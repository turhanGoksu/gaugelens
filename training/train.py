"""Train one way of D17, export it to ONNX and write its metrics.

    python -m training.train way1 --data <prepared> --out <dir>
        [--max-minutes 180] [--smoke] [--no-pretrained] [--workers 4]

- way1: the D-FINE detector; it also gives way 2 its dial boxes.
- way2: the heatmap model on dial crops.
- way3: Keypoint R-CNN.

``<prepared>`` holds ``train.jsonl`` and ``val.jsonl`` from
``python -m training.data prepare``. ``<out>`` gets ``model.pt`` (the best
weights), ``model.onnx`` and ``metrics.json``: the training history and GPU
time, the pixel error of each point on the validation images, and how far
the ONNX outputs are from PyTorch's.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training import common, data, detector, heatmap, kprcnn, preprocess
from training.torchutil import device as pick_device

SETTINGS = {
    # batch size, learning rate, eval every N steps
    "way1": {"batch": 8, "lr": 1e-4, "eval_every": 1000},
    "way2": {"batch": 64, "lr": 1e-3, "eval_every": 1000},
    "way3": {"batch": 4, "lr": 1e-4, "eval_every": 1000},
}
SMOKE = {"batch": 2, "eval_every": 3, "steps": 6, "train": 8, "val": 4}
PATIENCE = 5
VAL_LOSS_IMAGES = 400  # a fixed subset keeps each validation short
WARMUP = 500


def forever(loader) -> Iterator:
    while True:
        yield from loader


def every_kth(items: list, n: int) -> list:
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def optimizer_for(way: str, model, lr: float):
    if way == "way1":
        backbone = [p for n, p in model.named_parameters() if "backbone" in n]
        rest = [p for n, p in model.named_parameters() if "backbone" not in n]
        groups = [{"params": backbone, "lr": lr / 10}, {"params": rest, "lr": lr}]
    else:
        groups = [
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}
        ]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP))
    return opt, sched


def batch_loss(way: str, model, batch, dev) -> torch.Tensor:
    if way == "way1":
        pixels, targets = batch
        labels = [{k: v.to(dev) for k, v in t.items()} for t in targets]
        return model(pixel_values=pixels.to(dev), labels=labels).loss
    if way == "way2":
        x, maps, weights = (t.to(dev) for t in batch)
        return heatmap.loss_fn(model(x), maps, weights)
    images, targets = batch
    losses = model(
        [i.to(dev) for i in images],
        [{k: v.to(dev) for k, v in t.items()} for t in targets],
    )
    return sum(losses.values())


def build(way: str, pretrained: bool):
    if way == "way1":
        return detector.build_model(pretrained)
    if way == "way2":
        return heatmap.HeatmapNet(pretrained)
    return kprcnn.build_model(pretrained)


def loaders(way, train, val, batch, workers, rng):
    if way == "way2":
        return (
            heatmap.loader(train, heatmap.dial_order(train, rng), True, batch, workers),
            heatmap.loader(
                val,
                every_kth(heatmap.dial_order(val, random.Random(0)), VAL_LOSS_IMAGES),
                False,
                batch,
                0,
            ),
        )
    module = detector if way == "way1" else kprcnn
    val_order = every_kth(list(range(len(val))), VAL_LOSS_IMAGES)
    return (
        module.loader(train, common.training_order(train, rng), True, batch, workers),
        module.loader(val, val_order, False, batch, 0),
    )


def predicted_points(way: str, model, image, gauge, dev) -> dict[str, Any] | None:
    """The points this way finds for the labeled dial ``gauge``."""
    if way == "way2":  # the labeled box stands in for the detector's
        return heatmap.predict(model, image, gauge["box"], dev)
    if way == "way1":
        dials = detector.predict(model, image, dev)
    else:
        dial = kprcnn.predict(model, image, dev)
        dials = [dial] if dial else []
    if not dials:
        return None
    cx, cy = (
        (gauge["box"][0] + gauge["box"][2]) / 2,
        (gauge["box"][1] + gauge["box"][3]) / 2,
    )

    def distance(d):
        return ((d["box"][0] + d["box"][2]) / 2 - cx) ** 2 + (
            (d["box"][1] + d["box"][3]) / 2 - cy
        ) ** 2

    return min(dials, key=distance)["points"]


def point_report(way, model, val, dev) -> dict[str, Any]:
    rows: dict[str, list] = {"synthetic": [], "roboflow": []}
    for sample in val:
        image = common.load_rgb(sample["image"])
        kind = "synthetic" if sample["source"] == "synthetic" else "roboflow"
        for gauge in sample["gauges"]:
            found = predicted_points(way, model, image, gauge, dev)
            rows[kind].append(common.point_errors(found, gauge))
    return {kind: common.summarize_errors(r) for kind, r in rows.items() if r}


def max_diff(got, expected: torch.Tensor) -> float:
    if np.shape(got) != tuple(expected.shape):
        return float("inf")  # e.g. a different number of detections
    if expected.numel() == 0:
        return 0.0
    return float(np.abs(np.asarray(got, np.float64) - expected.numpy()).max())


def export(way: str, model, path: Path, sample: dict[str, Any]) -> dict[str, Any]:
    """Write the ONNX file and compare its outputs with PyTorch's on a real
    validation image: on random noise the detector's queries tie, and their
    order can differ between the two without any real difference."""
    import onnxruntime as ort

    model = model.cpu().eval()
    image = common.load_rgb(sample["image"])
    if way == "way1":
        module, x = detector.Exported(model), detector.to_pixels(image)[None]
        names, inputs, args, opset, dynamic = (
            ["scores", "boxes"],
            ["pixel_values"],
            (x,),
            17,
            None,
        )
    elif way == "way2":
        m = preprocess.crop_matrix(sample["gauges"][0]["box"])
        module = model
        x = torch.from_numpy(preprocess.crop_input(preprocess.crop(image, m)))[None]
        names, inputs, args, opset, dynamic = ["heatmaps"], ["image"], (x,), 17, None
    else:
        module, x = model, torch.from_numpy(preprocess.kprcnn_input(image)[0])
        names = ["boxes", "labels", "scores", "keypoints", "keypoints_scores"]
        args, inputs, opset = ([x],), ["image"], 11
        dynamic = {"image": {1: "height", 2: "width"}}
    # The reference comes first: tracing D-FINE leaves state behind that
    # changes the module's later outputs (seen with transformers 5.23), while
    # the ONNX file matches the module as it was before the export.
    with torch.no_grad():
        expected = module(*args)
    torch.onnx.export(
        module,
        args,
        str(path),
        input_names=inputs,
        output_names=names,
        opset_version=opset,
        dynamic_axes=dynamic,
        dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = session.run(None, {inputs[0]: x.numpy()})
    if way == "way3":
        want = {k: expected[0][k].numpy() for k in names}
        diff = common.detections_max_diff(dict(zip(names, got, strict=True)), want)
        return {
            "file": path.name,
            "opset": opset,
            "max_abs_diff": diff,
            "detections": len(want["boxes"]),
        }
    expected = [expected] if way == "way2" else list(expected)
    diffs = [max_diff(g, e) for g, e in zip(got, expected, strict=True)]
    return {"file": path.name, "opset": opset, "max_abs_diff": max(diffs)}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m training.train")
    parser.add_argument("way", choices=sorted(SETTINGS))
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-minutes", type=float, default=180.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--smoke", action="store_true", help="a few steps on a few images"
    )
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    rng = random.Random(0)
    dev = pick_device()
    train = data.read_jsonl(Path(args.data) / "train.jsonl")
    val = data.read_jsonl(Path(args.data) / "val.jsonl")
    settings = dict(SETTINGS[args.way])
    if args.smoke:
        train, val = train[: SMOKE["train"]], val[: SMOKE["val"]]
        settings |= {"batch": SMOKE["batch"], "eval_every": SMOKE["eval_every"]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = build(args.way, not args.no_pretrained).to(dev)
    opt, sched = optimizer_for(args.way, model, settings["lr"])
    train_loader, val_loader = loaders(
        args.way, train, val, settings["batch"], 0 if args.smoke else args.workers, rng
    )
    batches = forever(train_loader)
    steps = {"n": 0}

    def step() -> float:
        model.train()
        loss = batch_loss(args.way, model, next(batches), dev)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 0.1 if args.way == "way1" else 5.0
        )
        opt.step()
        sched.step()
        steps["n"] += 1
        return loss.item()

    @torch.no_grad()
    def validate() -> float:
        # Keypoint R-CNN returns losses only in training mode; its batch norm
        # is frozen, so that mode changes nothing here.
        model.train(args.way == "way3")
        losses = [float(batch_loss(args.way, model, b, dev)) for b in val_loader]
        return sum(losses) / max(1, len(losses))

    def save() -> None:
        torch.save(model.state_dict(), out / "model.pt")

    max_seconds = args.max_minutes * 60
    if args.smoke:
        max_seconds = min(max_seconds, 300.0)
    result = common.fit(
        step,
        validate,
        save,
        max_seconds=max_seconds,
        eval_every=settings["eval_every"],
        patience=1 if args.smoke else PATIENCE,
        max_steps=SMOKE["steps"] if args.smoke else None,
    )
    model.load_state_dict(torch.load(out / "model.pt", map_location=dev))
    metrics = {
        "way": args.way,
        "device": torch.cuda.get_device_name(0)
        if dev.type == "cuda"
        else platform.processor(),
        "versions": {"torch": torch.__version__},
        "images": {"train": len(train), "val": len(val)},
        "settings": settings,
        "fit": result,
        "val_points": point_report(args.way, model, val, dev),
    }
    metrics["onnx"] = export(args.way, model, out / "model.onnx", val[0])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(
        json.dumps(
            {k: metrics[k] for k in ("way", "device", "val_points", "onnx")}, indent=1
        )
    )
    print(
        f"stopped by {result['stopped']} after {result['steps']} steps,"
        f" {result['seconds']} s"
    )


if __name__ == "__main__":
    main()

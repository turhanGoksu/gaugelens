"""One sample format for every Design A trainer (D16, D17).

A sample is a dict:

- ``image``: path to the image file;
- ``gauges``: one dict per dial with ``box`` (x0, y0, x1, y1) and the four
  points ``center``, ``tip``, ``min`` and ``max`` as (x, y) pixels;
- ``group``: images of one scene share a group, so a split never puts the
  same scene on both sides;
- ``source``: ``synthetic`` or ``roboflow:<workspace>/<project>``.

    python -m training.data prepare <out> --synthetic <folder> ... \\
        [--roboflow data/external/roboflow]

writes ``<out>/train.jsonl`` and ``<out>/val.jsonl``. Synthetic folders are
whole splits as written by ``python -m gaugelens.synth generate``; the
Roboflow images are the kept rows of ``evaluation/train_roboflow/manifest.csv``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.aalborg import sha256_file
from evaluation.roboflow import DEST as ROBOFLOW_DIR
from evaluation.roboflow import MANIFEST

POINTS = ("center", "tip", "min", "max")
Sample = dict[str, Any]

# Roboflow class names for the dial box and each point (D16 card).
ROBOFLOW_CLASSES = {
    "project-kaqwa/keypoints-4lma6": {
        "box": "meter",
        "center": "center",
        "tip": "pointer_tip",
        "min": "min",
        "max": "max",
    },
    "computer-vision-ygqpn/gauge-reader-idem4": {
        "box": "gauge",
        "center": "base",
        "tip": "tip",
        "min": "min_value",
        "max": "max_value",
    },
}
# A point may sit on the rim, just outside a tight dial box.
BOX_MARGIN = 0.1


def synthetic_samples(folder: Path) -> list[Sample]:
    samples = []
    for line in (folder / "labels.jsonl").read_text().splitlines():
        label = json.loads(line)
        gauge = {
            "box": list(label["bbox"]),
            "center": list(label["center"]),
            "tip": list(label["needle_tip"]),
            "min": list(label["min_mark"]),
            "max": list(label["max_mark"]),
        }
        samples.append(
            {
                "image": str(folder / label["image"]),
                "gauges": [gauge],
                "group": f"synthetic:{label['seed']}",
                "source": "synthetic",
            }
        )
    return samples


def _inside(point: list[float], box: list[float]) -> bool:
    x0, y0, x1, y1 = box
    mx, my = BOX_MARGIN * (x1 - x0), BOX_MARGIN * (y1 - y0)
    return x0 - mx <= point[0] <= x1 + mx and y0 - my <= point[1] <= y1 + my


def _gauges(annotations: list[tuple[str, list[float]]], names: dict[str, str]):
    """Dial boxes with their points, or None if any dial is incomplete.

    A point belongs to the dial whose box holds it; when two boxes hold it,
    to the one whose centre is nearer.
    """
    boxes = [
        [x, y, x + w, y + h]
        for name, (x, y, w, h) in annotations
        if name == names["box"]
    ]
    gauges: list[dict[str, Any]] = [{"box": box} for box in boxes]
    for key in POINTS:
        for name, (x, y, w, h) in annotations:
            if name != names[key]:
                continue
            point = [x + w / 2, y + h / 2]
            holders = [g for g in gauges if _inside(point, g["box"])]
            if not holders:
                return None
            nearest = min(
                holders,
                key=lambda g: (
                    (point[0] - (g["box"][0] + g["box"][2]) / 2) ** 2
                    + (point[1] - (g["box"][1] + g["box"][3]) / 2) ** 2
                ),
            )
            if key in nearest:
                return None
            nearest[key] = point
    if not gauges or any(len(g) != 1 + len(POINTS) for g in gauges):
        return None
    return gauges


def roboflow_samples(
    dest: Path = ROBOFLOW_DIR, manifest: Path = MANIFEST
) -> tuple[list[Sample], Counter[str]]:
    """The kept images of the manifest, checked by SHA-256.

    Returns the samples and how many kept images were dropped, and why: an
    image whose dials do not each have exactly one of every point cannot be
    labeled reliably.
    """
    rows = [r for r in csv.DictReader(manifest.open()) if r["keep"] == "yes"]
    wanted = {(r["dataset"], r["split"], r["file"]): r for r in rows}
    samples: list[Sample] = []
    dropped: Counter[str] = Counter()
    seen = set()
    for dataset, version in sorted({(r["dataset"], r["version"]) for r in rows}):
        folder = dest / dataset / f"v{version}"
        names = ROBOFLOW_CLASSES[dataset]
        for ann in sorted(folder.glob("*/_annotations.coco.json")):
            coco = json.loads(ann.read_text())
            category = {c["id"]: c["name"] for c in coco["categories"]}
            by_image: dict[int, list[tuple[str, list[float]]]] = {}
            for a in coco["annotations"]:
                by_image.setdefault(a["image_id"], []).append(
                    (category[a["category_id"]], a["bbox"])
                )
            for image in coco["images"]:
                key = (dataset, ann.parent.name, image["file_name"])
                if key not in wanted:
                    continue
                seen.add(key)
                path = ann.parent / image["file_name"]
                if sha256_file(path) != wanted[key]["sha256"]:
                    raise ValueError(f"{path} differs from the manifest")
                gauges = _gauges(by_image.get(image["id"], []), names)
                if gauges is None:
                    dropped["points do not match dials"] += 1
                    continue
                samples.append(
                    {
                        "image": str(path),
                        "gauges": gauges,
                        "group": f"roboflow:{wanted[key]['scene']}",
                        "source": f"roboflow:{dataset}",
                    }
                )
    if missing := sorted(set(wanted) - seen):
        raise SystemExit(
            f"{len(missing)} kept images are missing, e.g. {missing[0]}. "
            "Run the downloads in evaluation/train_roboflow/README.md."
        )
    return samples, dropped


def split(samples: list[Sample], fraction: float) -> tuple[list[Sample], list[Sample]]:
    """Whole groups go to validation until they hold about ``fraction`` of the
    images; the same samples always split the same way.

    Groups are taken in a fixed order (the hash of their name) and skipped
    when they would push the share past ``fraction``. Scenes differ a lot in
    size: one video can be a third of the real images, and counting groups
    instead of images once sent 37% of them to validation. A group that large
    stays in training.
    """
    sizes = Counter(s["group"] for s in samples)
    chosen: set[str] = set()
    taken = 0
    for group in sorted(sizes, key=lambda g: hashlib.sha256(g.encode()).digest()):
        if taken + sizes[group] <= fraction * len(samples):
            chosen.add(group)
            taken += sizes[group]
    train = [s for s in samples if s["group"] not in chosen]
    val = [s for s in samples if s["group"] in chosen]
    return train, val


def write_jsonl(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def read_jsonl(path: Path) -> list[Sample]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m training.data")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("out")
    prepare.add_argument("--synthetic", nargs="+", default=[], help="train folders")
    prepare.add_argument("--synthetic-val", nargs="*", default=[], help="val folders")
    prepare.add_argument("--roboflow", default=None, help="download folder")
    prepare.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()

    train: list[Sample] = []
    val: list[Sample] = []
    for folder in args.synthetic:
        train += synthetic_samples(Path(folder))
    for folder in args.synthetic_val:
        val += synthetic_samples(Path(folder))
    if args.roboflow:
        real, dropped = roboflow_samples(Path(args.roboflow))
        real_train, real_val = split(real, args.val_fraction)
        train += real_train
        val += real_val
        print(
            f"roboflow: {len(real_train)} train, {len(real_val)} val,"
            f" dropped {dict(dropped)}"
        )
    out = Path(args.out)
    write_jsonl(train, out / "train.jsonl")
    write_jsonl(val, out / "val.jsonl")
    print(f"train: {len(train)}, val: {len(val)} -> {out}")


if __name__ == "__main__":
    main()

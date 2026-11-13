"""Run a design on a labeled dev set and print the scores.

    python -m evaluation.run_dev data/synthetic/dev --method a0 --workers 8
    python -m evaluation.run_dev measurebench --method a0 --workers 8
    python -m evaluation.run_dev aalborg --method a0
    python -m evaluation.run_dev aalborg-static --method a0 --workers 8

A folder is a synthetic set with ``labels.jsonl``; a name is one of the real
dev sets in ``evaluation/real_dev.py``. Predictions go to
runs/<dataset>-<method>.jsonl so failures can be inspected. Dev sets only:
the test set has its own one-time runner.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from evaluation.aalborg import read_frames
from evaluation.metrics import ANSWERED, Outcome, score, summarize
from evaluation.real_dev import LOADERS, Label
from gaugelens import __version__
from gaugelens.classical import read_classical

# Labels per job. Frames of one video share a job so the video is decoded once.
FILES_PER_JOB = 8
FRAMES_PER_JOB = 64


def _images(batch: list[Label]) -> list[np.ndarray]:
    if "video" not in batch[0]:
        return [np.asarray(Image.open(b["path"]).convert("RGB")) for b in batch]
    frames = read_frames(Path(batch[0]["video"]), (b["frame"] for b in batch))
    return [cv2.cvtColor(frames[b["frame"]], cv2.COLOR_BGR2RGB) for b in batch]


def _read_a0(batch: list[Label]) -> list[dict[str, Any]]:
    predictions = []
    for label, image in zip(batch, _images(batch), strict=True):
        start = time.perf_counter()
        reading = read_classical(image, label["min_value"], label["max_value"])
        predictions.append(
            {
                "image": label["image"],
                "status": reading.status,
                "value": reading.value,
                "notes": list(reading.notes),
                "seconds": time.perf_counter() - start,
            }
        )
    return predictions


METHODS = {"a0": _read_a0}


def batches(labels: list[Label]) -> list[list[Label]]:
    """Consecutive labels of one source, so a video is read in one pass."""
    jobs: list[list[Label]] = []
    for label in labels:
        video = label.get("video")
        size = FRAMES_PER_JOB if video else FILES_PER_JOB
        if jobs and len(jobs[-1]) < size and jobs[-1][0].get("video") == video:
            jobs[-1].append(label)
        else:
            jobs.append([label])
    return jobs


def _tilt_bucket(label: dict[str, Any]) -> str:
    tilt = max(abs(label["capture"]["tilt_x"]), abs(label["capture"]["tilt_y"]))
    for edge in (10, 20, 30):
        if tilt < edge:
            return f"tilt <{edge}"
    return "tilt >=30"


def load_synthetic(folder: Path) -> list[Label]:
    lines = (folder / "labels.jsonl").read_text().splitlines()
    return [
        label
        | {
            "path": str(folder / label["image"]),
            "group": label["image"],
            "buckets": [_tilt_bucket(label)],
        }
        for label in map(json.loads, lines)
    ]


def stability(
    labels: list[Label], outcomes: list[Outcome], predictions: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Per gauge whose needle does not move: how alike its frames were read.

    Spreads are in fractions of full scale, over the frames read as a number.
    """
    report: dict[str, dict[str, Any]] = {}
    for gauge in dict.fromkeys(label["group"] for label in labels):
        rows = [
            (lab, o, p)
            for lab, o, p in zip(labels, outcomes, predictions, strict=True)
            if lab["group"] == gauge
        ]
        lab = rows[0][0]
        span = lab["max_value"] - lab["min_value"]
        values = np.array([p["value"] for _, _, p in rows if p["value"] is not None])
        right = [
            o.status in ANSWERED if o.expected == "in_range" else o.status == o.expected
            for _, o, _ in rows
        ]
        report[gauge] = {
            "frames": len(rows),
            "expected": rows[0][1].expected,
            "statuses": dict(Counter(o.status for _, o, _ in rows)),
            "status_right": sum(right) / len(rows),
            "within_2%": sum(
                o.error is not None and o.error <= 0.02 for _, o, _ in rows
            )
            / len(rows),
            "numbers": len(values),
            "spread_p5_p95": float(np.subtract(*np.percentile(values, [95, 5]))) / span
            if len(values)
            else None,
            "spread_max_min": float(values.max() - values.min()) / span
            if len(values)
            else None,
        }
    return report


def _pct(v: float | None) -> str:
    return "   -  " if v is None else f"{v:6.1%}"


def _print_summary(title: str, summary: dict[str, Any]) -> None:
    print(
        f"{title:16s} n={summary['in_range_images']:4d}"
        f"  within 1/2/5% FS: {_pct(summary['within_1%'])} {_pct(summary['within_2%'])}"
        f" {_pct(summary['within_5%'])}"
        f"  confidently wrong (>2%): {_pct(summary['confidently_wrong_2%'])}"
        f"  abstained: {_pct(summary['abstained'])}"
        f"  said off-scale: {_pct(summary['false_out_of_range'])}"
        f"  median err: {_pct(summary['median_error'])}"
        + (
            f"  | off-scale n={summary['out_of_range_images']}"
            f" status right {_pct(summary['out_of_range_correct'])}"
            if "out_of_range_images" in summary
            else ""
        )
    )


def _print_stability(report: dict[str, dict[str, Any]]) -> None:
    print("stability, needle does not move (spreads over the frames read as a number):")
    print(
        "gauge  frames  expected     status right  within 2%  numbers"
        "  spread p5-p95  max-min  statuses"
    )
    for gauge, r in report.items():
        print(
            f"{gauge:6s} {r['frames']:6d}  {r['expected']:11s}"
            f"  {_pct(r['status_right'])}       {_pct(r['within_2%'])}"
            f"  {r['numbers']:7d}"
            f"  {_pct(r['spread_p5_p95'])}         {_pct(r['spread_max_min'])}"
            f"   {r['statuses']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help=f"a folder, or one of: {', '.join(LOADERS)}")
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if args.dataset in LOADERS:
        name, labels = args.dataset, LOADERS[args.dataset]()
    else:
        folder = Path(args.dataset)
        name, labels = folder.name, load_synthetic(folder)
    labels = labels[: args.limit]
    jobs = batches(labels)
    reader = METHODS[args.method]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            predictions = [p for done in pool.map(reader, jobs) for p in done]
    else:
        predictions = [p for job in jobs for p in reader(job)]

    outcomes = [
        score(lab, p["status"], p["value"])
        for lab, p in zip(labels, predictions, strict=True)
    ]
    out = Path("runs") / f"{name}-{args.method}.jsonl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        for prediction, outcome in zip(predictions, outcomes, strict=True):
            record = prediction | {"expected": outcome.expected, "error": outcome.error}
            f.write(json.dumps(record) + "\n")

    summary = summarize(outcomes)
    seconds = sorted(p["seconds"] for p in predictions)
    gauges = len({label["group"] for label in labels})
    print(f"{args.method} on {name} | gaugelens {__version__} | {platform.platform()}")
    print(f"images: {len(labels)} of {gauges} gauges | statuses: {summary['statuses']}")
    print(
        f"out-of-range needles: {summary.get('out_of_range_images', 0)}, "
        f"status right {summary.get('out_of_range_correct', 0):.1%}, "
        f"given as ok {summary.get('out_of_range_given_ok', 0):.1%}"
    )
    median_ms = seconds[len(seconds) // 2] * 1000
    print(f"latency per image (CPU, one process): median {median_ms:.0f} ms")
    _print_summary("all", summary)
    buckets: dict[str, list[Outcome]] = {}
    for label, outcome in zip(labels, outcomes, strict=True):
        for bucket in label["buckets"]:
            buckets.setdefault(bucket, []).append(outcome)
    for bucket in sorted(buckets):
        _print_summary(bucket, summarize(buckets[bucket]))
    if any("video" in label for label in labels):
        _print_stability(stability(labels, outcomes, predictions))
    print(f"predictions: {out}")


if __name__ == "__main__":
    main()

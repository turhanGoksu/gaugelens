"""Run a design on a labeled dev set and print the scores.

    python -m evaluation.run_dev data/synthetic/dev --method a0 --workers 8

Predictions go to runs/<dataset>-<method>.jsonl so failures can be inspected.
Dev sets only: the test set has its own one-time runner.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from evaluation.metrics import Outcome, score, summarize
from gaugelens import __version__
from gaugelens.classical import read_classical


def _read_a0(job: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    folder, label = job
    image = np.asarray(Image.open(Path(folder) / label["image"]).convert("RGB"))
    start = time.perf_counter()
    reading = read_classical(image, label["min_value"], label["max_value"])
    seconds = time.perf_counter() - start
    return {
        "image": label["image"],
        "status": reading.status,
        "value": reading.value,
        "notes": list(reading.notes),
        "seconds": seconds,
    }


METHODS = {"a0": _read_a0}


def _tilt_bucket(label: dict[str, Any]) -> str:
    tilt = max(abs(label["capture"]["tilt_x"]), abs(label["capture"]["tilt_y"]))
    for edge in (10, 20, 30):
        if tilt < edge:
            return f"tilt <{edge}"
    return "tilt >=30"


def _print_summary(title: str, summary: dict[str, Any]) -> None:
    def pct(v: float | None) -> str:
        return "   -  " if v is None else f"{v:6.1%}"

    print(
        f"{title:12s} n={summary['in_range_images']:4d}"
        f"  within 1/2/5% FS: {pct(summary['within_1%'])} {pct(summary['within_2%'])}"
        f" {pct(summary['within_5%'])}"
        f"  confidently wrong (>2%): {pct(summary['confidently_wrong_2%'])}"
        f"  abstained: {pct(summary['abstained'])}"
        f"  median err: {pct(summary['median_error'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    folder = Path(args.dataset)
    labels = [
        json.loads(line) for line in (folder / "labels.jsonl").read_text().splitlines()
    ]
    labels = labels[: args.limit]
    jobs = [(str(folder), label) for label in labels]
    reader = METHODS[args.method]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            predictions = list(pool.map(reader, jobs, chunksize=8))
    else:
        predictions = [reader(job) for job in jobs]

    outcomes = [
        score(lab, p["status"], p["value"])
        for lab, p in zip(labels, predictions, strict=True)
    ]
    out = Path("runs") / f"{folder.name}-{args.method}.jsonl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        for prediction, outcome in zip(predictions, outcomes, strict=True):
            f.write(
                json.dumps(
                    prediction | {"expected": outcome.expected, "error": outcome.error}
                )
                + "\n"
            )

    summary = summarize(outcomes)
    seconds = sorted(p["seconds"] for p in predictions)
    print(
        f"{args.method} on {folder} | gaugelens {__version__} | {platform.platform()}"
    )
    print(f"statuses: {summary['statuses']}")
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
        buckets.setdefault(_tilt_bucket(label), []).append(outcome)
    for name in ("tilt <10", "tilt <20", "tilt <30", "tilt >=30"):
        if name in buckets:
            _print_summary(name, summarize(buckets[name]))
    print(f"predictions: {out}")


if __name__ == "__main__":
    main()

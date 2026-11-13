"""Load the real dev sets in the shape the dev runner scores (D14, D15).

Every label is a dict with the keys ``metrics.score`` reads (``min_value``,
``max_value``, ``value``, ``fraction`` and, when the truth is a range,
``interval``) plus:

- ``image``: the id written to the predictions file,
- ``path`` for an image file, or ``video`` and ``frame`` for a video frame,
- ``group``: the gauge, since frames of one gauge are not independent (D13),
- ``buckets``: the subsets the runner reports on their own.

A needle labeled off the scale gets ``fraction`` -inf or +inf and no value.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from evaluation import aalborg

ROOT = Path(__file__).resolve().parents[1]
MEASUREBENCH_DIR = ROOT / "data" / "external" / "measurebench"
MEASUREBENCH_LABELS = Path(__file__).resolve().parent / "dev_measurebench"
AALBORG_DIR = ROOT / "data" / "external" / "aalborg"
# Needles that stay put for the whole video (evaluation/dev_aalborg/README.md).
STATIC_GAUGES = ("man4", "man5", "man6", "man7")

Label = dict[str, Any]


def _fraction(status: str, value: float | None, lo: float, hi: float) -> float:
    if status == "below_range":
        return -math.inf
    if status == "above_range":
        return math.inf
    assert value is not None
    return (value - lo) / (hi - lo)


def _require(path: Path, hint: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {path}. {hint}")


def load_measurebench(
    images: Path = MEASUREBENCH_DIR, labels_dir: Path = MEASUREBENCH_LABELS
) -> list[Label]:
    """The eligible photos; the truth is MeasureBench's interval for the scale."""
    _require(
        images / "manifest.jsonl",
        "Run: python -m evaluation.measurebench data/external/measurebench "
        "--test-images data/raw/commons",
    )
    manifest = {r["id"]: r for r in map(json.loads, (images / "manifest.jsonl").open())}
    labels = []
    for row in aalborg.read_csv(labels_dir / "labels.csv"):
        if row["eligible"] != "yes":
            continue
        entry = manifest[row["id"]]
        lo_value, hi_value = entry["readings"][int(row["reading_index"])]["interval"]
        lo, hi = float(row["min"]), float(row["max"])
        middle = (lo_value + hi_value) / 2
        labels.append(
            {
                "image": row["id"],
                "path": str(images / entry["file"]),
                "min_value": lo,
                "max_value": hi,
                "value": middle,
                "interval": [lo_value, hi_value],
                "fraction": (middle - lo) / (hi - lo),
                "group": row["id"],
                "buckets": [entry["image_type"]]
                + (["inferred_end"] if row["inferred_end"] else []),
            }
        )
    return labels


def load_aalborg(
    dest: Path = AALBORG_DIR, labels_dir: Path = aalborg.LABELS_DIR
) -> list[Label]:
    """The 32 labeled frames, as written by ``python -m evaluation.aalborg frames``."""
    labels = []
    for row in aalborg.read_csv(labels_dir / "labels.csv"):
        path = dest / "frames" / f"{row['id']}.png"
        _require(path, f"Run: python -m evaluation.aalborg frames {dest}")
        lo, hi = float(row["min"]), float(row["max"])
        value = float(row["value"]) if row["value"] else None
        labels.append(
            {
                "image": row["id"],
                "path": str(path),
                "min_value": lo,
                "max_value": hi,
                "value": value,
                "fraction": _fraction(row["status"], value, lo, hi),
                "group": row["gauge"],
                "buckets": [row["gauge"]]
                + (["sure: yes"] if row["sure"] == "yes" else []),
            }
        )
    return labels


def load_aalborg_static(
    dest: Path = AALBORG_DIR, labels_dir: Path = aalborg.LABELS_DIR
) -> list[Label]:
    """Every frame of the videos whose needle does not move.

    The needle stays put, so the labels of a gauge's first and last frame
    hold for all of its frames: their status, and for a number the range
    between the two readings.
    """
    videos = {v["gauge"]: v for v in aalborg.read_csv(labels_dir / "videos.csv")}
    rows = aalborg.read_csv(labels_dir / "labels.csv")
    labels = []
    for gauge in STATIC_GAUGES:
        own = [r for r in rows if r["gauge"] == gauge]
        statuses = {r["status"] for r in own}
        if len(statuses) != 1:
            raise ValueError(f"{gauge}: labeled frames disagree: {sorted(statuses)}")
        status = statuses.pop()
        lo, hi = float(own[0]["min"]), float(own[0]["max"])
        label: Label = {
            "min_value": lo,
            "max_value": hi,
            "group": gauge,
            "buckets": [gauge],
        }
        if status == "value":
            values = [float(r["value"]) for r in own]
            middle = (min(values) + max(values)) / 2
            label |= {
                "value": middle,
                "interval": [min(values), max(values)],
                "fraction": _fraction(status, middle, lo, hi),
            }
        else:
            label |= {"value": None, "fraction": _fraction(status, None, lo, hi)}
        video = dest / "files" / aalborg.VIDEOS / videos[gauge]["file"]
        _require(
            video,
            f'Run: python -m evaluation.aalborg download {dest} "{aalborg.VIDEOS}/*"',
        )
        if aalborg.sha256_file(video) != videos[gauge]["sha256"]:
            raise ValueError(f"{video.name} differs from the labeled video")
        labels += [
            label
            | {"image": f"{gauge}_f{frame:05d}", "video": str(video), "frame": frame}
            for frame in range(int(videos[gauge]["frames"]))
        ]
    return labels


LOADERS = {
    "measurebench": load_measurebench,
    "aalborg": load_aalborg,
    "aalborg-static": load_aalborg_static,
}

"""Pieces shared by the three Design A trainers (D17)."""

from __future__ import annotations

import math
import random
import statistics
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from training.data import POINTS, Sample

# Real photos are few (D16); each pass repeats them so they make up about
# this share of what the model sees.
REAL_SHARE = 0.25


def load_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def color_jitter(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Brightness, contrast and saturation. Geometry is untouched: a mirror
    would turn a clockwise scale counterclockwise (guideline rule 5)."""
    out = image.astype(np.float32)
    mean = out.mean()
    out = (out - mean) * rng.uniform(0.7, 1.3) + mean + rng.uniform(-25.0, 25.0)
    gray = out.mean(axis=2, keepdims=True)
    out = gray + (out - gray) * rng.uniform(0.6, 1.4)
    return np.clip(out, 0, 255).astype(np.uint8)


def training_order(samples: list[Sample], rng: random.Random) -> list[int]:
    """Sample indices for one pass: synthetic ones once, real ones repeated."""
    synthetic = [i for i, s in enumerate(samples) if s["source"] == "synthetic"]
    real = [i for i, s in enumerate(samples) if s["source"] != "synthetic"]
    repeat = 1
    if synthetic and real:
        share = REAL_SHARE / (1 - REAL_SHARE)
        repeat = max(1, round(share * len(synthetic) / len(real)))
    order = synthetic + real * repeat
    rng.shuffle(order)
    return order


def random_crop(
    image: np.ndarray, gauges: list[dict[str, Any]], rng: random.Random
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Cut a random window that still holds every dial whole, so the dials
    appear at varied sizes and places."""
    h, w = image.shape[:2]
    x0 = min(g["box"][0] for g in gauges)
    y0 = min(g["box"][1] for g in gauges)
    x1 = max(g["box"][2] for g in gauges)
    y1 = max(g["box"][3] for g in gauges)
    left = int(rng.uniform(0, max(0.0, x0)))
    top = int(rng.uniform(0, max(0.0, y0)))
    right = int(rng.uniform(min(float(w), x1), w))
    bottom = int(rng.uniform(min(float(h), y1), h))
    if right - left < 32 or bottom - top < 32:
        return image, gauges
    shifted = [
        {
            key: [v[0] - left, v[1] - top, v[2] - left, v[3] - top]
            if key == "box"
            else [v[0] - left, v[1] - top]
            for key, v in g.items()
        }
        for g in gauges
    ]
    return image[top:bottom, left:right], shifted


def dial_size(gauge: dict[str, Any]) -> float:
    x0, y0, x1, y1 = gauge["box"]
    return max(x1 - x0, y1 - y0)


def point_errors(
    found: dict[str, Any] | None, gauge: dict[str, Any]
) -> dict[str, float | None]:
    """Distance of each found point from its label, as a share of half the
    dial size (about its radius). None for a point that was not found."""
    half = dial_size(gauge) / 2
    errors: dict[str, float | None] = {}
    for key in POINTS:
        point = (found or {}).get(key)
        if point is None:
            errors[key] = None
        else:
            dx, dy = point[0] - gauge[key][0], point[1] - gauge[key][1]
            errors[key] = math.hypot(dx, dy) / half
    return errors


def summarize_errors(rows: list[dict[str, float | None]]) -> dict[str, Any]:
    out: dict[str, Any] = {"dials": len(rows)}
    for key in POINTS:
        values = sorted(r[key] for r in rows if r[key] is not None)
        out[key] = {
            "missing": sum(r[key] is None for r in rows),
            "median": statistics.median(values) if values else None,
            "p90": values[int(0.9 * (len(values) - 1))] if values else None,
        }
    return out


def fit(
    step: Callable[[], float],
    validate: Callable[[], float],
    save: Callable[[], None],
    *,
    max_seconds: float,
    eval_every: int,
    patience: int,
    max_steps: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train until the validation loss stops improving or time runs out (D17).

    ``step`` runs one batch and returns its loss; ``save`` keeps the current
    weights, and is called whenever the validation loss reaches a new best.
    """
    start = time.monotonic()
    best, bad, steps, recent = math.inf, 0, 0, []
    history: list[dict[str, float]] = []
    stopped = "patience"
    while True:
        recent.append(step())
        steps += 1
        elapsed = time.monotonic() - start
        out_of_time = elapsed > max_seconds
        out_of_steps = max_steps is not None and steps >= max_steps
        if steps % eval_every and not (out_of_time or out_of_steps):
            continue
        val = validate()
        train_loss = sum(recent) / len(recent)
        recent = []
        history.append(
            {
                "step": steps,
                "seconds": round(elapsed),
                "train_loss": train_loss,
                "val_loss": val,
            }
        )
        log(
            f"step {steps}  {elapsed / 60:.1f} min"
            f"  train {train_loss:.4f}  val {val:.4f}"
        )
        if val < best:
            best, bad = val, 0
            save()
        else:
            bad += 1
        if out_of_time or out_of_steps:
            stopped = "time" if out_of_time else "steps"
            break
        if bad >= patience:
            break
    return {
        "steps": steps,
        "seconds": round(time.monotonic() - start),
        "best_val_loss": best,
        "stopped": stopped,
        "history": history,
    }


def detections_max_diff(got: dict[str, Any], want: dict[str, Any]) -> float:
    """Largest difference between two lists of detections, in any order.

    Keypoint R-CNN lists its detections by score. When scores nearly tie,
    ONNX Runtime and PyTorch may order them differently, and comparing them
    position by position shows differences of hundreds of pixels that are
    not there. Each detection is paired with the nearest box of the other
    list instead. Lists of different lengths give infinity.
    """
    want_boxes = np.asarray(want["boxes"], np.float64)
    got_boxes = np.asarray(got["boxes"], np.float64)
    if len(want_boxes) != len(got_boxes):
        return math.inf
    if len(want_boxes) == 0:
        return 0.0
    cost = np.abs(want_boxes[:, None, :] - got_boxes[None, :, :]).sum(-1)
    pairs: list[tuple[int, int]] = []
    used_want: set[int] = set()
    used_got: set[int] = set()
    for flat in np.argsort(cost, axis=None):
        i, j = (int(v) for v in np.unravel_index(flat, cost.shape))
        if i not in used_want and j not in used_got:
            pairs.append((i, j))
            used_want.add(i)
            used_got.add(j)
    worst = 0.0
    for key in want:
        a = np.asarray(want[key], np.float64)
        b = np.asarray(got[key], np.float64)
        for i, j in pairs:
            worst = max(worst, float(np.abs(a[i] - b[j]).max()))
    return worst

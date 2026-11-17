"""Turn model outputs into dials and points (D17). Numpy only, so the ONNX
readers and the trainers share it."""

from __future__ import annotations

from typing import Any

import numpy as np

POINTS = ("center", "tip", "min", "max")
# Detector classes: the dial, then one class per point.
CLASSES = ("gauge", *POINTS)
MIN_DIAL_SCORE = 0.3
MIN_POINT_SCORE = 0.1
# A point may sit on the rim, just outside a tight dial box.
BOX_MARGIN = 0.1


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def detector_dials(
    scores: np.ndarray,
    boxes: np.ndarray,
    width: float,
    height: float,
    min_dial: float = MIN_DIAL_SCORE,
    min_point: float = MIN_POINT_SCORE,
) -> list[dict[str, Any]]:
    """Dials found by the detector, best first, each with its points.

    ``scores`` is (queries, classes) after the sigmoid; ``boxes`` is
    (queries, 4) as normalized (cx, cy, w, h). A point is the centre of its
    box and belongs to the dial whose box holds it (nearest centre on a tie);
    each dial keeps its best-scoring point of each kind, or None.
    """
    scale = np.array([width, height, width, height])
    xyxy = np.concatenate(
        [boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], 1
    )
    xyxy = xyxy * scale
    dials: list[dict[str, Any]] = []
    for q in np.argsort(-scores[:, 0]):
        if scores[q, 0] < min_dial:
            break
        if any(_iou(xyxy[q], d["_box"]) > 0.7 for d in dials):
            continue
        dials.append({"score": float(scores[q, 0]), "_box": xyxy[q], "points": {}})
    for c, key in enumerate(POINTS, start=1):
        best: dict[int, tuple[float, list[float]]] = {}
        for q in np.nonzero(scores[:, c] >= min_point)[0]:
            x = (xyxy[q, 0] + xyxy[q, 2]) / 2
            y = (xyxy[q, 1] + xyxy[q, 3]) / 2
            holders = []
            for i, d in enumerate(dials):
                x0, y0, x1, y1 = d["_box"]
                mx, my = BOX_MARGIN * (x1 - x0), BOX_MARGIN * (y1 - y0)
                if x0 - mx <= x <= x1 + mx and y0 - my <= y <= y1 + my:
                    holders.append(
                        (((x0 + x1) / 2 - x) ** 2 + ((y0 + y1) / 2 - y) ** 2, i)
                    )
            if holders:
                i = min(holders)[1]
                if i not in best or scores[q, c] > best[i][0]:
                    best[i] = (float(scores[q, c]), [float(x), float(y)])
        for i, d in enumerate(dials):
            d["points"][key] = best[i][1] if i in best else None
    for d in dials:
        d["box"] = [float(v) for v in d.pop("_box")]
    return dials


def _offset(left: float, centre: float, right: float) -> float:
    """Sub-pixel offset of a peak from a parabola through the logarithms of
    three neighbouring values; exact for a Gaussian peak."""
    if min(left, centre, right) <= 0:
        return 0.0
    a, b, c = np.log(left), np.log(centre), np.log(right)
    curvature = a - 2 * b + c
    if curvature >= 0:
        return 0.0
    return float(np.clip(0.5 * (a - c) / curvature, -0.5, 0.5))


def heatmap_peaks(heatmaps: np.ndarray) -> list[tuple[float, float, float]]:
    """(x, y, peak value) per channel, in heatmap pixels.

    The strongest pixel is refined below one pixel along each axis. A 3x3
    weighted mean would pull the peak towards the pixel's centre by up to a
    quarter of a pixel, which is about 1% of the dial here.
    """
    out = []
    _, h, w = heatmaps.shape
    for m in heatmaps:
        y, x = (int(v) for v in np.unravel_index(int(np.argmax(m)), m.shape))
        fx, fy = float(x), float(y)
        if 0 < x < w - 1:
            fx += _offset(m[y, x - 1], m[y, x], m[y, x + 1])
        if 0 < y < h - 1:
            fy += _offset(m[y - 1, x], m[y, x], m[y + 1, x])
        out.append((fx, fy, float(m[y, x])))
    return out

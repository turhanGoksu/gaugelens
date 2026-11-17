"""Read gauges with the ONNX files of D17's three ways, on CPU.

    <models>/way1/model.onnx   D-FINE: dials and points as boxes
    <models>/way2/model.onnx   heatmaps on a dial crop; the box comes from way 1
    <models>/way3/model.onnx   Keypoint R-CNN

ONNX Runtime, numpy and OpenCV only: these readers run where the library
will, and prepare images exactly as the trainers did (training/preprocess).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from gaugelens.points import GaugePoints, read_points
from training import preprocess
from training.decode import POINTS, detector_dials

WAYS = ("way1", "way2", "way3")


@cache
def session(path: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1  # the dev runner parallelises over images
    return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def _file(models: Path, way: str) -> str:
    path = models / way / "model.onnx"
    if not path.exists():
        raise SystemExit(f"Missing {path}: download it from the Kaggle run (D17).")
    return str(path)


def detector_dials_onnx(image: np.ndarray, models: Path) -> list[dict[str, Any]]:
    h, w = image.shape[:2]
    scores, boxes = session(_file(models, "way1")).run(
        None, {"pixel_values": preprocess.detector_input(image)[None]}
    )
    return detector_dials(scores[0], boxes[0], w, h)


def heatmap_points(image: np.ndarray, box: list[float], models: Path) -> dict[str, Any]:
    m = preprocess.crop_matrix(box)
    x = preprocess.crop_input(preprocess.crop(image, m))[None]
    (heatmaps,) = session(_file(models, "way2")).run(None, {"image": x})
    return preprocess.points_from_heatmaps(heatmaps[0], m)


def kprcnn_dial(image: np.ndarray, models: Path) -> dict[str, Any] | None:
    x, scale = preprocess.kprcnn_input(image)
    boxes, _, scores, keypoints, _ = session(_file(models, "way3")).run(
        None, {"image": x}
    )
    if len(scores) == 0:
        return None
    i = int(np.argmax(scores))
    return {
        "score": float(scores[i]),
        "box": [float(v) * scale for v in boxes[i]],
        "points": {
            key: [float(keypoints[i][j][0]) * scale, float(keypoints[i][j][1]) * scale]
            for j, key in enumerate(POINTS)
        },
    }


def find(way: str, image: np.ndarray, models: Path) -> dict[str, Any] | None:
    """The best dial and its points, or None when no dial is found."""
    if way == "way3":
        return kprcnn_dial(image, models)
    dials = detector_dials_onnx(image, models)
    if not dials:
        return None
    dial = dials[0]
    if way == "way2":
        dial = dial | {"points": heatmap_points(image, dial["box"], models)}
    return dial


def read(
    way: str, image: np.ndarray, min_value: float, max_value: float, models: Path
) -> dict[str, Any]:
    """Status and value as D17 defines them, plus what was found."""
    dial = find(way, image, models)
    if dial is None:
        return {"status": "unreadable", "value": None, "notes": ["no dial found"]}
    found = {"box": dial["box"], "points": dial["points"], "score": dial["score"]}
    missing = [key for key in POINTS if dial["points"].get(key) is None]
    if missing:
        notes = [f"not found: {', '.join(missing)}"]
        return {"status": "unreadable", "value": None, "notes": notes} | found
    p = dial["points"]
    try:
        reading = read_points(
            GaugePoints(
                tuple(p["center"]), tuple(p["tip"]), tuple(p["min"]), tuple(p["max"])
            ),
            min_value,
            max_value,
        )
    except ValueError as error:
        return {"status": "unreadable", "value": None, "notes": [str(error)]} | found
    status = "ok" if reading.position == "in_range" else reading.position
    return {"status": status, "value": reading.value, "notes": []} | found

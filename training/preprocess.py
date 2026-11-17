"""Image preparation shared by the trainers and the ONNX readers (D17).

Numpy and OpenCV only: the readers run without PyTorch, and they must
prepare images exactly as the models saw them in training.
"""

from __future__ import annotations

import random
from typing import Any

import cv2
import numpy as np

from training.decode import POINTS, heatmap_peaks

# Way 1: D-FINE sees the whole image stretched to a square, values in [0, 1].
DETECTOR_SIZE = 640

# Way 2: the dial is cut out with a margin and scaled to CROP x CROP.
CROP = 256
HEATMAP = 64
MARGIN = 1.3  # crop side as a multiple of the dial size
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Way 3: Keypoint R-CNN resizes to this side at most; larger input only costs time.
KPRCNN_MAX_SIDE = 1066


def detector_input(image: np.ndarray) -> np.ndarray:
    """RGB uint8 -> (3, DETECTOR_SIZE, DETECTOR_SIZE) float32 in [0, 1]."""
    resized = cv2.resize(
        image, (DETECTOR_SIZE, DETECTOR_SIZE), interpolation=cv2.INTER_LINEAR
    )
    return resized.astype(np.float32).transpose(2, 0, 1) / 255.0


def crop_matrix(box: list[float], rng: random.Random | None = None) -> np.ndarray:
    """2x3 affine map from the image to the crop. In training the crop is
    shifted, scaled and turned at random; turning a dial does not change
    its reading, unlike mirroring it."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = max(box[2] - box[0], box[3] - box[1]) * MARGIN
    angle = 0.0
    if rng is not None:
        side *= rng.uniform(0.9, 1.15)
        cx += rng.uniform(-0.06, 0.06) * side
        cy += rng.uniform(-0.06, 0.06) * side
        angle = rng.uniform(-30.0, 30.0)
    m = cv2.getRotationMatrix2D((cx, cy), angle, CROP / side)
    m[0, 2] += CROP / 2 - cx
    m[1, 2] += CROP / 2 - cy
    return m


def apply(m: np.ndarray, point: list[float]) -> list[float]:
    return [
        float(m[0, 0] * point[0] + m[0, 1] * point[1] + m[0, 2]),
        float(m[1, 0] * point[0] + m[1, 1] * point[1] + m[1, 2]),
    ]


def crop(image: np.ndarray, m: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(
        image, m, (CROP, CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )


def crop_input(crop_rgb: np.ndarray) -> np.ndarray:
    """(3, CROP, CROP) float32, normalized as the ImageNet backbone expects."""
    x = (crop_rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1).copy()


def points_from_heatmaps(heatmaps: np.ndarray, m: np.ndarray) -> dict[str, Any]:
    """Heatmap peaks back in image pixels, through the inverse crop map."""
    inverse = cv2.invertAffineTransform(m)
    return {
        key: apply(inverse, [u * CROP / HEATMAP, v * CROP / HEATMAP])
        for key, (u, v, _) in zip(POINTS, heatmap_peaks(heatmaps), strict=True)
    }


def shrink(
    image: np.ndarray, gauges: list[dict[str, Any]]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Scale an image down so its longer side is at most KPRCNN_MAX_SIDE."""
    h, w = image.shape[:2]
    s = min(1.0, KPRCNN_MAX_SIDE / max(h, w))
    if s == 1.0:
        return image, gauges
    image = cv2.resize(
        image, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA
    )
    gauges = [{k: [v * s for v in values] for k, values in g.items()} for g in gauges]
    return image, gauges


def kprcnn_input(image: np.ndarray) -> tuple[np.ndarray, float]:
    """(3, H, W) float32 in [0, 1], and the factor back to the image's pixels."""
    small, _ = shrink(image, [])
    scale = image.shape[1] / small.shape[1]
    return small.astype(np.float32).transpose(2, 0, 1) / 255.0, scale

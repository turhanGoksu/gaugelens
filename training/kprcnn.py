"""Way 3 of D17: torchvision's Keypoint R-CNN finds each dial and its four
points together.

Training starts from the COCO person-keypoint checkpoint; its box and
keypoint heads are replaced for one class (the dial) and four points.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from training.common import color_jitter, load_rgb, random_crop
from training.data import POINTS, Sample
from training.preprocess import kprcnn_input, shrink
from training.torchutil import worker_init

# Torchvision ignores a point outside its box; marks on the rim can sit just
# outside a tight dial box, so the training boxes are widened by this share.
BOX_GROW = 0.1


def build_model(pretrained: bool = True):
    from torchvision.models.detection import (
        KeypointRCNN_ResNet50_FPN_Weights,
        keypointrcnn_resnet50_fpn,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

    model = keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None,
        weights_backbone=None,
    )
    model.roi_heads.box_predictor = FastRCNNPredictor(1024, 2)
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(512, len(POINTS))
    return model


def targets(gauges: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    boxes, keypoints = [], []
    for g in gauges:
        x0, y0, x1, y1 = g["box"]
        gx, gy = BOX_GROW * (x1 - x0), BOX_GROW * (y1 - y0)
        boxes.append([x0 - gx, y0 - gy, x1 + gx, y1 + gy])
        keypoints.append([[*g[key], 2.0] for key in POINTS])
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.ones(len(gauges), dtype=torch.int64),
        "keypoints": torch.tensor(keypoints, dtype=torch.float32).reshape(
            -1, len(POINTS), 3
        ),
    }


def to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


class KeypointData(Dataset):
    def __init__(
        self, samples: list[Sample], order: list[int], train: bool, seed: int = 0
    ):
        self.samples, self.order, self.train = samples, order, train
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, k: int):
        sample = self.samples[self.order[k]]
        image, gauges = load_rgb(sample["image"]), sample["gauges"]
        if self.train:
            image, gauges = random_crop(image, gauges, self.rng)
            image = color_jitter(image, self.rng)
        image, gauges = shrink(image, gauges)
        return to_tensor(image), targets(gauges)


def collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


def loader(samples, order, train, batch_size, workers, seed=0) -> DataLoader:
    return DataLoader(
        KeypointData(samples, order, train, seed),
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        collate_fn=collate,
        drop_last=train,
        persistent_workers=workers > 0,
        worker_init_fn=worker_init,
    )


def best_dial(boxes, scores, keypoints) -> dict[str, Any] | None:
    """The highest-scoring dial and its points, from the model's outputs."""
    if len(scores) == 0:
        return None
    i = int(np.argmax(scores))
    return {
        "score": float(scores[i]),
        "box": [float(v) for v in boxes[i]],
        "points": {
            key: [float(keypoints[i][j][0]), float(keypoints[i][j][1])]
            for j, key in enumerate(POINTS)
        },
    }


@torch.no_grad()
def predict(model, image: np.ndarray, device) -> dict[str, Any] | None:
    model.eval()
    x, scale = kprcnn_input(image)
    out = model([torch.from_numpy(x).to(device)])[0]
    dial = best_dial(
        out["boxes"].cpu().numpy() * scale,
        out["scores"].cpu().numpy(),
        out["keypoints"].cpu().numpy() * scale,
    )
    return dial

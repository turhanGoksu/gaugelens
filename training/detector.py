"""Way 1 of D17: a D-FINE detector finds each dial and its four points as
small boxes. Way 2 uses its dial boxes too.

D-FINE comes from Hugging Face ``transformers`` (Apache-2.0); training starts
from the COCO-trained ``ustc-community/dfine-small-coco`` checkpoint.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from training.common import color_jitter, dial_size, load_rgb, random_crop
from training.data import POINTS, Sample
from training.decode import CLASSES, detector_dials
from training.preprocess import detector_input
from training.torchutil import worker_init

CHECKPOINT = "ustc-community/dfine-small-coco"
# A point is labeled as a square box this share of its dial's size.
POINT_BOX = 0.06
MIN_POINT_BOX = 6.0


def targets(
    gauges: list[dict[str, Any]], width: int, height: int
) -> dict[str, torch.Tensor]:
    """D-FINE targets: class ids and normalized (cx, cy, w, h) boxes."""
    labels, boxes = [], []
    for g in gauges:
        x0, y0, x1, y1 = (
            max(0.0, g["box"][0]),
            max(0.0, g["box"][1]),
            min(float(width), g["box"][2]),
            min(float(height), g["box"][3]),
        )
        labels.append(0)
        boxes.append([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0])
        side = max(MIN_POINT_BOX, POINT_BOX * dial_size(g))
        for c, key in enumerate(POINTS, start=1):
            labels.append(c)
            boxes.append([g[key][0], g[key][1], side, side])
    b = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    b /= torch.tensor([width, height, width, height], dtype=torch.float32)
    return {
        "class_labels": torch.tensor(labels, dtype=torch.long),
        "boxes": b.clamp(0, 1),
    }


def to_pixels(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(detector_input(image))


class DetectionData(Dataset):
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
        h, w = image.shape[:2]
        return to_pixels(image), targets(gauges, w, h)


def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def build_model(pretrained: bool = True):
    from transformers import DFineConfig, DFineForObjectDetection

    # The class count follows from the label map.
    labels = {
        "id2label": dict(enumerate(CLASSES)),
        "label2id": {c: i for i, c in enumerate(CLASSES)},
    }
    if pretrained:
        return DFineForObjectDetection.from_pretrained(
            CHECKPOINT, ignore_mismatched_sizes=True, **labels
        )
    config = DFineConfig.from_pretrained(CHECKPOINT, **labels)
    return DFineForObjectDetection(config)


class Exported(torch.nn.Module):
    """What the ONNX file holds: per-class scores and boxes of every query."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor):
        out = self.model(pixel_values=pixel_values)
        return out.logits.sigmoid(), out.pred_boxes


def loader(samples, order, train, batch_size, workers, seed=0) -> DataLoader:
    return DataLoader(
        DetectionData(samples, order, train, seed),
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        collate_fn=collate,
        drop_last=train,
        persistent_workers=workers > 0,
        worker_init_fn=worker_init,
    )


@torch.no_grad()
def predict(model, image: np.ndarray, device) -> list[dict[str, Any]]:
    model.eval()
    h, w = image.shape[:2]
    out = model(pixel_values=to_pixels(image)[None].to(device))
    scores = out.logits.sigmoid()[0].cpu().numpy()
    boxes = out.pred_boxes[0].cpu().numpy()
    return detector_dials(scores, boxes, w, h)

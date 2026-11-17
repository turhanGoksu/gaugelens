"""Way 2 of D17: find the four points in a crop of the dial.

The dial box (from the detector at inference, from the label in training)
is cut out with a margin, scaled to CROP x CROP, and a SimpleBaseline-style
network (Xiao et al., 2018: a ResNet followed by three upsampling layers)
draws one heatmap per point. Training targets are Gaussian peaks.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from training.common import REAL_SHARE, color_jitter, load_rgb
from training.data import POINTS, Sample
from training.preprocess import (
    CROP,
    HEATMAP,
    apply,
    crop,
    crop_input,
    crop_matrix,
    points_from_heatmaps,
)
from training.torchutil import worker_init

SIGMA = 2.0  # target peak width, in heatmap pixels


MARGIN = 1.3  # crop side as a multiple of the dial size
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def target_maps(points: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian peaks at the points (crop pixels), and a weight of 0 for a
    point that falls outside the crop."""
    ys, xs = np.mgrid[0:HEATMAP, 0:HEATMAP].astype(np.float32)
    maps = np.zeros((len(points), HEATMAP, HEATMAP), np.float32)
    weights = np.zeros(len(points), np.float32)
    for i, (x, y) in enumerate(points):
        u, v = x * HEATMAP / CROP, y * HEATMAP / CROP
        if 0 <= u < HEATMAP and 0 <= v < HEATMAP:
            maps[i] = np.exp(-((xs - u) ** 2 + (ys - v) ** 2) / (2 * SIGMA**2))
            weights[i] = 1.0
    return maps, weights


def dial_order(samples: list[Sample], rng: random.Random) -> list[tuple[int, int]]:
    """(sample, dial) pairs for one pass; real dials repeated to REAL_SHARE."""
    pairs = [(i, j) for i, s in enumerate(samples) for j in range(len(s["gauges"]))]
    synthetic = [p for p in pairs if samples[p[0]]["source"] == "synthetic"]
    real = [p for p in pairs if samples[p[0]]["source"] != "synthetic"]
    repeat = 1
    if synthetic and real:
        repeat = max(
            1, round(REAL_SHARE / (1 - REAL_SHARE) * len(synthetic) / len(real))
        )
    order = synthetic + real * repeat
    rng.shuffle(order)
    return order


class CropData(Dataset):
    def __init__(self, samples, order, train: bool, seed: int = 0):
        self.samples, self.order, self.train = samples, order, train
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, k: int):
        i, j = self.order[k]
        gauge = self.samples[i]["gauges"][j]
        image = load_rgb(self.samples[i]["image"])
        m = crop_matrix(gauge["box"], self.rng if self.train else None)
        patch = crop(image, m)
        if self.train:
            patch = color_jitter(patch, self.rng)
        maps, weights = target_maps([apply(m, gauge[key]) for key in POINTS])
        return (
            torch.from_numpy(crop_input(patch)),
            torch.from_numpy(maps),
            torch.from_numpy(weights),
        )


class HeatmapNet(nn.Module):
    """ResNet body (stride 32) and three 2x upsamplings: 256 -> 8 -> 64."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        body = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.body = nn.Sequential(*list(body.children())[:-2])
        layers: list[nn.Module] = []
        channels = 512
        for _ in range(3):
            layers += [
                nn.ConvTranspose2d(channels, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ]
            channels = 256
        self.head = nn.Sequential(*layers, nn.Conv2d(256, len(POINTS), 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def loss_fn(
    pred: torch.Tensor, maps: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    per_point = ((pred - maps) ** 2).mean(dim=(2, 3))
    return (per_point * weights).sum() / weights.sum().clamp(min=1.0)


def loader(samples, order, train, batch_size, workers, seed=0) -> DataLoader:
    return DataLoader(
        CropData(samples, order, train, seed),
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        drop_last=train,
        persistent_workers=workers > 0,
        worker_init_fn=worker_init,
    )


@torch.no_grad()
def predict(model, image: np.ndarray, box: list[float], device) -> dict[str, Any]:
    model.eval()
    m = crop_matrix(box)
    x = torch.from_numpy(crop_input(crop(image, m)))[None].to(device)
    return points_from_heatmaps(model(x)[0].cpu().numpy(), m)

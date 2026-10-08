"""Write synthetic scenes to disk as a dataset.

Layout of a dataset folder::

    dataset.json   how it was made: split, seeds, size, generator version,
                   background photos used
    labels.jsonl   one JSON object per image, in seed order
    images/        <seed>.jpg

Each scene depends only on its seed, so a dataset can be regenerated exactly
from ``dataset.json`` and the same gaugelens version, and parallel workers
produce the same files as a single one.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gaugelens import __version__
from gaugelens.synth.backgrounds import BackgroundPool, Split
from gaugelens.synth.capture import render_scene

# Train and dev seeds come from ranges that can never overlap.
SPLIT_SEED_START: dict[str, int] = {"train": 0, "dev": 100_000_000}
JPEG_QUALITY = 95


@dataclass(frozen=True)
class DatasetConfig:
    split: Split
    count: int
    seed_start: int | None = None
    """Defaults to the split's own range (SPLIT_SEED_START)."""
    width: int = 640
    height: int = 640
    out_of_range: float = 0.1
    backgrounds: str | None = None
    """Folder of background photos; None means procedural backgrounds only."""
    photo_probability: float = 0.7

    @property
    def seeds(self) -> range:
        start = (
            SPLIT_SEED_START[self.split] if self.seed_start is None else self.seed_start
        )
        return range(start, start + self.count)


_pools: dict[tuple[str | None, str, float], BackgroundPool] = {}


def _pool(config: DatasetConfig) -> BackgroundPool:
    """One pool per worker process, so its image cache is reused."""
    key = (config.backgrounds, config.split, config.photo_probability)
    if key not in _pools:
        if config.backgrounds is None:
            _pools[key] = BackgroundPool()
        else:
            _pools[key] = BackgroundPool.from_dir(
                config.backgrounds, config.photo_probability, split=config.split
            )
    return _pools[key]


def _render_one(job: tuple[int, str, DatasetConfig]) -> dict[str, Any]:
    seed, out_dir, config = job
    scene = render_scene(
        seed, _pool(config), config.width, config.height, config.out_of_range
    )
    name = f"images/{seed:09d}.jpg"
    scene.image.save(Path(out_dir) / name, quality=JPEG_QUALITY)
    return {"image": name, "seed": seed, **scene.labels.to_dict()}


def write_dataset(out_dir: str | Path, config: DatasetConfig, workers: int = 1) -> Path:
    """Render ``config.count`` scenes into ``out_dir`` and return its path."""
    out_dir = Path(out_dir)
    if (out_dir / "dataset.json").exists():
        raise FileExistsError(f"{out_dir} already holds a dataset; use a new folder")
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    pool = _pool(config)
    if config.backgrounds is not None and not pool.photos:
        raise ValueError(f"no {config.split} background photos in {config.backgrounds}")

    jobs = [(seed, str(out_dir), config) for seed in config.seeds]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_render_one, jobs, chunksize=16))
    else:
        rows = [_render_one(job) for job in jobs]

    with open(out_dir / "labels.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    metadata = {
        "gaugelens_version": __version__,
        "config": asdict(config),
        "seeds": [config.seeds.start, config.seeds.stop],
        "image_format": f"JPEG quality {JPEG_QUALITY}",
        "background_photos": sorted(p.name for p, _ in pool.photos),
    }
    (out_dir / "dataset.json").write_text(json.dumps(metadata, indent=1))
    return out_dir

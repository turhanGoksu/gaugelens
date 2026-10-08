"""Backgrounds for synthetic scenes: CC0 photos from Poly Haven and
procedural patterns.

Photos are downloaded once with :func:`download_polyhaven` into a folder
whose ``manifest.json`` records each file's source, authors and license.
Poly Haven assets are CC0. The Poly Haven API asks that users be told the
assets came from Poly Haven, hence the manifest and the credit in the README.
"""

from __future__ import annotations

import json
import math
import urllib.request
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from gaugelens import __version__

POLYHAVEN_API = "https://api.polyhaven.com"
_DOWNLOAD_PREFIX = "https://dl.polyhaven.org/"
USER_AGENT = f"gaugelens/{__version__} (+https://github.com/turhanGoksu/gaugelens)"

# Indoor HDRIs whose id or tags suggest an industrial scene.
_HDRI_KEYWORDS = (
    "industrial",
    "factory",
    "warehouse",
    "workshop",
    "garage",
    "pipe",
    "boiler",
    "machine",
    "powerplant",
    "hangar",
)
_TEXTURE_CATEGORIES = ("metal", "concrete", "plaster-concrete", "brick", "industrial")
# Panoramas are stored downscaled; crops come from their middle band, where
# the equirectangular projection distorts least.
_PANORAMA_WIDTH = 3072
_PANORAMA_BAND = (0.3, 0.7)

Fetch = Callable[[str], bytes]


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _spread(ids: list[str], count: int) -> list[str]:
    """Up to ``count`` ids spread evenly over the sorted list."""
    if len(ids) <= count:
        return ids
    return [ids[i * len(ids) // count] for i in range(count)]


def download_polyhaven(
    dest: str | Path,
    max_panoramas: int = 30,
    max_textures: int = 60,
    fetch: Fetch = _fetch,
) -> list[dict[str, str]]:
    """Download industrial indoor panoramas and surface textures from Poly
    Haven into ``dest`` and write ``manifest.json``. Files already listed in
    the manifest are skipped, so the call can be repeated after a failure."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "manifest.json"
    entries: dict[str, dict[str, str]] = {}
    if manifest_path.exists():
        entries = {e["file"]: e for e in json.loads(manifest_path.read_text())}

    hdris = json.loads(fetch(f"{POLYHAVEN_API}/assets?type=hdris&categories=indoor"))
    panorama_ids = sorted(
        asset_id
        for asset_id, asset in hdris.items()
        if any(w in asset_id or w in asset.get("tags", []) for w in _HDRI_KEYWORDS)
    )
    textures: dict[str, dict] = {}
    for category in _TEXTURE_CATEGORIES:
        url = f"{POLYHAVEN_API}/assets?type=textures&categories={category}"
        textures.update(json.loads(fetch(url)))

    jobs = [(i, hdris[i], "panorama") for i in _spread(panorama_ids, max_panoramas)]
    jobs += [
        (i, textures[i], "texture") for i in _spread(sorted(textures), max_textures)
    ]
    for asset_id, asset, kind in jobs:
        name = f"{kind}_{asset_id}.jpg"
        if name in entries and (dest / name).exists():
            continue
        files = json.loads(fetch(f"{POLYHAVEN_API}/files/{asset_id}"))
        try:
            if kind == "panorama":
                url = files["tonemapped"]["url"]
            else:
                url = files["Diffuse"]["1k"]["jpg"]["url"]
        except KeyError:
            continue  # no file in the format we use
        if not url.startswith(_DOWNLOAD_PREFIX):
            raise ValueError(f"unexpected download host: {url}")

        image = cv2.imdecode(np.frombuffer(fetch(url), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        if kind == "panorama" and image.shape[1] > _PANORAMA_WIDTH:
            height = round(image.shape[0] * _PANORAMA_WIDTH / image.shape[1])
            image = cv2.resize(
                image, (_PANORAMA_WIDTH, height), interpolation=cv2.INTER_AREA
            )
        cv2.imwrite(str(dest / name), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        entries[name] = {
            "file": name,
            "kind": kind,
            "asset_id": asset_id,
            "name": asset.get("name", asset_id),
            "authors": ", ".join(asset.get("authors", {})),
            "url": url,
            "page": f"https://polyhaven.com/a/{asset_id}",
            "license": "CC0-1.0",
            "provider": "Poly Haven",
        }
        # Written after every file, so an interrupted run keeps its progress.
        manifest_path.write_text(
            json.dumps(sorted(entries.values(), key=lambda e: e["file"]), indent=1)
        )
    return sorted(entries.values(), key=lambda e: e["file"])


@lru_cache(maxsize=16)
def _load_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read background image {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class BackgroundPool:
    """Samples backgrounds: with probability ``photo_probability`` a random
    crop of a photo (when there are photos), otherwise a procedural pattern."""

    def __init__(
        self, photos: Sequence[tuple[Path, bool]] = (), photo_probability: float = 0.7
    ) -> None:
        """``photos`` are (path, is_panorama) pairs."""
        self.photos = list(photos)
        self.photo_probability = photo_probability

    @classmethod
    def from_dir(
        cls, directory: str | Path, photo_probability: float = 0.7
    ) -> BackgroundPool:
        """Photos listed in ``manifest.json``, or every JPEG/PNG without one."""
        directory = Path(directory)
        manifest = directory / "manifest.json"
        if manifest.exists():
            photos = [
                (directory / e["file"], e["kind"] == "panorama")
                for e in json.loads(manifest.read_text())
            ]
        else:
            paths = sorted(directory.glob("*.jp*g")) + sorted(directory.glob("*.png"))
            photos = [(p, False) for p in paths]
        return cls(photos, photo_probability)

    def sample(
        self, rng: np.random.Generator, width: int, height: int
    ) -> tuple[np.ndarray, str]:
        """A (height, width, 3) float32 RGB image in [0, 1] and its source."""
        if self.photos and rng.random() < self.photo_probability:
            path, panorama = self.photos[rng.integers(len(self.photos))]
            return _crop(
                rng, _load_rgb(str(path)), panorama, width, height
            ), f"photo:{path.name}"
        kind = str(rng.choice(["gradient", "blotches", "pipes"]))
        image = _PROCEDURAL[kind](rng, width, height).astype(np.float32, copy=False)
        return image, f"procedural:{kind}"


def _crop(
    rng: np.random.Generator, image: np.ndarray, panorama: bool, width: int, height: int
) -> np.ndarray:
    rows, cols = image.shape[:2]
    top, bottom = (0, rows)
    if panorama:
        top, bottom = round(rows * _PANORAMA_BAND[0]), round(rows * _PANORAMA_BAND[1])
    # A random zoom: the crop covers 35-100% of the usable height.
    crop_h = int((bottom - top) * rng.uniform(0.35, 1.0))
    crop_w = int(crop_h * width / height)
    if crop_w > cols:
        crop_w, crop_h = cols, int(cols * height / width)
    y = int(rng.integers(top, bottom - crop_h + 1))
    x = int(rng.integers(0, cols - crop_w + 1))
    crop = image[y : y + crop_h, x : x + crop_w]
    if rng.random() < 0.5:
        crop = crop[:, ::-1]
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def _random_color(rng: np.random.Generator) -> np.ndarray:
    """Mostly muted colors, as on walls, machines and floors."""
    gray = rng.uniform(0.1, 0.9)
    saturation = rng.uniform(0.0, 0.6)
    color = gray + saturation * (rng.uniform(0.0, 1.0, 3) - gray)
    return np.clip(color, 0, 1).astype(np.float32)


def _gradient(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    angle = rng.uniform(0, 2 * math.pi)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    ramp = xs * math.cos(angle) + ys * math.sin(angle)
    ramp = (ramp - ramp.min()) / max(float(np.ptp(ramp)), 1e-6)
    a, b = _random_color(rng), _random_color(rng)
    return a * (1 - ramp[..., None]) + b * ramp[..., None]


def _blotches(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    """Low-frequency color noise with fine grain on top."""
    cells = int(rng.integers(3, 12))
    grid = np.stack([_random_color(rng) for _ in range(cells * cells)]).reshape(
        cells, cells, 3
    )
    image = cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC)
    image += rng.normal(0, 0.03, image.shape).astype(np.float32)
    return np.clip(image, 0, 1)


def _pipes(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    """A plain wall crossed by thick shaded bars, like pipes or rails."""
    image = np.empty((height, width, 3), np.float32)
    image[:] = _random_color(rng)
    for _ in range(int(rng.integers(1, 6))):
        color = _random_color(rng)
        thickness = int(rng.uniform(0.03, 0.15) * min(width, height))
        p0 = (
            int(rng.integers(-width, 2 * width)),
            int(rng.integers(-height, 2 * height)),
        )
        p1 = (
            int(rng.integers(-width, 2 * width)),
            int(rng.integers(-height, 2 * height)),
        )
        for step, shade in (
            (1.0, 0.7),
            (0.6, 1.0),
            (0.2, 1.25),
        ):  # darker rim, lighter core
            cv2.line(
                image, p0, p1, (color * shade).tolist(), max(1, int(thickness * step))
            )
    return np.clip(cv2.GaussianBlur(image, (0, 0), 1.0), 0, 1)


_PROCEDURAL = {"gradient": _gradient, "blotches": _blotches, "pipes": _pipes}

"""Put a rendered dial into a photo-like scene: camera angle, background,
lighting, glare, shadow, blur, noise and JPEG compression.

Every geometric change is one homography ``H`` (a 3x3 matrix) that maps
dial-image pixels to scene pixels. The keypoint labels go through the same
``H``, so they stay exact. The value does not change: tilting the camera
moves pixels, not the needle.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from gaugelens.synth.backgrounds import BackgroundPool
from gaugelens.synth.dial import (
    Point,
    RenderedDial,
    render_dial,
    sample_spec,
    sample_value,
)


@dataclass(frozen=True)
class CaptureParams:
    """How the scene is shot. Recorded in the labels so that errors can later
    be broken down by condition (e.g. by tilt or blur)."""

    width: int
    height: int
    dial_diameter: float
    """Target size of the dial in the scene, in pixels."""
    tilt_x: float
    """Degrees the dial plane is turned about the horizontal axis (top/bottom)."""
    tilt_y: float
    """Degrees the dial plane is turned about the vertical axis (left/right)."""
    roll: float
    """In-plane rotation in degrees, clockwise."""
    focal: float
    """Focal length in units of the dial image size; smaller = stronger perspective."""
    glare: float
    shadow: float
    light_gradient: float
    brightness: float
    contrast: float
    color_gains: tuple[float, float, float]
    blur_sigma: float
    motion_blur: float
    """Motion-blur kernel length in pixels; 0 means none."""
    noise_sigma: float
    jpeg_quality: int | None


@dataclass(frozen=True)
class SceneLabels:
    """Ground truth for one scene, in scene pixel coordinates."""

    value: float
    fraction: float
    min_value: float
    max_value: float
    unit: str
    center: Point
    """Where the dial's center lands. Under perspective this is not the
    center of the ellipse the dial becomes."""
    needle_tip: Point
    min_mark: Point
    max_mark: Point
    bbox: tuple[float, float, float, float]
    homography: tuple[tuple[float, float, float], ...]
    """Maps dial-image pixels to scene pixels; its inverse undoes the camera."""
    background: str
    capture: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Scene:
    image: Image.Image
    labels: SceneLabels


def sample_capture(
    rng: np.random.Generator, width: int = 640, height: int = 640
) -> CaptureParams:
    """Draw shooting conditions (domain randomization of the capture)."""
    short = min(width, height)

    def sometimes(probability: float, low: float, high: float) -> float:
        return float(rng.uniform(low, high)) if rng.random() < probability else 0.0

    return CaptureParams(
        width=width,
        height=height,
        dial_diameter=float(rng.uniform(0.3, 0.92) * short),
        tilt_x=float(np.clip(rng.normal(0, 14), -40, 40)),
        tilt_y=float(np.clip(rng.normal(0, 14), -40, 40)),
        roll=float(np.clip(rng.normal(0, 6), -25, 25)),
        focal=float(rng.uniform(1.0, 3.0)),
        glare=sometimes(0.4, 0.2, 0.8),
        shadow=sometimes(0.3, 0.2, 0.6),
        light_gradient=float(rng.uniform(0.0, 0.35)),
        brightness=float(rng.uniform(0.65, 1.25)),
        contrast=float(rng.uniform(0.75, 1.15)),
        color_gains=tuple(
            float(g) for g in np.clip(rng.normal(1, 0.05, 3), 0.85, 1.15)
        ),
        blur_sigma=sometimes(0.5, 0.3, 1.8) * short / 640,
        motion_blur=sometimes(0.15, 2.0, 10.0),
        noise_sigma=sometimes(0.7, 0.003, 0.03),
        jpeg_quality=int(rng.integers(35, 96)) if rng.random() < 0.8 else None,
    )


def _rotation(tilt_x: float, tilt_y: float, roll: float) -> np.ndarray:
    a, b, g = (math.radians(v) for v in (tilt_x, tilt_y, roll))
    rx = np.array(
        [[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]]
    )
    ry = np.array(
        [[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]]
    )
    rz = np.array(
        [[math.cos(g), -math.sin(g), 0], [math.sin(g), math.cos(g), 0], [0, 0, 1]]
    )
    return rz @ rx @ ry


def _transform(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply homography ``h`` to (N, 2) points."""
    ones = np.ones((len(points), 1))
    mapped = np.hstack([points, ones]) @ h.T
    return mapped[:, :2] / mapped[:, 2:3]


def _circle(center: Point, radius: float, count: int = 72) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return np.stack(
        [center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)], axis=1
    )


def _homography(
    size: int,
    center: Point,
    radius: float,
    params: CaptureParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pinhole camera looking at the tilted dial plane, then scaled to the
    target diameter and placed at a random position inside the frame."""
    src = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float64
    )
    focal = params.focal * size
    rotation = _rotation(params.tilt_x, params.tilt_y, params.roll)

    def project(points: np.ndarray) -> np.ndarray:
        plane = np.hstack([points - np.array(center), np.zeros((len(points), 1))])
        camera = plane @ rotation.T + np.array([0.0, 0.0, focal])
        return focal * camera[:, :2] / camera[:, 2:3]

    rim = project(_circle(center, radius))
    scale = params.dial_diameter / float(np.ptp(rim, axis=0).max())
    rim *= scale
    offset = []
    for axis, extent in ((0, params.width), (1, params.height)):
        low, high = -rim[:, axis].min(), extent - 1 - rim[:, axis].max()
        offset.append(rng.uniform(low, high) if low <= high else (low + high) / 2)
    dst = project(src) * scale + np.array(offset)
    return cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))


def _add_glare(
    rgba: np.ndarray,
    center: Point,
    radius: float,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """A soft elliptical highlight on the glass, in dial-image coordinates."""
    size = rgba.shape[0]
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    r = 0.7 * radius * math.sqrt(rng.random())
    t = rng.uniform(0, 2 * math.pi)
    gx, gy = center[0] + r * math.cos(t), center[1] + r * math.sin(t)
    a = rng.uniform(0.15, 0.6) * radius
    b = a * rng.uniform(0.15, 1.0)
    phi = rng.uniform(0, math.pi)
    u = (xs - gx) * math.cos(phi) + (ys - gy) * math.sin(phi)
    v = -(xs - gx) * math.sin(phi) + (ys - gy) * math.cos(phi)
    mask = strength * np.exp(-2.0 * ((u / a) ** 2 + (v / b) ** 2))
    out = rgba.copy()
    out[..., :3] += mask[..., None] * (1.0 - out[..., :3])
    return out


def _lighting(
    image: np.ndarray, params: CaptureParams, rng: np.random.Generator
) -> np.ndarray:
    height, width = image.shape[:2]
    image = image * np.array(params.color_gains, np.float32) * params.brightness
    mean = float(image.mean())
    image = (image - mean) * params.contrast + mean
    if params.light_gradient > 0:
        angle = rng.uniform(0, 2 * math.pi)
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        ramp = xs * math.cos(angle) + ys * math.sin(angle)
        ramp = 2 * (ramp - ramp.min()) / max(float(np.ptp(ramp)), 1e-6) - 1
        image = image * (1 + params.light_gradient * ramp)[..., None]
    if params.shadow > 0:
        # A soft-edged half-plane, as if something blocks part of the light.
        angle = rng.uniform(0, 2 * math.pi)
        px, py = rng.uniform(0, width), rng.uniform(0, height)
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        distance = (xs - px) * math.cos(angle) + (ys - py) * math.sin(angle)
        softness = rng.uniform(0.01, 0.1) * min(width, height)
        mask = 1 / (1 + np.exp(-distance / softness))
        image = image * (1 - params.shadow * mask)[..., None]
    return image


def _motion_blur(image: np.ndarray, length: float, angle: float) -> np.ndarray:
    size = max(3, int(round(length)) | 1)
    kernel = np.zeros((size, size), np.float32)
    c = size // 2
    dx, dy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    p0 = (round(c - dx * c), round(c - dy * c))
    p1 = (round(c + dx * c), round(c + dy * c))
    cv2.line(kernel, p0, p1, 1.0, 1)
    return cv2.filter2D(image, -1, kernel / kernel.sum())


def capture(
    dial: RenderedDial,
    params: CaptureParams,
    background: np.ndarray,
    rng: np.random.Generator,
    background_source: str = "",
) -> Scene:
    """Shoot ``dial`` under ``params`` in front of ``background``."""
    labels = dial.labels
    rgba = np.asarray(dial.image, dtype=np.float32) / 255.0
    size = rgba.shape[0]
    if params.glare > 0:
        rgba = _add_glare(rgba, labels.center, labels.radius * 0.9, params.glare, rng)
    # Premultiply alpha so that interpolation at the rim does not pull in
    # the black of the transparent pixels.
    rgba[..., :3] *= rgba[..., 3:4]

    h = _homography(size, labels.center, labels.radius, params, rng)
    warped = cv2.warpPerspective(
        rgba,
        h,
        (params.width, params.height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    image = background * (1 - warped[..., 3:4]) + warped[..., :3]

    image = _lighting(image, params, rng)
    if params.blur_sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), params.blur_sigma)
    if params.motion_blur > 0:
        image = _motion_blur(image, params.motion_blur, rng.uniform(0, 180))
    if params.noise_sigma > 0:
        image = image + rng.normal(0, params.noise_sigma, image.shape).astype(
            np.float32
        )
    pixels = (np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8)
    if params.jpeg_quality is not None:
        bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, params.jpeg_quality]
        )
        assert ok
        pixels = cv2.cvtColor(
            cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        )

    keypoints = np.array(
        [labels.center, labels.needle_tip, labels.min_mark, labels.max_mark]
    )
    center, tip, min_mark, max_mark = (
        tuple(p) for p in _transform(h, keypoints).tolist()
    )
    rim = _transform(h, _circle(labels.center, labels.radius))
    scene_labels = SceneLabels(
        value=labels.value,
        fraction=labels.fraction,
        min_value=labels.min_value,
        max_value=labels.max_value,
        unit=labels.unit,
        center=center,
        needle_tip=tip,
        min_mark=min_mark,
        max_mark=max_mark,
        bbox=(*rim.min(axis=0).tolist(), *rim.max(axis=0).tolist()),
        homography=tuple(tuple(row) for row in h.tolist()),
        background=background_source,
        capture=asdict(params),
    )
    return Scene(image=Image.fromarray(pixels), labels=scene_labels)


def render_scene(
    seed: int,
    pool: BackgroundPool | None = None,
    width: int = 640,
    height: int = 640,
    out_of_range: float = 0.1,
) -> Scene:
    """Sample a dial, a value, shooting conditions and a background from
    ``seed`` and render the scene."""
    rng = np.random.default_rng(seed)
    spec = sample_spec(rng)
    value = sample_value(spec.scale, rng, out_of_range)
    params = sample_capture(rng, width, height)
    # Render the dial near its final size, so the warp barely rescales it.
    dial = render_dial(spec, value, size=max(64, round(params.dial_diameter / 0.96)))
    background, source = (pool or BackgroundPool()).sample(rng, width, height)
    return capture(dial, params, background, rng, source)

import dataclasses
import json
import math
import urllib.request

import cv2
import numpy as np
import pytest

from gaugelens.geometry import GaugeScale, point_to_angle
from gaugelens.synth import (
    BackgroundPool,
    CaptureParams,
    backgrounds,
    capture,
    download_polyhaven,
    render_dial,
    render_scene,
    sample_spec,
)

GRAY = np.full((480, 640, 3), 0.5, np.float32)


def plain(**overrides) -> CaptureParams:
    """Shooting conditions with every photometric effect off."""
    base = {
        "width": 640,
        "height": 480,
        "dial_diameter": 400.0,
        "tilt_x": 0.0,
        "tilt_y": 0.0,
        "roll": 0.0,
        "focal": 2.0,
        "glare": 0.0,
        "shadow": 0.0,
        "light_gradient": 0.0,
        "brightness": 1.0,
        "contrast": 1.0,
        "color_gains": (1.0, 1.0, 1.0),
        "blur_sigma": 0.0,
        "motion_blur": 0.0,
        "noise_sigma": 0.0,
        "jpeg_quality": None,
    }
    return CaptureParams(**(base | overrides))


def red_needle_dial(fraction: float):
    spec = dataclasses.replace(
        sample_spec(np.random.default_rng(0)),
        face_color=(255, 255, 255),
        needle_color=(255, 0, 0),
        hub_color=(255, 0, 0),
        needle_style="line",
    )
    value = spec.scale.min_value + fraction * spec.scale.span
    return render_dial(spec, value, size=420), spec


def to_scene(homography, point) -> tuple[float, float]:
    x, y, w = np.array(homography) @ np.array([point[0], point[1], 1.0])
    return x / w, y / w


def to_dial(homography, point) -> tuple[float, float]:
    return to_scene(np.linalg.inv(np.array(homography)), point)


def reading_error(labels, points) -> float:
    """|error| in fractions of full scale when reading from ``points``."""
    center, tip, min_mark, max_mark = points
    scale = GaugeScale(
        min_value=labels.min_value,
        max_value=labels.max_value,
        min_angle=point_to_angle(center, min_mark),
        max_angle=point_to_angle(center, max_mark),
    )
    reading = scale.read_angle(point_to_angle(center, tip), edge_margin=0.0)
    return abs(reading.fraction - labels.fraction)


def test_same_seed_renders_identical_scene() -> None:
    a, b = render_scene(5), render_scene(5)
    assert a.labels == b.labels
    assert a.image.tobytes() == b.image.tobytes()


def test_keypoints_follow_the_needle_under_perspective() -> None:
    dial, spec = red_needle_dial(0.5)  # needle at 12 o'clock
    params = plain(tilt_x=25.0, tilt_y=-30.0, roll=10.0)
    scene = capture(dial, params, GRAY, np.random.default_rng(1))
    pixels = np.asarray(scene.image).astype(int)
    cx, cy = dial.labels.center
    face = dial.labels.radius * (1 - spec.bezel_width)

    def scene_color(angle: float) -> np.ndarray:
        a = math.radians(angle)
        x, y = to_scene(
            scene.labels.homography,
            (cx + 0.45 * face * math.sin(a), cy - 0.45 * face * math.cos(a)),
        )
        return pixels[round(y), round(x)]

    red = np.array([255, 0, 0])
    assert np.abs(scene_color(0.0) - red).max() < 40
    assert np.abs(scene_color(90.0) - red).max() > 100


def test_tilt_breaks_naive_angles_but_rectification_reads_exactly() -> None:
    dial, _ = red_needle_dial(0.75)
    scene = capture(dial, plain(tilt_y=35.0), GRAY, np.random.default_rng(2))
    labels = scene.labels
    points = (labels.center, labels.needle_tip, labels.min_mark, labels.max_mark)
    # Angles measured directly in the tilted photo are off by more than 1% FS.
    assert reading_error(labels, points) > 0.01
    # Undoing the camera (inverse homography) gives back the exact reading.
    rectified = [to_dial(labels.homography, p) for p in points]
    assert reading_error(labels, rectified) < 1e-9


def test_frontal_capture_keeps_the_target_size() -> None:
    dial, _ = red_needle_dial(0.3)
    x0, y0, x1, y1 = capture(dial, plain(), GRAY, np.random.default_rng(3)).labels.bbox
    assert x1 - x0 == pytest.approx(400.0, abs=0.5)
    assert y1 - y0 == pytest.approx(400.0, abs=0.5)


@pytest.mark.parametrize("seed", range(40))
def test_dial_and_keypoints_stay_inside_the_frame(seed) -> None:
    scene = render_scene(seed)
    labels = scene.labels
    x0, y0, x1, y1 = labels.bbox
    width, height = scene.image.size
    assert -0.5 <= x0 and x1 <= width - 0.5
    assert -0.5 <= y0 and y1 <= height - 0.5
    for x, y in (labels.center, labels.needle_tip, labels.min_mark, labels.max_mark):
        assert x0 <= x <= x1
        assert y0 <= y <= y1


def test_procedural_backgrounds_have_the_requested_shape() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        image, source = BackgroundPool().sample(rng, 320, 200)
        assert image.shape == (200, 320, 3)
        assert image.dtype == np.float32
        assert 0.0 <= image.min() and image.max() <= 1.0
        assert source.startswith("procedural:")


def _write_jpeg(path, width: int, height: int) -> None:
    image = np.random.default_rng(0).integers(
        0, 255, (height, width, 3), dtype=np.uint8
    )
    cv2.imwrite(str(path), image)


def test_pool_uses_photos_from_the_manifest(tmp_path) -> None:
    _write_jpeg(tmp_path / "panorama_a.jpg", 400, 200)
    _write_jpeg(tmp_path / "texture_b.jpg", 128, 128)
    manifest = [
        {"file": "panorama_a.jpg", "kind": "panorama"},
        {"file": "texture_b.jpg", "kind": "texture"},
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    pool = BackgroundPool.from_dir(tmp_path, photo_probability=1.0)
    rng = np.random.default_rng(0)
    sources = {pool.sample(rng, 64, 48)[1] for _ in range(30)}
    assert sources == {"photo:panorama_a.jpg", "photo:texture_b.jpg"}


def test_pool_without_manifest_uses_every_image(tmp_path) -> None:
    _write_jpeg(tmp_path / "wall.jpg", 100, 100)
    pool = BackgroundPool.from_dir(tmp_path)
    assert [p.name for p, _ in pool.photos] == ["wall.jpg"]


class FakePolyHaven:
    """Answers the few Poly Haven API calls the downloader makes."""

    def __init__(self, image_host: str = "https://dl.polyhaven.org/") -> None:
        ok, jpeg = cv2.imencode(".jpg", np.zeros((50, 100, 3), np.uint8))
        self.jpeg = jpeg.tobytes()
        self.image_host = image_host
        self.downloads: list[str] = []

    def __call__(self, url: str) -> bytes:
        api = backgrounds.POLYHAVEN_API
        if url == f"{api}/assets?type=hdris&categories=indoor":
            return json.dumps(
                {
                    "boiler_room": {
                        "name": "Boiler Room",
                        "tags": [],
                        "authors": {"A": "All"},
                    },
                    "cozy_bedroom": {"name": "Cozy Bedroom", "tags": ["home"]},
                }
            ).encode()
        if url.startswith(f"{api}/assets?type=textures"):
            return json.dumps({"rusty_metal": {"name": "Rusty Metal"}}).encode()
        if url == f"{api}/files/boiler_room":
            return json.dumps(
                {"tonemapped": {"url": f"{self.image_host}boiler.jpg"}}
            ).encode()
        if url == f"{api}/files/rusty_metal":
            image = {"jpg": {"url": f"{self.image_host}rust.jpg"}}
            return json.dumps({"Diffuse": {"1k": image}}).encode()
        self.downloads.append(url)
        return self.jpeg


def test_download_writes_images_and_a_cc0_manifest(tmp_path) -> None:
    fake = FakePolyHaven()
    entries = download_polyhaven(tmp_path, fetch=fake)
    assert [e["file"] for e in entries] == [
        "panorama_boiler_room.jpg",
        "texture_rusty_metal.jpg",
    ]
    assert {e["license"] for e in entries} == {"CC0-1.0"}
    assert {e["provider"] for e in entries} == {"Poly Haven"}
    assert json.loads((tmp_path / "manifest.json").read_text()) == entries
    assert (tmp_path / "panorama_boiler_room.jpg").exists()

    # A second run finds everything in the manifest and downloads nothing.
    fake.downloads.clear()
    download_polyhaven(tmp_path, fetch=fake)
    assert fake.downloads == []


def test_download_refuses_unexpected_hosts(tmp_path) -> None:
    with pytest.raises(ValueError, match="unexpected download host"):
        download_polyhaven(
            tmp_path, fetch=FakePolyHaven(image_host="https://example.com/")
        )


def test_requests_carry_a_gaugelens_user_agent(monkeypatch) -> None:
    seen = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request, timeout):
        seen.append(request.get_header("User-agent"))
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    backgrounds._fetch("https://api.polyhaven.com/assets")
    assert seen[0].startswith("gaugelens/")

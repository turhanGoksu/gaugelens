import dataclasses
import math
import subprocess
import sys

import numpy as np
import pytest

from gaugelens.geometry import GaugeScale, point_to_angle
from gaugelens.synth import render_dial, render_random, sample_spec, sample_value

SEEDS = range(200)


def test_same_seed_renders_identical_dial() -> None:
    a, b = render_random(7), render_random(7)
    assert a.labels == b.labels
    assert a.image.tobytes() == b.image.tobytes()


@pytest.mark.parametrize("seed", SEEDS)
def test_labeled_keypoints_read_back_the_labeled_value(seed) -> None:
    labels = render_random(seed).labels
    scale = GaugeScale(
        min_value=labels.min_value,
        max_value=labels.max_value,
        min_angle=point_to_angle(labels.center, labels.min_mark),
        max_angle=point_to_angle(labels.center, labels.max_mark),
    )
    reading = scale.read_angle(
        point_to_angle(labels.center, labels.needle_tip), edge_margin=0
    )
    assert reading.fraction == pytest.approx(labels.fraction, abs=1e-9)
    if 0.0 <= labels.fraction <= 1.0:
        assert reading.value == pytest.approx(labels.value, abs=1e-9 * scale.span)
    else:
        expected = "below_range" if labels.fraction < 0 else "above_range"
        assert reading.position == expected


def test_needle_is_drawn_where_the_labels_say() -> None:
    spec = dataclasses.replace(
        sample_spec(np.random.default_rng(0)),
        face_color=(255, 255, 255),
        needle_color=(255, 0, 0),
        hub_color=(255, 0, 0),
        needle_style="line",
    )
    scale = spec.scale
    # The sampled sweeps are symmetric about 6 o'clock, so mid-scale is 12 o'clock.
    rendered = render_dial(spec, (scale.min_value + scale.max_value) / 2, size=512)
    pixels = np.asarray(rendered.image.convert("RGB")).astype(int)
    cx, cy = rendered.labels.center
    face_radius = rendered.labels.radius * (1 - spec.bezel_width)

    def color_at(angle: float, radius: float) -> np.ndarray:
        x = cx + radius * math.sin(math.radians(angle))
        y = cy - radius * math.cos(math.radians(angle))
        return pixels[round(y), round(x)]

    red, white = np.array([255, 0, 0]), np.array([255, 255, 255])
    assert np.abs(color_at(0.0, 0.45 * face_radius) - red).max() < 40
    assert np.abs(color_at(90.0, 0.45 * face_radius) - white).max() < 40
    tip_x, tip_y = rendered.labels.needle_tip
    assert tip_x == pytest.approx(cx)
    assert tip_y < cy
    tip_radius = cy - tip_y
    assert np.abs(color_at(0.0, tip_radius - 2) - red).max() < 60


def test_outside_the_bezel_is_transparent() -> None:
    alpha = np.asarray(render_random(3).image)[..., 3]
    assert alpha[0, 0] == 0
    assert alpha[256, 256] == 255


@pytest.mark.parametrize("seed", SEEDS)
def test_sampled_specs_have_round_major_ticks(seed) -> None:
    spec = sample_spec(np.random.default_rng(seed))
    intervals = spec.scale.span / spec.major_step
    assert 4 <= round(intervals) <= 10
    assert math.isclose(intervals, round(intervals), abs_tol=1e-9)
    assert spec.scale.sweep in (180.0, 240.0, 270.0, 300.0)
    assert spec.needle_length <= spec.tick_outer


def test_sample_value_without_out_of_range_stays_on_the_scale() -> None:
    rng = np.random.default_rng(0)
    scale = GaugeScale(min_value=0.0, max_value=10.0, min_angle=225.0, max_angle=135.0)
    values = [sample_value(scale, rng, out_of_range=0.0) for _ in range(1000)]
    assert min(values) >= 0.0
    assert max(values) <= 10.0


@pytest.mark.parametrize("sweep", [180.0, 240.0, 270.0, 300.0])
def test_out_of_range_values_land_in_the_matching_dead_zone_half(sweep) -> None:
    rng = np.random.default_rng(0)
    gap = 360.0 - sweep
    scale = GaugeScale(
        min_value=0.0,
        max_value=10.0,
        min_angle=180.0 + gap / 2,
        max_angle=180.0 - gap / 2,
    )
    for _ in range(500):
        value = sample_value(scale, rng, out_of_range=1.0)
        reading = scale.read_angle(scale.value_to_angle(value), edge_margin=0.0)
        assert reading.position == ("below_range" if value < 0.0 else "above_range")


def test_core_import_does_not_load_pillow() -> None:
    code = "import sys, gaugelens, gaugelens.geometry; assert 'PIL' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)

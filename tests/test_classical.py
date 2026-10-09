import numpy as np
import pytest

from gaugelens.classical import read_classical
from gaugelens.synth import capture, render_dial, sample_spec, sample_value
from tests.test_capture import plain

GRAY = np.full((480, 640, 3), 0.5, np.float32)


def frontal_scene(seed: int):
    """A clean, frontal dial on a gray background with its needle on the scale."""
    rng = np.random.default_rng(seed)
    spec = sample_spec(rng)
    dial = render_dial(spec, sample_value(spec.scale, rng, out_of_range=0.0), size=420)
    return capture(dial, plain(), GRAY, rng)


@pytest.mark.parametrize("seed", range(12))
def test_reads_clean_frontal_dials_within_half_a_percent(seed) -> None:
    scene = frontal_scene(seed)
    labels = scene.labels
    reading = read_classical(
        np.asarray(scene.image), labels.min_value, labels.max_value
    )
    assert reading.status == "ok", reading.notes
    span = labels.max_value - labels.min_value
    truth = min(max(labels.value, labels.min_value), labels.max_value)
    assert abs(reading.value - truth) / span < 0.005


def test_given_angles_skip_the_tick_search() -> None:
    scene = frontal_scene(1)
    labels = scene.labels
    reading = read_classical(
        np.asarray(scene.image),
        labels.min_value,
        labels.max_value,
        # Deliberately wrong marks: the reader must use them as given.
        min_angle=270.0,
        max_angle=90.0,
    )
    assert (reading.min_angle, reading.max_angle) == (270.0, 90.0)


def test_blank_image_has_no_gauge() -> None:
    blank = np.full((480, 640, 3), 128, np.uint8)
    reading = read_classical(blank, 0.0, 10.0)
    assert reading.status == "no_gauge"
    assert reading.value is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_value": 10.0, "max_value": 0.0},
        {"min_angle": 225.0},
    ],
)
def test_invalid_input_raises(kwargs) -> None:
    args = {"min_value": 0.0, "max_value": 10.0} | kwargs
    with pytest.raises(ValueError):
        read_classical(np.zeros((64, 64, 3), np.uint8), **args)

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


@pytest.mark.parametrize("angle", [0.0, 35.0, 120.0])
def test_rectify_turns_an_ellipse_into_a_circle(angle) -> None:
    import cv2

    from gaugelens.classical import _rectify

    image = np.zeros((400, 400, 3), np.uint8)
    cv2.ellipse(image, ((200, 200), (300, 170), angle), (255, 255, 255), -1)
    contours, _ = cv2.findContours(
        image[..., 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    (cx, cy), (w, h), a = cv2.fitEllipse(contours[0])
    out, _, radius = _rectify(image, ((cx, cy), (w / 2, h / 2), a))
    contours, _ = cv2.findContours(
        out[..., 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    _, (w2, h2), _ = cv2.fitEllipse(max(contours, key=cv2.contourArea))
    assert min(w2, h2) / max(w2, h2) > 0.99
    assert radius == pytest.approx(150, abs=1)


@pytest.mark.parametrize("seed", range(6))
def test_tilted_dials_stay_within_five_percent_under_weak_perspective(seed) -> None:
    # The tilt fix is affine: with the camera far away (focal 3) the error
    # stays a few percent; closer cameras shift the true center further.
    rng = np.random.default_rng(seed)
    spec = sample_spec(rng)
    dial = render_dial(spec, sample_value(spec.scale, rng, out_of_range=0.0), size=420)
    scene = capture(dial, plain(tilt_y=20.0, focal=3.0), GRAY, rng)
    labels = scene.labels
    reading = read_classical(
        np.asarray(scene.image), labels.min_value, labels.max_value
    )
    assert reading.status == "ok", reading.notes
    truth = min(max(labels.value, labels.min_value), labels.max_value)
    assert abs(reading.value - truth) / (labels.max_value - labels.min_value) < 0.05

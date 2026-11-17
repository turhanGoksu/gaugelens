"""Design A training pieces that need no PyTorch (D17)."""

import math
import random
from pathlib import Path

import numpy as np
import pytest

from training import common, preprocess, readers
from training.decode import detector_dials, heatmap_peaks


def gauge(x0=100.0, y0=100.0, side=200.0) -> dict:
    return {
        "box": [x0, y0, x0 + side, y0 + side],
        "center": [x0 + 100, y0 + 100],
        "tip": [x0 + 150, y0 + 60],
        "min": [x0 + 40, y0 + 170],
        "max": [x0 + 160, y0 + 170],
    }


def test_fit_stops_when_validation_stops_improving() -> None:
    losses = iter([5.0, 4.0, 4.5, 4.6, 4.7, 4.8])
    saved = []
    result = common.fit(
        lambda: 1.0,
        lambda: next(losses),
        lambda: saved.append(1),
        max_seconds=60,
        eval_every=2,
        patience=3,
        log=lambda _: None,
    )
    assert result["stopped"] == "patience"
    assert result["best_val_loss"] == 4.0
    assert len(saved) == 2  # 5.0, then the new best 4.0
    assert [h["step"] for h in result["history"]] == [2, 4, 6, 8, 10]


def test_fit_stops_at_the_step_limit() -> None:
    result = common.fit(
        lambda: 1.0,
        lambda: 1.0,
        lambda: None,
        max_seconds=60,
        eval_every=100,
        patience=5,
        max_steps=3,
        log=lambda _: None,
    )
    assert (result["steps"], result["stopped"]) == (3, "steps")


def test_real_images_get_their_share() -> None:
    samples = [{"source": "synthetic"}] * 300 + [{"source": "roboflow:x"}] * 10
    order = common.training_order(samples, random.Random(0))
    real = sum(samples[i]["source"] != "synthetic" for i in order)
    assert real / len(order) == pytest.approx(common.REAL_SHARE, abs=0.02)


def test_random_crop_keeps_every_dial_whole() -> None:
    image = np.zeros((600, 800, 3), np.uint8)
    rng = random.Random(1)
    for _ in range(50):
        out, [g] = common.random_crop(image, [gauge()], rng)
        h, w = out.shape[:2]
        assert g["box"][0] >= 0 and g["box"][1] >= 0
        assert g["box"][2] <= w and g["box"][3] <= h
        assert g["center"][0] - g["box"][0] == 100


def test_point_errors_are_shares_of_the_dial_radius() -> None:
    g = gauge()
    found = {"center": [210.0, 200.0], "tip": None, "min": g["min"], "max": g["max"]}
    errors = common.point_errors(found, g)
    assert errors["center"] == pytest.approx(0.1)  # 10 px of a 100 px radius
    assert errors["tip"] is None and errors["min"] == 0.0
    summary = common.summarize_errors([errors, common.point_errors(None, g)])
    assert summary["tip"]["missing"] == 2 and summary["center"]["missing"] == 1


def test_detector_points_join_the_dial_that_holds_them() -> None:
    # Queries: two dials (one a duplicate), a centre for each place, a tip.
    boxes = np.array(
        [
            [0.25, 0.5, 0.4, 0.8],
            [0.26, 0.5, 0.4, 0.8],
            [0.75, 0.5, 0.4, 0.8],
            [0.25, 0.5, 0.02, 0.02],
            [0.75, 0.5, 0.02, 0.02],
            [0.3, 0.3, 0.02, 0.02],
        ]
    )
    scores = np.zeros((6, 5))
    scores[0, 0], scores[1, 0], scores[2, 0] = 0.9, 0.8, 0.6
    scores[3, 1], scores[4, 1], scores[5, 2] = 0.7, 0.5, 0.4
    dials = detector_dials(scores, boxes, 200, 100)
    assert [round(d["score"], 1) for d in dials] == [0.9, 0.6]  # duplicate dropped
    assert dials[0]["points"]["center"] == pytest.approx([50, 50])
    assert dials[1]["points"]["center"] == pytest.approx([150, 50])
    assert dials[0]["points"]["tip"] == pytest.approx([60, 30])
    assert dials[1]["points"]["tip"] is None


def test_heatmap_peak_is_refined_below_one_pixel() -> None:
    ys, xs = np.mgrid[0:64, 0:64]
    maps = np.exp(-((xs - 20.3) ** 2 + (ys - 41.6) ** 2) / 8.0)[None]
    ((x, y, peak),) = heatmap_peaks(maps)
    assert x == pytest.approx(20.3, abs=1e-6) and y == pytest.approx(41.6, abs=1e-6)
    assert peak > 0.9


def test_a_point_survives_crop_and_heatmap_round_trip() -> None:
    g = gauge()
    m = preprocess.crop_matrix(g["box"], random.Random(3))
    u, v = preprocess.apply(m, g["tip"])
    ys, xs = np.mgrid[0 : preprocess.HEATMAP, 0 : preprocess.HEATMAP]
    scale = preprocess.HEATMAP / preprocess.CROP
    peak = np.exp(-((xs - u * scale) ** 2 + (ys - v * scale) ** 2) / 8.0)
    found = preprocess.points_from_heatmaps(np.stack([peak] * 4), m)
    assert found["tip"] == pytest.approx(g["tip"], abs=1.0)


def test_kprcnn_input_scales_large_images_down() -> None:
    x, scale = preprocess.kprcnn_input(np.zeros((1080, 1920, 3), np.uint8))
    assert x.shape == (3, 600, 1066) and scale == pytest.approx(1920 / 1066)


def test_reader_turns_points_into_statuses(monkeypatch) -> None:
    def at(angle):
        t = math.radians(angle)
        return [200 + 100 * math.sin(t), 200 - 100 * math.cos(t)]

    dial = {
        "box": [100, 100, 300, 300],
        "score": 0.9,
        "points": {"center": [200, 200], "tip": at(0), "min": at(225), "max": at(135)},
    }
    monkeypatch.setattr(readers, "find", lambda way, image, models: dial)
    r = readers.read("way1", np.zeros((1, 1, 3)), 0.0, 10.0, Path("."))
    assert r["status"] == "ok" and r["value"] == pytest.approx(5.0)

    dial["points"]["tip"] = None
    r = readers.read("way1", np.zeros((1, 1, 3)), 0.0, 10.0, Path("."))
    assert (r["status"], r["notes"]) == ("unreadable", ["not found: tip"])

    dial["points"]["tip"] = [200, 200]  # on the pivot: no angle
    assert (
        readers.read("way1", np.zeros((1, 1, 3)), 0, 10, Path("."))["status"]
        == "unreadable"
    )

    monkeypatch.setattr(readers, "find", lambda way, image, models: None)
    r = readers.read("way2", np.zeros((1, 1, 3)), 0.0, 10.0, Path("."))
    assert (r["status"], r["notes"]) == ("unreadable", ["no dial found"])

"""Design A trainers; skipped where PyTorch is not installed (D17)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from training import detector, heatmap, kprcnn, train  # noqa: E402


def gauge() -> dict:
    return {
        "box": [100.0, 50.0, 300.0, 250.0],
        "center": [200.0, 150.0],
        "tip": [250.0, 100.0],
        "min": [140.0, 220.0],
        "max": [260.0, 220.0],
    }


def test_detector_targets_are_normalized_dial_and_point_boxes() -> None:
    t = detector.targets([gauge()], 400, 300)
    assert t["class_labels"].tolist() == [0, 1, 2, 3, 4]
    dial, center = t["boxes"][0].tolist(), t["boxes"][1].tolist()
    assert dial == pytest.approx([0.5, 0.5, 0.5, 200 / 300])
    side = detector.POINT_BOX * 200  # 6% of the dial size
    assert center == pytest.approx([0.5, 0.5, side / 400, side / 300])


def test_heatmap_targets_peak_at_the_points() -> None:
    maps, weights = heatmap.target_maps([[128.0, 64.0], [999.0, 5.0]])
    y, x = np.unravel_index(int(np.argmax(maps[0])), maps[0].shape)
    assert (x, y) == (32, 16)
    assert weights.tolist() == [1.0, 0.0]  # the second point lies outside the crop


def test_heatmap_net_draws_four_64x64_maps() -> None:
    net = heatmap.HeatmapNet(pretrained=False).eval()
    with torch.no_grad():
        assert net(torch.zeros(1, 3, 256, 256)).shape == (1, 4, 64, 64)


def test_kprcnn_targets_widen_the_dial_box() -> None:
    t = kprcnn.targets([gauge()])
    assert t["boxes"][0].tolist() == pytest.approx([80.0, 30.0, 320.0, 270.0])
    assert t["keypoints"].shape == (1, 4, 3)
    assert t["keypoints"][0, 0].tolist() == [200, 150, 2]


def test_onnx_check_flags_a_different_number_of_detections() -> None:
    assert train.max_diff(np.zeros((2, 4)), torch.zeros(3, 4)) == float("inf")
    assert train.max_diff(np.zeros((0, 4)), torch.zeros(0, 4)) == 0.0
    assert train.max_diff(np.ones(3), torch.zeros(3)) == 1.0

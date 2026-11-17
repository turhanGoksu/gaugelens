import math

import pytest

from gaugelens.points import GaugePoints, read_points


def at(angle: float, radius: float = 100.0, center=(200.0, 200.0)) -> tuple:
    """Pixel position at a clock angle (0 = 12 o'clock, clockwise)."""
    t = math.radians(angle)
    return (center[0] + radius * math.sin(t), center[1] - radius * math.cos(t))


def gauge(tip_angle: float) -> GaugePoints:
    # A usual scale: min mark at 225 (7:30), max mark at 135 (4:30), 270 sweep.
    return GaugePoints(center=(200.0, 200.0), tip=at(tip_angle), min_mark=at(225),
                       max_mark=at(135))  # fmt: skip


def test_the_middle_of_the_sweep_reads_half_scale() -> None:
    reading = read_points(gauge(0.0), 0.0, 10.0)
    assert reading.position == "in_range"
    assert reading.value == pytest.approx(5.0)


def test_the_marks_and_tip_need_not_be_at_the_same_radius() -> None:
    points = GaugePoints(center=(200.0, 200.0), tip=at(90.0, 60.0),
                         min_mark=at(225, 120.0), max_mark=at(135, 90.0))  # fmt: skip
    # 90 degrees is 225 degrees past the min mark: 225/270 of 0-6 bar.
    assert read_points(points, 0.0, 6.0).value == pytest.approx(5.0)


def test_a_needle_in_the_dead_zone_is_off_the_scale() -> None:
    assert read_points(gauge(200.0), 0.0, 10.0).position == "below_range"
    assert read_points(gauge(160.0), 0.0, 10.0).position == "above_range"


def test_degenerate_points_raise() -> None:
    with pytest.raises(ValueError):
        read_points(GaugePoints((1, 1), (1, 1), (0, 5), (5, 5)), 0.0, 1.0)
    with pytest.raises(ValueError):
        read_points(GaugePoints((0, 0), (0, -5), (3, 3), (6, 6)), 0.0, 1.0)

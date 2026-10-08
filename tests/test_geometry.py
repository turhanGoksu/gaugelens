import math

import pytest

from gaugelens.geometry import GaugeScale, point_to_angle

# A typical 0-10 bar dial: min mark at 7:30, max mark at 4:30, 270 degree
# sweep that crosses 12 o'clock.
BAR = GaugeScale(min_value=0.0, max_value=10.0, min_angle=225.0, max_angle=135.0)


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((0, -1), 0.0),  # 12 o'clock (y points down)
        ((1, 0), 90.0),  # 3 o'clock
        ((0, 1), 180.0),  # 6 o'clock
        ((-1, 0), 270.0),  # 9 o'clock
        ((1, -1), 45.0),
    ],
)
def test_point_to_angle_uses_clock_convention(point, expected) -> None:
    assert point_to_angle((0, 0), point) == pytest.approx(expected)


def test_point_to_angle_never_returns_360() -> None:
    # Just left of 12 o'clock: the raw angle is a tiny negative number.
    assert point_to_angle((0, 0), (-1e-300, -1)) == 0.0


def test_point_to_angle_rejects_point_at_center() -> None:
    with pytest.raises(ValueError):
        point_to_angle((5, 5), (5, 5))


def test_sweep_crosses_twelve_o_clock() -> None:
    assert BAR.sweep == 270.0


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (225.0, 0.0),  # min mark
        (270.0, 10 / 6),  # 9 o'clock, 45 of 270 degrees
        (0.0, 5.0),  # 12 o'clock; a naive (a - a_min) / (a_max - a_min) gives 25
        (135.0, 10.0),  # max mark
    ],
)
def test_read_angle_in_range(angle, expected) -> None:
    reading = BAR.read_angle(angle)
    assert reading.position == "in_range"
    assert reading.value == pytest.approx(expected)


def test_needle_resting_below_min_is_below_range() -> None:
    reading = BAR.read_angle(210.0)  # 7 o'clock, 15 degrees before the min mark
    assert reading.position == "below_range"
    assert reading.value is None
    assert reading.fraction == pytest.approx(-15 / 270)


def test_needle_past_max_is_above_range() -> None:
    reading = BAR.read_angle(150.0)  # 5 o'clock, 15 degrees past the max mark
    assert reading.position == "above_range"
    assert reading.value is None
    assert reading.fraction == pytest.approx(1 + 15 / 270)


def test_dead_zone_midpoint_counts_as_above_range() -> None:
    # 6 o'clock is exactly halfway between the max and the min mark.
    assert BAR.read_angle(180.0).position == "above_range"


@pytest.mark.parametrize(
    ("angle", "position", "value"),
    [
        (224.0, "in_range", 0.0),  # 1 degree = 0.37% FS before min: clamped
        (220.0, "below_range", None),  # 5 degrees = 1.85% FS before min
        (136.0, "in_range", 10.0),  # 1 degree past max: clamped
        (140.0, "above_range", None),  # 5 degrees past max
    ],
)
def test_edge_margin_clamps_only_small_overshoots(angle, position, value) -> None:
    reading = BAR.read_angle(angle)
    assert reading.position == position
    assert reading.value == value


def test_zero_edge_margin_disables_clamping() -> None:
    assert BAR.read_angle(224.0, edge_margin=0.0).position == "below_range"


@pytest.mark.parametrize(
    "scale",
    [
        BAR,
        GaugeScale(min_value=-1.0, max_value=3.0, min_angle=225.0, max_angle=135.0),
        GaugeScale(min_value=0.0, max_value=160.0, min_angle=0.0, max_angle=300.0),
        GaugeScale(min_value=20.0, max_value=120.0, min_angle=315.0, max_angle=45.0),
        GaugeScale(min_value=0.0, max_value=1.0, min_angle=90.0, max_angle=270.0),
    ],
)
def test_value_to_angle_round_trips(scale) -> None:
    for i in range(101):
        value = scale.min_value + scale.span * i / 100
        reading = scale.read_angle(scale.value_to_angle(value), edge_margin=0.0)
        assert reading.position == "in_range"
        assert reading.value == pytest.approx(value, abs=1e-9 * scale.span)


def test_value_outside_range_maps_into_dead_zone() -> None:
    angle = BAR.value_to_angle(-0.5)
    assert angle == pytest.approx(211.5)
    assert BAR.read_angle(angle).fraction == pytest.approx(-0.05)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_value": 10.0, "max_value": 10.0},
        {"min_value": 10.0, "max_value": 0.0},
        {"min_angle": 225.0, "max_angle": 585.0},  # same mark
        {"min_value": math.nan},
        {"max_angle": math.inf},
    ],
)
def test_invalid_scale_is_rejected(kwargs) -> None:
    args = {"min_value": 0.0, "max_value": 10.0, "min_angle": 225.0, "max_angle": 135.0}
    with pytest.raises(ValueError):
        GaugeScale(**(args | kwargs))


@pytest.mark.parametrize("margin", [-0.01, 0.5, math.nan])
def test_invalid_edge_margin_is_rejected(margin) -> None:
    with pytest.raises(ValueError):
        BAR.read_angle(0.0, edge_margin=margin)

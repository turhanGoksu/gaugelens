"""Gauge geometry: needle angle <-> value on a linear circular scale.

Angles use the clock convention: 0 degrees points to 12 o'clock and angles
grow clockwise, so 3 o'clock is 90 and 6 o'clock is 180. Pixel coordinates
have y pointing down. Values increase clockwise from the min mark to the max
mark, and the scale may cross 12 o'clock, which is the usual case (min mark
at 225, max mark at 135).

The gap between the max mark and the min mark is the dead zone. A needle
there is outside the scale: the half next to the max mark counts as above the
range, the half next to the min mark as below it.

This module assumes a frontal view; perspective is removed before angles are
measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

ScalePosition = Literal["in_range", "below_range", "above_range"]

DEFAULT_EDGE_MARGIN = 0.01
"""Provisional: a needle up to 1% of full scale beyond an end mark counts as
measurement noise and is clamped to that mark. The final value is chosen on
the dev set together with the tolerances (docs/decisions.md, D4)."""


def _normalize(angle: float) -> float:
    """Map any angle to [0, 360)."""
    angle %= 360.0
    # A tiny negative angle wraps to 360.0 after floating-point rounding.
    return 0.0 if angle == 360.0 else angle


def point_to_angle(center: tuple[float, float], point: tuple[float, float]) -> float:
    """Clock angle in [0, 360) of ``point`` as seen from ``center``.

    Both are (x, y) pixel coordinates with y pointing down.
    """
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    if dx == 0 and dy == 0:
        raise ValueError("point coincides with center; the angle is undefined")
    return _normalize(math.degrees(math.atan2(dx, -dy)))


@dataclass(frozen=True)
class ScaleReading:
    """Where a needle angle falls on a scale."""

    position: ScalePosition
    fraction: float
    """Position along the scale, unclamped: 0 at the min mark, 1 at the max
    mark, negative or above 1 outside the scale."""
    value: float | None
    """Reading in scale units, or None when the needle is outside the scale."""


@dataclass(frozen=True)
class GaugeScale:
    """A linear scale from ``min_value`` at ``min_angle`` clockwise to
    ``max_value`` at ``max_angle`` (clock-convention degrees)."""

    min_value: float
    max_value: float
    min_angle: float
    max_angle: float

    def __post_init__(self) -> None:
        for name in ("min_value", "max_value", "min_angle", "max_angle"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite, got {getattr(self, name)!r}")
        if self.min_value >= self.max_value:
            raise ValueError(
                "min_value must be less than max_value, "
                f"got {self.min_value} >= {self.max_value}"
            )
        if self.sweep == 0.0:
            raise ValueError("min_angle and max_angle point to the same mark")

    @property
    def span(self) -> float:
        """Full scale in value units."""
        return self.max_value - self.min_value

    @property
    def sweep(self) -> float:
        """Clockwise angle from the min mark to the max mark, in [0, 360)."""
        return _normalize(self.max_angle - self.min_angle)

    def value_to_angle(self, value: float) -> float:
        """Needle angle for ``value``; the inverse of :meth:`read_angle`.

        Values outside the range map into the dead zone, which lets the
        synthetic generator draw needles below the min or above the max mark.
        """
        fraction = (value - self.min_value) / self.span
        return _normalize(self.min_angle + fraction * self.sweep)

    def read_angle(
        self, angle: float, edge_margin: float = DEFAULT_EDGE_MARGIN
    ) -> ScaleReading:
        """Locate a needle angle on the scale.

        Within ``edge_margin`` (a fraction of full scale) beyond an end mark,
        the value is clamped to that mark. Further out, the reading has no
        value and its position says on which side the needle is.
        """
        if not 0.0 <= edge_margin < 0.5:
            raise ValueError(f"edge_margin must be in [0, 0.5), got {edge_margin!r}")

        offset = _normalize(angle - self.min_angle)
        dead_zone = 360.0 - self.sweep
        # The dead-zone half next to the min mark lies before the scale. An
        # exact tie goes to above_range: missing an overpressure is the worse
        # error.
        if offset > self.sweep + dead_zone / 2:
            offset -= 360.0
        fraction = offset / self.sweep

        if fraction < -edge_margin:
            return ScaleReading("below_range", fraction, None)
        if fraction > 1.0 + edge_margin:
            return ScaleReading("above_range", fraction, None)
        clamped = min(max(fraction, 0.0), 1.0)
        return ScaleReading("in_range", fraction, self.min_value + clamped * self.span)

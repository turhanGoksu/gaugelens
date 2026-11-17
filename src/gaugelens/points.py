"""From four points on a dial to a reading: Design A's shared geometry.

Design A finds the needle pivot, the needle tip and the min and max marks.
Seen from the pivot, the marks give the scale and the tip gives the reading.
Every way of finding the points ends here (docs/decisions.md, D17), so a
comparison between them measures only how well the points are found.

Like ``gaugelens.geometry``, this assumes a frontal view.
"""

from __future__ import annotations

from dataclasses import dataclass

from gaugelens.geometry import (
    DEFAULT_EDGE_MARGIN,
    GaugeScale,
    ScaleReading,
    point_to_angle,
)

Point = tuple[float, float]


@dataclass(frozen=True)
class GaugePoints:
    """Pixel coordinates (x, y), y pointing down."""

    center: Point
    tip: Point
    min_mark: Point
    max_mark: Point


def read_points(
    points: GaugePoints,
    min_value: float,
    max_value: float,
    edge_margin: float = DEFAULT_EDGE_MARGIN,
) -> ScaleReading:
    """Locate the needle on the scale the two marks define.

    Raises ``ValueError`` when the points cannot define a reading: a point on
    the pivot, or both marks at the same angle. A reader turns that into the
    status ``unreadable``.
    """
    scale = GaugeScale(
        min_value=min_value,
        max_value=max_value,
        min_angle=point_to_angle(points.center, points.min_mark),
        max_angle=point_to_angle(points.center, points.max_mark),
    )
    return scale.read_angle(point_to_angle(points.center, points.tip), edge_margin)

"""Design A0: a classical computer-vision baseline with no learning.

1. Find the dial with the Hough circle transform, then undo camera tilt: a
   tilted dial is an ellipse, so fill the face, fit an ellipse to its outline
   and stretch the ellipse back into a circle.
2. Unwrap the dial into a polar strip: rows are angles, columns are radii.
   The needle becomes the angle whose middle band is ink along its whole
   length; printed text covers that band only in part.
3. Find the scale ends. Major ticks are ink runs that start at the tick ring
   and reach inward; the dead zone is the widest empty gap between them, and
   the ticks on its two sides are the max and min marks.
4. Turn the needle angle into a value with :mod:`gaugelens.geometry`.

Stretching an ellipse into a circle is an affine correction. It removes most
of the tilt but not the perspective: under a strong perspective the dial's
center is not the ellipse's center, and a small error remains.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from gaugelens.geometry import DEFAULT_EDGE_MARGIN, GaugeScale
from gaugelens.status import Status

Point = tuple[float, float]

ANGLE_BINS = 1440  # 0.25 degree per polar row
RADIUS_BINS = 240
_POLAR_REACH = 1.05  # unwrap slightly past the detected circle
_CANNY_THRESHOLDS = (15, 45)  # low, so a blurred rim still forms a closed wall


@dataclass(frozen=True)
class ClassicalReading:
    """What A0 read, and the intermediate geometry for debugging."""

    status: Status
    value: float | None
    center: Point | None = None
    radius: float | None = None
    """Radius of the dial face in pixels, after the tilt is undone."""
    needle_angle: float | None = None
    min_angle: float | None = None
    max_angle: float | None = None
    notes: tuple[str, ...] = ()
    """Why the status is not ``ok``, in plain words."""


def _row_to_clock(row: float) -> float:
    # warpPolar rows start at 3 o'clock and run clockwise on screen.
    return (row * 360.0 / ANGLE_BINS + 90.0) % 360.0


def _find_dial(gray: np.ndarray) -> tuple[float, float, float] | None:
    """(x, y, radius) of the strongest circle. The ALT variant locates the
    center about twice as precisely; the classic one is the fallback for
    shapes that are not round enough for it (tilted dials)."""
    short = min(gray.shape)
    sizes = {"minRadius": int(0.12 * short), "maxRadius": int(0.5 * short)}
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (0, 0), 1.5),
        cv2.HOUGH_GRADIENT_ALT,
        dp=1.5,
        minDist=short // 4,
        param1=150,
        param2=0.8,
        **sizes,
    )
    if circles is None:
        circles = cv2.HoughCircles(
            cv2.medianBlur(gray, 5),
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=short,
            param1=120,
            param2=40,
            **sizes,
        )
    if circles is None:
        return None
    x, y, r = circles[0][0]
    return float(x), float(y), float(r)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of True runs in a circular boolean array."""
    if mask.all():
        return [(0, len(mask))]
    if not mask.any():
        return []
    shift = int(np.argmin(mask))  # start scanning at a False element
    rolled = np.roll(mask, -shift)
    edges = np.diff(np.concatenate([[0], rolled.astype(np.int8), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [((s + shift) % len(mask), e - s) for s, e in zip(starts, ends, strict=True)]


def _circular_mean_row(start: int, length: int, weights: np.ndarray) -> float:
    rows = np.arange(start, start + length)
    w = weights[rows % len(weights)]
    return float((rows * w).sum() / max(w.sum(), 1e-9))


def _find_needle(
    ink: np.ndarray, face_cols: int
) -> tuple[float, int, list[str]] | None:
    """Needle angle (polar row, fractional), its half-width in rows, notes."""
    band = ink[:, int(0.3 * face_cols) : int(0.5 * face_cols)]
    coverage = band.mean(axis=1)
    kernel = np.ones(5) / 5
    coverage = np.convolve(
        np.concatenate([coverage[-2:], coverage, coverage[:2]]), kernel, "valid"
    )
    peak = int(np.argmax(coverage))
    if coverage[peak] < 0.6:
        return None
    notes = []
    away = np.ones(ANGLE_BINS, bool)
    away[np.arange(peak - 40, peak + 41) % ANGLE_BINS] = False  # 10 degrees
    if coverage[away].max() > 0.9 * coverage[peak]:
        notes.append("two needle-like lines")
    above = coverage >= 0.5 * coverage[peak]
    start, length = next((s, n) for s, n in _runs(above) if (peak - s) % ANGLE_BINS < n)
    return _circular_mean_row(start, length, coverage), length // 2 + 1, notes


def _find_scale_ends(
    ink: np.ndarray, face_cols: int, needle_rows: np.ndarray
) -> tuple[float, float, list[str]] | None:
    """Clock angles of the min and max marks, and notes."""
    usable = np.ones(ANGLE_BINS, bool)
    usable[needle_rows] = False

    # The tick ring: the outermost radius where ink still appears at a few
    # percent of the angles (ticks are sparse when there are no minor ticks).
    # It stops short of the face edge, whose antialiased rim looks like ink.
    outer = np.arange(int(0.75 * face_cols), int(0.97 * face_cols))
    density = ink[usable][:, outer].mean(axis=0)
    if len(outer) == 0 or density.max() < 0.01:
        return None
    ring = int(outer[np.where(density >= max(0.01, 0.2 * density.max()))[0].max()])

    # A major tick is an ink run that starts at the ring and reaches inward
    # 6-25% of the face radius; the needle reaches further.
    run = np.zeros(ANGLE_BINS)
    for row in np.where(usable)[0]:
        # A slightly off center makes the ring wavy, so search a few columns.
        column = min(ring + 3, ink.shape[1] - 1)
        while column > ring - 6 and not ink[row, column]:
            column -= 1
        length = 0
        while column - length >= 0 and ink[row, column - length]:
            length += 1
        run[row] = length / face_cols
    major = (run >= 0.06) & (run <= 0.25)
    ticks = [
        _circular_mean_row(s, n, np.ones(ANGLE_BINS)) % ANGLE_BINS
        for s, n in _runs(major)
        if n >= 2
    ]
    if len(ticks) < 3:
        return None
    ticks.sort()

    # Score each gap between neighboring ticks by width x emptiness, where
    # emptiness looks between the numbers and the tick ring (not at the rim).
    annulus = ink[:, int(0.55 * face_cols) : ring + 1].any(axis=1) & usable
    scores = []
    for i, start in enumerate(ticks):
        end = ticks[(i + 1) % len(ticks)]
        width = (end - start) % ANGLE_BINS
        rows = np.arange(int(start) + 3, int(start + width) - 2) % ANGLE_BINS
        emptiness = 1.0 - (annulus[rows].mean() if len(rows) else 1.0)
        scores.append(width * emptiness)
    order = np.argsort(scores)[::-1]
    notes = []
    if scores[order[1]] > 0.8 * scores[order[0]]:
        notes.append("scale ends ambiguous")
    best = int(order[0])
    max_row, min_row = ticks[best], ticks[(best + 1) % len(ticks)]
    return _row_to_clock(min_row), _row_to_clock(max_row), notes


def _face_ellipse(
    image: np.ndarray, center: Point, radius: float
) -> tuple[Point, tuple[float, float], float] | None:
    """Fit an ellipse to the dial face, or None if the face cannot be isolated.

    Edges (Canny) act as walls: the face is filled from seeds around the
    center, the fill cannot cross an edge, and the outline of the filled
    region is the rim. Comparing colors instead lets the fill creep through
    blurred rims step by step; walls do not.
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (0, 0), 1.2)
    walls = cv2.dilate(cv2.Canny(gray, *_CANNY_THRESHOLDS), np.ones((3, 3), np.uint8))
    height, width = gray.shape
    mask = np.zeros((height + 2, width + 2), np.uint8)
    mask[1:-1, 1:-1] = walls > 0  # nonzero mask pixels stop the fill
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (2 << 8)
    for k in range(8):  # seeds between the hub and the numbers
        a = 2 * np.pi * k / 8
        x = int(center[0] + 0.4 * radius * np.cos(a))
        y = int(center[1] + 0.4 * radius * np.sin(a))
        if 0 <= x < width and 0 <= y < height and not mask[y + 1, x + 1]:
            cv2.floodFill(gray, mask, (x, y), 0, (255,), (255,), flags)
    face = (mask[1:-1, 1:-1] == 2).astype(np.uint8)
    area = float(np.count_nonzero(face))
    # Too small: the fill was blocked. Too large: it leaked past the bezel.
    if not 0.3 * np.pi * radius**2 <= area <= 1.6 * np.pi * radius**2:
        return None
    contours, _ = cv2.findContours(face, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    if len(hull) < 5:
        return None
    (ex, ey), (axis_w, axis_h), angle = cv2.fitEllipse(hull)
    return (float(ex), float(ey)), (axis_w / 2, axis_h / 2), float(angle)


def _rectify(
    image: np.ndarray, ellipse: tuple[Point, tuple[float, float], float]
) -> tuple[np.ndarray, np.ndarray, float]:
    """Stretch the ellipse along its minor axis into a circle.

    Returns the new image, the 2x3 affine map from the old image to the new
    one, and the circle's radius.
    """
    (ex, ey), (semi_w, semi_h), angle = ellipse
    t = np.radians(angle)
    along_w = np.array([np.cos(t), np.sin(t)])  # OpenCV's width axis
    along_h = np.array([-np.sin(t), np.cos(t)])
    if semi_w >= semi_h:
        minor, stretch, radius = along_h, semi_w / semi_h, semi_w
    else:
        minor, stretch, radius = along_w, semi_h / semi_w, semi_h
    linear = np.eye(2) + (stretch - 1.0) * np.outer(minor, minor)
    side = int(np.ceil(2.3 * radius))
    target = np.array([side / 2, side / 2])
    affine = np.hstack([linear, (target - linear @ np.array([ex, ey]))[:, None]])
    rectified = cv2.warpAffine(
        image,
        affine,
        (side, side),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rectified, affine, float(radius)


def read_classical(
    image: np.ndarray,
    min_value: float,
    max_value: float,
    *,
    min_angle: float | None = None,
    max_angle: float | None = None,
    edge_margin: float = DEFAULT_EDGE_MARGIN,
) -> ClassicalReading:
    """Read an RGB uint8 image of a single gauge.

    ``min_angle`` and ``max_angle`` (clock degrees) are optional; without
    them the scale ends are found from the ticks.
    """
    if (min_angle is None) != (max_angle is None):
        raise ValueError("give both min_angle and max_angle, or neither")
    if not min_value < max_value:
        raise ValueError(
            f"min_value must be less than max_value, got {min_value}, {max_value}"
        )

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dial = _find_dial(gray)
    if dial is None:
        return ClassicalReading("no_gauge", None, notes=("no circle found",))
    cx, cy, r = dial
    center = (cx, cy)
    ellipse = _face_ellipse(image, center, r)
    if ellipse is not None:
        image, _, r = _rectify(image, ellipse)
        center = (image.shape[1] / 2, image.shape[0] / 2)
        reported_center = ellipse[0]
    else:
        reported_center = center

    polar = cv2.warpPolar(
        image,
        (RADIUS_BINS, ANGLE_BINS),
        center,
        r * _POLAR_REACH,
        cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR,
    ).astype(np.float32)
    cols_per_radius = RADIUS_BINS / _POLAR_REACH
    face_color = np.median(
        polar[:, int(0.25 * cols_per_radius) : int(0.5 * cols_per_radius)].reshape(
            -1, 3
        ),
        axis=0,
    )
    distance = np.linalg.norm(polar - face_color, axis=2)
    scaled = np.clip(distance, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = distance > max(otsu, 30.0)

    # Where the face ends: from here on nearly every angle differs from the
    # face color. A scale arc covers only the sweep (at most ~85%).
    differs = (distance > 40).mean(axis=0)
    edge = np.where(differs[int(0.6 * cols_per_radius) :] > 0.9)[0]
    face_cols = (
        int(0.6 * cols_per_radius) + int(edge[0]) if len(edge) else int(cols_per_radius)
    )

    needle = _find_needle(ink, face_cols)
    if needle is None:
        return ClassicalReading(
            "unreadable", None, reported_center, r, notes=("no needle found",)
        )
    needle_row, half_width, notes = needle
    needle_angle = _row_to_clock(needle_row)
    if "two needle-like lines" in notes:
        return ClassicalReading(
            "unreadable", None, reported_center, r, needle_angle, notes=tuple(notes)
        )

    if min_angle is None or max_angle is None:
        rows = np.arange(
            int(needle_row) - half_width - 4, int(needle_row) + half_width + 5
        )
        ends = _find_scale_ends(ink, face_cols, rows % ANGLE_BINS)
        if ends is None:
            return ClassicalReading(
                "unreadable",
                None,
                reported_center,
                r,
                needle_angle,
                notes=("scale ticks not found",),
            )
        min_angle, max_angle, end_notes = ends
        notes += end_notes

    scale = GaugeScale(min_value, max_value, min_angle, max_angle)
    if not 90.0 <= scale.sweep <= 340.0:
        return ClassicalReading(
            "unreadable",
            None,
            reported_center,
            r,
            needle_angle,
            min_angle,
            max_angle,
            notes=(f"implausible sweep {scale.sweep:.0f} degrees",),
        )
    reading = scale.read_angle(needle_angle, edge_margin)
    if reading.position != "in_range":
        status: Status = reading.position
    else:
        status = "low_confidence" if notes else "ok"
    return ClassicalReading(
        status,
        reading.value,
        reported_center,
        r,
        needle_angle,
        min_angle,
        max_angle,
        tuple(notes),
    )

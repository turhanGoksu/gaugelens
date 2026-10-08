"""Render a frontal, undistorted gauge dial with exact labels.

The dial is drawn supersampled and then downscaled, so edges are
antialiased. The image is RGBA and everything outside the bezel is
transparent, which lets a later capture stage put the dial on any background.

Label coordinates are output pixels: pixel centers at integer coordinates,
y pointing down. The needle angle comes from
:meth:`gaugelens.geometry.GaugeScale.value_to_angle`, so labels and readers
share one angle convention.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gaugelens.geometry import GaugeScale

RGB = tuple[int, int, int]
Point = tuple[float, float]
NeedleStyle = Literal["line", "tapered"]

_SUPERSAMPLE = 3

# (min, max) ranges and the units they appear with. Pressure maxima follow
# the 1 / 1.6 / 2.5 / 4 / 6 series used on industrial gauges (EN 837-1).
_PRESSURE_MAXIMA = [1, 1.6, 2.5, 4, 6, 10, 16, 25, 40, 60, 100, 160, 250, 400, 600]
_RANGE_GROUPS: list[tuple[list[tuple[float, float]], list[str]]] = [
    ([(0.0, float(m)) for m in _PRESSURE_MAXIMA], ["bar", "psi", "kPa", "MPa"]),
    (
        [(-1.0, 0.0), (-1.0, 0.6), (-1.0, 1.5), (-1.0, 3.0), (-1.0, 5.0), (-1.0, 9.0)],
        ["bar"],
    ),
    (
        [(0.0, 120.0), (0.0, 160.0), (0.0, 200.0), (-20.0, 60.0), (0.0, 300.0)],
        ["°C", "°F"],
    ),
]
_RANGE_GROUP_WEIGHTS = [0.6, 0.15, 0.25]

_SWEEPS = [270.0, 240.0, 300.0, 180.0]
_SWEEP_WEIGHTS = [0.6, 0.2, 0.1, 0.1]


@dataclass(frozen=True)
class DialSpec:
    """Everything that defines how a dial looks. Lengths are fractions of the
    face radius unless noted otherwise."""

    scale: GaugeScale
    unit: str
    major_step: float
    minor_divisions: int
    decimals: int
    face_color: RGB
    ink_color: RGB
    needle_color: RGB
    hub_color: RGB
    bezel_color: RGB
    bezel_width: float  # fraction of the outer radius
    tick_outer: float
    major_tick_length: float
    major_tick_width: float
    number_size: float
    number_gap: float  # from the inner end of the major ticks to the numbers
    unit_offset: float  # below the center
    scale_arc: bool
    stop_pin: bool
    needle_style: NeedleStyle
    needle_length: float
    needle_tail: float
    needle_width: float
    hub_radius: float


@dataclass(frozen=True)
class DialLabels:
    """Ground truth for one rendered dial, in output pixel coordinates."""

    value: float
    """The value the needle points at; may lie outside [min_value, max_value]."""
    fraction: float
    """Needle position along the scale: 0 at the min mark, 1 at the max mark."""
    min_value: float
    max_value: float
    unit: str
    center: Point
    needle_tip: Point
    min_mark: Point
    """Midpoint of the min-mark tick."""
    max_mark: Point
    """Midpoint of the max-mark tick."""
    radius: float
    """Outer (bezel) radius in pixels."""
    bbox: tuple[float, float, float, float]
    """(x0, y0, x1, y1) of the bezel circle."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RenderedDial:
    image: Image.Image
    labels: DialLabels


def _jitter(rng: np.random.Generator, color: RGB, amount: int = 12) -> RGB:
    r, g, b = (
        int(np.clip(c + rng.integers(-amount, amount + 1), 0, 255)) for c in color
    )
    return (r, g, b)


def _nice_steps(min_value: float, max_value: float) -> list[float]:
    """Major steps that give 4-10 intervals and land on round numbers."""
    span = max_value - min_value
    steps = []
    for exponent in range(-2, 4):
        for mantissa in (1.0, 2.0, 2.5, 5.0):
            step = mantissa * 10.0**exponent
            intervals = span / step
            starts = min_value / step
            if (
                4 <= round(intervals) <= 10
                and math.isclose(intervals, round(intervals), abs_tol=1e-9)
                and math.isclose(starts, round(starts), abs_tol=1e-9)
            ):
                steps.append(step)
    return steps


def _decimals(*numbers: float) -> int:
    """Fewest decimals that print every number exactly."""
    for decimals in range(4):
        if all(math.isclose(n, round(n, decimals), abs_tol=1e-9) for n in numbers):
            return decimals
    return 4


def sample_spec(rng: np.random.Generator) -> DialSpec:
    """Draw a random dial appearance (domain randomization of the content)."""
    group = rng.choice(len(_RANGE_GROUPS), p=_RANGE_GROUP_WEIGHTS)
    ranges, units = _RANGE_GROUPS[group]
    min_value, max_value = ranges[rng.integers(len(ranges))]
    unit = str(rng.choice(units))
    major_step = float(rng.choice(_nice_steps(min_value, max_value)))

    # Symmetric about 6 o'clock: a 270 degree sweep runs from 225 to 135.
    sweep = float(rng.choice(_SWEEPS, p=_SWEEP_WEIGHTS))
    scale = GaugeScale(
        min_value=min_value,
        max_value=max_value,
        min_angle=180.0 + (360.0 - sweep) / 2,
        max_angle=180.0 - (360.0 - sweep) / 2,
    )

    if rng.random() < 0.8:  # light face, dark print
        face = _jitter(
            rng, [(250, 250, 250), (242, 238, 225), (225, 228, 230)][rng.integers(3)]
        )
        ink = _jitter(rng, (25, 25, 25), 20)
        needles = [(200, 30, 30), (20, 20, 20), (25, 45, 120), (230, 110, 20)]
    else:  # dark face, light print
        face = _jitter(rng, [(20, 20, 22), (35, 35, 40), (20, 30, 60)][rng.integers(3)])
        ink = _jitter(rng, (240, 240, 240), 15)
        needles = [(240, 240, 240), (220, 40, 40), (240, 140, 30), (240, 210, 40)]
    needle = _jitter(rng, needles[rng.integers(len(needles))])
    hub = needle if rng.random() < 0.6 else _jitter(rng, (30, 30, 30))
    bezel = _jitter(
        rng, [(190, 192, 195), (40, 40, 42), (175, 145, 80)][rng.integers(3)]
    )

    minor_divisions = int(rng.choice([1, 2, 4, 5]))
    tick_outer = float(rng.uniform(0.86, 0.95))
    major_tick_length = float(rng.uniform(0.08, 0.14))
    return DialSpec(
        scale=scale,
        unit=unit,
        major_step=major_step,
        minor_divisions=minor_divisions,
        decimals=_decimals(min_value, major_step),
        face_color=face,
        ink_color=ink,
        needle_color=needle,
        hub_color=hub,
        bezel_color=bezel,
        bezel_width=float(rng.uniform(0.04, 0.12)),
        tick_outer=tick_outer,
        major_tick_length=major_tick_length,
        major_tick_width=float(rng.uniform(0.010, 0.022)),
        number_size=float(rng.uniform(0.09, 0.13)),
        number_gap=float(rng.uniform(0.08, 0.12)),
        unit_offset=float(rng.uniform(0.30, 0.42)),
        scale_arc=bool(rng.random() < 0.5),
        stop_pin=bool(rng.random() < 0.5),
        needle_style="line" if rng.random() < 0.4 else "tapered",
        # The tip ends inside or just short of the tick band.
        needle_length=tick_outer - float(rng.uniform(0.0, 1.2)) * major_tick_length,
        needle_tail=float(rng.uniform(0.0, 0.25)),
        needle_width=float(rng.uniform(0.025, 0.06)),
        hub_radius=float(rng.uniform(0.04, 0.09)),
    )


def sample_value(
    scale: GaugeScale, rng: np.random.Generator, out_of_range: float = 0.1
) -> float:
    """Draw a needle value: uniform over the scale, or with probability
    ``out_of_range`` a little below the min or above the max mark."""
    if rng.random() >= out_of_range:
        return float(rng.uniform(scale.min_value, scale.max_value))
    # Stay well inside the correct half of the dead zone.
    limit = 0.8 * (360.0 - scale.sweep) / 2 / scale.sweep
    over = float(rng.uniform(0.02, min(0.15, limit)))
    fraction = -over if rng.random() < 0.5 else 1.0 + over
    return scale.min_value + fraction * scale.span


@lru_cache(maxsize=64)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Pillow's built-in font is Aileron Regular (CC0).
    return ImageFont.load_default(size=size)


def render_dial(spec: DialSpec, value: float, size: int = 512) -> RenderedDial:
    """Draw ``spec`` with the needle at ``value`` on a ``size`` x ``size`` canvas."""
    ss = _SUPERSAMPLE
    scale = spec.scale
    center = ((size - 1) / 2, (size - 1) / 2)
    outer = 0.48 * size
    face = outer * (1.0 - spec.bezel_width)

    def to_canvas(p: Point) -> Point:
        return ((p[0] + 0.5) * ss - 0.5, (p[1] + 0.5) * ss - 0.5)

    def polar(angle: float, radius: float) -> Point:
        # Clock convention: 0 = 12 o'clock, clockwise, y down.
        a = math.radians(angle)
        return (center[0] + radius * math.sin(a), center[1] - radius * math.cos(a))

    def circle(draw: ImageDraw.ImageDraw, p: Point, radius: float, fill: RGB) -> None:
        x, y = to_canvas(p)
        r = radius * ss
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)

    def bar(
        draw: ImageDraw.ImageDraw,
        angle: float,
        r0: float,
        r1: float,
        w0: float,
        w1: float,
        fill: RGB,
    ) -> None:
        """Quad along ``angle`` from radius r0 (width w0) to r1 (width w1)."""
        a = math.radians(angle)
        nx, ny = math.cos(a), math.sin(a)  # perpendicular to the radial direction
        p0, p1 = polar(angle, r0), polar(angle, r1)
        corners = [
            (p0[0] + nx * w0 / 2, p0[1] + ny * w0 / 2),
            (p1[0] + nx * w1 / 2, p1[1] + ny * w1 / 2),
            (p1[0] - nx * w1 / 2, p1[1] - ny * w1 / 2),
            (p0[0] - nx * w0 / 2, p0[1] - ny * w0 / 2),
        ]
        draw.polygon([to_canvas(c) for c in corners], fill=fill)

    canvas = Image.new("RGBA", (size * ss, size * ss), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    circle(draw, center, outer, spec.bezel_color)
    circle(draw, center, face, spec.face_color)

    tick_outer = spec.tick_outer * face
    major_len = spec.major_tick_length * face
    major_w = spec.major_tick_width * face
    tick_mid = tick_outer - major_len / 2

    if spec.scale_arc:
        x, y = to_canvas(center)
        r = tick_outer * ss
        draw.arc(
            (x - r, y - r, x + r, y + r),
            start=scale.min_angle - 90.0,  # Pillow measures from 3 o'clock
            end=scale.min_angle - 90.0 + scale.sweep,
            fill=spec.ink_color,
            width=max(1, round(major_w * ss / 2)),
        )

    intervals = round(scale.span / spec.major_step)
    for i in range(intervals * spec.minor_divisions + 1):
        tick_value = scale.min_value + i * spec.major_step / spec.minor_divisions
        angle = scale.value_to_angle(tick_value)
        if i % spec.minor_divisions == 0:
            bar(
                draw,
                angle,
                tick_outer - major_len,
                tick_outer,
                major_w,
                major_w,
                spec.ink_color,
            )
        else:
            length = major_len * 0.5
            bar(
                draw,
                angle,
                tick_outer - length,
                tick_outer,
                major_w / 2,
                major_w / 2,
                spec.ink_color,
            )

    number_radius = tick_outer - major_len - spec.number_gap * face
    numbers = []
    for i in range(intervals + 1):
        tick_value = scale.min_value + i * spec.major_step
        text = f"{tick_value:.{spec.decimals}f}"
        numbers.append((tick_value, text.lstrip("-") if float(text) == 0.0 else text))
    # Shrink the numbers when the widest one would not fit between neighbors.
    font_size = max(1, round(spec.number_size * face * ss))
    spacing = number_radius * ss * math.radians(scale.sweep) / intervals
    widest = max(_font(font_size).getlength(text) for _, text in numbers)
    if widest > 0.8 * spacing:
        font_size = max(1, int(font_size * 0.8 * spacing / widest))
    number_font = _font(font_size)
    for tick_value, text in numbers:
        anchor = to_canvas(polar(scale.value_to_angle(tick_value), number_radius))
        draw.text(anchor, text, font=number_font, fill=spec.ink_color, anchor="mm")

    unit_font = _font(max(1, round(spec.number_size * 1.1 * face * ss)))
    unit_anchor = to_canvas((center[0], center[1] + spec.unit_offset * face))
    draw.text(unit_anchor, spec.unit, font=unit_font, fill=spec.ink_color, anchor="mm")

    if spec.stop_pin:
        circle(
            draw, polar(scale.min_angle - 6.0, tick_mid), 0.02 * face, spec.ink_color
        )

    needle_angle = scale.value_to_angle(value)
    needle_len = spec.needle_length * face
    needle_w = spec.needle_width * face
    tip_w = needle_w if spec.needle_style == "line" else needle_w * 0.2
    bar(
        draw,
        needle_angle,
        -spec.needle_tail * face,
        needle_len,
        needle_w,
        tip_w,
        spec.needle_color,
    )
    circle(draw, center, spec.hub_radius * face, spec.hub_color)

    # BOX averages each ss x ss block: exact area sampling for an integer factor.
    image = canvas.resize((size, size), Image.Resampling.BOX)
    labels = DialLabels(
        value=value,
        fraction=(value - scale.min_value) / scale.span,
        min_value=scale.min_value,
        max_value=scale.max_value,
        unit=spec.unit,
        center=center,
        needle_tip=polar(needle_angle, needle_len),
        min_mark=polar(scale.min_angle, tick_mid),
        max_mark=polar(scale.max_angle, tick_mid),
        radius=outer,
        bbox=(
            center[0] - outer,
            center[1] - outer,
            center[0] + outer,
            center[1] + outer,
        ),
    )
    return RenderedDial(image=image, labels=labels)


def render_random(
    seed: int, size: int = 512, out_of_range: float = 0.1
) -> RenderedDial:
    """Sample a spec and a value from ``seed`` and render them."""
    rng = np.random.default_rng(seed)
    spec = sample_spec(rng)
    return render_dial(spec, sample_value(spec.scale, rng, out_of_range), size)

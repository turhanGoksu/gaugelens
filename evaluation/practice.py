"""A practice set for the person who labels the test set.

    python -m evaluation.practice data/practice

Writes synthetic gauges whose answers are known, and a labeling page in
practice mode: after each label it shows the right answer, the error in % of
full scale, and why. The images are synthetic, so the practice never touches
the test set. Each item illustrates one rule of docs/labeling-guideline.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from evaluation.label_tool import write_page
from gaugelens.geometry import GaugeScale
from gaugelens.synth import BackgroundPool, CaptureParams, capture, render_dial
from gaugelens.synth.dial import DialSpec, RenderedDial, sample_spec

SIZE = 640
LIGHT = {"face_color": (246, 245, 238), "ink_color": (30, 30, 30)}
DARK = {"face_color": (24, 24, 28), "ink_color": (236, 236, 232)}


@dataclass(frozen=True)
class Case:
    """One practice item. ``value`` is where the needle is drawn."""

    kind: str  # "reading", "drag_pointer", "two_gauges", "drawing"
    lo: float
    hi: float
    unit: str
    step: float
    minor: int
    value: float
    status: str = "value"
    explanation: str = ""
    sweep: float = 270.0
    tilt: float = 0.0
    face: str = "light"
    needle: tuple[int, int, int] = (200, 30, 30)
    glare: str = ""  # "away" from the tip, or "tip" to hide it
    stop_pin: bool = False


CASES = [
    Case(
        "reading",
        0,
        10,
        "bar",
        1,
        5,
        4.32,
        explanation="Smallest division 0.2 bar (5 small ticks per number), so read to "
        "0.04. The tip is 3/5 of the way from 4.2 to 4.4: 4.32.",
    ),
    Case(
        "drawing",
        0,
        6,
        "bar",
        1,
        5,
        2.4,
        explanation="A flat, perfectly clean graphic "
        "with no scene around it is a drawing, not a photo: not_a_photo.",
    ),
    Case(
        "reading",
        0,
        1.6,
        "MPa",
        0.2,
        4,
        0.73,
        tilt=12,
        explanation="Divisions of 0.05 MPa, so read to 0.01. The photo is slightly "
        "tilted: read it as it appears.",
    ),
    Case(
        "reading",
        0,
        6,
        "bar",
        1,
        5,
        -6 / 270 * 6,
        status="below_range",
        stop_pin=True,
        explanation="The needle rests on the stop pin, left of 0 by more than its own "
        "width: below_range, with no value.",
    ),
    Case(
        "reading",
        -1,
        3,
        "bar",
        0.5,
        5,
        1.46,
        face="dark",
        needle=(240, 140, 30),
        explanation="The scale starts below zero, so min is -1. Divisions of 0.1 bar, "
        "read to 0.02: 1.46.",
    ),
    Case(
        "two_gauges",
        0,
        10,
        "bar",
        1,
        5,
        6.0,
        explanation="Two complete gauges in one "
        "picture: which one is the subject? Excluded: several_gauges.",
    ),
    Case(
        "reading",
        0,
        600,
        "kPa",
        100,
        5,
        268,
        tilt=25,
        explanation="Strongly tilted. Read where the tip appears, without correcting "
        "in your head. Divisions of 20 kPa, read to 4: 268.",
    ),
    Case(
        "reading",
        0,
        4,
        "bar",
        0.5,
        5,
        0.0,
        explanation="The needle sits exactly on 0. That is status value with value 0, "
        "not below_range.",
    ),
    Case(
        "drag_pointer",
        0,
        16,
        "bar",
        2,
        4,
        7.0,
        explanation="A second, thin red needle "
        "(a drag pointer that remembers the maximum) makes it out_of_scope.",
    ),
    Case(
        "reading",
        0,
        160,
        "psi",
        20,
        4,
        103,
        glare="away",
        needle=(20, 20, 20),
        explanation="Divisions of 5 psi, read to 1. The glare does not cover the tip, "
        "so the gauge stays readable: 103.",
    ),
    Case(
        "reading",
        0,
        16,
        "bar",
        2,
        4,
        16 * 1.08,
        status="above_range",
        explanation="The needle is past 16 by more than its width: above_range. Do not "
        "write 16.",
    ),
    Case(
        "reading",
        0,
        120,
        "°C",
        20,
        2,
        56,
        sweep=240,
        explanation="One small tick between numbers: divisions of 10 °C, read to 2. "
        "The sweep is 240°, not 270°; the ends are where the end numbers are: 56.",
    ),
    Case(
        "reading",
        0,
        10,
        "bar",
        1,
        5,
        7.1,
        status="unreadable",
        glare="tip",
        explanation="A reflection hides the needle, so its direction cannot be "
        "found within one division: unreadable. The gauge itself is eligible. If "
        "only the tip were hidden, you would extend the visible part instead.",
    ),
    Case(
        "reading",
        0,
        25,
        "bar",
        5,
        5,
        18.4,
        sweep=300,
        needle=(25, 45, 120),
        explanation="Divisions of 1 bar, read to 0.2: 18.4. The sweep is 300°.",
    ),
]


def _spec(case: Case, seed: int) -> DialSpec:
    base = sample_spec(np.random.default_rng(seed))
    gap = 360.0 - case.sweep
    scale = GaugeScale(case.lo, case.hi, 180.0 + gap / 2, 180.0 - gap / 2)
    decimals = next(
        d
        for d in range(4)
        if all(math.isclose(n, round(n, d)) for n in (case.lo, case.step))
    )
    colors = DARK if case.face == "dark" else LIGHT
    return dataclasses.replace(
        base,
        scale=scale,
        unit=case.unit,
        major_step=case.step,
        minor_divisions=case.minor,
        decimals=decimals,
        needle_color=case.needle,
        hub_color=case.needle,
        needle_length=base.tick_outer - 0.3 * base.major_tick_length,
        stop_pin=case.stop_pin,
        scale_arc=False,
        **colors,
    )


def _blob(
    dial: RenderedDial,
    angle: float,
    radius_share: float,
    strength: float,
    spread_share: float = 0.22,
) -> None:
    """A soft white reflection centered at ``angle`` on the dial (in place)."""
    image = np.asarray(dial.image).astype(np.float32)
    cx, cy = dial.labels.center
    r = dial.labels.radius * radius_share
    a = math.radians(angle)
    bx, by = cx + r * math.sin(a), cy - r * math.cos(a)
    ys, xs = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    spread = spread_share * dial.labels.radius
    falloff = np.exp(-(((xs - bx) ** 2 + (ys - by) ** 2) / (2 * spread**2)))
    mask = np.minimum(strength * falloff, 1.0)  # strength > 1 gives a flat white core
    image[..., :3] += mask[..., None] * (255 - image[..., :3])
    dial.image.paste(Image.fromarray(image.clip(0, 255).astype(np.uint8), "RGBA"))


def _drag_pointer(dial: RenderedDial, spec: DialSpec, value: float) -> None:
    draw = ImageDraw.Draw(dial.image)
    cx, cy = dial.labels.center
    length = dial.labels.radius * (1 - spec.bezel_width) * (spec.tick_outer - 0.02)
    a = math.radians(spec.scale.value_to_angle(value))
    tip = (cx + length * math.sin(a), cy - length * math.cos(a))
    draw.line([dial.labels.center, tip], fill=(215, 30, 30, 255), width=4)


def _params(width: int, height: int, tilt: float) -> CaptureParams:
    return CaptureParams(
        width=width,
        height=height,
        dial_diameter=0.82 * min(width, height),
        tilt_x=0.4 * tilt,
        tilt_y=-0.9 * tilt,
        roll=3.0,
        focal=2.5,
        glare=0.0,
        shadow=0.0,
        light_gradient=0.15,
        brightness=1.0,
        contrast=1.0,
        color_gains=(1.0, 1.0, 1.0),
        blur_sigma=0.6,
        motion_blur=0.0,
        noise_sigma=0.008,
        jpeg_quality=88,
    )


def _render(case: Case, seed: int, pool: BackgroundPool) -> Image.Image:
    rng = np.random.default_rng(seed)
    spec = _spec(case, seed)
    if case.kind == "drawing":
        dial = render_dial(spec, case.value, size=SIZE - 40)
        canvas = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
        canvas.paste(dial.image, (20, 20), dial.image)
        return canvas
    if case.kind == "two_gauges":
        background, _ = pool.sample(rng, SIZE, SIZE)
        halves = []
        for i, half in enumerate(
            (background[:, : SIZE // 2], background[:, SIZE // 2 :])
        ):
            other = (
                _spec(dataclasses.replace(case, needle=(20, 20, 20)), seed + 1)
                if i
                else spec
            )
            dial = render_dial(other, case.value + 2.7 * i, size=300)
            params = dataclasses.replace(
                _params(SIZE // 2, SIZE, 0.0), dial_diameter=280.0
            )
            halves.append(np.asarray(capture(dial, params, half.copy(), rng).image))
        return Image.fromarray(np.hstack(halves))

    dial = render_dial(spec, case.value, size=560)
    if case.kind == "drag_pointer":
        _drag_pointer(dial, spec, case.hi * 0.81)
    needle_angle = spec.scale.value_to_angle(case.value)
    if case.glare == "tip":
        # Large enough to hide the whole needle, not just its tip: a careful
        # labeler could otherwise extend the visible part.
        _blob(dial, needle_angle, 0.45, 2.5, spread_share=0.38)
    elif case.glare == "away":
        _blob(dial, needle_angle + 150, 0.5, 0.75)
    background, _ = pool.sample(rng, SIZE, SIZE)
    return capture(dial, _params(SIZE, SIZE, case.tilt), background, rng).image


def _answer(case: Case) -> dict[str, Any]:
    if case.kind == "drawing":
        return {"eligible": "no", "exclusion_reason": "not_a_photo"}
    if case.kind == "two_gauges":
        return {"eligible": "no", "exclusion_reason": "several_gauges"}
    if case.kind == "drag_pointer":
        return {"eligible": "no", "exclusion_reason": "out_of_scope"}
    answer: dict[str, Any] = {
        "eligible": "yes",
        "min": case.lo,
        "max": case.hi,
        "unit": case.unit,
        "status": case.status,
    }
    if case.status == "value":
        division = case.step / case.minor
        fifth = division / 5
        answer |= {
            "exact": case.value,
            "value": round(round(case.value / fifth) * fifth, 6),
            "division": division,
        }
    return answer


def write_practice(dest: str | Path, backgrounds: str | Path | None = None) -> Path:
    """Render the practice items into ``dest`` and write its labeling page."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if backgrounds and Path(backgrounds).exists():
        pool = BackgroundPool.from_dir(backgrounds, photo_probability=1.0, split="dev")
    else:
        pool = BackgroundPool()
    rows = []
    for number, case in enumerate(CASES, start=1):
        name = f"practice_{number:02d}.jpg"
        _render(case, 7000 + number, pool).convert("RGB").save(dest / name, quality=92)
        rows.append(
            {
                "id": name.removesuffix(".jpg"),
                "file": name,
                "sha256": hashlib.sha256((dest / name).read_bytes()).hexdigest(),
                "title": f"Practice {number} of {len(CASES)}",
                "page": "",
                "license_name": "synthetic",
                "artist": "gaugelens",
                "answer": _answer(case) | {"explanation": case.explanation},
            }
        )
    (dest / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    return write_page(dest)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evaluation.practice")
    parser.add_argument("dest")
    parser.add_argument("--backgrounds", default="data/backgrounds")
    args = parser.parse_args()
    print(f"open {write_practice(args.dest, args.backgrounds)}")


if __name__ == "__main__":
    main()

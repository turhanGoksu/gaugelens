"""Score readings against labels.

Errors are in fractions of full scale (FS). A reading is judged by what the
library promised: a number with status ``ok`` must be right; ``low_confidence``
is a number with a warning; ``unreadable`` and ``no_gauge`` are explicit
failures; ``below_range`` / ``above_range`` must match the needle's side.

Tolerances here are for reporting only; the project's official tolerance is
decided before the test set is run.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

from gaugelens.geometry import DEFAULT_EDGE_MARGIN
from gaugelens.status import Status

Expected = Literal["in_range", "below_range", "above_range"]
ANSWERED: tuple[Status, ...] = ("ok", "low_confidence")
ABSTAINED: tuple[Status, ...] = ("unreadable", "no_gauge")


def expected_position(
    fraction: float, edge_margin: float = DEFAULT_EDGE_MARGIN
) -> Expected:
    if fraction < -edge_margin:
        return "below_range"
    if fraction > 1.0 + edge_margin:
        return "above_range"
    return "in_range"


@dataclass(frozen=True)
class Outcome:
    expected: Expected
    status: Status
    error: float | None
    """|reading - truth| in fractions of FS, when a number was expected and given."""


def score(label: dict[str, Any], status: Status, value: float | None) -> Outcome:
    expected = expected_position(label["fraction"])
    error = None
    if expected == "in_range" and status in ANSWERED and value is not None:
        span = label["max_value"] - label["min_value"]
        truth = min(max(label["value"], label["min_value"]), label["max_value"])
        error = abs(value - truth) / span
    return Outcome(expected, status, error)


def summarize(outcomes: list[Outcome], tolerances=(0.01, 0.02, 0.05)) -> dict[str, Any]:
    """Shares are of the in-range images unless the key says otherwise."""
    in_range = [o for o in outcomes if o.expected == "in_range"]
    outside = [o for o in outcomes if o.expected != "in_range"]
    n = max(len(in_range), 1)
    errors = sorted(o.error for o in in_range if o.error is not None)
    summary: dict[str, Any] = {
        "images": len(outcomes),
        "in_range_images": len(in_range),
        "statuses": dict(Counter(o.status for o in outcomes)),
        "answered": sum(o.status in ANSWERED for o in in_range) / n,
        "abstained": sum(o.status in ABSTAINED for o in in_range) / n,
        "false_out_of_range": sum(
            o.status in ("below_range", "above_range") for o in in_range
        )
        / n,
        "median_error": errors[len(errors) // 2] if errors else None,
    }
    for t in tolerances:
        key = f"{t:.0%}"
        summary[f"within_{key}"] = (
            sum(o.error is not None and o.error <= t for o in in_range) / n
        )
        summary[f"confidently_wrong_{key}"] = (
            sum(
                o.status == "ok" and o.error is not None and o.error > t
                for o in in_range
            )
            / n
        )
    if outside:
        summary["out_of_range_images"] = len(outside)
        summary["out_of_range_correct"] = sum(
            o.status == o.expected for o in outside
        ) / len(outside)
        # A number with status ok for a needle off the scale: the worst case.
        summary["out_of_range_given_ok"] = sum(o.status == "ok" for o in outside) / len(
            outside
        )
    return summary

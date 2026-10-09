"""Reading statuses shared by every design (docs/decisions.md, D1 and D4)."""

from typing import Literal

Status = Literal[
    "ok",
    "low_confidence",
    "unreadable",
    "no_gauge",
    "below_range",
    "above_range",
]

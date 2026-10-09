"""Write a local labeling page for test candidates (docs/labeling-guideline.md).

    python -m evaluation.label_tool data/raw/commons

Open ``label.html`` in that folder with a browser. Labels are kept in the
browser's local storage as you go; "Export CSV" downloads them. The page
loads the images from the same folder and never sends anything anywhere.

The relabel pass shows a fixed 10% of the candidates, chosen by image hash,
without the first labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXCLUSION_REASONS = [
    "not_a_photo",
    "several_gauges",
    "out_of_scope",
    "counterclockwise",
    "range_illegible",
]
STATUSES = ["value", "below_range", "above_range", "unreadable"]
COLUMNS = [
    "id",
    "file",
    "sha256",
    "eligible",
    "exclusion_reason",
    "min",
    "max",
    "unit",
    "status",
    "value",
    "sure",
    "notes",
    "labeled_at",
]
RELABEL_FRACTION = 0.10


def in_relabel_sample(sha256: str) -> bool:
    """A fixed ~10% of the images, chosen by hash so it never changes."""
    return int(sha256[:8], 16) / 2**32 < RELABEL_FRACTION


_TEMPLATE = Path(__file__).with_name("label_page.html")


def write_page(folder: str | Path) -> Path:
    """Write ``label.html`` next to the candidates listed in ``manifest.jsonl``."""
    folder = Path(folder)
    rows = [
        json.loads(line)
        for line in (folder / "manifest.jsonl").read_text().splitlines()
    ]
    items = [
        {
            "id": r["id"],
            "file": r["file"],
            "sha256": r["sha256"],
            "title": r["title"].removeprefix("File:"),
            "page": r["page"],
            "license_name": r["license_name"],
            "artist": r["artist"],
            "relabel": in_relabel_sample(r["sha256"]),
        }
        for r in rows
    ]
    data = {
        "items": items,
        "columns": COLUMNS,
        "reasons": EXCLUSION_REASONS,
        "statuses": STATUSES,
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = folder / "label.html"
    page.write_text(_TEMPLATE.read_text().replace("__DATA__", payload))
    return page


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evaluation.label_tool")
    parser.add_argument("folder")
    args = parser.parse_args()
    print(f"open {write_page(args.folder)}")


if __name__ == "__main__":
    main()

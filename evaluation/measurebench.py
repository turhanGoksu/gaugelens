"""Download MeasureBench dial gauges as real dev data (docs/decisions.md, D10).

MeasureBench (FlagEval, CC BY-SA 4.0) gives, for each real photo, an interval
of accepted readings and the unit. It gives no scale range, so ``min``,
``max`` and the scale's unit are labeled separately. Only pressure gauges and
blood pressure gauges with a dial are taken. A photo that also appears among
the test candidates is dropped here; the test set keeps it (D10).

    python -m evaluation.measurebench data/external/measurebench \\
        --test-images data/raw/commons
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from evaluation import dedup
from evaluation.commons import Fetch, _fetch

ROWS = "https://datasets-server.huggingface.co/rows"
DATASET = "FlagEval/MeasureBench"
SPLIT = "real_world"
DIAL_TYPES = ("pressure_gauge", "sphygmomanometer")


def fetch_rows(fetch: Fetch, split: str = SPLIT) -> Iterable[dict[str, Any]]:
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": 100,
            }
        )
        data = json.loads(fetch(f"{ROWS}?{query}"))
        for row in data["rows"]:
            yield row["row"]
        offset += len(data["rows"])
        if not data["rows"] or offset >= data["num_rows_total"]:
            return


def readings(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Accepted readings as [{"interval": [lo, hi], "units": [...]}, ...].

    A dial with two scales has one interval per scale."""
    kwargs = json.loads(row["evaluator_kwargs"])
    if "intervals" in kwargs:
        return [
            {"interval": [float(lo), float(hi)], "units": units}
            for (lo, hi), units in zip(
                kwargs["intervals"], kwargs["units"], strict=True
            )
        ]
    lo, hi = kwargs["interval"]
    return [{"interval": [float(lo), float(hi)], "units": kwargs["units"]}]


def revision(image_url: str) -> str:
    """The dataset commit an image URL was served from."""
    match = re.search(r"/--/([0-9a-f]{40})/--/", image_url)
    return match.group(1) if match else ""


def collect(
    dest: str | Path, fetch: Fetch = _fetch, test_images: str | Path | None = None
) -> Counter[str]:
    """Download dial pressure gauges into ``dest`` with a ``manifest.jsonl``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    rows = []
    for row in fetch_rows(fetch):
        counts["rows"] += 1
        if row["design"] != "dial" or row["image_type"] not in DIAL_TYPES:
            continue
        url = row["image"]["src"]
        ext = Path(urllib.parse.urlparse(url).path).suffix or ".jpg"
        name = f"{row['question_id']}{ext}"
        path = dest / name
        if not path.exists():
            path.write_bytes(fetch(url))
        rows.append(
            {
                "id": row["question_id"],
                "file": name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "image_type": row["image_type"],
                "question": row["question"],
                "readings": readings(row),
                "source": row["meta_info"]["source"],
                "uploader": row["meta_info"]["uploader"],
                "revision": revision(url),
            }
        )
        counts["kept"] += 1

    if test_images is not None:
        test_dir = Path(test_images)
        test = {
            r["id"]: dedup.dhash(test_dir / r["file"])
            for r in map(json.loads, (test_dir / "manifest.jsonl").open())
        }
        dev = {r["id"]: dedup.dhash(dest / r["file"]) for r in rows}
        dropped = {d for d, _, _ in dedup.near_duplicates(dev, test)}
        rows = [r for r in rows if r["id"] not in dropped]
        counts["dropped: also a test candidate"] = len(dropped)
        counts["kept"] -= len(dropped)

    with open(dest / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evaluation.measurebench")
    parser.add_argument("dest")
    parser.add_argument("--test-images", help="folder with the test candidates")
    args = parser.parse_args()
    for key, value in sorted(collect(args.dest, test_images=args.test_images).items()):
        print(f"{value:5d}  {key}")


if __name__ == "__main__":
    main()

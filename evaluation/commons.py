"""Collect test-set candidates from Wikimedia Commons (docs/decisions.md, D10).

Only metadata decides what is collected: category, license, format and size.
Whether a photo shows an eligible gauge, and what it reads, is decided by a
person following docs/labeling-guideline.md. No reader and no model looks at
these images before their labels are committed.

    python -m evaluation.commons data/raw/commons
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from gaugelens import __version__

API = "https://commons.wikimedia.org/w/api.php"
MEASUREBENCH_ROWS = "https://datasets-server.huggingface.co/rows"
USER_AGENT = f"gaugelens/{__version__} (+https://github.com/turhanGoksu/gaugelens)"

# (category, how many levels of subcategories to follow)
CATEGORIES = [
    ("Category:Pressure gauges", 2),
    ("Category:Manometers", 2),
    ("Category:Vacuum gauges", 2),
    ("Category:Gauges", 1),
]
MIN_SIDE = 400
DOWNLOAD_WIDTH = 1600
MIMES = {"image/jpeg": "jpg", "image/png": "png"}

Fetch = Callable[[str], bytes]


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def _api(fetch: Fetch, params: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params | {"format": "json", "maxlag": 5})
    return json.loads(fetch(f"{API}?{query}"))


def license_allowed(code: str) -> bool:
    """CC0, public domain, CC BY and CC BY-SA. Not GFDL-only, not unclear."""
    code = code.lower()
    return code == "cc0" or code.startswith("pd") or code.startswith("cc-by")


def _plain(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def walk(
    fetch: Fetch, category: str, depth: int, found: dict[str, str], seen: set[str]
) -> None:
    """Add every file under ``category`` to ``found`` (title -> category)."""
    if category in seen:
        return
    seen.add(category)
    params: dict[str, Any] = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": "file|subcat",
        "cmlimit": 500,
    }
    while True:
        data = _api(fetch, params)
        for member in data["query"]["categorymembers"]:
            if member["ns"] == 6:
                found.setdefault(member["title"], category)
            elif member["ns"] == 14 and depth > 0:
                walk(fetch, member["title"], depth - 1, found, seen)
        if "continue" not in data:
            return
        params |= data["continue"]


def measurebench_commons_files(fetch: Fetch) -> set[str]:
    """Commons file titles used by MeasureBench, whose readings are public."""
    titles: set[str] = set()
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "dataset": "FlagEval/MeasureBench",
                "config": "default",
                "split": "real_world",
                "offset": offset,
                "length": 100,
            }
        )
        data = json.loads(fetch(f"{MEASUREBENCH_ROWS}?{query}"))
        for row in data["rows"]:
            match = re.search(
                r"commons\.wikimedia\.org/wiki/(File:[^'\"\s]+)",
                str(row["row"]["meta_info"]),
            )
            if match:
                titles.add(urllib.parse.unquote(match.group(1)).replace("_", " "))
        offset += len(data["rows"])
        if not data["rows"] or offset >= data["num_rows_total"]:
            return titles


def _image_info(fetch: Fetch, titles: list[str]) -> Iterable[dict[str, Any]]:
    for i in range(0, len(titles), 50):
        data = _api(
            fetch,
            {
                "action": "query",
                "prop": "imageinfo",
                "iiprop": "url|size|mime|sha1|timestamp|extmetadata",
                "iiurlwidth": DOWNLOAD_WIDTH,
                "titles": "|".join(titles[i : i + 50]),
            },
        )
        yield from data["query"]["pages"].values()


def collect(dest: str | Path, fetch: Fetch = _fetch) -> Counter[str]:
    """Download eligible-by-metadata candidates into ``dest`` with a
    ``manifest.jsonl``; returns counts of what was kept and why not."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    found: dict[str, str] = {}
    seen: set[str] = set()
    for category, depth in CATEGORIES:
        walk(fetch, category, depth, found, seen)
    published = measurebench_commons_files(fetch)

    counts: Counter[str] = Counter(files=len(found), categories=len(seen))
    rows = []
    for page in _image_info(fetch, sorted(found)):
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        code = meta.get("License", {}).get("value", "")
        if page["title"] in published:
            counts["skipped: reading published in MeasureBench"] += 1
            continue
        if not license_allowed(code):
            counts["skipped: license"] += 1
            continue
        if info.get("mime") not in MIMES:
            counts["skipped: format"] += 1
            continue
        if min(info.get("width", 0), info.get("height", 0)) < MIN_SIDE:
            counts["skipped: too small"] += 1
            continue

        name = f"commons_{page['pageid']}.{MIMES[info['mime']]}"
        path = dest / name
        if not path.exists():
            path.write_bytes(fetch(info.get("thumburl") or info["url"]))
        rows.append(
            {
                "id": name.rsplit(".", 1)[0],
                "file": name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "title": page["title"],
                "page": info.get("descriptionurl", ""),
                "original_url": info["url"],
                "original_sha1": info.get("sha1", ""),
                "downloaded_url": info.get("thumburl") or info["url"],
                "uploaded": info.get("timestamp", ""),
                "category": found[page["title"]],
                "license": code,
                "license_name": meta.get("LicenseShortName", {}).get("value", ""),
                "license_url": meta.get("LicenseUrl", {}).get("value", ""),
                "artist": _plain(meta.get("Artist", {}).get("value", "")),
                "credit": _plain(meta.get("Credit", {}).get("value", "")),
            }
        )
        counts["kept"] += 1

    rows.sort(key=lambda r: r["id"])
    with open(dest / "manifest.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evaluation.commons")
    parser.add_argument("dest")
    args = parser.parse_args()
    for key, value in sorted(collect(args.dest).items()):
        print(f"{value:5d}  {key}")


if __name__ == "__main__":
    main()

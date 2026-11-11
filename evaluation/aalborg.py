"""Aalborg pressure gauge data from Kaggle, the second real dev source (D10).

The dataset is about 6.4 GB, so the first step only lists its files; the
download takes the files a pattern picks.

    python -m evaluation.aalborg info
    python -m evaluation.aalborg list data/external/aalborg
    python -m evaluation.aalborg download data/external/aalborg "*.json"
    python -m evaluation.aalborg frames data/external/aalborg

Credentials come from the environment (KAGGLE_USERNAME, KAGGLE_KEY) or, if
they are not set there, from the project's ``.env``. They are sent only to
kaggle.com and never printed. ``frames`` needs no credentials: it checks the
downloaded test videos against ``evaluation/dev_aalborg/videos.csv`` and
writes the labeled frames as PNG files.
"""

from __future__ import annotations

import argparse
import base64
import csv
import fnmatch
import hashlib
import io
import json
import os
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

API = "https://www.kaggle.com/api/v1"
DATASET = "juliusgrassme/pressure-gauge-reader-data"
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
LABELS_DIR = Path(__file__).resolve().parent / "dev_aalborg"
# The paper's seven test gauges, cropped to the dial and scaled to 512x512.
VIDEOS = "Data/2 Test videos/edited videos"

Fetch = Callable[[str], bytes]


def credentials(env_file: Path = ENV_FILE) -> tuple[str, str]:
    values = {k: os.environ.get(k, "") for k in ("KAGGLE_USERNAME", "KAGGLE_KEY")}
    if not all(values.values()) and env_file.exists():
        for line in env_file.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() in values and not values[key.strip()]:
                values[key.strip()] = value.strip().strip("'\"")
    if not all(values.values()):
        raise SystemExit("Set KAGGLE_USERNAME and KAGGLE_KEY (see .env.example).")
    return values["KAGGLE_USERNAME"], values["KAGGLE_KEY"]


def make_fetch(user: str, key: str) -> Fetch:
    token = base64.b64encode(f"{user}:{key}".encode()).decode()
    headers = {"Authorization": f"Basic {token}", "User-Agent": "gaugelens-dev"}

    def fetch(url: str) -> bytes:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.read()

    return fetch


def info(fetch: Fetch, dataset: str = DATASET) -> dict[str, Any]:
    data = json.loads(fetch(f"{API}/datasets/view/{dataset}"))
    keys = ("title", "licenseName", "totalBytes", "currentVersionNumber", "lastUpdated")
    return {k: data.get(k) for k in keys} | {"description": data.get("description", "")}


def list_files(fetch: Fetch, dataset: str = DATASET) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    token = ""
    while True:
        query = urllib.parse.urlencode(
            {"pageSize": 200} | ({"pageToken": token} if token else {})
        )
        data = json.loads(fetch(f"{API}/datasets/list/{dataset}?{query}"))
        files += [
            {"name": f["name"], "bytes": f.get("totalBytes", 0)}
            for f in data.get("datasetFiles") or []
        ]
        token = data.get("nextPageToken") or ""
        if not token:
            return files


def summary(files: list[dict[str, Any]]) -> dict[str, Any]:
    """File counts and sizes by top folder and by extension."""
    folders: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    for f in files:
        top = f["name"].split("/")[0] if "/" in f["name"] else "."
        folders[top] += 1
        sizes[top] += f["bytes"]
    exts = Counter(Path(f["name"]).suffix.lower() or "(none)" for f in files)
    return {
        "files": len(files),
        "gigabytes": round(sum(f["bytes"] for f in files) / 1e9, 2),
        "folders": {k: [folders[k], round(sizes[k] / 1e6, 1)] for k in sorted(folders)},
        "extensions": dict(exts.most_common()),
    }


def unwrap(data: bytes, name: str) -> bytes:
    """Kaggle sends larger files inside a zip that holds only that file."""
    if data[:4] != b"PK\x03\x04" or name.lower().endswith(".zip"):
        return data
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.namelist()
        if members != [Path(name).name]:
            raise ValueError(f"{name}: unexpected zip members {members}")
        return archive.read(members[0])


def download(fetch: Fetch, dest: Path, pattern: str, dataset: str = DATASET) -> int:
    listing = json.loads((dest / "files.json").read_text())
    count = 0
    for f in listing:
        if not fnmatch.fnmatch(f["name"], pattern):
            continue
        path = dest / "files" / f["name"]
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        name = urllib.parse.quote(f["name"], safe="")
        data = unwrap(fetch(f"{API}/datasets/download/{dataset}/{name}"), f["name"])
        if len(data) != f["bytes"]:
            raise ValueError(
                f"{f['name']}: got {len(data)} bytes, listing says {f['bytes']}"
            )
        partial = path.with_name(path.name + ".part")
        partial.write_bytes(data)
        partial.replace(path)
        count += 1
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_frames(video: Path, indices: Iterable[int]) -> dict[int, np.ndarray]:
    """Frames by index, decoded in order: seeking is not frame-exact in every codec."""
    wanted = set(indices)
    found: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video}")
    index = 0
    try:
        while len(found) < len(wanted):
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                found[index] = frame
            index += 1
    finally:
        capture.release()
    if missing := sorted(wanted - found.keys()):
        raise ValueError(f"{video.name}: no frames {missing}")
    return found


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_frames(dest: Path, labels_dir: Path = LABELS_DIR) -> int:
    """Write every labeled frame to ``dest/frames/<id>.png``.

    Each video must match the size and SHA-256 in ``videos.csv``, so the
    frame numbers in ``labels.csv`` point at the frames that were labeled.
    """
    videos = {v["gauge"]: v for v in read_csv(labels_dir / "videos.csv")}
    by_gauge: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(labels_dir / "labels.csv"):
        by_gauge[row["gauge"]].append(row)
    out = dest / "frames"
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for gauge, rows in by_gauge.items():
        video = dest / "files" / VIDEOS / videos[gauge]["file"]
        if not video.exists():
            raise SystemExit(
                f"Missing {video}. Run: python -m evaluation.aalborg download "
                f'{dest} "{VIDEOS}/*"'
            )
        if sha256_file(video) != videos[gauge]["sha256"]:
            raise ValueError(f"{video.name} differs from the labeled video")
        frames = read_frames(video, (int(r["frame"]) for r in rows))
        for r in rows:
            cv2.imwrite(str(out / f"{r['id']}.png"), frames[int(r["frame"])])
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evaluation.aalborg")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info")
    p_list = sub.add_parser("list")
    p_list.add_argument("dest")
    p_down = sub.add_parser("download")
    p_down.add_argument("dest")
    p_down.add_argument("pattern")
    p_frames = sub.add_parser("frames")
    p_frames.add_argument("dest")
    args = parser.parse_args()

    if args.command == "frames":
        print(f"wrote {write_frames(Path(args.dest))} frames")
        return
    fetch = make_fetch(*credentials())
    if args.command == "info":
        print(json.dumps(info(fetch), indent=1, ensure_ascii=False))
    elif args.command == "list":
        dest = Path(args.dest)
        dest.mkdir(parents=True, exist_ok=True)
        files = list_files(fetch)
        (dest / "files.json").write_text(json.dumps(files, indent=1))
        print(json.dumps(summary(files), indent=1))
    else:
        print(f"downloaded {download(fetch, Path(args.dest), args.pattern)} files")


if __name__ == "__main__":
    main()

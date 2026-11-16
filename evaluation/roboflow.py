"""Roboflow Universe datasets, candidates for Design A's training data (D10).

    python -m evaluation.roboflow info
    python -m evaluation.roboflow info <workspace>/<project> ...
    python -m evaluation.roboflow download <workspace>/<project> <version>
    python -m evaluation.roboflow select

``info`` prints each project's type, classes, image count and versions; with
no project it goes through ``CANDIDATES``. ``download`` fetches one version as
COCO JSON into ``data/external/roboflow/<workspace>/<project>/v<version>/``
and prints the license lines of the export's README: that, not the search
listing, is the license to verify before use (D10). ``select`` applies D16
to the downloaded ``SOURCES`` and writes ``evaluation/train_roboflow/manifest.csv``:
one row per image, kept or not and why.

The key comes from ROBOFLOW_API_KEY in the environment or ``.env``. It is
sent only to api.roboflow.com, never to the storage host of an export link,
and never printed.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation import dedup, env
from evaluation.aalborg import sha256_file

API = "https://api.roboflow.com"
DEST = Path(__file__).resolve().parents[1] / "data" / "external" / "roboflow"
# From a web search on 2026-11-16; every field is checked through the API.
CANDIDATES = (
    "computer-vision-ygqpn/gauge-reader-idem4",
    "project-kaqwa/keypoints-4lma6",
    "analog-meter-reading/gauge-analog-tndze",
    "ndkcs-studio/gauge-location-detection",
    "itu-weijie-wang/dial-meter-detection-4-points",
    "university-bswxt/gauge-mgyts",
    "azham/ai-camera-pressure-gauge-esj2i-3twce",
    "connecthingz-sdn-bhd/gauge-meter-reading",
    "yolo-lxxyl/analog-gauge-meter",
)

# D16: the versions used for training; neither is augmented.
SOURCES = {
    "project-kaqwa/keypoints-4lma6": "1",
    "computer-vision-ygqpn/gauge-reader-idem4": "11",
}
# D16: what keeps an image out of training. The first matching rule wins.
RULES: tuple[tuple[str, str, str], ...] = (
    # The TS 0-10 bar gauge of MeasureBench's pressure_gauge_0 (dev).
    ("project-kaqwa/keypoints-4lma6", r"^meter_a", "same_gauge_as_dev"),
    # Frames of the Aalborg test video man3 (dev).
    ("project-kaqwa/keypoints-4lma6", r"cropclipscale", "aalborg_dev_video"),
    # Stock footage with a Storyblocks watermark: not the uploader's to license.
    ("project-kaqwa/keypoints-4lma6", r"^needle-on-pressure-gauge", "stock_watermark"),
    # Web catalog and stock photos, whose license the uploader cannot grant.
    ("computer-vision-ygqpn/gauge-reader-idem4", r"^Screenshot", "web_image"),
    ("computer-vision-ygqpn/gauge-reader-idem4", r"^g-test", "web_image"),
    (
        "computer-vision-ygqpn/gauge-reader-idem4",
        r"^\d+_(jpe?g|png|webp)\.rf\.",
        "web_image",
    ),
    # Phone photos of Aalborg dev gauges at their pump stations: man2, man4,
    # man5 and man6 on 2022-06-10; man1 at 13:53 on 2022-06-13.
    ("computer-vision-ygqpn/gauge-reader-idem4", r"^20220610_", "same_gauge_as_dev"),
    (
        "computer-vision-ygqpn/gauge-reader-idem4",
        r"^20220613_1353",
        "same_gauge_as_dev",
    ),
)
MANIFEST = Path(__file__).resolve().parent / "train_roboflow" / "manifest.csv"
MANIFEST_COLUMNS = (
    "dataset",
    "version",
    "split",
    "file",
    "sha256",
    "keep",
    "reason",
    "scene",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".gif"}

Fetch = Callable[[str], bytes]


def make_fetch(key: str) -> Fetch:
    """GET a URL; the key is added only for api.roboflow.com."""

    def fetch(url: str) -> bytes:
        full = url
        if url.startswith(API + "/"):
            sep = "&" if "?" in url else "?"
            full = f"{url}{sep}api_key={urllib.parse.quote(key, safe='')}"
        request = urllib.request.Request(full, headers={"User-Agent": "gaugelens-dev"})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            # Say which URL failed without the query string, which holds the key.
            raise RuntimeError(f"HTTP {error.code} for {url.split('?')[0]}") from None

    return fetch


def _date(value: Any) -> Any:
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, UTC).date().isoformat()
    return value


def _license_fields(data: Any, path: str = "") -> dict[str, Any]:
    """Every field whose name mentions a license, wherever it sits."""
    found: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            where = f"{path}.{key}" if path else key
            if "licen" in key.lower():
                found[where] = value
            found |= _license_fields(value, where)
    elif isinstance(data, list):
        for i, value in enumerate(data):
            found |= _license_fields(value, f"{path}[{i}]")
    return found


def info(fetch: Fetch, project_id: str, dest: Path = DEST) -> dict[str, Any]:
    data = json.loads(fetch(f"{API}/{project_id}"))
    folder = dest / project_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "project.json").write_text(json.dumps(data, indent=1))
    project = data.get("project", {})
    return {
        "id": project_id,
        "type": project.get("type"),
        "images": project.get("images"),
        "classes": project.get("classes"),
        "updated": _date(project.get("updated")),
        "versions": [
            {
                "version": str(v.get("id", "")).rsplit("/", 1)[-1],
                "images": v.get("images"),
                "splits": v.get("splits"),
                "created": _date(v.get("created")),
            }
            for v in data.get("versions", [])
        ],
        "license_fields": _license_fields(data),
    }


def extract(data: bytes, folder: Path) -> int:
    """Unpack a zip into ``folder``; paths that leave it, and links, are refused."""
    folder.mkdir(parents=True, exist_ok=True)
    root = folder.resolve()
    count = 0
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.infolist():
            target = (folder / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"zip path leaves the folder: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"zip holds a link: {member.filename}")
            if member.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
            count += 1
    return count


def coco_summary(folder: Path) -> dict[str, Any]:
    """Images, labels per class and keypoint names in each split's COCO file."""
    splits = {}
    for path in sorted(folder.glob("*/_annotations.coco.json")):
        coco = json.loads(path.read_text())
        names = {c["id"]: c["name"] for c in coco.get("categories", [])}
        keypoints = {
            c["name"]: c["keypoints"]
            for c in coco.get("categories", [])
            if c.get("keypoints")
        }
        labels = Counter(names.get(a["category_id"], "?") for a in coco["annotations"])
        splits[path.parent.name] = {
            "images": len(coco.get("images", [])),
            "labels": dict(labels),
            "keypoints": keypoints,
        }
    return splits


def license_lines(folder: Path) -> list[str]:
    lines = []
    for name in ("README.dataset.txt", "README.roboflow.txt"):
        path = folder / name
        if path.exists():
            lines += [
                f"{name}: {line.strip()}"
                for line in path.read_text(errors="replace").splitlines()
                if "licen" in line.lower()
            ]
    return lines


def download(
    fetch: Fetch, project_id: str, version: str, dest: Path = DEST, wait: float = 5.0
) -> dict[str, Any]:
    url = f"{API}/{project_id}/{version}/coco"
    for _ in range(60):
        meta = json.loads(fetch(url))
        link = (meta.get("export") or {}).get("link")
        if link:
            break
        time.sleep(wait)  # Roboflow builds the export on the first request.
    else:
        raise SystemExit(f"{project_id} v{version}: the export was not ready in time")
    folder = dest / project_id / f"v{version}"
    files = extract(fetch(link), folder)
    return {
        "id": project_id,
        "version": version,
        "folder": str(folder),
        "files": files,
        "license": license_lines(folder),
        "splits": coco_summary(folder),
    }


def rule_reason(dataset: str, file_name: str) -> str | None:
    for source, pattern, reason in RULES:
        if source == dataset and re.search(pattern, file_name):
            return reason
    return None


def reference_hashes(root: Path = DEST.parents[2]) -> dict[str, dict[str, int]]:
    """Perceptual hashes of the test candidates and the dev images."""
    test = root / "data" / "raw" / "commons"
    dev = [
        *(root / "data" / "external" / "measurebench").glob("*"),
        *(root / "data" / "external" / "aalborg" / "frames").glob("*.png"),
    ]
    return {
        "test": {
            p.name: dedup.dhash(p)
            for p in sorted(test.iterdir())
            if p.suffix.lower() in IMAGE_SUFFIXES
        },
        "dev": {
            p.name: dedup.dhash(p)
            for p in sorted(dev)
            if p.suffix.lower() in IMAGE_SUFFIXES
        },
    }


def select(
    dest: Path,
    reference: Mapping[str, Mapping[str, int]],
    sources: Mapping[str, str] = SOURCES,
) -> list[dict[str, str]]:
    """One row per image of the used versions: kept or not, and why (D16).

    An image no rule excludes is still dropped when it is within
    ``dedup.MAX_BITS`` of a test or dev image (D10). Kept images that are
    near-copies share a ``scene``, so a later train/validation split can keep
    each scene on one side.
    """
    rows: list[dict[str, str]] = []
    hashes: list[int] = []
    for dataset, version in sources.items():
        folder = dest / dataset / f"v{version}"
        if not folder.exists():
            raise SystemExit(
                f"Missing {folder}. Run: python -m evaluation.roboflow download "
                f"{dataset} {version}"
            )
        for ann in sorted(folder.glob("*/_annotations.coco.json")):
            for image in json.loads(ann.read_text())["images"]:
                path = ann.parent / image["file_name"]
                h = dedup.dhash(path)
                reason = rule_reason(dataset, image["file_name"])
                for name, known in reference.items():
                    if reason is None and any(
                        dedup.distance(h, k) <= dedup.MAX_BITS for k in known.values()
                    ):
                        reason = f"near_{name}"
                rows.append(
                    {
                        "dataset": dataset,
                        "version": version,
                        "split": ann.parent.name,
                        "file": image["file_name"],
                        "sha256": sha256_file(path),
                        "keep": "no" if reason else "yes",
                        "reason": reason or "",
                        "scene": "",
                    }
                )
                hashes.append(h)
    kept = [i for i, row in enumerate(rows) if row["keep"] == "yes"]
    parent = {i: i for i in kept}

    def root_of(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, i in enumerate(kept):
        for j in kept[a + 1 :]:
            if dedup.distance(hashes[i], hashes[j]) <= dedup.MAX_BITS:
                parent[root_of(i)] = root_of(j)
    numbers: dict[int, int] = {}
    for i in kept:
        rows[i]["scene"] = str(numbers.setdefault(root_of(i), len(numbers)))
    return rows


def write_manifest(rows: list[dict[str, str]], path: Path = MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evaluation.roboflow")
    sub = parser.add_subparsers(dest="command", required=True)
    p_info = sub.add_parser("info")
    p_info.add_argument("projects", nargs="*", default=list(CANDIDATES))
    p_down = sub.add_parser("download")
    p_down.add_argument("project")
    p_down.add_argument("version")
    sub.add_parser("select")
    args = parser.parse_args()

    if args.command == "select":
        rows = select(DEST, reference_hashes())
        write_manifest(rows)
        counts = Counter((r["dataset"], r["reason"] or "kept") for r in rows)
        for (dataset, reason), n in sorted(counts.items()):
            print(f"{n:5d}  {dataset}  {reason}")
        kept = [r for r in rows if r["keep"] == "yes"]
        print(f"kept {len(kept)} images in {len({r['scene'] for r in kept})} scenes")
        print(f"manifest: {MANIFEST}")
        return
    fetch = make_fetch(env.read(("ROBOFLOW_API_KEY",))["ROBOFLOW_API_KEY"])

    if args.command == "info":
        for project_id in args.projects:
            try:
                result = info(fetch, project_id)
            except RuntimeError as error:
                result = {"id": project_id, "error": str(error)}
            print(json.dumps(result, ensure_ascii=False))
    else:
        result = download(fetch, args.project, args.version)
        print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

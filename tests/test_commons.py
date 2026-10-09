import json
import urllib.parse

import pytest

from evaluation import commons


@pytest.mark.parametrize(
    ("code", "allowed"),
    [
        ("cc0", True),
        ("pd", True),
        ("pd-old-70", True),
        ("cc-by-4.0", True),
        ("cc-by-sa-3.0", True),
        ("gfdl", False),
        ("", False),
        ("attribution", False),
    ],
)
def test_license_allowlist(code, allowed) -> None:
    assert commons.license_allowed(code) is allowed


def _info(pageid, title, code, mime="image/jpeg", size=(1200, 900)):
    return {
        "pageid": pageid,
        "title": title,
        "imageinfo": [
            {
                "url": f"https://upload.wikimedia.org/{pageid}.jpg",
                "thumburl": f"https://upload.wikimedia.org/thumb/{pageid}.jpg",
                "descriptionurl": f"https://commons.wikimedia.org/wiki/{title}",
                "width": size[0],
                "height": size[1],
                "mime": mime,
                "sha1": "abc",
                "timestamp": "2020-01-01T00:00:00Z",
                "extmetadata": {
                    "License": {"value": code},
                    "Artist": {"value": "<a href='x'>Ada</a>"},
                },
            }
        ],
    }


class FakeCommons:
    def __init__(self) -> None:
        self.pages = {
            "File:Ok.jpg": _info(1, "File:Ok.jpg", "cc-by-sa-4.0"),
            "File:Gfdl.jpg": _info(2, "File:Gfdl.jpg", "gfdl"),
            "File:Tiny.jpg": _info(3, "File:Tiny.jpg", "cc0", size=(300, 200)),
            "File:Svg.svg": _info(4, "File:Svg.svg", "cc0", mime="image/svg+xml"),
            "File:Published.jpg": _info(5, "File:Published.jpg", "cc0"),
        }

    def __call__(self, url: str) -> bytes:
        if url.startswith(commons.MEASUREBENCH_ROWS):
            meta = "{'source': 'https://commons.wikimedia.org/wiki/File:Published.jpg'}"
            return json.dumps(
                {"rows": [{"row": {"meta_info": meta}}], "num_rows_total": 1}
            ).encode()
        if url.startswith(commons.API):
            params = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
            if params.get("list") == "categorymembers":
                members = [{"ns": 6, "title": t} for t in self.pages]
                return json.dumps({"query": {"categorymembers": members}}).encode()
            titles = params["titles"].split("|")
            pages = {str(i): self.pages[t] for i, t in enumerate(titles)}
            return json.dumps({"query": {"pages": pages}}).encode()
        return b"image bytes"


def test_collect_keeps_only_allowed_unpublished_images(tmp_path) -> None:
    counts = commons.collect(tmp_path, fetch=FakeCommons())
    assert counts["kept"] == 1
    assert counts["skipped: license"] == 1
    assert counts["skipped: too small"] == 1
    assert counts["skipped: format"] == 1
    assert counts["skipped: reading published in MeasureBench"] == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "manifest.jsonl").read_text().splitlines()
    ]
    assert [r["title"] for r in rows] == ["File:Ok.jpg"]
    assert rows[0]["artist"] == "Ada"
    assert len(rows[0]["sha256"]) == 64
    assert (tmp_path / rows[0]["file"]).read_bytes() == b"image bytes"

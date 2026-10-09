import hashlib
import json
import re
import shutil
import subprocess

import pytest

from evaluation.label_tool import in_relabel_sample, write_page


def test_relabel_sample_is_about_ten_percent_and_stable() -> None:
    hashes = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(5000)]
    picked = [in_relabel_sample(h) for h in hashes]
    assert picked == [in_relabel_sample(h) for h in hashes]
    assert 0.08 < sum(picked) / len(picked) < 0.12


@pytest.fixture
def page(tmp_path):
    rows = [
        {
            "id": f"commons_{i}",
            "file": f"commons_{i}.jpg",
            "sha256": hashlib.sha256(str(i).encode()).hexdigest(),
            "title": f"File:Gauge </script> {i}.jpg",
            "page": f"https://commons.wikimedia.org/wiki/File:Gauge_{i}.jpg",
            "license_name": "CC BY-SA 4.0",
            "artist": "Ada",
        }
        for i in range(3)
    ]
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return write_page(tmp_path).read_text()


def test_page_embeds_every_candidate(page) -> None:
    payload = re.search(
        r'<script type="application/json" id="data">(.*?)</script>', page, re.S
    )
    data = json.loads(payload.group(1).replace("<\\/", "</"))
    assert [item["id"] for item in data["items"]] == [
        "commons_0",
        "commons_1",
        "commons_2",
    ]
    assert data["columns"][0] == "id"


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_page_script_is_valid_javascript(page, tmp_path) -> None:
    script = page.rsplit("<script>", 1)[1].split("</script>", 1)[0]
    path = tmp_path / "page.js"
    path.write_text(script)
    subprocess.run(["node", "--check", str(path)], check=True)

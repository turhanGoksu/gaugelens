import io
import json
import urllib.error
import zipfile

import pytest

from evaluation import roboflow

PROJECT = {
    "project": {"type": "keypoint-detection", "images": 3, "classes": {"gauge": 3},
                "updated": 1700000000},
    "versions": [{"id": "ws/p/2", "images": 3, "splits": {"train": 2, "valid": 1},
                  "created": 1700000000}],
    "workspace": {"license": "CC BY 4.0"},
}  # fmt: skip


def make_zip(files: dict[str, str | bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


COCO = {
    "images": [{"id": 1}, {"id": 2}],
    "categories": [
        {"id": 0, "name": "gauges"},
        {"id": 1, "name": "gauge", "keypoints": ["center", "tip", "min", "max"]},
    ],
    "annotations": [{"category_id": 1}, {"category_id": 1}],
}


def test_key_goes_only_to_the_api_and_never_into_errors(monkeypatch) -> None:
    seen = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def urlopen(request, timeout):
        seen.append(request.full_url)
        if "fail" in request.full_url:
            raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)
        return Response(b"ok")

    monkeypatch.setattr(roboflow.urllib.request, "urlopen", urlopen)
    fetch = roboflow.make_fetch("s3cret")
    fetch(f"{roboflow.API}/ws/p")
    fetch("https://storage.example/export.zip")
    assert seen == [
        f"{roboflow.API}/ws/p?api_key=s3cret",
        "https://storage.example/export.zip",
    ]
    with pytest.raises(RuntimeError) as error:
        fetch(f"{roboflow.API}/ws/fail")
    assert "s3cret" not in str(error.value)
    assert "403" in str(error.value)


def test_info_summarizes_and_finds_license_fields(tmp_path) -> None:
    result = roboflow.info(lambda url: json.dumps(PROJECT).encode(), "ws/p", tmp_path)
    assert result["type"] == "keypoint-detection"
    assert result["versions"] == [
        {"version": "2", "images": 3, "splits": {"train": 2, "valid": 1},
         "created": "2023-11-14"}
    ]  # fmt: skip
    assert result["license_fields"] == {"workspace.license": "CC BY 4.0"}
    assert (tmp_path / "ws" / "p" / "project.json").exists()


def test_download_waits_for_the_export_and_reads_license(tmp_path) -> None:
    export = make_zip(
        {
            "README.dataset.txt": "Provided by a Roboflow user\nLicense: CC BY 4.0\n",
            "train/_annotations.coco.json": json.dumps(COCO),
            "train/a.jpg": b"jpg",
        }
    )
    calls = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        if url.startswith(roboflow.API):
            ready = len(calls) > 1
            body = (
                {"export": {"link": "https://storage.example/x.zip"}} if ready else {}
            )
            return json.dumps(body).encode()
        return export

    result = roboflow.download(fetch, "ws/p", "2", tmp_path, wait=0)
    assert result["license"] == ["README.dataset.txt: License: CC BY 4.0"]
    assert result["splits"]["train"]["images"] == 2
    assert result["splits"]["train"]["labels"] == {"gauge": 2}
    assert result["splits"]["train"]["keypoints"] == {
        "gauge": ["center", "tip", "min", "max"]
    }
    assert (tmp_path / "ws" / "p" / "v2" / "train" / "a.jpg").read_bytes() == b"jpg"


def test_extract_refuses_paths_that_leave_the_folder(tmp_path) -> None:
    with pytest.raises(ValueError, match="leaves"):
        roboflow.extract(make_zip({"../evil.txt": "x"}), tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


GR = "computer-vision-ygqpn/gauge-reader-idem4"
KQ = "project-kaqwa/keypoints-4lma6"


@pytest.mark.parametrize(
    ("dataset", "name", "reason"),
    [
        (GR, "Screenshot-2024-07-19-1_png.rf.ab.jpg", "web_image"),
        (GR, "16_webp.rf.ab.jpg", "web_image"),
        (GR, "10_jpg.rf.ab.jpg", "web_image"),
        (GR, "g-test-4_png.rf.ab.jpg", "web_image"),
        (GR, "3_mp4-0004_jpg.rf.ab.jpg", None),  # a video frame, not a web file
        (GR, "1-13-00-14HTD21CP501-1_jpg.rf.ab.jpg", None),
        (GR, "20220610_172152933_iOS_jpg.rf.ab.jpg", "same_gauge_as_dev"),
        (GR, "20220613_135304519_iOS_jpg.rf.ab.jpg", "same_gauge_as_dev"),
        (GR, "20220613_133226396_iOS_jpg.rf.ab.jpg", None),
        (KQ, "meter_a_mp4-12_jpg.rf.ab.jpg", "same_gauge_as_dev"),
        (KQ, "meter_b_MOV-12_jpg.rf.ab.jpg", None),
        (KQ, "man3cropclipscale_mp4-22_jpg.rf.ab.jpg", "aalborg_dev_video"),
        (KQ, "needle-on-pressure-gauge-falling-to-zero_x_mp4-1_jpg.rf.ab.jpg",
         "stock_watermark"),
        (GR, "meter_a_mp4-12_jpg.rf.ab.jpg", None),  # rules are per dataset
    ],
)  # fmt: skip
def test_rules(dataset, name, reason) -> None:
    assert roboflow.rule_reason(dataset, name) == reason


def test_select_applies_rules_hashes_and_scenes(tmp_path, monkeypatch) -> None:
    import cv2
    import numpy as np

    rng = np.random.default_rng(0)
    split = tmp_path / "ws" / "p" / "v1" / "train"
    split.mkdir(parents=True)
    images = {
        "a.jpg": rng.integers(0, 255, (64, 64, 3), np.uint8),
        "Screenshot-1.jpg": rng.integers(0, 255, (64, 64, 3), np.uint8),
        "c.jpg": rng.integers(0, 255, (64, 64, 3), np.uint8),
    }
    images["b.jpg"] = images["a.jpg"].copy()  # a near-copy: same scene as a.jpg
    for name, pixels in images.items():
        cv2.imwrite(str(split / name), pixels)
    coco = {"images": [{"id": i, "file_name": n} for i, n in enumerate(images)]}
    (split / "_annotations.coco.json").write_text(json.dumps(coco))
    reference = {"dev": {"x": roboflow.dedup.dhash(split / "c.jpg")}}
    monkeypatch.setattr(roboflow, "RULES", (("ws/p", r"^Screenshot", "web_image"),))
    rows = roboflow.select(tmp_path, reference, {"ws/p": "1"})
    by_name = {r["file"]: r for r in rows}
    assert by_name["Screenshot-1.jpg"]["reason"] == "web_image"
    assert by_name["c.jpg"]["reason"] == "near_dev"
    assert by_name["a.jpg"]["keep"] == "yes"
    assert by_name["a.jpg"]["scene"] == by_name["b.jpg"]["scene"] != ""
    assert by_name["c.jpg"]["scene"] == ""

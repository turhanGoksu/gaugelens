import io
import json
import urllib.parse
import zipfile

import pytest

from evaluation import aalborg

FILES = [
    {"name": "images/a.jpg", "totalBytes": 1000},
    {"name": "images/b.jpg", "totalBytes": 2000},
    {"name": "labels/a.json", "totalBytes": 10},
    {"name": "README.txt", "totalBytes": 5},
]


def fake_fetch(url: str) -> bytes:
    if "/datasets/list/" in url:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        start = int(query.get("pageToken", ["0"])[0])
        page = FILES[start : start + 3]
        token = str(start + 3) if start + 3 < len(FILES) else ""
        return json.dumps({"datasetFiles": page, "nextPageToken": token}).encode()
    if "/datasets/download/" in url:
        name = urllib.parse.unquote(url.rsplit("/", 1)[1])
        return b"x" * next(f["totalBytes"] for f in FILES if f["name"] == name)
    raise AssertionError(url)


def test_list_files_follows_pages_and_summarizes() -> None:
    files = aalborg.list_files(fake_fetch)
    assert [f["name"] for f in files] == [f["name"] for f in FILES]
    s = aalborg.summary(files)
    assert s["files"] == 4
    assert s["folders"]["images"][0] == 2
    assert s["extensions"][".jpg"] == 2


def test_download_takes_only_matching_files(tmp_path) -> None:
    (tmp_path / "files.json").write_text(json.dumps(aalborg.list_files(fake_fetch)))
    assert aalborg.download(fake_fetch, tmp_path, "labels/*") == 1
    assert (tmp_path / "files" / "labels" / "a.json").exists()
    assert not (tmp_path / "files" / "images").exists()
    # A second run skips what is already there.
    assert aalborg.download(fake_fetch, tmp_path, "labels/*") == 0


def test_download_unwraps_kaggle_zip_and_checks_size(tmp_path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("b.jpg", b"y" * 2000)
    wrapped = buffer.getvalue()
    (tmp_path / "files.json").write_text(json.dumps(aalborg.list_files(fake_fetch)))
    assert aalborg.download(lambda url: wrapped, tmp_path, "images/b.jpg") == 1
    assert (tmp_path / "files" / "images" / "b.jpg").read_bytes() == b"y" * 2000
    with pytest.raises(ValueError, match="listing says 1000"):
        aalborg.download(lambda url: b"short", tmp_path, "images/a.jpg")
    assert not (tmp_path / "files" / "images" / "a.jpg").exists()


def test_credentials_come_from_env_first(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("KAGGLE_USERNAME=file_user\nKAGGLE_KEY='file_key'\n")
    monkeypatch.setenv("KAGGLE_USERNAME", "env_user")
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    assert aalborg.credentials(env_file) == ("env_user", "file_key")


def test_credentials_missing_stops_with_a_hint(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    with pytest.raises(SystemExit, match="env.example"):
        aalborg.credentials(tmp_path / ".env")


def make_video(path, count: int) -> None:
    """Frame i is a flat grey image of level 20 * i."""
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    for i in range(count):
        writer.write(np.full((32, 32, 3), 20 * i, np.uint8))
    writer.release()


def test_read_frames_returns_the_asked_frames(tmp_path) -> None:
    video = tmp_path / "v.avi"
    make_video(video, 8)
    frames = aalborg.read_frames(video, [1, 6])
    assert sorted(frames) == [1, 6]
    assert abs(frames[6].mean() - 120) < 3
    with pytest.raises(ValueError, match="no frames"):
        aalborg.read_frames(video, [9])


def test_write_frames_checks_the_video_hash(tmp_path) -> None:
    video = tmp_path / "files" / aalborg.VIDEOS / "g1.avi"
    make_video(video, 5)
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "labels.csv").write_text("id,gauge,frame\ng1_f00003,g1,3\n")
    good = aalborg.sha256_file(video)
    (labels / "videos.csv").write_text(f"gauge,file,sha256\ng1,g1.avi,{good}\n")
    assert aalborg.write_frames(tmp_path, labels) == 1
    assert (tmp_path / "frames" / "g1_f00003.png").exists()
    (labels / "videos.csv").write_text(f"gauge,file,sha256\ng1,g1.avi,{'0' * 64}\n")
    with pytest.raises(ValueError, match="differs"):
        aalborg.write_frames(tmp_path, labels)

import csv
import json

import pytest

from evaluation.aalborg import sha256_file
from training import data

NAMES = {"box": "gauge", "center": "c", "tip": "t", "min": "lo", "max": "hi"}


def point(name: str, x: float, y: float) -> tuple[str, list[float]]:
    return (name, [x - 2, y - 2, 4, 4])  # a 4x4 box centred on (x, y)


def dial(x0: float, extra: float = 0.0) -> list[tuple[str, list[float]]]:
    """A 100x100 dial box at x0 with its four points."""
    return [
        ("gauge", [x0, 0, 100, 100]),
        point("c", x0 + 50, 50),
        point("t", x0 + 80, 30),
        point("lo", x0 + 20 - extra, 90),
        point("hi", x0 + 80, 90),
    ]


def test_points_go_to_the_dial_that_holds_them() -> None:
    gauges = data._gauges(dial(0) + dial(300), NAMES)
    assert [g["center"] for g in gauges] == [[50, 50], [350, 50]]
    assert gauges[1]["min"] == [320, 90]


def test_a_point_just_outside_a_tight_box_still_counts() -> None:
    # The min mark sits 25 px left of the box edge's 20, within the 10% margin.
    [gauge] = data._gauges(dial(0, extra=25.0), NAMES)
    assert gauge["min"] == [-5, 90]


def test_incomplete_or_doubled_points_drop_the_image() -> None:
    assert data._gauges(dial(0)[:-1], NAMES) is None
    assert data._gauges(dial(0) + [point("t", 60, 40)], NAMES) is None
    assert data._gauges(dial(0) + [point("t", 900, 900)], NAMES) is None


def test_synthetic_labels_become_samples(tmp_path) -> None:
    label = {"image": "images/7.jpg", "seed": 7, "bbox": [1, 2, 3, 4],
             "center": [5, 6], "needle_tip": [7, 8], "min_mark": [9, 10],
             "max_mark": [11, 12]}  # fmt: skip
    (tmp_path / "labels.jsonl").write_text(json.dumps(label) + "\n")
    [sample] = data.synthetic_samples(tmp_path)
    assert sample["group"] == "synthetic:7"
    assert sample["gauges"] == [
        {"box": [1, 2, 3, 4], "center": [5, 6], "tip": [7, 8], "min": [9, 10],
         "max": [11, 12]}
    ]  # fmt: skip


def write_roboflow(tmp_path, monkeypatch, digest=None):
    monkeypatch.setitem(data.ROBOFLOW_CLASSES, "ws/p", NAMES)
    split = tmp_path / "rf" / "ws" / "p" / "v1" / "train"
    split.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        (split / name).write_bytes(name.encode())
    names = ["gauge", "c", "t", "lo", "hi"]
    anns = dial(0)
    coco = {
        "categories": [{"id": i, "name": n} for i, n in enumerate(names)],
        "images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
        "annotations": [
            {"image_id": 1, "category_id": names.index(n), "bbox": b} for n, b in anns
        ]
        + [  # b.jpg has a dial with no points
            {"image_id": 2, "category_id": 0, "bbox": [0, 0, 10, 10]}
        ],
    }
    (split / "_annotations.coco.json").write_text(json.dumps(coco))
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["dataset", "version", "split", "file", "sha256", "keep", "reason", "scene"]
        )
        for name in ("a.jpg", "b.jpg"):
            sha = digest or sha256_file(split / name)
            w.writerow(["ws/p", "1", "train", name, sha, "yes", "", "3"])
    return tmp_path / "rf", manifest


def test_roboflow_samples_check_hashes_and_drop_bad_images(
    tmp_path, monkeypatch
) -> None:
    dest, manifest = write_roboflow(tmp_path, monkeypatch)
    samples, dropped = data.roboflow_samples(dest, manifest)
    assert [s["image"].rsplit("/", 1)[1] for s in samples] == ["a.jpg"]
    assert samples[0]["group"] == "roboflow:3"
    assert dropped == {"points do not match dials": 1}


def test_roboflow_samples_refuse_changed_files(tmp_path, monkeypatch) -> None:
    dest, manifest = write_roboflow(tmp_path, monkeypatch, digest="0" * 64)
    with pytest.raises(ValueError, match="differs"):
        data.roboflow_samples(dest, manifest)


def test_split_keeps_each_group_on_one_side() -> None:
    samples = [{"group": f"g{i % 50}"} for i in range(500)]
    train, val = data.split(samples, 0.2)
    assert {s["group"] for s in train}.isdisjoint({s["group"] for s in val})
    assert 0.05 < len(val) / len(samples) < 0.4
    assert data.split(samples, 0.2) == (train, val)

import json
import math

import pytest

from evaluation import aalborg, real_dev, run_dev
from evaluation.metrics import score
from tests.test_aalborg import make_video


def write(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_measurebench_keeps_eligible_rows_with_their_interval(tmp_path) -> None:
    manifest = [
        {"id": "a", "file": "a.jpg", "image_type": "pressure_gauge",
         "readings": [{"interval": [9, 9]}, {"interval": [2.0, 2.5]}]},
        {"id": "b", "file": "b.jpg", "image_type": "pressure_gauge",
         "readings": [{"interval": [1, 1]}]},
    ]  # fmt: skip
    write(tmp_path / "img" / "manifest.jsonl", "\n".join(map(json.dumps, manifest)))
    write(
        tmp_path / "lab" / "labels.csv",
        "id,eligible,min,max,inferred_end,reading_index\na,yes,0,10,yes,1\nb,no,,,,\n",
    )
    [label] = real_dev.load_measurebench(tmp_path / "img", tmp_path / "lab")
    assert label["interval"] == [2.0, 2.5]
    assert label["fraction"] == pytest.approx(0.225)
    assert label["buckets"] == ["pressure_gauge", "inferred_end"]
    assert label["path"].endswith("a.jpg")


def test_aalborg_off_scale_labels_have_no_value(tmp_path) -> None:
    write(
        tmp_path / "lab" / "labels.csv",
        "id,gauge,min,max,status,value,sure\n"
        "g_f00001,g,0,6,value,1.5,yes\n"
        "g_f00002,g,0,6,below_range,,no\n",
    )
    with pytest.raises(SystemExit, match="evaluation.aalborg frames"):
        real_dev.load_aalborg(tmp_path, tmp_path / "lab")
    for name in ("g_f00001", "g_f00002"):
        write(tmp_path / "frames" / f"{name}.png", "")
    on, below = real_dev.load_aalborg(tmp_path, tmp_path / "lab")
    assert on["fraction"] == pytest.approx(0.25)
    assert on["buckets"] == ["g", "sure: yes"]
    assert below["value"] is None and below["fraction"] == -math.inf
    assert score(below, "below_range", None).expected == "below_range"


def static_setup(tmp_path, rows: str):
    video = tmp_path / "files" / aalborg.VIDEOS / "g.avi"
    make_video(video, 4)
    digest = aalborg.sha256_file(video)
    write(
        tmp_path / "lab" / "videos.csv",
        f"gauge,file,sha256,frames\ng,g.avi,{digest},4\n",
    )
    write(tmp_path / "lab" / "labels.csv", "id,gauge,min,max,status,value\n" + rows)


def test_static_gauge_label_holds_for_every_frame(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(real_dev, "STATIC_GAUGES", ("g",))
    static_setup(
        tmp_path, "g_f00000,g,-1,1.5,value,0.08\ng_f00003,g,-1,1.5,value,0.09\n"
    )
    labels = real_dev.load_aalborg_static(tmp_path, tmp_path / "lab")
    assert [lab["frame"] for lab in labels] == [0, 1, 2, 3]
    assert labels[2]["interval"] == [0.08, 0.09]
    assert score(labels[2], "ok", 0.085).error == 0.0


def test_static_gauge_with_disagreeing_labels_is_refused(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(real_dev, "STATIC_GAUGES", ("g",))
    static_setup(tmp_path, "g_f00000,g,0,6,value,0\ng_f00003,g,0,6,below_range,\n")
    with pytest.raises(ValueError, match="disagree"):
        real_dev.load_aalborg_static(tmp_path, tmp_path / "lab")


def test_batches_keep_one_video_per_job() -> None:
    labels = [{"path": "a"}] * 10 + [{"video": "v", "frame": i} for i in range(70)]
    labels += [{"video": "w", "frame": 0}]
    sizes = [(job[0].get("video"), len(job)) for job in run_dev.batches(labels)]
    assert sizes == [(None, 8), (None, 2), ("v", 64), ("v", 6), ("w", 1)]


def test_stability_measures_the_spread_of_one_gauge() -> None:
    label = {"min_value": 0.0, "max_value": 10.0, "value": 5.0, "fraction": 0.5,
             "group": "g"}  # fmt: skip
    values = [5.0, 5.1, 4.9, 7.0]
    predictions = [{"status": "ok", "value": v} for v in values]
    predictions.append({"status": "unreadable", "value": None})
    labels = [label] * 5
    outcomes = [score(label, p["status"], p["value"]) for p in predictions]
    report = run_dev.stability(labels, outcomes, predictions)["g"]
    assert report["frames"] == 5
    assert report["numbers"] == 4
    assert report["status_right"] == pytest.approx(4 / 5)
    assert report["within_2%"] == pytest.approx(3 / 5)
    assert report["spread_max_min"] == pytest.approx(0.21)

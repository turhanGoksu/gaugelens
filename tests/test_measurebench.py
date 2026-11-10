import json
import urllib.parse

import cv2
import numpy as np

from evaluation import dedup, measurebench

REV = "93c001671055f739bc396c73aa6b2d5e6171abed"


def _image(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    pixels = cv2.resize(rng.integers(0, 256, (12, 12, 3), dtype=np.uint8), (240, 240))
    return cv2.imencode(".jpg", pixels)[1].tobytes()


def _row(qid, image_type, design, kwargs):
    return {
        "question_id": qid,
        "question": "What is the reading of the instrument?",
        "image": {"src": f"https://assets/--/{REV}/--/{qid}/image.jpg?Expires=1"},
        "image_type": image_type,
        "design": design,
        "evaluator_kwargs": json.dumps(kwargs),
        "meta_info": {"source": "https://example.org", "uploader": "Ada"},
    }


ROWS = [
    _row(
        "pressure_gauge_0",
        "pressure_gauge",
        "dial",
        {"interval": [3, 3.5], "units": ["bar"]},
    ),
    _row(
        "pressure_gauge_1",
        "pressure_gauge",
        "dial",
        {"intervals": [[-20, -10], [-0.2, -0.1]], "units": [["kPa"], ["bar"]]},
    ),
    _row(
        "pressure_gauge_2",
        "pressure_gauge",
        "digital",
        {"interval": [1, 1], "units": ["bar"]},
    ),
    _row("ammeter_0", "ammeter", "dial", {"interval": [9, 9], "units": ["A"]}),
    _row(
        "sphygmomanometer_0",
        "sphygmomanometer",
        "dial",
        {"interval": [0, 2], "units": ["mmHg"]},
    ),
]


def fake_fetch(url: str) -> bytes:
    if url.startswith(measurebench.ROWS):
        offset = int(
            urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["offset"][0]
        )
        page = ROWS[offset : offset + 100]
        return json.dumps(
            {"rows": [{"row": r} for r in page], "num_rows_total": len(ROWS)}
        ).encode()
    qid = url.split("/--/")[2].split("/")[0]
    return _image(sum(map(ord, qid)))


def test_readings_handle_one_and_two_scales() -> None:
    assert measurebench.readings(ROWS[0]) == [
        {"interval": [3.0, 3.5], "units": ["bar"]}
    ]
    assert measurebench.readings(ROWS[1]) == [
        {"interval": [-20.0, -10.0], "units": ["kPa"]},
        {"interval": [-0.2, -0.1], "units": ["bar"]},
    ]


def test_collect_keeps_dial_pressure_gauges(tmp_path) -> None:
    counts = measurebench.collect(tmp_path, fetch=fake_fetch)
    rows = [
        json.loads(line)
        for line in (tmp_path / "manifest.jsonl").read_text().splitlines()
    ]
    assert [r["id"] for r in rows] == [
        "pressure_gauge_0",
        "pressure_gauge_1",
        "sphygmomanometer_0",
    ]
    assert counts["kept"] == 3
    assert all(r["revision"] == REV for r in rows)
    assert all((tmp_path / r["file"]).exists() for r in rows)


def test_collect_drops_photos_that_are_test_candidates(tmp_path) -> None:
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    # The test candidate is a resized copy of pressure_gauge_0.
    same = cv2.imdecode(
        np.frombuffer(fake_fetch(f"x/--/{REV}/--/pressure_gauge_0/i.jpg"), np.uint8), 1
    )
    cv2.imwrite(str(test_dir / "commons_1.jpg"), cv2.resize(same, (500, 500)))
    (test_dir / "manifest.jsonl").write_text(
        json.dumps({"id": "commons_1", "file": "commons_1.jpg"})
    )

    counts = measurebench.collect(
        tmp_path / "dev", fetch=fake_fetch, test_images=test_dir
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "dev" / "manifest.jsonl").read_text().splitlines()
    ]
    assert "pressure_gauge_0" not in [r["id"] for r in rows]
    assert counts["dropped: also a test candidate"] == 1


def test_dhash_separates_copies_from_other_photos(tmp_path) -> None:
    for name, seed in (("a", 1), ("b", 2)):
        (tmp_path / f"{name}.jpg").write_bytes(_image(seed))
    a = cv2.imread(str(tmp_path / "a.jpg"))
    cv2.imwrite(str(tmp_path / "a_small.png"), cv2.resize(a, (120, 120)))
    ha, hb, hs = (dedup.dhash(tmp_path / f) for f in ("a.jpg", "b.jpg", "a_small.png"))
    assert dedup.distance(ha, hs) <= dedup.MAX_BITS
    assert dedup.distance(ha, hb) > dedup.MAX_BITS

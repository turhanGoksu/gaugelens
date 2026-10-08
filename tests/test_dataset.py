import json

import cv2
import numpy as np
import pytest

from gaugelens.synth import BackgroundPool
from gaugelens.synth.__main__ import main
from gaugelens.synth.backgrounds import background_split
from gaugelens.synth.dataset import DatasetConfig, write_dataset


def test_background_split_is_stable_and_about_one_fifth_dev() -> None:
    names = [f"texture_{i}.jpg" for i in range(2000)]
    splits = [background_split(n) for n in names]
    assert splits == [background_split(n) for n in names]
    assert 0.17 < splits.count("dev") / len(names) < 0.23


def _photo_dir(tmp_path, count: int = 30):
    rng = np.random.default_rng(0)
    for i in range(count):
        image = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"wall_{i}.jpg"), image)
    return tmp_path


def test_train_and_dev_pools_share_no_photo(tmp_path) -> None:
    folder = _photo_dir(tmp_path)
    train = {p.name for p, _ in BackgroundPool.from_dir(folder, split="train").photos}
    dev = {p.name for p, _ in BackgroundPool.from_dir(folder, split="dev").photos}
    assert train and dev
    assert not train & dev
    assert len(train | dev) == 30


def test_train_and_dev_seeds_never_overlap() -> None:
    train = DatasetConfig(split="train", count=10_000_000).seeds
    dev = DatasetConfig(split="dev", count=10_000_000).seeds
    assert train.stop <= dev.start


def _read(folder):
    lines = (folder / "labels.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_parallel_workers_write_the_same_dataset(tmp_path) -> None:
    config = DatasetConfig(split="dev", count=6, width=160, height=160)
    one = write_dataset(tmp_path / "one", config, workers=1)
    two = write_dataset(tmp_path / "two", config, workers=2)
    assert _read(one) == _read(two)
    for row in _read(one):
        assert (one / row["image"]).read_bytes() == (two / row["image"]).read_bytes()


def test_dataset_records_how_it_was_made(tmp_path) -> None:
    photos = _photo_dir(tmp_path)
    config = DatasetConfig(
        split="dev", count=4, width=128, height=128, backgrounds=str(photos)
    )
    out = write_dataset(tmp_path / "ds", config)
    rows = _read(out)
    assert [r["seed"] for r in rows] == list(range(100_000_000, 100_000_004))
    assert all((out / r["image"]).exists() for r in rows)
    metadata = json.loads((out / "dataset.json").read_text())
    assert metadata["seeds"] == [100_000_000, 100_000_004]
    assert metadata["background_photos"]
    assert all(background_split(n) == "dev" for n in metadata["background_photos"])


def test_refuses_to_overwrite_a_dataset(tmp_path) -> None:
    config = DatasetConfig(split="train", count=1, width=96, height=96)
    write_dataset(tmp_path, config)
    with pytest.raises(FileExistsError):
        write_dataset(tmp_path, config)


def test_cli_generates_a_dataset(tmp_path, capsys) -> None:
    out = tmp_path / "cli"
    args = ["generate", str(out), "--split", "train", "--count", "3", "--size", "128"]
    assert main(args) == 0
    assert len(_read(out)) == 3
    assert "3 train scenes" in capsys.readouterr().out

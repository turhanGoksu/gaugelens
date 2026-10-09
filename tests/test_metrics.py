import pytest

from evaluation.metrics import expected_position, score, summarize


def label(value: float, lo: float = 0.0, hi: float = 10.0) -> dict:
    return {
        "value": value,
        "min_value": lo,
        "max_value": hi,
        "fraction": (value - lo) / (hi - lo),
    }


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [
        (-0.05, "below_range"),
        (-0.005, "in_range"),
        (0.5, "in_range"),
        (1.005, "in_range"),
        (1.05, "above_range"),
    ],
)
def test_expected_position_uses_the_edge_margin(fraction, expected) -> None:
    assert expected_position(fraction) == expected


def test_error_is_a_fraction_of_full_scale() -> None:
    assert score(label(5.0), "ok", 5.3).error == pytest.approx(0.03)


def test_needle_within_the_edge_margin_is_scored_against_the_end_mark() -> None:
    # 10.05 bar is 0.5% FS past the max mark: the right answer is 10.
    assert score(label(10.05), "ok", 10.0).error == pytest.approx(0.0)


def test_abstentions_have_no_error() -> None:
    assert score(label(5.0), "unreadable", None).error is None


def test_summary_separates_right_wrong_and_abstained() -> None:
    outcomes = [
        score(label(5.0), "ok", 5.05),  # 0.5% FS: right
        score(label(5.0), "ok", 6.0),  # 10% FS: confidently wrong
        score(label(5.0), "low_confidence", 6.0),  # wrong, but flagged
        score(label(5.0), "unreadable", None),
        score(label(11.0), "above_range", None),  # off the scale, status right
        score(label(11.0), "ok", 10.0),  # off the scale, given as ok
    ]
    summary = summarize(outcomes)
    assert summary["in_range_images"] == 4
    assert summary["within_2%"] == pytest.approx(1 / 4)
    assert summary["confidently_wrong_2%"] == pytest.approx(1 / 4)
    assert summary["abstained"] == pytest.approx(1 / 4)
    assert summary["out_of_range_correct"] == pytest.approx(1 / 2)
    assert summary["out_of_range_given_ok"] == pytest.approx(1 / 2)

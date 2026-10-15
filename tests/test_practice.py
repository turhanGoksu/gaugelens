import json
import math

from evaluation.practice import CASES, write_practice


def test_practice_set_has_an_answer_for_every_image(tmp_path) -> None:
    page = write_practice(tmp_path, backgrounds=None)
    rows = [
        json.loads(line)
        for line in (tmp_path / "manifest.jsonl").read_text().splitlines()
    ]
    assert len(rows) == len(CASES)
    assert all((tmp_path / row["file"]).exists() for row in rows)
    assert '"practice": true' in page.read_text()
    for row in rows:
        answer = row["answer"]
        assert answer["explanation"]
        if answer["eligible"] == "no":
            assert answer["exclusion_reason"]
        elif answer["status"] == "value":
            # Readings sit on the 1/5-division grid that the guideline asks for.
            fifth = answer["division"] / 5
            steps = (answer["exact"] - answer["min"]) / fifth
            assert math.isclose(steps, round(steps), abs_tol=1e-6)
            assert answer["min"] <= answer["value"] <= answer["max"]

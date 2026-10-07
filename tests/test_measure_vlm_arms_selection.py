"""`measure_vlm_arms --worst-case`: the peak must come from the tail (#163).

The script measured the head of the file (96 of 9,441 samples) and said that the
peak was a lower bound. #163 asks for the number that decides whether a long run
dies, which lives in the distribution's tail. These cover the selection only —
the run itself needs a GPU and an engine venv.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "measure_vlm_arms", REPO / "scripts" / "measure_vlm_arms.py")
mva = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mva
_SPEC.loader.exec_module(mva)


def jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def row(text: str, width: int = 100) -> dict:
    return {"image": "data/pages/p.jpg", "text": text, "source_type": "line",
            "bbox": [0, 0, width, 40], "page": "data/pages/p.xml"}


def test_the_head_is_still_the_head(tmp_path):
    """The default must not move: #137's figures are quoted in the docstring."""
    src = jsonl(tmp_path / "train.jsonl", [row("a"), row("b" * 9000), row("c")])
    dest = tmp_path / "out.jsonl"
    meta = mva.take_lines(src, dest, 2, worst_case=False)
    assert meta == {"selection": "head", "written": 2}
    texts = [json.loads(ln)["text"] for ln in dest.read_text().splitlines()]
    assert texts == ["a", "b" * 9000]


def test_the_worst_case_takes_the_expensive_tail(tmp_path):
    """Nine thousand characters of logits, not the first line of the file."""
    rows = [row("x") for _ in range(20)]
    rows.insert(7, row("a" * 9000))
    src = jsonl(tmp_path / "train.jsonl", rows)
    dest = tmp_path / "out.jsonl"
    meta = mva.take_lines(src, dest, 1, worst_case=True)
    assert meta["selection"] == "worst-case"
    assert meta["longest_chars"] == 9000
    assert meta["considered"] == 21
    assert json.loads(dest.read_text().splitlines()[0])["text"] == "a" * 9000


def test_the_report_can_tell_a_floor_from_a_ceiling(tmp_path):
    """The point of the field: a head peak may not be read as a limit."""
    src = jsonl(tmp_path / "train.jsonl", [row("a" * 10, width=4000), row("b" * 500)])
    head = mva.take_lines(src, tmp_path / "h.jsonl", 1, worst_case=False)
    tail = mva.take_lines(src, tmp_path / "t.jsonl", 2, worst_case=True)
    assert head["selection"] != tail["selection"]
    assert "considered" not in head, "a head selection cannot claim coverage"
    assert tail["considered"] == 2
    assert tail["widest_pixels"] == 4000 * 40


def test_the_written_samples_stay_loadable(tmp_path):
    """Round-trip: the engine reads these lines back with the same reader."""
    from atr_training.vlm_dataset import read_jsonl

    src = jsonl(tmp_path / "train.jsonl", [row("a" * 30), row("b" * 60)])
    dest = tmp_path / "out.jsonl"
    mva.take_lines(src, dest, 2, worst_case=True)
    back = list(read_jsonl(dest))
    assert sorted(len(s.text) for s in back) == [30, 60]
    assert all(s.image == "data/pages/p.jpg" for s in back), \
        "image paths must stay relative to the job dir, or the run dies on data/data/"


def test_asking_for_more_than_the_file_holds_does_not_raise(tmp_path):
    """The head path raises StopIteration here; the worst-case path must not."""
    src = jsonl(tmp_path / "train.jsonl", [row("a")])
    meta = mva.take_lines(src, tmp_path / "out.jsonl", 96, worst_case=True)
    assert meta["written"] == 1
    assert meta["considered"] == 1

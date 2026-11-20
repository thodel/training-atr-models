"""The scoring half of scripts/measure_party.py (#432).

party, kraken and torch are not in the repo venv; the reader is imported lazily
for that reason, and what is worth testing — how a page's lines are paired and
how a level is summarised — needs none of them.
"""

from __future__ import annotations

import json

import pytest

from scripts.measure_party import measure_page_set, page_items, summarise


def test_lines_pair_by_position_and_a_missing_one_is_empty():
    lines, page = page_items("p.xml", ["a b", "c"], ["a b"], 1.5)
    assert [(i["ref"], i["hyp"]) for i in lines] == [("a b", "a b"), ("c", "")]
    assert page["ref"] == "a b\nc" and page["hyp"] == "a b" and page["lines"] == 2


def test_a_level_reports_the_failure_shapes():
    items = [{"ref": "abcdef", "hyp": "abcdef", "sec": 1.0},
             {"ref": "abcdef", "hyp": "a", "sec": 1.0},
             {"ref": "ab", "hyp": "ab ab ab", "sec": 1.0, "error": None},
             {"ref": "abc", "hyp": "", "sec": 0.0, "error": "OutOfMemoryError: …"}]
    level = summarise("gt-line", items)
    assert level["n"] == 4 and level["ref_chars"] == 17
    assert level["collapsed<1/3"] == 2 and level["runaway>1.5x"] == 1 and level["failed"] == 1
    assert 0 < level["cer"] < 1


def test_line_breaks_are_layout_not_errors():
    level = summarise("gt-page", [{"ref": "eins\nzwei", "hyp": "eins zwei", "sec": 0}])
    assert level["cer"] == 0.0 and level["length_ratio"] == 1.0


class _Fake:
    """The reader's two calls, answering from a script instead of a model."""

    def gt(self, xml, image):
        return ["Erste Zeile", "zweite"], ["Erste Zeile", "zweyte"]

    def blla(self, image):
        return ["Erste Zeile zweite"]


def _page_set(tmp_path, md5=None):
    import hashlib

    rows = [{"image": "data/pages/a.jpg", "text": "Erste Zeile\nzweite", "page": "data/pages/a.xml"}]
    draw = tmp_path / "pages.jsonl"
    draw.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "benchmark.json").write_text(json.dumps({
        "set": "s", "side": "val", "draw": "pages.jsonl",
        "draw_md5": md5 or hashlib.md5(draw.read_bytes()).hexdigest()}))
    return tmp_path


def test_a_page_set_is_read_in_both_modes(tmp_path):
    raw = (tmp_path / "raw.jsonl").open("w")
    report = measure_page_set(_Fake(), _page_set(tmp_path), ("gt", "blla"), None, raw)
    levels = {lv["level"]: lv for lv in report["levels"]}
    assert set(levels) == {"gt-line", "gt-page", "blla-page"}
    assert levels["gt-line"]["n"] == 2 and levels["gt-page"]["n"] == 1
    assert levels["blla-page"]["cer"] == 0.0
    assert report["set"]["set"] == "s"


def test_an_edited_draw_is_refused(tmp_path):
    from eval_granularity import PageSetError

    with pytest.raises(PageSetError):
        measure_page_set(_Fake(), _page_set(tmp_path, md5="0" * 32), ("gt",), None,
                         (tmp_path / "raw.jsonl").open("w"))

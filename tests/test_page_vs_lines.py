"""#74: a page arm against line arms, on the same pages and one reference.

The measurement joins a line arm's outputs per page and scores them against the
page draw's reference — so the joining order, the matching of lines to pages, and
the sign of the paired difference are what decide the answer. Each is pinned.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lop = _load("lines_of_pages", "scripts/lines_of_pages.py")
cmp_ = _load("compare_page_and_lines", "scripts/compare_page_and_lines.py")
rescore = _load("rescore_on_shared_draw", "ubelix/rescore_on_shared_draw.py")

PAGES = [{"image": "data/pages/000001_111_0001_9.jpg", "text": "erste zeile\nzweite zeile"},
         {"image": "data/pages/000002_222_0001_8.jpg", "text": "dritte"}]
LINES = [{"image": "data/crops/val/0000000.jpg", "text": "erste zeile",
          "page": "data/pages/000001_111_0001_9.xml"},
         {"image": "data/crops/val/0000001.jpg", "text": "zweite zeile",
          "page": "data/pages/000001_111_0001_9.xml"},
         {"image": "data/crops/val/0000002.jpg", "text": "nicht gezogen",
          "page": "data/pages/000099_999_0001_1.xml"},
         {"image": "data/crops/val/0000003.jpg", "text": "dritte",
          "page": "data/pages/000002_222_0001_8.xml"}]


# ── which lines ─────────────────────────────────────────────────────────────

def test_only_the_lines_of_the_drawn_pages_are_kept_in_corpus_order():
    kept, per_page = lop.lines_of_pages(PAGES, LINES)

    assert [r["text"] for r in kept] == ["erste zeile", "zweite zeile", "dritte"]
    assert per_page == {"000001_111_0001_9": 2, "000002_222_0001_8": 1}


def test_a_drawn_page_without_lines_is_refused(tmp_path):
    """Then the corpora do not hold the same pages, and the comparison would
    score the line arm on fewer pages than the page arm without saying so."""
    pages = tmp_path / "p.jsonl"
    pages.write_text("".join(json.dumps(r) + "\n" for r in PAGES + [
        {"image": "data/pages/000003_333_0001_7.jpg", "text": "x"}]))
    lines = tmp_path / "l.jsonl"
    lines.write_text("".join(json.dumps(r) + "\n" for r in LINES))

    assert lop.main(["--pages", str(pages), "--lines", str(lines),
                     "--out", str(tmp_path / "o.jsonl")]) == 1
    assert not (tmp_path / "o.jsonl").exists()


# ── joining and scoring ─────────────────────────────────────────────────────

def test_a_line_arm_page_is_its_lines_joined_in_order():
    kept, _ = lop.lines_of_pages(PAGES, LINES)
    raw = [{"image": "data/crops/val/0000001.jpg", "raw": "zweite zeile "},
           {"image": "data/crops/val/0000000.jpg", "raw": " erste zeile"},
           {"image": "data/crops/val/0000003.jpg", "raw": "dritte"}]

    pages = cmp_.page_outputs_of_lines(kept, raw)

    assert pages["000001_111_0001_9"] == "erste zeile\nzweite zeile", \
        "the raw report's order must not decide the page text"


def test_a_line_without_an_output_is_an_error_not_a_blank():
    kept, _ = lop.lines_of_pages(PAGES, LINES)

    with pytest.raises(ValueError, match="no output"):
        cmp_.page_outputs_of_lines(kept, [{"image": "data/crops/val/0000000.jpg", "raw": "x"}])


def test_perfect_lines_score_zero_against_the_page_reference():
    kept, _ = lop.lines_of_pages(PAGES, LINES)
    raw = [{"image": r["image"], "raw": r["text"]} for r in kept]

    summary = cmp_.summarise(cmp_.page_outputs_of_lines(kept, raw), PAGES)

    assert summary["cer_pooled"] == 0.0
    assert summary["length_ratio_median"] == 1.0


def test_coverage_says_how_much_page_text_the_lines_hold():
    kept, _ = lop.lines_of_pages(PAGES, LINES)

    cover = cmp_.coverage(kept[:1] + kept[2:], PAGES)

    assert cover["000002_222_0001_8"] == 1.0
    assert cover["000001_111_0001_9"] == pytest.approx(len("erste zeile") /
                                                       len("erste zeilezweite zeile"))


# ── the paired difference ───────────────────────────────────────────────────

def test_the_paired_difference_is_line_minus_page_with_an_interval_around_it():
    line = {"a": {"cer": 0.10}, "b": {"cer": 0.20}, "c": {"cer": 0.30}}
    page = {"a": {"cer": 0.30}, "b": {"cer": 0.30}, "c": {"cer": 0.30}}

    result = cmp_.paired(line, page)

    assert result["mean_diff"] == pytest.approx(-0.10)
    assert result["ci95"][0] <= result["mean_diff"] <= result["ci95"][1]
    assert (result["a_better"], result["b_better"], result["ties"]) == (2, 0, 1)


def test_the_interval_is_reproducible():
    line = {k: {"cer": i / 10} for i, k in enumerate("abcdefgh")}
    page = {k: {"cer": 0.3} for k in "abcdefgh"}

    assert cmp_.paired(line, page) == cmp_.paired(line, page)


# ── the rescore entry point ─────────────────────────────────────────────────

@pytest.mark.parametrize("argv", [
    ["--job", "j"],                                                  # neither
    ["--job", "j", "--draw-from", "d", "--draw-file", "f.jsonl"],   # both
    ["--job", "j", "--draw-file", "f.jsonl", "--label", "x"],       # no data root
    ["--job", "j", "--draw-file", "f.jsonl", "--data-root", "r"],   # no label
])
def test_a_draw_file_comes_with_its_root_and_a_name_and_never_with_draw_from(argv):
    """Refused before the job store is read, so a typo costs no allocation."""
    assert rescore.main(argv) == 2


def test_a_draw_file_is_scored_whole_not_capped_at_the_runs_eval_samples(tmp_path):
    """19709423 scored 200 of 3,890 lines: the run's eval_samples capped a draw
    that already was the selection."""
    from atr_training.contracts import VlmTrainParams
    from atr_training.vlm_cmd import evaluate_cmd

    draw = tmp_path / "lines.jsonl"
    draw.write_text("".join(json.dumps({"image": f"data/crops/val/{i:07d}.jpg",
                                        "text": "x"}) + "\n" for i in range(3890)))
    params = rescore.params_for_draw_file(VlmTrainParams(eval_samples=200), draw)

    cmd = evaluate_cmd("python", params=params, base_model="Qwen/Qwen3.5-4B",
                       adapter_dir=tmp_path, val_jsonl=draw, data_root=tmp_path,
                       report=tmp_path / "r.json")
    assert cmd[cmd.index("--max-samples") + 1] == "3890"

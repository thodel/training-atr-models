"""scripts/stratified_eval_set.py — an evaluation subset that covers every source.

Pinned because the defect it works around was invisible: the test stage scores
the head of val.jsonl, and for the German corpus the head is one source.

Moved from serving-atr-inference with the script (#6), **and trimmed**. Four of
its seven tests covered `eval_subset` itself, against the arithmetic this
repository has since replaced: until #28 a dataset's span started at the pages
*written* so far, so the ranges overlapped by the previous dataset's skipped
pages and the reader compensated by dropping them. Writer and reader now consume
the same indices, and `tests/test_eval_subset.py` asserts that. Carrying the old
expectations across would have been a regression wearing the clothes of
coverage. What is left is what only this script does: the draw and the CLI.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stratified_eval_set.py"
spec = importlib.util.spec_from_file_location("stratified_eval_set", SCRIPT)
sev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sev)

# Two datasets. A consumed 13 indices (10 written, 3 skipped), so B starts at 13
# — the post-#28 arithmetic, which is what `atr_training.eval_subset` implements.
COUNTS = [{"hf_repo": "dh-unibe/image-text_a", "pages_written": 10, "pages_skipped": 3},
          {"hf_repo": "dh-unibe/image-text_b", "pages_written": 10, "pages_skipped": 0}]


def img(index: int, doc: str) -> str:
    return f"data/pages/{index:06d}_{doc}_0001_999.jpg"


def test_every_source_gets_its_share_even_when_the_file_starts_with_one(tmp_path):
    """The whole point: val.jsonl's head is one source."""
    rows = [{"image": img(i, f"a{i}"), "text": "x"} for i in range(8)] + \
           [{"image": img(13 + i, f"b{i}"), "text": "y"} for i in range(5)]
    owner = sev.attribute([r["image"] for r in rows], sev.source_spans(COUNTS))
    picked = sev.stratify(rows, owner, per_source=3, seed=1)
    sources = [owner[r["image"]] for r in picked]
    assert sources.count("a") == 3 and sources.count("b") == 3
    assert rows[:6] != picked                      # not simply the head


def test_the_draw_is_reproducible():
    rows = [{"image": img(i, f"a{i}"), "text": "x"} for i in range(9)]
    owner = sev.attribute([r["image"] for r in rows], sev.source_spans(COUNTS))
    assert sev.stratify(rows, owner, 4, seed=42) == sev.stratify(rows, owner, 4, seed=42)


def test_the_output_records_each_page_s_source(tmp_path):
    job = tmp_path / "job"
    (job / "data").mkdir(parents=True)
    (job / "job.json").write_text(json.dumps({"progress": {"dataset_counts": COUNTS}}))
    val = [{"image": img(1, "a1"), "text": "x"}, {"image": img(14, "b1"), "text": "y"}]
    (job / "data" / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in val))
    (job / "data" / "train.jsonl").write_text("")
    out = tmp_path / "eval.jsonl"
    assert sev.main([str(job), "--per-source", "5", "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert {r["source"] for r in rows} == {"a", "b"}

"""scripts/materialize_benchmark.py — the Federal Council lines as an evaluation set (#875).

The directory the scoring jobs read (``DRAW_FILE=$E/data/val.jsonl DATA_ROOT=$E``)
was hand-built once and lost to the scratch purge. The script that replaces it
has to produce the same shape — image paths that resolve under ``--out``, one row
per line, the reference text of each line — and must not overwrite a set other
reports were already scored on. No hub: a fake source yields rows the way the
real ``Image(decode=False)`` column does.
"""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "materialize_benchmark.py"
spec = importlib.util.spec_from_file_location("materialize_benchmark", SCRIPT)
mb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mb)

LINE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="x.jpg" imageWidth="600" imageHeight="80">
    <TextRegion id="r1">
      <TextLine id="l1"><Coords points="8,10 590,10 590,70 8,70"/>
        <Baseline points="8,60 590,60"/>
        <TextEquiv><Unicode>{text}</Unicode></TextEquiv></TextLine>
    </TextRegion>
  </Page>
</PcGts>
"""
TEXTS = ["Der Bundesrat beschliesst", "in Anwesenheit des Kanzlers",
         "wird dem Departement des Innern", ""]          # the last one: no transcription


def _jpeg(width: int = 600, height: int = 80) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="JPEG")
    return buf.getvalue()


class FakeSource:
    def __init__(self, texts=TEXTS):
        self.texts = texts
        self.calls: list[tuple[str, list[str], str | None]] = []

    def stream(self, hf_repo, data_files, revision=None):
        self.calls.append((hf_repo, list(data_files), revision))
        for i, text in enumerate(self.texts):
            yield {"image": {"bytes": _jpeg(), "path": f"{i}.jpg"},
                   "xml_content": LINE_XML.format(text=text),
                   "filename": f"{i}_page_9{i}_line_1_docId_5151{i % 2}.jpg",
                   "project_name": mb.FEDERAL_MINUTES_PROJECT}


def test_the_set_has_the_shape_the_scoring_jobs_read(tmp_path):
    out = tmp_path / "federal-minutes"
    source = FakeSource()
    record = mb.materialize_benchmark(source, out, hf_repo=mb.FEDERAL_MINUTES_REPO,
                                      project=mb.FEDERAL_MINUTES_PROJECT, revision="a" * 40)

    # it read the one project, at the given revision
    assert source.calls == [(mb.FEDERAL_MINUTES_REPO,
                             ["data/train/TEST_federal_minutes/*.parquet"], "a" * 40)]
    rows = [json.loads(line) for line in (out / "data" / "val.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert [r["text"] for r in rows] == TEXTS[:3]          # the empty page is skipped
    for r in rows:
        # DATA_ROOT=$E: every image resolves under --out, and it is a crop, not the page
        assert not Path(r["image"]).is_absolute()
        assert r["image"].startswith("data/crops/benchmark_TEST_federal_minutes/")
        assert (out / r["image"]).is_file()
        assert "bbox" not in r or r["bbox"] is None
        assert r["page"].startswith("data/benchmarks/TEST_federal_minutes/")
    assert record["lines"] == 3 and record["pages_skipped"] == 1
    assert record["chars"] == sum(len(t) for t in TEXTS[:3])
    assert record["documents"] == 2                        # docId_51510, docId_51511
    assert record["revision"] == "a" * 40
    on_disk = json.loads((out / "evalset.json").read_text(encoding="utf-8"))
    assert on_disk["val_jsonl_md5"] == record["val_jsonl_md5"] and len(record["val_jsonl_md5"]) == 32
    assert (out / "data" / "pages_benchmark_TEST_federal_minutes.lst").is_file()


def test_the_rows_read_back_as_samples_the_evaluators_take(tmp_path):
    """evaluate_qlora and evaluate_trocr read val.jsonl through read_jsonl."""
    from atr_training.vlm_dataset import read_jsonl

    out = tmp_path / "e"
    mb.materialize_benchmark(FakeSource(), out, hf_repo="r", project="P")
    samples = list(read_jsonl(out / "data" / "val.jsonl"))
    assert len(samples) == 3 and all(s.bbox is None for s in samples)
    with Image.open(out / samples[0].image) as im:
        assert im.width <= 600 and im.height <= 80


def test_an_existing_set_is_never_overwritten(tmp_path):
    out = tmp_path / "e"
    mb.materialize_benchmark(FakeSource(), out, hf_repo="r", project="P")
    before = (out / "data" / "val.jsonl").read_bytes()
    with pytest.raises(SystemExit, match="move the old set aside"):
        mb.materialize_benchmark(FakeSource(["anders"]), out, hf_repo="r", project="P")
    assert (out / "data" / "val.jsonl").read_bytes() == before


def test_the_cli_fails_when_the_line_count_is_not_the_published_one(tmp_path, capsys):
    rc = mb.main(["--out", str(tmp_path / "a"), "--expect-lines", "3"], source=FakeSource())
    assert rc == 0
    rc = mb.main(["--out", str(tmp_path / "b"), "--expect-lines", "2751"], source=FakeSource())
    assert rc == 1
    assert "3 lines, expected 2751" in capsys.readouterr().err


def test_a_commit_is_taken_as_given_without_asking_the_hub():
    assert mb.resolve_revision("any/repo", "0123456789abcdef0123456789abcdef01234567") == \
        "0123456789abcdef0123456789abcdef01234567"

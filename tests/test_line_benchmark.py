"""scripts/build_line_benchmark.py: a frozen page set's lines, as a line arm's draw.

The lines must be the ones ``eval_granularity.py --page-set`` reads at its line
level, cut and written the way a run's compile writes them, and every row must
name its page — that is what ``rescore_on_shared_draw.py --heldout`` reads the
documents from. The page set itself is never touched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from atr_training.vlm_dataset import page_sample, read_jsonl, write_jsonl
from scripts.eval_granularity import page_lines

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lines_bench = _load("build_line_benchmark", "scripts/build_line_benchmark.py")
rescore = _load("rescore_on_shared_draw", "ubelix/rescore_on_shared_draw.py")

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
#: ``<pool index>_<docId>_<folio>_<image>`` — field 1 is the document.
PAGES = {
    "000001_1001_0001_1": [((100, 100, 500, 140), "Item ontfaen van Janne"),
                           ((100, 160, 520, 200), "anno 1431 / vigilia")],
    "000002_1001_0002_2": [((90, 300, 700, 350), "zweite seite")],
    "000003_1002_0001_3": [((0, 0, 0, 0), "kein lesbarer ausschnitt")],   # no line survives
}


def _xml(lines) -> str:
    body = "".join(
        f'<TextLine id="l{i}"><Coords points="{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"/>'
        f"<TextEquiv><Unicode>{text}</Unicode></TextEquiv></TextLine>"
        for i, ((x0, y0, x1, y1), text) in enumerate(lines))
    return (f'<PcGts xmlns="{NS}"><Page imageFilename="p.jpg" imageWidth="1000" '
            f'imageHeight="800"><TextRegion id="r"><Coords points="0,0 1000,0 1000,800 0,800"/>'
            f"{body}</TextRegion></Page></PcGts>")


@pytest.fixture
def page_set(tmp_path):
    """A set laid out as scripts/build_page_benchmark.py writes one."""
    root = tmp_path / "eval_sets" / "toy-v1" / "vlm-pages-test"
    pages = root / "data" / "pages"
    pages.mkdir(parents=True)
    for stem, lines in PAGES.items():
        (pages / f"{stem}.xml").write_text(_xml(lines), encoding="utf-8")
        Image.new("RGB", (1000, 800), "white").save(pages / f"{stem}.jpg")
    samples = [page_sample(pages / f"{stem}.xml", root=root) for stem in PAGES]
    write_jsonl(root / "pages.jsonl", samples)
    record = {"set": "toy-v1", "side": "test", "documents": 2, "pages": 3,
              "page_digest": "0123456789abcdef", "samples": 3,
              "reference_chars": sum(len(s.text) for s in samples), "draw": "pages.jsonl",
              "draw_md5": hashlib.md5((root / "pages.jsonl").read_bytes()).hexdigest()}
    (root / "benchmark.json").write_text(json.dumps(record), encoding="utf-8")
    return root


def _snapshot(directory: Path) -> dict[str, str]:
    return {str(p.relative_to(directory)): hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}


# ── the draw ────────────────────────────────────────────────────────────────
def test_the_lines_are_the_ones_eval_granularity_reads(page_set, tmp_path):
    out = tmp_path / "eval_sets" / "toy-v1" / "vlm-lines-test"
    lines_bench.build(page_set=page_set, out=out)

    rows = list(read_jsonl(out / "lines.jsonl"))
    expected = [r for stem in PAGES
                for r in page_lines(page_set, f"data/pages/{stem}.xml", (1000, 800))]
    assert [r.text for r in rows] == [r["text"] for r in expected]
    assert [r.page for r in rows] == [r["page"] for r in expected]
    assert {r.source_type for r in rows} == {"line"}


def test_every_crop_is_on_disk_relative_to_out_and_the_size_of_its_box(page_set, tmp_path):
    out = tmp_path / "vlm-lines-test"
    lines_bench.build(page_set=page_set, out=out)

    rows = list(read_jsonl(out / "lines.jsonl"))
    boxes = [r["bbox"] for stem in PAGES
             for r in page_lines(page_set, f"data/pages/{stem}.xml", (1000, 800))]
    for row, (left, top, right, bottom) in zip(rows, boxes, strict=True):
        assert not Path(row.image).is_absolute() and row.bbox is None
        with Image.open(out / row.image) as crop:
            assert crop.size == (right - left, bottom - top)


def test_the_record_says_what_was_cut_from_what(page_set, tmp_path):
    out = tmp_path / "vlm-lines-test"
    record = lines_bench.build(page_set=page_set, out=out)

    saved = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))
    assert saved == record
    assert (record["set"], record["side"], record["level"]) == ("toy-v1", "test", "line")
    assert record["page_set"]["path"] == str(page_set)
    assert record["page_set"]["draw_md5"] == json.loads(
        (page_set / "benchmark.json").read_text())["draw_md5"]
    assert (record["pages"], record["pages_with_lines"], record["samples"]) == (3, 2, 3)
    # Named, not lost: a page with text whose only line no crop survives.
    assert record["pages_without_lines"] == ["data/pages/000003_1002_0001_3.xml"]
    assert record["reference_chars"] == len("Item ontfaen van Janne"
                                            "anno 1431 / vigilia" "zweite seite")
    assert record["draw_md5"] == hashlib.md5((out / "lines.jsonl").read_bytes()).hexdigest()


def test_the_page_set_is_not_touched(page_set, tmp_path):
    before = _snapshot(page_set)
    lines_bench.build(page_set=page_set, out=tmp_path / "vlm-lines-test")
    assert _snapshot(page_set) == before


# ── refusals ────────────────────────────────────────────────────────────────
def test_an_edited_page_set_is_refused(page_set, tmp_path):
    with open(page_set / "pages.jsonl", "a", encoding="utf-8") as fh:
        fh.write("\n")
    with pytest.raises(lines_bench.LineBenchmarkError, match="not the frozen draw"):
        lines_bench.build(page_set=page_set, out=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_an_out_inside_the_page_set_is_refused(page_set):
    with pytest.raises(lines_bench.LineBenchmarkError, match="frozen"):
        lines_bench.build(page_set=page_set, out=page_set / "lines")


def test_an_existing_draw_is_not_overwritten(page_set, tmp_path):
    out = tmp_path / "vlm-lines-test"
    lines_bench.build(page_set=page_set, out=out)
    before = (out / "lines.jsonl").read_bytes()
    with pytest.raises(lines_bench.LineBenchmarkError, match="move the old draw aside"):
        lines_bench.build(page_set=page_set, out=out)
    assert (out / "lines.jsonl").read_bytes() == before


def test_the_cli_writes_beside_the_page_set_and_refuses_with_1(page_set, capsys):
    assert lines_bench.main(["--page-set", str(page_set)]) == 0
    out = page_set.parent / "vlm-lines-test"
    assert (out / "lines.jsonl").is_file()
    assert "3 lines from 2 of 3 pages" in capsys.readouterr().out

    assert lines_bench.main(["--page-set", str(page_set)]) == 1
    assert "refused:" in capsys.readouterr().err


# ── what the held-out check reads from it ───────────────────────────────────
def _manifest(directory: Path, stems: list[str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "pages_train.lst"
    path.write_text("".join(f"/scratch/jobs/x/data/pages/{s}.xml\n" for s in stems))
    return path


def test_the_heldout_check_reads_the_documents_of_the_lines(page_set, tmp_path):
    out = tmp_path / "vlm-lines-test"
    lines_bench.build(page_set=page_set, out=out)

    clean = rescore.heldout_check(out / "lines.jsonl",
                                  _manifest(tmp_path / "a", ["000009_1009_0001_9"]))
    assert clean["checked"] is True and clean["shared"] == []
    assert (clean["draw_samples"], clean["draw_documents"]) == (3, 1)

    crossed = rescore.heldout_check(out / "lines.jsonl",
                                    _manifest(tmp_path / "b", ["000077_1001_0009_7"]))
    assert crossed["shared"] == ["1001"]

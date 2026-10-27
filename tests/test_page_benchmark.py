"""#301: a held-out page set becomes a draw only if it is the set, and a score on it
is held out only if the run never trained on its documents.

Two halves, both offline:

* ``scripts/build_page_benchmark.py`` — refuses a list whose documents or page
  digest differ from the registry, refuses a list with a page missing on disk,
  copies what it accepts and writes a draw a VLM evaluator reads.
* ``ubelix/rescore_on_shared_draw.py --heldout`` — finds the training manifest of
  a run that adopted its corpus, and reads "unknown" as unknown, never as clean.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from atr_training.split_identity import page_digest
from atr_training.vlm_dataset import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load("build_page_benchmark", "scripts/build_page_benchmark.py")
rescore = _load("rescore_on_shared_draw", "ubelix/rescore_on_shared_draw.py")

PAGE_XML = """<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="{name}.jpg">
    <TextRegion>{lines}</TextRegion>
  </Page>
</PcGts>"""

#: ``<pool index>_<docId>_<folio>_<image>`` — field 1 is the document.
TEST_PAGES = ["000001_1001_0001_1", "000002_1001_0002_2", "000003_1002_0001_3"]


def write_page(directory: Path, stem: str, lines: list[str]) -> None:
    body = "".join(f"<TextLine><TextEquiv><Unicode>{t}</Unicode></TextEquiv></TextLine>"
                   for t in lines)
    (directory / f"{stem}.xml").write_text(PAGE_XML.format(name=stem, lines=body),
                                           encoding="utf-8")
    (directory / f"{stem}.jpg").write_bytes(b"\xff\xd8 not a real jpeg, not opened here")


def registry(tmp_path: Path, *, test_documents, test_pages) -> Path:
    path = tmp_path / "heldout.json"
    path.write_text(json.dumps({"sets": [{
        "name": "toy-v1",
        "test_documents": sorted(test_documents),
        "val_documents": ["2001"],
        "page_digests": {"test": page_digest(f"{p}.xml" for p in test_pages),
                         "val": "unused"},
    }]}), encoding="utf-8")
    return path


@pytest.fixture
def pool(tmp_path):
    pages = tmp_path / "pool" / "data" / "pages"
    pages.mkdir(parents=True)
    write_page(pages, TEST_PAGES[0], ["Item ontfaen van Janne", "anno 1431 / vigilia"])
    write_page(pages, TEST_PAGES[1], ["zweite seite"])
    write_page(pages, TEST_PAGES[2], ["dritte seite"])
    write_page(pages, "000004_1003_0001_4", ["eine trainingsseite"])
    lst = tmp_path / "pool" / "pages_test.lst"
    lst.write_text("".join(f"/somewhere/else/data/pages/{p}.xml\n" for p in TEST_PAGES))
    reg = registry(tmp_path, test_documents={"1001", "1002"}, test_pages=TEST_PAGES)
    return {"pages": pages, "list": lst, "registry": reg, "out": tmp_path / "eval_sets" / "toy"}


def build(pool, **over):
    args = dict(name="toy-v1", side="test", pages_list=pool["list"],
                pages_dir=pool["pages"], out=pool["out"], registry=pool["registry"])
    args.update(over)
    return bench.build(**args)


# ── building the draw ───────────────────────────────────────────────────────

def test_a_list_that_is_the_set_becomes_a_frozen_draw(pool):
    record = build(pool)

    assert (record["documents"], record["pages"], record["samples"]) == (2, 3, 3)
    samples = list(read_jsonl(pool["out"] / "pages.jsonl"))
    assert [s.source_type for s in samples] == ["page"] * 3
    assert samples[0].text == "Item ontfaen van Janne\nanno 1431 / vigilia"
    # Relative to --out, so the draw moves with its directory, and every image is
    # a copy: the pool is a job directory and those are cleaned (#183).
    for s in samples:
        assert not Path(s.image).is_absolute()
        assert (pool["out"] / s.image).is_file()
        assert (pool["out"] / s.page).is_file()
    saved = json.loads((pool["out"] / "benchmark.json").read_text())
    assert saved["page_digest"] == record["page_digest"]
    assert saved["reference_chars"] == sum(len(s.text) for s in samples)
    assert saved["draw_md5"] == hashlib.md5((pool["out"] / "pages.jsonl").read_bytes()).hexdigest()


def test_a_missing_document_refuses_and_writes_nothing(pool):
    pool["list"].write_text("".join(f"/x/{p}.xml\n" for p in TEST_PAGES[:2]))
    with pytest.raises(bench.BenchmarkError, match="1002"):
        build(pool)
    assert not pool["out"].exists()


def test_a_foreign_document_refuses(pool):
    pool["list"].write_text("".join(f"/x/{p}.xml\n" for p in TEST_PAGES + ["000004_1003_0001_4"]))
    with pytest.raises(bench.BenchmarkError, match="never test documents"):
        build(pool)


def test_the_right_documents_with_other_pages_refuse_on_the_digest(pool, tmp_path):
    """The case documents alone cannot catch (split_identity): same docIds, a
    different page of one of them."""
    write_page(pool["pages"], "000005_1001_0003_5", ["eine andere seite"])
    pool["list"].write_text("".join(f"/x/{p}.xml\n" for p in
                                    [TEST_PAGES[0], "000005_1001_0003_5", TEST_PAGES[2]]))
    with pytest.raises(bench.BenchmarkError, match="page digest"):
        build(pool)


def test_a_set_without_a_recorded_digest_is_refused(pool, tmp_path):
    raw = json.loads(pool["registry"].read_text())
    raw["sets"][0]["page_digests"] = {}
    pool["registry"].write_text(json.dumps(raw))
    with pytest.raises(bench.BenchmarkError, match="no test page digest"):
        build(pool)


def test_a_page_missing_on_disk_refuses(pool):
    (pool["pages"] / f"{TEST_PAGES[1]}.jpg").unlink()
    with pytest.raises(bench.BenchmarkError, match="no XML\\+JPEG pair"):
        build(pool)


def test_a_page_without_text_is_named_not_lost(pool):
    write_page(pool["pages"], TEST_PAGES[2], [])
    record = build(pool)
    assert record["samples"] == 2
    assert record["pages_without_text"] == [f"{TEST_PAGES[2]}.xml"]


def test_the_cli_says_refused_and_exits_1(pool, capsys):
    pool["list"].write_text(f"/x/{TEST_PAGES[0]}.xml\n")
    code = bench.main(["--set", "toy-v1", "--side", "test", "--pages-list", str(pool["list"]),
                       "--pages-dir", str(pool["pages"]), "--out", str(pool["out"]),
                       "--registry", str(pool["registry"])])
    assert code == 1
    assert "refused" in capsys.readouterr().err


# ── the held-out check of the rescore ───────────────────────────────────────

@pytest.fixture
def draw(pool):
    build(pool)
    return pool["out"] / "pages.jsonl"


def manifest(directory: Path, pages: list[str], name: str = "pages_train.lst") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("".join(f"/scratch/jobs/x/data/pages/{p}.xml\n" for p in pages))
    return path


def test_a_run_that_never_saw_the_documents_is_checked_and_clean(draw, tmp_path):
    train = manifest(tmp_path / "job" / "data", ["000004_1003_0001_4", "000009_1009_0001_9"])
    check = rescore.heldout_check(draw, train)
    assert check["checked"] is True and check["shared"] == []
    assert (check["draw_documents"], check["train_documents"]) == (2, 2)


def test_one_shared_document_is_reported_by_id(draw, tmp_path):
    """Another page of a held-out document in training is enough: same hand."""
    train = manifest(tmp_path / "job" / "data", ["000004_1003_0001_4", "000077_1002_0009_7"])
    assert rescore.heldout_check(draw, train)["shared"] == ["1002"]


@pytest.mark.parametrize("setup, why", [
    ("none", "no training manifest"),
    ("chunk_plan", "chunked run"),
    ("crops", "no document id readable in the training manifest"),
])
def test_an_unknown_answer_is_never_clean(draw, tmp_path, setup, why):
    data = tmp_path / "job" / "data"
    path = None
    if setup == "chunk_plan":
        path = manifest(data, [], name=rescore.CHUNK_PLAN_FILENAME)
    elif setup == "crops":
        path = manifest(data, ["crop_0000001", "crop_0000002"])
    check = rescore.heldout_check(draw, path)
    assert check["checked"] is False and why in check["why"]


def test_an_adopted_run_is_checked_against_its_corpus(tmp_path):
    """Every medieval olmOCR page arm adopted `2ad500ee8414`; none has a manifest
    of its own. The cache entry is asked first, the builder second."""
    jobs = tmp_path / "jobs"
    own = jobs / "20261019T180118Z-arm" / "data"
    own.mkdir(parents=True)
    builder = manifest(jobs / "20261001T093909Z-builder" / "data", ["000004_1003_0001_4"])
    note = "2ad500ee8414 reused, built by 20261001T093909Z-builder"

    assert rescore.train_manifest_of(own, artefact_note=note, cache_entry=None,
                                     jobs_root=jobs) == builder

    entry = tmp_path / "artefacts" / "2ad500ee8414abcd"
    cached = manifest(entry / "data", ["000004_1003_0001_4"])
    assert rescore.train_manifest_of(own, artefact_note=note, cache_entry=entry,
                                     jobs_root=jobs) == cached

    assert rescore.train_manifest_of(own, artefact_note=None, cache_entry=None,
                                     jobs_root=jobs) is None


def test_chunk_manifests_are_read_together(draw, tmp_path):
    data = tmp_path / "job" / "data"
    first = manifest(data, ["000004_1003_0001_4"], name="pages_train_000.lst")
    manifest(data, ["000077_1001_0009_7"], name="pages_train_001.lst")
    assert rescore.heldout_check(draw, first)["shared"] == ["1001"]


def test_heldout_needs_a_draw_file():
    assert rescore.main(["--job", "x", "--draw-from", "y", "--heldout"]) == 2


def test_a_given_manifest_needs_heldout_and_must_exist(tmp_path):
    draw = tmp_path / "pages.jsonl"
    draw.write_text("{}\n")
    base = ["--job", "x", "--draw-file", str(draw), "--data-root", str(tmp_path),
            "--label", "toy"]
    assert rescore.main(base + ["--train-manifest", str(draw)]) == 2
    assert rescore.main(base + ["--heldout", "--train-manifest",
                                str(tmp_path / "gone.lst")]) == 2

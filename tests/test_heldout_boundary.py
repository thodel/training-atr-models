"""#247: the census must never count a run it could not check as unaffected.

The question is which of 54 published CERs were measured on models that trained
on material today's registry holds back. `progress.reserved_pages` cannot answer
it — it is a corpus fingerprint, and for 42 of the runs it is `None` or `0` — so
the check runs against the manifest each job still holds.

The whole risk of such a census is in one direction. A run whose manifest is
gone, or whose manifest names no document at all, produces zero reserved pages
exactly like a run that was really checked and really is clean. Count those as
clean and the answer comes out optimistic, which is the shape of the defect
being measured in the first place.

So every state has a test here, and the two non-answers are tested *against*
being read as answers. The fourth state is new: #247 names three, and reading
the code for it turned up manifests that name no document — a line corpus of
foreign scans, and a chunked run whose train side was materialised per chunk and
discarded. Both recorded `reserved_pages: 0`.

Offline: tmp_path job stores, a registry built in the test. The real stores are
on the research share and in UBELIX scratch; the script says so and counts
nothing when it cannot read them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atr_training.chunking import CHUNK_PLAN_FILENAME
from atr_training.heldout import HeldOutDocuments
from atr_training.heldout_boundary import (
    Verdict,
    check_run,
    check_store,
    find_manifest,
    resolve_adoptions,
    summarize,
)

RESERVED_DOC = "018155"
OTHER_DOC = "099999"


@pytest.fixture
def held() -> HeldOutDocuments:
    return HeldOutDocuments(frozenset({RESERVED_DOC}), {"german-medieval-v1": 1})


def _page(document: str, folio: str = "f001r") -> str:
    """A materialised page name, in the shape `document_of` reads."""
    return f"0_{document}_{folio}_IMG001.xml"


def _run(root: Path, job_id: str, *, manifest: str | None = None,
         entries: list[str] | None = None, cer: float | None = 0.0712,
         reserved_pages: int | None = None, record: bool = True,
         artefact: str | None = None) -> Path:
    job_dir = root / job_id
    (job_dir / "data").mkdir(parents=True)
    if manifest is not None:
        (job_dir / "data" / manifest).write_text("\n".join(entries or []) + "\n",
                                                 encoding="utf-8")
    if record:
        body = {"id": job_id, "status": "completed",
                "progress": {"reserved_pages": reserved_pages, "artefact": artefact},
                "metrics": {"cer": cer} if cer is not None else {}}
        (job_dir / "job.json").write_text(json.dumps(body), encoding="utf-8")
    return job_dir


# ── the two real answers ────────────────────────────────────────────────────

def test_a_run_that_trained_on_reserved_material_is_crossed(tmp_path, held):
    job = _run(tmp_path, "20260910T110352Z-qwen3vl-german-pages-v3",
               manifest="pages_train.lst",
               entries=[_page(RESERVED_DOC), _page(OTHER_DOC), _page(RESERVED_DOC, "f002v")])

    check = check_run(job, store="share", held=held)

    assert check.verdict is Verdict.CROSSED
    assert check.reserved == 2
    assert check.entries == 3
    assert check.documents == [RESERVED_DOC], "the write-up names the documents"
    assert "reserved today" in check.why


def test_a_run_whose_pages_are_all_elsewhere_is_clean(tmp_path, held):
    job = _run(tmp_path, "20260920T000000Z-kraken-xix", manifest="pages_train.lst",
               entries=[_page(OTHER_DOC), _page("088888")])

    check = check_run(job, store="share", held=held)

    assert check.verdict is Verdict.CLEAN
    assert check.reserved == 0
    assert check.identified == 2, "clean means documents were identified and ruled out"


# ── the two non-answers, which must not read as the first one ───────────────

def test_a_purged_data_directory_is_undecidable_not_clean(tmp_path, held):
    """Scratch is purged after 30 days. The manifest is the evidence; without it
    there is no verdict about the number, and `clean` would be a claim."""
    job = _run(tmp_path, "20260801T000000Z-kraken-old", manifest=None)

    check = check_run(job, store="scratch", held=held)

    assert check.verdict is Verdict.GONE
    assert check.reserved == 0
    assert check.verdict is not Verdict.CLEAN


def test_a_manifest_naming_no_document_is_uncheckable_not_clean(tmp_path, held):
    """The fourth state, and the one #247 does not have.

    A line corpus whose pages are foreign scans has no `docId` the registry
    could match. Nothing is reserved — and nothing was ruled out either. Before
    this, such a run recorded `reserved_pages: 0`, which is also what a real
    check that found nothing records.
    """
    job = _run(tmp_path, "20260925T000000Z-kraken-towerbooks-lines",
               manifest="lines_train.jsonl", reserved_pages=0,
               entries=[json.dumps({"image": "images/000001_x.jpg", "text": "a",
                                    "page": "scan_0001.jpg"}),
                        json.dumps({"image": "images/000002_y.jpg", "text": "b",
                                    "page": "scan_0002.jpg"})])

    check = check_run(job, store="share", held=held)

    assert check.verdict is Verdict.UNCHECKABLE
    assert check.verdict is not Verdict.CLEAN
    assert check.recorded_reserved == 0, "the record claimed a check that did not happen"
    assert "no document id" in check.why


def test_the_crop_name_is_not_accepted_as_a_document(tmp_path, held):
    """A sample without a `page` is unidentifiable, and the image beside it must
    not stand in for one.

    `images/000001_x.jpg` parses as a page name — index `000001`, "document"
    `x` — so a fallback to the image would make this run count as *identified*
    and come out `clean`: nothing matched, because nothing could. That is the
    false negative this whole census is built to avoid, produced by a line of
    convenience.
    """
    job = _run(tmp_path, "20260927T000000Z-kraken-no-page",
               manifest="lines_train.jsonl", reserved_pages=0,
               entries=[json.dumps({"image": "images/000001_x.jpg", "text": "a"}),
                        json.dumps({"image": "images/000002_y.jpg", "text": "b"})])

    check = check_run(job, store="share", held=held)

    assert check.verdict is Verdict.UNCHECKABLE
    assert check.identified == 0, "a crop name was accepted as a document id"


def test_a_chunked_run_is_uncheckable_and_says_why(tmp_path, held):
    """`train_plan.json` is what a chunked run leaves where the manifest would
    be. It is nine lines of JSON; read as a page list it names no document."""
    job = _run(tmp_path, "20261001T000000Z-kraken-corpus", manifest=CHUNK_PLAN_FILENAME,
               reserved_pages=0,
               entries=json.dumps({"hf_repo": "dh-unibe/medieval",
                                   "data_files": ["data/train-00000.parquet"],
                                   "chunk_pages": 5000}, indent=2).splitlines())

    check = check_run(job, store="share", held=held)

    assert check.verdict is Verdict.UNCHECKABLE
    assert "chunked" in check.why
    assert "per chunk" in check.why


def test_a_line_manifest_whose_pages_are_materialised_is_checked(tmp_path, held):
    """The counter-test: uncheckable is a property of the manifest, not of the
    format. A line corpus cropped from materialised pages carries the document
    in its `page` field and is checked like any other."""
    job = _run(tmp_path, "20260926T000000Z-kraken-lines-medieval",
               manifest="lines_train.jsonl",
               entries=[json.dumps({"image": "images/000001_x.jpg", "text": "a",
                                    "page": _page(RESERVED_DOC)}),
                        json.dumps({"image": "images/000002_y.jpg", "text": "b",
                                    "page": _page(OTHER_DOC)})])

    check = check_run(job, store="share", held=held)

    assert check.verdict is Verdict.CROSSED
    assert check.reserved == 1


# ── the chunked compile case, where the pages do exist ──────────────────────

def test_every_chunk_manifest_is_counted(tmp_path, held):
    """A chunked run that reached compile has one manifest per chunk. Checking
    only the first would under-count — and under-counting is the one direction
    this census must not fail in."""
    job = _run(tmp_path, "20261002T000000Z-kraken-corpus", manifest=None)
    data = job / "data"
    for index, document in enumerate([RESERVED_DOC, OTHER_DOC, RESERVED_DOC]):
        (data / f"pages_train_{index:04d}.lst").write_text(
            _page(document) + "\n", encoding="utf-8")

    check = check_run(job, store="share", held=held)

    assert find_manifest(data).name == "pages_train_0000.lst"
    assert check.entries == 3, "all three chunk manifests were read"
    assert check.reserved == 2
    assert check.verdict is Verdict.CROSSED


# ── the census over a store ─────────────────────────────────────────────────

def test_only_runs_with_a_number_are_counted_by_default(tmp_path, held):
    """#247 counts the 54 runs that have a CER: those are the numbers that were
    published. A queued or failed run has nothing to correct."""
    _run(tmp_path, "20260910T000000Z-with-cer", manifest="pages_train.lst",
         entries=[_page(OTHER_DOC)], cer=0.0712)
    _run(tmp_path, "20260911T000000Z-no-cer", manifest="pages_train.lst",
         entries=[_page(RESERVED_DOC)], cer=None)

    counted = check_store(tmp_path, store="share", held=held)
    everything = check_store(tmp_path, store="share", held=held, completed_only=False)

    assert [c.job_id for c in counted] == ["20260910T000000Z-with-cer"]
    assert len(everything) == 2


def test_an_unreadable_record_gets_a_row_rather_than_vanishing(tmp_path, held):
    """A run missing from the table without a reason is how a census loses a
    case: 54 rows have to be 54 rows."""
    job = _run(tmp_path, "20260912T000000Z-broken", manifest="pages_train.lst",
               entries=[_page(RESERVED_DOC)], record=False)
    (job / "job.json").write_text("{not json", encoding="utf-8")

    checks = check_store(tmp_path, store="share", held=held)

    assert [c.verdict for c in checks] == [Verdict.GONE]
    assert "unreadable" in checks[0].why


def test_the_summary_separates_the_answers_from_the_non_answers(tmp_path, held):
    _run(tmp_path, "20260901T000000Z-a", manifest="pages_train.lst",
         entries=[_page(RESERVED_DOC)] * 2 + [_page(OTHER_DOC)])
    _run(tmp_path, "20260902T000000Z-b", manifest="pages_train.lst",
         entries=[_page(OTHER_DOC)])
    _run(tmp_path, "20260903T000000Z-c", manifest=None)
    _run(tmp_path, "20260904T000000Z-d", manifest="lines_train.jsonl",
         entries=[json.dumps({"image": "i.jpg", "page": "scan_1.jpg"})])

    counts = summarize(check_store(tmp_path, store="share", held=held))

    assert counts == {"clean": 1, "crossed": 1, "uncheckable": 1, "gone": 1,
                      "runs": 4, "reserved_pages": 2}
    assert counts["clean"] + counts["crossed"] != counts["runs"], (
        "two of these four runs are not a verdict about their number")


def test_a_directory_that_is_not_a_run_gets_no_row(tmp_path, held):
    """`evalset-federal-minutes` holds benchmark reports beside the jobs. Read as
    a run it was a purged one, and the undecidable column grew by a run that
    never trained."""
    (tmp_path / "evalset-federal-minutes" / "data").mkdir(parents=True)
    (tmp_path / "evalset-federal-minutes" / "report_qwen35-4b.json").write_text("{}")
    job = _run(tmp_path, "20260912T000000Z-broken", manifest=None, record=False)
    job.joinpath("data").rmdir()

    checks = check_store(tmp_path, store="scratch", held=held)

    assert [c.job_id for c in checks] == ["20260912T000000Z-broken"], (
        "a job-shaped directory without a record still gets its row")


def test_a_store_that_is_not_there_contributes_no_rows_and_no_verdicts(tmp_path, held):
    """Not an error, and not an empty 'all clean' either — the caller prints the
    store as unreadable, and nothing is counted."""
    assert check_store(tmp_path / "never-mounted", store="share", held=held) == []


# ── the row that goes into the docs page ────────────────────────────────────

def test_the_row_shows_the_record_beside_the_measurement(tmp_path, held):
    """Where the two disagree, the table has to show both: the record said `0`
    for runs that were never checked, and overwriting it would hide exactly the
    finding #247 is looking for."""
    job = _run(tmp_path, "20261001T000000Z-kraken-corpus", manifest=CHUNK_PLAN_FILENAME,
               reserved_pages=0, cer=0.1234,
               entries=json.dumps({"hf_repo": "x", "chunk_pages": 5000}, indent=2).splitlines())

    row = check_run(job, store="share", held=held).row
    # By column, not by substring: `| 0 |` also matches the "pages fallen"
    # column, so the first version of this check passed with the recorded value
    # replaced by "n/a" — it was reading the wrong cell.
    cells = [cell.strip() for cell in row.strip().strip("|").split("|")]

    assert cells == ["`20261001T000000Z-kraken-corpus`", "share", "0.1234", "0",
                     "uncheckable", "0",
                     "chunked run: the train side was materialised per chunk and discarded"]


# ── a run that adopted its corpus has the builder's manifest ────────────────
#
# Measured 2026-10-17: 16 of the 25 runs without a manifest had adopted a
# compiled corpus, and their records name the builder. Every case below is one
# that occurred: the adopter in scratch and the builder in the share, a builder
# without a CER of its own (a cancelled first attempt), and an adopter whose
# "built by" names a later job that rebuilt the same key.

KEY = "2439ac90f727"


def _adopter(root: Path, job_id: str, builder: str, *, key: str = KEY,
             reserved_pages: int | None = 0) -> Path:
    return _run(root, job_id, manifest=None, reserved_pages=reserved_pages,
                artefact=f"{key} reused, built by {builder}")


def _builder(root: Path, job_id: str, pages: list[str], *, key: str = KEY,
             cer: float | None = 0.4265) -> Path:
    return _run(root, job_id, manifest="pages_train.lst", entries=pages, cer=cer,
                artefact=f"{key} built by this job")


def _census(stores, held):
    checks = []
    for label, root in stores:
        checks += check_store(root, store=label, held=held)
    return resolve_adoptions(checks, stores, held=held)


def test_an_adopter_takes_the_clean_verdict_of_its_builder(tmp_path, held):
    """The medieval page arms: the record said `0`, the builder had held back
    3173 pages, and the run is clean — not because of the `0`."""
    scratch = tmp_path / "scratch"
    _builder(scratch, "20260923T000000Z-qwen3vl-page", [_page(OTHER_DOC)])
    _adopter(scratch, "20260926T000000Z-qwen35-page", "20260923T000000Z-qwen3vl-page")

    checks = {c.job_id: c for c in _census([("scratch", scratch)], held)}
    adopter = checks["20260926T000000Z-qwen35-page"]

    assert adopter.verdict is Verdict.CLEAN
    assert adopter.inherited_from == "20260923T000000Z-qwen3vl-page"
    assert adopter.recorded_reserved == 0, "the record stays beside the verdict"
    assert adopter.cer == 0.0712, "the adopter keeps its own number"
    assert f"adopted {KEY}" in adopter.why


def test_an_adopter_of_a_crossed_corpus_is_crossed(tmp_path, held):
    """Inheritance has to carry the bad answer as readily as the good one."""
    scratch = tmp_path / "scratch"
    _builder(scratch, "20260925T000000Z-ladder-27b",
             [_page(RESERVED_DOC), _page(OTHER_DOC)])
    _adopter(scratch, "20260930T000000Z-ladder-9b", "20260925T000000Z-ladder-27b")

    adopter = {c.job_id: c for c in _census([("scratch", scratch)], held)}[
        "20260930T000000Z-ladder-9b"]

    assert adopter.verdict is Verdict.CROSSED
    assert adopter.reserved == 1
    assert adopter.documents == [RESERVED_DOC]


def test_the_builder_is_found_in_another_store_and_without_a_cer(tmp_path, held):
    """`qwen3vl-german-pages-v5-asteraix`: the builder was cancelled, has no CER,
    and so is not in the census at all — it still holds the manifest."""
    share, scratch = tmp_path / "share", tmp_path / "scratch"
    _builder(share, "20260916T115939Z-pages-v5", [_page(RESERVED_DOC)], key="82db328c96d7",
             cer=None)
    _adopter(scratch, "20260916T205123Z-pages-v5", "20260916T115939Z-pages-v5",
             key="82db328c96d7")

    checks = _census([("scratch", scratch), ("share", share)], held)

    assert [c.job_id for c in checks] == ["20260916T205123Z-pages-v5"]
    assert checks[0].verdict is Verdict.CROSSED
    assert "(share)" in checks[0].why


def test_a_builder_that_does_not_confirm_the_key_is_not_trusted(tmp_path, held):
    """The name in an adopter's record is a label. A job of that name that says
    it built a different key cannot vouch for this corpus."""
    scratch = tmp_path / "scratch"
    _builder(scratch, "20261001T000000Z-olmocr-page", [_page(OTHER_DOC)], key="2ad500ee8414")
    _adopter(scratch, "20261002T000000Z-sweep", "20261001T000000Z-olmocr-page")

    adopter = {c.job_id: c for c in _census([("scratch", scratch)], held)}[
        "20261002T000000Z-sweep"]

    assert adopter.verdict is Verdict.GONE
    assert adopter.verdict is not Verdict.CLEAN
    assert "confirms building" in adopter.why


def test_a_builder_that_records_its_copy_time_still_confirms(tmp_path, held):
    """#238 appended the cache copy time: `<key> built by this job, stored in
    12 min`. An exact match on the old form would make every builder from then
    on fail to confirm, and leave its adopters `gone` without anyone noticing —
    GitHub merged the two changes as textually independent."""
    scratch = tmp_path / "scratch"
    _run(scratch, "20261020T000000Z-builder", manifest="pages_train.lst",
         entries=[_page(OTHER_DOC)], artefact=f"{KEY} built by this job, stored in 12 min")
    _adopter(scratch, "20261021T000000Z-adopter", "20261020T000000Z-builder")

    adopter = {c.job_id: c for c in _census([("scratch", scratch)], held)}[
        "20261021T000000Z-adopter"]

    assert adopter.verdict is Verdict.CLEAN


def test_a_builder_of_a_longer_key_with_the_same_prefix_does_not_confirm(tmp_path, held):
    """The prefix match must stop at the key: `<key>0` is another corpus."""
    scratch = tmp_path / "scratch"
    _run(scratch, "20261020T000000Z-builder", manifest="pages_train.lst",
         entries=[_page(OTHER_DOC)], artefact=f"{KEY}0 built by this job")
    _adopter(scratch, "20261021T000000Z-adopter", "20261020T000000Z-builder")

    adopter = {c.job_id: c for c in _census([("scratch", scratch)], held)}[
        "20261021T000000Z-adopter"]

    assert adopter.verdict is Verdict.GONE


def test_a_builder_whose_manifest_is_gone_too_leaves_the_adopter_gone(tmp_path, held):
    scratch = tmp_path / "scratch"
    _run(scratch, "20260901T000000Z-builder", manifest=None,
         artefact=f"{KEY} built by this job")
    _adopter(scratch, "20260902T000000Z-adopter", "20260901T000000Z-builder")

    checks = {c.job_id: c for c in _census([("scratch", scratch)], held)}

    assert checks["20260902T000000Z-adopter"].verdict is Verdict.GONE


def test_a_builder_in_no_store_leaves_the_adopter_gone(tmp_path, held):
    scratch = tmp_path / "scratch"
    _adopter(scratch, "20260902T000000Z-adopter", "20260901T000000Z-nowhere")

    (adopter,) = _census([("scratch", scratch)], held)

    assert adopter.verdict is Verdict.GONE
    assert "in no store" in adopter.why


def test_a_run_with_its_own_manifest_is_judged_on_it_not_on_its_builder(tmp_path, held):
    """Adoption is consulted only for a run that has nothing of its own: the
    material it holds wins over what its record says about where it came from."""
    scratch = tmp_path / "scratch"
    _builder(scratch, "20260901T000000Z-builder", [_page(OTHER_DOC)])
    _run(scratch, "20260902T000000Z-adopter", manifest="pages_train.lst",
         entries=[_page(RESERVED_DOC)], artefact=f"{KEY} reused, built by 20260901T000000Z-builder")

    adopter = {c.job_id: c for c in _census([("scratch", scratch)], held)}[
        "20260902T000000Z-adopter"]

    assert adopter.verdict is Verdict.CROSSED
    assert adopter.inherited_from is None

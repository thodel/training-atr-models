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
         reserved_pages: int | None = None, record: bool = True) -> Path:
    job_dir = root / job_id
    (job_dir / "data").mkdir(parents=True)
    if manifest is not None:
        (job_dir / "data" / manifest).write_text("\n".join(entries or []) + "\n",
                                                 encoding="utf-8")
    if record:
        body = {"id": job_id, "status": "completed",
                "progress": {"reserved_pages": reserved_pages},
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

    assert counts == {"clean": 1, "enforced": 0, "crossed": 1, "uncheckable": 1,
                      "gone": 1, "runs": 4, "decided": 2, "reserved_pages": 2}
    assert counts["decided"] != counts["runs"], (
        "two of these four runs are not a verdict about their number")


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


# ── decided without a manifest: the registry's own fingerprint ──────────────
#
# `progress.heldout_fingerprint` records which registry a run's corpus was
# prepared under, and the artefact payload carries it across a cache hit — an
# artefact built under a different registry has a different key and is never
# adopted. So a run that applied today's registry needs no manifest to be
# decided, which is the only way a purged run can be decided at all.
#
# The trap is that the field is written at the *top* of
# `_reserve_eval_documents`, before any of its checks: it says "consulted", not
# "applied". The two tests at the end of this block are that trap.

TODAY = "6a47d9b94d2ac739"


def _record(root: Path, job_id: str, *, fingerprint: str | None, reserved: int | None,
            source: str | None = "prepare", manifest: str | None = None,
            entries: list[str] | None = None) -> Path:
    job_dir = root / job_id
    (job_dir / "data").mkdir(parents=True)
    if manifest is not None:
        (job_dir / "data" / manifest).write_text("\n".join(entries or []) + "\n",
                                                 encoding="utf-8")
    (job_dir / "job.json").write_text(json.dumps({
        "id": job_id, "status": "completed",
        "progress": {"heldout_fingerprint": fingerprint, "reserved_pages": reserved,
                     "reserved_pages_source": source},
        "metrics": {"cer": 0.1224},
    }), encoding="utf-8")
    return job_dir


def test_a_run_prepared_under_todays_registry_is_decided_without_its_manifest(tmp_path, held):
    """The 3,173-page runs of the medieval ladder: the reservation really ran and
    really dropped pages, so the number stands — and the scratch may be gone."""
    job = _record(tmp_path, "20261016T000000Z-qwen3.5-27b-right-of-the-boundary",
                  fingerprint=TODAY, reserved=3173, manifest=None)

    check = check_run(job, store="share", held=held, today=TODAY)

    assert check.verdict is Verdict.ENFORCED
    assert "3173" in check.why and TODAY in check.why
    assert check.verdict is not Verdict.GONE, "no manifest was needed"


def test_an_artefact_hit_answers_for_the_registry_it_was_built_under(tmp_path, held):
    """A reused corpus skips prepare, and the payload carries the fingerprint —
    so the run answers for the registry the *artefact* was built under, which is
    the registry its corpus actually had."""
    job = _record(tmp_path, "20261016T000001Z-reused",
                  fingerprint=TODAY, reserved=3173,
                  source="artefact f7188cd0b798 built by 20261015T120000Z-first")

    check = check_run(job, store="share", held=held, today=TODAY)

    assert check.verdict is Verdict.ENFORCED
    assert "artefact f7188cd0b798" in check.why


def test_an_older_registry_is_not_decided_by_its_fingerprint(tmp_path, held):
    """It dropped *some* registry's documents, and today's set is a different
    one. Only the manifest can say whether the difference is in this corpus."""
    job = _record(tmp_path, "20260910T110352Z-under-an-older-registry",
                  fingerprint="0000000000000000", reserved=198,
                  manifest="pages_train.lst",
                  entries=[_page(RESERVED_DOC), _page(OTHER_DOC)])

    check = check_run(job, store="share", held=held, today=TODAY)

    assert check.verdict is Verdict.CROSSED, "the manifest decided it, not the record"
    assert check.reserved == 1


def test_todays_fingerprint_with_nothing_dropped_is_not_decided(tmp_path, held):
    """The trap, and the reason this rule needs two fields.

    The fingerprint is written before the reservation's own checks, so a chunked
    run from before #255 carries today's value and dropped nothing: it read
    `train_plan.json` as a page list, found no document, and recorded `0`. From
    the record alone that is indistinguishable from a page run that was really
    checked and really holds nothing reserved — so it falls through, and here
    there is no manifest left to fall through to.
    """
    job = _record(tmp_path, "20261001T000000Z-chunked-before-255",
                  fingerprint=TODAY, reserved=0, manifest=None)

    check = check_run(job, store="share", held=held, today=TODAY)

    assert check.verdict is Verdict.GONE
    assert check.verdict is not Verdict.ENFORCED, (
        "a 0 beside today's fingerprint was taken as proof that it was applied")


def test_a_consulted_but_unchecked_registry_is_not_decided(tmp_path, held):
    """What #247 made the record say: `reserved_pages: None` with a reason. The
    fingerprint is there — the registry was consulted — and nothing was applied."""
    job = _record(tmp_path, "20261002T000000Z-chunked-after-255",
                  fingerprint=TODAY, reserved=None,
                  source="not checked: chunked prepare defers the train side to compile",
                  manifest=None)

    check = check_run(job, store="share", held=held, today=TODAY)

    assert check.verdict is Verdict.GONE
    assert check.verdict is not Verdict.ENFORCED


def test_the_fingerprint_is_the_trainers_own_definition():
    """Compared value against value, not implementation against implementation:
    the same function the trainer writes onto the record and folds into the
    artefact key."""
    from atr_training.artefact_cache import heldout_fingerprint
    from atr_training.heldout_boundary import today_fingerprint

    assert today_fingerprint() == heldout_fingerprint()


def test_without_a_fingerprint_to_compare_nothing_is_decided_by_it(tmp_path, held):
    """`today=None` is how a caller says "do not use this rule" — and then every
    run goes to its manifest, as before."""
    job = _record(tmp_path, "20261016T000002Z-no-rule",
                  fingerprint=TODAY, reserved=3173, manifest="pages_train.lst",
                  entries=[_page(OTHER_DOC)])

    assert check_run(job, store="share", held=held, today=None).verdict is Verdict.CLEAN
    assert check_run(job, store="share", held=held, today=TODAY).verdict is Verdict.ENFORCED

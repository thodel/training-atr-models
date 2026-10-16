"""#255: the held-out boundary has to hold for a chunked run too.

In chunked mode `prepare` writes no page manifest. It writes `train_plan.json`
and defers the train side to `compile`, so the reservation step received the
plan — nine lines of JSON — read it as a page list, found no document, reserved
nothing, and recorded `reserved_pages: 0`. That is the same value a real check
that found nothing records, which is what #247 then has to untangle across 54
runs.

The pages of a chunked run exist for one chunk at a time, inside the compile
loop, and nowhere else: they are discarded as soon as the chunk's arrow is
written. So that loop is the only place the registry can be applied, and these
tests put a corpus through it whose documents are partly reserved.

Two things are harder here than in the page path and have a test each:

* the **line count**. A chunk's `written.lines` counts every page it
  materialised; the reserved ones are not trained on. `train_lines` divides the
  step count (#72), so counting them in flatters the configuration — which is
  what that field's own comment says it exists to prevent.
* the **resume** (#72). A chunk a previous attempt finished is adopted without
  being rebuilt, and its pages are gone. Its reserved count has to come out of
  its record; for a record written before that key existed, the total for the
  run is unknowable, and `None` is the only honest answer.
"""

from __future__ import annotations

import json

import pytest

from atr_training import heldout
from atr_training.chunking import chunk_record
from kraken_train_svc.runner import Pipeline
from tests.test_train_svc_pipeline import (  # noqa: F401 — fixtures
    EMPTY_XML,
    PAGE_XML,
    THUN_TEST,
    THUN_TRAIN,
    FakeRunner,
    request_with,
    settings,
    store,
)

#: Two documents, one reserved and one not. `page_stem` keeps the filename's
#: stem after an index prefix, so `018155_f001r.jpg` materialises as
#: `000000_018155_f001r.xml` — field 1, which is what `document_of` reads.
RESERVED_DOC = "018155"
OTHER_DOC = "099999"


class DocumentSource:
    """A page stream whose filenames carry a Transkribus-shaped document id.

    `FakeSource` names every train page `train_<i>.jpg`, so they all share the
    document "train" — fine for counting chunks, useless for a boundary that
    draws a line *between* documents.
    """

    def __init__(self, documents: list[str], eval_pages: int = 2) -> None:
        self.documents = documents
        self.eval_pages = eval_pages

    def stream(self, hf_repo, data_files, revision=None):
        if any(THUN_TEST in f for f in data_files):
            for index in range(self.eval_pages):
                yield self._row(f"eval_{index}", THUN_TEST)
            return
        for index, document in enumerate(self.documents):
            yield self._row(f"{document}_f{index:03d}r", THUN_TRAIN)

    @staticmethod
    def _row(stem: str, project: str) -> dict:
        return {"image": {"bytes": b"\xff\xd8" + stem.encode(), "path": f"{stem}.jpg"},
                "xml_content": PAGE_XML, "filename": f"{stem}.jpg",
                "project_name": project}


@pytest.fixture
def held(monkeypatch):
    """One reserved document, installed where both paths read it."""
    from atr_training import runner_base

    documents = heldout.HeldOutDocuments(frozenset({RESERVED_DOC}),
                                        {"german-medieval-v1": 1})
    monkeypatch.setattr(runner_base, "load_heldout", lambda *a, **kw: documents)
    monkeypatch.setattr("kraken_train_svc.runner.load_heldout", lambda *a, **kw: documents)
    return documents


def _chunked(settings, chunk_pages=2):  # noqa: F811
    settings.chunk_pages = chunk_pages
    return settings


def _run(store, settings, source, runner=None):  # noqa: F811
    job = store.create(request_with(force=True))
    runner = runner or FakeRunner()
    Pipeline(store, settings, runner=runner, source=source).execute(job.id)
    return store.load(job.id)


# ── the boundary holds ──────────────────────────────────────────────────────

def test_reserved_pages_are_dropped_from_every_chunk(store, settings, held):  # noqa: F811
    """Six pages over three chunks, two of them reserved and in different
    chunks: both have to go, and the count has to be the sum."""
    source = DocumentSource([RESERVED_DOC, OTHER_DOC, OTHER_DOC,
                             RESERVED_DOC, OTHER_DOC, OTHER_DOC])

    job = _run(store, _chunked(settings, chunk_pages=2), source)

    assert job.progress.reserved_pages == 2
    assert job.progress.reserved_pages_source == "compile (per chunk)"
    assert job.progress.pages_written == 4, "the record counts trained pages"


def test_no_manifest_lists_a_reserved_page(store, settings, held):  # noqa: F811
    """The record is a claim; the manifests are what ketos reads."""
    source = DocumentSource([RESERVED_DOC, OTHER_DOC, RESERVED_DOC, OTHER_DOC])

    job = _run(store, _chunked(settings, chunk_pages=2), source)

    listed = "\n".join(path.read_text(encoding="utf-8") for path in
                       sorted((store.paths(job.id).data).glob("pages_train_*.lst")))
    assert listed.strip(), "no chunk manifest was written"
    assert RESERVED_DOC not in listed
    assert OTHER_DOC in listed


def test_the_line_count_follows_the_pages_that_were_kept(store, settings, held):  # noqa: F811
    """`train_lines` divides the step count (#72). A reserved page's lines are
    on disk and in no arrow, so counting them would promise training data that
    does not exist."""
    all_other = DocumentSource([OTHER_DOC] * 4)
    half_reserved = DocumentSource([RESERVED_DOC, OTHER_DOC, RESERVED_DOC, OTHER_DOC])

    full = _run(store, _chunked(settings, chunk_pages=2), all_other)
    reduced = _run(store, _chunked(settings, chunk_pages=2), half_reserved)

    assert full.progress.train_lines > 0
    assert reduced.progress.train_lines == full.progress.train_lines // 2, (
        "half the pages were reserved, so half the training lines are gone")
    assert reduced.progress.lines_written == reduced.progress.train_lines


def test_a_chunk_that_is_entirely_reserved_is_skipped_not_fatal(store, settings, held):  # noqa: F811
    """One chunk of eval material does not condemn the run — the next chunk may
    be all training pages."""
    source = DocumentSource([RESERVED_DOC, RESERVED_DOC, OTHER_DOC, OTHER_DOC])

    job = _run(store, _chunked(settings, chunk_pages=2), source)

    assert job.status == "completed"
    assert job.progress.reserved_pages == 2
    assert job.progress.pages_written == 2
    arrows = sorted(p.name for p in store.paths(job.id).data.glob("train_*.arrow"))
    assert arrows == ["train_0001.arrow"], "the all-reserved chunk compiled nothing"


def test_a_selection_that_is_entirely_the_eval_set_is_refused(store, settings, held):  # noqa: F811
    """The same refusal the page path makes, made once and after the loop —
    before it, "this chunk is all eval" and "everything is eval" look alike."""
    source = DocumentSource([RESERVED_DOC] * 4)

    job = store.create(request_with(force=True))
    pipeline = Pipeline(store, _chunked(settings, chunk_pages=2),
                        runner=FakeRunner(), source=source)
    pipeline.execute(job.id)

    record = store.load(job.id)
    assert record.status == "failed"
    assert "eval set, not a training corpus" in (record.error or "")


# ── the resume, where the pages are already gone ────────────────────────────

def test_an_adopted_chunk_contributes_its_recorded_count(store, settings, held):  # noqa: F811
    """A chunk a previous attempt finished is not rebuilt, so its reserved count
    can only come from its record."""
    source = DocumentSource([RESERVED_DOC, OTHER_DOC, RESERVED_DOC, OTHER_DOC])
    settings = _chunked(settings, chunk_pages=2)

    first = _run(store, settings, source)
    recorded = json.loads(
        chunk_record(store.paths(first.id).data / "train_0000.arrow").read_text())
    assert recorded["reserved"] == 1, "the record carries the count"

    # A second run over the same corpus, resuming into the first one's chunks.
    second = store.create(request_with(force=True))
    for name in ("train_0000.arrow", "train_0001.arrow"):
        source_arrow = store.paths(first.id).data / name
        dest = store.paths(second.id).data
        dest.mkdir(parents=True, exist_ok=True)
        (dest / name).write_bytes(source_arrow.read_bytes())
        chunk_record(dest / name).write_text(
            chunk_record(source_arrow).read_text(), encoding="utf-8")

    Pipeline(store, settings, runner=FakeRunner(),
             source=DocumentSource([RESERVED_DOC, OTHER_DOC, RESERVED_DOC, OTHER_DOC])
             ).execute(second.id)

    resumed = store.load(second.id)
    assert resumed.progress.reserved_pages == 2
    assert resumed.progress.reserved_pages_source == "compile (per chunk)"
    # The same numbers as the run that built the chunks. A resume reads them
    # back out of the records rather than recounting — the pages are gone — so
    # a record that stored the *materialised* count would make a resumed run
    # report training pages it does not have.
    assert resumed.progress.pages_written == first.progress.pages_written == 2
    assert resumed.progress.train_lines == first.progress.train_lines


def test_a_record_without_the_count_makes_the_total_unknown(store, settings, held):  # noqa: F811
    """A chunk compiled before this count existed. Its pages are discarded and
    cannot be re-examined, so the run's total is unknowable — and `None` is the
    answer, not a sum that silently omits it (#119).
    """
    source = DocumentSource([RESERVED_DOC, OTHER_DOC, RESERVED_DOC, OTHER_DOC])
    settings = _chunked(settings, chunk_pages=2)

    first = _run(store, settings, source)

    second = store.create(request_with(force=True))
    dest = store.paths(second.id).data
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("train_0000.arrow", "train_0001.arrow"):
        source_arrow = store.paths(first.id).data / name
        (dest / name).write_bytes(source_arrow.read_bytes())
        old = json.loads(chunk_record(source_arrow).read_text())
        old.pop("reserved")                      # a record from before #255
        chunk_record(dest / name).write_text(json.dumps(old), encoding="utf-8")

    Pipeline(store, settings, runner=FakeRunner(),
             source=DocumentSource([RESERVED_DOC, OTHER_DOC, RESERVED_DOC, OTHER_DOC])
             ).execute(second.id)

    resumed = store.load(second.id)
    assert resumed.progress.reserved_pages is None
    assert "incomplete" in (resumed.progress.reserved_pages_source or "")


# ── and the record no longer claims a check that did not happen ─────────────

def test_the_plan_itself_is_never_counted_as_a_page_list(store, settings, held):  # noqa: F811
    """What `prepare` records for a chunked run. It cannot check anything — the
    train side does not exist yet — so it says so, and `compile` overwrites it
    with the real count."""
    from atr_training.chunking import CHUNK_PLAN_FILENAME

    source = DocumentSource([OTHER_DOC] * 4)
    settings = _chunked(settings, chunk_pages=2)
    job = store.create(request_with(force=True))
    pipeline = Pipeline(store, settings, runner=FakeRunner(), source=source)

    train, _val = pipeline._prepare(store.load(job.id))

    assert train.name == CHUNK_PLAN_FILENAME
    mid = store.load(job.id)
    assert mid.progress.reserved_pages is None
    assert "chunked" in (mid.progress.reserved_pages_source or "")
    assert json.loads(train.read_text(encoding="utf-8"))["hf_repo"], (
        "the plan was rewritten as a page list")

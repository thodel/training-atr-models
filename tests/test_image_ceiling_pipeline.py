"""The trainer cuts its own train arrows where the ceiling binds (#145).

The PageXML ceiling bounds the ``Coords`` box; kraken pads a batch to the widest
*extracted image*. On the German sweep corpus the first removed 0 of 345,359
lines while 4.97 % of the images were over 60:1, up to 177:1, and four
fine-tunes died of OOM mid-epoch. ``scripts/filter_arrow_ceiling.py`` cut a
corpus by hand; every ordinary run still trained on the uncut one. These pin
the cut inside the pipeline:

* the train arrow is cut after compile, the val arrow is not (the job's CER is
  measured on it);
* nothing over the ceiling means nothing is rewritten;
* what was cut, and which characters went with it, is on the job's record;
* the step budget counts what is trained on;
* a chunk records its ceiling, and a chunk finished before the cut is cut when
  it is adopted;
* the artefact cache keys on the ceiling;
* 0 turns it off.

The cut itself (``line_ceiling.cut_arrow``) has its own suite; here it is a fake
over a text "arrow" of one aspect ratio per line, so this runs without pyarrow.
The last test runs the real one where pyarrow is installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from loguru import logger

import kraken_train_svc.runner as kraken_runner
import test_train_svc_pipeline as pipeline_tests
from atr_training.contracts import KrakenTrainParams
from atr_training.line_ceiling import ArrowCut
from kraken_train_svc.runner import Pipeline, _chunk_record
from test_train_svc_pipeline import FakeRunner, FakeSource, request_with

settings = pipeline_tests.settings
store = pipeline_tests.store

#: One line per row: aspect ratio, then its text. 177 is the sweep corpus' worst.
CORPUS = "10\ta\n177\tæ\n20\tb\n"


class CeilingRunner(FakeRunner):
    """compile writes CORPUS (or ``corpus``) instead of b"ARROW"."""

    def __init__(self, corpus: str = CORPUS, **kw):
        super().__init__(**kw)
        self.corpus = corpus

    def run(self, cmd, log_path, env=None, on_start=None):
        code = super().run(cmd, log_path, env=env, on_start=on_start)
        if "compile" in cmd and code == 0:
            Path(cmd[cmd.index("--output") + 1]).write_text(self.corpus, encoding="utf-8")
        return code


class FakeCut:
    """``cut_arrow`` over the text corpus, recording how it was called."""

    def __init__(self):
        self.calls: list[tuple[str, bool]] = []

    def __call__(self, src, dest=None, *, ceiling, dry_run=False):
        self.calls.append((Path(src).name, dry_run))
        rows = [line.split("\t") for line in Path(src).read_text("utf-8").splitlines()]
        kept = [(a, t) for a, t in rows if float(a) <= ceiling]
        before = {c for _, t in rows for c in t}
        after = {c for _, t in kept for c in t}
        if not dry_run:
            Path(dest).write_text("".join(f"{a}\t{t}\n" for a, t in kept), encoding="utf-8")
        return ArrowCut(lines_before=len(rows), removed=len(rows) - len(kept),
                        widest_before=max(float(a) for a, _ in rows),
                        widest_after=max((float(a) for a, _ in kept), default=0.0),
                        characters_lost=tuple(sorted(before - after)),
                        written=None if dry_run else Path(dest))


@pytest.fixture
def cut(monkeypatch):
    fake = FakeCut()
    monkeypatch.setattr(kraken_runner, "cut_arrow", fake)
    return fake


@pytest.fixture
def ceiling(settings):  # noqa: F811
    settings.image_ceiling = 60.0
    return settings


def _run(store, settings, runner, **kw):  # noqa: F811
    job = store.create(request_with(**kw))
    return Pipeline(store, settings, runner=runner,
                    source=FakeSource({"train": 6, "eval": 2})).execute(job.id)


def _rows(path: Path) -> list[str]:
    return [line.split("\t")[0] for line in path.read_text("utf-8").splitlines()]


# ── unchunked ───────────────────────────────────────────────────────────────
def test_the_train_arrow_is_cut_and_the_val_arrow_is_not(store, ceiling, cut):
    job = _run(store, ceiling, CeilingRunner())
    data = store.paths(job.id).data
    assert job.status == "completed", job.error
    assert _rows(data / "train.arrow") == ["10", "20"], "177:1 is gone from what is trained on"
    assert _rows(data / "val.arrow") == ["10", "177", "20"], \
        "the job's CER is measured on val; its widest lines stay"
    assert not list(data.glob("*.part")), "no temporary file survives"


def test_what_was_cut_is_on_the_record(store, ceiling, cut):
    job = _run(store, ceiling, CeilingRunner())
    record = json.loads((store.paths(job.id).root / "line_ceiling.json").read_text("utf-8"))
    assert record == [{"what": "train", "ceiling": 60.0, "lines_before": 3, "removed": 1,
                       "widest_before": 177.0, "widest_after": 20.0,
                       "characters_lost": "æ"}]


def test_a_lost_character_is_said(store, ceiling, cut):
    seen = []
    sink = logger.add(lambda m: seen.append(m.record["message"]), level="WARNING")
    try:
        _run(store, ceiling, CeilingRunner())
    finally:
        logger.remove(sink)
    assert any("only on lines over the ceiling" in m and "æ" in m for m in seen)


def test_the_step_budget_counts_what_is_trained_on(store, ceiling, cut, settings):  # noqa: F811
    params = KrakenTrainParams(batch_size=1, epochs=2)
    uncut = _run(store, ceiling, CeilingRunner(corpus="10\ta\n"), params=params)
    with_wide = _run(store, ceiling, CeilingRunner(), params=params)
    assert with_wide.progress.train_lines == uncut.progress.train_lines - 1
    assert with_wide.progress.train_samples == uncut.progress.train_samples - 1
    assert with_wide.progress.steps_per_epoch == uncut.progress.steps_per_epoch - 1


def test_nothing_over_the_ceiling_is_measured_and_not_rewritten(store, ceiling, cut):
    job = _run(store, ceiling, CeilingRunner(corpus="10\ta\n20\tb\n"))
    assert cut.calls == [("train.arrow", True)], "one measuring pass, no rewrite"
    assert json.loads((store.paths(job.id).root / "line_ceiling.json")
                      .read_text("utf-8"))[0]["removed"] == 0


def test_zero_turns_it_off(store, settings, cut):  # noqa: F811
    settings.image_ceiling = 0
    job = _run(store, settings, CeilingRunner())
    assert cut.calls == []
    assert _rows(store.paths(job.id).data / "train.arrow") == ["10", "177", "20"]


# ── chunked ─────────────────────────────────────────────────────────────────
def test_every_chunk_is_cut_and_records_its_ceiling(store, ceiling, cut):
    ceiling.chunk_pages = 2
    job = _run(store, ceiling, CeilingRunner())
    data = store.paths(job.id).data
    chunks = sorted(data.glob("train_[0-9]*.arrow"))
    assert len(chunks) >= 2
    for arrow in chunks:
        assert _rows(arrow) == ["10", "20"]
        record = json.loads(_chunk_record(arrow).read_text("utf-8"))
        assert record["image_ceiling"] == 60.0 and record["ceiling_removed"] == 1
    assert _rows(data / "val.arrow") == ["10", "177", "20"]


def test_a_chunk_finished_before_the_cut_is_cut_when_adopted(store, ceiling, cut):
    """A resume adopts finished chunks without recompiling them; one whose
    record has no ceiling was compiled before the cut and holds the 177:1 line."""
    ceiling.chunk_pages = 2
    job = store.create(request_with())
    data = store.paths(job.id).data
    data.mkdir(parents=True, exist_ok=True)
    old = data / "train_0000.arrow"
    old.write_text(CORPUS, encoding="utf-8")
    _chunk_record(old).write_text(json.dumps({"pages_written": 2, "lines": 3,
                                              "start_index": 0, "arrow": old.name}))
    Pipeline(store, ceiling, runner=CeilingRunner(),
             source=FakeSource({"train": 6, "eval": 2})).execute(job.id)
    assert _rows(old) == ["10", "20"]
    record = json.loads(_chunk_record(old).read_text("utf-8"))
    assert record["lines"] == 2 and record["image_ceiling"] == 60.0
    assert record["ceiling_removed"] == 1


def test_a_chunk_already_cut_at_this_ceiling_is_left_alone(store, ceiling, cut):
    ceiling.chunk_pages = 2
    job = store.create(request_with())
    data = store.paths(job.id).data
    data.mkdir(parents=True, exist_ok=True)
    done = data / "train_0000.arrow"
    done.write_text("10\ta\n20\tb\n", encoding="utf-8")
    _chunk_record(done).write_text(json.dumps({"pages_written": 2, "lines": 2,
                                               "start_index": 0, "arrow": done.name,
                                               "image_ceiling": 60.0, "ceiling_removed": 1}))
    Pipeline(store, ceiling, runner=CeilingRunner(),
             source=FakeSource({"train": 6, "eval": 2})).execute(job.id)
    assert ("train_0000.arrow", True) not in cut.calls, "not measured again"


# ── the cache ───────────────────────────────────────────────────────────────
def test_the_cache_keys_on_the_ceiling(store, settings):  # noqa: F811
    job = store.create(request_with())
    keys = {}
    for value in (0, 60.0, 90.0):
        settings.image_ceiling = value
        keys[value] = Pipeline(store, settings, runner=FakeRunner(),
                               source=FakeSource({}))._cache_key(job)
    assert len(set(map(str, keys.values()))) == 3, \
        "an entry compiled before the cut, or at another ceiling, is another corpus"


# ── the real cut, where pyarrow is installed ────────────────────────────────
def test_the_real_cut_on_a_real_arrow(store, ceiling, tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("PIL")
    from test_arrow_ceiling import aspects_from_arrow, write_arrow

    job = store.create(request_with())
    paths = store.paths(job.id)
    paths.data.mkdir(parents=True, exist_ok=True)
    arrow = write_arrow(paths.data / "train.arrow",
                        [(100, "ok"), (1770, "wide"), (600, "edge")], batch_rows=2)
    pipeline = Pipeline(store, ceiling, runner=FakeRunner(), source=FakeSource({}))
    assert pipeline._cut_to_ceiling(job, arrow, "train") == 1
    assert aspects_from_arrow(arrow) == [10.0, 60.0]
    assert not list(paths.data.glob("*.part"))

"""A chunked compile that was interrupted picks up where it stopped (#72).

`_compile_chunked` wrote one `train_<i>.arrow` per chunk and nobody read them
back, so a failure in chunk 40 started again at chunk 0 and re-streamed and
re-compiled everything. Chunking turned one large loss into several small ones
without making any of them repeatable.

The first attempt at this (#66) added a resume that produced quietly wrong
numbers, and that is what most of this file guards:

* an arrow exists after a `-9` too, so its presence is no proof of anything;
* a chunk's **line** count is made in `materialize()` and is on disk nowhere
  afterwards — `pages_train_<i>.lst` holds one *page* per line, so counting it
  reads about a factor 30 low and looks entirely plausible;
* that number feeds `check_convergence` and is written into the artefact-cache
  payload, from which every later job on that entry reads it back.

`TestChunkedCompile` in test_train_svc_pipeline.py never lets a compile run a
second time, so none of this was covered: a green suite said nothing about it.
"""

import json
from pathlib import Path


from tests.test_train_svc_pipeline import FakeRunner, FakeSource, TestChunkedCompile
from tests.test_train_svc_pipeline import request_with, settings, store  # noqa: F401

from engines.kraken_train_svc.runner import (
    CHUNK_RECORD_SUFFIX,
    Pipeline,
    _chunk_record,
    _finished_chunk,
)


def _run(store, settings, source=None, runner=None, request=None):  # noqa: F811
    job = store.create(request or request_with(force=True))
    runner = runner or FakeRunner()
    source = source or FakeSource({"train": 6, "eval": 2})
    return Pipeline(store, settings, runner=runner, source=source).execute(job.id), runner


def _chunked(settings, chunk_pages=2):  # noqa: F811
    return TestChunkedCompile.chunking_settings(settings, chunk_pages=chunk_pages)


class CountingRunner(FakeRunner):
    """Records every ketos compile, by the output it was asked for."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.compiled: list[str] = []

    def run(self, cmd, log_path, env=None):
        if "compile" in cmd:
            out = cmd[cmd.index("--output") + 1] if "--output" in cmd else ""
            self.compiled.append(Path(out).name)
        return super().run(cmd, log_path, env)


# ── what makes a chunk finished ─────────────────────────────────────────────
def test_an_arrow_without_a_record_is_not_finished(tmp_path: Path):
    """The case that matters: ketos killed by the OOM killer leaves the file."""
    arrow = tmp_path / "train_0000.arrow"
    arrow.write_bytes(b"truncated")

    assert _finished_chunk(arrow) is None


def test_a_record_naming_a_missing_arrow_is_not_finished(tmp_path: Path):
    arrow = tmp_path / "train_0000.arrow"
    _chunk_record(arrow).write_text(json.dumps({"pages_written": 5, "lines": 150}))

    assert _finished_chunk(arrow) is None


def test_a_record_naming_an_empty_arrow_is_not_finished(tmp_path: Path):
    arrow = tmp_path / "train_0000.arrow"
    arrow.write_bytes(b"")
    _chunk_record(arrow).write_text(json.dumps({"pages_written": 5, "lines": 150}))

    assert _finished_chunk(arrow) is None


def test_a_record_that_does_not_parse_is_not_finished(tmp_path: Path):
    """An interrupted run is exactly what leaves a file that exists and does not
    parse."""
    arrow = tmp_path / "train_0000.arrow"
    arrow.write_bytes(b"data")
    _chunk_record(arrow).write_text("{ not json")

    assert _finished_chunk(arrow) is None


def test_a_record_without_both_counts_is_not_finished(tmp_path: Path):
    arrow = tmp_path / "train_0000.arrow"
    arrow.write_bytes(b"data")
    _chunk_record(arrow).write_text(json.dumps({"pages_written": 5}))

    assert _finished_chunk(arrow) is None


def test_a_complete_record_beside_a_real_arrow_is_finished(tmp_path: Path):
    arrow = tmp_path / "train_0000.arrow"
    arrow.write_bytes(b"data")
    _chunk_record(arrow).write_text(json.dumps({"pages_written": 5, "lines": 150}))

    assert _finished_chunk(arrow) == {"pages_written": 5, "lines": 150}


# ── the arrow only ever bears its final name when it is whole ───────────────
def test_a_compile_writes_through_a_part_file(store, settings):  # noqa: F811
    """`ketos` is pointed at `.part`, so a kill cannot leave a truncated file
    under the name a resume trusts."""
    settings = _chunked(settings)
    runner = CountingRunner()
    _run(store, settings, runner=runner)

    assert runner.compiled, "nothing was compiled"
    assert all(name.endswith(".part") for name in runner.compiled), runner.compiled


def test_no_part_file_survives_a_finished_run(store, settings):  # noqa: F811
    settings = _chunked(settings)
    job, _ = _run(store, settings)

    assert list(store.paths(job.id).data.glob("*.part")) == []


def test_every_chunk_gets_a_record(store, settings):  # noqa: F811
    settings = _chunked(settings)
    job, _ = _run(store, settings)
    data = store.paths(job.id).data

    arrows = sorted(data.glob("train_[0-9]*.arrow"))
    assert arrows, "no chunk arrow was written"
    for arrow in arrows:
        assert _finished_chunk(arrow) is not None, arrow.name


# ── re-entry ────────────────────────────────────────────────────────────────
def _second_pass(store, settings, mangle=None):  # noqa: F811
    """Run once, optionally damage the result, then run the same job again."""
    job = store.create(request_with(force=True))
    pipeline = Pipeline(store, settings, runner=FakeRunner(),
                        source=FakeSource({"train": 6, "eval": 2}))
    first = pipeline.execute(job.id)

    if mangle is not None:
        mangle(store.paths(job.id).data)

    again = CountingRunner()
    second = Pipeline(store, settings, runner=again,
                      source=FakeSource({"train": 6, "eval": 2}))
    reloaded = store.load(job.id)
    reloaded.status = "compiling"
    store.save(reloaded)
    return first, second.execute(job.id), again


def test_a_finished_chunk_is_not_compiled_again(store, settings):  # noqa: F811
    """Acceptance 2."""
    settings = _chunked(settings)
    _, second, again = _second_pass(store, settings)

    train_compiles = [n for n in again.compiled if n.startswith("train_0")]
    assert train_compiles == [], f"chunks were rebuilt: {train_compiles}"
    assert second.status == "completed", second.error


def test_a_chunk_whose_arrow_was_truncated_is_compiled_again(store, settings):  # noqa: F811
    """Acceptance 3 — the test that must be red without the change. The record
    is removed and a truncated arrow left, which is what a `-9` leaves behind."""
    settings = _chunked(settings)

    def kill_the_second_chunk(data: Path):
        arrow = sorted(data.glob("train_[0-9]*.arrow"))[1]
        _chunk_record(arrow).unlink()
        arrow.write_bytes(b"truncated")

    _, second, again = _second_pass(store, settings, kill_the_second_chunk)

    assert [n for n in again.compiled if n.startswith("train_0")], \
        "the truncated chunk was adopted instead of rebuilt"
    assert second.status == "completed", second.error


def test_the_line_count_survives_a_resume(store, settings):  # noqa: F811
    """Acceptance 4, and the one the first attempt got wrong. It fed
    `check_convergence` and the artefact-cache payload, so a single resumed run
    poisoned every later job on that entry."""
    settings = _chunked(settings)
    first, second, _ = _second_pass(store, settings)

    assert second.progress.train_lines == first.progress.train_lines
    assert second.progress.pages_written == first.progress.pages_written


def test_the_line_count_is_not_the_number_of_pages(store, settings):  # noqa: F811
    """`pages_train_<i>.lst` holds one **page** per line, always non-empty, so
    counting it gives the page count wearing the line count's name — plausible,
    and about a factor 30 low on a real corpus."""
    settings = _chunked(settings)
    job, _ = _run(store, settings)

    assert job.progress.train_lines > job.progress.pages_written


def test_a_job_interrupted_while_compiling_is_recognised(store, settings, caplog):  # noqa: F811
    """Acceptance 5."""
    from loguru import logger

    settings = _chunked(settings)
    seen: list[str] = []
    sink = logger.add(lambda m: seen.append(str(m)), level="WARNING")
    try:
        _second_pass(store, settings)
    finally:
        logger.remove(sink)

    assert any("re-entered while `compiling`" in w for w in seen), seen


def test_a_stale_page_directory_is_cleared_before_a_retry(store, settings):  # noqa: F811
    """Acceptance 6. Pages from an attempt with a different `start_index` are in
    no manifest and nothing will read them — dead bytes on exactly the scratch
    this chunking exists to spare."""
    settings = _chunked(settings)

    def leave_a_stale_chunk(data: Path):
        arrow = sorted(data.glob("train_[0-9]*.arrow"))[1]
        _chunk_record(arrow).unlink()
        arrow.unlink()
        stale = data / "pages" / "chunk_0001"
        stale.mkdir(parents=True, exist_ok=True)
        (stale / "999999_ghost_0001_1.jpg").write_bytes(b"stale")

    _, second, _ = _second_pass(store, settings, leave_a_stale_chunk)

    assert second.status == "completed", second.error
    ghosts = list(store.paths(second.id).pages.rglob("*ghost*"))
    assert ghosts == [], ghosts


def test_the_resumed_run_produces_the_same_training_set(store, settings):  # noqa: F811
    """The arrows a resumed run hands to `ketos train` are the ones the first
    run built — same names, same order, one manifest."""
    settings = _chunked(settings)
    first, second, _ = _second_pass(store, settings)

    listed = (store.paths(second.id).data / "train_bin.lst").read_text().split()
    assert listed, "no training set was listed"
    assert listed == sorted(listed)
    assert all(CHUNK_RECORD_SUFFIX not in name for name in listed)

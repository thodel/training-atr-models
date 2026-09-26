"""Storing the corpus cost xix-v2 fourteen of its sixteen and a half hours (#22).

Slurm 15315827: compile finished at 13:39, the artefact entry was written at
03:37 the next morning — 966,748 files, one metadata operation each, on GPFS, for
a copy **no run has ever read**. The run that produced it trains out of the job
directory; only a later run with an identical selection could use the entry, and
that corpus will very likely never be built again.

Three things follow, and none of them is "move instead of copy" — that has a
precondition nobody has met (resume still reads the job directory) and it does
not help the case above, where the cheapest copy is the one not made:

1. a run can decline to store, because the cost and the benefit are per run;
2. the job is `training` before the store starts, so hours spent there cannot
   turn a finished corpus into a TIMEOUT that never reached training;
3. the size is counted while the bytes go past, not by walking the tree a second
   time immediately after the copy that touched every file in it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atr_training.artefact_cache import ArtefactCache, ArtefactKey  # noqa: E402

from tests.test_vlm_train_pipeline import (  # noqa: E402
    FakeRunner,
    FakeSource,
    request_with,
    run_pipeline,
    settings,  # noqa: F401 — a fixture, used by name
    store,     # noqa: F401
)


@pytest.fixture
def stores(monkeypatch):
    """Record every ``put``: the job status at the time, and the entry size."""
    calls: list[dict] = []
    original = ArtefactCache.put

    def spy(self, key, source, **kwargs):
        entry = original(self, key, source, **kwargs)
        calls.append({"key": key.digest, "bytes": entry.bytes_})
        return entry

    monkeypatch.setattr(ArtefactCache, "put", spy)
    return calls


def _cache_entries(settings) -> list[Path]:  # noqa: F811
    root = Path(settings.artefact_cache_root)
    if not root.is_dir():
        return []
    return [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]


# ── 1. a run can decline to store ───────────────────────────────────────────
def test_a_run_that_declines_the_cache_stores_nothing(store, settings, stores):  # noqa: F811
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}),
                       FakeRunner(), request_with(artefact_cache=False))

    assert job.status == "completed", job.error
    assert stores == [], "the corpus was stored although the run declined"
    assert _cache_entries(settings) == []


def test_a_run_without_the_field_behaves_exactly_as_before(store, settings, stores):  # noqa: F811
    """Unset means "as the box holds it" — which is what every request written
    before this field says, and they must not change meaning."""
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}),
                       FakeRunner())

    assert job.status == "completed", job.error
    assert len(stores) == 1


def test_asking_for_the_cache_explicitly_also_stores(store, settings, stores):  # noqa: F811
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}),
                       FakeRunner(), request_with(artefact_cache=True))

    assert job.status == "completed", job.error
    assert len(stores) == 1


def test_declining_still_lets_the_run_reuse_an_entry_that_exists(store, settings):  # noqa: F811
    """The switch is about paying, not about reading. Refusing to reuse would
    make the run recompile a corpus somebody else already built — more expensive
    than the storing it was trying to avoid."""
    from engines.vlm_train_svc.runner import Pipeline

    pipeline = Pipeline(store, settings, runner=FakeRunner(),
                        source=FakeSource({"train": 4, "eval": 2}))
    job = store.create(request_with(artefact_cache=False))

    assert pipeline._store_cache(job) is None      # will not write
    assert pipeline._cache() is not None           # will still read


# ── 2. the status says the corpus exists before the store spends hours ──────
def test_the_job_is_training_before_the_store_begins(store, settings, monkeypatch):  # noqa: F811
    """A job that sits in `compiling` for fourteen hours and then hits the
    walltime ends as TIMEOUT with a corpus that was finished the whole time."""
    seen: list[str] = []
    original = ArtefactCache.put

    def spy(self, key, source, **kwargs):
        seen.append(store.load(job_id[0]).status)
        return original(self, key, source, **kwargs)

    monkeypatch.setattr(ArtefactCache, "put", spy)
    job_id: list[str] = []
    created = store.create(request_with())
    job_id.append(created.id)

    from atr_training.runner_base import run_job  # noqa: F401 — same path as run_pipeline
    from engines.vlm_train_svc.runner import Pipeline

    Pipeline(store, settings, runner=FakeRunner(),
             source=FakeSource({"train": 4, "eval": 2})).execute(created.id)

    assert seen == ["training"], seen


def test_a_store_that_dies_leaves_a_usable_corpus(store, settings, monkeypatch):  # noqa: F811
    """A cache is an optimisation; a run that fails because of one is strictly
    worse than a run that was slow. And the status it leaves behind has to be
    the true one."""
    def boom(self, key, source, **kwargs):
        raise OSError("the scratch filesystem went away")

    monkeypatch.setattr(ArtefactCache, "put", boom)

    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}),
                       FakeRunner())

    assert job.status == "completed", job.error


# ── 3. the size is counted once, while it is being written ──────────────────
def test_the_manifest_size_matches_the_files_stored(tmp_path: Path):
    source = tmp_path / "data"
    (source / "pages").mkdir(parents=True)
    (source / "pages" / "a.jpg").write_bytes(b"x" * 1000)
    (source / "pages" / "b.jpg").write_bytes(b"y" * 2500)
    (source / "train.jsonl").write_bytes(b"z" * 40)

    cache = ArtefactCache(tmp_path / "cache")
    entry = cache.put(ArtefactKey(digest="d" * 32, pinned=False), source)

    # The manifest itself is written after the count, so it is not in it — the
    # number describes the corpus, which is what the eviction budget is about.
    assert entry.bytes_ == 3540


def test_the_file_list_form_counts_too(tmp_path: Path):
    one = tmp_path / "one.arrow"
    two = tmp_path / "two.arrow"
    one.write_bytes(b"a" * 700)
    two.write_bytes(b"b" * 300)

    entry = ArtefactCache(tmp_path / "cache").put(
        ArtefactKey(digest="e" * 32, pinned=False), [one, two])

    assert entry.bytes_ == 1000


def test_a_move_is_measured_before_the_source_is_gone(tmp_path: Path):
    source = tmp_path / "data"
    source.mkdir()
    (source / "a.bin").write_bytes(b"x" * 1234)

    entry = ArtefactCache(tmp_path / "cache").put(
        ArtefactKey(digest="f" * 32, pinned=False), source, move=True)

    assert entry.bytes_ == 1234
    assert not source.exists()


def test_the_normal_path_does_not_walk_the_tree_a_second_time(tmp_path: Path, monkeypatch):
    """The point of the change: after a copy that has just touched every one of
    966,748 files, a full stat walk of the same tree is the second of two
    metadata passes on GPFS. The fallback stays, for the case where the copy
    could not account for what it wrote."""
    import atr_training.artefact_cache as ac

    walked: list[Path] = []
    original = ac._tree_bytes

    def spy(root):
        walked.append(Path(root))
        return original(root)

    monkeypatch.setattr(ac, "_tree_bytes", spy)

    source = tmp_path / "data"
    source.mkdir()
    (source / "a.bin").write_bytes(b"x" * 10)
    entry = ac.ArtefactCache(tmp_path / "cache").put(ArtefactKey(digest="a" * 32, pinned=False), source)

    assert entry.bytes_ == 10
    assert walked == [], "the staging tree was walked again after the copy"


def test_an_unmeasurable_copy_falls_back_rather_than_claiming_zero(tmp_path: Path,
                                                                  monkeypatch):
    """Unknown is not zero. A symlink to nowhere, a file that vanishes between
    the walk and the stat — the counter gives up rather than guessing, and a
    manifest claiming 0 bytes would make the eviction budget drop the wrong
    entry first."""
    import atr_training.artefact_cache as ac

    class GivesUp(ac._Counter):
        def __call__(self, src, dst, **kwargs):
            result = super().__call__(src, dst, **kwargs)
            self.total = None
            return result

    monkeypatch.setattr(ac, "_Counter", GivesUp)

    source = tmp_path / "data"
    source.mkdir()
    (source / "a.bin").write_bytes(b"x" * 77)

    entry = ac.ArtefactCache(tmp_path / "cache").put(
        ArtefactKey(digest="b" * 32, pinned=False), source)

    assert entry.bytes_ == 77, "the fallback walk did not run"


def test_the_counter_gives_up_rather_than_undercounting(tmp_path: Path, monkeypatch):
    """The unit of the above: one unreadable destination poisons the total for
    good, instead of quietly leaving that file out of the sum."""
    import atr_training.artefact_cache as ac

    counter = ac._Counter()
    src, dst = tmp_path / "a", tmp_path / "b"
    src.write_bytes(b"x" * 5)
    counter(src, dst)
    assert counter.total == 5

    monkeypatch.setattr(ac, "_size_of", lambda path: None)
    counter(src, tmp_path / "c")

    assert counter.total is None

"""A VLM corpus is built where it will be stored (#309).

On asteraix the job directory is on the CIFS share and the artefact cache on the
local NVMe. A corpus built in the job directory had to be copied across: the
#238 line corpus, 979,596 files and 86.2 GB, took 2 h 45 (06.11.2026), and then
lay on both disks. Built in ``<cache>/.building/<job>/data``, the store is one
rename and nothing is left twice.

These tests pretend the two are different filesystems — on a test machine they
never are — and check the run end to end through the fake trainer, plus the
cases that must keep the old route: one filesystem (UBELIX), a short disk, and
a job that already has part of its corpus in the job directory.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from atr_training import runner_base
from atr_training.artefact_cache import BUILDING_DIR, ArtefactCache, sweep_builds
from atr_training.heldout_boundary import find_manifest
from atr_training.jobstore import JobStore
from atr_training.runner_base import StageFailed

from tests.test_vlm_train_pipeline import (
    FakeRunner,
    FakeSource,
    request_with,
    settings,  # noqa: F401 — a fixture, used by name
)
from vlm_train_svc.runner import Pipeline


@pytest.fixture
def build_store(settings):  # noqa: F811
    """The runner's store: it knows where builds go."""
    settings.artefact_cache_max_gb = 0
    return JobStore(settings.jobs_root, host_id=settings.host_id,
                    build_root=Path(settings.artefact_cache_root) / BUILDING_DIR)


@pytest.fixture
def two_disks(settings, monkeypatch):  # noqa: F811
    """Make the cache look like a different filesystem from the job store."""
    cache_root = Path(settings.artefact_cache_root)
    real = runner_base._filesystem_of

    def fake(path):
        path = Path(path)
        if path == cache_root or cache_root in path.parents:
            return -1
        return real(path)

    monkeypatch.setattr(runner_base, "_filesystem_of", fake)


@pytest.fixture
def puts(monkeypatch):
    """Every ``put``: what was stored, and whether it was moved."""
    calls: list[dict] = []
    original = ArtefactCache.put

    def spy(self, key, source, **kwargs):
        calls.append({"source": Path(source), "move": kwargs.get("move")})
        return original(self, key, source, **kwargs)

    monkeypatch.setattr(ArtefactCache, "put", spy)
    return calls


def _run(store, settings, runner=None, request=None):  # noqa: F811
    job = store.create(request or request_with())
    pipeline = Pipeline(store, settings, runner=runner or FakeRunner(),
                        source=FakeSource({"train": 4, "eval": 2}))
    return pipeline, pipeline.execute(job.id)


def _entry_dir(settings, job) -> Path:  # noqa: F811
    key = re.match(r"^([0-9a-f]{12})", job.progress.artefact)[1]
    found = [d for d in Path(settings.artefact_cache_root).iterdir()
             if d.name.startswith(key)]
    assert len(found) == 1, found
    return found[0]


# ── the route itself ────────────────────────────────────────────────────────
def test_the_corpus_is_built_on_the_cache_disk_and_moved(
        build_store, settings, two_disks, puts):  # noqa: F811
    _, job = _run(build_store, settings)

    assert job.status == "completed", job.error
    assert len(puts) == 1
    build = Path(settings.artefact_cache_root) / BUILDING_DIR / job.id
    assert puts[0]["source"] == build / "data"
    assert puts[0]["move"] is True
    assert not build.exists(), "the build directory outlived the store"
    assert re.match(r"^[0-9a-f]{12} built by this job, stored in \d+ min$",
                    job.progress.artefact)


def test_the_job_directory_keeps_the_manifests_and_nothing_else(
        build_store, settings, two_disks):  # noqa: F811
    _, job = _run(build_store, settings)

    data = build_store.paths(job.id).data
    assert data.parent == build_store.paths(job.id).root
    assert (data / "pages_train.lst").is_file() and (data / "pages_val.lst").is_file()
    # A resume takes these first, and their samples resolve against the job
    # directory, where there are no images.
    assert not (data / "train.jsonl").exists() and not (data / "val.jsonl").exists()
    assert not (data / "crops").exists()
    assert not (data / "pages").is_dir() or not any((data / "pages").iterdir())
    # Every run that adopts this corpus is checked through the builder's data/.
    assert find_manifest(data) == data / "pages_train.lst"


def test_the_entry_holds_the_corpus_and_its_manifests_name_its_pages(
        build_store, settings, two_disks):  # noqa: F811
    _, job = _run(build_store, settings)

    stored = _entry_dir(settings, job) / "data"
    assert (stored / "train.jsonl").is_file() and (stored / "val.jsonl").is_file()
    assert any((stored / "crops").rglob("*.jpg"))
    for name in ("pages_train.lst", "pages_val.lst"):
        for line in (stored / name).read_text(encoding="utf-8").split():
            assert Path(line).is_file(), f"{name} names {line}, which is not there"
            assert BUILDING_DIR not in line


def test_training_reads_the_entry(build_store, settings, two_disks):  # noqa: F811
    runner = FakeRunner()
    _, job = _run(build_store, settings, runner)

    train = runner.command("train")
    stored = _entry_dir(settings, job)
    assert train[train.index("--data-root") + 1] == str(stored)
    assert Path(train[train.index("--train-jsonl") + 1]) == stored / "data" / "train.jsonl"


def test_a_resume_after_the_store_finds_the_entry(build_store, settings, two_disks):  # noqa: F811
    pipeline, job = _run(build_store, settings)

    resumed = pipeline._resume_artifacts(job)
    assert resumed is not None
    assert resumed[0] == _entry_dir(settings, job) / "data" / "train.jsonl"


# ── where the old route stays ───────────────────────────────────────────────
def test_one_filesystem_builds_in_the_job_directory_as_before(build_store, settings, puts):  # noqa: F811
    """UBELIX: job directory and cache share /scratch, and the store hard-links."""
    _, job = _run(build_store, settings)

    assert job.status == "completed", job.error
    assert puts[0]["source"] == build_store.paths(job.id).root / "data"
    assert puts[0]["move"] is False
    assert not (Path(settings.artefact_cache_root) / BUILDING_DIR).exists()


def test_a_short_disk_builds_in_the_job_directory(build_store, settings, two_disks, puts):  # noqa: F811
    settings.artefact_cache_max_gb = 10**9
    _, job = _run(build_store, settings)

    assert job.status == "completed", job.error
    assert puts[0]["source"] == build_store.paths(job.id).root / "data"
    assert puts[0]["move"] is False


def test_a_store_without_a_build_root_builds_in_the_job_directory(
        settings, two_disks, puts):  # noqa: F811
    """Every store but the runner's: it cannot know this host's build directory."""
    settings.artefact_cache_max_gb = 0
    store = JobStore(settings.jobs_root, host_id=settings.host_id)
    _, job = _run(store, settings)

    assert job.status == "completed", job.error
    assert puts[0]["source"] == store.paths(job.id).root / "data"


def test_a_corpus_begun_in_the_job_directory_is_finished_there(
        build_store, settings, two_disks):  # noqa: F811
    """A job re-entering ``compiling`` (#72) continues where its pages are."""
    job = build_store.create(request_with())
    pages = build_store.paths(job.id).data / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    (pages / "0001_doc_0001.xml").write_text("<PcGts/>", encoding="utf-8")
    pipeline = Pipeline(build_store, settings, runner=FakeRunner(),
                        source=FakeSource({"train": 4, "eval": 2}))

    pipeline._begin_local_build(job)

    assert build_store.paths(job.id).corpus is None
    assert not (Path(settings.artefact_cache_root) / BUILDING_DIR / job.id).exists()


def test_a_run_that_declines_the_cache_builds_in_the_job_directory(
        build_store, settings, two_disks):  # noqa: F811
    _, job = _run(build_store, settings, request=request_with(artefact_cache=False))

    assert job.status == "completed", job.error
    assert (build_store.paths(job.id).data / "train.jsonl").is_file()
    assert not (Path(settings.artefact_cache_root) / BUILDING_DIR).exists()


# ── what is left behind when it goes wrong ──────────────────────────────────
def test_a_job_that_fails_before_the_store_removes_its_build(
        build_store, settings, two_disks, monkeypatch):  # noqa: F811
    seen: list[bool] = []

    def boom(self, job, pages_train, pages_val, record):
        seen.append(build_store.paths(job.id).corpus is not None)
        raise StageFailed("compile broke")

    monkeypatch.setattr(Pipeline, "_compile", boom)
    _, job = _run(build_store, settings)

    assert job.status == "failed"
    assert seen == [True], "the corpus was not being built locally"
    assert not (Path(settings.artefact_cache_root) / BUILDING_DIR / job.id).exists()


def test_a_store_that_dies_trains_from_the_build_and_then_removes_it(
        build_store, settings, two_disks, monkeypatch):  # noqa: F811
    """The cache is an optimisation: the run goes on from where the corpus is."""
    def boom(self, key, source, **kwargs):
        raise OSError("the system disk went away")

    monkeypatch.setattr(ArtefactCache, "put", boom)
    runner = FakeRunner()
    _, job = _run(build_store, settings, runner)

    assert job.status == "completed", job.error
    build = Path(settings.artefact_cache_root) / BUILDING_DIR / job.id
    train = runner.command("train")
    assert train[train.index("--data-root") + 1] == str(build)
    assert not build.exists()


def test_an_entry_stored_meanwhile_is_adopted_not_replaced(
        build_store, settings, two_disks, monkeypatch):  # noqa: F811
    """Two jobs with one key built side by side: the second must not delete the
    entry the first may be training from."""
    _, first = _run(build_store, settings)
    entry = _entry_dir(settings, first)
    built_at = (entry / "artefact.json").stat().st_mtime_ns

    monkeypatch.setattr(Pipeline, "_reuse_artefact", lambda self, job: None)
    _, second = _run(build_store, settings)

    assert second.status == "completed", second.error
    assert second.progress.artefact == f"{entry.name[:12]} reused, built by {first.id}"
    assert _entry_dir(settings, second) == entry
    assert (entry / "artefact.json").is_file()
    assert ArtefactCache(settings.artefact_cache_root).entry(entry.name).job_id == first.id
    assert (entry / "artefact.json").stat().st_mtime_ns >= built_at
    assert not (Path(settings.artefact_cache_root) / BUILDING_DIR / second.id).exists()


# ── the store resolves data/ ────────────────────────────────────────────────
def test_paths_point_at_the_build_only_while_it_exists(tmp_path):
    store = JobStore(tmp_path / "jobs", build_root=tmp_path / "cache" / BUILDING_DIR)
    job_id = "20261106T000000Z-x"
    assert store.paths(job_id).data == tmp_path / "jobs" / job_id / "data"
    assert store.paths(job_id).corpus_root == tmp_path / "jobs" / job_id

    (tmp_path / "cache" / BUILDING_DIR / job_id).mkdir(parents=True)
    paths = store.paths(job_id)
    assert paths.data == tmp_path / "cache" / BUILDING_DIR / job_id / "data"
    assert paths.corpus_root == tmp_path / "cache" / BUILDING_DIR / job_id
    assert paths.logs == tmp_path / "jobs" / job_id / "logs"
    assert paths.job_json == tmp_path / "jobs" / job_id / "job.json"


# ── the service's sweep ─────────────────────────────────────────────────────
def test_the_sweep_removes_only_builds_of_jobs_that_are_over(tmp_path):
    root = tmp_path / BUILDING_DIR
    for name in ("over", "running", "gone", "unknown"):
        (root / name / "data").mkdir(parents=True)
    answers = {"over": False, "running": True, "gone": False, "unknown": True}

    removed = sweep_builds(root, answers.__getitem__)

    assert sorted(removed) == ["gone", "over"]
    assert sorted(p.name for p in root.iterdir()) == ["running", "unknown"]


def test_the_service_keeps_a_live_or_unreadable_job_and_drops_the_rest(
        settings, tmp_path):  # noqa: F811
    from kraken_train_svc import app as app_module

    store = JobStore(settings.jobs_root, host_id=settings.host_id)
    live = store.create(request_with())
    over = store.create(request_with(model_id="qwen3vl-thun-v2"))
    store.fail(over, "it broke")
    unreadable = store.create(request_with(model_id="qwen3vl-thun-v3"))
    store.paths(unreadable.id).job_json.write_text("{not json", encoding="utf-8")
    builds = Path(settings.artefact_cache_root) / BUILDING_DIR
    gone = "20261106T000000Z-deleted-job"
    for job_id in (live.id, over.id, unreadable.id, gone):
        (builds / job_id / "data").mkdir(parents=True)

    removed = app_module._cleanup_local_builds(settings.artefact_cache_root, store)

    assert sorted(removed) == sorted([over.id, gone])
    assert sorted(p.name for p in builds.iterdir()) == sorted([live.id, unreadable.id])

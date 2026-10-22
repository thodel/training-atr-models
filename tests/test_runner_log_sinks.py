"""#21: ``runner.log`` is written where a share failure cannot take it.

``prepare`` and the VLM ``compile`` run in-process and leave nothing but
``runner.log``; ``run_job`` hung that one loguru sink under the jobs root — on
asteraix the share — so the network failure that killed an 11½-hour prepare was
the one that could not keep its reason. Now two sinks: the local one beside the
checkpoint root is the file that has to survive, the share one is best effort,
and where the two paths coincide there is one file and every line once.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from loguru import logger

from atr_training.jobstore import JobStore
from atr_training.runner_base import (
    SubprocessRunner,
    install_runner_log_sinks,
    local_log_path,
)
from atr_training.settings import TrainerSettings
from kraken_train_svc.runner import Pipeline

from test_train_svc_pipeline import FakeRunner, request_with


def _settings(tmp_path: Path, *, jobs: str = "share", checkpoints: str = "nvme") -> TrainerSettings:
    (tmp_path / "registry").mkdir(exist_ok=True)
    return TrainerSettings(
        jobs_root=tmp_path / jobs, trained_root=tmp_path / "trained",
        registry_root=tmp_path / "registry", checkpoint_root=tmp_path / checkpoints,
        ketos=tmp_path / "ketos", min_free_disk_gb=0.0, gpu=1,
        artefact_cache=False, artefact_cache_root=tmp_path / "artefacts")


@pytest.fixture
def sinks():
    """Removes every sink a test installs, whatever happens in between."""
    installed: list = []
    yield installed
    for runner_log in installed:
        for sink_id in runner_log.sink_ids:
            try:
                logger.remove(sink_id)
            except ValueError:
                pass


def _install(settings, store, job_id, sinks):
    runner_log = install_runner_log_sinks(store, job_id, settings)
    sinks.append(runner_log)
    return runner_log


# ── the rule, shared with the stage log ─────────────────────────────────────

def test_the_local_path_mirrors_the_share_path_under_the_local_root(tmp_path):
    share, local = tmp_path / "share", tmp_path / "nvme"
    log = share / "jobs" / "x" / "logs" / "runner.log"
    assert local_log_path(log, local, share) == local / "jobs" / "x" / "logs" / "runner.log"
    assert local_log_path(log, share, share) is None, "same path: one file"
    assert local_log_path(tmp_path / "elsewhere" / "runner.log", local, share) is None
    assert local_log_path(log, None, share) is None and local_log_path(log, local, None) is None


def test_the_stage_log_follows_the_same_rule(tmp_path):
    share, local = tmp_path / "share", tmp_path / "nvme"
    runner = SubprocessRunner(local_root=local, share_root=share)
    for log in (share / "j" / "logs" / "train.log", share / "x", tmp_path / "other" / "train.log"):
        assert runner._local_path(log) == local_log_path(log, local, share)


# ── two sinks ───────────────────────────────────────────────────────────────

def test_a_line_lands_in_both_files_when_the_roots_differ(tmp_path, sinks):
    settings = _settings(tmp_path)
    store = JobStore(settings.jobs_root, host_id=settings.host_id)
    job = store.create(request_with())

    runner_log = _install(settings, store, job.id, sinks)
    logger.info("prepare: streaming pages 37/1640")

    assert runner_log.path == tmp_path / "nvme" / job.id / "logs" / "runner.log"
    assert runner_log.mirror == store.paths(job.id).logs / "runner.log"
    assert "streaming pages 37/1640" in runner_log.path.read_text(encoding="utf-8")
    assert "streaming pages 37/1640" in runner_log.mirror.read_text(encoding="utf-8")


def test_a_share_that_cannot_be_opened_does_not_stop_the_run(tmp_path, sinks):
    """The share is gone at start — here: a directory sits where the file
    would be, so it cannot be opened as a file, root or not (a chmod would not
    stop root, and CI runs as root). Setting up must not raise, and the line
    is in the local file."""
    settings = _settings(tmp_path)
    store = JobStore(settings.jobs_root, host_id=settings.host_id)
    job = store.create(request_with())
    (store.paths(job.id).logs / "runner.log").mkdir(parents=True, exist_ok=True)

    runner_log = _install(settings, store, job.id, sinks)
    logger.info("prepare: DatasetGenerationError while generating the dataset")

    assert runner_log.mirror is None and len(runner_log.sink_ids) == 1
    assert "DatasetGenerationError" in runner_log.path.read_text(encoding="utf-8")


def test_coinciding_roots_give_one_file_and_every_line_once(tmp_path, sinks):
    """A box whose jobs root is local: not two handles on one file, not two
    copies of each line — the mistake the stage log's first cut made."""
    settings = _settings(tmp_path, jobs="local", checkpoints="local")
    store = JobStore(settings.jobs_root, host_id=settings.host_id)
    job = store.create(request_with())

    runner_log = _install(settings, store, job.id, sinks)
    logger.info("once-only-marker")

    assert runner_log.mirror is None and len(runner_log.sink_ids) == 1
    assert runner_log.path == store.paths(job.id).logs / "runner.log"
    assert runner_log.path.read_text(encoding="utf-8").count("once-only-marker") == 1


def test_a_second_install_for_the_same_job_is_a_second_pair_of_sinks(tmp_path, sinks):
    """Not a requirement, a boundary: the installer does not deduplicate across
    calls. run_job calls it once; a test that calls it twice removes both."""
    settings = _settings(tmp_path)
    store = JobStore(settings.jobs_root, host_id=settings.host_id)
    job = store.create(request_with())
    first = _install(settings, store, job.id, sinks)
    second = _install(settings, store, job.id, sinks)
    assert set(first.sink_ids).isdisjoint(second.sink_ids)


# ── the failure log comes from where it was written ─────────────────────────

def _failing_prepare_pipeline(settings, store):
    class Exploding:
        def stream(self, hf_repo, data_files, revision=None):
            raise RuntimeError("An error occurred while generating the dataset")
            yield  # pragma: no cover

    return Pipeline(store, settings, runner=FakeRunner(), source=Exploding())


def test_a_failed_prepare_brings_its_reason_from_the_local_runner_log(tmp_path):
    """The share copy is missing — the case the issue is about — and the local
    one holds the traceback. The record must carry it."""
    settings = _settings(tmp_path)
    store = JobStore(settings.jobs_root, host_id=settings.host_id)
    job = store.create(request_with())
    local = tmp_path / "nvme" / job.id / "logs" / "runner.log"
    local.parent.mkdir(parents=True)
    local.write_text("line 1\nDatasetGenerationError: the real reason\n", encoding="utf-8")

    done = _failing_prepare_pipeline(settings, store).execute(job.id)

    assert done.status == "failed"
    assert any("the real reason" in line for line in done.log_tail), done.log_tail


def test_a_readable_share_copy_is_still_preferred(tmp_path):
    """The counter-test: with the share copy there, nothing changes."""
    settings = _settings(tmp_path)
    store = JobStore(settings.jobs_root, host_id=settings.host_id)
    job = store.create(request_with())
    share = store.paths(job.id).logs / "runner.log"
    share.parent.mkdir(parents=True, exist_ok=True)
    share.write_text("share copy: the reason\n", encoding="utf-8")
    local = tmp_path / "nvme" / job.id / "logs" / "runner.log"
    local.parent.mkdir(parents=True)
    local.write_text("local copy: stale\n", encoding="utf-8")

    done = _failing_prepare_pipeline(settings, store).execute(job.id)

    assert any("share copy" in line for line in done.log_tail)
    assert not any("local copy" in line for line in done.log_tail)


def test_run_job_installs_the_sinks_through_the_function():
    """run_job is `pragma: no cover`; what it must call is read off its source."""
    import ast
    import inspect

    from atr_training import runner_base
    tree = ast.parse(inspect.getsource(runner_base.run_job))
    calls = {node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "install_runner_log_sinks" in calls
    attrs = {node.func.attr for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert "add" not in attrs, "run_job still adds a sink of its own"

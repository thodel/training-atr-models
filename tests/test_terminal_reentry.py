"""#153: a requeue that lands after the run ended is not a failure.

Job 16298912 (``20260926T080902Z-ladder-xix-gemma4-e4b``) finished on its
sixteenth attempt — ``status: completed``, CER 11.1 % over 200 samples, adapter
on disk — and Slurm's ``--requeue`` started a seventeenth fourteen hours later.
That attempt fell through ``execute()`` to ``advance(job, "preparing")``, threw
``IllegalTransition``, threw it again from the failure path that tried to set
``failed``, and ``sacct`` recorded a week of GPU as FAILED.

``execute()`` had a re-entry branch for ``training`` (a resume) and one for
``compiling`` (a recompile), and none for a record that is already done. These
tests enter it with one and pin that nothing is written and the finished job is
what comes back — so ``run_job`` exits 0 on it, as it did on attempt sixteen.
"""

from __future__ import annotations

import signal

import pytest

from atr_training.jobstore import JobStore
from atr_training.registration import read_registration
from atr_training.runner_base import run_job
from atr_training.settings import TrainerSettings
from kraken_train_svc.runner import Pipeline

from test_train_svc_pipeline import FakeRunner, FakeSource, request_with, run_pipeline

MODEL = "kraken-thun-missiven-v1"


@pytest.fixture
def settings(tmp_path) -> TrainerSettings:
    (tmp_path / "registry").mkdir()
    return TrainerSettings(
        jobs_root=tmp_path / "training", trained_root=tmp_path / "trained",
        registry_root=tmp_path / "registry",
        checkpoint_root=tmp_path / "local-scratch" / "checkpoints",
        ketos=tmp_path / "ketos", min_free_disk_gb=0.0, gpu=1,
        artefact_cache=False, artefact_cache_root=tmp_path / "artefacts",
    )


@pytest.fixture
def store(settings) -> JobStore:
    return JobStore(settings.jobs_root, host_id=settings.host_id)


def _ended(store, settings, how: str):
    """A record in the terminal state *how*, the way a real run leaves it."""
    if how == "completed":
        job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}), FakeRunner())
    elif how == "failed":
        job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}),
                           FakeRunner(fail_on="train"))
    else:
        job = store.create(request_with())
        store.advance(job, "preparing")
        job = store.advance(job, "cancelled")
    assert job.status == how, job.error
    return job


def _record_bytes(store, job_id: str) -> bytes:
    return store.paths(job_id).job_json.read_bytes()


# ── the seventeenth attempt ─────────────────────────────────────────────────

@pytest.mark.parametrize("how", ["completed", "failed", "cancelled"])
def test_re_entering_a_finished_job_changes_nothing_and_returns_it(store, settings, how):
    ended = _ended(store, settings, how)
    before = _record_bytes(store, ended.id)
    runner, source = FakeRunner(), FakeSource({"train": 4, "eval": 2})

    again = Pipeline(store, settings, runner=runner, source=source).execute(ended.id)

    assert again.status == how
    assert again.id == ended.id
    assert _record_bytes(store, ended.id) == before, (
        "the record was rewritten — a finished job must not even gain a pid")
    assert runner.commands == [], "a stage ran on a finished job"


def test_a_requeue_after_completion_keeps_the_result_and_the_registration(store, settings):
    """The case that cost a week: the metrics and the registration from attempt
    sixteen are what attempt seventeen hands back."""
    ended = _ended(store, settings, "completed")

    again = Pipeline(store, settings, runner=FakeRunner(),
                     source=FakeSource({"train": 4, "eval": 2})).execute(ended.id)

    assert again.metrics is not None and again.metrics.cer == ended.metrics.cer
    assert read_registration(settings.registry_root, MODEL) is not None
    assert again.error is None


def test_a_prepare_attempt_onto_a_finished_record_is_no_error_either(store, settings):
    """`--stop-after compile` enters through the same door (#153, the second
    `advance(job, "preparing")`)."""
    ended = _ended(store, settings, "completed")
    before = _record_bytes(store, ended.id)

    again = Pipeline(store, settings, runner=FakeRunner(),
                     source=FakeSource({"train": 4, "eval": 2})).execute(
        ended.id, stop_after="compile")

    assert again.status == "completed"
    assert _record_bytes(store, ended.id) == before


# ── the failure path does not touch a finished record ───────────────────────

def test_a_failure_after_the_record_was_written_does_not_rewrite_it(store, settings, monkeypatch):
    """`_finish` wrote `completed`; then releasing the artefact claim blew up.
    The old except branch called `store.fail`, which `advance` refuses for a
    terminal record — the second IllegalTransition in 16298912's log."""
    boom = RuntimeError("share vanished while releasing the claim")

    def release_fails(self, job):
        if job.is_terminal:
            raise boom

    monkeypatch.setattr(Pipeline, "_release_artefact", release_fails)
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}), FakeRunner())

    assert job.status == "completed", job.error
    assert store.load(job.id).status == "completed"
    assert job.error is None


def test_a_failure_before_the_record_is_terminal_still_fails_it(store, settings, monkeypatch):
    """The counter-test: the guard is for terminal records only. A stage that
    raises on a live job lands on the record as before."""
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}),
                       FakeRunner(fail_on="train"))
    assert job.status == "failed"
    assert "train" in (job.error or "")


# ── what Slurm sees ─────────────────────────────────────────────────────────

def _exit_code(settings, job_id: str, *extra: str) -> int:
    """`run_job` as the batch script calls it, on a record already on disk."""
    try:
        return run_job(Pipeline, "kraken", ["--root", str(settings.jobs_root),
                                            "--job-id", job_id, *extra])
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)


def test_the_requeued_attempt_exits_zero_on_a_completed_record(store, settings, monkeypatch):
    """`sacct` records the exit code; 0 is what attempt sixteen earned."""
    ended = _ended(store, settings, "completed")
    monkeypatch.setenv("ATR_TRAIN_JOBS_ROOT", str(settings.jobs_root))

    assert _exit_code(settings, ended.id) == 0
    assert _exit_code(settings, ended.id, "--stop-after", "compile") == 0


def test_the_requeued_attempt_exits_one_on_a_failed_record(store, settings, monkeypatch):
    ended = _ended(store, settings, "failed")
    monkeypatch.setenv("ATR_TRAIN_JOBS_ROOT", str(settings.jobs_root))

    assert _exit_code(settings, ended.id) == 1

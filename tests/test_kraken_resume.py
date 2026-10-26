"""A kraken run that was interrupted continues where it stopped (#171).

UBELIX gives a GPU job 24 hours and takes the node back when it likes, and the
asteraix service is restarted now and then. Until now a kraken job re-entered
while `training` called ``ketos train`` afresh: epoch 0, into the same
checkpoint directory. Three things pin the way back:

* kraken 7.0.2 has ``--resume CKPT`` (model, optimizer, epoch, data module from
  the checkpoint) and refuses it beside ``--load``;
* the checkpoint is the newest one, and ``checkpoint_abort.ckpt`` — what kraken
  writes when Lightning turns a SIGTERM into ``SIGTERMException`` — only if it
  is newer than the epoch checkpoints;
* the SIGTERM has to reach ``ketos`` at all. It runs in a session of its own, so
  neither the service's cancel (``killpg`` on the runner's group) nor Slurm's
  signal to the batch script got there: the runner raised ``Cancelled`` or
  ``Preempted`` and ``ketos`` trained on, holding the card, with nobody left to
  report it — the orphan of #24. The runner now passes the stop on.

And the two-job UBELIX path (compile on a CPU node, train on a GPU node) needs
the train stage to find the corpus in the job's own directory, not only in the
artefact cache.
"""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

import pytest

import kraken_train_svc.runner as kraken_runner
import test_train_svc_pipeline as pipeline_tests
from atr_training import runner_base
from atr_training.contracts import KrakenTrainParams
from atr_training.ketos_cmd import (
    ABORT_CHECKPOINT,
    KetosCommandError,
    resume_point,
    train_cmd,
)
from atr_training.runner_base import Cancelled, Preempted, SubprocessRunner
from kraken_train_svc.runner import Pipeline
from test_train_svc_pipeline import FakeRunner, FakeSource, request_with

settings = pipeline_tests.settings
store = pipeline_tests.store


# ── the command ─────────────────────────────────────────────────────────────
def _cmd(**kw):
    return train_cmd("/opt/kraken-train/bin/ketos", params=KrakenTrainParams(),
                     training_manifest="t.lst", evaluation_manifest="v.lst",
                     checkpoint_dir="ckpt", **kw)


def test_a_resumed_run_passes_its_checkpoint_and_neither_load_nor_spec():
    cmd = _cmd(resume="ckpt/checkpoint_04-0.8100.ckpt")
    assert cmd[cmd.index("--resume") + 1] == "ckpt/checkpoint_04-0.8100.ckpt"
    assert "--load" not in cmd and "--spec" not in cmd and "--resize" not in cmd


def test_resume_beside_load_is_refused_as_ketos_would():
    with pytest.raises(KetosCommandError, match="mutually exclusive"):
        _cmd(resume="a.ckpt", load="base.mlmodel")


def test_without_a_checkpoint_the_command_is_what_it_was():
    assert "--resume" not in _cmd() and "--spec" in _cmd()
    loaded = _cmd(load="base.mlmodel")
    assert "--load" in loaded and "--resume" not in loaded


# ── which checkpoint ────────────────────────────────────────────────────────
def _touch(path: Path, at: float) -> Path:
    path.write_bytes(b"ckpt")
    os.utime(path, (at, at))
    return path


def test_no_checkpoint_means_a_fresh_start(tmp_path):
    assert resume_point(tmp_path) is None
    (tmp_path / "best_0.9000.mlmodel").write_bytes(b"w")
    assert resume_point(tmp_path) is None, "weights are not a checkpoint"


def test_the_newest_epoch_checkpoint_is_the_resume_point(tmp_path):
    _touch(tmp_path / "checkpoint_03-0.7000.ckpt", 1000)
    newest = _touch(tmp_path / "checkpoint_02-0.7500.ckpt", 2000)
    assert resume_point(tmp_path) == newest, "by time, not by the best metric or the name"


def test_a_fresh_abort_checkpoint_wins_a_stale_one_does_not(tmp_path):
    _touch(tmp_path / "checkpoint_05-0.8000.ckpt", 2000)
    abort = _touch(tmp_path / ABORT_CHECKPOINT, 3000)
    assert resume_point(tmp_path) == abort
    _touch(abort, 1000)
    assert resume_point(tmp_path).name == "checkpoint_05-0.8000.ckpt", \
        "an abort from an earlier interruption the run has trained past since"


# ── the pipeline ────────────────────────────────────────────────────────────
def _resumable_job(store, settings, **kw):
    """A job in `training` with its corpus and an interrupted attempt's checkpoints."""
    job = store.create(request_with(**kw))
    data = store.paths(job.id).data
    data.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val"):
        (data / f"{name}.arrow").write_bytes(b"ARROW")
        (data / f"{name}_bin.lst").write_text(str(data / f"{name}.arrow") + "\n")
    job.status = "training"
    store.save(job)
    ckpt = settings.checkpoint_root / job.id
    ckpt.mkdir(parents=True, exist_ok=True)
    return job, ckpt


def test_a_requeued_job_resumes_from_its_checkpoint(store, settings):
    """The UBELIX case: compiled by one job, trained by another, interrupted, requeued."""
    job, ckpt = _resumable_job(store, settings)
    _touch(ckpt / "checkpoint_02-0.6000.ckpt", time.time() - 60)
    runner = FakeRunner()
    done = Pipeline(store, settings, runner=runner, source=FakeSource({})).execute(job.id)
    (train,) = runner.commands_named("train")
    assert train[train.index("--resume") + 1] == str(ckpt / "checkpoint_02-0.6000.ckpt")
    assert "--load" not in train
    assert done.status == "completed", done.error
    assert runner.commands_named("compile") == [], "the corpus is not rebuilt"


def test_a_requeued_fine_tune_resumes_instead_of_loading_its_base_again(store, settings,
                                                                          monkeypatch):
    job, ckpt = _resumable_job(store, settings, base_model="kraken-catmus_medieval")
    _touch(ckpt / ABORT_CHECKPOINT, time.time())
    monkeypatch.setattr(Pipeline, "_resolve_base_model",
                        lambda self, base: pytest.fail("the base is in the checkpoint"))
    runner = FakeRunner()
    Pipeline(store, settings, runner=runner, source=FakeSource({})).execute(job.id)
    (train,) = runner.commands_named("train")
    assert train[train.index("--resume") + 1].endswith(ABORT_CHECKPOINT)


def test_the_corpus_is_found_in_the_jobs_own_directory_without_the_cache(store, settings):
    """settings.artefact_cache is off here: the only place the corpus can be is
    the job directory, where `--stop-after compile` left it."""
    assert settings.artefact_cache is False
    job, _ = _resumable_job(store, settings)
    pipeline = Pipeline(store, settings, runner=FakeRunner(), source=FakeSource({}))
    data = store.paths(job.id).data
    assert pipeline._resume_artifacts(job) == (data / "train_bin.lst", data / "val_bin.lst")


def test_manifests_naming_a_purged_arrow_are_not_a_corpus(store, settings):
    job, _ = _resumable_job(store, settings)
    (store.paths(job.id).data / "train.arrow").unlink()
    pipeline = Pipeline(store, settings, runner=FakeRunner(), source=FakeSource({}))
    assert pipeline._resume_artifacts(job) is None, "and with no cache there is nothing else"


# ── the stop reaches the child ──────────────────────────────────────────────
#: Writes a marker on SIGTERM and leaves; with "stubborn" it ignores SIGTERM.
CHILD = """import signal, sys, time
marker = sys.argv[1]
def leave(*_):
    open(marker, "w").write("TERM")
    sys.exit(0)
signal.signal(signal.SIGTERM, signal.SIG_IGN if sys.argv[2] == "stubborn" else leave)
open(marker + ".started", "w").close()
time.sleep(60)
"""


def _raise_in(seconds: float, exc_type):
    def handler(*_):
        raise exc_type()
    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    return previous


@pytest.mark.parametrize("exc_type", [Cancelled, Preempted])
@pytest.mark.parametrize("piped", [False, True], ids=["direct", "mirrored"])
def test_a_stopped_runner_passes_sigterm_to_the_child(tmp_path, exc_type, piped):
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    marker = tmp_path / "marker"
    share = tmp_path / "share"
    runner = (SubprocessRunner(local_root=tmp_path / "local", share_root=share) if piped
              else SubprocessRunner())
    pids = []
    previous = _raise_in(1.0, exc_type)
    try:
        with pytest.raises(exc_type):
            runner.run([sys.executable, str(script), str(marker), "polite"],
                       share / "train.log", on_start=pids.append)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert marker.read_text() == "TERM", "the child got the SIGTERM and could save its state"
    with pytest.raises(ProcessLookupError):
        os.kill(pids[0], 0)


def test_a_child_that_ignores_sigterm_is_killed_after_the_grace(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_base, "CHILD_STOP_GRACE_S", 0.5)
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    pids = []
    previous = _raise_in(1.0, Cancelled)
    started = time.monotonic()
    try:
        with pytest.raises(Cancelled):
            SubprocessRunner().run([sys.executable, str(script), str(tmp_path / "m"), "stubborn"],
                                   tmp_path / "train.log", on_start=pids.append)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert time.monotonic() - started < 10
    with pytest.raises(ProcessLookupError):
        os.kill(pids[0], 0)


def test_an_undisturbed_child_is_waited_for_as_before(tmp_path):
    code = SubprocessRunner().run([sys.executable, "-c", "import sys; sys.exit(3)"],
                                  tmp_path / "x.log")
    assert code == 3


def test_the_grace_fits_inside_slurms_warning():
    """The batch files get SIGTERM 120 s before the kill (--signal=B:TERM@120)."""
    assert runner_base.CHILD_STOP_GRACE_S < 120
    assert kraken_runner.resume_point is resume_point

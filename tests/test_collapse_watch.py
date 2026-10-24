"""A run from scratch that reads nothing is stopped while it runs (#131).

The step-count guard let through two runs on the German sweep corpus that
learned nothing — 2,778 steps for CER 1.0000, 10,186 steps for CER 0.985 in 4.9
hours — and no floor fixes that, because the material decides. These pin the
watch that reads the material's answer off the checkpoint names instead:

* the rule: exactly 0 at three checkpoints past the floor, never a zero before
  it, never once any epoch read a character;
* the watch ends the process group — TERM, then KILL — and keeps the reason;
* in the pipeline: a collapsed run fails with that reason and is neither tested
  nor registered, also when ketos leaves on SIGTERM with exit 0; ``force`` and a
  fine-tune are not watched.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from atr_training import collapse_watch as cw
from atr_training.contracts import KrakenTrainParams
from atr_training.convergence import COLLAPSE_PATIENCE, FLOOR_FROM_SCRATCH, collapse_reason
from atr_training.curves import EpochPoint

import kraken_train_svc.runner as kraken_runner
import test_train_svc_pipeline as pipeline_tests
from test_train_svc_pipeline import FakeRunner, FakeSource, request_with, run_pipeline

#: The pipeline suite's fixtures, so these runs use the same trainer settings.
settings = pipeline_tests.settings
store = pipeline_tests.store


def pts(*metrics, start=0):
    return [EpochPoint(epoch=start + i, val_metric=m) for i, m in enumerate(metrics)]


# ── the rule ────────────────────────────────────────────────────────────────
def test_the_10186_step_corpus_is_stopped_at_epoch_five():
    """926 steps an epoch: the floor of 2,000 is passed at epoch index 2 (2,778
    steps), so epochs 2, 3, 4 are the three — stopped after 4,630 of 10,186."""
    reason = collapse_reason(pts(0, 0, 0, 0, 0), steps_per_epoch=926)
    assert reason and "epochs 2–4" in reason and "4,630 steps" in reason
    assert "base_model" in reason and '"force": true' in reason


def test_zeros_before_the_floor_are_what_learning_looks_like():
    assert collapse_reason(pts(0, 0, 0, 0), steps_per_epoch=100) is None, \
        "400 steps, all below the floor"
    assert collapse_reason(pts(0, 0), steps_per_epoch=5000) is None, \
        "past the floor, but only two checkpoints"


def test_one_character_read_anywhere_is_not_a_collapse():
    assert collapse_reason(pts(0, 0, 0.0001, 0, 0, 0), steps_per_epoch=926) is None
    assert collapse_reason(pts(0.7057, 0, 0, 0), steps_per_epoch=3249) is None


def test_the_order_of_the_files_does_not_matter():
    shuffled = [EpochPoint(4, 0.0), EpochPoint(2, 0.0), EpochPoint(3, 0.0)]
    assert collapse_reason(shuffled, steps_per_epoch=926)


def test_without_a_step_count_there_is_no_floor_to_place():
    assert collapse_reason(pts(0, 0, 0, 0, 0), steps_per_epoch=0) is None


def test_the_patience_is_three():
    assert COLLAPSE_PATIENCE == 3 and FLOOR_FROM_SCRATCH == 2_000


# ── the watch ───────────────────────────────────────────────────────────────
def _ckpt(directory: Path, epoch: int, metric: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"checkpoint_{epoch:02d}-{metric:.4f}.ckpt").touch()


class Signals:
    def __init__(self, dies_on=signal.SIGTERM):
        self.sent = []
        self.dies_on = dies_on
        self.dead = False

    def killpg(self, pid, sig):
        self.sent.append((pid, sig))
        if sig in (self.dies_on, signal.SIGKILL):
            self.dead = True

    def alive(self, pid):
        return not self.dead


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_the_watch_ends_the_group_on_collapse_and_keeps_why(tmp_path):
    signals = Signals()
    watch = cw.CollapseWatch(tmp_path, 926, poll_s=0.01, grace_s=1,
                             killpg=signals.killpg, alive=signals.alive).start(4242)
    for epoch in range(3):
        _ckpt(tmp_path, epoch, 0.0)
    assert not _wait(lambda: signals.sent, timeout=0.2), "two past the floor are not three"
    _ckpt(tmp_path, 3, 0.0)
    _ckpt(tmp_path, 4, 0.0)
    assert _wait(lambda: signals.sent)
    watch.stop()
    assert signals.sent == [(4242, signal.SIGTERM)]
    assert watch.reason and watch.reason.startswith("collapsed:")


def test_a_process_that_ignores_term_gets_kill(tmp_path):
    signals = Signals(dies_on=None)
    for epoch in range(5):
        _ckpt(tmp_path, epoch, 0.0)
    watch = cw.CollapseWatch(tmp_path, 926, poll_s=0.01, grace_s=0.3,
                             killpg=signals.killpg, alive=signals.alive).start(7)
    assert _wait(lambda: (7, signal.SIGKILL) in signals.sent)
    watch.stop()
    assert signals.sent == [(7, signal.SIGTERM), (7, signal.SIGKILL)]


def test_a_run_that_reads_is_never_signalled(tmp_path):
    signals = Signals()
    for epoch, metric in enumerate([0, 0, 0.01, 0.2, 0.4, 0.5]):
        _ckpt(tmp_path, epoch, metric)
    watch = cw.CollapseWatch(tmp_path, 926, poll_s=0.01,
                             killpg=signals.killpg, alive=signals.alive).start(7)
    time.sleep(0.2)
    watch.stop()
    assert signals.sent == [] and watch.reason is None


# ── in the pipeline ─────────────────────────────────────────────────────────
#: A ketos stand-in: sleeps in its own session until signalled. With
#: ``EXIT0_ON_TERM`` it leaves with 0 on SIGTERM, as Lightning can.
FAKE_KETOS = """import signal, sys, time
if sys.argv[1] == "exit0":
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
time.sleep(30)
"""


class CollapsingRunner(FakeRunner):
    """train: a real process, and checkpoints that read exactly 0."""

    def __init__(self, tmp_path: Path, *, exit0_on_term=False, metric=0.0, **kw):
        super().__init__(**kw)
        self.script = tmp_path / "fake_ketos.py"
        self.script.write_text(FAKE_KETOS)
        self.mode = "exit0" if exit0_on_term else "plain"
        self.metric = metric
        self.train_seconds = None

    def run(self, cmd, log_path: Path, env=None, on_start=None):
        if "train" not in cmd:
            return super().run(cmd, log_path, env=env, on_start=on_start)
        self.commands.append(list(cmd))
        out = Path(cmd[cmd.index("--output") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "best_0.0000.mlmodel").write_bytes(b"WEIGHTS")
        started = time.monotonic()
        proc = subprocess.Popen([sys.executable, str(self.script), self.mode],
                                start_new_session=True)
        if on_start:
            on_start(proc.pid)
        # Epoch numbers far past the floor for any steps_per_epoch the fixture has.
        for epoch in (2000, 2001, 2002):
            _ckpt(out, epoch, self.metric)
        try:
            code = proc.wait(timeout=20)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        self.train_seconds = time.monotonic() - started
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("train\n", encoding="utf-8")
        return code


@pytest.fixture
def fast_watch(monkeypatch):
    monkeypatch.setattr(kraken_runner, "COLLAPSE_POLL_S", 0.05)


def _converging_request(**kw):
    """From scratch, enough steps for the step guard: the collapse is the run's."""
    return request_with(params=KrakenTrainParams(batch_size=1, epochs=500), **kw)


def test_a_collapsed_run_fails_with_the_reason_and_is_never_tested(store, settings, tmp_path,
                                                                   fast_watch):
    runner = CollapsingRunner(tmp_path)
    job = run_pipeline(store, settings, FakeSource({"train": 6, "eval": 2}), runner,
                       request=_converging_request(force=False))
    assert job.status == "failed"
    assert "train stopped: collapsed: validation accuracy exactly 0" in job.error
    assert runner.train_seconds < 15, "stopped, not run to the end"
    assert runner.commands_named("test") == [], "nothing collapsed is tested"
    assert not list((settings.trained_root).glob("**/metadata.json")), "nor registered"


def test_a_term_that_exits_zero_still_fails_the_run(store, settings, tmp_path, fast_watch):
    """Lightning can leave on SIGTERM with exit 0; a stopped run must not go on
    to test and register as if it had finished."""
    runner = CollapsingRunner(tmp_path, exit0_on_term=True)
    job = run_pipeline(store, settings, FakeSource({"train": 6, "eval": 2}), runner,
                       request=_converging_request(force=False))
    assert job.status == "failed" and "train stopped: collapsed" in job.error
    assert runner.commands_named("test") == []


def test_force_lets_a_run_from_scratch_go_on(store, settings, tmp_path, fast_watch, monkeypatch):
    monkeypatch.setattr(CollapsingRunner, "run", _quick_train(CollapsingRunner.run))
    runner = CollapsingRunner(tmp_path)
    job = run_pipeline(store, settings, FakeSource({"train": 6, "eval": 2}), runner,
                       request=_converging_request(force=True))
    assert "collapsed" not in (job.error or "")
    assert runner.commands_named("test"), "it ran on to test"


def test_a_run_that_reads_is_left_alone(store, settings, tmp_path, fast_watch, monkeypatch):
    monkeypatch.setattr(CollapsingRunner, "run", _quick_train(CollapsingRunner.run))
    runner = CollapsingRunner(tmp_path, metric=0.0123)
    job = run_pipeline(store, settings, FakeSource({"train": 6, "eval": 2}), runner,
                       request=_converging_request(force=False))
    assert "collapsed" not in (job.error or "")
    assert runner.commands_named("test")


def _quick_train(run):
    """The fake ketos ends by itself after a second instead of 30, so a run the
    watch leaves alone finishes."""
    def wrapped(self, cmd, log_path, env=None, on_start=None):
        self.script.write_text(FAKE_KETOS.replace("time.sleep(30)", "time.sleep(1)"))
        return run(self, cmd, log_path, env=env, on_start=on_start)
    return wrapped


def test_only_a_run_from_scratch_without_force_is_watched(store, settings):
    pipeline = kraken_runner.Pipeline(store, settings, runner=FakeRunner(),
                                      source=FakeSource({}))
    job = store.create(_converging_request(force=False))
    job.progress.steps_per_epoch = 926
    assert pipeline._collapse_watch(job, Path("/x"), from_scratch=True) is not None
    assert pipeline._collapse_watch(job, Path("/x"), from_scratch=False) is None, \
        "a fine-tune starts from weights that read; its zero means something else"
    job.progress.steps_per_epoch = None
    assert pipeline._collapse_watch(job, Path("/x"), from_scratch=True) is None
    forced = store.create(_converging_request(force=True))
    forced.progress.steps_per_epoch = 926
    assert pipeline._collapse_watch(forced, Path("/x"), from_scratch=True) is None

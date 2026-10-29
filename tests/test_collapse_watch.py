"""A run from scratch that reads nothing is stopped while it runs (#131).

The step-count guard let through two runs on the German sweep corpus that
learned nothing — 2,778 steps for CER 1.0000, 10,186 steps for CER 0.985 in 4.9
hours — and no floor fixes that, because the material decides. These pin the
watch that asks the run's own checkpoints how much they write:

* the rule: a length ratio at or below 0.05 at three checkpoints past the floor, never
  a checkpoint before it, never once any checkpoint wrote more;
* the probe: a sample of the validation pages, compiled once; each checkpoint
  converted (``ketos test`` cannot read one on kraken 7.0.2), read, and the
  weights removed; a probe that cannot measure gives None and stops nothing;
* the watch measures each checkpoint once, stands down once the run reads, and
  ends the process group — TERM, then KILL — keeping the reason;
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
from atr_training.convergence import (
    COLLAPSE_LENGTH_RATIO,
    COLLAPSE_PATIENCE,
    FLOOR_FROM_SCRATCH,
    collapse_reason,
    past_the_floor,
)
from atr_training.noise_floor import is_collapsed

import kraken_train_svc.runner as kraken_runner
import test_train_svc_pipeline as pipeline_tests
from test_train_svc_pipeline import FakeRunner, FakeSource, request_with, run_pipeline

#: The pipeline suite's fixtures, so these runs use the same trainer settings.
settings = pipeline_tests.settings
store = pipeline_tests.store

FIXTURES = Path(__file__).parent / "fixtures"

#: The #131 probe's ten checkpoints (epochs 1–10, 770 steps an epoch), measured
#: afterwards on 58 validation pages, 53,331 characters (asteraix, 28.10.2026).
PROBE_131 = {1: 0.0, 2: 0.00002, 3: 0.00013, 4: 0.00075, 5: 0.0044,
             6: 0.0080, 7: 0.0095, 8: 0.0098, 9: 0.0104, 10: 0.0106}


def upto(ratios, last):
    return {e: r for e, r in ratios.items() if e <= last}


# ── the rule ────────────────────────────────────────────────────────────────
def test_the_131_probe_is_stopped_at_3850_steps():
    """The collapse that crept past the exact-0 rule: val accuracy 0.0001 at
    epoch 3, 0.0083 at epoch 10, and never more than 1 % of the reference length.
    Epoch 2 (2,310 steps) is the first past the floor, so 2, 3, 4 are the three."""
    assert collapse_reason(upto(PROBE_131, 3), steps_per_epoch=770) is None, \
        "epoch 1 is before the floor; two past it are not three"
    reason = collapse_reason(upto(PROBE_131, 4), steps_per_epoch=770)
    assert reason and "epochs 2–4" in reason and "3,850 steps" in reason
    assert "0.0008" in reason, "the measurements are in the error"
    assert "base_model" in reason and '"force": true' in reason


def test_a_whole_run_of_it_is_a_collapse_at_every_length():
    assert collapse_reason(PROBE_131, steps_per_epoch=770)


def test_checkpoints_before_the_floor_are_what_learning_looks_like():
    assert collapse_reason({0: 0, 1: 0, 2: 0, 3: 0}, steps_per_epoch=100) is None, \
        "400 steps, all below the floor"
    assert collapse_reason({0: 0, 1: 0}, steps_per_epoch=5000) is None, \
        "past the floor, but only two checkpoints"
    assert past_the_floor(2, 770) and not past_the_floor(1, 770)


def test_one_checkpoint_that_writes_is_not_a_collapse():
    assert collapse_reason({2: 0.01, 3: 0.02, 4: 0.0501, 5: 0.0}, steps_per_epoch=926) is None
    assert collapse_reason({2: 0.6, 3: 0.0, 4: 0.0, 5: 0.0}, steps_per_epoch=926) is None


def test_a_poor_reading_of_full_length_is_not_a_collapse():
    """What the old rule could not tell apart: accuracy near 0, but text written."""
    assert collapse_reason({2: 0.9, 3: 1.1, 4: 0.95}, steps_per_epoch=926) is None


def test_the_order_does_not_matter():
    shuffled = {4: 0.0, 2: 0.0, 3: 0.0}
    assert collapse_reason(shuffled, steps_per_epoch=926)


def test_without_a_step_count_there_is_no_floor_to_place():
    assert collapse_reason({0: 0, 1: 0, 2: 0, 3: 0, 4: 0}, steps_per_epoch=0) is None


def test_the_threshold_is_the_one_that_calls_a_result_collapsed():
    """One definition of "no reading": the noise floor's, which the sweep uses
    to throw out an empty hypothesis after the run."""
    assert COLLAPSE_LENGTH_RATIO == 0.05
    assert is_collapsed(0.99, 0.05) and not is_collapsed(0.9, 0.0501)
    assert collapse_reason({2: 0.05, 3: 0.05, 4: 0.05}, steps_per_epoch=926)
    assert collapse_reason({2: 0.05, 3: 0.0501, 4: 0.05}, steps_per_epoch=926) is None


def test_the_patience_is_three():
    assert COLLAPSE_PATIENCE == 3 and FLOOR_FROM_SCRATCH == 2_000


# ── the probe ───────────────────────────────────────────────────────────────
class FakeKetos:
    """``subprocess.run`` for compile, convert and test, writing what ketos would."""

    def __init__(self, *, report=None, fail=()):
        self.report = report or (FIXTURES / "ketos_test_report_collapse_probe.txt").read_text()
        self.fail = set(fail)
        self.calls = []
        self.envs = []

    def __call__(self, cmd, *, stdout, stderr, env, timeout):
        verb = next(v for v in ("compile", "convert", "test") if v in cmd)
        self.calls.append((verb, list(cmd)))
        self.envs.append(env)
        if verb in self.fail:
            return subprocess.CompletedProcess(cmd, 1)
        if verb == "compile":
            Path(cmd[cmd.index("--output") + 1]).write_bytes(b"ARROW")
        elif verb == "convert":
            Path(cmd[cmd.index("--output") + 1]).write_bytes(b"WEIGHTS")
        else:
            stdout.write(self.report)
        return subprocess.CompletedProcess(cmd, 0)

    def named(self, verb):
        return [c for v, c in self.calls if v == verb]


def _pages(tmp_path, n):
    pages = [tmp_path / "pages" / f"p{i}.xml" for i in range(n)]
    (tmp_path / "pages").mkdir()
    for page in pages:
        page.touch()
    manifest = tmp_path / "pages_val.lst"
    manifest.write_text("\n".join(str(p) for p in pages) + "\n")
    return manifest


def test_the_probe_reads_a_checkpoint_off_a_sample(tmp_path):
    ketos = FakeKetos()
    probe = cw.LengthProbe("ketos", _pages(tmp_path, 5), tmp_path / "probe", sample_pages=2,
                           device="cuda:0", env={"CUDA_VISIBLE_DEVICES": "1"}, run=ketos)
    ckpt = tmp_path / "checkpoint_04-0.0007.ckpt"
    ckpt.touch()
    ratio = probe(ckpt)
    assert ratio == pytest.approx(40 / 53331), "53,331 characters, 53,291 of them missing"
    sample = (tmp_path / "probe" / "sample_pages.lst").read_text().split()
    assert [Path(p).name for p in sample] == ["p0.xml", "p3.xml"], "every n-th page, spread"
    (convert,) = ketos.named("convert")
    assert str(ckpt) in convert and convert[convert.index("--weights-format") + 1] == "coreml"
    (test,) = ketos.named("test")
    assert test[test.index("--model") + 1].endswith("checkpoint_04-0.0007.mlmodel")
    assert not list((tmp_path / "probe").glob("*.mlmodel")), "the converted weights are removed"
    assert all(e["CUDA_VISIBLE_DEVICES"] == "1" and "PATH" in e for e in ketos.envs), \
        "the job's card, on top of the service's environment"
    probe(ckpt)
    assert len(ketos.named("compile")) == 1, "the sample is compiled once"


def test_a_sample_that_does_not_compile_measures_nothing_and_is_not_retried(tmp_path):
    ketos = FakeKetos(fail={"compile"})
    probe = cw.LengthProbe("ketos", _pages(tmp_path, 3), tmp_path / "probe", run=ketos)
    ckpt = tmp_path / "checkpoint_04-0.0000.ckpt"
    assert probe(ckpt) is None and probe(ckpt) is None
    assert len(ketos.named("compile")) == 1 and not ketos.named("convert")


@pytest.mark.parametrize("step", ["convert", "test"])
def test_a_failed_conversion_or_test_is_none(tmp_path, step):
    probe = cw.LengthProbe("ketos", _pages(tmp_path, 3), tmp_path / "probe",
                           run=FakeKetos(fail={step}))
    assert probe(tmp_path / "checkpoint_04-0.0000.ckpt") is None
    assert not list((tmp_path / "probe").glob("*.mlmodel"))


def test_no_validation_pages_is_none(tmp_path):
    probe = cw.LengthProbe("ketos", tmp_path / "missing.lst", tmp_path / "probe", run=FakeKetos())
    assert probe(tmp_path / "checkpoint_04-0.0000.ckpt") is None


# ── the watch ───────────────────────────────────────────────────────────────
def _ckpt(directory: Path, epoch: int, metric: float = 0.0) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"checkpoint_{epoch:02d}-{metric:.4f}.ckpt").touch()


class Measure:
    """A probe that answers from a table by epoch, and counts what it was asked."""

    def __init__(self, ratios, *, raises=()):
        self.ratios = ratios
        self.raises = set(raises)
        self.asked = []

    def __call__(self, path):
        epoch = int(path.name.split("_")[1].split("-")[0])
        self.asked.append(epoch)
        if epoch in self.raises:
            raise RuntimeError("card busy")
        return self.ratios.get(epoch)


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
    measure = Measure(PROBE_131)
    watch = cw.CollapseWatch(tmp_path, 770, measure, poll_s=0.01, grace_s=1,
                             killpg=signals.killpg, alive=signals.alive).start(4242)
    for epoch in range(4):
        _ckpt(tmp_path, epoch)
    assert not _wait(lambda: signals.sent, timeout=0.2), "two past the floor are not three"
    _ckpt(tmp_path, 4, 0.0007)
    assert _wait(lambda: signals.sent)
    watch.stop()
    assert signals.sent == [(4242, signal.SIGTERM)]
    assert watch.reason and watch.reason.startswith("collapsed: length ratio at or below 0.05")
    assert sorted(measure.asked) == [2, 3, 4], "past the floor only, and each once"


def test_a_process_that_ignores_term_gets_kill(tmp_path):
    signals = Signals(dies_on=None)
    for epoch in range(5):
        _ckpt(tmp_path, epoch)
    watch = cw.CollapseWatch(tmp_path, 926, Measure(dict.fromkeys(range(5), 0.0)),
                             poll_s=0.01, grace_s=0.3,
                             killpg=signals.killpg, alive=signals.alive).start(7)
    assert _wait(lambda: (7, signal.SIGKILL) in signals.sent)
    watch.stop()
    assert signals.sent == [(7, signal.SIGTERM), (7, signal.SIGKILL)]


def test_a_run_that_reads_is_never_signalled_and_the_watch_stands_down(tmp_path):
    signals = Signals()
    measure = Measure({2: 0.01, 3: 0.3, 4: 0.0, 5: 0.0, 6: 0.0})
    for epoch in range(4):
        _ckpt(tmp_path, epoch)
    watch = cw.CollapseWatch(tmp_path, 926, measure, poll_s=0.01,
                             killpg=signals.killpg, alive=signals.alive).start(7)
    assert _wait(lambda: watch.reading)
    for epoch in (4, 5, 6):
        _ckpt(tmp_path, epoch)
    time.sleep(0.2)
    watch.stop()
    assert signals.sent == [] and watch.reason is None
    assert measure.asked == [2, 3], "nothing after the run read is measured"


def test_a_checkpoint_that_cannot_be_measured_does_not_count(tmp_path):
    signals = Signals()
    measure = Measure({2: 0.0, 4: 0.0, 5: 0.0}, raises={3})
    for epoch in range(5):
        _ckpt(tmp_path, epoch)
    watch = cw.CollapseWatch(tmp_path, 926, measure, poll_s=0.01,
                             killpg=signals.killpg, alive=signals.alive).start(7)
    assert not _wait(lambda: signals.sent, timeout=0.2), "2 and 4 measured, 3 not: two"
    _ckpt(tmp_path, 5)
    assert _wait(lambda: signals.sent)
    watch.stop()
    assert "epochs 2–5" in watch.reason
    assert measure.asked.count(3) == 1, "a failed measurement is not repeated every minute"


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


class FakeProbe:
    """Stands in for :class:`~atr_training.collapse_watch.LengthProbe` in the
    pipeline: every checkpoint writes ``ratio`` of the reference length."""

    ratio = 0.0
    made = []

    def __init__(self, ketos, pages_val, workdir, **kw):
        FakeProbe.made.append((Path(pages_val), Path(workdir), kw))

    def __call__(self, checkpoint):
        return FakeProbe.ratio


@pytest.fixture
def fast_watch(monkeypatch):
    monkeypatch.setattr(kraken_runner, "COLLAPSE_POLL_S", 0.05)
    monkeypatch.setattr(kraken_runner, "LengthProbe", FakeProbe)
    monkeypatch.setattr(FakeProbe, "ratio", 0.0)
    monkeypatch.setattr(FakeProbe, "made", [])


def _converging_request(**kw):
    """From scratch, enough steps for the step guard: the collapse is the run's."""
    return request_with(params=KrakenTrainParams(batch_size=1, epochs=500), **kw)


def test_a_collapsed_run_fails_with_the_reason_and_is_never_tested(store, settings, tmp_path,
                                                                   fast_watch):
    runner = CollapsingRunner(tmp_path)
    job = run_pipeline(store, settings, FakeSource({"train": 6, "eval": 2}), runner,
                       request=_converging_request(force=False))
    assert job.status == "failed"
    assert "train stopped: collapsed: length ratio at or below 0.05" in job.error
    assert runner.train_seconds < 15, "stopped, not run to the end"
    assert runner.commands_named("test") == [], "nothing collapsed is tested"
    assert not list((settings.trained_root).glob("**/metadata.json")), "nor registered"
    ((pages_val, workdir, kw),) = FakeProbe.made
    assert pages_val == store.paths(job.id).data / "pages_val.lst" and pages_val.exists(), \
        "measured on this job's own validation pages"
    assert workdir.parent == Path(job.checkpoint_dir), "and cleaned up with its checkpoints"
    assert kw["device"] == job.request.params.device


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
    FakeProbe.ratio = 0.62
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

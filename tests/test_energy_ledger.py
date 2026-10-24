"""Energy survives an interrupted stage (#184).

`PeakSampler` integrates `power.draw` in memory, and the total reached the job
record only when the stage ended in order. A preempted or requeued stage dies
before that, so an interrupted run kept only the energy of its last attempt:
the medieval olmOCR r16 arm drew on ~46 GPU-hours and recorded 880 Wh, which
is its last 3.6-hour attempt alone (UBELIX, 24.10.2026).

The sampler now writes its running total to a ledger in the job directory every
minute, and the next attempt adds whatever a dead one left behind.
"""
from __future__ import annotations

import json

from atr_training import gpu
from atr_training.contracts import Progress, StageRecord
from atr_training.jobstore import JobPaths

UUID_A = "GPU-aaaa"


def fake_smi(watts):
    def _smi(query, *, per_app):
        if per_app:
            return [["4242", "20000", UUID_A]]
        if query == "uuid":
            return [[UUID_A]]
        if query == "memory.used":
            return [["21000"]]
        if query == "power.draw":
            return [[str(w)] for w in watts]
        raise AssertionError(f"unexpected query {query!r}")
    return _smi


def sampled(monkeypatch, *, watts=400.0, hours=1.0, ledger=None):
    """A sampler that has integrated ``watts`` over ``hours`` on one card."""
    monkeypatch.setattr(gpu, "_smi", fake_smi([watts]))
    monkeypatch.setattr(gpu, "_ancestors", lambda pid, limit=32: [pid, 77])
    sampler = gpu.PeakSampler(77, ledger=ledger)
    clock = iter([0.0, hours * 3600.0])
    monkeypatch.setattr(gpu.time, "monotonic", lambda: next(clock, hours * 3600.0))
    sampler._once()
    sampler._once()
    return sampler


# ── the ledger ──────────────────────────────────────────────────────────────
def test_a_flush_writes_the_running_total(tmp_path, monkeypatch):
    ledger = tmp_path / "energy-pending.train.1.json"
    sampler = sampled(monkeypatch, watts=400.0, hours=2.0, ledger=ledger)

    sampler.flush()

    assert gpu.read_ledger(ledger) == (800.0, False)
    assert not ledger.with_name(ledger.name + ".tmp").exists(), "written atomically"


def test_nothing_drawn_writes_no_ledger(tmp_path, monkeypatch):
    """A card that answers [N/A] integrates 0 Wh, and 0 is not a measurement."""
    ledger = tmp_path / "energy-pending.train.1.json"
    sampler = sampled(monkeypatch, watts=0.0, ledger=ledger)

    sampler.flush()

    assert not ledger.exists()


def test_the_loop_flushes_on_its_own(tmp_path, monkeypatch):
    """Not only on demand: the process that should call flush is the one that dies."""
    ledger = tmp_path / "energy-pending.train.1.json"
    monkeypatch.setattr(gpu, "_smi", fake_smi([300.0]))
    monkeypatch.setattr(gpu, "_ancestors", lambda pid, limit=32: [pid, 77])
    sampler = gpu.PeakSampler(77, interval_s=0.01, ledger=ledger, flush_s=0.0)

    sampler.start()
    deadline = gpu.time.monotonic() + 5
    while not ledger.exists() and gpu.time.monotonic() < deadline:
        gpu.time.sleep(0.01)
    sampler.stop()

    assert ledger.exists()


def test_a_damaged_ledger_reads_as_none_not_zero(tmp_path):
    ledger = tmp_path / "energy-pending.train.1.json"
    ledger.write_text("{not json", encoding="utf-8")

    assert gpu.read_ledger(ledger) is None


# ── the runner adds what a dead attempt left ────────────────────────────────
class _Store:
    def __init__(self, root) -> None:
        self.root = root
        self.saves = 0

    def paths(self, job_id) -> JobPaths:
        return JobPaths(root=self.root)

    def save(self, job) -> None:
        self.saves += 1


class _Job:
    id = "20261024T000000Z-test"

    def __init__(self) -> None:
        self.progress = Progress()


class _Settings:
    def env_for_child(self):
        return {}


class _Runner:
    """Starts nothing; hands the sampler a pid and reports success.

    With ``until_ledger`` it returns only once the sampler has written a ledger
    into that directory, so a stage that ends in order has one to clean up.
    """

    def __init__(self, until_ledger=None) -> None:
        self.until_ledger = until_ledger

    def run(self, cmd, log_path, *, env=None, on_start=None):
        if on_start is not None:
            on_start(4242)
        if self.until_ledger is not None:
            deadline = gpu.time.monotonic() + 5
            while (not list(self.until_ledger.glob("energy-pending.*.json"))
                   and gpu.time.monotonic() < deadline):
                gpu.time.sleep(0.01)
        return 0


def _pipeline(root):
    from atr_training.runner_base import BasePipeline

    class _P:
        store = _Store(root)
        runner = _Runner()
        settings = _Settings()

    p = _P()
    for name in ("_run", "_recover_energy", "_record_peak"):
        setattr(p, name, getattr(BasePipeline, name).__get__(p))
    p._slurm_job_id = None
    return p


def test_the_next_attempt_adds_a_dead_attempts_energy(tmp_path, monkeypatch):
    ledger = tmp_path / "energy-pending.train.999.json"
    sampled(monkeypatch, watts=390.0, hours=3.0, ledger=ledger).flush()
    monkeypatch.setattr(gpu, "_smi", fake_smi([0.0]))  # the new attempt draws nothing
    job, p = _Job(), _pipeline(tmp_path)

    p._run(job, "train", ["true"], StageRecord(name="train"))

    assert job.progress.energy_wh == 1170.0
    assert job.progress.energy_recovered_wh == 1170.0
    assert not ledger.exists(), "counted once, then gone"


def test_an_ordered_stage_leaves_no_ledger_to_count_twice(tmp_path, monkeypatch):
    """The stage that ended in order already counted its energy; its ledger
    must not be found again by the next stage and added a second time."""
    monkeypatch.setattr(gpu, "_smi", fake_smi([400.0]))
    monkeypatch.setattr(gpu, "_ancestors", lambda pid, limit=32: [pid, 77])
    monkeypatch.setattr(gpu, "LEDGER_FLUSH_S", 0.0)
    job, p = _Job(), _pipeline(tmp_path)
    p.runner = _Runner(until_ledger=tmp_path)

    p._run(job, "train", ["true"], StageRecord(name="train"))
    assert job.progress.energy_wh > 0, "the stage drew something"
    p.runner = _Runner()
    monkeypatch.setattr(gpu, "_smi", fake_smi([0.0]))
    p._run(job, "test", ["true"], StageRecord(name="test"))

    assert list(tmp_path.glob("energy-pending.*")) == []
    assert job.progress.energy_recovered_wh == 0.0, "counted twice"


def test_an_unreadable_ledger_is_left_and_not_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu, "_smi", fake_smi([0.0]))
    bad = tmp_path / "energy-pending.train.999.json"
    bad.write_text(json.dumps({"energy_wh": "garbage"}), encoding="utf-8")
    job, p = _Job(), _pipeline(tmp_path)

    p._run(job, "train", ["true"], StageRecord(name="train"))

    assert job.progress.energy_wh == 0.0
    assert bad.exists(), "left for a person to look at, not deleted unread"

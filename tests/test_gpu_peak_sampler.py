"""Peak GPU memory for an engine that holds no torch handle (#163).

`kraken` drives `ketos` as an external CLI, so there is no allocator to ask, and
the log carries no numbers either — ketos renders through `rich`. The only route
to the figure the first branch of `docs/WHERE_A_RUN_RUNS.md` needs is to watch the
card from outside while the subprocess runs.

The attribution is the module's own idiom from `gpu.inspect`: a compute app
belongs to this job when the pid we spawned is in its ancestor chain.
"""
from __future__ import annotations

import os
import sys

from atr_training import gpu
from atr_training.contracts import Progress, StageRecord

UUID_A = "GPU-aaaa"
UUID_B = "GPU-bbbb"


def fake_smi(apps, used, uuids=(UUID_A,)):
    """Stand in for `_smi`, which is the only thing that shells out."""
    def _smi(query, *, per_app):
        if per_app:
            return [list(row) for row in apps]
        if query == "uuid":
            return [[u] for u in uuids]
        if query == "memory.used":
            return [[str(v)] for v in used]
        raise AssertionError(f"unexpected query {query!r}")
    return _smi


# ── attribution ─────────────────────────────────────────────────────────────
def test_only_our_own_process_tree_is_counted(monkeypatch):
    """A neighbour on the same card is not this run's footprint."""
    monkeypatch.setattr(gpu, "_smi", fake_smi(
        apps=[["4242", "21675", UUID_A], ["999", "9000", UUID_A]], used=[31000]))
    monkeypatch.setattr(gpu, "_ancestors", lambda pid, limit=32: {
        4242: [4242, 77], 999: [999, 1234]}.get(pid, []))

    sampler = gpu.PeakSampler(77)
    sampler._once()

    assert sampler.peak.own_mib == {0: 21675}, "the neighbour's 9000 MiB is not ours"
    assert sampler.peak.card_mib == {0: 31000}
    assert sampler.peak.attributed is True


def test_a_descendant_counts_because_ketos_forks(monkeypatch):
    """start_new_session makes our pid a session leader; children stay reachable."""
    monkeypatch.setattr(gpu, "_smi", fake_smi(apps=[["5000", "1200", UUID_A]], used=[1500]))
    monkeypatch.setattr(gpu, "_ancestors", lambda pid, limit=32: [5000, 4999, 77, 1])

    sampler = gpu.PeakSampler(77)
    sampler._once()

    assert sampler.peak.own_mib == {0: 1200}


def test_two_cards_stay_apart(monkeypatch):
    monkeypatch.setattr(gpu, "_smi", fake_smi(
        apps=[["1", "13761", UUID_A], ["2", "18625", UUID_B]], used=[14000, 19000],
        uuids=(UUID_A, UUID_B)))
    monkeypatch.setattr(gpu, "_ancestors", lambda pid, limit=32: [pid, 77])

    sampler = gpu.PeakSampler(77)
    sampler._once()

    assert sampler.peak.own_mib == {0: 13761, 1: 18625}


def test_the_maximum_is_kept_not_the_last_reading(monkeypatch):
    readings = iter([
        fake_smi(apps=[["1", "5000", UUID_A]], used=[6000]),
        fake_smi(apps=[["1", "27763", UUID_A]], used=[28000]),
        fake_smi(apps=[["1", "900", UUID_A]], used=[1000]),
    ])
    monkeypatch.setattr(gpu, "_ancestors", lambda pid, limit=32: [1, 77])

    sampler = gpu.PeakSampler(77)
    for _ in range(3):
        monkeypatch.setattr(gpu, "_smi", next(readings))
        sampler._once()

    assert sampler.peak.own_mib == {0: 27763}
    assert sampler.peak.card_mib == {0: 28000}
    assert sampler.peak.readings == 3


# ── "could not look" is not "used nothing" (#165) ───────────────────────────
def test_an_untraceable_process_does_not_read_as_zero(monkeypatch):
    """Behind a PID namespace the ancestry breaks; `attributed` says so."""
    monkeypatch.setattr(gpu, "_smi", fake_smi(apps=[["31337", "22000", UUID_A]], used=[22500]))
    monkeypatch.setattr(gpu, "_ancestors", lambda pid, limit=32: [31337, 1])

    sampler = gpu.PeakSampler(77)
    sampler._once()

    assert sampler.peak.attributed is False
    assert sampler.peak.card_mib == {0: 22500}, "the card figure is still honest"
    assert "NOT attributed" in sampler.peak.summary()


def test_a_failing_smi_is_counted_and_does_not_raise(monkeypatch):
    def boom(query, *, per_app):
        raise FileNotFoundError("nvidia-smi is not on PATH")

    monkeypatch.setattr(gpu, "_smi", boom)
    sampler = gpu.PeakSampler(77)
    sampler._once()

    assert sampler.peak.failures == 1
    assert sampler.peak.readings == 0
    assert "no GPU reading taken" in sampler.peak.summary()


def test_a_card_with_no_uuid_match_is_skipped(monkeypatch):
    """An app on a card we did not enumerate must not land on card 0."""
    monkeypatch.setattr(gpu, "_smi", fake_smi(
        apps=[["1", "5000", "GPU-elsewhere"]], used=[100]))
    monkeypatch.setattr(gpu, "_ancestors", lambda pid, limit=32: [1, 77])

    sampler = gpu.PeakSampler(77)
    sampler._once()

    assert sampler.peak.own_mib == {}


# ── the thread, and the short stage ─────────────────────────────────────────
def test_a_stage_shorter_than_one_interval_still_gets_a_reading(monkeypatch):
    """A peak of nothing would be the worst kind of measurement: plausible."""
    monkeypatch.setattr(gpu, "_smi", fake_smi(apps=[["1", "700", UUID_A]], used=[800]))
    monkeypatch.setattr(gpu, "_ancestors", lambda pid, limit=32: [1, 77])

    peak = gpu.PeakSampler(77, interval_s=30).start().stop()

    assert peak.readings >= 1
    assert peak.own_mib == {0: 700}


# ── the pid reaches the sampler from the real runner ────────────────────────
def test_the_subprocess_runner_hands_over_a_live_pid(tmp_path):
    from atr_training.runner_base import SubprocessRunner

    seen: list[int] = []
    runner = SubprocessRunner()
    code = runner.run([sys.executable, "-c", "print('x')"], tmp_path / "log.txt",
                      on_start=seen.append)

    assert code == 0
    assert len(seen) == 1 and seen[0] > 0
    assert seen[0] != os.getpid(), "that is the child's pid, not ours"


def test_a_watcher_that_throws_does_not_lose_the_child(tmp_path, caplog):
    """The process is already running; nothing else is holding it."""
    from atr_training.runner_base import SubprocessRunner

    def explode(pid: int) -> None:
        raise RuntimeError("no nvidia-smi here")

    code = SubprocessRunner().run([sys.executable, "-c", "pass"],
                                  tmp_path / "log.txt", on_start=explode)

    assert code == 0


# ── onto the record ─────────────────────────────────────────────────────────
class _Store:
    def save(self, job) -> None:
        pass


class _Job:
    def __init__(self) -> None:
        self.progress = Progress()


def _pipeline():
    """`_record_peak` alone, which needs no engine."""
    from atr_training.runner_base import BasePipeline

    class _P:
        store = _Store()

    p = _P()
    p._record_peak = BasePipeline._record_peak.__get__(p)
    return p


def test_the_peak_lands_on_the_stage_and_on_the_job():
    job, record = _Job(), StageRecord(name="train")
    peak = gpu.Peak(own_mib={0: 21675}, card_mib={0: 22000}, readings=5, attributed=True)

    _pipeline()._record_peak(job, record, peak)

    assert record.peak_gpu_mib == {"gpu0": {"own_mib": 21675, "card_mib": 22000}}
    assert job.progress.peak_gpu_mib["gpu0"]["own_mib"] == 21675


def test_a_cheaper_later_stage_cannot_lower_the_job_peak():
    """The eval loop is where the OOM behind drop_long_samples happened — the
    maximum over stages is the figure, not the last one."""
    job = _Job()
    p = _pipeline()

    p._record_peak(job, StageRecord(name="train"),
                   gpu.Peak(own_mib={0: 27763}, card_mib={0: 28000},
                            readings=3, attributed=True))
    p._record_peak(job, StageRecord(name="test"),
                   gpu.Peak(own_mib={0: 900}, card_mib={0: 1000},
                            readings=3, attributed=True))

    assert job.progress.peak_gpu_mib["gpu0"] == {"own_mib": 27763, "card_mib": 28000}


def test_an_unattributed_peak_omits_own_rather_than_claiming_zero():
    job, record = _Job(), StageRecord(name="train")
    peak = gpu.Peak(own_mib={}, card_mib={0: 30000}, readings=4, attributed=False)

    _pipeline()._record_peak(job, record, peak)

    assert record.peak_gpu_mib == {"gpu0": {"card_mib": 30000}}
    assert "own_mib" not in record.peak_gpu_mib["gpu0"]


def test_torch_marks_and_sampled_marks_share_one_row():
    """A row carrying the allocator's floor and the card's figure is the row
    worth having, so the two writers merge instead of overwriting."""
    job, record = _Job(), StageRecord(name="train")
    job.progress.peak_gpu_mib = {"gpu0": {"reserved_mib": 27763, "allocated_mib": 26104}}

    _pipeline()._record_peak(job, record,
                             gpu.Peak(own_mib={0: 28500}, card_mib={0: 29000},
                                      readings=2, attributed=True))

    assert sorted(job.progress.peak_gpu_mib["gpu0"]) == [
        "allocated_mib", "card_mib", "own_mib", "reserved_mib"]


def test_nothing_measured_leaves_the_record_untouched():
    job, record = _Job(), StageRecord(name="compile")

    _pipeline()._record_peak(job, record, gpu.Peak())

    assert record.peak_gpu_mib == {}
    assert job.progress.peak_gpu_mib == {}

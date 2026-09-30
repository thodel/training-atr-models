"""The sweep runs continuously, and gives the card back on demand (#117).

Two halves, and the second is the one with history behind it.

**Standing queue.** The driver does not run the ladder once and stop; it takes
the next open configuration whenever the card is free, and waits when there is
nothing to do. A configuration added to the manifest days later is picked up
without anybody restarting anything.

**Last tenant.** On 15.09.2026 the gateway started a vLLM model during a run's
`prepare` stage, because prepare did not claim the card. When the run reached
`train` the model still held 16.5 GB, and three minutes later it died with 841
MiB free. The lock was then changed so a job claims the card from its first
stage (serving-atr-inference#129).

A permanently running sweep inverts that danger: it is the process that is
always there, so claiming the card like a requested run would block every
inference and every real training from now on. It therefore needs the opposite
treatment — claimed, but preemptible — and the rule that makes it safe is that a
preempted cell is **not a result**. It ran on part of its budget, and half a
budget is not a measurement.
"""

import json
from pathlib import Path

import pytest

from atr_training.sweep_driver import (
    IDLE_S, Preempted, SweepDriver, SweepError, SweepState,
)
from atr_training.sweep_manifest import ManifestError

from tests.test_sweep_driver import FakeTrainer, manifest, scores_for


def driver(trainer, m=None, *, naps=None, policy="cancel", train_lines=120_000):
    m = m or manifest()
    state = SweepState.for_manifest(m, "benchmark_cer")
    state.train_lines = train_lines
    return SweepDriver(m, state, trainer,
                       sleep=(naps.append if naps is not None else (lambda _s: None)),
                       yield_policy=policy)


def all_good(m):
    return FakeTrainer(scores_for(m, [0.200 + i * 0.003 for i in range(12)]))


# ── the ceiling ─────────────────────────────────────────────────────────────
def test_a_manifest_can_state_a_ceiling():
    m = manifest(**{"budget.max_total_steps": 500_000})

    assert m.max_total_steps == 500_000


def test_no_ceiling_is_allowed_for_a_sweep_somebody_watches():
    assert manifest().max_total_steps is None


@pytest.mark.parametrize("bad", [0, -1, "lots", 1.5, True])
def test_a_ceiling_must_be_a_positive_whole_number(bad):
    with pytest.raises(ManifestError, match="max_total_steps"):
        manifest(**{"budget.max_total_steps": bad})


def test_the_sweep_stops_when_it_reaches_its_ceiling():
    """A continuously running process without an upper bound is not a process,
    it is a leak."""
    m = manifest(**{"budget.max_total_steps": 20_000})
    trainer = all_good(m)
    d = driver(trainer, m)

    d.run()

    assert d.state.spent_steps() <= 20_000 + 4000, d.state.spent_steps()
    assert len(trainer.submitted) < 12, "the ceiling did not stop anything"


def test_the_ceiling_is_checked_before_a_cell_not_after():
    """A ceiling crossed by the cell that crossed it has not held."""
    m = manifest(**{"budget.max_total_steps": 4_000})
    trainer = all_good(m)
    d = driver(trainer, m)

    d.run()

    assert len(trainer.submitted) == 1


def test_a_preempted_cell_still_counts_against_the_ceiling():
    """It spent the card whether or not it produced a number."""
    m = manifest()
    state = SweepState.for_manifest(m, "benchmark_cer")
    state.put(0, "c1", {"job_id": "j1", "status": "cancelled", "preempted": True,
                        "score": None, "raw": None, "steps": 4000})

    assert state.spent_steps() == 4000


def test_the_stop_says_what_to_change():
    m = manifest(**{"budget.max_total_steps": 1})
    d = driver(all_good(m), m)
    d.state.put(0, "c1", {"job_id": "j", "status": "completed", "steps": 5000})

    assert "budget.max_total_steps" in d.over_ceiling()


# ── who else wants the card ─────────────────────────────────────────────────
def test_a_job_this_sweep_did_not_submit_is_a_requested_job():
    m = manifest()
    trainer = all_good(m)
    trainer.someone_wants_the_card("operator-run-1")
    d = driver(trainer, m)

    assert d.requested_jobs() == ["operator-run-1"]


def test_our_own_jobs_are_not_mistaken_for_requests():
    m = manifest()
    trainer = all_good(m)
    d = driver(trainer, m)
    d.run()

    assert d.requested_jobs() == []


def test_a_finished_foreign_job_does_not_hold_the_card():
    m = manifest()
    trainer = all_good(m)
    trainer.someone_wants_the_card("done-1", status="completed")

    assert driver(trainer, m).requested_jobs() == []


def test_a_trainer_that_cannot_be_asked_counts_as_occupied():
    """Never let a failed poll look like a free card. The failure this yields to
    killed a run with 841 MiB free."""
    class Mute(FakeTrainer):
        def jobs(self):
            raise ConnectionError("trainer unreachable")

    assert driver(Mute()).requested_jobs() == ["<unknown>"]


# ── yielding ────────────────────────────────────────────────────────────────
class Slow(FakeTrainer):
    """A trainer whose jobs take a few looks to finish, so a request can arrive
    while one is running."""

    def __init__(self, *a, wants_card_after: int = 1, **kw):
        super().__init__(*a, **kw)
        self._looks = 0
        self._wants_after = wants_card_after

    def job(self, job_id):
        self._looks += 1
        if self._looks == self._wants_after:
            self.someone_wants_the_card()
        record = dict(self.records[job_id])
        if self._looks <= self._wants_after:
            record["status"] = "training"
        return record


def test_a_requested_job_takes_the_card_without_anybody_intervening():
    """#117's second acceptance criterion, on a case rather than a report."""
    m = manifest()
    trainer = Slow(scores_for(m, [0.2] * 12))
    d = driver(trainer, m)

    with pytest.raises(Preempted):
        d.run()

    assert trainer.cancelled, "the running cell was not given up"


def test_a_preempted_cell_leaves_no_score():
    """#117's third criterion: no half measurement in the leaderboard."""
    m = manifest()
    trainer = Slow(scores_for(m, [0.2] * 12))
    d = driver(trainer, m)

    with pytest.raises(Preempted):
        d.run()

    entry = next(iter(d.state.results["0"].values()))
    assert entry["score"] is None and entry["raw"] is None
    assert entry["preempted"] is True


def test_a_preempted_cell_is_run_again_rather_than_counted():
    """`scored()` must not accept it, or the next pass would skip it and the
    sweep would rank a configuration on part of a budget."""
    m = manifest()
    state = SweepState.for_manifest(m, "benchmark_cer")
    state.put(0, "c1", {"job_id": "j", "status": "cancelled", "preempted": True,
                        "score": None, "raw": None})

    assert state.scored("c1", 0) is False


def test_an_ordinary_cancellation_is_still_terminal():
    """Somebody cancelling a job by hand is not a preemption, and re-running it
    for them would be its own surprise."""
    m = manifest()
    state = SweepState.for_manifest(m, "benchmark_cer")
    state.put(0, "c1", {"job_id": "j", "status": "cancelled", "score": None, "raw": None})

    assert state.scored("c1", 0) is True


def test_nothing_new_is_submitted_while_the_card_is_wanted():
    m = manifest()
    trainer = all_good(m)
    trainer.someone_wants_the_card()
    d = driver(trainer, m)

    with pytest.raises(Preempted):
        d.run()

    assert trainer.submitted == []


def test_the_finish_cell_policy_does_not_cancel():
    """The other side of the decision #117 leaves open: at the last rung a cell
    is many hours and throwing them away is expensive — but so is waiting them
    out, which is the block this exists to prevent."""
    m = manifest()
    trainer = Slow(scores_for(m, [0.2] * 12))
    d = driver(trainer, m, policy="finish_cell")

    with pytest.raises(Preempted):
        d.run()

    assert trainer.cancelled == []


def test_the_finish_cell_policy_still_records_no_score():
    """Whichever policy, the cell ran on part of its budget."""
    m = manifest()
    trainer = Slow(scores_for(m, [0.2] * 12))
    d = driver(trainer, m, policy="finish_cell")

    with pytest.raises(Preempted):
        d.run()

    assert next(iter(d.state.results["0"].values()))["score"] is None


def test_an_unknown_yield_policy_is_refused():
    with pytest.raises(SweepError, match="yield policy"):
        driver(all_good(manifest()), policy="ignore-everyone")


def test_the_record_says_who_the_card_went_to():
    m = manifest()
    trainer = Slow(scores_for(m, [0.2] * 12))
    d = driver(trainer, m)

    with pytest.raises(Preempted):
        d.run()

    assert next(iter(d.state.results["0"].values()))["preempted_for"] == ["requested-1"]


# ── the standing queue ──────────────────────────────────────────────────────
def test_serving_keeps_going_after_the_ladder_is_done():
    """It does not run once and stop: that is the difference between a sweep
    somebody starts and a sweep that is a process."""
    m = manifest()
    naps: list[float] = []
    d = driver(all_good(m), m, naps=naps)

    assert d.serve(passes=3) == 3
    assert naps, "a finished pass did not wait before looking again"


def test_serving_waits_while_somebody_else_has_the_card():
    m = manifest()
    trainer = all_good(m)
    trainer.someone_wants_the_card()
    naps: list[float] = []
    d = driver(trainer, m, naps=naps)

    d.serve(passes=2)

    assert trainer.submitted == []
    assert naps == [IDLE_S, IDLE_S]


def test_serving_picks_up_a_configuration_added_to_the_manifest():
    """The queue is the file. Adding a fourth learning rate is how a sweep is
    extended, and `config_id` is stable under exactly that edit (#113)."""
    narrow = manifest(**{"axes.lrate": [3.0e-5, 1.0e-4], "budget.rungs": [8, 2, 1]})
    wide = manifest(**{"axes.lrate": [3.0e-5, 1.0e-4, 3.0e-4], "budget.rungs": [12, 4, 1]})
    trainer = FakeTrainer(scores_for(wide, [0.200 + i * 0.003 for i in range(12)]))
    d = driver(trainer, narrow)
    manifests = iter([narrow, wide])

    d.serve(passes=2, reload=lambda: next(manifests))

    ran = {r["model_id"].rsplit("-", 1)[-1] for r in trainer.submitted}
    assert {c.config_id for c in wide.configs()} <= ran


def test_serving_stops_at_the_ceiling_rather_than_looping_forever():
    m = manifest(**{"budget.max_total_steps": 8_000})
    d = driver(all_good(m), m)

    assert d.serve(passes=50) < 50


def test_the_outstanding_queue_is_what_has_no_score_yet():
    m = manifest()
    d = driver(all_good(m), m)

    assert len(d.outstanding()) == 12
    d.run()
    assert d.outstanding() == []


def test_a_preempted_configuration_is_back_in_the_queue():
    m = manifest()
    d = driver(all_good(m), m)
    first = m.configs()[0].config_id
    d.state.put(0, first, {"job_id": "j", "status": "cancelled", "preempted": True,
                           "score": None, "raw": None})

    assert first in d.outstanding()


# ── it survives a restart ───────────────────────────────────────────────────
def test_a_served_sweep_resumes_from_its_state_file(tmp_path: Path):
    """#117's first criterion. The state file is what survives the trainer
    restarting, the driver being killed, and the machine rebooting."""
    m = manifest()
    path = tmp_path / "s.json"
    first = FakeTrainer(scores_for(m, [0.200 + i * 0.003 for i in range(12)]))
    state = SweepState.for_manifest(m, "benchmark_cer", path)
    state.train_lines = 120_000
    SweepDriver(m, state, first, sleep=lambda _s: None).serve(passes=1)

    again = FakeTrainer(scores_for(m, [0.200 + i * 0.003 for i in range(12)]))
    reloaded = SweepState.load(path, m, "benchmark_cer")
    SweepDriver(m, reloaded, again, sleep=lambda _s: None).serve(passes=1)

    assert again.submitted == [], "a resumed sweep re-ran finished work"


def test_the_preemption_survives_the_restart_too(tmp_path: Path):
    """A cell given up before a reboot must still be unscored afterwards, or the
    sweep ranks it on part of a budget."""
    m = manifest()
    path = tmp_path / "s.json"
    state = SweepState.for_manifest(m, "benchmark_cer", path)
    state.train_lines = 120_000
    first = m.configs()[0].config_id
    state.put(0, first, {"job_id": "j", "status": "cancelled", "preempted": True,
                         "score": None, "raw": None, "steps": 4000})
    state.save()

    reloaded = SweepState.load(path, m, "benchmark_cer")

    assert json.loads(path.read_text())["results"]["0"][first]["preempted"] is True
    assert reloaded.scored(first, 0) is False


def test_a_preempted_cell_is_submitted_afresh_not_reattached():
    """It keeps the id of the job that was given up. Re-attaching would find a
    cancelled job, record no score, and repeat that on every later pass — the
    configuration would never be measured and nothing would say why."""
    m = manifest()
    trainer = all_good(m)
    d = driver(trainer, m)
    first = m.configs()[0].config_id
    d.state.put(0, first, {"job_id": "given-up", "status": "cancelled",
                           "preempted": True, "score": None, "raw": None,
                           "steps": 4000})

    d.run()

    assert d.state.at(0, first)["job_id"] != "given-up"
    assert d.state.at(0, first)["score"] is not None
    assert not d.state.at(0, first).get("preempted")

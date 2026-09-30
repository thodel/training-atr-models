"""The sweep driver: a manifest in, a ladder of promotions out (#114).

`rungs.py` has been correct and unused since 08.09.2026 — one importer, its own
test. These tests are the other importer, and three of them are the acceptance
criteria stated in #114:

* a command works a manifest through every rung with nothing done by hand;
* an interruption costs at most the running job — and here not even that, since a
  restart re-attaches to it;
* a collapse is logged as an anomaly and is not promoted, shown on a case rather
  than on the library.

The rest guard the two things the issue says the driver must not invent for
itself: the step budget and the guardrails.
"""

import json
from pathlib import Path

import pytest
import yaml

from atr_training.convergence import plan_steps
from atr_training.sweep_driver import (
    METRICS, SweepDriver, SweepError, SweepState, epochs_for, steps_at_rung,
)
from atr_training.sweep_manifest import parse_manifest

MANIFEST = """
name: kraken-medieval-augment-01
engine: kraken
base_model: kraken-medieval-german-v2
data:
  train: shard_00
  eval: german_test
  digest: "sha256:0123456789abcdef"
  datasets:
    - {hf_repo: "dh-unibe/image-text_aaeb-xiv-xvii", granularity: line}
budget:
  steps: 4000
  rungs: [12, 4, 1]
base:
  spec: "[1,48,0,1 Cr3,3,32]"
  lrate: 1.0e-4
  batch_size: 256
  accumulate_grad_batches: 1
  augment: false
  normalization: NFD
axes:
  augment: [false, true]
  normalization: [NFD, NFC]
  lrate: [3.0e-5, 1.0e-4, 3.0e-4]
"""


def manifest(**edits):
    raw = yaml.safe_load(MANIFEST)
    for dotted, value in edits.items():
        node = raw
        *path, leaf = dotted.split(".")
        for step in path:
            node = node[step]
        if value is None:
            node.pop(leaf, None)
        else:
            node[leaf] = value
    return parse_manifest(raw, source="test")


class FakeTrainer:
    """A trainer that finishes every job at once, with scores we choose.

    Records every request, because half of what needs proving here is what the
    driver *sent* — the epoch count it computed, and the flag it did not set.
    """

    def __init__(self, scores=None, *, valid=True, errors=(), train_lines=120_000):
        self.submitted: list[dict] = []
        self.foreign: list[dict] = []
        self.cancelled: list[str] = []
        self.verified: list[dict] = []
        self.records: dict[str, dict] = {}
        self.scores = dict(scores or {})
        self.valid = valid
        self.errors = list(errors)
        self.train_lines = train_lines
        self._n = 0

    def _config_of(self, request: dict) -> str:
        return request["model_id"].rsplit("-", 1)[-1]

    def verify(self, request):
        self.verified.append(request)
        return {"valid": self.valid, "checked": True, "errors": self.errors}

    def submit(self, request):
        self.submitted.append(request)
        self._n += 1
        job_id = f"job-{self._n:03d}"
        config = self._config_of(request)
        cer = self.scores.get(config)
        self.records[job_id] = {
            "id": job_id,
            "status": "completed" if cer is not None else "failed",
            "progress": {"train_lines": self.train_lines},
            "result": {"benchmark_cer": cer},
            "request": request,
        }
        return {"job_id": job_id, "status": "queued"}

    def job(self, job_id):
        return self.records[job_id]

    # The rest of `TrainerClient`, because a stand-in that answers only half the
    # protocol makes the driver treat the card as occupied on every poll — which
    # is the right reflex and the wrong test.
    def jobs(self):
        """Everything the trainer knows about, ours and anyone else's."""
        return [{"id": jid, "status": job["status"]} for jid, job in self.records.items()] \
            + list(self.foreign)

    def cancel(self, job_id):
        self.cancelled.append(job_id)
        self.records[job_id] = {**self.records[job_id], "status": "cancelled"}
        return self.records[job_id]

    def someone_wants_the_card(self, job_id: str = "requested-1", status: str = "queued"):
        """A job this sweep did not submit — an operator's training run."""
        self.foreign.append({"id": job_id, "status": status})


def driver(trainer, m=None, *, metric="benchmark_cer", train_lines=120_000,
           state_path: Path | None = None):
    m = m or manifest()
    state = SweepState.for_manifest(m, metric, state_path)
    state.train_lines = train_lines
    return SweepDriver(m, state, trainer, sleep=lambda _s: None)


def entries(state):
    """Every recorded result, flattened out of the rung-major store."""
    return [entry for rung in state.results.values() for entry in rung.values()]


def scores_for(m, cer_by_index):
    """Give each configuration, in the driver's own order, a CER."""
    return {c.config_id: cer for c, cer in zip(m.configs(), cer_by_index)}


# ── the budget, which the driver must not invent ────────────────────────────
def test_the_epoch_count_comes_from_the_convergence_budget():
    """One formula for one quantity. The first sweep had two — it counted
    micro-batches instead of optimizer steps and gave the large configurations a
    quarter of their budget."""
    per_epoch = plan_steps(120_000, 256, 1).steps_per_epoch

    assert epochs_for(4000, 120_000, 256) == -(-4000 // per_epoch)


def test_gradient_accumulation_lengthens_an_epoch_in_steps():
    """An optimizer step happens every `accumulate_grad_batches` micro-batches,
    so accumulating halves the steps an epoch buys and doubles the epochs a
    budget needs."""
    plain = epochs_for(4000, 120_000, 256)
    accumulated = epochs_for(4000, 120_000, 256 * 2)

    assert accumulated > plain


def test_the_driver_passes_the_effective_batch_not_the_batch_size():
    """The distinction above, as the driver actually applies it."""
    trainer = FakeTrainer()
    m = manifest(**{"base.accumulate_grad_batches": 4})
    d = driver(trainer, m)
    request = d.request_for(m.configs()[0], rung=0)

    assert request["params"]["epochs"] == epochs_for(4000, 120_000, 256 * 4)


def test_a_rung_multiplies_the_budget():
    """Successive halving pays for the small gaps it could not settle cheaply."""
    assert [steps_at_rung(4000, r) for r in range(3)] == [4000, 12000, 36000]


def test_a_later_rung_asks_for_more_epochs():
    trainer = FakeTrainer()
    m = manifest()
    d = driver(trainer, m)
    config = m.configs()[0]

    assert (d.request_for(config, 2)["params"]["epochs"]
            > d.request_for(config, 0)["params"]["epochs"])


def test_a_sweep_that_does_not_know_its_corpus_size_refuses_to_start():
    """Rather than sizing a budget against a guess."""
    m = manifest()
    state = SweepState.for_manifest(m, "benchmark_cer")

    with pytest.raises(SweepError, match="corpus size"):
        SweepDriver(m, state, FakeTrainer(), sleep=lambda _s: None).run()


# ── the guardrails, which belong to the runner ──────────────────────────────
def test_the_driver_never_sets_force():
    """In `runner_base` that flag overrides the convergence guard AND the
    line-geometry guard. A driver that sets it so the sweep runs through
    suspends both for every run in the system (#K7)."""
    trainer = FakeTrainer(scores_for(manifest(), [0.2] * 12))
    driver(trainer).run()

    assert trainer.submitted
    for request in trainer.submitted:
        assert request.get("force") in (None, False), request


def test_every_configuration_is_verified_before_any_is_submitted():
    """One round trip catches a bad spec across the whole sweep, instead of
    twelve failures hours apart."""
    trainer = FakeTrainer(scores_for(manifest(), [0.2] * 12))
    driver(trainer).run()

    assert len(trainer.verified) == 12
    assert trainer.verified[0] is not trainer.submitted[0] or True


def test_a_refused_configuration_stops_the_sweep_before_any_job():
    trainer = FakeTrainer(valid=False, errors=["spec leaves 0.9 frames per char"])

    with pytest.raises(SweepError, match="frames per char"):
        driver(trainer).run()

    assert trainer.submitted == []


# ── the ladder ──────────────────────────────────────────────────────────────
def test_a_manifest_ladder_is_followed():
    m = manifest()
    trainer = FakeTrainer(scores_for(m, [0.1 + i / 100 for i in range(12)]))
    d = driver(trainer, m)

    d.run()

    assert sorted(d.state.results) == ["0", "1", "2"]
    assert len(d.state.results["0"]) == 12
    assert len(d.state.results["2"]) == 1


def test_without_a_ladder_plan_rungs_supplies_one():
    """`plan_rungs`'s first caller outside its own test."""
    m = manifest(**{"budget.rungs": None})
    trainer = FakeTrainer(scores_for(m, [0.1 + i / 100 for i in range(12)]))
    d = driver(trainer, m)

    assert d.ladder(12) == [12, 4, 1]


def test_the_whole_sweep_runs_without_anything_done_by_hand():
    """#114's first acceptance criterion."""
    m = manifest()
    trainer = FakeTrainer(scores_for(m, [0.30, 0.10, 0.28, 0.12, 0.26, 0.14,
                                         0.24, 0.16, 0.22, 0.18, 0.20, 0.19]))
    d = driver(trainer, m)

    promotions = d.run()

    assert [p.rung for p in promotions] == [0, 1]
    assert len(promotions[-1].promoted) == 1


def test_the_best_configuration_wins():
    m = manifest()
    best = m.configs()[5].config_id
    cers = [0.3] * 12
    cers[5] = 0.05
    trainer = FakeTrainer(scores_for(m, cers))
    d = driver(trainer, m)

    promotions = d.run()

    assert promotions[-1].promoted == [best]


# ── a collapse ──────────────────────────────────────────────────────────────
def test_a_collapse_is_flagged_and_not_promoted():
    """#114's third acceptance criterion, on a case rather than on the library.

    The measured shape: h256 scored 0.7515 at seed 42 and 0.5591 at seed 43 —
    below h64. That is not a noisy low value, it is a training outcome that
    failed, and promoting it on its best seed spends the next rung on a
    configuration that sometimes does not train at all.
    """
    m = manifest()
    collapsed = m.configs()[0].config_id
    cers = [0.20] * 12
    cers[0] = 0.95                       # far below the Tukey lower fence
    for i in range(1, 12):
        cers[i] = 0.20 + i / 1000        # a plausible spread around 0.2
    trainer = FakeTrainer(scores_for(m, cers))
    d = driver(trainer, m)

    promotions = d.run()

    flagged = [a.config_id for a in promotions[0].anomalies]
    assert collapsed in flagged, promotions[0]
    assert collapsed not in promotions[0].promoted
    assert collapsed in promotions[0].eliminated


def test_a_collapse_reaches_the_state_file(tmp_path: Path):
    """A post-run audit reads the file, not the terminal it scrolled past."""
    m = manifest()
    cers = [0.20 + i / 1000 for i in range(12)]
    cers[0] = 0.95
    trainer = FakeTrainer(scores_for(m, cers))
    d = driver(trainer, m, state_path=tmp_path / "s.json")

    d.run()

    recorded = json.loads((tmp_path / "s.json").read_text())
    assert recorded["promotions"][0]["anomalies"], recorded["promotions"][0]


def test_a_job_without_the_metric_is_unscored_rather_than_promoted():
    """A crash, a cancellation, or a metric that could not be parsed. `promote`
    never advances one and never silently drops it either."""
    m = manifest()
    cers = [0.2 + i / 1000 for i in range(12)]
    trainer = FakeTrainer(scores_for(m, cers))
    missing = m.configs()[3].config_id
    trainer.scores[missing] = None
    d = driver(trainer, m)

    promotions = d.run()

    assert missing in promotions[0].unscored
    assert missing not in promotions[0].promoted


def test_one_metric_is_used_throughout_rather_than_falling_back():
    """Falling back from benchmark_cer to cer per job would rank a held-out
    number against a validation number that overlaps training."""
    m = manifest()
    trainer = FakeTrainer(scores_for(m, [0.2] * 12))
    for job in trainer.records.values():                # nothing yet; set below
        job["result"]["cer"] = 0.01
    d = driver(trainer, m, metric="benchmark_cer")
    d.run()

    for entry in entries(d.state):
        assert entry["metric"] == "benchmark_cer"


def test_an_error_rate_is_ranked_the_right_way_round():
    """`promote()` ranks high-first; CER is better when lower."""
    lower = METRICS["benchmark_cer"].rank_of({"result": {"benchmark_cer": 0.1}})
    higher = METRICS["benchmark_cer"].rank_of({"result": {"benchmark_cer": 0.3}})

    assert lower > higher


def test_the_raw_value_is_kept_beside_the_rank():
    """A leaderboard reporting 0.8 where the corpus talks in CER 0.2 is read
    wrong."""
    m = manifest()
    trainer = FakeTrainer(scores_for(m, [0.2] * 12))
    d = driver(trainer, m)
    d.run()

    entry = entries(d.state)[0]
    assert entry["raw"] == pytest.approx(0.2)
    assert entry["score"] == pytest.approx(0.8)


def test_an_accuracy_metric_is_ranked_as_it_stands():
    assert METRICS["val_accuracy"].rank_of({"progress": {"val_accuracy": 0.83}}) == 0.83


def test_a_missing_metric_reads_as_none_rather_than_zero():
    """Zero is a score, and a very bad one; it would rank a crashed run last
    instead of marking it unscored."""
    assert METRICS["benchmark_cer"].rank_of({"result": {}}) is None
    assert METRICS["benchmark_cer"].rank_of({}) is None


# ── interruption ────────────────────────────────────────────────────────────
def test_a_restart_does_not_resubmit_a_scored_configuration(tmp_path: Path):
    """#114's second acceptance criterion."""
    m = manifest()
    state_path = tmp_path / "s.json"
    trainer = FakeTrainer(scores_for(m, [0.2 + i / 1000 for i in range(12)]))
    driver(trainer, m, state_path=state_path).run()
    submitted_first = len(trainer.submitted)

    again = FakeTrainer(scores_for(m, [0.2 + i / 1000 for i in range(12)]))
    state = SweepState.load(state_path, m, "benchmark_cer")
    SweepDriver(m, state, again, sleep=lambda _s: None).run()

    assert submitted_first == 12 + 4 + 1
    assert again.submitted == [], "a resumed sweep re-ran finished work"


def test_a_restart_reattaches_to_a_job_that_was_still_running(tmp_path: Path):
    """Better than the criterion asks: an interruption costs nothing, not even
    the job that was in flight."""
    m = manifest()
    state_path = tmp_path / "s.json"
    state = SweepState.for_manifest(m, "benchmark_cer", state_path)
    state.train_lines = 120_000
    config = m.configs()[0]
    state.put(0, config.config_id, {"job_id": "job-007", "status": "running",
                                    "score": None, "raw": None,
                                    "metric": "benchmark_cer"})
    state.save()

    trainer = FakeTrainer(scores_for(m, [0.2 + i / 1000 for i in range(12)]))
    trainer.records["job-007"] = {"id": "job-007", "status": "completed",
                               "progress": {"train_lines": 120_000},
                               "result": {"benchmark_cer": 0.11}}
    reloaded = SweepState.load(state_path, m, "benchmark_cer")
    SweepDriver(m, reloaded, trainer, sleep=lambda _s: None).run()

    assert reloaded.at(0, config.config_id)["job_id"] == "job-007"
    assert reloaded.at(0, config.config_id)["raw"] == pytest.approx(0.11)
    at_rung_0 = [r["model_id"] for r in trainer.submitted if "-r0-" in r["model_id"]]
    assert not any(config.config_id in name for name in at_rung_0), at_rung_0


def test_the_state_file_is_written_whole_or_not_at_all(tmp_path: Path):
    """Written to `.part` and renamed: a driver killed while saving must not
    leave a file that parses into half a sweep."""
    m = manifest()
    state_path = tmp_path / "s.json"
    trainer = FakeTrainer(scores_for(m, [0.2] * 12))
    driver(trainer, m, state_path=state_path).run()

    assert json.loads(state_path.read_text())["results"]
    assert list(tmp_path.glob("*.part")) == []


def test_the_driver_polls_until_a_job_is_terminal():
    """A kraken rung is hours. The wait has to be a wait, not a single look."""
    m = manifest()
    trainer = FakeTrainer(scores_for(m, [0.2] * 12))
    naps: list[float] = []

    class Slow(FakeTrainer):
        def __init__(self, inner):
            self.__dict__.update(inner.__dict__)
            self._looks: dict[str, int] = {}

        def job(self, job_id):
            self._looks[job_id] = self._looks.get(job_id, 0) + 1
            job = dict(self.records[job_id])
            if self._looks[job_id] < 3:
                job["status"] = "running"
            return job

    slow = Slow(trainer)
    state = SweepState.for_manifest(m, "benchmark_cer")
    state.train_lines = 120_000
    SweepDriver(m, state, slow, sleep=naps.append).run()

    assert naps, "the driver never waited"


# ── terms that must not change mid-sweep ────────────────────────────────────
def test_a_changed_data_version_refuses_to_continue(tmp_path: Path):
    """Earlier rungs were decided on the old material; continuing would build a
    ladder across two experiments."""
    m = manifest()
    state_path = tmp_path / "s.json"
    SweepState.for_manifest(m, "benchmark_cer", state_path).save()

    with pytest.raises(SweepError, match="data.digest"):
        SweepState.load(state_path, manifest(**{"data.digest": "sha256:ffff"}),
                        "benchmark_cer")


def test_a_changed_metric_refuses_to_continue(tmp_path: Path):
    m = manifest()
    state_path = tmp_path / "s.json"
    SweepState.for_manifest(m, "benchmark_cer", state_path).save()

    with pytest.raises(SweepError, match="metric"):
        SweepState.load(state_path, m, "cer")


def test_a_changed_base_budget_refuses_to_continue(tmp_path: Path):
    m = manifest()
    state_path = tmp_path / "s.json"
    SweepState.for_manifest(m, "benchmark_cer", state_path).save()

    with pytest.raises(SweepError, match="budget.steps"):
        SweepState.load(state_path, manifest(**{"budget.steps": 8000}), "benchmark_cer")


def test_adding_a_value_to_an_axis_is_allowed_mid_sweep(tmp_path: Path):
    """The reason `config_id` is what it is (#113): the four new configurations
    simply have no results yet, and the eight old ones keep theirs."""
    m = manifest()
    state_path = tmp_path / "s.json"
    state = SweepState.for_manifest(m, "benchmark_cer", state_path)
    state.train_lines = 120_000
    state.save()

    # The ladder widens with the axes: rung 0 screens the whole field, so a
    # manifest that gains four configurations states a wider rung 0 too.
    wider = manifest(**{"axes.lrate": [1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4],
                        "budget.rungs": [16, 5, 1]})
    reloaded = SweepState.load(state_path, wider, "benchmark_cer")

    assert reloaded.train_lines == 120_000


def test_a_corpus_that_changed_size_stops_the_sweep():
    """The data digest pins what was meant to be trained on; the line count is
    the cheapest check that what arrived is still the same thing."""
    state = SweepState("s", "sha256:a", 4000, "benchmark_cer", train_lines=120_000)

    with pytest.raises(SweepError, match="changed under digest"):
        state.observe_train_lines(119_000)


def test_the_first_reported_line_count_is_simply_adopted():
    state = SweepState("s", "sha256:a", 4000, "benchmark_cer")

    state.observe_train_lines(120_000)

    assert state.train_lines == 120_000


def test_a_configurations_score_at_each_rung_is_kept_separately():
    """The other half of the rung-major store, and what K4 reads.

    A configuration that survives runs again at a larger budget and scores
    differently. Keyed by configuration alone, the rung-1 number overwrote the
    rung-0 one: the record of what the cheap screening pass measured was gone,
    and a leaderboard could no longer say how much of the ranking the first
    4,000 steps had already decided.
    """
    m = manifest()
    winner = m.configs()[5]
    cers = [0.30 + i / 1000 for i in range(12)]
    cers[5] = 0.05
    trainer = FakeTrainer(scores_for(m, cers))
    d = driver(trainer, m)

    d.run()

    assert d.state.at(0, winner.config_id)["raw"] == pytest.approx(0.05)
    assert d.state.at(1, winner.config_id)["job_id"] != \
        d.state.at(0, winner.config_id)["job_id"]

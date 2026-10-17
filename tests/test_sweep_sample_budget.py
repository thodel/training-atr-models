"""A budget in samples, for a sweep whose effective batch is on an axis (#138).

The first rung 0 of the XIX sweep gave every cell 3,000 optimizer steps. At equal
steps the effective batch of 64 sees four times the lines of the batch of 16, so
the ladder would have ranked data, not learning rates: 0.3010 at eff 16 against
0.1455 at eff 32 is mostly the second cell having read twice as much. These pin
that a ``samples`` budget gives every cell the same number of lines, that the
ceiling counts what the budget counts, and that a state file cannot switch unit
underneath its earlier rungs.
"""

from __future__ import annotations

import pytest
import yaml

from atr_training.sweep_driver import (
    SweepDriver, SweepError, SweepState, cell_steps, effective_batch,
)
from atr_training.sweep_manifest import ManifestError, parse_manifest

MANIFEST = """
name: vlm-samples-01
engine: vllm
base_model: Qwen/Qwen3-VL-4B-Instruct
data:
  train: t
  eval: e
  digest: "sha256:0123456789abcdef"
  datasets:
    - {hf_repo: "dh-unibe/image-text_kurrent-xix", granularity: page}
budget:
  samples: 48000
  rungs: [3, 1]
  max_total_samples: 336000
base:
  granularity: line
  batch_size: 16
  accumulate_grad_batches: 1
  lrate: 2.0e-4
  seed: 42
axes:
  effective_batch:
    - {batch_size: 16, accumulate_grad_batches: 1, lrate: 2.0e-4}
    - {batch_size: 16, accumulate_grad_batches: 2, lrate: 4.0e-4}
    - {batch_size: 16, accumulate_grad_batches: 4, lrate: 8.0e-4}
"""


def raw(**budget):
    data = yaml.safe_load(MANIFEST)
    if budget:
        data["budget"] = {"rungs": [3, 1], **budget}
    return data


def manifest(**budget):
    return parse_manifest(raw(**budget), source="test")


class FakeTrainer:
    def __init__(self):
        self.submitted = []

    def verify(self, request):
        return {"valid": True}

    def submit(self, request):
        self.submitted.append(request)
        return {"job_id": f"job-{len(self.submitted)}", "status": "queued"}

    def job(self, job_id):
        return {"id": job_id, "status": "completed", "metrics": {"cer": 0.2},
                "progress": {"train_lines": 405_000}}

    def jobs(self):
        return []

    def cancel(self, job_id):
        return {}


def driver(m, trainer=None):
    state = SweepState.for_manifest(m, "cer")
    state.train_lines = 405_000
    return SweepDriver(m, state, trainer or FakeTrainer(), sleep=lambda _s: None)


# ── the manifest ────────────────────────────────────────────────────────────
def test_samples_are_a_budget_unit():
    m = manifest()
    assert m.budget_unit == "samples" and m.steps == 48000
    assert m.max_total_steps == 336000


def test_a_steps_manifest_is_unchanged():
    m = manifest(steps=3000, max_total_steps=22000)
    assert m.budget_unit == "steps" and m.steps == 3000


@pytest.mark.parametrize("budget", [{}, {"steps": 3000, "samples": 48000}])
def test_exactly_one_unit(budget):
    data = raw()
    data["budget"] = {"rungs": [3, 1], **budget}
    with pytest.raises(ManifestError, match="exactly one"):
        parse_manifest(data, source="test")


def test_a_ceiling_in_the_other_unit_is_refused():
    with pytest.raises(ManifestError, match="max_total_samples"):
        manifest(samples=48000, max_total_steps=22000)
    with pytest.raises(ManifestError, match="max_total_steps"):
        manifest(steps=3000, max_total_samples=48000)


# ── the cells ───────────────────────────────────────────────────────────────
def test_every_cell_sees_the_same_samples():
    m = manifest()
    d = driver(m)
    for config in m.configs():
        steps = d.request_for(config, 0)["params"]["max_steps"]
        assert steps * effective_batch(config.params) == 48000


def test_the_xix_cells_get_3000_1500_750_and_three_times_that_at_rung_1():
    m = manifest()
    d = driver(m)
    by_batch = {effective_batch(c.params): c for c in m.configs()}
    assert [d.request_for(by_batch[b], 0)["params"]["max_steps"] for b in (16, 32, 64)] \
        == [3000, 1500, 750]
    assert [d.request_for(by_batch[b], 1)["params"]["max_steps"] for b in (16, 32, 64)] \
        == [9000, 4500, 2250]


def test_a_budget_that_does_not_divide_rounds_up():
    """Never fewer samples than the budget: a cell short-changed by a remainder
    would lose by arithmetic."""
    assert cell_steps(1000, "samples", {"batch_size": 3}, 0) == 334
    assert cell_steps(1000, "steps", {"batch_size": 3}, 0) == 1000


def test_the_entry_records_steps_and_samples():
    m = manifest()
    trainer = FakeTrainer()
    d = driver(m, trainer)
    d.run()
    assert sorted(e["samples"] for e in d.state.results["0"].values()) == [48000] * 3
    assert [e["samples"] for e in d.state.results["1"].values()] == [144000]


# ── the ceiling counts what the budget counts ───────────────────────────────
def test_the_ceiling_counts_samples():
    m = manifest(samples=48000, max_total_samples=96000)
    d = driver(m)
    for config in m.configs()[:2]:
        d.state.put(0, config.config_id, {"status": "completed", "steps": 1,
                                          "samples": 48000})
    assert "96,000 ceiling" in d.over_ceiling()
    assert "training samples" in d.over_ceiling()


def test_under_the_ceiling_it_runs():
    m = manifest()
    d = driver(m)
    assert d.over_ceiling() is None


# ── a state file cannot change unit underneath its rungs ────────────────────
def test_a_steps_state_refuses_a_samples_manifest(tmp_path):
    steps = manifest(steps=3000, max_total_steps=22000)
    path = tmp_path / "s.json"
    SweepState.for_manifest(steps, "cer", path).save()
    with pytest.raises(SweepError, match="budget.unit"):
        SweepState.load(path, manifest(), "cer")


def test_an_old_state_file_without_a_unit_reads_as_steps(tmp_path):
    import json
    steps = manifest(steps=3000, max_total_steps=22000)
    path = tmp_path / "s.json"
    SweepState.for_manifest(steps, "cer", path).save()
    data = json.loads(path.read_text())
    data.pop("budget_unit")
    path.write_text(json.dumps(data))
    assert SweepState.load(path, steps, "cer").budget_unit == "steps"

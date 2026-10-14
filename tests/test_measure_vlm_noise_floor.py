"""The VLM noise floor goes through the trainer (#138, #115), without a GPU.

What can go wrong silently: a request that differs from the sweep's own cell in
anything but the seed measures the variance of two things at once; a restart
that submits the seeds again pays for hours twice; a collapsed run with a
spread of 0.0000 reads as perfect resolution.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from atr_training.sweep_manifest import load_manifest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "measure_vlm_noise_floor.py"
MANIFEST = ROOT / "config" / "sweeps" / "vlm-xix-effective-batch-01.yaml"

spec = importlib.util.spec_from_file_location("measure_vlm_noise_floor", SCRIPT)
mvnf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mvnf)


class FakeTrainer:
    """Finishes every job at once, with the scores we choose per seed."""

    def __init__(self, scores: dict[int, tuple[float | None, float | None]], *,
                 valid: bool = True):
        self.scores = scores
        self.valid = valid
        self.submitted: list[dict] = []
        self.verified: list[dict] = []
        self.records: dict[str, dict] = {}

    def verify(self, request):
        self.verified.append(request)
        return {"valid": self.valid, "errors": [] if self.valid else ["no such base"]}

    def submit(self, request):
        self.submitted.append(request)
        seed = request["params"]["seed"]
        job_id = f"job-s{seed}"
        cer, ratio = self.scores[seed]
        self.records[job_id] = {
            "id": job_id, "status": "completed" if cer is not None else "failed",
            "metrics": {"cer": cer, "length_ratio": ratio},
            "started_at": "2026-10-12T10:00:00Z", "finished_at": "2026-10-12T12:00:00Z",
            "code": {"commit": "abc1234"},
        }
        return {"job_id": job_id, "status": "queued"}

    def job(self, job_id):
        return self.records[job_id]


@pytest.fixture
def manifest():
    return load_manifest(MANIFEST)


@pytest.fixture
def cell(manifest):
    return mvnf.pick_cell(manifest, None)


# ── the request is the cell's, seed aside ───────────────────────────────────
def test_only_the_seed_the_name_and_the_note_differ(manifest, cell):
    requests = mvnf.requests_for(manifest, cell, [42, 43, 44], "cer")
    first = requests[42]
    for seed, other in requests.items():
        assert other["params"]["seed"] == seed
        assert {k for k in first if first[k] != other[k]} <= {"model_id", "notes", "params"}
        assert {k for k in first["params"] if first["params"][k] != other["params"][k]} \
            <= {"seed"}
        assert other["params"]["eval_seed"] == first["params"]["eval_seed"]
        assert other["model_id"].endswith(f"-noise-s{seed}")


def test_every_seed_scores_the_same_draw(manifest, cell):
    """Four seeds, four draws was the first measurement's mistake: the draw is
    taken with the training seed unless `eval_seed` says otherwise."""
    requests = mvnf.requests_for(manifest, cell, [42, 43, 44, 45], "cer")
    assert {r["params"]["eval_seed"] for r in requests.values()} == {42}
    assert [r["params"]["seed"] for r in requests.values()] == [42, 43, 44, 45]


def test_the_request_is_the_sweep_s_own_rung_zero(manifest, cell):
    """Built by the driver's `request_for`, so the floor and the cells cannot
    drift: same datasets, same budget, same base."""
    from atr_training.sweep_driver import SweepDriver, SweepState

    ours = mvnf.requests_for(manifest, cell, [42], "cer")[42]
    state = SweepState.for_manifest(manifest, "cer")
    theirs = SweepDriver(manifest, state, mvnf._NoTrainer()).request_for(cell, 0)
    assert ours["datasets"] == theirs["datasets"]
    assert ours["base_model"] == theirs["base_model"]
    assert ours["params"]["max_steps"] == theirs["params"]["max_steps"] == manifest.steps
    assert "force" not in ours


def test_the_default_cell_is_the_baseline_s_setting(manifest, cell):
    """The manifest reserves the first cell for v2's own hyperparameters."""
    assert cell.axes["effective_batch"] == {"accumulate_grad_batches": 1,
                                            "batch_size": 16, "lrate": 2.0e-4}


def test_a_cell_can_be_named_by_a_prefix(manifest):
    full = manifest.configs()[1].config_id
    assert mvnf.pick_cell(manifest, full[:6]).config_id == full
    with pytest.raises(SystemExit, match="matches 0"):
        mvnf.pick_cell(manifest, "zzzz")


def test_a_kraken_manifest_is_sent_to_the_other_script():
    from tests.test_sweep_driver import manifest as kraken_manifest
    m = kraken_manifest()
    with pytest.raises(SystemExit, match="measure_noise_floor.py"):
        mvnf.requests_for(m, m.configs()[0], [42, 43], "cer")


# ── the measurement ─────────────────────────────────────────────────────────
def test_the_spread_excludes_a_collapsed_run(manifest, cell, tmp_path):
    trainer = FakeTrainer({42: (0.050, 1.0), 43: (0.052, 1.0), 44: (0.049, 1.0),
                           45: (1.0, 0.0)})
    requests = mvnf.requests_for(manifest, cell, [42, 43, 44, 45], "cer")
    state = mvnf.NoiseState(tmp_path / "noise.json")
    mvnf.submit_missing(trainer, requests, state, cell, manifest.data_digest)
    jobs = mvnf.await_all(trainer, state.jobs, sleep=lambda _s: None)
    runs = mvnf.read_runs(jobs, "cer")
    summary = mvnf.summarise(manifest, cell, requests, runs, "cer")

    assert summary["collapsed_seeds"] == [45]
    assert summary["noise_floor"]["n"] == 3
    assert summary["noise_floor"]["spread"] == pytest.approx(0.003)
    assert summary["steps"] == 3000 and summary["cell"] == cell.config_id
    assert "seed" not in summary["params"]


def test_fewer_than_two_usable_runs_state_no_floor(manifest, cell, tmp_path):
    trainer = FakeTrainer({42: (0.050, 1.0), 43: (None, None)})
    requests = mvnf.requests_for(manifest, cell, [42, 43], "cer")
    state = mvnf.NoiseState(tmp_path / "noise.json")
    mvnf.submit_missing(trainer, requests, state, cell, manifest.data_digest)
    runs = mvnf.read_runs(mvnf.await_all(trainer, state.jobs, sleep=lambda _s: None), "cer")
    assert "noise_floor" not in mvnf.summarise(manifest, cell, requests, runs, "cer")


# ── a restart re-attaches ───────────────────────────────────────────────────
def test_a_restart_submits_nothing_twice(manifest, cell, tmp_path):
    requests = mvnf.requests_for(manifest, cell, [42, 43], "cer")
    first = FakeTrainer({42: (0.05, 1.0), 43: (0.05, 1.0)})
    mvnf.submit_missing(first, requests, mvnf.NoiseState(tmp_path / "n.json"),
                        cell, manifest.data_digest)
    assert len(first.submitted) == 2

    again = FakeTrainer({42: (0.05, 1.0), 43: (0.05, 1.0)})
    state = mvnf.NoiseState(tmp_path / "n.json")
    mvnf.submit_missing(again, requests, state, cell, manifest.data_digest)
    assert again.submitted == []
    assert state.jobs == {42: "job-s42", 43: "job-s43"}


def test_a_new_seed_joins_the_old_jobs(manifest, cell, tmp_path):
    requests = mvnf.requests_for(manifest, cell, [42, 43], "cer")
    mvnf.submit_missing(FakeTrainer({42: (0.05, 1.0), 43: (0.05, 1.0)}), requests,
                        mvnf.NoiseState(tmp_path / "n.json"), cell, manifest.data_digest)
    more = FakeTrainer({42: (0.05, 1.0), 43: (0.05, 1.0), 44: (0.05, 1.0)})
    state = mvnf.NoiseState(tmp_path / "n.json")
    mvnf.submit_missing(more, mvnf.requests_for(manifest, cell, [42, 43, 44], "cer"),
                        state, cell, manifest.data_digest)
    assert [r["params"]["seed"] for r in more.submitted] == [44]


def test_the_state_refuses_another_cell_or_corpus(manifest, cell, tmp_path):
    (tmp_path / "n.json").write_text(json.dumps(
        {"cell": "other", "data_digest": manifest.data_digest, "jobs": {"42": "job-1"}}))
    with pytest.raises(SystemExit, match="belongs to one configuration"):
        mvnf.NoiseState(tmp_path / "n.json").check_against(cell, manifest.data_digest)


def test_a_refusal_inside_a_200_stops_the_measurement(manifest, cell, tmp_path):
    """`/jobs/verify` answers 200 with `valid: false`; a client that only looks
    at the status code would submit four refused jobs."""
    trainer = FakeTrainer({}, valid=False)
    requests = mvnf.requests_for(manifest, cell, [42, 43], "cer")
    with pytest.raises(SystemExit, match="no such base"):
        mvnf.submit_missing(trainer, requests, mvnf.NoiseState(tmp_path / "n.json"),
                            cell, manifest.data_digest)
    assert trainer.submitted == []


def test_a_dry_run_verifies_and_submits_nothing(manifest, cell, tmp_path):
    trainer = FakeTrainer({})
    requests = mvnf.requests_for(manifest, cell, [42, 43], "cer")
    state = mvnf.NoiseState(tmp_path / "n.json")
    mvnf.submit_missing(trainer, requests, state, cell, manifest.data_digest, dry_run=True)
    assert len(trainer.verified) == 2 and trainer.submitted == [] and state.jobs == {}


# ── into the manifest, with provenance and without losing the comments ──────
def measured(manifest, cell, tmp_path):
    trainer = FakeTrainer({42: (0.050, 1.0), 43: (0.052, 1.0), 44: (0.049, 1.0)})
    requests = mvnf.requests_for(manifest, cell, [42, 43, 44], "cer")
    state = mvnf.NoiseState(tmp_path / "n.json")
    mvnf.submit_missing(trainer, requests, state, cell, manifest.data_digest)
    runs = mvnf.read_runs(mvnf.await_all(trainer, state.jobs, sleep=lambda _s: None), "cer")
    return mvnf.summarise(manifest, cell, requests, runs, "cer")


def test_the_floor_is_spliced_in_and_parses_back(manifest, cell, tmp_path):
    from atr_training.noise_floor import write_floor

    copy = tmp_path / "sweep.yaml"
    shutil.copy(MANIFEST, copy)
    before = copy.read_text(encoding="utf-8")
    write_floor(copy, mvnf.manifest_block(measured(manifest, cell, tmp_path)))
    after = copy.read_text(encoding="utf-8")

    parsed = load_manifest(copy)
    assert parsed.noise_floor == pytest.approx(0.003)
    assert parsed.noise_floor_provenance["cell"] == cell.config_id
    assert parsed.noise_floor_provenance["steps"] == 3000
    assert parsed.noise_floor_provenance["metric"] == "cer"
    assert parsed.noise_floor_provenance["eval_seed"] == 42
    assert parsed.noise_floor_provenance["provenance"] == "measured"
    # Every comment line survived; the reasoning is in them.
    assert [ln for ln in before.splitlines() if ln.lstrip().startswith("#")] == \
           [ln for ln in after.splitlines() if ln.lstrip().startswith("#")]


def test_a_floor_from_other_data_is_refused(manifest, cell, tmp_path):
    from atr_training.noise_floor import write_floor

    copy = tmp_path / "sweep.yaml"
    shutil.copy(MANIFEST, copy)
    block = {**mvnf.manifest_block(measured(manifest, cell, tmp_path)),
             "measured_on": "sha256:" + "f" * 64}
    with pytest.raises(ValueError, match="license a ranking"):
        write_floor(copy, block)

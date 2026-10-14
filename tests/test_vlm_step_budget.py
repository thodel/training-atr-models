"""A VLM sweep rung budgets in optimizer steps, and the engine now takes them (#138).

Found on 12.10.2026, one check before the first run: the XIX sweep's rung 0 says
3,000 steps, and the only budget the VLM engine took was epochs. On a 405k-line
corpus at effective batch 16 the smallest epoch count is ~25,000 steps, so every
"3,000-step" cell would have trained a whole epoch — three cells at eight times
the budget, and rung 1 identical to rung 0. These pin that the step budget
reaches the Trainer, that the convergence guard judges the capped run, and that
the driver hands a step-taking engine its steps.
"""

from __future__ import annotations

import pytest
import yaml

from atr_training.contracts import VLM_BASE_MODEL, VlmTrainParams
from atr_training.convergence import check_convergence, plan_steps
from atr_training.sweep_driver import SweepDriver, SweepState, steps_at_rung
from atr_training.sweep_manifest import parse_manifest
from atr_training.vlm_cmd import evaluate_cmd, train_cmd
from vlm_train_svc.train_qlora import parse_args as parse_train_args

VLM_MANIFEST = """
name: vlm-test-01
engine: vllm
base_model: Qwen/Qwen3-VL-4B-Instruct
data:
  train: four corpora
  eval: their validation partition
  digest: "sha256:0123456789abcdef"
  datasets:
    - {hf_repo: "dh-unibe/image-text_kurrent-xix", granularity: page}
budget:
  steps: 3000
  rungs: [3, 1]
base:
  granularity: line
  batch_size: 16
  accumulate_grad_batches: 1
  lrate: 2.0e-4
  seed: 42
axes:
  lrate: [2.0e-4, 4.0e-4, 8.0e-4]
"""


def vlm_manifest():
    return parse_manifest(yaml.safe_load(VLM_MANIFEST), source="test")


class NoTrainer:
    def __getattr__(self, name):
        raise AssertionError(f"the trainer must not be called here: {name}")


def driver(m, train_lines=None):
    state = SweepState.for_manifest(m, "cer")
    state.train_lines = train_lines
    return SweepDriver(m, state, NoTrainer(), sleep=lambda _s: None)


# ── the contract and the command ────────────────────────────────────────────
def test_the_budget_reaches_the_engine_s_argv():
    cmd = train_cmd("python", params=VlmTrainParams(max_steps=3000), base_model=VLM_BASE_MODEL,
                    train_jsonl="/j/t.jsonl", val_jsonl="/j/v.jsonl", data_root="/j",
                    output_dir="/o")
    assert cmd[cmd.index("--max-steps") + 1] == "3000"
    assert parse_train_args(cmd[3:]).max_steps == 3000


def test_without_a_budget_the_flag_is_absent_and_the_engine_reads_none():
    cmd = train_cmd("python", params=VlmTrainParams(), base_model=VLM_BASE_MODEL,
                    train_jsonl="/j/t.jsonl", val_jsonl="/j/v.jsonl", data_root="/j",
                    output_dir="/o")
    assert "--max-steps" not in cmd
    assert parse_train_args(cmd[3:]).max_steps is None


def test_the_eval_command_never_sees_the_budget():
    """`evaluate_qlora` has no such flag and exits 2 on an unknown one."""
    cmd = evaluate_cmd("python", params=VlmTrainParams(max_steps=3000),
                       base_model=VLM_BASE_MODEL, adapter_dir="/o", val_jsonl="/j/v.jsonl",
                       data_root="/j", report="/j/r.json")
    assert "--max-steps" not in cmd


def test_a_budget_below_one_step_is_refused():
    with pytest.raises(ValueError):
        VlmTrainParams(max_steps=0)


# ── the guard judges the run that will happen ───────────────────────────────
def test_the_cap_bounds_the_total_and_not_the_epoch():
    budget = plan_steps(405_000, 16, 1, cap=3000)
    assert budget.steps_per_epoch == 25_313
    assert budget.total_steps == 3000


def test_a_cap_above_the_epochs_changes_nothing():
    assert plan_steps(783, 16, 1, cap=3000).total_steps == plan_steps(783, 16, 1).total_steps


def test_the_guard_refuses_a_capped_run_below_the_floor_and_says_so():
    """25,000 steps of epochs would pass; 10 of cap do not, and the reason has
    to name the cap or the reader raises epochs and changes nothing."""
    verdict = check_convergence("vllm", from_scratch=False, train_lines=405_000,
                                effective_batch=16, epochs=1, max_steps=10)
    assert verdict.ok is False
    assert verdict.budget.total_steps == 10
    assert "max_steps 10" in verdict.reason


def test_the_xix_rung_passes_the_guard_at_its_real_budget():
    verdict = check_convergence("vllm", from_scratch=False, train_lines=405_000,
                                effective_batch=16, epochs=1, max_steps=3000)
    assert verdict.ok is True
    assert verdict.budget.total_steps == 3000


# ── the driver hands a step-taking engine its steps ─────────────────────────
def test_a_vlm_cell_carries_the_rung_budget_as_max_steps():
    m = vlm_manifest()
    request = driver(m, train_lines=405_000).request_for(m.configs()[0], rung=0)
    assert request["params"]["max_steps"] == 3000
    assert request["params"]["epochs"] == 1


def test_a_later_rung_triples_the_cap_not_the_epochs():
    m = vlm_manifest()
    d = driver(m, train_lines=405_000)
    config = m.configs()[0]
    assert d.request_for(config, 1)["params"]["max_steps"] == steps_at_rung(3000, 1)
    assert d.request_for(config, 1)["params"]["epochs"] == 1


def test_the_epochs_still_cover_a_cap_longer_than_the_corpus():
    """A small corpus and a big budget: the Trainer repeats the data, and the
    epoch count has to say so or the guard's total_steps lies."""
    m = vlm_manifest()
    request = driver(m, train_lines=783).request_for(m.configs()[0], rung=0)
    assert request["params"]["max_steps"] == 3000
    assert request["params"]["epochs"] == -(-3000 // -(-783 // 16))


def test_a_vlm_sweep_needs_no_corpus_size_to_build_a_request():
    """`epochs` was the only reason the kraken driver had to know it."""
    m = vlm_manifest()
    request = driver(m, train_lines=None).request_for(m.configs()[0], rung=0)
    assert request["params"] == {**request["params"], "max_steps": 3000, "epochs": 1}
    assert driver(m, train_lines=None).budgets_in_steps is True


def test_a_kraken_sweep_still_budgets_in_epochs():
    from tests.test_sweep_driver import manifest as kraken_manifest
    m = kraken_manifest()
    d = driver(m, train_lines=120_000)
    assert d.budgets_in_steps is False
    assert "max_steps" not in d.request_for(m.configs()[0], rung=0)["params"]


# ── the evaluation draw is its own decision (#138) ──────────────────────────
def test_the_draw_follows_eval_seed_when_given():
    from vlm_train_svc.runner import Pipeline  # noqa: F401 — the field, not the class
    assert VlmTrainParams().eval_seed is None
    assert VlmTrainParams(seed=43, eval_seed=42).eval_seed == 42


def test_eval_seed_is_not_part_of_the_argv():
    """It decides which rows the runner writes to val_eval.jsonl; the engine
    never sees it, so an unknown flag cannot kill a run after the queue."""
    cmd = train_cmd("python", params=VlmTrainParams(eval_seed=7), base_model=VLM_BASE_MODEL,
                    train_jsonl="/j/t.jsonl", val_jsonl="/j/v.jsonl", data_root="/j",
                    output_dir="/o")
    assert not any("eval" in c and "seed" in c for c in cmd)

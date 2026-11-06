"""`measure_vlm_arms --like/--card`: a real run's geometry, on a chosen card (#379).

The peak table in docs/WHERE_A_RUN_RUNS.md has rows that are lower bounds or
H100 figures. Turning one into an A40 point means measuring *that* run's batch,
granularity and pixel budget, not #137's fixed page arms. These cover the
parameters and the plan; the run itself needs a GPU and an engine venv.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "measure_vlm_arms_like", REPO / "scripts" / "measure_vlm_arms.py")
mva = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mva
_SPEC.loader.exec_module(mva)

#: The request of `20261029T133027Z-qwen3.5-4b-german-kurrent-xvi-xix-page-v2`,
#: trimmed to what decides a step, plus a step budget the measurement must drop.
LIKE = {"granularity": "page", "load_in_4bit": False, "batch_size": 2,
        "accumulate_grad_batches": 1, "max_steps": 100, "epochs": 3,
        "save_steps": 50, "lora_r": 16, "obsolete_key": "an old record"}


def test_without_like_the_arms_are_137s():
    """The 03.10.2026 figures in the docstring must stay reproducible."""
    params = mva.arm_params(False, 1)
    assert params.granularity == "page"
    assert (params.batch_size, params.accumulate_grad_batches) == (1, 16)
    assert (params.epochs, params.max_epochs, params.save_steps) == (1, 1, 10_000)


def test_like_keeps_the_runs_geometry_and_drops_its_step_budget():
    params = mva.arm_params(False, 1, LIKE)
    assert (params.granularity, params.batch_size, params.accumulate_grad_batches,
            params.lora_r) == ("page", 2, 1, 16)
    # 100 steps over 96 samples would loop the selection; one epoch is the unit.
    assert params.max_steps is None
    assert (params.epochs, params.max_epochs, params.save_steps) == (1, 1, 10_000)


def test_load_like_reads_the_request(tmp_path):
    record = tmp_path / "job.json"
    record.write_text(json.dumps({"id": "x", "request": {
        "engine": "vllm", "base_model": "Qwen/Qwen3.5-4B", "params": LIKE}}))
    params, base = mva.load_like(record)
    assert base == "Qwen/Qwen3.5-4B"
    assert params["batch_size"] == 2


def test_like_is_one_arm_quantised_as_the_run_was():
    assert mva.plan_arms(1, LIKE) == [("like_job", "1", False)]
    assert mva.plan_arms(0, {**LIKE, "load_in_4bit": True}) == [("like_job", "0", True)]


def test_card_moves_the_one_card_arms_and_leaves_the_two_card_arm():
    """On 06.11.2026 a ketos run held card 0; the arms were hard-wired to it."""
    assert mva.plan_arms(1, None) == [("fourbit_one_card", "1", True),
                                      ("bf16_one_card", "1", False),
                                      ("bf16_two_cards", "0,1", False)]

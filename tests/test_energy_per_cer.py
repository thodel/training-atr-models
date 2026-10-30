"""#333: energy and CER side by side, only where the two CER values compare.

The number the issue wanted — kWh per CER point along the ladder — is cheap to
compute and easy to get wrong in three ways, each of which has a test here. A
cost can be paired with the wrong CER: the issue's own draft put the Qwen3.8-27B
energy next to the Qwen3.5-27B CER, so pairing goes by the record and by the
draw, never by the arm's name. A run with a CER and no energy can be skipped in
silence, which makes the ladder look complete; it is named under the table
instead. And a worse, costlier run can be stepped over as if it had never been
trained; it is listed as dominated, with infinite cost per point.

Offline: tmp_path job stores with the two files the pairing reads.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from atr_training.energy_per_cer import (
    Run,
    as_json,
    ladders,
    load_store,
    markdown,
    run_of,
)

DRAW_A = "line one\nline two\n"
DRAW_B = "line one\nline three\n"


def _job(root: Path, job_id: str, *, base: str, cer: float | None, wh: float | None,
         draw: str = DRAW_A, reserved: int | None = 3173, four_bit: bool = True,
         status: str = "completed", host: str = "ubelix", shared: bool = False,
         granularity: str = "line", train_wh: float | None = None) -> Path:
    job_dir = root / job_id
    (job_dir / "data").mkdir(parents=True)
    (job_dir / "data" / "val_eval.jsonl").write_text(draw, encoding="utf-8")
    progress = {"reserved_pages": reserved, "gpu_shared": shared}
    if wh is not None:
        progress["energy_wh"] = wh
    stages = [{"name": "prepare", "status": "completed"},
              {"name": "train", "status": "completed",
               "started_at": "2026-10-15T15:57:00Z", "finished_at": "2026-10-15T22:01:58Z",
               "energy_wh": ({"gpu0": train_wh} if train_wh is not None else {})},
              {"name": "test", "status": "completed"}]
    record = {"id": job_id, "status": status, "host": host,
              "request": {"base_model": base, "params": {"granularity": granularity,
                                                          "load_in_4bit": four_bit}},
              "progress": progress, "stages": stages,
              "metrics": ({"cer": cer, "samples": 200} if cer is not None else None)}
    (job_dir / "job.json").write_text(json.dumps(record), encoding="utf-8")
    return job_dir


@pytest.fixture
def ladder_store(tmp_path: Path) -> Path:
    """The medieval ladder as the records have it: three measured 4-bit arms, one
    bf16 arm, and the Qwen3.5-27B arm whose record carries no energy."""
    root = tmp_path / "jobs"
    root.mkdir()
    _job(root, "4b", base="Qwen/Qwen3.5-4B", cer=0.1594, wh=2377.38, train_wh=2368.8)
    _job(root, "9b", base="Qwen/Qwen3.5-9B", cer=0.1375, wh=2972.96)
    _job(root, "27b-38", base="Qwen/Qwen3.8-27B", cer=0.1215, wh=8606.46)
    _job(root, "27b-35", base="Qwen/Qwen3.5-27B", cer=0.1224, wh=None)
    _job(root, "4b-bf16", base="Qwen/Qwen3.5-4B", cer=0.1474, wh=2101.96, four_bit=False)
    return root


def test_a_record_is_read_with_both_axes_and_the_train_stage(ladder_store: Path):
    run = run_of(ladder_store / "4b", store="scratch")
    assert run is not None
    assert run.measured
    assert run.kwh == pytest.approx(2.37738)
    assert run.cer_points == pytest.approx(15.94)
    assert run.train_wh == pytest.approx(2368.8)
    assert run.train_hours == pytest.approx(6.08, abs=0.01)
    assert run.draw_lines == 2 and run.host == "ubelix"


def test_the_four_bit_ladder_prices_a_cer_point_as_the_issue_computed(ladder_store: Path):
    runs = load_store(ladder_store, store="scratch")
    [lad] = ladders(runs, filters={"load_in_4bit": "true"})
    assert [r.id for r in lad.frontier] == ["4b", "9b", "27b-38"]
    first, second = lad.steps
    assert first.delta_kwh == pytest.approx(0.5956, abs=1e-3)
    assert first.delta_points == pytest.approx(-2.19)
    assert first.kwh_per_point == pytest.approx(0.27, abs=0.005)
    assert second.kwh_per_point == pytest.approx(3.52, abs=0.01)


def test_the_arm_without_energy_is_named_not_priced_and_not_skipped_in_silence(
        ladder_store: Path):
    """The 12.24 % of Qwen3.5-27B has no kilowatt-hours behind it. The issue's
    draft paired it with the Qwen3.8 arm's 8.59 kWh; the table must not."""
    runs = load_store(ladder_store, store="scratch")
    [lad] = ladders(runs, filters={"load_in_4bit": "true"})
    assert [r.id for r in lad.unmeasured] == ["27b-35"]
    text = markdown(lad)
    assert "`27b-35`" in text and "no energy figure" in text
    assert "12.24 %" in text
    # and it never sits in a priced row
    priced = [line for line in text.splitlines() if line.startswith("| Qwen3.5-27B")]
    assert priced == []


def test_a_costlier_worse_run_is_dominated_and_the_frontier_steps_past_it(ladder_store: Path):
    runs = load_store(ladder_store, store="scratch")
    [lad] = ladders(runs)
    assert [r.id for r in lad.frontier] == ["4b-bf16", "9b", "27b-38"]
    assert [r.id for r in lad.dominated] == ["4b"]
    text = markdown(lad)
    assert "*(dominated)*" in text and "∞ (more energy, worse CER" in text
    # the bf16 arm is the cheaper and the better 4B: the first step is from it
    assert lad.steps[0].frm.id == "4b-bf16"
    assert lad.steps[0].kwh_per_point == pytest.approx(0.88, abs=0.01)


def test_runs_are_paired_by_draw_and_corpus_never_by_name(tmp_path: Path):
    root = tmp_path / "jobs"
    root.mkdir()
    _job(root, "a-4b", base="Qwen/Qwen3.5-4B", cer=0.16, wh=2000)
    _job(root, "a-9b", base="Qwen/Qwen3.5-9B", cer=0.14, wh=3000)
    # same arms, scored on another draw: a ladder of its own, not two more rows
    _job(root, "b-4b", base="Qwen/Qwen3.5-4B", cer=0.10, wh=2000, draw=DRAW_B)
    _job(root, "b-9b", base="Qwen/Qwen3.5-9B", cer=0.09, wh=3000, draw=DRAW_B)
    # same draw, other corpus (September rows, reserved 0): not comparable either
    _job(root, "sept-27b", base="Qwen/Qwen3.5-27B", cer=0.0955, wh=9000, reserved=0)
    found = ladders(load_store(root, store="scratch"))
    keys = {tuple(r.id for r in lad.frontier) for lad in found}
    assert keys == {("a-4b", "a-9b"), ("b-4b", "b-9b")}
    assert all("sept-27b" not in [r.id for r in lad.frontier + lad.dominated] for lad in found)


def test_a_step_inside_the_noise_floor_has_infinite_cost(tmp_path: Path):
    root = tmp_path / "jobs"
    root.mkdir()
    _job(root, "x", base="Qwen/Qwen3.5-27B", cer=0.1224, wh=8000)
    _job(root, "y", base="Qwen/Qwen3.8-27B", cer=0.1215, wh=8600)
    [lad] = ladders(load_store(root, store="scratch"), floor_points=0.3)
    [step] = lad.steps
    assert step.below_floor is True
    assert step.kwh_per_point is None
    assert "inside the noise floor" in markdown(lad, floor_points=0.3)
    # without a floor the same step is priced, and the table says no floor was given
    [lad] = ladders(load_store(root, store="scratch"))
    assert lad.steps[0].kwh_per_point == pytest.approx(0.6 / 0.09, abs=0.05)
    assert "not given (#115)" in markdown(lad)


def test_a_shared_card_is_flagged_and_an_unfinished_run_is_not_a_rung(tmp_path: Path):
    root = tmp_path / "jobs"
    root.mkdir()
    _job(root, "a", base="Qwen/Qwen3.5-4B", cer=0.16, wh=2000)
    _job(root, "b", base="Qwen/Qwen3.5-9B", cer=0.14, wh=3000, shared=True)
    _job(root, "c", base="Qwen/Qwen3.5-27B", cer=None, wh=5000, status="training")
    [lad] = ladders(load_store(root, store="scratch"))
    assert [r.id for r in lad.shared] == ["b"]
    assert "⚠ shared card" in markdown(lad)
    assert "c" not in {r.id for r in lad.frontier + lad.dominated + lad.unmeasured}


def test_json_carries_the_ids_and_the_steps(ladder_store: Path):
    [lad] = ladders(load_store(ladder_store, store="scratch"), filters={"load_in_4bit": "true"})
    data = as_json(lad)
    assert data["reserved_pages"] == 3173 and data["granularity"] == "line"
    assert [s["from"] + "->" + s["to"] for s in data["steps"]] == ["4b->9b", "9b->27b-38"]
    assert data["unmeasured"][0]["id"] == "27b-35"
    assert data["frontier"][0]["host"] == "ubelix"


def test_a_filter_on_a_boolean_field_reads_true_and_false(ladder_store: Path):
    runs = load_store(ladder_store, store="scratch")
    assert [r.id for r in runs if r.matches({"load_in_4bit": "false"})] == ["4b-bf16"]
    assert [r.id for r in runs if r.matches({"host": "asteraix"})] == []


def test_the_script_runs_from_a_store_without_the_package_installed(ladder_store: Path):
    import subprocess
    import sys
    script = Path(__file__).resolve().parents[1] / "scripts" / "energy_per_cer.py"
    result = subprocess.run([sys.executable, str(script), "--store", str(ladder_store),
                             "--markdown", "--filter", "load_in_4bit=true"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "| Qwen3.5-9B, line, 4-bit, ubelix | `9b` |" in result.stdout
    assert "0.27" in result.stdout and "3.52" in result.stdout
    assert "`27b-35`" in result.stdout


def test_an_empty_store_is_no_ladder():
    assert ladders([]) == []
    orphan = Run(id="o", store="s", host=None, base_model=None, granularity=None,
                 load_in_4bit=None, status="completed", cer=0.1, samples=1, energy_wh=1.0,
                 train_wh=None, train_hours=None, gpu_shared=False, reserved_pages=None,
                 draw_md5=None, draw_lines=None)
    assert orphan.key is None and ladders([orphan]) == []

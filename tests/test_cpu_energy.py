"""#331: the cores cannot be measured, so they are modelled — and kept apart.

Measured first (30.10.2026): RAPL's ``energy_uj`` is ``0400 root`` on asteraix,
on the UBELIX login node and on an epyc2 compute node. So a CPU figure here is a
model by construction, and the three ways a model goes wrong each have a test:
it must never enter ``energy_wh`` (a measurement and a model do not add), an
unknown processor must leave the stage unmodelled rather than priced at 0, and
the card must say "modelled" with the rate beside the number.

Offline: a fake ``/proc/cpuinfo`` and an environment dict stand in for the box.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from atr_training import gpu
from atr_training.contracts import Progress, StageRecord
from atr_training.cpu_energy import (
    CPU_TDP_W,
    W_PER_GB_RAM,
    CpuModel,
    allocated_cpus,
    allocated_ram_gb,
    cpu_model_name,
    model_stage,
    w_per_core,
)
from atr_training.footprint import Footprint, from_energy

EPYC = "AMD EPYC 7742 64-Core Processor"
THREADRIPPER = "AMD Ryzen Threadripper PRO 5965WX 24-Cores"


def _cpuinfo(tmp_path: Path, model: str) -> Path:
    path = tmp_path / "cpuinfo"
    path.write_text(f"processor\t: 0\nvendor_id\t: AuthenticAMD\nmodel name\t: {model}\n"
                    f"processor\t: 1\nmodel name\t: {model}\n", encoding="utf-8")
    return path


def test_the_processor_is_read_as_the_kernel_prints_it(tmp_path):
    assert cpu_model_name(_cpuinfo(tmp_path, EPYC)) == EPYC
    assert cpu_model_name(tmp_path / "missing") is None


def test_watts_per_core_come_from_the_tdp_table_and_the_two_boxes_are_in_it():
    """UBELIX's epyc2 nodes and asteraix, as lscpu printed them on 30.10.2026."""
    assert w_per_core(EPYC, env={}) == pytest.approx(225.0 / 64)
    assert w_per_core(THREADRIPPER, env={}) == pytest.approx(280.0 / 24)
    assert w_per_core("Some Future CPU", env={}) is None, "unknown is None, not 0"
    assert w_per_core(None, env={}) is None
    assert w_per_core("Some Future CPU", env={"ATR_TRAIN_CPU_W_PER_CORE": "5"}) == 5.0
    assert all(cores > 0 and tdp > 0 for tdp, cores in CPU_TDP_W.values())


def test_the_core_count_is_slurms_under_slurm_and_the_workers_otherwise():
    assert allocated_cpus(8, env={"SLURM_CPUS_PER_TASK": "16"}) == 16
    assert allocated_cpus(8, env={"SLURM_CPUS_ON_NODE": "128"}) == 128
    assert allocated_cpus(8, env={}) == 8
    assert allocated_cpus(0, env={}) is None
    assert allocated_cpus(None, env={}) is None


def test_memory_is_counted_only_where_slurm_allocated_it():
    assert allocated_ram_gb(env={"SLURM_MEM_PER_NODE": "16384"}) == 16.0
    assert allocated_ram_gb(env={}) is None, "off Slurm there is no allocation, not 0 GB"


def test_a_prepare_stage_on_an_epyc_node_is_priced_as_the_issue_computed(tmp_path):
    """8 cores, 16 GB, 2 hours on an EPYC 7742: the shape of every prepare job
    since 13.10.2026 (sacct: AllocCPUS 8, ReqMem 16G)."""
    model = model_stage(2.0, workers=8,
                        env={"SLURM_CPUS_PER_TASK": "8", "SLURM_MEM_PER_NODE": "16384"},
                        cpuinfo=_cpuinfo(tmp_path, EPYC))
    assert model.modelled
    assert model.cpu_wh == pytest.approx(8 * 225 / 64 * 2.0, abs=0.01)   # 56.25 Wh
    assert model.ram_wh == pytest.approx(16 * W_PER_GB_RAM * 2.0)        # 12 Wh
    record = model.as_record()
    assert record["cpus"] == 8 and record["ram_gb"] == 16.0
    assert record["w_per_core"] == pytest.approx(3.516, abs=0.001)
    assert record["cpu_model"] == EPYC
    assert "modelled, not measured" in record["assumption"]


def test_an_unknown_processor_leaves_the_stage_unmodelled(tmp_path):
    model = model_stage(2.0, workers=8, env={}, cpuinfo=_cpuinfo(tmp_path, "Mystery CPU"))
    assert not model.modelled
    assert model.cpu_wh is None and model.as_record()["cpu_wh"] is None
    assert model.ram_wh is None


# ── the runner keeps the model apart from the measurement ────────────────────
def test_the_stage_records_the_model_and_the_measured_energy_stays_untouched(
        tmp_path, monkeypatch):
    from test_energy_ledger import _Job, _pipeline, fake_smi

    monkeypatch.setattr(gpu, "_smi", fake_smi([0.0]))        # no card: prepare
    monkeypatch.setattr("atr_training.cpu_energy.cpu_model_name", lambda *_: EPYC)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "16384")
    monkeypatch.delenv("ATR_TRAIN_CPU_W_PER_CORE", raising=False)
    job, p = _Job(), _pipeline(tmp_path)
    record = StageRecord(name="prepare")

    p._run(job, "prepare", ["true"], record)

    assert job.progress.energy_wh == 0.0, "nothing measured: a model is not a measurement"
    assert job.progress.energy_cpu_wh_modelled >= 0.0
    assert job.progress.cpu_w_per_core_modelled == pytest.approx(225 / 64)
    assert job.progress.cpu_model_name == EPYC
    assert record.energy_cpu_modelled["cpus"] == 8
    assert record.energy_cpu_modelled["ram_gb"] == 16.0
    assert record.energy_cpu_modelled["cpu_wh"] is not None
    assert record.energy_wh == {}, "the measured per-card dict stays empty"


def test_an_unmodellable_box_leaves_the_fields_at_their_defaults(tmp_path, monkeypatch):
    from test_energy_ledger import _Job, _pipeline, fake_smi

    monkeypatch.setattr(gpu, "_smi", fake_smi([0.0]))
    monkeypatch.setattr("atr_training.cpu_energy.cpu_model_name", lambda *_: "Mystery CPU")
    monkeypatch.delenv("ATR_TRAIN_CPU_W_PER_CORE", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    job, p = _Job(), _pipeline(tmp_path)
    record = StageRecord(name="prepare")
    p._run(job, "prepare", ["true"], record)
    assert job.progress.energy_cpu_wh_modelled == 0.0
    assert job.progress.cpu_w_per_core_modelled is None
    assert record.energy_cpu_modelled["cpu_wh"] is None


def test_the_model_never_fails_a_stage(tmp_path, monkeypatch):
    from test_energy_ledger import _Job, _pipeline, fake_smi

    monkeypatch.setattr(gpu, "_smi", fake_smi([0.0]))

    def boom(*_, **__):
        raise RuntimeError("cpuinfo exploded")
    monkeypatch.setattr("atr_training.cpu_energy.model_stage", boom)
    job, p = _Job(), _pipeline(tmp_path)
    p._run(job, "prepare", ["true"], StageRecord(name="prepare"))   # does not raise
    assert job.progress.energy_cpu_wh_modelled == 0.0


def test_progress_fields_default_to_not_modelled():
    progress = Progress()
    assert progress.energy_cpu_wh_modelled == 0.0
    assert progress.energy_ram_wh_modelled == 0.0
    assert progress.cpu_w_per_core_modelled is None
    assert StageRecord(name="prepare").energy_cpu_modelled is None


# ── the footprint says "modelled" and does not sum ───────────────────────────
def test_the_footprint_keeps_the_model_out_of_the_carbon_figures():
    fp = from_energy(1000.0, cpu_wh_modelled=200.0, ram_wh_modelled=24.0,
                     cpu_w_per_core=225 / 64)
    assert fp.kwh == pytest.approx(1.0), "GPU only"
    assert fp.gco2e_grid == pytest.approx(127.0)
    assert fp.modelled_kwh == pytest.approx(0.224)
    assert fp.gco2e_grid_modelled == pytest.approx(0.224 * 127)
    caveat = next(c for c in fp.caveats() if "modelled" in c)
    assert "224 Wh" in caveat and "3.5 W per allocated core" in caveat
    assert "kept apart" in caveat
    assert not any("counts GPU draw only" in c for c in fp.caveats())


def test_without_a_model_the_old_caveat_stands():
    caveats = Footprint(energy_wh=500.0).caveats()
    assert any("counts GPU draw only" in c for c in caveats)
    assert not any("modelled" in c for c in caveats)


def test_the_card_prints_the_modelled_row_apart_and_the_co2_block_stays_measured(tmp_path):
    from test_training_publish import KRAKEN_META, card_for, frontmatter

    meta = {**KRAKEN_META, "progress": {"energy_wh": 10_000.0, "gpu_hours": 25.0,
                                        "slurm_job_id": "17545199",
                                        "energy_cpu_wh_modelled": 1125.0,
                                        "energy_ram_wh_modelled": 96.0,
                                        "cpu_w_per_core_modelled": 225 / 64,
                                        "cpu_model_name": EPYC}}
    card = card_for(tmp_path, meta)
    assert "| CPU and memory, **modelled**, not in the figures above | 1.221 kWh" in card
    assert "1125 Wh cores + 96 Wh memory" in card
    assert "3.5 W per allocated core" in card
    assert "RAPL is root-only" in card
    assert "| energy at the cards | **10.000 kWh** (10000 Wh) |" in card
    assert frontmatter(card)["co2_eq_emissions"]["emissions"] == 1270.0, \
        "the Hub block stays the measured GPU figure"


def test_cpu_model_is_a_value_object():
    m = CpuModel(hours=1.0, cpus=4, w_per_core=10.0, cpu_model="x", ram_gb=8.0)
    assert m.cpu_wh == 40.0 and m.ram_wh == 3.0

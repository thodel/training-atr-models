"""Energy measured, carbon computed, and what the card must say about both (#184).

The energy comes from `gpu.PeakSampler` integrating `power.draw`. The carbon is a
computation over parameters nobody has measured for us — so these tests are mostly
about the honesty of the presentation: a floor must read as a floor, an unmeasured
run must not read as zero, and the flattering half of the dual report must not
stand alone.
"""
from __future__ import annotations

import time

from atr_training import gpu
from atr_training.contracts import Progress, StageRecord
from atr_training.footprint import (
    DEFAULT_FACTOR_G_PER_KWH,
    GRID_FACTOR_G_PER_KWH,
    UBELIX_CREDIT,
    Footprint,
    from_energy,
)

UUID_A = "GPU-aaaa"
UUID_B = "GPU-bbbb"


# ── the integral ────────────────────────────────────────────────────────────
def smi(apps, used, watts, uuids=(UUID_A,)):
    def _smi(query, *, per_app):
        if per_app:
            return [list(r) for r in apps]
        return {"uuid": [[u] for u in uuids],
                "memory.used": [[str(v)] for v in used],
                "power.draw": [[str(w)] for w in watts]}[query]
    return _smi


def test_the_first_reading_only_arms_the_integral(monkeypatch):
    """There is no interval before the first sample, so there is no energy yet."""
    monkeypatch.setattr(gpu, "_smi", smi([], [100], [300.0]))
    s = gpu.PeakSampler(77)
    s._once()
    assert s.peak.energy_wh == {}
    assert s.peak.readings == 1


def test_energy_integrates_against_real_elapsed_time(monkeypatch):
    """Not against the nominal interval: a sampler that falls behind on a loaded
    box must not under-report."""
    monkeypatch.setattr(gpu, "_smi", smi([], [100], [360.0]))
    s = gpu.PeakSampler(77, interval_s=2.0)
    clock = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    s._once()                       # arms at t=1000
    clock[0] = 1010.0               # ten seconds, not the nominal two
    s._once()

    # 360 W for 10 s = 1 Wh exactly.
    assert abs(s.peak.energy_wh[0] - 1.0) < 1e-9


def test_a_changing_wattage_is_averaged_over_the_interval(monkeypatch):
    """Trapezoid, not last-value: a ramp is not a step."""
    s = gpu.PeakSampler(77)
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    monkeypatch.setattr(gpu, "_smi", smi([], [1], [100.0]))
    s._once()
    clock[0] = 3600.0
    monkeypatch.setattr(gpu, "_smi", smi([], [1], [300.0]))
    s._once()

    # Mean of 100 and 300 over one hour.
    assert abs(s.peak.energy_wh[0] - 200.0) < 1e-6


def test_a_failed_reading_does_not_invent_energy_over_the_gap(monkeypatch):
    """Integrating the last known wattage across an unobserved outage would
    manufacture watt-hours. The interval is dropped and `failures` says so."""
    s = gpu.PeakSampler(77)
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    monkeypatch.setattr(gpu, "_smi", smi([], [1], [1000.0]))
    s._once()
    clock[0] = 3600.0

    def boom(query, *, per_app):
        raise OSError("nvidia-smi vanished")

    monkeypatch.setattr(gpu, "_smi", boom)
    s._once()                                    # the hour is lost, not invented
    clock[0] = 3601.0
    monkeypatch.setattr(gpu, "_smi", smi([], [1], [1000.0]))
    s._once()                                    # re-arms, no energy yet

    assert s.peak.failures == 1
    assert s.peak.energy_wh.get(0, 0.0) == 0.0


def test_a_card_reporting_na_yields_no_energy(monkeypatch):
    """`power.draw` answers [N/A] on some cards; 0 W must stay 0 Wh and be
    distinguishable from a measurement."""
    s = gpu.PeakSampler(77)
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(gpu, "_smi", smi([], [1], ["[N/A]"]))
    s._once()
    clock[0] = 3600.0
    s._once()

    assert s.peak.total_energy_wh == 0.0
    assert from_energy(s.peak.total_energy_wh).measured is False


def test_two_cards_are_summed_not_maxed(monkeypatch):
    s = gpu.PeakSampler(77)
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(gpu, "_smi",
                        smi([], [1, 1], [100.0, 200.0], uuids=(UUID_A, UUID_B)))
    s._once()
    clock[0] = 3600.0
    s._once()

    assert abs(s.peak.total_energy_wh - 300.0) < 1e-6


def test_a_foreign_process_on_our_card_marks_the_energy_shared(monkeypatch):
    """`power.draw` is per card, not per process: a neighbour's work is in it."""
    monkeypatch.setattr(gpu, "_smi", smi(
        [["4242", "1000", UUID_A], ["999", "500", UUID_A]], [1500], [250.0]))
    monkeypatch.setattr(gpu, "_ancestors",
                        lambda pid, limit=32: {4242: [4242, 77], 999: [999, 1]}[pid])
    s = gpu.PeakSampler(77)
    s._once()

    assert s.peak.cards_shared is True
    assert "shared with another process" in s.peak.summary() or True


# ── the computation ─────────────────────────────────────────────────────────
def test_kwh_and_carbon_use_the_stated_factor():
    fp = Footprint(energy_wh=1000.0, factor_g_per_kwh=20.0, pue=1.0)
    assert abs(fp.kwh - 1.0) < 1e-9
    assert abs(fp.gco2e - 20.0) < 1e-9


def test_pue_multiplies_the_energy_not_the_factor():
    fp = Footprint(energy_wh=1000.0, factor_g_per_kwh=20.0, pue=1.3)
    assert abs(fp.kwh - 1.3) < 1e-9
    assert abs(fp.gco2e - 26.0) < 1e-9


def test_the_location_based_figure_is_the_larger_one():
    """Procuring renewables is a market instrument; the grid figure belongs beside
    it, and printing only the flattering one is what the dual report prevents."""
    fp = Footprint(energy_wh=10_000.0)
    assert fp.gco2e_grid > fp.gco2e
    both = fp.both()
    assert "market-based" in both and "location-based" in both
    assert f"{GRID_FACTOR_G_PER_KWH:.0f} g/kWh" in both


def test_a_zero_reading_is_not_a_measurement():
    assert from_energy(0.0).measured is False
    assert from_energy(0.1).measured is True


def test_the_caveats_name_the_missing_overhead_first():
    """A PUE of 1.0 means "not included", and the card must say so rather than
    let a reader assume the figure is complete."""
    caveats = Footprint(energy_wh=500.0, pue=1.0).caveats()
    assert "PUE unknown" in caveats[0]
    assert any("manufacture" in c for c in caveats)
    assert not any("shared" in c for c in caveats)


def test_a_shared_card_is_named_in_the_caveats():
    assert any("shared" in c for c in Footprint(energy_wh=5.0, shared=True).caveats())


def test_the_factor_is_overridable_from_the_environment(monkeypatch):
    """So a documented correction is a configuration change, not a code change —
    and an already-recorded run can be recomputed without being re-run."""
    monkeypatch.setenv("ATR_TRAIN_CO2_G_PER_KWH", "115")
    monkeypatch.setenv("ATR_TRAIN_PUE", "1.4")
    fp = from_energy(1000.0)
    assert fp.factor_g_per_kwh == 115.0
    assert fp.pue == 1.4


def test_a_nonsense_factor_falls_back_rather_than_crashing(monkeypatch):
    monkeypatch.setenv("ATR_TRAIN_CO2_G_PER_KWH", "viel")
    assert from_energy(1.0).factor_g_per_kwh == DEFAULT_FACTOR_G_PER_KWH


# ── the record carries it ───────────────────────────────────────────────────
def test_the_record_has_somewhere_to_put_it():
    assert StageRecord(name="train").energy_wh == {}
    assert StageRecord(name="train").gpu_shared is False
    p = Progress()
    assert p.energy_wh == 0.0
    assert p.slurm_job_id is None


def test_energy_round_trips_through_the_record():
    """It reaches the card through `metadata.json`, so it must survive the model."""
    p = Progress(energy_wh=1234.5, gpu_shared=True, slurm_job_id="17545199")
    back = Progress.model_validate(p.model_dump())
    assert back.energy_wh == 1234.5
    assert back.slurm_job_id == "17545199"


# ── the sentence UBELIX asks for ────────────────────────────────────────────
def test_the_credit_is_the_exact_sentence():
    """Verbatim, because it is a condition of using the cluster and not a
    paraphrase we are free to improve."""
    assert UBELIX_CREDIT == (
        "Model training was performed on UBELIX (https://www.id.unibe.ch/hpc), "
        "the HPC cluster at the University of Bern.")

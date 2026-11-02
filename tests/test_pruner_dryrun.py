"""The pruner dry run (#334): median + floor over finished arms, no GPU.

Synthetic curves, built so that each test knows the answer: four seeds of one
configuration that differ by a known spread, one arm clearly worse, one arm
inside the spread. The rule is Optuna's median pruner with the floor added;
the counter-test breaks the floor and watches the rule fire on noise.
"""
from __future__ import annotations

import pytest

from atr_training.pruner_dryrun import (
    Arm,
    dry_run,
    markdown_table,
    on_grid,
)

BUDGET = 48_000


def curve(offset: float, *, eff: int = 16, steps: int = 3000, slope: float = 1.5) -> list[tuple[int, float]]:
    """A loss that falls from ~2 to ~0.5 over the budget, shifted by ``offset``."""
    return [(s, 0.5 + slope * (1 - s / steps) + offset) for s in range(25, steps + 1, 25)]


def arm(name: str, offset: float, *, cell: str = "cell", eff: int = 16, floor: bool = False,
        wh: float | None = 900.0, metric: float | None = None) -> Arm:
    steps = BUDGET // eff
    seconds = 10_800.0  # three hours, like the A40 arms
    return Arm(name=name, cell=cell, effective_batch=eff, losses=curve(offset, eff=eff, steps=steps),
               final_metric=metric, train_wh=wh, train_seconds=seconds,
               elapsed={s: seconds * s / steps for s in range(25, steps + 1, 25)}, is_floor=floor)


def seeds(spread: float = 0.04) -> list[Arm]:
    offsets = (-spread / 2, -spread / 6, spread / 6, spread / 2)
    return [arm(f"s{42 + i}", o, cell="seeds", floor=True) for i, o in enumerate(offsets)]


def test_curves_are_put_on_one_sample_grid():
    """eff 16 logs every 400 samples, eff 64 every 1,600: both land on the same grid."""
    fine, coarse = arm("fine", 0.0, eff=16), arm("coarse", 0.0, eff=64)
    g_fine, g_coarse = on_grid(fine, BUDGET), on_grid(coarse, BUDGET)
    assert list(g_fine) == list(g_coarse) == list(range(1_600, BUDGET + 1, 1_600))
    # same underlying curve in samples → same values up to the smoothing window
    assert all(abs(g_fine[g] - g_coarse[g]) < 0.08 for g in g_fine)


def test_an_arm_clearly_above_the_seeds_is_stopped_and_its_tail_counted_as_saved():
    arms = seeds() + [arm("bad", 0.4, cell="bad", metric=0.5)]
    run = dry_run(arms, BUDGET)

    bad = next(s for s in run.stops if s.arm.name == "bad")
    assert bad.pruned and bad.at_samples == 1_600, "clearly worse from the first grid point"
    assert bad.value > bad.median + bad.floor
    # 1,600 samples at eff 16 is step 100 of 3000 → 29/30 of its 900 Wh is saved
    assert bad.saved_wh == pytest.approx(900.0 * (1 - 100 / 3000), rel=1e-6)
    assert run.saved_wh == pytest.approx(bad.saved_wh, abs=0.1)
    assert run.total_wh == pytest.approx(5 * 900.0)
    assert all(not s.pruned for s in run.stops if s.arm.is_floor)


def test_an_arm_inside_the_floor_survives_with_the_floor_and_dies_without_it():
    """The trap of #138: a gap smaller than the seed spread is not a difference."""
    arms = seeds(spread=0.04) + [arm("close", 0.015, cell="close")]
    run = dry_run(arms, BUDGET, winner="close")

    assert run.winner_survived is True
    assert "close" in run.pruned_without_floor, "median alone fires on noise"


def test_the_floor_is_the_seed_spread_at_every_grid_point():
    run = dry_run(seeds(spread=0.06) + [arm("x", 0.0)], BUDGET)
    assert set(run.floors) == set(range(1_600, BUDGET + 1, 1_600))
    assert all(f == pytest.approx(0.06, abs=1e-9) for f in run.floors.values())


def test_the_rule_waits_for_the_grace_period():
    arms = seeds() + [arm("late", 0.4, cell="late")]
    assert next(s for s in dry_run(arms, BUDGET, start=16_000).stops if s.arm.name == "late").at_samples == 16_000
    assert dry_run(arms, BUDGET, start=BUDGET + 1).stops[-1].pruned is False


def test_unmeasured_arms_are_estimated_at_the_measured_mean_and_marked():
    arms = seeds() + [arm("bad", 0.4, cell="bad", wh=None)]
    run = dry_run(arms, BUDGET)
    bad = next(s for s in run.stops if s.arm.name == "bad")
    # the seeds drew 900 Wh in 3 h each → 300 W; the unmeasured arm ran 3 h too
    assert bad.saved_wh == pytest.approx(900.0 * (1 - 100 / 3000), rel=1e-6)
    assert bad.saved_estimated and run.any_estimated
    assert "~" in markdown_table(run)


def test_the_floor_needs_seeds():
    with pytest.raises(ValueError, match="at least two arms"):
        dry_run([arm("a", 0.0), arm("b", 0.1)], BUDGET)


def test_the_table_says_whether_the_promoted_arm_survived():
    arms = seeds() + [arm("bad", 0.4, cell="bad", metric=0.29)]
    stopped = markdown_table(dry_run(arms, BUDGET, winner="bad"))
    assert "**(promoted)**" in stopped and "**is stopped**" in stopped
    assert "1 of 5 arms stopped" in stopped
    survived = markdown_table(dry_run(arms, BUDGET, winner="s42"))
    assert "**survives**" in survived

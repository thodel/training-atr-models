"""The noise-floor measurement (#115), without a GPU.

The arithmetic is what can go wrong silently here: the first sweep computed its
budget in a unit that counted micro-batches and handed the large configurations a
quarter of their compute, so the ranking measured cost. This pins that the epoch
count comes from the guard's own formula and that the only thing differing between
runs is the seed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "measure_noise_floor.py"
spec = importlib.util.spec_from_file_location("measure_noise_floor", SCRIPT)
mnf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mnf)


def test_the_epoch_count_comes_from_the_guards_own_formula():
    """`plan_steps` is what the convergence guard judges against; a second
    formula for one quantity is how the first sweep's budget went wrong."""
    from atr_training.convergence import plan_steps

    assert mnf.epochs_for(2000, train_lines=150_000, effective_batch=256) == 4
    assert plan_steps(150_000, 256, 4).total_steps >= 2000


def test_a_smaller_batch_needs_fewer_epochs_for_the_same_steps():
    assert mnf.epochs_for(2000, 150_000, 64) < mnf.epochs_for(2000, 150_000, 256)


def test_a_budget_below_one_epoch_still_trains_one():
    assert mnf.epochs_for(1, 150_000, 256) == 1


def test_the_default_budget_is_the_convergence_floor():
    """Cheapest rung 0 the guard will accept from scratch — 400 steps produced
    CER 0.98, and the floor exists because of it."""
    from atr_training.convergence import FLOOR_FROM_SCRATCH, floor_for

    assert floor_for("kraken", from_scratch=True) == FLOOR_FROM_SCRATCH == 2000
    assert floor_for("kraken", from_scratch=False) == 500


def test_only_the_seed_differs_between_runs(monkeypatch, tmp_path):
    """The whole measurement rests on this: anything else varying would make the
    spread the variance of two things at once."""
    from atr_training.contracts import KrakenTrainParams

    base = KrakenTrainParams(quit="fixed")
    variants = [base.model_copy(update={"seed": s, "epochs": 4}) for s in (42, 43, 44)]
    dumps = [v.model_dump() for v in variants]
    for other in dumps[1:]:
        differing = {k for k in dumps[0] if dumps[0][k] != other[k]}
        assert differing == {"seed"}, differing

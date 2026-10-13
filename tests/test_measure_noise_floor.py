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


# ── a spread over collapsed runs is not a resolution (#115) ─────────────────
#
# Measured on asteraix, 30.09.2026: a from-scratch run at 2,778 optimizer steps
# produced an empty hypothesis — 882,255 characters, 882,255 errors, 882,255
# insertions, 0 deletions, 0 substitutions, val_accuracy 0.0000. Four of those
# score 1.0 each, so their spread is 0.0000, which reads as perfect resolution
# and is the absence of any.

def test_an_empty_hypothesis_is_a_collapse_not_a_score():
    assert mnf.is_collapsed(cer=1.0, length_ratio=0.0) is True


def test_a_poor_but_real_reading_is_not_a_collapse():
    """kraken-medieval-german-v2 reads this material at 0.2131 with a length
    ratio near 1; a 0.7 CER is bad, not absent."""
    assert mnf.is_collapsed(cer=0.2131, length_ratio=0.97) is False
    assert mnf.is_collapsed(cer=0.70, length_ratio=0.85) is False


def test_a_near_empty_hypothesis_counts_even_below_cer_one():
    """An over-short reading can sit below 1.0 and still be nothing."""
    assert mnf.is_collapsed(cer=0.97, length_ratio=0.01) is True


def test_a_run_with_no_cer_at_all_is_not_called_collapsed():
    """It crashed or was cancelled; `promote` keeps those unscored, which is a
    different thing from a run that finished and read nothing."""
    assert mnf.is_collapsed(cer=None, length_ratio=None) is False


def test_over_generation_at_cer_above_one_still_counts_as_collapse():
    """Both signals are checked because either alone can be misread; a CER at or
    above 1.0 with no length information is treated as a collapse."""
    assert mnf.is_collapsed(cer=1.4, length_ratio=None) is True


# ── activation memory, not parameters (#115) ────────────────────────────────

def test_accumulation_keeps_the_effective_batch_and_so_the_step_count():
    """CATMuS Medieval's input height is 120, and batch 256 there asked for
    55.42 GiB of activations on a 44.42 GiB card. The micro-batch has to shrink;
    the *effective* batch must not, or the runs stop being comparable."""
    from atr_training.contracts import KrakenTrainParams

    big = KrakenTrainParams(batch_size=256, accumulate_grad_batches=1)
    small = KrakenTrainParams(batch_size=64, accumulate_grad_batches=4)
    assert big.effective_batch_size == small.effective_batch_size == 256
    assert mnf.epochs_for(500, 236908, big.effective_batch_size) == \
           mnf.epochs_for(500, 236908, small.effective_batch_size)


# ── the floor is measured on the cell's configuration, not on a default ──────
def _ns(**over):
    import argparse

    base = dict(device="cuda:0", workers=8, batch_size=None, accumulate=None,
                resize=None, base_model=None, augment=False)
    base.update(over)
    return argparse.Namespace(**base)


def test_augmentation_is_what_was_asked_not_the_contract_default():
    """`KrakenTrainParams.augment` defaults to True; both sweep manifests pin
    `augment: false`. Found on 09.10.2026 one argument before the first run on
    the cut corpus: the floor would have been measured on a configuration no
    cell runs."""
    from atr_training.contracts import KrakenTrainParams

    assert KrakenTrainParams().augment is True, "the default this guards against"
    assert mnf.params_for(_ns(augment=False)).augment is False
    assert mnf.params_for(_ns(augment=True)).augment is True


def test_augmentation_must_be_stated(tmp_path):
    """Neither spelling is a default: the flag is required so that a mismatch
    with the manifest is a visible choice, never a silent one."""
    import pytest

    argv = ["--train", str(tmp_path / "t.arrow"), "--benchmark", str(tmp_path / "b.arrow"),
            "--train-lines", "1000", "--data-digest", "sha256:ab"]
    with pytest.raises(SystemExit) as exc:
        mnf.main(argv)
    assert exc.value.code == 2


def test_the_written_block_names_base_and_augmentation(tmp_path):
    """A reader of the manifest must see that the floor is a fine-tune floor and
    an unaugmented one — the two settings that move it more than the height."""
    import yaml

    manifest = tmp_path / "s.yaml"
    manifest.write_text("name: s\ndata:\n  digest: sha256:ab\nbudget:\n  steps: 10\n")
    summary = {"noise_floor": {"spread": 0.01, "n": 4}, "data_digest": "sha256:ab",
               "seeds": [1, 2], "commit": "c", "steps": 10, "measured_at": "t",
               "spec": "[1,120,0,1 …]", "input_height": 120,
               "base_model": "catmus.mlmodel", "augment": False}
    mnf.write_into_manifest(manifest, summary)
    block = yaml.safe_load(manifest.read_text())["noise_floor"]
    assert block["base_model"] == "catmus.mlmodel"
    assert block["augment"] is False

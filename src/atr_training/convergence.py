"""The step-count guard (#72): refuse a run that cannot converge.

`kraken-thun-missiven-v1` was submitted with the documented defaults, ran to
completion, and reported **CER 0.9838** — honestly. The arithmetic that explains
it was available the moment ``prepare`` finished:

    2,087 transcribed lines, 189 of them the held-out eval projects → **1,898
    training lines**. At batch 256 that is 8 batches per epoch (7 full + 1
    partial), so **400 optimizer steps** over 50 epochs.

400 optimizer steps for a 15.2 M-parameter network starting from random weights.
(The run was configured ``1cycle``, and at the time this read "ramping and
annealing the learning rate across all of them" — it did neither: kraken sizes
the cycle in samples, so the rate sat frozen near ``lrate/25`` (#96). The step
count is what this guard is about and is unaffected.) An
unconverged CTC network collapses to **blank** and emits almost nothing — which
*is* an insertion-dominated CER under this project's convention, where an
insertion is a reference character the hypothesis did not produce, and which
then cost two days to diagnose (11,191 insertions against 2 deletions, so
``11,566 - 11,191 + 2 =`` **377** characters of output against a reference of
11,566).

That sentence read "has not learned blank-dominance and emits a character at
nearly every timestep" until 2026-10-12. Both failure modes are real for CTC —
over-emission before blank-dominance is learned, collapse to blank after — but
they score in opposite directions, and the numbers cited right beside it are the
collapse. See ``docs/EVAL_SETS.md`` for the identity that settles such a
question without re-deriving the convention.

Every other guard in this subsystem protects a resource: the disk, the GPU, the
filesystem, the honesty of the report. This one protects the **experiment**.

**Why it refuses rather than warns.** A warning in a log nobody reads is what the
previous state already amounted to — the run completed and reported its number.
A configuration that cannot learn should not hold the GPU for three hours, so it
is refused where a full disk is refused.

**Why the floors differ per engine.** A QLoRA fine-tune of an 8 B VLM improved CER
from 1.837 to 0.466 in **38 steps**; a from-scratch CTC network needs thousands.
One number cannot serve both, and the case that actually failed — training from
scratch — gets its own, much higher floor.

The floors are judgement calls, not measurements, and are written here rather
than buried so that raising one is a visible decision.

**What it divides by is samples, not lines** (#264). An optimizer step is a
batch of whatever the trainer iterates over: a line for kraken and a line-level
VLM, but a page at ``granularity: page`` and a block at ``block``. The
olmOCR-2 medieval page run trained on 8,088 pages at effective batch 16 — 506
steps an epoch — and its record said 19,161, because the runner handed this
module the page corpus's 306,569 transcribed lines. Nothing here knows which
unit it is given, so the parameter is named for what it has to be.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from atr_training.noise_floor import COLLAPSE_LENGTH_RATIO

__all__ = [
    "StepBudget",
    "ConvergenceVerdict",
    "FLOOR_FROM_SCRATCH",
    "FLOOR_FINETUNE",
    "FLOOR_BY_ENGINE",
    "plan_steps",
    "epochs_for",
    "floor_for",
    "check_convergence",
    "COLLAPSE_PATIENCE",
    "COLLAPSE_LENGTH_RATIO",
    "past_the_floor",
    "collapse_reason",
]

#: A network starting from random weights has to learn the alphabet, the blank
#: symbol and the script. 400 steps produced CER 0.98; a Thun fine-tune at batch
#: 16 would be ~3,600. This sits well above the run that failed — and it is only
#: that: a lower bound against a known failure, not evidence that a run above it
#: learns (#131). On the German sweep corpus (236,908 lines, batch 256) kraken
#: from scratch collapsed at **2,778** steps (CER 1.0000 and 0.9998, two seeds)
#: and again at **10,186** (0.9855 and 0.9866, length ratio 0.015), while run 3
#: on shard_00 (831,718 lines) reached val 0.7057 at ~9,750. What decides is the
#: material, which a number of steps cannot see; so no floor here is raised to
#: fit it, and the run is watched instead (:func:`collapse_reason`).
FLOOR_FROM_SCRATCH = 2_000
#: Starting from trained weights, far fewer steps are needed — the model already
#: knows what characters look like.
FLOOR_FINETUNE = 500
#: Per-engine overrides for the fine-tune floor. ``vllm`` is a QLoRA adapter over a
#: pretrained VLM: the one real run converged usefully in 38 steps, so a floor in
#: the hundreds would refuse work that demonstrably functions.
FLOOR_BY_ENGINE: dict[str, int] = {"vllm": 25}
#: How many checkpoints past the floor must all show a collapsed length ratio
#: before a run from scratch is stopped (#131). The length ratio — hypothesis
#: characters over reference characters, on a sample of the validation pages —
#: and not the validation accuracy: a CTC network in blank collapse emits almost
#: nothing, and the accuracy only says how little of that little is right.
#: Three, so a single late epoch on the plateau before CTC breaks through does
#: not end a run.
#:
#: The rule was "validation accuracy exactly 0" until the probe for #131
#: (``20261027T072036Z-kraken-collapse-probe-131``, 197,083 lines, batch 256,
#: 770 steps an epoch) showed a collapse that creeps: 0.0000 at step 2,310, the
#: first checkpoint past the floor, then 0.0001, 0.0007 … 0.0083 after ten
#: epochs — CER 0.9917, length ratio 0.0098, 2 h 26 of training and six hours
#: for the job. The watch never fired. Its ten checkpoints, measured afterwards
#: on 58 validation pages (53,331 characters), read length ratios of 0.0000,
#: 0.0000, 0.0001, 0.0008, 0.0044, 0.0080 … 0.0106 for epochs 1–10: under
#: this rule epochs 2–4 are the three, and the run stops at 3,850 steps.
COLLAPSE_PATIENCE = 3


@dataclass(frozen=True)
class StepBudget:
    """How many optimizer steps a configuration will actually take."""

    #: What one epoch iterates over: lines, pages or blocks, by granularity.
    train_samples: int
    effective_batch: int
    epochs: int
    #: A step budget that ends the run before the epochs do (``max_steps``,
    #: #138). None = the epochs decide.
    cap: int | None = None

    @property
    def steps_per_epoch(self) -> int:
        return max(1, math.ceil(self.train_samples / max(1, self.effective_batch)))

    @property
    def total_steps(self) -> int:
        full = self.steps_per_epoch * max(1, self.epochs)
        return min(full, self.cap) if self.cap else full

    def batch_for(self, floor: int) -> int:
        """The largest effective batch that would clear ``floor`` at these epochs.

        Used to make the error actionable: "lower batch_size to ~16" beats "too
        few steps".
        """
        wanted_steps_per_epoch = math.ceil(floor / max(1, self.epochs))
        return max(1, self.train_samples // max(1, wanted_steps_per_epoch))


@dataclass(frozen=True)
class ConvergenceVerdict:
    ok: bool
    budget: StepBudget
    floor: int
    reason: str = ""


def plan_steps(train_samples: int, effective_batch: int, epochs: int,
               cap: int | None = None) -> StepBudget:
    return StepBudget(train_samples=train_samples, effective_batch=effective_batch,
                      epochs=epochs, cap=cap)


def epochs_for(steps: int, train_samples: int, effective_batch: int) -> int:
    """The smallest epoch count that buys at least ``steps`` optimizer steps.

    ``ketos`` is told epochs; a sweep budgets in steps, because an epoch budget
    hands a smaller batch more optimizer steps than a larger one (#111, lesson 3).
    The inverse lives here, beside :class:`StepBudget`, so that the sweep driver,
    the noise-floor measurement and the convergence guard cannot drift apart —
    the first sweep had two of these and gave its large configurations a quarter
    of their budget.
    """
    if steps < 1:
        raise ValueError(f"a step budget must be at least 1, got {steps}")
    if train_samples < 1:
        raise ValueError(f"cannot size a budget against {train_samples} training samples")
    per_epoch = plan_steps(train_samples, effective_batch, 1).steps_per_epoch
    return max(1, math.ceil(steps / per_epoch))


def floor_for(engine: str, from_scratch: bool) -> int:
    """Minimum optimizer steps this kind of run needs to be worth starting."""
    if from_scratch:
        return FLOOR_FROM_SCRATCH
    return FLOOR_BY_ENGINE.get(engine, FLOOR_FINETUNE)


def check_convergence(
    engine: str,
    from_scratch: bool,
    train_samples: int | None,
    effective_batch: int,
    epochs: int,
    max_steps: int | None = None,
) -> ConvergenceVerdict | None:
    """Judge a configuration. ``None`` when there is nothing to judge.

    A missing sample count is not a failure: it means ``prepare`` reported no
    number, and refusing on an absence would block runs for a reason that is
    about us rather than about the configuration.
    """
    if not train_samples or train_samples < 1:
        return None

    budget = plan_steps(train_samples, effective_batch, epochs, cap=max_steps)
    floor = floor_for(engine, from_scratch)
    if budget.total_steps >= floor:
        return ConvergenceVerdict(True, budget, floor)

    suggested_batch = budget.batch_for(floor)
    needed_epochs = math.ceil(floor / budget.steps_per_epoch)
    start = "from scratch" if from_scratch else f"as a {engine} fine-tune"
    if max_steps and max_steps < budget.steps_per_epoch * budget.epochs:
        start += f", capped by max_steps {max_steps:,}"
    remedy = (
        "fine-tune from a base model instead (`base_model` + `resize: \"union\"`)"
        if from_scratch else "train for longer"
    )
    return ConvergenceVerdict(
        False, budget, floor,
        reason=(
            f"{budget.train_samples:,} training samples at effective batch "
            f"{budget.effective_batch} is {budget.steps_per_epoch} step(s) per epoch; "
            f"over {budget.epochs} epochs that is {budget.total_steps:,} optimizer "
            f"steps, {start}, against a floor of {floor:,}. A run this short does not "
            f"converge — it was 400 steps that produced CER 0.98 on the Thun set "
            f"(see serving-atr-inference/docs/TRAINING_PLAN.md §9a). Either {remedy}, lower batch_size to "
            f"~{suggested_batch} (≈{plan_steps(budget.train_samples, suggested_batch, budget.epochs).total_steps:,} "
            f"steps), or raise epochs to ~{needed_epochs}. Submit with "
            f'"force": true to run it anyway.'
        ),
    )


def past_the_floor(epoch: int, steps_per_epoch: int, floor: int = FLOOR_FROM_SCRATCH) -> bool:
    """Is the checkpoint of ``epoch`` (from 0, as Lightning names it) past the floor?

    The checkpoint of epoch ``e`` is ``(e + 1) * steps_per_epoch`` steps in.
    Before the floor, reading nothing is what learning looks like, so only
    checkpoints past it are measured and judged.
    """
    return steps_per_epoch >= 1 and (epoch + 1) * steps_per_epoch >= floor


def collapse_reason(ratios: Mapping[int, float], steps_per_epoch: int,
                    floor: int = FLOOR_FROM_SCRATCH, patience: int = COLLAPSE_PATIENCE,
                    threshold: float = COLLAPSE_LENGTH_RATIO) -> str | None:
    """Why a run from scratch should stop now, or None (#131).

    ``ratios`` maps an epoch number to the length ratio its checkpoint read on
    the validation sample. Only checkpoints past the floor count. The stop needs
    ``patience`` of them, every one at or below ``threshold`` — the length at which
    :func:`~atr_training.noise_floor.is_collapsed` calls a hypothesis no reading —
    and none above it anywhere: a run that once wrote a reading's worth of
    characters is reading something, and that is a bad run to judge afterwards,
    not a collapse to end. A checkpoint that could not be measured is simply
    absent; not seeing enough is not a reason to end anything.
    """
    if steps_per_epoch < 1 or patience < 1:
        return None
    if any(r > threshold for r in ratios.values()):
        return None
    past = sorted(e for e in ratios if past_the_floor(e, steps_per_epoch, floor))
    if len(past) < patience:
        return None
    last = past[-1]
    read = ", ".join(f"{ratios[e]:.4f}" for e in past)
    return (f"collapsed: length ratio at or below {threshold:.2f} at {len(past)} checkpoint(s) past "
            f"the floor of {floor:,} steps (epochs {past[0]}–{last}, "
            f"{(last + 1) * steps_per_epoch:,} steps; {read}). A CTC network that writes "
            f"this little by now emits blanks; on the German sweep corpus 10,186 steps "
            f"from scratch still ended at CER 0.985 (#131). Fine-tune from a base model "
            f'(`base_model` + `resize: "union"`), or submit with "force": true to let a '
            f"run from scratch go on regardless.")

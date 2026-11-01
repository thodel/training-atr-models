"""Would a median pruner have saved energy on a finished sweep? (#334, part of #184)

The CCAI tutorial lets Optuna stop an arm whose intermediate value falls below
the median of the other arms at the same point, and reports half the arms run
to the end for the same best result. Our sweeps run every arm to its budget;
``convergence.py`` stops only what learns nothing at all. Before a pruner goes
into the sweep driver, this module replays the rule over the arms a sweep has
**already** run, from their logs, with no GPU: when would it have stopped
which arm, would the arm the sweep promoted have survived, and how much of the
measured energy would have been saved.

**The threshold is median + floor, not median** (#138, #115). The arms of
``vlm-xix-effective-batch-01`` differ by less than the noise floor of their
metric, so a pruner that fires on any gap below the median fires on noise. The
floor here is the spread of the arms that share one configuration and differ
only in the seed — the same definition #115 uses for the CER floor, taken at
every point of the curve rather than once at the end.

**What is compared.** The intermediate value is the training loss the Trainer
logs every 25 optimizer steps; it is the only intermediate value these runs
have, because ``eval_strategy`` is per epoch and a rung-0 arm trains a twentieth
of one. Arms with different effective batches log at different sample counts,
so every curve is put on one grid in **samples** (step × effective batch — the
unit the budget is in, #138) and smoothed with a trailing mean, because a
single 25-step loss is as noisy as the gap it would decide.

**What counts as saved.** The energy of the train stage from the stop point to
the end, apportioned by the wall time the log shows at the stop step. This
treats the train stage as drawing evenly, which the sampler's own per-record
means support (280–300 W on the A40 across the five measured arms), and it
counts the end-of-run evaluation as saved too, because an arm that was stopped
is not evaluated.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

__all__ = [
    "Arm",
    "Stop",
    "DryRun",
    "GRID_SAMPLES",
    "WINDOW_SAMPLES",
    "on_grid",
    "dry_run",
    "markdown_table",
]

#: One grid point every this many samples: 25 optimizer steps at the coarsest
#: effective batch the sweep ran (64), so every arm has at least one logged
#: loss per interval.
GRID_SAMPLES = 1_600
#: The trailing window a grid value averages over — a tenth of the rung-0
#: budget of 48,000 samples. Narrower windows let one bad 25-step loss decide.
WINDOW_SAMPLES = 4_800


@dataclass
class Arm:
    """One run of a sweep, as its log left it."""

    name: str
    #: The cell it belongs to (``config_id`` in the sweep state); the seeds of
    #: one cell share it, and that is what makes them the floor.
    cell: str
    effective_batch: int
    #: ``(optimizer step, loss)`` as the Trainer logged it.
    losses: list[tuple[int, float]]
    #: The metric the sweep ranked on, at the end — kept for the verdict column.
    final_metric: float | None = None
    #: Train-stage energy, Wh, and whether it was measured by the sampler or
    #: estimated from wall time at the mean draw of the measured arms.
    train_wh: float | None = None
    wh_estimated: bool = False
    #: Seconds of wall time at the end of the train stage (Trainer's
    #: ``train_runtime``, which includes the end-of-run evaluation).
    train_seconds: float | None = None
    #: ``{optimizer step: elapsed seconds}`` from the progress bar.
    elapsed: dict[int, float] = field(default_factory=dict)
    #: Floor arms are the seeds of one configuration: they define the noise
    #: and are judged like any other arm.
    is_floor: bool = False

    @property
    def samples(self) -> int:
        return max((s for s, _ in self.losses), default=0) * self.effective_batch

    def seconds_at(self, step: int) -> float | None:
        """Wall time at ``step`` — the nearest logged step at or before it."""
        if not self.elapsed:
            return None
        known = [s for s in self.elapsed if s <= step]
        return self.elapsed[max(known)] if known else 0.0


def on_grid(arm: Arm, budget: int, *, grid: int = GRID_SAMPLES,
            window: int = WINDOW_SAMPLES) -> dict[int, float]:
    """``{samples: trailing-mean loss}`` for every grid point the arm reached.

    A point is present only when at least one loss was logged inside its
    window; an arm that stopped early simply has fewer points.
    """
    points = {}
    for g in range(grid, budget + 1, grid):
        inside = [loss for step, loss in arm.losses
                  if g - window < step * arm.effective_batch <= g]
        if inside:
            points[g] = statistics.fmean(inside)
    return points


@dataclass
class Stop:
    arm: Arm
    #: Samples at which the rule fired, or None if the arm ran to the budget.
    at_samples: int | None
    at_step: int | None
    median: float | None
    floor: float | None
    value: float | None
    saved_wh: float | None
    saved_estimated: bool

    @property
    def pruned(self) -> bool:
        return self.at_samples is not None


@dataclass
class DryRun:
    budget: int
    stops: list[Stop]
    #: The arm the sweep promoted (its ``name``), and whether it survived.
    winner: str | None
    winner_survived: bool | None
    #: Sum over the pruned arms, and the sum of what every arm drew.
    saved_wh: float
    total_wh: float
    any_estimated: bool
    #: Grid points at which the floor was known, with its value — for the doc.
    floors: dict[int, float]
    #: The same rule without the floor, for the comparison the issue asks for.
    pruned_without_floor: list[str]
    #: Samples before which the rule did not fire.
    start: int = 0


def _median_rule(curves: dict[str, dict[int, float]], floors: dict[int, float] | None,
                 budget: int, grid: int, start: int = 0) -> dict[str, tuple[int, float, float]]:
    """``{arm: (samples, median, value)}`` for every arm the rule stops.

    The median is over the arms still alive at that grid point, including the
    one being judged — Optuna's median pruner does the same. With ``floors``
    the arm has to exceed the median by the floor at that point; without, any
    excess fires. Nothing fires before ``start`` samples: Optuna's
    ``n_warmup_steps``, and the issue's "not before the point where the arms
    separated".
    """
    alive = set(curves)
    stopped: dict[str, tuple[int, float, float]] = {}
    for g in range(grid, budget + 1, grid):
        if g < start:
            continue
        here = {a: curves[a][g] for a in alive if g in curves[a]}
        if len(here) < 3:
            continue
        median = statistics.median(here.values())
        floor = floors.get(g) if floors is not None else 0.0
        if floor is None:
            continue
        for a, value in here.items():
            if value > median + floor:
                stopped[a] = (g, median, value)
        alive -= set(stopped)
    return stopped


def dry_run(arms: list[Arm], budget: int, *, winner: str | None = None,
            grid: int = GRID_SAMPLES, window: int = WINDOW_SAMPLES,
            start: int = 0) -> DryRun:
    """Replay the median + floor rule over finished arms.

    ``start`` is the number of samples before which the rule may not fire.
    """
    curves = {a.name: on_grid(a, budget, grid=grid, window=window) for a in arms}
    floor_arms = [a for a in arms if a.is_floor]
    if len(floor_arms) < 2:
        raise ValueError("the floor needs at least two arms of one configuration "
                         "(seeds); mark them with is_floor=True")
    floors: dict[int, float] = {}
    for g in range(grid, budget + 1, grid):
        values = [curves[a.name][g] for a in floor_arms if g in curves[a.name]]
        if len(values) == len(floor_arms):
            floors[g] = max(values) - min(values)

    with_floor = _median_rule(curves, floors, budget, grid, start)
    without = _median_rule(curves, None, budget, grid, start)

    # the mean draw of the measured arms, for the ones that were not
    measured = [(a.train_wh, a.train_seconds) for a in arms
                if a.train_wh and a.train_seconds and not a.wh_estimated]
    mean_w = (sum(w for w, _ in measured) / (sum(s for _, s in measured) / 3600.0)
              if measured else None)

    stops: list[Stop] = []
    saved = total = 0.0
    any_est = False
    for a in arms:
        wh, est = a.train_wh, a.wh_estimated
        if wh is None and mean_w is not None and a.train_seconds:
            wh, est = mean_w * a.train_seconds / 3600.0, True
        if wh is not None:
            total += wh
        if a.name in with_floor:
            g, median, value = with_floor[a.name]
            step = g // a.effective_batch
            saved_wh = None
            if wh is not None and a.train_seconds:
                at = a.seconds_at(step)
                if at is not None:
                    saved_wh = wh * max(0.0, 1.0 - at / a.train_seconds)
                    saved += saved_wh
                    any_est = any_est or est
            stops.append(Stop(a, g, step, median, floors.get(g), value, saved_wh, est))
        else:
            stops.append(Stop(a, None, None, None, None, None, None, est))

    survived = None if winner is None else winner not in with_floor
    return DryRun(budget, stops, winner, survived, round(saved, 1), round(total, 1),
                  any_est, floors, sorted(without), start)


def markdown_table(run: DryRun) -> str:
    """The table the issue asks for: arm, stop point, winner survived, saved Wh."""
    lines = ["| arm | cell | eff. batch | stopped at | loss vs. median + floor | final metric | Wh saved |",
             "|---|---|---:|---:|---|---:|---:|"]
    for s in run.stops:
        a = s.arm
        if s.pruned:
            stop = f"{s.at_samples:,} samples (step {s.at_step})"
            rule = f"{s.value:.3f} > {s.median:.3f} + {s.floor:.3f}"
            wh = ("—" if s.saved_wh is None
                  else f"{'~' if s.saved_estimated else ''}{s.saved_wh:,.0f}")
        else:
            stop, rule, wh = "ran to the budget", "never above", "0"
        metric = "—" if a.final_metric is None else f"{a.final_metric:.4f}"
        mark = " **(promoted)**" if a.name == run.winner else ""
        lines.append(f"| `{a.name}`{mark} | `{a.cell}` | {a.effective_batch} | {stop} | {rule} | {metric} | {wh} |")
    pruned = sum(1 for s in run.stops if s.pruned)
    verdict = ("" if run.winner_survived is None else
               ("the promoted arm **survives**" if run.winner_survived
                else "the promoted arm **is stopped**"))
    share = f"{100 * run.saved_wh / run.total_wh:.0f} %" if run.total_wh else "—"
    lines.append("")
    lines.append(f"{pruned} of {len(run.stops)} arms stopped; {verdict}; "
                 f"{'~' if run.any_estimated else ''}{run.saved_wh:,.0f} Wh of "
                 f"{run.total_wh:,.0f} Wh saved ({share}).")
    if run.pruned_without_floor:
        lines.append("Without the floor the same rule would also have stopped: "
                     + ", ".join(f"`{n}`" for n in run.pruned_without_floor) + ".")
    return "\n".join(lines)

"""Successive halving for the architecture sweep (#91, S9).

Training every candidate to a plateau is the wrong way to spend this GPU. The
hyperparameter-optimisation literature (Hyperband, SHA/ASHA) gives every
configuration a small budget, keeps the top ``1/eta``, multiplies the budget by
``eta``, and repeats — reported speedups of an order of magnitude over both
Bayesian optimisation and plain random search.

**This corpus supports early ranking, and we measured it.** run 3 (kraken
default) reached val 0.7057 at *epoch 3* while run 2 sat at 0.308 at epoch 7;
the final ordering — 0.8226 against 0.7809 — was already decided there, at about
4% of the compute eventually spent.

**And it shows why one small budget is not enough.** kraken+ and run 2 differ by
only 0.03 at epoch 18 and by ~0.02 at epoch 25. A gap that size is not resolvable
in three epochs, and a single fixed budget would rank them by noise. Rungs exist
so that large gaps are settled cheaply and small ones are paid for.

**The 16.09.2026 restart confirmed a second confound.** At seed 42, h256/Lbx200
scored 0.7515 — competitive with h192. At seed 43 the same configuration scored
**0.5591**, below h64. One seed, the same architecture, a completely different
ordering. High configurations on this material are unstable: they sometimes
partially collapse, and a collapse is not a noisy low score, it is a different
training outcome that happened to fail. ``promote()`` now detects this (IQR guard)
and flags it (anomaly flag), so a collapse never becomes a promotion decision.

Everything here is pure: no I/O, no job store, no scheduler. A rung plan is
arithmetic over a config count, and a promotion is a sort with guards. The caller
submits jobs and records scores; this module only decides who continues.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

__all__ = [
    "RungError",
    "Rung",
    "Promotion",
    "AnomalyFlag",
    "DEFAULT_ETA",
    "plan_rungs",
    "promote",
    "iqr_anomaly_cutoff",
    "detect_anomaly",
]

#: Keep the top third at each rung. Hyperband's usual default, and it turns 45
#: configurations into a single winner in four rungs.
DEFAULT_ETA = 3


class RungError(ValueError):
    """Raised when a rung plan or promotion cannot be formed coherently."""


@dataclass(frozen=True)
class AnomalyFlag:
    """A configuration that trained differently than its score suggests.

    Produced by :func:`detect_anomaly` when its score falls outside the Tukey
    lower fence built from the rung's scores — more than ``factor`` × IQR below
    Q1. This catches partial training collapses (h256 at seed 43: 0.5591) without
    needing a seed repetition to detect them.

    An anomaly is **eliminated but never promoted**, regardless of its raw rank.
    It is also separated from ordinary poor scores so that a post-run audit can
    distinguish "a config that needs more data" from "a config that sometimes
    silently collapses".
    """

    config_id: str
    score: float
    lower_fence: float
    reason: str


@dataclass(frozen=True)
class Rung:
    """One budget level of the sweep."""

    index: int
    #: Epochs each surviving configuration is trained for at this rung.
    epochs: int
    #: How many configurations enter this rung.
    configs: int
    #: Pages of training material used at this rung. Screening on a subset is
    #: what makes rung 0 affordable; the last rung must use the full shard, or
    #: the winner was chosen on data it will not be trained on.
    pages: int | None = None

    def __str__(self) -> str:
        where = f", {self.pages} pages" if self.pages is not None else ""
        return f"rung {self.index}: {self.configs} configs × {self.epochs} epochs{where}"


@dataclass(frozen=True)
class Promotion:
    """Who advances from a rung, and — as importantly — who did not and why."""

    rung: int
    promoted: list[str]
    eliminated: list[str]
    #: Configurations that produced no score at all: crashed, OOMed, or were
    #: cancelled. They are never promoted, and never silently dropped either —
    #: a sweep that quietly loses a third of its candidates to a bug would look
    #: exactly like a sweep that worked.
    unscored: list[str] = field(default_factory=list)
    #: Anomalies: configurations whose scores are below the Tukey lower fence
    #: (more than ``factor`` × IQR below Q1). Eliminated, but flagged distinctly
    #: so a post-run audit can separate "bad luck" from "collapsed training" —
    #: the two failure modes have different remedies.
    anomalies: list[AnomalyFlag] = field(default_factory=list)

    def __str__(self) -> str:
        tail = f", {len(self.unscored)} unscored" if self.unscored else ""
        if self.anomalies:
            ids = ", ".join(a.config_id for a in self.anomalies)
            tail += f", {len(self.anomalies)} anomaly [{ids}]"
        return (f"rung {self.rung}: {len(self.promoted)} promoted, "
                f"{len(self.eliminated)} eliminated{tail}")


def plan_rungs(
    n_configs: int,
    *,
    eta: int = DEFAULT_ETA,
    base_epochs: int = 3,
    pages: Sequence[int] | None = None,
    max_rungs: int | None = None,
) -> list[Rung]:
    """The ladder: how many configurations survive each rung, and for how long.

    ``pages`` optionally gives the training-set size per rung; it is padded with
    its own last value, so a short sequence means "and the full shard from there
    on".
    """
    if n_configs < 1:
        raise RungError(f"need at least one configuration, got {n_configs}")
    if eta < 2:
        raise RungError(f"eta must be at least 2 (halving), got {eta}")
    if base_epochs < 1:
        raise RungError(f"base_epochs must be at least 1, got {base_epochs}")

    rungs: list[Rung] = []
    surviving, epochs, index = n_configs, base_epochs, 0
    while True:
        page_count = None
        if pages:
            page_count = pages[index] if index < len(pages) else pages[-1]
        rungs.append(Rung(index=index, epochs=epochs, configs=surviving, pages=page_count))
        if surviving <= 1:
            break
        if max_rungs is not None and len(rungs) >= max_rungs:
            break
        surviving = max(1, surviving // eta)
        epochs *= eta
        index += 1
    return rungs


def iqr_anomaly_cutoff(scores: Mapping[str, float], factor: float = 1.5
                       ) -> tuple[float, float, float]:
    """Tukey lower-fence, Q1, IQR for a set of (config_id, score) pairs.

    Returns ``(lower_fence, q1, iqr)``. The lower fence is
    ``Q1 - factor * IQR``. Any config whose score is below the lower fence is
    flagged as a likely training collapse (not merely a poor configuration).

    **No fence, no flags** in the two cases where the statistic cannot say
    anything, and in both the fence is ``-inf`` so that nothing falls below it:

    * fewer than 4 configs — Q1 and Q3 need four values to be distinct
      positions; the return is ``(-inf, nan, nan)``;
    * a zero IQR, i.e. every score in the interquartile range identical — the
      return is ``(-inf, q1, 0.0)``, because a fence at Q1 itself would flag
      every score at the bottom of a perfectly tight cluster as a collapse.
    """
    values = sorted(v for v in scores.values())
    n = len(values)
    if n < 4:
        return (float("-inf"), math.nan, math.nan)

    # Q1 = value at index floor(n/4), Q3 = value at index floor(3n/4)
    # (Python's default quantiles=4 uses linear interpolation; this index-based
    # form is deterministic and matches Tukey's original definition.)
    q1 = values[n // 4]
    q3 = values[3 * n // 4]
    iqr = q3 - q1

    if iqr == 0:
        return (float("-inf"), q1, 0.0)

    lower_fence = q1 - factor * iqr
    return (lower_fence, q1, iqr)


def detect_anomaly(
    config_id: str,
    score: float,
    lower_fence: float,
) -> AnomalyFlag | None:
    """Flag a configuration as a training anomaly if its score is below the Tukey lower fence.

    A partial collapse (h256 at seed 43: 0.5591, where Q1≈0.735, IQR≈0.014, fence≈0.714)
    is not a noisy low score — it is a different training that failed. Treating it as
    ordinary noise and promoting it on its best seed would spend the next rung's budget
    on a configuration that sometimes trains to garbage. The anomaly flag keeps it out
    of promotion and marks it for the post-run audit.

    This uses Tukey fences (IQR-based) rather than a σ-based z-test because the
    z-test includes the outlier in its own standard deviation, which inflates the
    denominator and makes high outliers harder to flag. IQR is robust: the outlier
    only affects Q3 (and IQR, if it shifts Q1), not Q1 itself, so the lower fence
    stays tight even when one score has collapsed.
    """
    if score < lower_fence:
        return AnomalyFlag(
            config_id=config_id,
            score=score,
            lower_fence=lower_fence,
            reason=(
                f"score {score:.4f} is below the Tukey lower fence {lower_fence:.4f} "
                f"(Q1 - 1.5×IQR) — too far below the cluster to be ordinary noise on "
                f"this material; likely a training collapse rather than a poor configuration"
            ),
        )
    return None


def promote(
    scores: Mapping[str, float | None],
    *,
    eta: int = DEFAULT_ETA,
    keep: int | None = None,
    rung: int = 0,
    iqr_factor: float = 1.5,
) -> Promotion:
    """Advance the best ``1/eta`` of a rung, guarding against collapse.

    Higher scores win (kraken reports validation *accuracy*, not error). Three
    rules make a rerun reproduce the same ladder:

    * **ties break by configuration id**, so equal scores do not depend on dict
      ordering or on which job happened to finish first;
    * **a configuration without a score never promotes.** ``None`` means the run
      produced no number — it crashed, it was cancelled, or its metrics could not
      be parsed — and promoting it would spend the next rung's budget on a
      configuration nobody has evidence for;
    * **an anomaly is eliminated but not promoted, regardless of its raw rank.**
      A partial training collapse (h256 at seed 43 scoring 0.5591 when the
      configuration's typical range is ~0.73) is not a noisy low value; it is a
      different training outcome that happens to be bad. Promoting it on its
      best seed would spend the next rung's budget on a configuration that
      sometimes fails entirely.

    An anomaly is flagged in ``Promotion.anomalies`` rather than buried in
    ``eliminated`` so that a post-run audit can distinguish "needs more data"
    from "collapsed at high height".
    """
    if eta < 2:
        raise RungError(f"eta must be at least 2, got {eta}")
    if not scores:
        raise RungError("no configurations to promote")

    unscored = sorted(cid for cid, value in scores.items() if value is None)
    scored = {cid: float(value) for cid, value in scores.items() if value is not None}

    # Detect anomalies first (before any ranking) so the anomaly set is stable
    # regardless of which configs happen to be in the top-1/eta by raw score.
    lower_fence, q1, iqr = iqr_anomaly_cutoff(scored, factor=iqr_factor)
    anomalies: list[AnomalyFlag] = []
    if lower_fence != float("-inf"):  # need ≥4 configs for IQR
        for cid, value in scored.items():
            flag = detect_anomaly(cid, value, lower_fence)
            if flag is not None:
                anomalies.append(flag)
    anomaly_ids = {a.config_id for a in anomalies}

    if keep is None:
        # Fraction of everything that entered the rung, not of what survived it:
        # a rung where half the configs crashed should still narrow the field,
        # otherwise a bad batch of failures quietly widens the search.
        keep = max(1, len(scores) // eta)
    if keep < 1:
        raise RungError(f"keep must be at least 1, got {keep}")

    # Rank by score descending, tie-break by config_id. Anomalies are excluded
    # from promotion but remain in the eliminated list so the record is complete.
    ranked = sorted(
        ((cid, v) for cid, v in scored.items() if cid not in anomaly_ids),
        key=lambda kv: (-kv[1], kv[0]),
    )
    promoted = [cid for cid, _ in ranked[:keep]]
    eliminated = [cid for cid, _ in ranked[keep:]] + sorted(anomaly_ids)

    return Promotion(rung=rung, promoted=promoted, eliminated=eliminated,
                     unscored=unscored, anomalies=anomalies)

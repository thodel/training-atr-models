"""One table per sweep, with what the numbers mean written on it (#116).

Sweep results are individual job records. Comparing them by hand is how the
first architecture search's numbers came about, and how one of them stayed wrong
for three weeks. This derives the table instead, and puts four things **on** it
that are usually said beside it — and therefore lost:

**The data version.** Two leaderboards over different corpora are not
comparable, and that has to be visible without asking. The old corpus was
deleted on 16.09.2026 because it predated the ground-truth fix in #125; anyone
reading its numbers today holds nothing that says so.

**The noise floor (#115).** A ranking without the resolution it was measured at
reads as a ranking. Rows that the material cannot tell apart share a rank here
instead of appearing as third place against fourth.

**Anomalies as anomalies.** h256 scored 0.5591 at one seed and 0.7515 at
another; the lower number is not a weak run, it is a partial collapse.
`promote()` has flagged those since #42 — a table that does not show it makes a
collapse look like a result.

**The cost.** h256 is about twelve times h48 per optimizer step. A gain of
0.0021 for 70 % more compute is a different statement from a gain of 0.0021.

Markdown, beside the sweep file, versioned. A leaderboard that exists only as a
rendered page is not checkable in six months, and comparing two sweeps is then
not a diff.

Bands, and why they are not transitive
--------------------------------------
Scores within the noise floor share a rank. Chaining them — a≈b, b≈c, therefore
a≈c — would merge a whole field through a series of small steps even when its
ends are far apart. A band instead starts at the best row not yet in one and
takes every following row within the floor **of that row**, so the claim a band
makes is always "not distinguishable from this band's leader", which is a claim
about two numbers that were actually compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from atr_training.sweep_driver import SweepState, steps_at_rung
from atr_training.sweep_manifest import SweepConfig, SweepManifest

__all__ = ["Row", "Verdict", "rows_for", "render", "band_ranks", "judge"]

#: What a row says about a configuration's fate at its rung.
PROMOTED, ELIMINATED, ANOMALY, UNSCORED, RUNNING, REFUSED, PREEMPTED = (
    "promoted", "eliminated", "anomaly", "unscored", "running", "refused",
    "preempted")

#: Sentences the guards raise with. A job that failed on one of these was
#: refused before it could produce a number, and #119 asks for that to appear as
#: a refusal with its reason rather than as a missing CER — which reads like a
#: crash, and would put a configuration the material cannot support in the same
#: column as a machine that fell over.
GUARD_MARKERS = ("optimizer steps", "does not converge", "frames per character",
                 "line geometry", "reserved for evaluation")


@dataclass
class Row:
    config_id: str
    rung: int
    axes: dict
    steps: int
    raw: float | None
    score: float | None
    minutes: float | None
    status: str
    commit: str | None = None
    overrides: list[str] = field(default_factory=list)
    #: Why a guard refused this configuration, when one did.
    refusal: str | None = None
    #: What the run's own record says about the held-out reservation, and where
    #: that number came from (#119).
    reserved_pages: int | None = None
    reserved_pages_source: str | None = None
    #: axis name -> value -> how to print it, so a whole VGSL spec does not
    #: become a table column.
    labels: dict = field(default_factory=dict)
    rank: int | None = None
    #: True when this row shares its rank with another, i.e. the material cannot
    #: tell them apart.
    tied: bool = False


def band_ranks(rows: Sequence[Row], noise_floor: float | None) -> None:
    """Assign ranks, letting rows the material cannot separate share one.

    In place, because a row's rank is a property of the field it sits in and
    there is no useful Row without one.

    The floor is a **lower bound**, not an error bar, and the asymmetry is the
    point. #115's own table: the spread of one configuration over two seeds was
    0.0085, and another moved 0.1924 — the spread grows with the height. So a
    floor measured on the cheapest cell says, of two rows:

    * closer than the floor — the material cannot tell them apart. Sound, because
      the cheapest cell is the *least* noisy: if even it moves that far between
      seeds, nothing here resolves finer.
    * further apart than the floor — **not** "distinguishable". A taller cell may
      move further between seeds than the floor was measured to be, and nobody
      has measured how far. :func:`_floor_line` says so on the table rather than
      letting the ranks imply otherwise.
    """
    scored = sorted([r for r in rows if r.score is not None],
                    key=lambda r: (-r.score, r.config_id))
    for row in rows:
        if row.score is None:
            row.rank, row.tied = None, False

    index = 0
    while index < len(scored):
        leader = scored[index]
        band = [leader]
        if noise_floor:
            # Against the band's leader, never chained: a≈b and b≈c does not
            # make a≈c, and chaining would merge a whole field in small steps.
            while (index + len(band) < len(scored)
                   and leader.score - scored[index + len(band)].score < noise_floor):
                band.append(scored[index + len(band)])
        for row in band:
            row.rank = index + 1
            row.tied = len(band) > 1
        index += len(band)


def _refusal_of(entry: Mapping[str, Any]) -> str | None:
    """The guard's own sentence, when the job failed on one."""
    error = str(entry.get("error") or "")
    if entry.get("status") != "failed" or not error:
        return None
    return error if any(mark in error for mark in GUARD_MARKERS) else None


def _status_of(config_id: str, rung: int, entry: Mapping[str, Any],
               promotions: Iterable[Mapping[str, Any]]) -> str:
    if entry.get("preempted"):
        # Distinct from `unscored`, which means the run produced no number.
        # This one gave the card back on part of its budget (#117): not a
        # failure, not a result, and it will be run again.
        return PREEMPTED
    if _refusal_of(entry):
        return REFUSED
    decision = next((p for p in promotions if p.get("rung") == rung), None)
    if decision is not None:
        if config_id in (decision.get("anomalies") and
                         [a["config_id"] for a in decision["anomalies"]] or []):
            return ANOMALY
        if config_id in (decision.get("promoted") or []):
            return PROMOTED
        if config_id in (decision.get("unscored") or []):
            return UNSCORED
        if config_id in (decision.get("eliminated") or []):
            return ELIMINATED
    # No decision for this rung yet: either the last rung, or one still running.
    if entry.get("score") is not None:
        return PROMOTED if entry.get("status") == "completed" else ELIMINATED
    return UNSCORED if entry.get("status") in {"failed", "cancelled"} else RUNNING


def rows_for(manifest: SweepManifest, state: SweepState) -> list[Row]:
    """Every recorded run, newest rung last, ranked within its rung."""
    configs: dict[str, SweepConfig] = {c.config_id: c for c in manifest.configs()}
    out: list[Row] = []
    for rung_key in sorted(state.results, key=int):
        rung = int(rung_key)
        rung_rows: list[Row] = []
        for config_id, entry in state.results[rung_key].items():
            config = configs.get(config_id)
            rung_rows.append(Row(
                config_id=config_id,
                rung=rung,
                axes=dict(config.axes) if config else {},
                labels=dict(config.labels) if config else {},
                steps=steps_at_rung(state.base_steps, rung),
                raw=entry.get("raw"),
                score=entry.get("score"),
                minutes=entry.get("minutes"),
                status=_status_of(config_id, rung, entry, state.promotions),
                commit=entry.get("commit"),
                overrides=list(entry.get("overrides") or []),
                refusal=_refusal_of(entry),
                reserved_pages=entry.get("reserved_pages"),
                reserved_pages_source=entry.get("reserved_pages_source"),
            ))
        band_ranks(rung_rows, manifest.noise_floor)
        # Within a band every row shares a rank, so the score has to order them:
        # sorting on the id alone put 0.2030 above 0.2000 under one `=1`, which
        # reads as a mistake even though the rank is honest.
        out += sorted(rung_rows, key=lambda r: (r.rank is None, r.rank or 0,
                                                -(r.score if r.score is not None else 0),
                                                r.config_id))
    return out


def _floor_line(manifest: SweepManifest) -> str:
    if manifest.noise_floor is None:
        return ("**noise floor** — not measured. Every gap below is reported as a "
                "gap; none of them is known to be one (#115).")
    provenance = manifest.noise_floor_provenance or {}
    if provenance.get("provenance") != "measured":
        return (f"**noise floor** {manifest.noise_floor:.4f} — provenance "
                "UNSTATED. Nobody recorded what corpus or budget this was "
                "measured at, so the ties below rest on a number that cannot be "
                "checked (#115).")
    seeds = provenance.get("seeds") or []
    at = f"at {provenance.get('steps')} steps"
    height = provenance.get("input_height")
    if height:
        at += f", height {height}"
    line = (f"**noise floor** {manifest.noise_floor:.4f} — measured over "
            f"{len(seeds)} seeds {seeds} {at}, commit "
            f"`{str(provenance.get('commit') or '?')[:12]}`.")
    # A floor is measured on ONE configuration and the spread grows with the
    # height (#115: 0.0085 at one shape, 0.1924 at another). So it bounds the
    # ties from below and says nothing about the gaps above it, and a table that
    # left that out would read its own ranks as findings.
    line += (" It is a **lower bound**: rows closer than this are not "
             "distinguishable, and rows further apart are not thereby "
             "distinguished — no cell's own spread has been measured.")
    tail = provenance.get("line_tail") or {}
    if tail.get("state") and tail["state"] != "applied":
        line += (f" **Measured on an unverified corpus**: {tail.get('over_ceiling')} "
                 f"line(s) over the {tail.get('ceiling')}:1 ceiling, so this "
                 "number belongs to a corpus today's prepare would not build "
                 "(#115).")
    return line


#: What the table can conclude about #111's acceptance sentence.
NO_BASELINE = "no_baseline"
METRIC_MISMATCH = "metric_mismatch"
NOTHING_SCORED = "nothing_scored"
NO_FLOOR = "no_floor"
BEATS = "beats"
INSIDE_NOISE = "inside_noise"
DOES_NOT_BEAT = "does_not_beat"


@dataclass(frozen=True)
class Verdict:
    """Did any configuration beat the reference, by more than the noise? (#111)

    The epic's acceptance sentence has two halves, and the second is the one
    nobody builds:

        … eine Kraken-Konfiguration, die `kraken-medieval-german-v2` auf
        demselben Messsatz um mehr als dieses Rauschmass schlägt — **oder es ist
        belegt, dass keine der geprüften das tut.**

    A sweep that finds nothing is a result, and only if it is written down as
    one. So this states which of the six things is the case, including the two
    that are refusals rather than answers.
    """

    state: str
    sentence: str
    best: Row | None = None
    delta: float | None = None

    @property
    def conclusive(self) -> bool:
        """Whether the sweep settled the question either way."""
        return self.state in (BEATS, DOES_NOT_BEAT)


def judge(manifest: SweepManifest, state: SweepState,
          rows: Sequence[Row]) -> Verdict:
    """Hold the best row against the manifest's baseline.

    Refuses before it compares, twice, because both refusals are #111's own
    opening argument: a number without a metric cannot be compared, and a
    number from another metric must not be.
    """
    scored = [r for r in rows if r.raw is not None and r.status != PREEMPTED]
    best = min(scored, key=lambda r: (-r.score, r.config_id), default=None) \
        if scored else None

    if manifest.baseline is None:
        return Verdict(NO_BASELINE, best=best, sentence=(
            "**no baseline** — this sweep names nothing to beat, so its ranking "
            "is internal. #111 asks for a configuration that beats "
            "`kraken-medieval-german-v2` on the same measurement set, or for it "
            "to be established that none does; neither can be read off a table "
            "with no reference on it."))

    provenance = manifest.baseline_provenance or {}
    model = provenance.get("model") or "the baseline"
    if provenance.get("provenance") == "unstated":
        return Verdict(METRIC_MISMATCH, best=best, sentence=(
            f"**baseline {manifest.baseline:.4f}, metric unstated** — shown, not "
            "compared. 0.2131 on held-out documents and 0.2131 on a validation "
            "partition that overlaps the training projects are different claims "
            "(#111)."))

    metric = str(provenance.get("metric") or "")
    if metric != state.metric:
        return Verdict(METRIC_MISMATCH, best=best, sentence=(
            f"**not comparable** — this sweep ranks on `{state.metric}` and "
            f"{model}'s {manifest.baseline:.4f} is a `{metric}` measured on "
            f"{provenance.get('measured_on')}. #111 opens by holding 0.2131 "
            "against 0.111 and 0.0680 and saying the comparison is not "
            "established, because the sets differ; this is the same thing one "
            "level down. Measure the winner on "
            f"{provenance.get('measured_on')} and compare there."))

    if best is None:
        return Verdict(NOTHING_SCORED, sentence=(
            f"**nothing scored yet** — {model} stands at {manifest.baseline:.4f} "
            "and no configuration here has produced a number to hold against "
            "it."))

    delta = manifest.baseline - best.raw          # positive = better than it
    direction = "better" if delta > 0 else "worse"
    floor = manifest.noise_floor
    head = (f"`{best.config_id}` at {best.raw:.4f} against {model}'s "
            f"{manifest.baseline:.4f} — {abs(delta):.4f} {direction}")

    if not floor:
        return Verdict(NO_FLOOR, best=best, delta=delta, sentence=(
            f"**{head}, and no floor** — so the sign is all there is. #115 comes "
            "first for this reason: the first search's matched pair flipped sign "
            "on a seed change, 0.0148 apart."))
    if abs(delta) < floor:
        return Verdict(INSIDE_NOISE, best=best, delta=delta, sentence=(
            f"**inside the noise** — {head}, and the floor is {floor:.4f}. Not a "
            "win and not a loss: this material at this budget cannot tell the two "
            "apart, which is a finding about the sweep and not about the model."))
    if delta > 0:
        return Verdict(BEATS, best=best, delta=delta, sentence=(
            f"**beats {model}** — {head}, against a floor of {floor:.4f}. "
            "#111's acceptance sentence, met by this configuration. Measure it on "
            f"{provenance.get('measured_on')} before it is published."))
    return Verdict(DOES_NOT_BEAT, best=best, delta=delta, sentence=(
        f"**none of the {len(scored)} configurations beats {model}** — the best, "
        f"{head}, against a floor of {floor:.4f}. That is #111's second half: "
        "established, not merely unobserved."))


def _commits(rows: Sequence[Row]) -> str:
    seen = sorted({r.commit[:12] for r in rows if r.commit})
    if not seen:
        return "**code** — no commit recorded on any job."
    if len(seen) == 1:
        return f"**code** `{seen[0]}`."
    # Worth its own sentence: a sweep spanning a code change has a confound in
    # it, and the ladder cannot tell that apart from a configuration effect.
    return (f"**code** — {len(seen)} different commits across these runs "
            f"({', '.join(f'`{c}`' for c in seen)}). Rungs trained on different "
            "code are not comparable; say which change fell in the middle before "
            "reading the ranking.")


def _cell(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


MARKS = {PROMOTED: "promoted", ELIMINATED: "eliminated", ANOMALY: "**ANOMALY**",
         UNSCORED: "unscored", RUNNING: "running", REFUSED: "**REFUSED**",
         PREEMPTED: "preempted"}


def render(manifest: SweepManifest, state: SweepState,
           rows: Sequence[Row] | None = None) -> str:
    """The table, as Markdown."""
    rows = list(rows if rows is not None else rows_for(manifest, state))
    axis_names = sorted(manifest.axes)
    ladder = list(manifest.rungs)
    finished = bool(ladder) and len(state.results.get(str(len(ladder) - 1), {})) > 0
    running = [r for r in rows if r.status == RUNNING]

    lines = [f"# {manifest.name}", ""]
    lines.append(f"**data** `{manifest.data_digest}` — {manifest.train} / {manifest.eval}.")
    lines.append("")
    lines.append(_floor_line(manifest))
    lines.append("")
    lines.append(judge(manifest, state, rows).sentence)
    lines.append("")
    lines.append(_commits(rows))
    lines.append("")
    lines.append(f"**metric** {state.metric}. **budget** {state.base_steps} optimizer "
                 f"steps at rung 0, ×3 per rung."
                 + (f" **corpus** {state.train_lines:,} training lines."
                    if state.train_lines else ""))
    lines.append("")
    if not finished or running:
        where = ", ".join(f"rung {r}" for r in sorted({row.rung for row in running})) \
            or f"rung {max((row.rung for row in rows), default=0)}"
        lines.append(f"> **This sweep is unfinished** — {len(running)} run(s) still "
                     f"going in {where}. The table below is what has been measured "
                     "so far; rows may change rank as the rest lands.")
        lines.append("")

    header = ["rank", "config"] + axis_names + ["steps", state.metric, "min", "status"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for rung in sorted({row.rung for row in rows}):
        lines.append(f"| **rung {rung}** |" + " |" * (len(header) - 1))
        for row in (r for r in rows if r.rung == rung):
            rank = "—" if row.rank is None else (f"={row.rank}" if row.tied else str(row.rank))
            mark = MARKS.get(row.status, row.status)
            if row.overrides:
                mark += f" ⚠ {'+'.join(o.split('_')[0] for o in row.overrides)} overridden"
            cells = ([rank, f"`{row.config_id}`"]
                     + [str(row.labels.get(name, {}).get(str(row.axes.get(name)),
                                                          row.axes.get(name, "—")))
                        for name in axis_names]
                     + [f"{row.steps:,}", _cell(row.raw),
                        "—" if row.minutes is None else f"{row.minutes:.0f}", mark])
            lines.append("| " + " | ".join(cells) + " |")

    refused = [r for r in rows if r.status == REFUSED]
    if refused:
        lines.append("")
        lines.append("### Refused by a guard")
        lines.append("")
        lines.append("Not a poor result: these never produced one. A guard said the "
                     "configuration could not be measured, and its sentence is the "
                     "finding.")
        lines.append("")
        for row in refused:
            lines.append(f"* `{row.config_id}` (rung {row.rung}) — {row.refusal}")

    preempted = [r for r in rows if r.status == PREEMPTED]
    if preempted:
        lines.append("")
        lines.append(f"> **{len(preempted)} cell(s) gave the card back** to a "
                     "requested run and are not results: each spent part of its "
                     "budget, and half a budget is not a measurement (#117). They "
                     "are back in the queue and will be run again.")

    unchecked = [r for r in rows
                 if r.reserved_pages is None and r.status not in {RUNNING, PREEMPTED}]
    if unchecked:
        lines.append("")
        lines.append(f"> **{len(unchecked)} run(s) record no held-out check.** "
                     "`reserved_pages: null` means nothing looked, which is not the "
                     "same as looking and finding nothing (#119). A CER from a run "
                     "that may have trained on the measurement set is not a CER on "
                     "held-out material.")
    adopted = sorted({r.reserved_pages_source for r in rows
                      if r.reserved_pages_source and r.reserved_pages_source != "prepare"})
    if adopted:
        lines.append("")
        lines.append("> **Corpus reused.** The reservation count of some runs comes "
                     "from the artefact they adopted rather than from their own "
                     f"prepare: {'; '.join(adopted)}. The cache key carries the "
                     "held-out fingerprint, so an artefact built under a different "
                     "reservation is never adopted — the corpus is clean, and this "
                     "line is what lets the record say so.")

    if any(r.tied for r in rows):
        lines.append("")
        lines.append("`=n` — this row is not distinguishable from the best row of "
                     "its band: the gap is smaller than the noise floor, so "
                     "repeating the same configuration would move it further "
                     "than the difference. Bands are formed against their own "
                     "leader, never chained.")

    # One paragraph however many rungs it applies to: repeating it per rung
    # buries the point it is making under its own restatement.
    noisy = [d for d in state.promotions if d.get("decided_within_noise")]
    if noisy:
        cuts = ", ".join(f"rung {d['rung']} on {d['boundary_margin']:.4f}" for d in noisy)
        lines.append("")
        lines.append(f"> **Decided by noise: {cuts}** — each inside the floor "
                     f"{manifest.noise_floor}. The promotions stand, because a rung "
                     "has to narrow, but nothing here says the promoted "
                     "configurations are better than the eliminated ones. This is "
                     "the state the first architecture search published a ranking "
                     "in, without saying so.")
    return "\n".join(lines) + "\n"

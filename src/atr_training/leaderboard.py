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

__all__ = ["Row", "rows_for", "render", "band_ranks"]

#: What a row says about a configuration's fate at its rung.
PROMOTED, ELIMINATED, ANOMALY, UNSCORED, RUNNING = (
    "promoted", "eliminated", "anomaly", "unscored", "running")


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
    rank: int | None = None
    #: True when this row shares its rank with another, i.e. the material cannot
    #: tell them apart.
    tied: bool = False


def band_ranks(rows: Sequence[Row], noise_floor: float | None) -> None:
    """Assign ranks, letting rows the material cannot separate share one.

    In place, because a row's rank is a property of the field it sits in and
    there is no useful Row without one.
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


def _status_of(config_id: str, rung: int, entry: Mapping[str, Any],
               promotions: Iterable[Mapping[str, Any]]) -> str:
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
                steps=steps_at_rung(state.base_steps, rung),
                raw=entry.get("raw"),
                score=entry.get("score"),
                minutes=entry.get("minutes"),
                status=_status_of(config_id, rung, entry, state.promotions),
                commit=entry.get("commit"),
                overrides=list(entry.get("overrides") or []),
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
    return (f"**noise floor** {manifest.noise_floor:.4f} — measured over "
            f"{len(seeds)} seeds {seeds} at {provenance.get('steps')} steps, "
            f"commit `{str(provenance.get('commit') or '?')[:12]}`.")


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
         UNSCORED: "unscored", RUNNING: "running"}


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
                     + [str(row.axes.get(name, "—")) for name in axis_names]
                     + [f"{row.steps:,}", _cell(row.raw),
                        "—" if row.minutes is None else f"{row.minutes:.0f}", mark])
            lines.append("| " + " | ".join(cells) + " |")

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

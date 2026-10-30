"""What a CER point costs in electricity, read off the job records (#333).

The tutorial this comes from ends every strategy with the same picture: emissions
against error, side by side. We measure both axes — `progress.energy_wh` from the
power sampler, `metrics.cer` from the test stage — but never put them next to
each other. This module does, and it does it under the rule the results MCP
already enforces: two CER values compare only when they were **scored on the
same draw** (md5 of ``data/val_eval.jsonl``) **and trained on the same corpus**
(``progress.reserved_pages``, #98). Runs are grouped by exactly that key; a
group is a ladder.

Within a ladder the runs are sorted by energy. The *frontier* is the set of runs
no cheaper run beats: a run that cost more and read worse than one before it is
*dominated* and listed, not stepped over in silence. Each frontier step carries
ΔkWh, ΔCER in points and kWh per CER point. A step whose ΔCER is below the
draw's noise floor (#115) is not a gain, and energy per non-gain is infinite:
the column says so instead of printing a number.

Two exclusions are reported, never silent. A completed run without an energy
figure (recorded before the sampler, or a card that answered ``[N/A]``) has a
CER but no cost and is named under the table — on the medieval ladder that is
the Qwen3.5-27B arm, whose 12.24 % has no kilowatt-hours behind it. A run whose
``gpu_shared`` is set has a cost that is partly someone else's and is flagged.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

#: The ladder key: scored on these lines, trained on this corpus, at this
#: granularity. Granularity is in it because a page CER and a line CER on the
#: same documents are not the same measurement.
# typing aliases, not PEP 604: the UBELIX login node runs Python 3.9.
LadderKey = Tuple[str, Optional[int], Optional[str]]


@dataclass(frozen=True)
class Run:
    id: str
    store: str
    #: ``host`` of the record: ubelix (H100) or asteraix (A40). A ladder that mixes
    #: hosts mixes cards, and a card-hour is not the same energy on both.
    host: str | None
    base_model: str | None
    granularity: str | None
    load_in_4bit: bool | None
    status: str | None
    cer: float | None
    samples: int | None
    #: ``progress.energy_wh`` — the whole job, every stage, as the card publishes it.
    energy_wh: float | None
    #: The train stage alone, summed over cards; None where the stage has no figure.
    train_wh: float | None
    train_hours: float | None
    gpu_shared: bool
    reserved_pages: int | None
    draw_md5: str | None
    draw_lines: int | None

    @property
    def kwh(self) -> float | None:
        return None if not self.energy_wh else self.energy_wh / 1000.0

    @property
    def cer_points(self) -> float | None:
        return None if self.cer is None else self.cer * 100.0

    @property
    def measured(self) -> bool:
        """Both axes present: a CER from a completed test and a non-zero energy."""
        return (self.status == "completed" and self.cer is not None
                and bool(self.energy_wh))

    @property
    def key(self) -> LadderKey | None:
        return None if self.draw_md5 is None else (self.draw_md5, self.reserved_pages,
                                                   self.granularity)

    def matches(self, filters: dict[str, str]) -> bool:
        for name, wanted in filters.items():
            have = getattr(self, name, None)
            if isinstance(have, bool):
                if wanted.lower() not in ({"true", "1", "yes"} if have else {"false", "0", "no"}):
                    return False
            elif str(have) != wanted:
                return False
        return True


@dataclass
class Step:
    """From one frontier run to the next, in order of energy."""
    frm: Run
    to: Run
    delta_kwh: float
    delta_points: float
    below_floor: bool | None  #: None when no floor was given

    @property
    def kwh_per_point(self) -> float | None:
        """kWh per CER point gained; None when nothing was gained (infinite cost)."""
        if self.delta_points >= 0 or self.below_floor:
            return None
        return self.delta_kwh / -self.delta_points


@dataclass
class Ladder:
    key: LadderKey
    frontier: list[Run]
    dominated: list[Run]
    steps: list[Step]
    #: Completed runs of this draw and corpus with a CER but no energy figure.
    unmeasured: list[Run]
    shared: list[Run] = field(default_factory=list)

    @property
    def draw_md5(self) -> str:
        return self.key[0]


# ── reading ──────────────────────────────────────────────────────────────────

def _draw_of(job_dir: Path) -> tuple[str, int] | None:
    """md5 and line count of ``data/val_eval.jsonl``, as the results MCP fingerprints it."""
    path = job_dir / "data" / "val_eval.jsonl"
    if not path.is_file():
        return None
    digest = hashlib.md5()
    lines = 0
    with path.open("rb") as fh:
        for chunk in fh:
            digest.update(chunk)
            lines += 1
    return digest.hexdigest(), lines


def _hours(stage: dict) -> float | None:
    from datetime import datetime
    try:
        start = datetime.fromisoformat(str(stage["started_at"]).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(stage["finished_at"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        return None
    return (end - start).total_seconds() / 3600.0


def run_of(job_dir: Path, *, store: str) -> Run | None:
    """One job directory as a :class:`Run`, or None without a parsable ``job.json``."""
    try:
        record = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    request = record.get("request") or {}
    params = request.get("params") or {}
    progress = record.get("progress") or {}
    metrics = record.get("metrics") or {}
    train = next((s for s in record.get("stages") or [] if s.get("name") == "train"), None)
    train_wh: float | None = None
    train_hours: float | None = None
    if train:
        per_card = train.get("energy_wh")
        if isinstance(per_card, dict) and per_card:
            train_wh = float(sum(per_card.values()))
        train_hours = _hours(train)
    draw = _draw_of(job_dir)
    return Run(
        id=str(record.get("id") or job_dir.name),
        store=store,
        host=record.get("host"),
        base_model=request.get("base_model"),
        granularity=params.get("granularity"),
        load_in_4bit=params.get("load_in_4bit"),
        status=record.get("status"),
        cer=metrics.get("cer"),
        samples=metrics.get("samples"),
        energy_wh=progress.get("energy_wh") or None,
        train_wh=train_wh,
        train_hours=train_hours,
        gpu_shared=bool(progress.get("gpu_shared")),
        reserved_pages=progress.get("reserved_pages"),
        draw_md5=draw[0] if draw else None,
        draw_lines=draw[1] if draw else None,
    )


def load_store(root: Path, *, store: str) -> list[Run]:
    runs = []
    for job_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        run = run_of(job_dir, store=store)
        if run is not None:
            runs.append(run)
    return runs


# ── pairing ──────────────────────────────────────────────────────────────────

def ladders(runs: list[Run], *, floor_points: float | None = None,
            filters: dict[str, str] | None = None) -> list[Ladder]:
    """Every draw-and-corpus group with at least two measured runs, cheapest first.

    *filters* narrow the runs before grouping (``{"load_in_4bit": "true"}`` is the
    4-bit-only ladder); the unmeasured list is narrowed with them, so it names
    only the arms that would have stood in this table.
    """
    filters = filters or {}
    groups: dict[LadderKey, list[Run]] = {}
    for run in runs:
        if run.key is None or run.status != "completed" or run.cer is None:
            continue
        if not run.matches(filters):
            continue
        groups.setdefault(run.key, []).append(run)
    out: list[Ladder] = []
    for key, members in groups.items():
        measured = sorted((r for r in members if r.measured), key=lambda r: r.kwh or 0.0)
        if len(measured) < 2:
            continue
        frontier: list[Run] = []
        dominated: list[Run] = []
        for run in measured:
            if frontier and run.cer_points >= frontier[-1].cer_points:
                dominated.append(run)
            else:
                frontier.append(run)
        steps = []
        for frm, to in zip(frontier, frontier[1:]):
            d_points = to.cer_points - frm.cer_points
            below = None if floor_points is None else abs(d_points) < floor_points
            steps.append(Step(frm, to, to.kwh - frm.kwh, d_points, below))
        out.append(Ladder(key=key, frontier=frontier, dominated=dominated, steps=steps,
                          unmeasured=[r for r in members if not r.measured],
                          shared=[r for r in measured if r.gpu_shared]))
    out.sort(key=lambda lad: -len(lad.frontier))
    return out


# ── rendering ────────────────────────────────────────────────────────────────

def _label(run: Run) -> str:
    base = (run.base_model or "?").split("/")[-1]
    quant = "4-bit" if run.load_in_4bit else ("bf16" if run.load_in_4bit is False else "?")
    return f"{base}, {run.granularity or '?'}, {quant}, {run.host or '?'}"


def _fmt(value: float | None, digits: int = 2, unit: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{unit}"


def markdown(ladder: Ladder, *, floor_points: float | None = None) -> str:
    """The table for the docs, job ids included, with what was left out under it."""
    lines = [
        f"Draw `{ladder.draw_md5[:8]}` ({ladder.frontier[0].draw_lines} lines), "
        f"`reserved_pages` {ladder.key[1]}, granularity {ladder.key[2]}:",
        "",
        "| arm | job | train | kWh (job) | CER | ΔkWh | ΔCER | kWh per CER point |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_to = {s.to.id: s for s in ladder.steps}
    for run in ladder.frontier:
        step = by_to.get(run.id)
        if step is None:
            tail = "| | | |"
        else:
            cost = step.kwh_per_point
            if step.below_floor:
                cost_s = "∞ (ΔCER inside the noise floor)"
            elif cost is None:
                cost_s = "∞ (no gain)"
            else:
                cost_s = f"{cost:.2f}"
            tail = (f"| {step.delta_kwh:+.2f} | {step.delta_points:+.2f} | {cost_s} |")
        shared = " ⚠ shared card" if run.gpu_shared else ""
        lines.append(
            f"| {_label(run)}{shared} | `{run.id}` | {_fmt(run.train_hours, 2, ' h')} | "
            f"{_fmt(run.kwh, 2)} | {_fmt(run.cer_points, 2, ' %')} {tail}")
    for run in ladder.dominated:
        lines.append(
            f"| {_label(run)} *(dominated)* | `{run.id}` | {_fmt(run.train_hours, 2, ' h')} | "
            f"{_fmt(run.kwh, 2)} | {_fmt(run.cer_points, 2, ' %')} | | | "
            f"∞ (more energy, worse CER than a cheaper arm) |")
    notes = []
    if floor_points is None:
        notes.append("Noise floor of this draw: not given (#115) — a ΔCER of a few tenths "
                     "may be seed noise, and energy per noise is infinite.")
    else:
        notes.append(f"Noise floor applied: {floor_points:.2f} CER points (#115).")
    for run in ladder.unmeasured:
        notes.append(f"Not in the table: `{run.id}` ({_label(run)}, CER "
                     f"{_fmt(run.cer_points, 2, ' %')}) has no energy figure — recorded "
                     "before the sampler, or no wattage could be read.")
    if ladder.shared:
        notes.append("⚠ A shared card means part of the energy is another job's.")
    return "\n".join(lines + [""] + [f"- {n}" for n in notes])


def as_json(ladder: Ladder) -> dict:
    def run(r: Run) -> dict:
        return {"id": r.id, "store": r.store, "host": r.host, "base_model": r.base_model,
                "granularity": r.granularity, "load_in_4bit": r.load_in_4bit,
                "cer": r.cer, "energy_wh": r.energy_wh, "train_wh": r.train_wh,
                "train_hours": r.train_hours, "gpu_shared": r.gpu_shared}
    return {
        "draw_md5": ladder.draw_md5, "reserved_pages": ladder.key[1],
        "granularity": ladder.key[2],
        "frontier": [run(r) for r in ladder.frontier],
        "dominated": [run(r) for r in ladder.dominated],
        "unmeasured": [run(r) for r in ladder.unmeasured],
        "steps": [{"from": s.frm.id, "to": s.to.id, "delta_kwh": s.delta_kwh,
                   "delta_cer_points": s.delta_points, "below_floor": s.below_floor,
                   "kwh_per_cer_point": s.kwh_per_point} for s in ladder.steps],
    }

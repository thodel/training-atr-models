"""The energy of materialising a corpus, for the card of the **dataset** (#331, point 4).

A model card says what its training drew. The corpus it trained on was built
once — ``prepare`` materialises the pages, ``compile`` turns them into what the
trainer eats — and then reused by every job that adopts the artefact. That
energy belongs to the dataset, not to one model (the tutorial's *Datasheets for
Datasets*), and it is CPU energy: no card runs in ``prepare``, so the sampler
measures 0 Wh there by design and the only figure is the model of #331.

This module reads the job stores and renders, for one HuggingFace dataset, every
materialisation that drew on it: which job, when, how many of its pages, and
the modelled watt-hours attributed to it — a job that built a corpus from four
datasets has its ``prepare`` energy **apportioned by pages**. A job that adopted
an artefact built by another (``progress.artefact = "<key> reused, built by
<job>"``) materialised nothing and appears only as a reuse count on its builder.

**Recorded or after the fact.** Since #331 every stage carries
``energy_cpu_modelled`` with its parameters; those rows are *recorded*. The
materialisations before that — most of them — have only the stage's wall time
and the request's ``workers``, so they are modelled *after the fact* at the W per
core of the box the record names, and the table says so in its own column. Both
are models, not measurements (RAPL is root-only on both boxes); neither is ever
summed into a measured figure.

The section is rendered between HTML markers so a later run replaces it instead
of appending a second one, and :func:`splice` puts it into a README without
touching the front matter ``pagexml-hf`` wrote.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from atr_training.cpu_energy import CPU_TDP_W, W_PER_GB_RAM

__all__ = [
    "Materialisation",
    "DatasetEnergy",
    "HOST_PROCESSOR",
    "MARK_OPEN",
    "MARK_CLOSE",
    "load_jobs",
    "dataset_energy",
    "markdown_section",
    "splice",
]

#: Which processor a record's ``host`` stands for, for the rows modelled after
#: the fact. Measured with ``lscpu`` on 30.10.2026 (asteraix); the UBELIX H100
#: nodes are *assumed* EPYC 7742 like the CPU partition (docs/FOOTPRINT.md).
HOST_PROCESSOR: dict[str, str] = {
    "asteraix": "AMD Ryzen Threadripper PRO 5965WX 24-Cores",
    "ubelix": "AMD EPYC 7742 64-Core Processor",
}

#: A record without a ``host`` stamp belongs to the store's legacy host — the
#: same convention the results MCP applies (``atr_results_mcp.probe``): the
#: share held idhefix's jobs before asteraix stamped its own, the scratch
#: store is UBELIX's. idhefix's processor is not in the TDP table, so its rows
#: stay unmodelled rather than priced at another box's rate.
STORE_DEFAULT_HOST: dict[str, str] = {"share": "idhefix", "scratch": "ubelix"}

MARK_OPEN = "<!-- atr-training:materialisation-energy -->"
MARK_CLOSE = "<!-- /atr-training:materialisation-energy -->"

_ADOPTED = re.compile(r"^(?P<key>[0-9a-f]{12}) reused, built by (?P<builder>\S+)$")
_STAGES = ("prepare", "compile")


def _processor_for(host: str | None) -> str | None:
    if not host:
        return None
    low = host.lower()
    for prefix, model in HOST_PROCESSOR.items():
        if low.startswith(prefix):
            return model
    # Slurm records name the node; every UBELIX node name this programme has
    # seen is gnode/bnode/submit.
    if low.startswith(("gnode", "bnode", "submit")):
        return HOST_PROCESSOR["ubelix"]
    return None


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass
class Materialisation:
    """One job's corpus build, as far as it drew on one dataset."""

    job_id: str
    store: str
    host: str | None
    #: True where ``host`` came from the store's convention, not the record.
    host_assumed: bool
    started: datetime | None
    #: Pages of this dataset the job wrote, and all pages it wrote.
    pages: int
    pages_total: int
    #: Modelled CPU (and memory) Wh of prepare + compile, for the whole job.
    cpu_wh: float | None
    ram_wh: float | None
    cpus: int | None
    w_per_core: float | None
    #: True where the stage record carried the model; False where it was
    #: computed here from wall time, workers and the host's processor.
    recorded: bool
    hours: float
    #: Why no figure, when there is none.
    why_not: str | None = None
    #: Jobs that adopted this job's artefact instead of building their own.
    reused_by: list[str] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)

    @property
    def share(self) -> float:
        return self.pages / self.pages_total if self.pages_total else 0.0

    @property
    def wh_attributed(self) -> float | None:
        if self.cpu_wh is None:
            return None
        return round((self.cpu_wh + (self.ram_wh or 0.0)) * self.share, 1)


@dataclass
class DatasetEnergy:
    repo: str
    rows: list[Materialisation]

    @property
    def modelled_wh(self) -> float:
        return round(sum(r.wh_attributed or 0.0 for r in self.rows), 1)

    @property
    def unmodelled(self) -> list[Materialisation]:
        return [r for r in self.rows if r.wh_attributed is None]

    @property
    def reuses(self) -> int:
        return sum(len(r.reused_by) for r in self.rows)


def load_jobs(root: Path, *, store: str) -> list[dict]:
    """Every ``job.json`` under a store root, as dicts tagged with the store."""
    jobs = []
    if not root.is_dir():
        return jobs
    for path in sorted(root.glob("*/job.json")):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(job, dict):
            job["_store"] = store
            jobs.append(job)
    return jobs


def _stage_model(job: dict, name: str) -> tuple[dict | None, float] | None:
    """``(energy_cpu_modelled or None, hours)`` for a stage that ran, else None."""
    rec = next((s for s in job.get("stages") or [] if s.get("name") == name), None)
    if rec is None:
        return None
    started, finished = _parse_dt(rec.get("started_at")), _parse_dt(rec.get("finished_at"))
    if not (started and finished):
        return rec.get("energy_cpu_modelled"), None
    return rec.get("energy_cpu_modelled"), max((finished - started).total_seconds(), 0.0) / 3600.0


def _materialisation(job: dict, repo: str) -> Materialisation | None:
    progress = job.get("progress") or {}
    if _ADOPTED.match(str(progress.get("artefact") or "")):
        return None                       # built nothing; counted as a reuse
    counts = [c for c in progress.get("dataset_counts") or [] if isinstance(c, dict)]
    mine = next((c for c in counts if c.get("hf_repo") == repo), None)
    if mine is None:
        return None
    pages = int(mine.get("pages_written") or 0)
    total = sum(int(c.get("pages_written") or 0) for c in counts)
    if total == 0:                        # nothing written: split the job evenly
        pages, total = 1, len(counts)

    host = job.get("host")
    host_assumed = not host
    host = host or STORE_DEFAULT_HOST.get(str(job.get("_store") or ""))
    workers = ((job.get("request") or {}).get("params") or {}).get("workers")
    processor = _processor_for(host)
    cpu_wh = ram_wh = 0.0
    cpus = w_core = None
    recorded, hours, stages, why_not = True, 0.0, [], None
    any_stage = False
    for name in _STAGES:
        found = _stage_model(job, name)
        if found is None:
            continue
        any_stage = True
        model, stage_hours = found
        stages.append(name)
        if model and model.get("cpu_wh") is not None:
            cpu_wh += float(model["cpu_wh"])
            ram_wh += float(model.get("ram_wh") or 0.0)
            cpus, w_core = model.get("cpus"), model.get("w_per_core")
            hours += float(model.get("hours") or stage_hours or 0.0)
            continue
        recorded = False
        if stage_hours is None:
            why_not = f"no times on the {name} stage"
        elif processor not in CPU_TDP_W:
            why_not = f"processor of {host or 'an unknown host'} not in the TDP table"
        elif not workers:
            why_not = "the request names no workers"
        if why_not:
            cpu_wh = None  # type: ignore[assignment]
            break
        tdp, cores = CPU_TDP_W[processor]
        w_core = tdp / cores
        cpus = int(workers)
        hours += stage_hours
        cpu_wh += cpus * w_core * stage_hours
        ram = model.get("ram_gb") if model else None
        if ram:
            ram_wh += float(ram) * W_PER_GB_RAM * stage_hours
    if not any_stage:
        return None
    started = _parse_dt(job.get("started_at")) or _parse_dt(job.get("created_at"))
    return Materialisation(job_id=str(job.get("id")), store=str(job.get("_store") or ""),
                           host=host, host_assumed=host_assumed, started=started,
                           pages=pages, pages_total=total,
                           cpu_wh=None if cpu_wh is None else round(cpu_wh, 1),
                           ram_wh=None if cpu_wh is None else round(ram_wh, 1),
                           cpus=cpus, w_per_core=w_core, recorded=recorded,
                           hours=round(hours, 2), stages=stages, why_not=why_not)


def dataset_energy(jobs: list[dict], repo: str) -> DatasetEnergy:
    rows = {}
    for job in jobs:
        m = _materialisation(job, repo)
        if m is not None:
            rows[m.job_id] = m
    for job in jobs:
        hit = _ADOPTED.match(str((job.get("progress") or {}).get("artefact") or ""))
        if hit and hit.group("builder") in rows:
            rows[hit.group("builder")].reused_by.append(str(job.get("id")))
    ordered = sorted(rows.values(), key=lambda r: r.started.isoformat() if r.started else "")
    return DatasetEnergy(repo=repo, rows=ordered)


def markdown_section(energy: DatasetEnergy, *, generated: str) -> str:
    """The card section, between the markers, for :func:`splice`."""
    lines = [
        MARK_OPEN,
        "## Energy of materialising this dataset",
        "",
        "Training on this dataset starts by materialising a corpus from it (pages "
        "out of the parquet shards, lines cut, a trainer-specific compile). That is "
        "CPU work and no card is involved, so it is **modelled, not measured**: "
        "allocated cores × W per core at TDP × hours, memory by allocation where "
        "known (RAPL is root-only on both boxes; "
        "[method](https://github.com/thodel/training-atr-models/blob/main/docs/FOOTPRINT.md)). "
        "A job that built from several datasets has its energy apportioned by the "
        "pages each contributed. Jobs that adopted an existing corpus instead of "
        "building one are counted as reuses: the dataset is built once and trained "
        "many times.",
        "",
        "| built (UTC) | job | box | pages of this dataset | share | hours | modelled Wh attributed | basis | reused by |",
        "|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for r in energy.rows:
        when = r.started.strftime("%Y-%m-%d") if r.started else "—"
        wh = "—" if r.wh_attributed is None else f"{r.wh_attributed:,.1f}"
        basis = (f"not modelled: {r.why_not}" if r.wh_attributed is None
                 else ("recorded by the stage" if r.recorded
                       else f"after the fact: {r.cpus} cores × {r.w_per_core:.1f} W"))
        box = (r.host or "—") + (" (by store)" if r.host_assumed and r.host else "")
        lines.append(f"| {when} | `{r.job_id}` | {box} | {r.pages:,} | {100 * r.share:.0f} % "
                     f"| {r.hours:.2f} | {wh} | {basis} | {len(r.reused_by)} |")
    n = len(energy.rows)
    lines += [
        "",
        f"**{energy.modelled_wh:,.0f} Wh modelled** over {n} materialisation{'s' if n != 1 else ''}"
        + (f", reused {energy.reuses} time{'s' if energy.reuses != 1 else ''}" if energy.reuses else "")
        + (f"; {len(energy.unmodelled)} could not be modelled and are not in the sum" if energy.unmodelled else "")
        + ". Read from the job stores of the training programme on "
        f"{generated}; a later run replaces this section.",
        MARK_CLOSE,
    ]
    return "\n".join(lines)


def splice(readme: str, section: str) -> str:
    """Put *section* into a README: in place of an earlier one, else at the end.

    The front matter and everything ``pagexml-hf`` wrote stay as they are.
    """
    if MARK_OPEN in readme and MARK_CLOSE in readme:
        head, rest = readme.split(MARK_OPEN, 1)
        _, tail = rest.split(MARK_CLOSE, 1)
        return head + section + tail
    return readme.rstrip("\n") + "\n\n" + section + "\n"

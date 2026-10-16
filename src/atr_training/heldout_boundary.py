"""Which published numbers cross the held-out boundary? (#247)

The boundary (#98) is proven for the medieval ladder: the September corpus held
306,582 training lines, the October one 219,261, because
``_reserve_eval_documents`` keeps 3,173 of 11,277 pages for
``german-medieval-v1``. Unknown is which *other* published numbers cross it.

``progress.reserved_pages`` cannot answer that. It is a corpus fingerprint, not
a boundary marker — the number hangs on the dataset cut, not on the date — and
for 42 of 54 completed runs it is ``None`` or ``0``. So the check is done
against the material: every run still holds the manifest it trained on, and
today's held-out set can be applied to it.

Four states, not three
----------------------
#247 names three outcomes and warns that the third must not fall into the
first. There is a fourth, found while reading the code for this:

* ``CLEAN`` — entries name documents, and none of them is reserved today.
* ``CROSSED`` — some are. The number is optimistic by that many pages.
* ``UNCHECKABLE`` — the manifest names no document at all. A line corpus whose
  pages are foreign scans has no ``docId`` the registry could match; so does a
  chunk plan, which is what a chunked run leaves behind instead of a page list.
  This is **not** ``CLEAN``: nothing was ruled out.
* ``GONE`` — the data directory is empty (scratch purge after 30 days). Also
  not ``CLEAN``, and the reason to read both job stores: the share held 61 runs
  against 58 in scratch.

Reading ``UNCHECKABLE`` as ``CLEAN`` is the mistake this module exists to
prevent, and it is the same mistake the records themselves made: before #247
both of those runs recorded ``reserved_pages: 0``, which is also what a real
check that found nothing records.

Offline
-------
No GPU, no network, no model: today's registry against stored manifests. It
needs the job stores, so it runs where they are mounted (asteraix, or a box
with the research share) — not in a container that has neither.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from atr_training.chunking import CHUNK_PLAN_FILENAME
from atr_training.heldout import HeldOutDocuments, document_of_entry, load_heldout

__all__ = ["Verdict", "RunCheck", "TRAIN_MANIFESTS", "check_run", "check_store",
           "find_manifest", "summarize"]


class Verdict(str, Enum):
    """Why a run is or is not affected. A ``str`` enum so JSON output is plain."""

    #: The manifest was read and holds no document reserved today.
    CLEAN = "clean"
    #: The record shows this corpus was prepared under **today's** registry and
    #: that pages were really dropped. Same answer as ``CLEAN`` — the number
    #: stands — on different evidence, and kept apart from it for that reason:
    #: this one needs no manifest, so it decides runs whose scratch was purged.
    ENFORCED = "enforced"
    CROSSED = "crossed"
    UNCHECKABLE = "uncheckable"
    GONE = "gone"


#: Verdicts that answer the question #247 asks. ``ENFORCED`` and ``CLEAN`` both
#: mean the published number stands; the other three mean nobody knows yet.
DECIDED = (Verdict.CLEAN, Verdict.ENFORCED, Verdict.CROSSED)


#: Train manifests in the order they are looked for. A chunked run's plan is
#: last and is recognised as unreadable rather than parsed: it is not a manifest.
TRAIN_MANIFESTS = ("pages_train.lst", "lines_train.jsonl", CHUNK_PLAN_FILENAME)


@dataclass
class RunCheck:
    """One run's verdict, with the counts that produced it."""

    job_id: str
    store: str
    verdict: Verdict
    why: str
    manifest: str | None = None
    entries: int = 0
    identified: int = 0
    reserved: int = 0
    cer: float | None = None
    recorded_reserved: int | None = None
    #: Which documents of the manifest are held out today, for the write-up.
    documents: list[str] = field(default_factory=list)

    @property
    def row(self) -> str:
        """One Markdown table row, for ``docs/HELDOUT_BOUNDARY.md``."""
        cer = "—" if self.cer is None else f"{self.cer:.4f}"
        recorded = "None" if self.recorded_reserved is None else str(self.recorded_reserved)
        return (f"| `{self.job_id}` | {self.store} | {cer} | {recorded} | "
                f"{self.verdict.value} | {self.reserved} | {self.why} |")


def enforced_by_record(job: dict, today: str) -> bool:
    """Did this run's corpus get **today's** registry applied to it?

    Two fields, and the second one is the reason this is not a one-liner.

    ``progress.heldout_fingerprint`` is written at the *top* of
    ``_reserve_eval_documents``, before any of its checks, so it says "this
    registry was consulted" and not "it was applied". A chunked run from before
    #255 carries today's fingerprint and dropped nothing: the reservation read
    ``train_plan.json`` as a page list, found no document, and recorded ``0``.

    So a match is only decisive together with ``reserved_pages > 0`` — pages
    were really removed, and only a reservation that worked removes pages. A
    ``0`` beside today's fingerprint cannot be told apart, from the record
    alone, from that broken path, and falls through to the manifest.

    A **cache hit** is covered: the artefact payload carries the fingerprint and
    is copied onto the record, and an artefact built under a different registry
    has a different key and is never adopted (``runner_base`` says so at the
    hit). So a reused corpus answers for the registry it was built under, which
    is the honest answer.
    """
    progress = job.get("progress") or {}
    if progress.get("heldout_fingerprint") != today:
        return False
    reserved = progress.get("reserved_pages")
    return isinstance(reserved, int) and reserved > 0


def find_manifest(data_dir: Path) -> Path | None:
    """The train manifest of a run, or None when the directory holds none.

    Chunked runs are the reason this returns the plan rather than skipping it:
    its presence is the evidence that the train side was never a manifest, which
    is a different finding from an empty directory.
    """
    for name in TRAIN_MANIFESTS:
        candidate = data_dir / name
        if candidate.is_file():
            return candidate
    chunks = sorted(data_dir.glob("pages_train_[0-9]*.lst"))
    return chunks[0] if chunks else None


def check_run(job_dir: Path, *, store: str, held: HeldOutDocuments,
              job: dict | None = None, today: str | None = None) -> RunCheck:
    """Apply *held* to the manifest in *job_dir*; never guess a verdict.

    *job* is the parsed ``job.json`` when the caller has it, for the CER and the
    recorded ``reserved_pages`` — both go in the table beside the measurement, so
    a record that disagrees with the material is visible rather than overwritten.
    When it is not given it is read here: a caller that forgets would otherwise
    get a row with no CER and no recorded count, which is the comparison this
    census is for, missing without saying so.
    """
    if job is None:
        job = _read_record(job_dir) or {}
    progress = job.get("progress") or {}
    metrics = job.get("metrics") or {}
    common = dict(
        job_id=job_dir.name, store=store,
        cer=metrics.get("cer") if isinstance(metrics.get("cer"), (int, float)) else None,
        recorded_reserved=progress.get("reserved_pages"),
    )

    if today and enforced_by_record(job, today):
        reserved = (job.get("progress") or {}).get("reserved_pages")
        source = (job.get("progress") or {}).get("reserved_pages_source") or "?"
        return RunCheck(
            verdict=Verdict.ENFORCED, reserved=0,
            why=(f"prepared under today's registry ({today}), which dropped "
                 f"{reserved} page(s) — source: {source}"),
            **common)

    manifest = find_manifest(job_dir / "data")
    if manifest is None:
        return RunCheck(verdict=Verdict.GONE, why="no train manifest in data/", **common)

    if manifest.name == CHUNK_PLAN_FILENAME:
        return RunCheck(
            verdict=Verdict.UNCHECKABLE, manifest=manifest.name,
            why="chunked run: the train side was materialised per chunk and discarded",
            **common)

    try:
        text = manifest.read_text(encoding="utf-8")
    except OSError as exc:
        return RunCheck(verdict=Verdict.GONE, manifest=manifest.name,
                        why=f"manifest unreadable: {exc.strerror or exc}", **common)

    entries = [line for line in text.splitlines() if line.strip()]
    # Chunked compile writes one manifest per chunk; a run that got that far has
    # all of them, and checking only the first would under-count.
    if manifest.name.startswith("pages_train_"):
        for extra in sorted((job_dir / "data").glob("pages_train_[0-9]*.lst"))[1:]:
            entries += [line for line in extra.read_text(encoding="utf-8").splitlines()
                        if line.strip()]

    result = held.reserve(entries)
    if not result.measurable:
        return RunCheck(
            verdict=Verdict.UNCHECKABLE, manifest=manifest.name, entries=result.total,
            why=f"no document id in any of {result.total} entries", **common)

    documents = sorted({d for d in map(document_of_entry, result.reserved) if d})
    if result.reserved:
        return RunCheck(
            verdict=Verdict.CROSSED, manifest=manifest.name, entries=result.total,
            identified=result.identified, reserved=len(result.reserved),
            documents=documents,
            why=(f"{len(result.reserved)} of {result.total} training pages are "
                 f"reserved today ({len(documents)} documents)"), **common)

    return RunCheck(
        verdict=Verdict.CLEAN, manifest=manifest.name, entries=result.total,
        identified=result.identified,
        why=f"{result.identified} of {result.total} entries name a document; none reserved",
        **common)


def _read_record(job_dir: Path) -> dict | None:
    """The parsed ``job.json``, or None when it is missing or not parsable."""
    record = job_dir / "job.json"
    try:
        parsed = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def check_store(root: Path, *, store: str, held: HeldOutDocuments,
                completed_only: bool = True, today: str | None = None) -> list[RunCheck]:
    """Every run under *root*, newest first. Unreadable records are not skipped.

    A job whose ``job.json`` cannot be parsed still gets a row: #247 counts 54
    completed runs, and a row missing from the table without a reason is how a
    run quietly leaves a census.
    """
    checks: list[RunCheck] = []
    if not root.is_dir():
        return checks
    for job_dir in sorted((d for d in root.iterdir() if d.is_dir()), reverse=True):
        job = _read_record(job_dir)
        if job is None:
            checks.append(RunCheck(job_id=job_dir.name, store=store, verdict=Verdict.GONE,
                                   why="job.json missing or unreadable"))
            continue
        if completed_only and not _has_cer(job):
            continue
        checks.append(check_run(job_dir, store=store, held=held, job=job, today=today))
    return checks


def _has_cer(job: dict) -> bool:
    """A run with a CER is a run whose number was published somewhere (#247)."""
    metrics = job.get("metrics") or {}
    return any(isinstance(metrics.get(key), (int, float))
               for key in ("cer", "benchmark_cer"))


def summarize(checks: list[RunCheck]) -> dict[str, int]:
    """How many runs per verdict — the reach #247 asks for, in one line.

    ``decided`` is the number that are an answer about a published number;
    everything else is "nobody knows yet" and must never be read as a pass.
    """
    counts = {verdict.value: 0 for verdict in Verdict}
    for check in checks:
        counts[check.verdict.value] += 1
    counts["runs"] = len(checks)
    counts["decided"] = sum(counts[v.value] for v in DECIDED)
    counts["reserved_pages"] = sum(c.reserved for c in checks)
    return counts


def load_today(registry: str | Path | None = None) -> HeldOutDocuments:
    """Today's held-out set — the registry in this checkout, not the run's."""
    return load_heldout(registry)


def today_fingerprint() -> str:
    """The fingerprint a run prepared **now** would record.

    The same function the trainer writes onto the record and folds into the
    artefact key, so the comparison in :func:`enforced_by_record` is between two
    values of one definition rather than between two implementations of one idea.
    """
    from atr_training.artefact_cache import heldout_fingerprint

    return heldout_fingerprint()

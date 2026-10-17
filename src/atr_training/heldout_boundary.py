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

A run that **adopted** a compiled corpus instead of building one never had a
manifest of its own to lose: it trained on another job's bytes, and its record
says whose — ``progress.artefact = "<key> reused, built by <job>"``. Such a run
is ``GONE`` only until the builder is found. :func:`resolve_adoptions` takes the
builder's verdict over, but only when the builder's own record confirms the key
(``"<key> built by this job"``): two jobs can have built the same key, and the
name in an adopter's record is a label, the key is the identity. Measured on
2026-10-17: 16 of 24 runs without a manifest resolve this way, all clean, and
three of them recorded ``reserved_pages: 0`` where their builder had held pages
back — an adoption does not re-run the reservation, so the record keeps
whatever the job started with.

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
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from atr_training.chunking import CHUNK_PLAN_FILENAME
from atr_training.heldout import HeldOutDocuments, document_of_entry, load_heldout

__all__ = ["Verdict", "RunCheck", "TRAIN_MANIFESTS", "check_run", "check_store",
           "find_manifest", "resolve_adoptions", "summarize"]


class Verdict(str, Enum):
    """Why a run is or is not affected. A ``str`` enum so JSON output is plain."""

    CLEAN = "clean"
    CROSSED = "crossed"
    UNCHECKABLE = "uncheckable"
    GONE = "gone"


#: Train manifests in the order they are looked for. A chunked run's plan is
#: last and is recognised as unreadable rather than parsed: it is not a manifest.
TRAIN_MANIFESTS = ("pages_train.lst", "lines_train.jsonl", CHUNK_PLAN_FILENAME)

#: ``progress.artefact`` of a job that adopted a corpus, and of one that built it
#: (``runner_base._reuse_artefact`` / ``_publish_artefact``).
_ADOPTED = re.compile(r"^(?P<key>[0-9a-f]{12}) reused, built by (?P<builder>\S+)$")
_BUILT = "{key} built by this job"

#: What a job directory is called. A store also holds directories that are not
#: runs — ``evalset-federal-minutes`` keeps benchmark reports — and one of those
#: without a ``job.json`` is not a run whose manifest was lost.
_JOB_ID = re.compile(r"^\d{8}T\d{6}Z-")


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
    #: ``progress.artefact`` when the run adopted a corpus rather than built one.
    adopted: str | None = None
    #: The job whose manifest the verdict was read from, when it is not this one.
    inherited_from: str | None = None

    @property
    def row(self) -> str:
        """One Markdown table row, for ``docs/HELDOUT_BOUNDARY.md``."""
        cer = "—" if self.cer is None else f"{self.cer:.4f}"
        recorded = "None" if self.recorded_reserved is None else str(self.recorded_reserved)
        return (f"| `{self.job_id}` | {self.store} | {cer} | {recorded} | "
                f"{self.verdict.value} | {self.reserved} | {self.why} |")


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
              job: dict | None = None) -> RunCheck:
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
        adopted=(progress.get("artefact")
                 if _ADOPTED.match(str(progress.get("artefact") or "")) else None),
    )

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
                completed_only: bool = True) -> list[RunCheck]:
    """Every run under *root*, newest first. Unreadable records are not skipped.

    A job whose ``job.json`` cannot be parsed still gets a row: #247 counts 54
    completed runs, and a row missing from the table without a reason is how a
    run quietly leaves a census. A directory that is not named like a job and
    has no record is not a run, and gets none.

    *completed_only* keeps the runs whose record carries a CER. A published
    number can live outside the record — the medieval Gemma line arms are
    ``failed`` with no metrics, and their CER was recovered by re-scoring the
    adapter — so the census of what was *published* needs ``completed_only=False``
    as well (2026-10-17: all eight Gemma attempts crossed, two published figures).
    """
    checks: list[RunCheck] = []
    if not root.is_dir():
        return checks
    for job_dir in sorted((d for d in root.iterdir() if d.is_dir()), reverse=True):
        job = _read_record(job_dir)
        if job is None and not _JOB_ID.match(job_dir.name) \
                and not (job_dir / "job.json").exists():
            continue
        if job is None:
            checks.append(RunCheck(job_id=job_dir.name, store=store, verdict=Verdict.GONE,
                                   why="job.json missing or unreadable"))
            continue
        if completed_only and not _has_cer(job):
            continue
        checks.append(check_run(job_dir, store=store, held=held, job=job))
    return checks


def _has_cer(job: dict) -> bool:
    """A run with a CER is a run whose number was published somewhere (#247)."""
    metrics = job.get("metrics") or {}
    return any(isinstance(metrics.get(key), (int, float))
               for key in ("cer", "benchmark_cer"))


def resolve_adoptions(checks: list[RunCheck], stores: list[tuple[str, Path]], *,
                      held: HeldOutDocuments) -> list[RunCheck]:
    """Give a ``GONE`` run that adopted its corpus the verdict of the builder.

    An adopter trained on the builder's bytes, so the builder's manifest is the
    adopter's manifest. The builder is looked up in every store, whether or not
    it has a CER of its own — the run that built a corpus is often a cancelled
    first attempt. It must confirm the key in its own record: the adopter's
    "built by" names whoever registered the cache entry last, and a key can be
    rebuilt. A builder that does not confirm, or whose manifest is gone too,
    leaves the run ``GONE`` with the reason extended — never ``CLEAN``.
    """
    resolved: list[RunCheck] = []
    for check in checks:
        match = _ADOPTED.match(check.adopted or "")
        if check.verdict is not Verdict.GONE or match is None:
            resolved.append(check)
            continue
        key, builder = match["key"], match["builder"]
        source = _find_builder(builder, key, stores, held)
        if isinstance(source, str):
            check.why = f"{check.why}; adopted {key} from {builder}, but {source}"
            resolved.append(check)
            continue
        if source.verdict not in (Verdict.CLEAN, Verdict.CROSSED):
            check.why = (f"{check.why}; adopted {key} from {builder}, "
                         f"whose verdict is {source.verdict.value}: {source.why}")
            resolved.append(check)
            continue
        resolved.append(RunCheck(
            job_id=check.job_id, store=check.store, verdict=source.verdict,
            why=f"adopted {key} from {builder} ({source.store}): {source.why}",
            manifest=source.manifest, entries=source.entries,
            identified=source.identified, reserved=source.reserved,
            documents=source.documents, cer=check.cer,
            recorded_reserved=check.recorded_reserved, adopted=check.adopted,
            inherited_from=builder))
    return resolved


def _find_builder(job_id: str, key: str, stores: list[tuple[str, Path]],
                  held: HeldOutDocuments) -> RunCheck | str:
    """The builder's own check, or the reason it cannot stand in (a string)."""
    found_record = False
    for label, root in stores:
        job_dir = Path(root) / job_id
        job = _read_record(job_dir)
        if job is None:
            continue
        found_record = True
        artefact = str((job.get("progress") or {}).get("artefact") or "")
        if artefact != _BUILT.format(key=key):
            continue
        check = check_run(job_dir, store=label, held=held, job=job)
        if check.verdict is not Verdict.GONE:
            return check
    if not found_record:
        return "its record is in no store"
    return f"no copy of it confirms building {key} and still holds a manifest"


def summarize(checks: list[RunCheck]) -> dict[str, int]:
    """How many runs per verdict — the reach #247 asks for, in one line."""
    counts = {verdict.value: 0 for verdict in Verdict}
    for check in checks:
        counts[check.verdict.value] += 1
    counts["runs"] = len(checks)
    counts["reserved_pages"] = sum(c.reserved for c in checks)
    return counts


def load_today(registry: str | Path | None = None) -> HeldOutDocuments:
    """Today's held-out set — the registry in this checkout, not the run's."""
    return load_heldout(registry)

"""Take the held-out lines back out of ``train_lines`` written before #256.

Since ``2ebebb8`` the reservation corrects ``train_lines`` itself
(``runner_base._drop_reserved_from_train_lines``). Everything prepared before
that still carries the count from **before** the reservation — and the copy
that matters is not the job record but the artefact-cache payload: a job that
adopts a compiled corpus copies ``train_lines`` out of the payload into its own
record, and for a line-level run the step-count guard (#72) divides by it
(page and block runs divide by ``train_samples`` since #264). The cache key did not change with the fix, so those entries stay adoptable and
hand the old number to every new run that uses them.

The correction is the one the fix makes, made after the fact: the transcribed
lines of the pages in ``pages_reserved.lst``, counted with the same
:func:`~atr_training.prepare.transcribed_lines_of`, subtracted from what the
payload or record says. Two counts travel beside it and are reported, never
used: the same function over ``pages_train.lst``, and the rows of
``train.jsonl``. Where all three agree the correction is proven; where they do
not, the row says so. Measured 19.10.2026 they do not, and for a reason that is
not the reservation: the old value was itself an estimate (``lines *
partition`` for the datasets split by seed), 2,167 lines (1 %) above the exact
sum on the medieval corpus. The correction is the fix's own subtraction, so a
corrected payload says what a rebuild with today's code would say. And for a
page corpus ``train.jsonl`` counts pages, not lines.

What is never done:

* **a correction from incomplete evidence.** A reserved page whose file cannot
  be found makes the row ``skip`` — a partial subtraction is a number nobody
  measured;
* **a second subtraction.** A cache payload that was corrected carries
  ``train_lines_recount``, a job record a ``train_lines_recount.json`` beside
  it; a job whose prepare log shows the #256 line, or that prepared its own
  corpus after the fix, was corrected at the time;
* **touching a live job.** Only terminal records are rewritten, and only the
  value of ``progress.train_lines`` — the record is the run's own evidence, so
  the old value goes into this tool's report instead.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from atr_training.prepare import transcribed_lines_of

__all__ = ["FIX_LANDED", "Recount", "recount_cache_entry", "recount_job",
           "resolve_pages", "TERMINAL"]

#: When ``5b5ec6b`` (#257, which carries ``2ebebb8``) reached ``main``: a corpus
#: prepared from then on corrected itself.
FIX_LANDED = datetime(2026, 10, 16, 19, 16, 50, tzinfo=timezone.utc)

#: Statuses whose record nobody writes any more.
TERMINAL = frozenset({"completed", "failed", "cancelled"})

#: The log line the fix writes; its presence means the job corrected itself.
FIX_LOG_LINE = "held-out line(s) (#256)"

#: Beside ``job.json``: this tool's record of a correction it made. Not a field
#: in the record, because the progress model drops a field it does not know the
#: next time anything saves the job.
SIDECAR = "train_lines_recount.json"

_ADOPTED = re.compile(r"^(?P<key>[0-9a-f]{12}) reused, built by \S+$")
_JOB_STAMP = re.compile(r"^(\d{8}T\d{6}Z)-")


@dataclass
class Recount:
    """One payload or record: what it says, what it should say, and why."""

    kind: str                     # "cache" or "job"
    ident: str                    # cache key (12) or job id
    verdict: str                  # "correct", "already", "skip"
    why: str
    before: int | None = None
    removed: int | None = None
    after: int | None = None
    #: The two counts reported beside the correction, never used to make it.
    train_pages_lines: int | None = None
    train_jsonl_rows: int | None = None
    key: str | None = None        # the corpus, for adopters and their builders

    @property
    def agrees(self) -> bool | None:
        """Do the independent counts confirm ``after``? None if none was taken."""
        checks = [c for c in (self.train_pages_lines, self.train_jsonl_rows) if c is not None]
        if self.after is None or not checks:
            return None
        return all(c == self.after for c in checks)


def resolve_pages(manifest: Path, data_dir: Path) -> tuple[list[Path], int]:
    """The files a page manifest names, and how many of them cannot be found.

    Manifests carry absolute paths into the job directory that built them, and
    that directory may be gone while the cache entry still holds the pages; so
    a path that does not exist is looked up by name in ``data_dir/pages``.
    """
    found: list[Path] = []
    missing = 0
    for line in manifest.read_text(encoding="utf-8").split():
        path = Path(line)
        if not path.is_file():
            path = data_dir / "pages" / path.name
        if path.is_file():
            found.append(path)
        else:
            missing += 1
    return found, missing


def _count(data_dir: Path) -> tuple[int | None, int, int | None, int | None]:
    """``(removed, missing, train_pages_lines, train_jsonl_rows)`` for *data_dir*."""
    reserved_list = data_dir / "pages_reserved.lst"
    if not reserved_list.is_file():
        return None, 0, None, None
    reserved, missing = resolve_pages(reserved_list, data_dir)
    removed = transcribed_lines_of(reserved) if not missing else None
    train_lines = None
    train_list = data_dir / "pages_train.lst"
    if train_list.is_file():
        train, train_missing = resolve_pages(train_list, data_dir)
        train_lines = transcribed_lines_of(train) if not train_missing else None
    rows = None
    jsonl = data_dir / "train.jsonl"
    if jsonl.is_file():
        with jsonl.open(encoding="utf-8") as handle:
            rows = sum(1 for line in handle if line.strip())
    return removed, missing, train_lines, rows


def recount_cache_entry(entry_dir: Path) -> Recount:
    """The payload of one artefact-cache entry, recounted from its own pages."""
    key = entry_dir.name[:12]
    try:
        meta = json.loads((entry_dir / "artefact.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return Recount("cache", key, "skip", f"artefact.json unreadable: {exc}", key=key)
    payload = meta.get("payload") or {}
    before = payload.get("train_lines")
    if not payload.get("reserved_pages"):
        return Recount("cache", key, "already", "nothing was reserved", before=before, key=key)
    if payload.get("train_lines_recount"):
        return Recount("cache", key, "already",
                       "recounted on %s" % payload["train_lines_recount"].get("at"),
                       before=before, key=key)
    built = datetime.fromtimestamp(float(meta.get("built_at") or 0), tz=timezone.utc)
    if built >= FIX_LANDED:
        return Recount("cache", key, "already", "built after the fix corrected itself",
                       before=before, key=key)
    if not isinstance(before, int):
        return Recount("cache", key, "skip", "payload carries no train_lines", key=key)
    removed, missing, pages_lines, rows = _count(entry_dir / "data")
    if removed is None:
        why = (f"{missing} reserved page(s) not found" if missing
               else "no pages_reserved.lst in the entry")
        return Recount("cache", key, "skip", why, before=before, key=key,
                       train_pages_lines=pages_lines, train_jsonl_rows=rows)
    return Recount("cache", key, "correct", "built before the fix",
                   before=before, removed=removed, after=max(0, before - removed),
                   train_pages_lines=pages_lines, train_jsonl_rows=rows, key=key)


def recount_job(job_dir: Path, by_key: dict[str, Recount]) -> Recount:
    """One job record: recounted from its own pages, or through its corpus.

    *by_key* holds the recount of every corpus seen so far, from the cache and
    from builders: an adopter takes the builder's correction, but only when its
    record still holds the builder's old value — anything else was written by
    something this tool does not know about.
    """
    job_id = job_dir.name
    try:
        record = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return Recount("job", job_id, "skip", f"job.json unreadable: {exc}")
    progress = record.get("progress") or {}
    before = progress.get("train_lines")
    if not progress.get("reserved_pages"):
        return Recount("job", job_id, "already", "nothing was reserved", before=before)
    if (job_dir / SIDECAR).is_file():
        return Recount("job", job_id, "already", f"recounted before ({SIDECAR})", before=before)
    if record.get("status") not in TERMINAL:
        return Recount("job", job_id, "skip", f"status {record.get('status')!r} — not touched",
                       before=before)
    if not isinstance(before, int):
        return Recount("job", job_id, "skip", "record carries no train_lines")
    # An adopter first, and regardless of its date: the fix corrects the
    # reservation step, which an adopter skips, so a run that adopted an old
    # payload *after* the fix still copied the old number (measured 19.10.2026:
    # a throughput run of 17.10. holding 306,569).
    adopted = _ADOPTED.match(str(progress.get("artefact") or ""))
    if adopted:
        key = adopted["key"]
        corpus = by_key.get(key)
        if corpus is None or corpus.verdict != "correct":
            return Recount("job", job_id, "skip",
                           f"adopted {key}, whose recount is not available", before=before, key=key)
        if before != corpus.before:
            return Recount("job", job_id, "skip",
                           f"adopted {key}, but holds {before} where the corpus said "
                           f"{corpus.before}", before=before, key=key)
        return Recount("job", job_id, "correct", f"adopted {key}",
                       before=before, removed=corpus.removed, after=corpus.after,
                       train_pages_lines=corpus.train_pages_lines,
                       train_jsonl_rows=corpus.train_jsonl_rows, key=key)

    stamp = _JOB_STAMP.match(job_id)
    prepared = (datetime.strptime(stamp.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                if stamp else None)
    if _log_shows_fix(job_dir) or (prepared is not None and prepared >= FIX_LANDED):
        return Recount("job", job_id, "already", "prepared with the fix", before=before)

    removed, missing, pages_lines, rows = _count(job_dir / "data")
    key_match = re.match(r"^([0-9a-f]{12}) built by this job", str(progress.get("artefact") or ""))
    key = key_match.group(1) if key_match else None
    if removed is None:
        corpus = by_key.get(key) if key else None
        if corpus is not None and corpus.verdict == "correct" and before == corpus.before:
            return Recount("job", job_id, "correct", f"built {key}; recounted from the cache entry",
                           before=before, removed=corpus.removed, after=corpus.after,
                           train_pages_lines=corpus.train_pages_lines,
                           train_jsonl_rows=corpus.train_jsonl_rows, key=key)
        why = (f"{missing} reserved page(s) not found" if missing
               else "no pages_reserved.lst left")
        return Recount("job", job_id, "skip", why, before=before, key=key)
    return Recount("job", job_id, "correct", "prepared before the fix",
                   before=before, removed=removed, after=max(0, before - removed),
                   train_pages_lines=pages_lines, train_jsonl_rows=rows, key=key)


def _log_shows_fix(job_dir: Path) -> bool:
    for log in (job_dir / "logs").glob("*.log"):
        try:
            if FIX_LOG_LINE in log.read_text(encoding="utf-8", errors="replace"):
                return True
        except OSError:
            continue
    return False


def apply_to_cache(entry_dir: Path, recount: Recount) -> None:
    """Write the corrected payload, with the old value beside it.

    The cache's own ``_amend`` is the model: read, change, write, no lock — the
    worst a race with a concurrent ``touch`` can cost is one ``last_used``
    stamp. ``train_lines_recount`` is not a progress field, so an adopter that
    copies the payload into its record ignores it.
    """
    manifest = entry_dir / "artefact.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    payload = data.setdefault("payload", {})
    payload["train_lines_recount"] = {
        "before": recount.before, "removed": recount.removed,
        "at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "issue": "#256"}
    payload["train_lines"] = recount.after
    _write_atomically(manifest, data)


def apply_to_job(job_dir: Path, recount: Recount) -> None:
    """Change ``progress.train_lines`` and nothing else in the record.

    The sidecar goes first: a crash between the two writes leaves a correction
    missing, which a later run reports, rather than one applied twice.
    """
    record_path = job_dir / "job.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("status") not in TERMINAL:
        raise RuntimeError(f"{job_dir.name} became {record.get('status')!r} — not touched")
    _write_atomically(job_dir / SIDECAR, {
        "before": recount.before, "removed": recount.removed, "after": recount.after,
        "why": recount.why, "at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "issue": "#256"})
    record["progress"]["train_lines"] = recount.after
    _write_atomically(record_path, record)


def _write_atomically(path: Path, data: dict) -> None:
    handle, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise

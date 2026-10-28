"""Rewrite ``steps_per_epoch`` / ``total_steps`` of page runs in samples (#264).

Until #264 the step-count guard divided ``progress.train_lines`` by the
effective batch, and wrote the result into the record. For a VLM run at
``granularity: page``, ``block`` or ``mixed`` that is the wrong unit: the
trainer steps over pages or blocks, not over the lines they contain. The
olmOCR-2 medieval page run trained on 8,088 pages at batch 16 — 506 steps an
epoch, ``trainer_state.json`` says 1,518 after three — and its record says
19,161, which is ``ceil(306,569 / 16)``: the line count from before #256, at
that. The #256 recount corrected ``train_lines`` and left these two fields as
they were.

The correction is what the guard computes since #264, from the count the
trainer was handed: the rows of the run's ``train.jsonl``, through
:func:`~atr_training.convergence.plan_steps` with the record's own effective
batch, epochs and ``max_steps``. Where the job adopted its corpus from the
artefact cache, the rows are those of the corpus it adopted — the builder's
``train.jsonl`` or the cache entry's, found by the key in ``progress.artefact``.

Two counts travel beside it and are reported, never used: ``train_samples``
from the trainer's own ``training_summary.json``, and the steps per epoch the
trainer took (``global_step / epoch`` of its last checkpoint, when that ended on
a whole epoch). Where they agree the correction is proven; where they do not,
the row says so.

``total_steps`` stays what the guard means by it: the steps of the *requested*
epochs. A run that early-stopping continuation (``max_epochs``) carried past
them took more, and the trainer column shows that rather than the record
pretending otherwise.

What is never done:

* **line runs.** At ``granularity: line`` a sample is a line, and the budget was
  already in the right unit; such records are not listed;
* **a correction without evidence.** No ``train.jsonl`` anywhere for the corpus
  makes the row ``skip``;
* **a second rewrite.** A record that was rewritten carries a
  ``step_budget_recount.json`` beside it;
* **touching a live job.** Only terminal records, and only the two fields —
  the old values go into the sidecar.

It runs on UBELIX, where the scratch store is mounted, and **inside the
container**: importing ``atr_training`` imports ``contracts``, which needs
pydantic, and the login node's ``python3`` (3.9) has none. The first ``--apply``
on 28.10.2026 died of exactly that, at import, before reading a record. The
dry run of 19.10. had passed only because it ran from a copy with an empty
package ``__init__``. The command is in ``scripts/recount_step_budget.py``.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from atr_training.convergence import plan_steps

__all__ = ["SIDECAR", "TERMINAL", "BudgetRecount", "apply_to_job", "count_rows",
           "recount_job"]

#: Statuses whose record nobody writes any more.
TERMINAL = frozenset({"completed", "failed", "cancelled"})

#: Beside ``job.json``: this tool's record of a rewrite it made. Not a field in
#: the record, because the progress model drops a field it does not know the
#: next time anything saves the job.
SIDECAR = "step_budget_recount.json"

_ADOPTED = re.compile(r"^(?P<key>[0-9a-f]{12}) reused, built by (?P<builder>\S+)$")
_BUILT = re.compile(r"^(?P<key>[0-9a-f]{12}) built by this job")


@dataclass
class BudgetRecount:
    """One job record: what its budget says, what it should say, and why."""

    ident: str
    verdict: str                  # "correct", "already", "skip", "line"
    why: str
    granularity: Optional[str] = None
    samples: Optional[int] = None
    source: Optional[str] = None  # where the samples were counted
    effective_batch: Optional[int] = None
    epochs: Optional[int] = None
    max_steps: Optional[int] = None
    steps_before: Optional[int] = None
    steps_after: Optional[int] = None
    total_before: Optional[int] = None
    total_after: Optional[int] = None
    #: The two counts reported beside the correction, never used to make it.
    trainer_samples: Optional[int] = None
    trainer_steps_per_epoch: Optional[int] = None

    @property
    def agrees(self) -> Optional[bool]:
        """Do the trainer's own counts confirm the rewrite? None if none exists."""
        checks = []
        if self.trainer_samples is not None and self.samples is not None:
            checks.append(self.trainer_samples == self.samples)
        if self.trainer_steps_per_epoch is not None and self.steps_after is not None:
            checks.append(self.trainer_steps_per_epoch == self.steps_after)
        return all(checks) if checks else None


def count_rows(jsonl: Path) -> int:
    with jsonl.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def recount_job(job_dir: Path, jobs_by_id: Dict[str, Path],
                caches_by_key: Dict[str, Path]) -> BudgetRecount:
    """One job record, against the corpus it trained on.

    *jobs_by_id* maps every job id in the stores to its directory, so an
    adopter finds its builder; *caches_by_key* maps a 12-character artefact key
    to its cache entry, for a builder whose job directory is gone.
    """
    job_id = job_dir.name
    try:
        record = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return BudgetRecount(job_id, "skip", f"job.json unreadable: {exc}")
    request = record.get("request") or {}
    params = request.get("params") or {}
    progress = record.get("progress") or {}
    granularity = params.get("granularity")
    out = BudgetRecount(job_id, "skip", "", granularity=granularity,
                        steps_before=progress.get("steps_per_epoch"),
                        total_before=progress.get("total_steps"))
    if request.get("engine") != "vllm" or granularity in (None, "line"):
        out.verdict, out.why = "line", "a sample is a line here"
        return out
    if out.steps_before is None:
        out.verdict, out.why = "line", "no budget recorded"
        return out
    if (job_dir / SIDECAR).is_file():
        out.verdict, out.why = "already", f"rewritten before ({SIDECAR})"
        return out
    if record.get("status") not in TERMINAL:
        out.why = f"status {record.get('status')!r} — not touched"
        return out

    out.effective_batch = (int(params.get("batch_size") or 1)
                           * int(params.get("accumulate_grad_batches") or 1))
    out.epochs = int(params.get("epochs") or 1)
    out.max_steps = params.get("max_steps")
    out.trainer_samples, out.trainer_steps_per_epoch = _trainer_counts(record)

    found = _corpus_rows(job_dir, progress, jobs_by_id, caches_by_key)
    if isinstance(found, str):
        out.why = found
        return out
    out.samples, out.source = found

    budget = plan_steps(out.samples, out.effective_batch, out.epochs, cap=out.max_steps)
    out.steps_after, out.total_after = budget.steps_per_epoch, budget.total_steps
    if (out.steps_after, out.total_after) == (out.steps_before, out.total_before):
        out.verdict, out.why = "already", "the budget is already in samples"
        return out
    out.verdict, out.why = "correct", f"{granularity} run budgeted in lines"
    return out


def _corpus_rows(job_dir: Path, progress: dict, jobs_by_id: Dict[str, Path],
                 caches_by_key: Dict[str, Path]):
    """``(rows, source)`` of the train.jsonl this run trained on, or why not."""
    own = job_dir / "data" / "train.jsonl"
    artefact = str(progress.get("artefact") or "")
    adopted = _ADOPTED.match(artefact)
    if adopted is None:
        if own.is_file():
            return count_rows(own), "own train.jsonl"
        built = _BUILT.match(artefact)
        entry = caches_by_key.get(built["key"]) if built else None
        if entry is not None and (entry / "data" / "train.jsonl").is_file():
            return count_rows(entry / "data" / "train.jsonl"), f"cache {built['key']}"
        return "no train.jsonl left"
    key, builder = adopted["key"], adopted["builder"]
    builder_dir = jobs_by_id.get(builder)
    if builder_dir is not None and (builder_dir / "data" / "train.jsonl").is_file():
        try:
            built_record = json.loads((builder_dir / "job.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            built_record = {}
        built_artefact = str((built_record.get("progress") or {}).get("artefact") or "")
        if built_artefact.startswith(key):
            return count_rows(builder_dir / "data" / "train.jsonl"), f"builder {builder}"
    entry = caches_by_key.get(key)
    if entry is not None and (entry / "data" / "train.jsonl").is_file():
        return count_rows(entry / "data" / "train.jsonl"), f"cache {key}"
    return f"adopted {key}; neither its builder nor the cache entry holds train.jsonl"


def _trainer_counts(record: dict):
    """``(train_samples, steps_per_epoch)`` as the trainer itself recorded them."""
    checkpoint = record.get("checkpoint_dir")
    if not checkpoint:
        return None, None
    root = Path(checkpoint)
    samples = None
    try:
        summary = json.loads((root / "training_summary.json").read_text(encoding="utf-8"))
        if isinstance(summary.get("train_samples"), int):
            samples = summary["train_samples"]
    except (OSError, ValueError):
        pass
    per_epoch = None
    states = sorted(root.glob("checkpoint-*/trainer_state.json"),
                    key=lambda p: int(p.parent.name.split("-")[-1])
                    if p.parent.name.split("-")[-1].isdigit() else -1)
    if states:
        try:
            state = json.loads(states[-1].read_text(encoding="utf-8"))
            epoch, step = float(state.get("epoch") or 0), int(state.get("global_step") or 0)
            # Only a checkpoint that ended on a whole epoch says how long one was.
            if epoch >= 1 and abs(epoch - round(epoch)) < 1e-6 and step % round(epoch) == 0:
                per_epoch = step // round(epoch)
        except (OSError, ValueError, TypeError):
            pass
    return samples, per_epoch


def apply_to_job(job_dir: Path, recount: BudgetRecount) -> None:
    """Change ``steps_per_epoch`` and ``total_steps`` and nothing else.

    The sidecar goes first: a crash between the two writes leaves a rewrite
    missing, which a later run reports, rather than one applied twice.
    """
    record_path = job_dir / "job.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("status") not in TERMINAL:
        raise RuntimeError(f"{job_dir.name} became {record.get('status')!r} — not touched")
    _write_atomically(job_dir / SIDECAR, {
        "steps_per_epoch": {"before": recount.steps_before, "after": recount.steps_after},
        "total_steps": {"before": recount.total_before, "after": recount.total_after},
        "train_samples": recount.samples, "source": recount.source,
        "effective_batch": recount.effective_batch, "epochs": recount.epochs,
        "max_steps": recount.max_steps, "why": recount.why,
        "at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "issue": "#264"})
    record["progress"]["steps_per_epoch"] = recount.steps_after
    record["progress"]["total_steps"] = recount.total_after
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

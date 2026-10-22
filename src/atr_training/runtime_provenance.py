"""Which image produced a record's number, after the fact (#206).

Since 22.10.2026 every stage the runner opens carries ``runtime`` (image and
transformers). For the records before that the answer has to be reconstructed,
and there is exactly one place it can come from: the Slurm log of the attempt
that ran the stage, which since 10.10.2026 prints

    == SIF: /storage/homefs/th19c587/ubelix/vlm-train-tf5.sif (transformers 5.17.0)

before anything loads. A log without that line — every run before 10.10., and
every log that was deleted — leaves the record **unknown**, and unknown is what
this writes down, never a guess from the model type or the date: the image a
job ran in came from the submitting environment, which the record does not
hold.

What is never done:

* a stage that already carries ``runtime`` is not touched;
* a live job is not touched;
* a second rewrite: a record that was rewritten carries a
  ``runtime_provenance.json`` beside it, with the log the image was read from.

Python 3.9-compatible on purpose: it runs on the UBELIX login node.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from atr_results_mcp.probe import _attempt_in, read_log_edges

__all__ = ["SIDECAR", "TERMINAL", "RuntimeRecovery", "apply_to_job", "recover_job", "sif_line"]

TERMINAL = frozenset({"completed", "failed", "cancelled"})
SIDECAR = "runtime_provenance.json"
#: The line the batch scripts print since 10.10.2026 (#206).
_SIF = re.compile(r"^== SIF: (\S+)(?: \(transformers ([^)]+)\))?")
#: The GPU stages: the ones a Slurm attempt runs, and the ones a CER depends on.
CARD_STAGES = ("train", "test")


@dataclass
class RuntimeRecovery:
    ident: str
    verdict: str              # "recover", "already", "unknown", "skip"
    why: str
    image: Optional[str] = None
    transformers: Optional[str] = None
    log: Optional[str] = None
    stages: Tuple[str, ...] = ()


def sif_line(lines: List[str]) -> Optional[Tuple[str, Optional[str]]]:
    """``(image, transformers)`` from the LAST ``== SIF:`` line: a requeue appends."""
    for line in reversed(lines):
        match = _SIF.match(line)
        if match:
            version = match.group(2)
            return match.group(1), (None if version in (None, "nicht abfragbar") else version)
    return None


def logs_of(job_id: str, logs_dir: Path) -> List[Path]:
    """The Slurm logs whose attempt line names *job_id*, oldest first."""
    found = []
    for path in sorted(logs_dir.glob("*.out")):
        try:
            edges = read_log_edges(path)
        except OSError:
            continue
        attempt = _attempt_in(edges["tail"]) or _attempt_in(edges["head"])
        if attempt is not None and attempt[0] == job_id:
            found.append((path.stat().st_mtime, path))
    return [path for _mtime, path in sorted(found)]


def recover_job(job_dir: Path, logs_dir: Path) -> RuntimeRecovery:
    job_id = job_dir.name
    try:
        record = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return RuntimeRecovery(job_id, "skip", "job.json unreadable: %s" % exc)
    stages = [s for s in record.get("stages") or [] if s.get("name") in CARD_STAGES]
    if not stages:
        return RuntimeRecovery(job_id, "skip", "no GPU stage ran")
    missing = tuple(s["name"] for s in stages if "runtime" not in s)
    if not missing:
        return RuntimeRecovery(job_id, "already", "every GPU stage carries a runtime",
                               stages=tuple(s["name"] for s in stages))
    if (job_dir / SIDECAR).is_file():
        return RuntimeRecovery(job_id, "already", "rewritten before (%s)" % SIDECAR)
    if record.get("status") not in TERMINAL:
        return RuntimeRecovery(job_id, "skip", "status %r — not touched" % record.get("status"))
    # The newest log with a SIF line: the last attempt is the one that wrote
    # the stages that stand.
    for path in reversed(logs_of(job_id, logs_dir)):
        edges = read_log_edges(path)
        found = sif_line(edges["head"]) or sif_line(edges["tail"])
        if found:
            image, version = found
            return RuntimeRecovery(job_id, "recover", "from the SIF line of %s" % path.name,
                                   image=image, transformers=version, log=str(path),
                                   stages=missing)
    return RuntimeRecovery(job_id, "unknown",
                           "no Slurm log of this record prints a SIF line (before 10.10.2026, "
                           "or the log is gone)", stages=missing)


def apply_to_job(job_dir: Path, recovery: RuntimeRecovery) -> None:
    """Write ``runtime`` into the stages named, and nothing else. Sidecar first."""
    if recovery.verdict != "recover":
        raise RuntimeError("%s: nothing to apply (%s)" % (job_dir.name, recovery.verdict))
    record_path = job_dir / "job.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("status") not in TERMINAL:
        raise RuntimeError("%s became %r — not touched" % (job_dir.name, record.get("status")))
    _write_atomically(job_dir / SIDECAR, {
        "runtime": {"image": recovery.image, "transformers": recovery.transformers},
        "stages": list(recovery.stages), "from": recovery.log,
        "at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "issue": "#206"})
    for stage in record.get("stages") or []:
        if stage.get("name") in recovery.stages and "runtime" not in stage:
            stage["runtime"] = {"image": recovery.image, "transformers": recovery.transformers}
    _write_atomically(record_path, record)


def _write_atomically(path: Path, data: Dict) -> None:
    handle, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise

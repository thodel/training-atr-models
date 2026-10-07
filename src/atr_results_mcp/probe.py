#!/usr/bin/env python3
"""The half of the results MCP that runs ON UBELIX.

Constraints this file lives under, and the test suite enforces:

- **Python 3.9.** The login node (submit02) has ``/usr/bin/python3`` 3.9.25 and
  nothing else outside a container. No ``match``, no ``X | Y`` at runtime, no
  ``zip(strict=)``. ``from __future__ import annotations`` keeps the annotations
  harmless.
- **Standard library only.** It is sent over ``ssh host python3 - <cmd> <json>``
  with its source on stdin, so there is nothing to install and nothing to keep in
  sync on the cluster.
- **Read-only.** It calls ``squeue``, ``sacct``, ``scontrol show``, ``sinfo``,
  ``df`` and ``git`` (``fetch`` included, which touches ``.git`` but not the tree),
  and it reads files. No ``sbatch``, no ``scancel``, no deletion.

Every entry point takes ``run`` (how to execute a command) and ``paths`` (where
things are) so the suite can drive it on a temporary directory with canned Slurm
output and never needs the cluster.

Times: Slurm prints local time already. ``job.json`` carries UTC with a ``Z``;
those are converted to Europe/Zurich and labelled ``…_cest``. Epoch seconds in
``artefact.json`` are converted the same way.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

ZURICH = ZoneInfo("Europe/Zurich")

#: A job whose Slurm attempt lasted less than this did not run: it is a requeue
#: that found the record finished, or a refusal before the first real step (#153).
SHORT_RUN_SECONDS = 10

#: Unpinned artefacts expire this long after ``built_at`` (artefact_cache).
ARTEFACT_TTL = timedelta(days=7)
#: ``/scratch`` purges what has not been touched for this long.
SCRATCH_PURGE = timedelta(days=30)

#: If the projected total time of a running job is closer than this to its wall,
#: say "at risk", not "fits": on 05.10.2026 a job at 150/164 that was called
#: "makes it, barely" ran into TIMEOUT.
WALL_MARGIN = 0.15

Runner = Callable[[List[str]], str]


class ProbeError(RuntimeError):
    """A command the probe depends on failed; the message says which and why."""


# ── where things are ────────────────────────────────────────────────────────
class Paths:
    """The cluster layout, resolvable from the environment or set by a test."""

    def __init__(self, user: Optional[str] = None, home: Optional[Path] = None,
                 scratch: Optional[Path] = None) -> None:
        self.user = user or os.environ.get("USER") or "th19c587"
        self.home = Path(home or os.environ.get("HOME") or ("/storage/homefs/" + self.user))
        self.scratch = Path(scratch or ("/scratch/network/users/" + self.user))

    @property
    def jobs_root(self) -> Path:
        return self.scratch / "runs" / "jobs"

    @property
    def artefacts_root(self) -> Path:
        return self.scratch / "expA" / "artefacts"

    @property
    def logs(self) -> Path:
        return self.home / "ubelix" / "logs"

    @property
    def checkout(self) -> Path:
        return self.home / "training-atr-models"


def shell_runner(argv: List[str], timeout: float = 60.0) -> str:
    """Run a command and return its stdout; a non-zero exit is a ProbeError."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        raise ProbeError("%s: not found on this host" % argv[0])
    except subprocess.TimeoutExpired:
        raise ProbeError("%s: no answer within %.0f s" % (" ".join(argv[:2]), timeout))
    if proc.returncode != 0:
        raise ProbeError("%s: exit %d: %s" % (" ".join(argv[:3]), proc.returncode,
                                               proc.stderr.strip()[-400:]))
    return proc.stdout


# ── time helpers ────────────────────────────────────────────────────────────
def utc_to_cest(value: Optional[str]) -> Optional[str]:
    """``2026-10-05T21:42:21.235264Z`` → ``2026-10-05 23:42 CEST``."""
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return value
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return _label(stamp.astimezone(ZURICH))


def epoch_to_cest(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return _label(datetime.fromtimestamp(float(value), tz=ZURICH))


def _label(stamp: datetime) -> str:
    return stamp.strftime("%Y-%m-%d %H:%M %Z")


def slurm_seconds(text: str) -> Optional[int]:
    """``27:58`` → 1678, ``1-00:00:00`` → 86400, ``12:00:16`` → 43216. Unknown → None."""
    text = (text or "").strip()
    if not text or text in ("UNLIMITED", "N/A", "Unknown", "INVALID"):
        return None
    days = 0
    if "-" in text:
        day_part, text = text.split("-", 1)
        days = int(day_part)
    parts = [int(p) for p in text.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, (minutes, seconds) = 0, parts
    elif len(parts) == 1:
        hours, minutes, seconds = 0, 0, parts[0]
    else:
        return None
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


# ── log reading ─────────────────────────────────────────────────────────────
#: Lines that say nothing to a reader: per-crop DEBUG spam, weight-loading and
#: download progress bars, and the bare ``k/n`` counters (kept separately as the
#: job's progress).
_NOISE = re.compile(r"\| DEBUG +\||Loading weights:|Downloading data:|Generating train split:"
                    r"|Loading checkpoint shards:|^\s*\d+/\d+\s*$")
_PROGRESS = re.compile(r"(?<![\d:\-/.])(\d+)/(\d+)(?![\d:\-/.])")
#: ``k/n`` counters that are not the job's progress: git checking out the pinned
#: worktree at the start of every attempt.
_NOT_PROGRESS = re.compile(r"Updating files:|Preparing worktree|Checking out files:")
_DRIFT = re.compile(r"stage \w+: running [0-9a-f]+, but the job was created with [0-9a-f]+")
_CODE = re.compile(r"^== code: .*")
_CAUSES = [
    re.compile(r"CANCELLED AT .* DUE TO TIME LIMIT"),
    re.compile(r"CANCELLED AT .* DUE TO PREEMPTION"),
    re.compile(r"oom-kill|Out Of Memory|OutOfMemory|Killed process|MemoryError"),
    re.compile(r"already completed — nothing to do"),
    re.compile(r"already \w+ — nothing to do"),
    re.compile(r"IllegalTransition"),
    re.compile(r"REFUSING"),
    re.compile(r"^\w*(Error|Exception)\b.*"),
    re.compile(r"\berror:"),
    re.compile(r"No such file or directory"),
    re.compile(r"set JOB_ID=|usage: "),
]

HEAD_BYTES = 64 * 1024
TAIL_BYTES = 64 * 1024


def read_log_edges(path: Path, head_bytes: int = HEAD_BYTES,
                   tail_bytes: int = TAIL_BYTES) -> Dict[str, Any]:
    """The first and last ``*_bytes`` of a log as lines, without reading the middle.

    A cropping log is 348 MB; everything a reader needs is at its edges: the code
    pin and stage lines at the start, the cause and the last progress at the end.
    """
    size = path.stat().st_size
    with path.open("rb") as fh:
        head = fh.read(head_bytes)
        if size > head_bytes + tail_bytes:
            fh.seek(size - tail_bytes)
            tail = fh.read(tail_bytes)
            truncated = True
        else:
            tail = head + fh.read()
            head = tail
            truncated = False
    head_lines = head.decode("utf-8", "replace").splitlines()
    tail_lines = tail.decode("utf-8", "replace").splitlines()
    if truncated:
        tail_lines = tail_lines[1:]  # the first line of a byte window is a fragment
    return {"size": size, "head": head_lines, "tail": tail_lines, "truncated": truncated}


def _strip_noise(lines: List[str]) -> List[str]:
    return [line for line in lines if not _NOISE.search(line)]


def last_progress(lines: List[str]) -> Optional[Dict[str, Any]]:
    """The last ``k/n`` counter in the lines, with the line it came from."""
    for line in reversed(lines):
        if _NOT_PROGRESS.search(line):
            continue
        found = _PROGRESS.findall(line)
        if not found:
            continue
        done, total = (int(x) for x in found[-1])
        if total > 1 and done <= total:
            return {"done": done, "total": total, "line": line.strip()[:200]}
    return None


def log_notes(head: List[str], tail: List[str]) -> Dict[str, Any]:
    """What a reader scans a log for: the code pin, commit drift, the cause.

    The cause is looked for in the tail first. A requeued job appends every
    attempt to the same file, so the head holds the preemption of attempt one and
    the tail holds why the LAST attempt ended; the last attempt is the one sacct
    reports.
    """
    lines = head + tail
    code = next((m.group(0) for line in lines for m in [_CODE.match(line)] if m), None)
    drift = sorted({m.group(0) for line in lines for m in [_DRIFT.search(line)] if m})
    cause = _cause_in(tail) or _cause_in(head)
    return {"code": code, "code_drift": drift, "cause": cause}


def _cause_in(lines: List[str]) -> Optional[str]:
    """The LAST line that looks like a cause: position wins over pattern, because
    a log holds every attempt and the final line says how the final one ended."""
    for line in reversed(lines):
        if any(pattern.search(line) for pattern in _CAUSES):
            return line.strip()[:300]
    return None


def find_log(paths: Paths, slurm_job_id: str) -> Optional[Path]:
    hits = sorted(paths.logs.glob("*-%s.out" % slurm_job_id))
    return hits[-1] if hits else None


# ── job records ─────────────────────────────────────────────────────────────
def _load_job(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def iter_jobs(paths: Paths):
    for job_json in sorted(paths.jobs_root.glob("*/job.json")):
        record = _load_job(job_json)
        if record is not None and isinstance(record.get("request"), dict):
            yield job_json.parent, record


def draw_of(job_dir: Path) -> Optional[Dict[str, Any]]:
    """Fingerprint of a job's evaluation draw: md5 and line count of ``val_eval.jsonl``.

    Two CER values are comparable only if their draws match (#108). The sample
    count is not evidence: the same 200 can be 200 different lines.
    """
    draw = job_dir / "data" / "val_eval.jsonl"
    if not draw.is_file():
        return None
    digest = hashlib.md5()
    lines = 0
    with draw.open("rb") as fh:
        for chunk in fh:
            digest.update(chunk)
            lines += 1
    return {"md5": digest.hexdigest(), "lines": lines}


def _params(record: Dict[str, Any]) -> Dict[str, Any]:
    return (record.get("request") or {}).get("params") or {}


def _slurm_ids(record: Dict[str, Any]) -> List[str]:
    text = " ".join(str(record.get(key) or "") for key in ("registration", "promotion_reason"))
    return sorted(set(re.findall(r"Slurm job (\d+)", text)))


def summarise_job(job_dir: Path, record: Dict[str, Any], with_draw: bool = True) -> Dict[str, Any]:
    request = record.get("request") or {}
    params = _params(record)
    metrics = record.get("metrics") or {}
    created = (record.get("code") or {}).get("commit")
    stages = []
    drift = []
    for stage in record.get("stages") or []:
        commit = (stage.get("code") or {}).get("commit")
        stages.append({
            "name": stage.get("name"), "status": stage.get("status"),
            "started_cest": utc_to_cest(stage.get("started_at")),
            "finished_cest": utc_to_cest(stage.get("finished_at")),
            "exit_code": stage.get("exit_code"), "commit": (commit or "")[:12] or None,
        })
        if commit and created and commit != created:
            drift.append("stage %s: running %s, but the job was created with %s"
                         % (stage.get("name"), commit[:12], created[:12]))
    return {
        "id": record.get("id") or job_dir.name,
        "status": record.get("status"),
        "stage": record.get("stage"),
        "model_id": request.get("model_id"),
        "engine": request.get("engine"),
        "base_model": request.get("base_model"),
        "granularity": params.get("granularity"),
        "load_in_4bit": params.get("load_in_4bit"),
        "epochs": params.get("epochs"),
        "lora_r": params.get("lora_r"),
        "created_cest": utc_to_cest(record.get("created_at")),
        "started_cest": utc_to_cest(record.get("started_at")),
        "finished_cest": utc_to_cest(record.get("finished_at")),
        "host": record.get("host"),
        "metrics": {k: metrics.get(k) for k in
                    ("cer", "wer", "length_ratio", "truncated_cer", "samples",
                     "benchmark_cer", "benchmark_samples")} if metrics else None,
        "error": record.get("error"),
        "promoted": record.get("promoted"),
        "registration": (record.get("registration") or "").split("\n")[0][:200] or None,
        "created_commit": (created or "")[:12] or None,
        "stages": stages,
        "code_drift": drift,
        "artefact": (record.get("progress") or {}).get("artefact"),
        "checkpoint_dir": record.get("checkpoint_dir"),
        "slurm_jobs": _slurm_ids(record),
        "draw": draw_of(job_dir) if with_draw else None,
    }


# ── the questions ───────────────────────────────────────────────────────────
SQUEUE_FIELDS = ["JobID", "Name", "Partition", "QOS", "State", "TimeUsed", "TimeLimit",
                 "StartTime", "Reason", "NodeList", "NumCPUs", "tres-alloc"]

REASONS = {
    "Priority": "waiting its turn; normal",
    "Resources": "waiting for free resources; normal",
    "MaxCpuRunMinsPerUser": "the QoS job_gratis caps CPUs x remaining walltime of the RUNNING "
                            "jobs at 11,520 CPU-minutes; this job starts when a running one ends",
    "ReqNodeNotAvail": "a node Slurm names is unavailable; check unavailable_nodes before "
                       "reading this as missing capacity",
    "QOSMaxGRESPerUser": "the QoS caps GPUs per user; this job starts when another releases one",
    "None": "running",
}


def _reason_key(reason: str) -> str:
    return (reason or "").split(",")[0].split(" ")[0].strip("()")


def node_states(run: Runner, nodes: List[str]) -> List[Dict[str, str]]:
    if not nodes:
        return []
    out = run(["sinfo", "-n", ",".join(nodes), "-h", "-N", "-o", "%N|%T|%E|%G"])
    seen = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) >= 3 and parts[0] not in seen:
            seen[parts[0]] = {"node": parts[0], "state": parts[1], "reason": parts[2],
                              "gres": parts[3] if len(parts) > 3 else ""}
    return list(seen.values())


def judge_wall(elapsed: Optional[int], limit: Optional[int],
               progress: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Will a running job finish inside its wall? Projected from its progress line."""
    if not elapsed or not limit or not progress or not progress.get("done"):
        return {"verdict": "unknown", "why": "no progress counter in the log"}
    projected = elapsed * progress["total"] / float(progress["done"])
    fraction = projected / float(limit)
    if fraction > 1.0:
        verdict = "will hit the wall"
    elif fraction > 1.0 - WALL_MARGIN:
        verdict = "at risk"
    else:
        verdict = "fits"
    return {"verdict": verdict, "projected_total_s": int(projected), "limit_s": limit,
            "fraction_of_wall": round(fraction, 3),
            "why": "%d/%d after %d s projects to %d s of %d" % (
                progress["done"], progress["total"], elapsed, projected, limit)}


def queue(run: Runner, paths: Paths) -> Dict[str, Any]:
    fmt = ",".join("%s:|" % f for f in SQUEUE_FIELDS)
    out = run(["squeue", "-u", paths.user, "-h", "-O", fmt])
    rows = []
    unavailable: List[str] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")]
        row = dict(zip(SQUEUE_FIELDS, parts))
        key = _reason_key(row.get("Reason", ""))
        entry = {
            "slurm_job_id": row.get("JobID"), "name": row.get("Name"),
            "partition": row.get("Partition"), "qos": row.get("QOS"),
            "state": row.get("State"), "elapsed": row.get("TimeUsed"),
            "limit": row.get("TimeLimit"), "start": row.get("StartTime"),
            "reason": row.get("Reason"), "reason_meaning": REASONS.get(key, ""),
            "nodes": row.get("NodeList"), "cpus": row.get("NumCPUs"),
            "tres": row.get("tres-alloc"),
        }
        if row.get("State") == "RUNNING":
            log = find_log(paths, row.get("JobID", ""))
            progress = None
            if log is not None:
                edges = read_log_edges(log)
                progress = last_progress(edges["tail"])
                entry["log"] = str(log)
            entry["progress"] = progress
            entry["wall"] = judge_wall(slurm_seconds(row.get("TimeUsed", "")),
                                       slurm_seconds(row.get("TimeLimit", "")), progress)
        match = re.search(r"UnavailableNodes:([\w,\-\[\]]+)", row.get("Reason", ""))
        if match:
            entry["unavailable_nodes"] = match.group(1).split(",")
            unavailable.extend(entry["unavailable_nodes"])
        rows.append(entry)
    states = node_states(run, sorted(set(unavailable))) if unavailable else []
    by_node = {s["node"]: s for s in states}
    for entry in rows:
        if "unavailable_nodes" in entry:
            entry["unavailable_nodes"] = [by_node.get(n, {"node": n}) for n in
                                          entry["unavailable_nodes"]]
    return {"user": paths.user, "jobs": rows,
            "running": sum(1 for r in rows if r["state"] == "RUNNING"),
            "pending": sum(1 for r in rows if r["state"] == "PENDING")}


SACCT_FIELDS = ["JobID", "JobName", "State", "Elapsed", "Timelimit", "Start", "End",
                "NodeList", "ExitCode", "NCPUS"]
BAD_STATES = ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "CANCELLED", "PREEMPTED")


def finished(run: Runner, paths: Paths, days: int = 2) -> Dict[str, Any]:
    out = run(["sacct", "-u", paths.user, "-S", "now-%ddays" % max(1, int(days)), "-X", "-P",
               "-o", ",".join(SACCT_FIELDS)])
    rows = []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < len(SACCT_FIELDS) or parts[0] == "JobID":
            continue
        row = dict(zip(SACCT_FIELDS, parts))
        state = row["State"].split(" ")[0]
        if state in ("RUNNING", "PENDING", "REQUEUED"):
            continue
        elapsed = slurm_seconds(row["Elapsed"]) or 0
        entry = {
            "slurm_job_id": row["JobID"], "name": row["JobName"], "state": row["State"],
            "elapsed": row["Elapsed"], "limit": row["Timelimit"], "start": row["Start"],
            "end": row["End"], "nodes": row["NodeList"], "exit_code": row["ExitCode"],
            "short_run": elapsed < SHORT_RUN_SECONDS,
        }
        log = find_log(paths, row["JobID"])
        if log is not None and (state in BAD_STATES or entry["short_run"]):
            edges = read_log_edges(log)
            notes = log_notes(edges["head"], edges["tail"])
            entry["log"] = str(log)
            entry["cause"] = notes["cause"]
            entry["code_drift"] = notes["code_drift"]
            entry["progress"] = last_progress(edges["tail"])
        if entry["short_run"]:
            entry["note"] = ("%d s is not a run: a requeue that found the record finished, "
                             "or a refusal before the first step; read cause" % elapsed)
        rows.append(entry)
    return {"since_days": days, "jobs": rows}


def job(paths: Paths, job_id: str) -> Dict[str, Any]:
    job_dir = paths.jobs_root / job_id
    record = _load_job(job_dir / "job.json")
    if record is None:
        return {"error": "no job.json under %s" % job_dir}
    return summarise_job(job_dir, record)


def results(paths: Paths, status: Optional[str] = None, granularity: Optional[str] = None,
            base_model: Optional[str] = None) -> Dict[str, Any]:
    rows = []
    for job_dir, record in iter_jobs(paths):
        summary = summarise_job(job_dir, record)
        if status and summary["status"] != status:
            continue
        if granularity and summary["granularity"] != granularity:
            continue
        if base_model and base_model.lower() not in (summary["base_model"] or "").lower():
            continue
        metrics = summary["metrics"] or {}
        draw = summary["draw"] or {}
        rows.append({
            "id": summary["id"], "status": summary["status"], "model_id": summary["model_id"],
            "base_model": summary["base_model"], "granularity": summary["granularity"],
            "load_in_4bit": summary["load_in_4bit"],
            "cer": metrics.get("cer"), "wer": metrics.get("wer"),
            "length_ratio": metrics.get("length_ratio"),
            "truncated_cer": metrics.get("truncated_cer"), "samples": metrics.get("samples"),
            "draw_md5": (draw.get("md5") or "")[:8] or None, "draw_lines": draw.get("lines"),
            "finished_cest": summary["finished_cest"],
            "code_drift": bool(summary["code_drift"]),
        })
    rows.sort(key=lambda r: (r["finished_cest"] or "", r["id"]))
    return {"jobs_root": str(paths.jobs_root), "rows": rows,
            "note": "two CER values compare only when draw_md5 matches (#108)"}


def draw(paths: Paths, job_id: str) -> Dict[str, Any]:
    own = draw_of(paths.jobs_root / job_id)
    if own is None:
        return {"job_id": job_id, "draw": None,
                "error": "no data/val_eval.jsonl: the test stage has not drawn yet"}
    shared = []
    for job_dir, record in iter_jobs(paths):
        if job_dir.name == job_id:
            continue
        other = draw_of(job_dir)
        if other and other["md5"] == own["md5"]:
            metrics = record.get("metrics") or {}
            shared.append({"id": job_dir.name, "status": record.get("status"),
                           "cer": metrics.get("cer")})
    return {"job_id": job_id, "draw": own, "shared_with": shared}


def _dir_mtime(path: Path) -> Optional[float]:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def prepared(paths: Paths) -> Dict[str, Any]:
    """Jobs whose corpus exists but whose GPU stage never finished."""
    completed_models = {}
    waiting = []
    for job_dir, record in iter_jobs(paths):
        model_id = (record.get("request") or {}).get("model_id")
        if record.get("status") == "completed":
            completed_models.setdefault(model_id, []).append(job_dir.name)
        elif record.get("status") in ("preparing", "training", "queued"):
            waiting.append((job_dir, record))
    rows = []
    for job_dir, record in waiting:
        summary = summarise_job(job_dir, record, with_draw=False)
        mtime = _dir_mtime(job_dir / "data") or _dir_mtime(job_dir)
        rows.append({
            "id": summary["id"], "status": summary["status"], "model_id": summary["model_id"],
            "base_model": summary["base_model"], "granularity": summary["granularity"],
            "load_in_4bit": summary["load_in_4bit"], "created_cest": summary["created_cest"],
            "data_touched_cest": epoch_to_cest(mtime),
            "scratch_purge_cest": epoch_to_cest(mtime + SCRATCH_PURGE.total_seconds())
            if mtime else None,
            "artefact": summary["artefact"],
            "superseded_by": completed_models.get(summary["model_id"], []),
        })
    return {"rows": rows,
            "note": "superseded_by lists completed jobs of the same model_id: a leftover, "
                    "not a next step"}


def deadlines(run: Runner, paths: Paths) -> Dict[str, Any]:
    now = datetime.now(tz=ZURICH)
    statuses = {job_dir.name: record.get("status") for job_dir, record in iter_jobs(paths)}
    waiting = prepared(paths)["rows"]
    superseded = {row["id"]: row["superseded_by"] for row in waiting if row["superseded_by"]}
    artefacts = []
    for meta in sorted(paths.artefacts_root.glob("*/artefact.json")):
        record = _load_job(meta)
        if record is None:
            continue
        built = float(record.get("built_at") or 0)
        pinned = bool(record.get("pinned"))
        expires = None if pinned or not built else datetime.fromtimestamp(built, tz=ZURICH) \
            + ARTEFACT_TTL
        claims = {k: epoch_to_cest(v) for k, v in (record.get("claims") or {}).items()}
        claimants = {k: statuses.get(k, "no record") for k in claims}
        # A claimant that a completed job of the same model_id superseded is a
        # leftover record, not a job waiting for this corpus.
        live = [k for k, s in claimants.items()
                if s in ("preparing", "training", "queued") and k not in superseded]
        artefacts.append({
            "key": (record.get("key") or meta.parent.name)[:12], "pinned": pinned,
            "built_cest": epoch_to_cest(built or None),
            "expires_cest": _label(expires) if expires else None,
            "expires_in_days": round((expires - now).total_seconds() / 86400, 1)
            if expires else None,
            "gib": round(float(record.get("bytes") or 0) / 2 ** 30, 1),
            "job_id": record.get("job_id"), "claims": claims, "claimant_status": claimants,
            "claimant_superseded_by": {k: superseded[k] for k in claims if k in superseded},
            "last_used_cest": epoch_to_cest(record.get("last_used")),
            "at_risk": bool(expires and live and (expires - now) < timedelta(days=2)),
        })
    purge = []
    for row in waiting:
        if row["scratch_purge_cest"]:
            purge.append({"id": row["id"], "status": row["status"],
                          "scratch_purge_cest": row["scratch_purge_cest"],
                          "superseded_by": row["superseded_by"]})
    disk = None
    try:
        line = run(["df", "-P", str(paths.home)]).splitlines()[-1].split()
        disk = {"filesystem": line[0], "use_percent": line[4],
                "avail_gib": round(int(line[3]) / 2 ** 20, 1), "mounted_on": line[5]}
    except (ProbeError, IndexError, ValueError):
        pass
    return {"now_cest": _label(now), "artefacts": artefacts, "scratch_purge": purge,
            "home": disk,
            "note": "at_risk: an unpinned artefact within 2 days of expiry whose claimant "
                    "is still preparing/training/queued and not superseded"}


def log(paths: Paths, slurm_job_id: str, lines: int = 40) -> Dict[str, Any]:
    path = find_log(paths, slurm_job_id)
    if path is None:
        return {"error": "no %s/*-%s.out" % (paths.logs, slurm_job_id)}
    edges = read_log_edges(path)
    head = _strip_noise(edges["head"])[:lines]
    tail = _strip_noise(edges["tail"])[-lines:]
    notes = log_notes(edges["head"], edges["tail"])
    notes["progress"] = last_progress(edges["tail"])
    # A log that is nothing but progress bars at its end filters to an empty tail;
    # the last raw lines still say where the job is.
    last_raw = [line.strip()[:200] for line in edges["tail"][-3:] if line.strip()]
    return {"log": str(path), "size_bytes": edges["size"],
            "modified_cest": epoch_to_cest(path.stat().st_mtime),
            "truncated_middle": edges["truncated"], "head": head, "tail": tail,
            "last_raw": last_raw, "notes": notes}


REPORT_SCALARS = ("cer", "wer", "length_ratio", "truncated_cer", "samples", "truncated_at_cap",
                  "load_in_4bit", "base_model", "granularity", "max_new_tokens", "max_pixels",
                  "eval_selection")


def report(paths: Paths, evalset: str = "evalset-federal-minutes",
           tag: Optional[str] = None) -> Dict[str, Any]:
    root = paths.jobs_root / evalset
    if not root.is_dir():
        return {"error": "no %s" % root}
    if tag:
        path = root / ("report_%s.json" % tag)
        record = _load_job(path)
        if record is None:
            return {"error": "no %s" % path}
        return {"evalset": evalset, "tag": tag, "modified_cest": epoch_to_cest(path.stat().st_mtime),
                "report": {k: v for k, v in record.items() if not isinstance(v, (list, dict))}}
    rows = []
    for path in sorted(root.glob("report_*.json")):
        record = _load_job(path)
        if record is None:
            continue
        row = {k: record.get(k) for k in REPORT_SCALARS}
        row["tag"] = path.stem[len("report_"):]
        row["adapter"] = Path(record.get("adapter") or "").name or None
        row["modified_cest"] = epoch_to_cest(path.stat().st_mtime)
        rows.append(row)
    rows.sort(key=lambda r: (r["cer"] is None, r["cer"] or 0))
    return {"evalset": evalset, "rows": rows,
            "note": "same val.jsonl for every row; quantisation (load_in_4bit) may differ"}


SCONTROL_KEEP = ("JobId", "JobName", "JobState", "Reason", "Dependency", "Partition", "QOS",
                 "RunTime", "TimeLimit", "SubmitTime", "StartTime", "EndTime", "NodeList",
                 "ReqTRES", "AllocTRES", "Command", "WorkDir", "StdOut", "NumCPUs", "Restarts",
                 "Requeue", "ExitCode", "BatchHost")


def slurm_job(run: Runner, paths: Paths, slurm_job_id: str) -> Dict[str, Any]:
    try:
        out = run(["scontrol", "show", "job", str(slurm_job_id)])
    except ProbeError as exc:
        if "Invalid job id" in str(exc):
            gone = finished(run, paths, days=30)
            match = [j for j in gone["jobs"] if j["slurm_job_id"] == str(slurm_job_id)]
            return {"slurm_job_id": str(slurm_job_id), "error": "not in scontrol any more",
                    "accounting": match[0] if match else None}
        raise
    fields = dict(re.findall(r"(\w+)=(\S*)", out))
    entry = {k: fields.get(k) for k in SCONTROL_KEEP if k in fields}
    entry["slurm_job_id"] = str(slurm_job_id)
    entry["reason_meaning"] = REASONS.get(_reason_key(fields.get("Reason", "")), "")
    match = re.search(r"UnavailableNodes:([\w,\-\[\]]+)", fields.get("Reason", ""))
    if match:
        entry["unavailable_nodes"] = node_states(run, match.group(1).split(","))
    log_path = find_log(paths, str(slurm_job_id))
    if log_path is not None:
        edges = read_log_edges(log_path)
        entry["log"] = str(log_path)
        entry["notes"] = log_notes(edges["head"], edges["tail"])
        entry["progress"] = last_progress(edges["tail"])
    return entry


def checkout(run: Runner, paths: Paths) -> Dict[str, Any]:
    repo = str(paths.checkout)
    git = ["git", "-C", repo]
    head = run(git + ["rev-parse", "HEAD"]).strip()
    subject = run(git + ["log", "-1", "--format=%h %s"]).strip()
    dirty = [line for line in run(git + ["status", "--porcelain"]).splitlines() if line.strip()]
    fetched = True
    try:
        run(git + ["fetch", "-q", "origin"])
    except ProbeError:
        fetched = False
    counts = run(git + ["rev-list", "--left-right", "--count", "HEAD...origin/main"]).split()
    ahead, behind = (int(counts[0]), int(counts[1])) if len(counts) == 2 else (None, None)
    origin = run(git + ["log", "-1", "--format=%h %s", "origin/main"]).strip()
    return {"checkout": repo, "head": head[:12], "head_subject": subject,
            "origin_main": origin, "fetched": fetched, "ahead": ahead, "behind": behind,
            "dirty_files": len(dirty),
            "note": "submit.sh refuses a checkout that is behind origin/main or dirty"}


# ── entry point ─────────────────────────────────────────────────────────────
def dispatch(cmd: str, args: Dict[str, Any], run: Runner, paths: Paths) -> Dict[str, Any]:
    if cmd == "queue":
        return queue(run, paths)
    if cmd == "finished":
        return finished(run, paths, days=int(args.get("days", 2)))
    if cmd == "job":
        return job(paths, str(args["job_id"]))
    if cmd == "results":
        return results(paths, status=args.get("status"), granularity=args.get("granularity"),
                       base_model=args.get("base_model"))
    if cmd == "draw":
        return draw(paths, str(args["job_id"]))
    if cmd == "prepared":
        return prepared(paths)
    if cmd == "deadlines":
        return deadlines(run, paths)
    if cmd == "log":
        return log(paths, str(args["slurm_job_id"]), lines=int(args.get("lines", 40)))
    if cmd == "report":
        return report(paths, evalset=args.get("evalset") or "evalset-federal-minutes",
                      tag=args.get("tag"))
    if cmd == "slurm_job":
        return slurm_job(run, paths, str(args["slurm_job_id"]))
    if cmd == "checkout":
        return checkout(run, paths)
    raise ProbeError("unknown command %r" % cmd)


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        sys.stderr.write("usage: python3 - <cmd> [json-args]\n")
        return 2
    cmd = argv[1]
    args = json.loads(argv[2]) if len(argv) > 2 and argv[2].strip() else {}
    try:
        result = dispatch(cmd, args, shell_runner, Paths())
    except ProbeError as exc:
        json.dump({"error": str(exc), "cmd": cmd}, sys.stdout)
        return 1
    result.setdefault("_host", os.uname().nodename)
    json.dump(result, sys.stdout, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

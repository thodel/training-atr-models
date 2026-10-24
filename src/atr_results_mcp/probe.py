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

**Two job stores.** UBELIX runs keep their records under ``/scratch``; the jobs
that the ``atr-train`` service on asteraix accepts keep theirs in the shared
store on the research share, which the login node mounts under
``/storage/research``. The record-reading questions (``results``, ``job``,
``draw``, ``prepared``, ``live``) read both and say on each row which ``host``
ran it. #156 said asteraix had no job store; it looked under ``~/atr-cache``,
which holds only hand-run measurements. The Slurm questions (``queue``,
``finished``, ``slurm_job``, ``log``) stay UBELIX's: asteraix has no Slurm, and
its live state is what ``live`` reads from the record and the stage log.

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
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from zoneinfo import ZoneInfo

ZURICH = ZoneInfo("Europe/Zurich")

#: The research share as the UBELIX login node mounts it; asteraix and idhefix
#: mount the same filesystem under ``/mnt/wbkolleg_dh_1``.
SHARE_ON_UBELIX = "/storage/research/wbkolleg_dh_1/Textrecognition_Training"

#: ``host`` a shared-store record without one belongs to: the retired trainer on
#: idhefix wrote 48 such records before #15 (``ATR_TRAIN_LEGACY_JOB_HOST``).
LEGACY_HOST = "idhefix"
#: Every record under ``/scratch`` was written by ``submit_job.py`` or
#: ``fanout.py``, which stamp ``host: ubelix``; an older one without the stamp is
#: still a Slurm job.
SCRATCH_HOST = "ubelix"

#: A record in one of these statuses has a runner, or waits for one.
LIVE_STATUSES = ("queued", "preparing", "compiling", "training", "testing", "registering")

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
                 scratch: Optional[Path] = None, share: Optional[Path] = None) -> None:
        self.user = user or os.environ.get("USER") or "th19c587"
        self.home = Path(home or os.environ.get("HOME") or ("/storage/homefs/" + self.user))
        self.scratch = Path(scratch or ("/scratch/network/users/" + self.user))
        self.share = Path(share or os.environ.get("ATR_RESULTS_SHARE") or SHARE_ON_UBELIX)

    @property
    def jobs_root(self) -> Path:
        """The UBELIX job store: what every ``*.sbatch`` sets ``ATR_TRAIN_JOBS_ROOT`` to."""
        return self.scratch / "runs" / "jobs"

    @property
    def shared_jobs_root(self) -> Path:
        """The shared job store: ``ATR_TRAIN_JOBS_ROOT`` of the service on asteraix."""
        return self.share / "training_folder" / "jobs"

    @property
    def job_roots(self) -> List[Tuple[str, Path]]:
        """``(store, root)`` for every job store this probe reads, scratch first."""
        return [("scratch", self.jobs_root), ("share", self.shared_jobs_root)]

    @property
    def artefacts_root(self) -> Path:
        return self.scratch / "expA" / "artefacts"

    @property
    def logs(self) -> Path:
        return self.home / "ubelix" / "logs"

    @property
    def checkout(self) -> Path:
        return self.home / "training-atr-models"

    @property
    def storage_script(self) -> Path:
        """Das Prüfskript für den research-storage, wie es ausgerollt ist.

        Nicht im Checkout, sondern unter ``~/ubelix/``: der Checkout ist für
        Trainingsjobs an einen Commit geheftet, und eine tägliche Prüfung darf
        nicht davon abhängen, auf welchem Stand er gerade steht.
        """
        return self.home / "ubelix" / "research_storage.py"


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
#: ``k/n`` counters that are not the job's progress. This is a list of KNOWN
#: false sources, and it grows: git checking out the pinned worktree at the start
#: of every attempt; the retry counter of a hub request that timed out (read as
#: "2/5, fits" on 09.10.2026, #203); the shard counter of a model load (the last
#: "progress" of a job that died in OOM while loading, 17611939).
_NOT_PROGRESS = re.compile(r"Updating files:|Preparing worktree|Checking out files:"
                           r"|Retry \d+/\d+|Retrying in|Loading weights:"
                           r"|Loading checkpoint shards:")
#: The attempt line every ``*.sbatch`` prints: names the record a Slurm job runs,
#: which is how the Slurm log leads to ``<jobs_root>/<job_id>/logs/<stage>.log``.
_ATTEMPT = re.compile(r"== attempt (\d+) — job (\S+) \((\w+)\)")
#: The image line ``train.sbatch`` prints: the container the attempt ran in,
#: which a resubmission has to repeat (#206, #226).
_SIF = re.compile(r"^== SIF: (\S+)")
#: tqdm's suffix ``[elapsed<remaining, rate]``; the remaining time is computed
#: from the rate of the running attempt and is the number to trust after a resume.
_TQDM = re.compile(r"\[(\d+(?::\d+){1,2})<(\d+(?::\d+){1,2}),\s*([\d.]+)\s*(s/it|it/s)\]")
_DRIFT = re.compile(r"stage \w+: running [0-9a-f]+, but the job was created with [0-9a-f]+")
_CODE = re.compile(r"^== code: .*")
_CAUSES = [
    re.compile(r"CANCELLED AT .* DUE TO TIME LIMIT"),
    re.compile(r"CANCELLED AT .* DUE TO PREEMPTION"),
    re.compile(r"oom-kill|Out Of Memory|OutOfMemory|Killed process|MemoryError"),
    re.compile(r"already completed — nothing to do"),
    re.compile(r"already \w+ — nothing to do"),
    re.compile(r"re-entered in terminal state"),  # #153: a requeue after the run ended
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


def last_progress(lines: List[str], source: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The last ``k/n`` counter in the lines, in one of three states (#203).

    ``None``: no counter at all. ``state: found``: a counter that is the job's
    progress, with ``done``/``total`` and the line — the only state a wall
    projection may use. ``state: unattributable``: the only counters are from a
    known false source (a retry count, a shard load, the worktree checkout);
    the line is returned so a reader can see what was there, and ``done`` is
    deliberately absent. "Found something that looks like it" is not "measured".
    ``source`` names the log the counter came from.
    """
    rejected = None
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index]
        found = _PROGRESS.findall(line)
        if not found:
            continue
        done, total = (int(x) for x in found[-1])
        if not (total > 1 and done <= total):
            continue
        if _NOT_PROGRESS.search(line):
            if rejected is None:
                match = _NOT_PROGRESS.search(line)
                rejected = {"state": "unattributable", "line": line.strip()[:200],
                            "source": source,
                            "why": "%d/%d is a known non-progress counter (%s)"
                                   % (done, total, match.group(0).strip(" :"))}
            continue
        found_progress = {"state": "found", "done": done, "total": total,
                          "line": line.strip()[:200], "source": source}
        # tqdm's own estimate, from the rate of THIS attempt: after a resume
        # `done` counts earlier attempts' steps and the ratio lies (#216). The
        # last line of a log being written is often cut mid-suffix
        # (`11830/13704 [17:52:52<2:46:18,`), so the previous complete line of
        # the same bar is allowed to supply it: one step stale, seconds off.
        for candidate in lines[max(0, index - 3):index + 1][::-1]:
            tqdm = _TQDM.search(candidate)
            if not tqdm:
                continue
            counter = _PROGRESS.findall(candidate)
            if counter and int(counter[-1][1]) == total:
                found_progress["remaining_s"] = slurm_seconds(tqdm.group(2))
                found_progress["rate"] = tqdm.group(3) + tqdm.group(4)
                break
        return found_progress
    return rejected


#: The stages one ``*.sbatch`` runs in one Slurm job: ``prepare.sbatch`` the
#: first group, ``train.sbatch`` the second. A counter from ``train`` has ``test``
#: and ``register`` still ahead of it inside the same wall.
STAGE_GROUPS = (("prepare", "compile"), ("train", "test", "register"))


def following_stages(stage: Optional[str]) -> List[str]:
    for group in STAGE_GROUPS:
        if stage in group:
            return list(group[group.index(stage) + 1:])
    return []


def _stage_seconds(record: Dict[str, Any]) -> Dict[str, int]:
    """Measured duration of every completed stage in a record, by name."""
    seconds = {}
    for stage in record.get("stages") or []:
        if stage.get("status") != "completed":
            continue
        try:
            started = datetime.fromisoformat((stage.get("started_at") or "").replace("Z", "+00:00"))
            finished = datetime.fromisoformat((stage.get("finished_at") or "").replace("Z", "+00:00"))
        except ValueError:
            continue
        seconds[stage.get("name")] = int((finished - started).total_seconds())
    return seconds


def reference_for(paths: "Paths", record: Dict[str, Any], stages: List[str]
                  ) -> Optional[Dict[str, Any]]:
    """How long ``stages`` took for a comparable completed job: measured from its
    ``stages[]``, never estimated (#216).

    Comparable means the same granularity in the same store, and the same base
    model when such a job exists; the most recently finished one wins. The
    answer names the record it came from, so the number is checkable.
    """
    granularity = _params(record).get("granularity")
    base_model = (record.get("request") or {}).get("base_model")
    best = None
    for other_dir, other in iter_jobs(paths, stores=[record.get("_store") or "scratch"]):
        if other.get("status") != "completed" or other_dir.name == record.get("id"):
            continue
        if _params(other).get("granularity") != granularity:
            continue
        measured = _stage_seconds(other)
        if not all(name in measured for name in stages):
            continue
        same_model = (other.get("request") or {}).get("base_model") == base_model
        key = (same_model, other.get("finished_at") or "")
        if best is None or key > best[0]:
            best = (key, other_dir.name, {name: measured[name] for name in stages})
    if best is None:
        return None
    return {"from": best[1], "match": "base_model+granularity" if best[0][0] else "granularity",
            "seconds": best[2]}


def _attempt_in(lines: List[str]) -> Optional[Tuple[str, str, int]]:
    """``(job_id, status, attempt)`` of the LAST attempt line: a requeued job appends."""
    for line in reversed(lines):
        match = _ATTEMPT.search(line)
        if match:
            return match.group(2), match.group(3), int(match.group(1))
    return None


def _sif_in(lines: List[str]) -> Optional[str]:
    """The container of the LAST attempt in the lines, from ``== SIF: <path> (…)``."""
    for line in reversed(lines):
        match = _SIF.match(line)
        if match:
            return match.group(1)
    return None


def slurm_progress(paths: "Paths", edges: Dict[str, Any]) -> Dict[str, Any]:
    """Progress of a Slurm job: from its stdout, else from the stage log.

    A VLM run writes its step counter to ``<jobs_root>/<job_id>/logs/train.log``
    and nothing to the Slurm stdout after the attempt line; read only there, a
    healthy run at 100 % GPU looked like "no progress counter" (#203). The
    attempt line in the stdout names the record, the record names its stage,
    and :func:`_stage_log` names the file. The stdout wins when it carries a
    counter of its own (cropping and dataset jobs write theirs there).
    """
    progress = last_progress(edges["tail"], source="slurm")
    answer: Dict[str, Any] = {"progress": progress}
    attempt = _attempt_in(edges["tail"]) or _attempt_in(edges["head"])
    if attempt is None:
        return answer
    job_id = attempt[0]
    answer["job_id"] = job_id
    # ``SLURM_RESTART_COUNT + 1``: a requeue appends a new attempt line to the
    # same log, so "attempt 2 running" is readable instead of guessed (#226).
    answer["attempt"] = attempt[2]
    job_dir, record = find_job(paths, job_id)
    if job_dir is None or record is None:
        return answer
    stage = record.get("stage")
    if progress is None or progress["state"] != "found":
        stage_log = _stage_log(job_dir, stage)
        if stage_log is not None:
            answer["stage_log"] = str(stage_log)
            answer["stage_log_modified_cest"] = epoch_to_cest(stage_log.stat().st_mtime)
            from_stage = last_progress(read_log_edges(stage_log)["tail"], source=stage_log.name)
            if from_stage is not None and (progress is None or from_stage["state"] == "found"):
                answer["progress"] = progress = from_stage
    if progress is not None and progress["state"] == "found" and stage:
        progress["stage"] = stage
        ahead = following_stages(stage)
        if ahead:
            answer["following"] = reference_for(paths, record, ahead) or {
                "from": None, "seconds": None,
                "note": "no completed job of the same granularity measures %s" % ", ".join(ahead)}
            answer["following"]["stages"] = ahead
    return answer


def log_notes(head: List[str], tail: List[str]) -> Dict[str, Any]:
    """What a reader scans a log for: the code pin, commit drift, the cause.

    A requeued job appends every attempt to the same file, so the log holds
    the preemption of attempt one *before* the attempt line of attempt two.
    ``cause`` is therefore the last cause line **after the last attempt line**
    — how the attempt sacct reports ended, or None while it is still running —
    and a cause found before it is ``previous_attempt_cause``, with the number
    of the attempt it belongs to. Taking the last cause-looking line anywhere
    hung "CANCELLED … DUE to SIGNAL Terminated" of attempt 1 on a running
    attempt 2 (17890142, 12.10.2026, #231). Lines before the first attempt line
    (a refusal in the batch script's preamble) belong to that first attempt.
    """
    lines = head + tail
    code = next((m.group(0) for line in lines for m in [_CODE.match(line)] if m), None)
    drift = sorted({m.group(0) for line in lines for m in [_DRIFT.search(line)] if m})
    causes = _causes_by_attempt(lines)
    last = _attempt_in(lines)
    previous = None
    if last is None:
        cause = causes[-1][1] if causes else None
    else:
        own = [line for attempt, line in causes if attempt == last[2]]
        earlier = [(attempt, line) for attempt, line in causes if attempt < last[2]]
        cause = own[-1] if own else None
        if earlier:
            previous = {"attempt": earlier[-1][0], "cause": earlier[-1][1]}
    return {"code": code, "code_drift": drift, "cause": cause,
            "previous_attempt_cause": previous}


def _causes_by_attempt(lines: List[str]) -> List[Tuple[int, str]]:
    """Every cause-looking line with the attempt it followed, in log order.

    Lines before the first attempt line get that first attempt's number: the
    batch script prints its refusals (``set JOB_ID=``, ``usage:``) before it
    prints the attempt line, and they are that attempt's cause, not an earlier
    one's. A log without any attempt line numbers everything 0.
    """
    first = next((int(m.group(1)) for line in lines for m in [_ATTEMPT.search(line)] if m), 0)
    current = first
    out: List[Tuple[int, str]] = []
    for line in lines:
        match = _ATTEMPT.search(line)
        if match:
            current = int(match.group(1))
            continue
        if any(pattern.search(line) for pattern in _CAUSES):
            out.append((current, line.strip()[:300]))
    return out


def find_log(paths: Paths, slurm_job_id: str) -> Optional[Path]:
    hits = sorted(paths.logs.glob("*-%s.out" % slurm_job_id))
    return hits[-1] if hits else None


# ── job records ─────────────────────────────────────────────────────────────
def _load_job(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _host_of(record: Dict[str, Any], store: str) -> str:
    """The host a record belongs to, with the two stores' defaults for a missing stamp."""
    return record.get("host") or (LEGACY_HOST if store == "share" else SCRATCH_HOST)


def iter_jobs(paths: Paths, stores: Optional[List[str]] = None
              ) -> Iterator[Tuple[Path, Dict[str, Any]]]:
    """Every readable record in every store, ``host`` and ``_store`` filled in.

    A store whose root does not exist yields nothing here; :func:`store_states`
    is where "not mounted" is said, so that an empty answer is never mistaken for
    an empty store (#165).
    """
    for store, root in paths.job_roots:
        if stores and store not in stores:
            continue
        if not root.is_dir():
            continue
        for job_json in sorted(root.glob("*/job.json")):
            record = _load_job(job_json)
            if record is not None and isinstance(record.get("request"), dict):
                record["host"] = _host_of(record, store)
                record["_store"] = store
                yield job_json.parent, record


def find_job(paths: Paths, job_id: str) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """The directory and record of ``job_id`` in whichever store holds it."""
    for store, root in paths.job_roots:
        record = _load_job(root / job_id / "job.json")
        if record is not None:
            record["host"] = _host_of(record, store)
            record["_store"] = store
            return root / job_id, record
    return None, None


def store_states(paths: Paths) -> Dict[str, Dict[str, Any]]:
    """Per store: where it is, whether it is there, and how many records it holds.

    The share is a mount; when it is away, its store reads as absent, and the
    answer must say so rather than show zero asteraix rows.
    """
    states = {}
    for store, root in paths.job_roots:
        if root.is_dir():
            states[store] = {"root": str(root), "mounted": True,
                             "jobs": sum(1 for _ in root.glob("*/job.json"))}
        else:
            states[store] = {"root": str(root), "mounted": False, "jobs": None,
                             "note": "not found on this host: its records are not in this answer"}
    return states


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


#: The stages whose runtime decides what a CER means: the test stage measured
#: it, the train stage made the weights it measured. In that order of relevance.
RUNTIME_STAGES = ("test", "train")


def runtime_of(record: Dict[str, Any]) -> Dict[str, Any]:
    """Which image and transformers produced this record's number (#206).

    Three answers, never two. ``known`` is True when a GPU stage carries the
    ``runtime`` field the runner writes since #206; then ``image`` is the
    image's file name or None for **no container** (asteraix runs its venvs
    bare — a measured fact). ``known`` is False on every record from before the
    field, with ``image`` and ``transformers`` None and ``why`` saying so: that
    is "nobody recorded it", which must not read like "no container".
    """
    for name in RUNTIME_STAGES:
        for stage in record.get("stages") or []:
            if stage.get("name") != name or "runtime" not in stage:
                continue
            runtime = stage.get("runtime") or {}
            image = runtime.get("image")
            return {"known": True, "stage": name,
                    "image": Path(image).name if image else None,
                    "transformers": runtime.get("transformers"),
                    "text": "%s / transformers %s" % (Path(image).name if image else "no container",
                                                      runtime.get("transformers") or "none")}
    return {"known": False, "stage": None, "image": None, "transformers": None,
            "text": "unknown", "why": "no GPU stage of this record carries a runtime: "
                                     "recorded before #206, or the stage never ran"}


def summarise_job(job_dir: Path, record: Dict[str, Any], with_draw: bool = True) -> Dict[str, Any]:
    request = record.get("request") or {}
    params = _params(record)
    metrics = record.get("metrics") or {}
    created = (record.get("code") or {}).get("commit")
    stages = []
    drift = []
    for stage in record.get("stages") or []:
        commit = (stage.get("code") or {}).get("commit")
        runtime = stage.get("runtime") if "runtime" in stage else None
        stages.append({
            "name": stage.get("name"), "status": stage.get("status"),
            "started_cest": utc_to_cest(stage.get("started_at")),
            "finished_cest": utc_to_cest(stage.get("finished_at")),
            "exit_code": stage.get("exit_code"), "commit": (commit or "")[:12] or None,
            # Absent (not None) when the record predates the field: a reader
            # that sees the key missing knows nobody looked.
            **({"image": Path(runtime["image"]).name if runtime.get("image") else None,
                "transformers": runtime.get("transformers")} if runtime is not None else {}),
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
        # Dieselbe Frage wie der Host, eine Ebene tiefer: zwei Abbilder, eine
        # transformers-Hauptversion Unterschied, und bis #206 stand in keinem
        # Datensatz, welches den Wert erzeugt hat.
        "runtime": runtime_of(record),
        # Der Held-out-Zustand ist die zweite Bedingung der Vergleichbarkeit neben
        # der Ziehung, und er lag schon im Datensatz, ohne je gelesen zu werden:
        # das September-Korpus der medieval-Leiter hatte `reserved_pages: 0`, das
        # Oktober-Korpus 3173 — 28 % weniger Trainingszeilen bei identischer
        # Spezifikation, drei Wochen unbemerkt (#98, gefunden 14.10.2026).
        "heldout": _heldout_of(record),
        "store": record.get("_store"),
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
    "QOSGrpGRES": "the QoS caps GPUs of this type over ALL its users' running jobs (job_gratis "
                  "shares a few rtx4090 and h100); this job starts when any job of the QoS "
                  "releases one — seen 12.10.2026 on two rescore jobs",
    "DependencyNeverSatisfied": "the job this one depends on (--dependency=afterok) ended "
                                "without success; Slurm will never start it — scancel it "
                                "and resubmit after the dependency is fixed (seen 12.10.2026 "
                                "on a prep/train chain after a 1 h prep TIMEOUT)",
    "None": "running",
}


def _reason_key(reason: str) -> str:
    return (reason or "").split(",")[0].split(" ")[0].strip("()")


def node_states(run: Runner, nodes: List[str]) -> List[Dict[str, str]]:
    """The state of each named node from sinfo — or ``unknown`` with the reason
    when sinfo could not be asked (#165): the queue answer is still the queue,
    and a node nobody could look at is not a node without a problem."""
    if not nodes:
        return []
    try:
        out = run(["sinfo", "-n", ",".join(nodes), "-h", "-N", "-o", "%N|%T|%E|%G"])
    except ProbeError as exc:
        return [{"node": n, "state": "unknown", "reason": "", "gres": "",
                 "why": "sinfo could not be asked: %s" % exc} for n in nodes]
    seen = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) >= 3 and parts[0] not in seen:
            seen[parts[0]] = {"node": parts[0], "state": parts[1], "reason": parts[2],
                              "gres": parts[3] if len(parts) > 3 else ""}
    for node in nodes:
        if node not in seen:
            seen[node] = {"node": node, "state": "unknown", "reason": "", "gres": "",
                          "why": "sinfo answered, but not for this node"}
    return list(seen.values())


def judge_wall(elapsed: Optional[int], limit: Optional[int],
               progress: Optional[Dict[str, Any]],
               following: Optional[Dict[str, Any]] = None,
               attempt: Optional[int] = None) -> Dict[str, Any]:
    """Will a running job finish inside its wall? The sum is written out in ``why``.

    Only a counter in state ``found`` projects. An unattributable one says so in
    ``why`` — "unknown" invites a look, a made-up "fits" does not (#203).

    The rest of the running stage comes from tqdm's own estimate when the line
    carries one (``remaining_s``), else from the ratio ``elapsed × total / done``.
    After a resume the ratio lies: ``done`` counts earlier attempts' steps and
    ``elapsed`` only this one (#216). When the line has no tqdm estimate and
    *attempt* says this is a resume, the ratio is therefore a **lower bound**:
    the sum is still written out, ``lower_bound`` says so, and a "fits" from it
    becomes "at risk" — a floor cannot prove a fit, and the direction of the
    error is the one #203 already failed in. The stages still ahead in the same
    wall (``test`` after ``train``) are added from a measured reference; without
    one the verdict says the sum is short of them, and a close call is not
    "fits".
    """
    if progress and progress.get("state") == "unattributable":
        return {"verdict": "unknown",
                "why": "the last counter is not progress: %s" % progress.get("why")}
    if not elapsed or not limit or not progress or not progress.get("done"):
        return {"verdict": "unknown", "why": "no progress counter in the log"}
    stage = progress.get("stage") or "stage"
    if progress.get("remaining_s") is not None:
        rest = progress["remaining_s"]
        method = "tqdm"
    else:
        rest = elapsed * (progress["total"] - progress["done"]) / float(progress["done"])
        method = "ratio"
    parts = ["%s: %d s + rest %d s (%s %d/%d)" % (stage, elapsed, rest, method,
                                                 progress["done"], progress["total"])]
    projected = elapsed + rest
    uncounted = []
    if following and following.get("stages"):
        if following.get("seconds"):
            for name in following["stages"]:
                projected += following["seconds"][name]
                parts.append("%s ≈ %d s" % (name, following["seconds"][name]))
            parts[-1] += " (from %s)" % following["from"]
        else:
            uncounted = list(following["stages"])
    fraction = projected / float(limit)
    lower_bound = method == "ratio" and attempt is not None and attempt > 1
    if fraction > 1.0:
        verdict = "will hit the wall"
    elif fraction > 1.0 - WALL_MARGIN:
        verdict = "at risk"
    elif uncounted and fraction > 1.0 - 2 * WALL_MARGIN:
        verdict = "at risk"
    elif lower_bound:
        verdict = "at risk"
    else:
        verdict = "fits"
    why = " + ".join(parts) + " = %d s of %d" % (projected, limit)
    if uncounted:
        why += "; %s not counted: no completed job of this granularity to measure it" \
               % ", ".join(uncounted)
    if lower_bound:
        why += ("; a lower bound: attempt %d resumed from a checkpoint, so %d/%d counts "
                "earlier attempts' steps while %d s is this attempt alone, and the line "
                "carries no tqdm estimate — not a fit until a tqdm line says so"
                % (attempt, progress["done"], progress["total"], elapsed))
    return {"verdict": verdict, "projected_total_s": int(projected), "limit_s": limit,
            "fraction_of_wall": round(fraction, 3), "method": method,
            "lower_bound": lower_bound, "uncounted_stages": uncounted, "why": why}


_SBATCH_OPTION = re.compile(r"^#SBATCH\s+--([\w-]+)(?:=(\S+))?")


def batch_file_for(paths: Paths, job_name: str) -> Optional[Path]:
    """The ``ubelix/*.sbatch`` whose ``#SBATCH --job-name`` is ``job_name``.

    Slurm names a job after that header, and the log is ``<name>-<id>.out``,
    so the name is the one thing a running job and its batch file share
    without asking Slurm. Read from the checkout, which may be behind the
    commit the job pinned: the requeue settings do not move between commits
    the way the code does, and the answer names the file so it can be checked.
    """
    return _batch_files(paths, job_name)[0]


_SOURCED = re.compile(r'^\s*(?:source|\.)\s+"?[^"\s]*ubelix/([\w.-]+\.sh)"?')


def _sourced_scripts(batch: Path, text: str) -> List[str]:
    """The bodies of the ``ubelix/*.sh`` files *text* sources, from beside *batch*.

    Only that directory: the batch files source ``"$REPO/ubelix/<name>.sh"``, and
    the checkout the probe reads is the one ``$REPO`` names. A sourced file that
    is not there contributes nothing rather than a guess.
    """
    bodies = []
    for line in text.splitlines():
        match = _SOURCED.match(line)
        if not match:
            continue
        try:
            bodies.append((batch.parent / match.group(1)).read_text(
                encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return bodies


def _batch_files(paths: Paths, job_name: str) -> Tuple[Optional[Path], List[str]]:
    """``(the batch file named job_name, the ones that could not be read)``.

    A file that could not be read is reported, not skipped: "no batch file is
    named X" and "one of them could not be opened" are different findings, and
    the second must not wear the first's words (#165).
    """
    root = paths.checkout / "ubelix"
    if not root.is_dir():
        return None, []
    unreadable: List[str] = []
    for path in sorted(root.glob("*.sbatch")):
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                match = _SBATCH_OPTION.match(line)
                if match and match.group(1) == "job-name" and match.group(2) == job_name:
                    return path, unreadable
        except OSError as exc:
            unreadable.append("ubelix/%s: %s" % (path.name, exc.__class__.__name__))
    return None, unreadable


def on_timeout(paths: Paths, job_name: str) -> Dict[str, Any]:
    """What happens to a job of this name when it runs into its wall (#226).

    Read from the batch file's ``#SBATCH`` header and body, never assumed:
    ``--requeue`` alone covers a preemption, which Slurm requeues on its own,
    and NOT a wall — a wall ends the job unless the script asks for a signal
    before it (``--signal=B:TERM@N``) and its trap runs ``scontrol requeue``.
    ``train.sbatch`` does both; the scoring and rescoring files do neither, and
    for them TIMEOUT is the end. ``meaning`` is the sentence a report can print.
    """
    path, unreadable = _batch_files(paths, job_name)
    if path is None:
        if unreadable:
            return {"file": None, "unreadable": unreadable,
                    "meaning": "unknown: no readable ubelix/*.sbatch is named %r, and %s "
                               "could not be read" % (job_name, ", ".join(unreadable))}
        return {"file": None, "meaning": "unknown: no ubelix/*.sbatch in the checkout is "
                                         "named %r" % job_name}
    options: Dict[str, Optional[str]] = {}
    trap_requeues = False
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        match = _SBATCH_OPTION.match(line)
        if match:
            options[match.group(1)] = match.group(2)
    # The trap may live in a script the batch file sources: since #280 it is
    # ubelix/requeue_trap.sh, and reading the batch file alone told every report
    # that a wall ends a training job.
    for body in [text] + _sourced_scripts(path, text):
        for line in body.splitlines():
            if "scontrol requeue" in line and not line.lstrip().startswith("#"):
                trap_requeues = True
    signal = options.get("signal")
    requeue = "requeue" in options
    seconds = None
    if signal and "@" in signal:
        try:
            seconds = int(signal.rsplit("@", 1)[1])
        except ValueError:
            seconds = None
    answer = {"file": "ubelix/" + path.name, "signal": signal, "requeue": requeue,
              "trap_requeues": trap_requeues, "signal_before_wall_s": seconds}
    if signal and trap_requeues and requeue:
        answer["meaning"] = ("%s %s before the wall; the trap requeues the same Slurm job and "
                             "the trainer resumes from its newest checkpoint"
                             % (signal.split("@")[0].split(":")[-1],
                                "%d s" % seconds if seconds is not None else "shortly"))
    elif requeue:
        answer["meaning"] = ("requeue on preemption only: a wall ends the job (no --signal "
                             "with a requeueing trap)")
    else:
        answer["meaning"] = "none: TIMEOUT ends the job"
    return answer


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
        # Every row with a log, not only the running ones: a PENDING job whose
        # log already exists is a requeue waiting to continue, and the attempt
        # line in that log names the record — which is how a second live
        # attempt of the same record is seen at all (#231).
        log = find_log(paths, row.get("JobID", ""))
        edges = read_log_edges(log) if log is not None else None
        if edges is not None:
            entry["log"] = str(log)
            attempt = _attempt_in(edges["tail"]) or _attempt_in(edges["head"])
            if attempt is not None:
                entry["job_id"], entry["attempt"] = attempt[0], attempt[2]
            entry["pinned_to"] = _pin_in(log_notes(edges["head"], edges["tail"])["code"])
        if row.get("State") == "RUNNING":
            progress = None
            if edges is not None:
                found = slurm_progress(paths, edges)
                progress = found.pop("progress")
                entry.update(found)
            entry["progress"] = progress
            entry["wall"] = judge_wall(slurm_seconds(row.get("TimeUsed", "")),
                                       slurm_seconds(row.get("TimeLimit", "")), progress,
                                       entry.get("following"), attempt=entry.get("attempt"))
            entry["wall"]["on_timeout"] = on_timeout(paths, row.get("Name", ""))
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
            "pending": sum(1 for r in rows if r["state"] == "PENDING"),
            "duplicates": _mark_duplicates(rows)}


def _pin_in(code_line: Optional[str]) -> Optional[str]:
    """The commit from ``== code: pinned to <hash> …``, or None."""
    match = re.search(r"pinned to ([0-9a-f]{7,40})", code_line or "")
    return match.group(1) if match else None


def _mark_duplicates(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Two live Slurm jobs on one record are never right, and the queue has to
    say so rather than show two healthy jobs (#231, 12.10.2026: 17890142, the
    requeue, and 17961661, submitted by hand, both RUNNING on
    ``…-olmocr2-7b-medieval-page-px4m`` with the same progress, because both
    wrote one ``train.log``).

    Every row of a record with more than one live row gets ``duplicate_of``
    (the other Slurm ids) and ``duplicate_note``; the answer's ``duplicates``
    maps the record to its attempts, oldest first. Which one to cancel is a
    decision and stays with the reader — the note says what is known: which
    is older, which is a requeue (attempt > 1) and which a fresh submission.
    """
    by_record: Dict[str, List[Dict[str, Any]]] = {}
    for entry in rows:
        if entry.get("job_id") and entry.get("state") in LIVE_SLURM_STATES:
            by_record.setdefault(entry["job_id"], []).append(entry)
    duplicates: Dict[str, List[Dict[str, Any]]] = {}
    for job_id, group in by_record.items():
        if len(group) < 2:
            continue
        group.sort(key=lambda e: (e.get("start") or "", e["slurm_job_id"]))
        duplicates[job_id] = [{"slurm_job_id": e["slurm_job_id"], "state": e["state"],
                               "start": e.get("start"), "elapsed": e.get("elapsed"),
                               "attempt": e.get("attempt"), "pinned_to": e.get("pinned_to")}
                              for e in group]
        oldest, newest = group[0], group[-1]
        how = ("the newer one is a fresh submission (attempt 1), not a requeue"
               if newest.get("attempt") == 1 else
               "both carry an attempt number above 1: two requeues of one record")
        note = ("%d live attempts of one record: %s (attempt %s, since %s) is the oldest, "
                "%s (attempt %s, since %s) the newest; %s. Both resume from the same "
                "checkpoint directory and write the same job.json and train.log, and the "
                "second to finish fails its test stage on a record the first already "
                "completed (#153): cancel all but one before that (#231)"
                % (len(group), oldest["slurm_job_id"], oldest.get("attempt"),
                   oldest.get("start"), newest["slurm_job_id"], newest.get("attempt"),
                   newest.get("start"), how))
        for entry in group:
            entry["duplicate_of"] = [o["slurm_job_id"] for o in group if o is not entry]
            entry["duplicate_note"] = note
    return duplicates


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
            entry.update(slurm_progress(paths, edges))
        if entry["short_run"]:
            entry["note"] = ("%d s is not a run: a requeue that found the record finished, "
                             "or a refusal before the first step; read cause" % elapsed)
        rows.append(entry)
    return {"since_days": days, "jobs": rows}


#: Slurm states of an attempt that still holds, or waits for, an allocation.
LIVE_SLURM_STATES = ("RUNNING", "PENDING", "REQUEUED", "SUSPENDED", "COMPLETING",
                     "CONFIGURING", "RESIZING")
#: End states after which a resubmission is a resume and not a decision: the
#: wall, a preemption that did not requeue, a node, a hand. FAILED and
#: OUT_OF_MEMORY carry a cause and need a human first.
RESUMABLE_END_STATES = ("TIMEOUT", "CANCELLED", "NODE_FAIL", "PREEMPTED", "DEADLINE")


def slurm_attempts(run: Runner, paths: Paths, record: Dict[str, Any],
                   unreadable: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Every Slurm job that ran this record, from the attempt line in the Slurm
    logs, with its accounting state.

    A record names its Slurm jobs only once registered (#17), so a live or
    broken one has to be found the other way round: each ``~/ubelix/logs/*.out``
    written since the record was created whose attempt line names it. The
    accounting state comes from one ``sacct -j`` over those ids.

    Three states, never two (#165): a log that could not be read is appended to
    *unreadable* when the caller passes a list — it may be this record's live
    attempt, and nobody could look; an id sacct does not list keeps
    ``state: None`` and says in ``state_why`` that accounting did not answer
    for it, which is "unknown", not "did not run".
    """
    created = (record.get("created_at") or "").replace("Z", "+00:00")
    try:
        since = datetime.fromisoformat(created).timestamp() if created else 0.0
    except ValueError:
        since = 0.0
    found: Dict[str, Dict[str, Any]] = {}
    for path in sorted(paths.logs.glob("*.out")):
        match = re.search(r"-(\d+)\.out$", path.name)
        if match is None:
            continue
        try:
            if path.stat().st_mtime < since:
                continue
            edges = read_log_edges(path)
        except OSError as exc:
            if unreadable is not None:
                unreadable.append("%s: %s" % (path.name, exc.__class__.__name__))
            continue
        attempt = _attempt_in(edges["tail"]) or _attempt_in(edges["head"])
        if attempt is None or attempt[0] != record.get("id"):
            continue
        notes = log_notes(edges["head"], edges["tail"])
        found[match.group(1)] = {"slurm_job_id": match.group(1), "log": str(path),
                                 "attempt": attempt[2],
                                 "sif": _sif_in(edges["tail"]) or _sif_in(edges["head"]),
                                 "cause": notes["cause"], "state": None, "end": None}
    if not found:
        return []
    out = run(["sacct", "-j", ",".join(sorted(found)), "-X", "-P", "-n",
               "-o", "JobID,JobName,State,Elapsed,End"])
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) >= 5 and parts[0] in found:
            found[parts[0]].update({"name": parts[1], "state": parts[2], "elapsed": parts[3],
                                    "end": parts[4]})
    for entry in found.values():
        if entry["state"] is None:
            entry["state_why"] = ("sacct did not list job %s: outside its window, or "
                                  "accounting did not answer — unknown, not \"did not run\""
                                  % entry["slurm_job_id"])
    return sorted(found.values(), key=lambda a: (a.get("end") or "", a["slurm_job_id"]))


def _newest_checkpoint(checkpoint_dir: Optional[str]) -> Optional[str]:
    if not checkpoint_dir:
        return None
    best = None
    try:
        for entry in Path(checkpoint_dir).iterdir():
            match = re.match(r"checkpoint-(\d+)$", entry.name)
            if match and entry.is_dir() and (best is None or int(match.group(1)) > best[0]):
                best = (int(match.group(1)), entry.name)
    except OSError:
        return None
    return best[1] if best else None


def resubmit_for(run: Runner, paths: Paths, record: Dict[str, Any]) -> Dict[str, Any]:
    """The one command that puts a stopped UBELIX training back in the queue —
    written out, pinned, never run from here (#226).

    ``command`` is set only when nothing of this record is still in Slurm and
    its last attempt ended in a way a resume answers (the wall, a preemption,
    a node, a hand); a FAILED attempt carries its cause instead. The pin is the
    commit the last attempt ran, not HEAD: every drift line in the records so
    far came from a resubmission by hand that let ``submit.sh`` pin whatever
    HEAD had become. The container is the one the last attempt printed. A
    checkout behind ``origin/main`` is named, because ``submit.sh`` refuses it
    and pulling is a write that stays with a person.
    """
    status = record.get("status")
    job_id = record.get("id")
    if record.get("host") != SCRATCH_HOST:
        return {"command": None,
                "why": "host %s: only a UBELIX record is resubmitted with train.sbatch"
                       % record.get("host")}
    if status not in ("training", "queued"):
        return {"command": None, "why": "status %s: nothing to resubmit" % status}
    unreadable: List[str] = []
    attempts = slurm_attempts(run, paths, record, unreadable=unreadable)
    live = [a for a in attempts if (a.get("state") or "").split(" ")[0] in LIVE_SLURM_STATES]
    answer: Dict[str, Any] = {"attempts": [{k: a.get(k) for k in
                                            ("slurm_job_id", "attempt", "state", "end", "cause",
                                             "state_why") if k in a}
                                           for a in attempts]}
    if unreadable:
        answer["unreadable_logs"] = unreadable
    if live:
        answer["command"] = None
        answer["why"] = "Slurm job %s is %s for this record: a second submission would run " \
                        "two attempts on one checkpoint directory" \
                        % (live[-1]["slurm_job_id"], live[-1]["state"])
        return answer
    if unreadable:
        # One of those logs may be this record's live attempt, and nobody could
        # look. "Could not look" is not "nothing there" (#165).
        answer["command"] = None
        answer["why"] = "%d Slurm log(s) since this record was created could not be read " \
                        "(%s): whether an attempt of this record is still live is not known " \
                        "from here" % (len(unreadable), ", ".join(unreadable))
        return answer
    last = attempts[-1] if attempts else None
    last_state = (last.get("state") or "").split(" ")[0] if last else None
    if last and last_state not in RESUMABLE_END_STATES:
        answer["command"] = None
        if last_state in ("FAILED", "OUT_OF_MEMORY"):
            answer["why"] = "the last attempt %s ended %s%s: a failure needs a person, not a " \
                            "requeue" % (last["slurm_job_id"], last_state,
                                         " (%s)" % last["cause"] if last.get("cause") else "")
        else:
            answer["why"] = "the last attempt %s ended %s while the record is still %s: read " \
                            "its log before resubmitting" % (last["slurm_job_id"],
                                                             last_state or "unknown", status)
        return answer
    # The pin: the commit the last GPU stage ran, else the one the job was created with.
    pinned = None
    pin_source = None
    for stage in record.get("stages") or []:
        commit = (stage.get("code") or {}).get("commit")
        if stage.get("name") in ("train", "test") and commit:
            pinned, pin_source = commit, "stage %s of this record" % stage.get("name")
    if pinned is None:
        pinned = (record.get("code") or {}).get("commit")
        pin_source = "the commit the job was created with" if pinned else None
    sif = last.get("sif") if last else None
    checkpoint = _newest_checkpoint(record.get("checkpoint_dir"))
    parts = ["cd %s &&" % paths.checkout]
    if pinned:
        parts.append("ATR_CODE_COMMIT=%s" % pinned)
    parts.append("JOB_ID=%s ubelix/submit.sh ubelix/train.sbatch" % job_id)
    if sif:
        parts.append("-- --export=ALL,SIF=%s" % sif)
    answer["command"] = " ".join(parts)
    answer["kind"] = ("resume" if checkpoint else
                      "first train submission" if last is None else "restart without checkpoint")
    answer["resume_from"] = checkpoint
    answer["pinned_to"] = pinned
    answer["pin_source"] = pin_source
    answer["sif"] = sif
    answer["sif_source"] = ("log of Slurm job %s" % last["slurm_job_id"] if sif else
                            "no attempt printed one: train.sbatch's default applies")
    # Unattributed pending jobs: a fresh submission has no log and no attempt
    # line yet, so it cannot be tied to a record — said, not assumed away (#165).
    pending_without_log = []
    for line in run(["squeue", "-u", paths.user, "-h", "-O", "JobID:|,Name:|,State:|"]).splitlines():
        cols = [c.strip() for c in line.split("|")]
        if len(cols) >= 3 and cols[2] == "PENDING" and find_log(paths, cols[0]) is None:
            pending_without_log.append("%s (%s)" % (cols[0], cols[1]))
    answer["unattributed_pending"] = pending_without_log
    behind_why = None
    try:
        counts = run(["git", "-C", str(paths.checkout), "rev-list", "--left-right", "--count",
                      "HEAD...origin/main"]).split()
        behind = int(counts[1]) if len(counts) == 2 else None
        if behind is None:
            behind_why = "git rev-list answered %r, not two counts" % " ".join(counts)
    except (ProbeError, ValueError) as exc:
        behind = None
        behind_why = "git could not be asked: %s" % exc
    answer["checkout_behind"] = behind
    if behind_why:
        answer["checkout_behind_why"] = behind_why + " — whether submit.sh would refuse " \
                                        "this checkout is not known from here"
    why = []
    if last is None and checkpoint:
        # Seen 12.10.2026 on throughput-qwen35-4b: checkpoint-400 on disk, no Slurm
        # log since created_at names the record (purged, renamed, or the train
        # stage ran off Slurm). The checkpoint is the fact; the log is missing.
        why.append("no Slurm log since %s names this record, but %s exists under %s: the "
                   "resubmission resumes from it"
                   % (utc_to_cest(record.get("created_at")), checkpoint,
                      record.get("checkpoint_dir")))
    elif last is None:
        why.append("never ran on Slurm: this is the first train submission, a decision rather "
                   "than a resume")
    else:
        why.append("the last attempt %s ended %s%s" % (
            last["slurm_job_id"], last_state, " at %s" % last["end"] if last.get("end") else ""))
        why.append("resumes from %s" % checkpoint if checkpoint else
                   "no checkpoint-N under %s: a resubmission starts the stage over"
                   % record.get("checkpoint_dir"))
    if pinned:
        why.append("pinned to %s (%s) so the resume runs the code it started with; drop "
                   "ATR_CODE_COMMIT on purpose to pick up a fix" % (pinned[:12], pin_source))
    if pending_without_log:
        why.append("pending without a log yet, so not attributable: %s"
                   % ", ".join(pending_without_log))
    if behind:
        why.append("the checkout is %d behind origin/main and submit.sh will refuse: pull first"
                   % behind)
    answer["why"] = "; ".join(why)
    return answer


def job(paths: Paths, job_id: str, run: Optional[Runner] = None) -> Dict[str, Any]:
    job_dir, record = find_job(paths, job_id)
    if job_dir is None or record is None:
        return {"error": "no %s/job.json in any store" % job_id, "stores": store_states(paths)}
    summary = summarise_job(job_dir, record)
    if run is not None:
        summary["resubmit"] = resubmit_for(run, paths, record)
    return summary


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
            "id": summary["id"], "host": summary["host"], "status": summary["status"],
            "model_id": summary["model_id"], "engine": summary["engine"],
            "base_model": summary["base_model"], "granularity": summary["granularity"],
            "load_in_4bit": summary["load_in_4bit"],
            "cer": metrics.get("cer"), "wer": metrics.get("wer"),
            "length_ratio": metrics.get("length_ratio"),
            "truncated_cer": metrics.get("truncated_cer"), "samples": metrics.get("samples"),
            "draw_md5": (draw.get("md5") or "")[:8] or None, "draw_lines": draw.get("lines"),
            "finished_cest": summary["finished_cest"],
            "code_drift": bool(summary["code_drift"]),
            "reserved_pages": summary["heldout"]["reserved_pages"],
            "heldout_fp": summary["heldout"]["fingerprint"],
            "runtime": summary["runtime"]["text"],
        })
    rows.sort(key=lambda r: (r["finished_cest"] or "", r["id"]))
    return {"jobs_root": str(paths.jobs_root), "stores": store_states(paths), "rows": rows,
            "note": "two CER values compare only when draw_md5 AND reserved_pages "
                    "match: the draw says they were scored on the same lines (#108), "
                    "reserved_pages that they were trained on the same corpus (#98). "
                    "The medieval ladder's September rows have reserved_pages 0 or "
                    "None and its October rows 3173 — 28 % fewer training lines from "
                    "the same spec, so they do not compare in either direction. "
                    "reserved_pages None means nobody looked, which is not 0. Host "
                    "says which machine trained the row; runtime says in which "
                    "image and transformers the number was measured — 'unknown' "
                    "on every record from before 22.10.2026, which is not 'no "
                    "container' (#206)"}


def draw(paths: Paths, job_id: str) -> Dict[str, Any]:
    job_dir, own_record = find_job(paths, job_id)
    own = draw_of(job_dir) if job_dir is not None else None
    if own is None:
        return {"job_id": job_id, "draw": None,
                "error": "no data/val_eval.jsonl: the test stage has not drawn yet"
                if job_dir is not None else "no %s/job.json in any store" % job_id}
    shared = []
    for other_dir, record in iter_jobs(paths):
        if other_dir.name == job_id:
            continue
        other = draw_of(other_dir)
        if other and other["md5"] == own["md5"]:
            metrics = record.get("metrics") or {}
            shared.append({"id": other_dir.name, "host": record["host"],
                           "status": record.get("status"), "cer": metrics.get("cer"),
                           "heldout": _heldout_of(record)})
    mine = _heldout_of(own_record or {})
    # Eine geteilte Ziehung heisst, dass zwei Laeufe auf denselben Zeilen gemessen
    # wurden — nicht, dass sie auf demselben Korpus trainiert haben. Genau diese
    # Luecke hat die medieval-Leiter drei Wochen lang falsch gelesen.
    mismatched = [o["id"] for o in shared
                  if o["heldout"]["reserved_pages"] != mine["reserved_pages"]]
    answer = {"job_id": job_id, "draw": own, "heldout": mine, "shared_with": shared}
    if mismatched:
        answer["not_comparable"] = mismatched
        answer["why"] = _why_not_comparable(mine["reserved_pages"], [
            o for o in shared if o["id"] in mismatched])
    return answer


def _why_not_comparable(mine: Any, others: List[Dict[str, Any]]) -> str:
    """Why rows that share a draw are still not compared — without overclaiming.

    Only two *positive* counts that differ prove a different corpus: a positive
    count matched the run's manifest in every case the census could check
    (#247). ``0`` and ``None`` prove no check at all — a run that adopted a
    compiled corpus records ``0`` without reserving anything — so against them
    the records can confirm neither the same corpus nor a different one. The
    September 27B (``0``) and 9B (``None``) shared a draw and, measured, a
    corpus; "trained on a different corpus" was the wrong reason to refuse them.
    """
    def counted(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    def values(rows: List[Dict[str, Any]]) -> str:
        return ", ".join(sorted({repr(o["heldout"]["reserved_pages"]) for o in rows}))

    differ = [o for o in others if counted(mine) and counted(o["heldout"]["reserved_pages"])]
    unsure = [o for o in others if o not in differ]
    parts = []
    if differ:
        parts.append("%s share the draw but were trained on a different corpus: "
                     "reserved_pages %r here against %s there."
                     % (", ".join(o["id"] for o in differ), mine, values(differ)))
    if unsure:
        parts.append("%s share the draw, and the records cannot confirm the same "
                     "corpus: reserved_pages %r here against %s there, and 0 or "
                     "None proves no check (an adopted corpus records 0 without "
                     "reserving; docs/HELDOUT_BOUNDARY.md has the measured answer)."
                     % (", ".join(o["id"] for o in unsure), mine, values(unsure)))
    parts.append("A shared draw is scoring, not training (#98).")
    return " ".join(parts)


def _heldout_of(record: Dict[str, Any]) -> Dict[str, Any]:
    """Wie viele Seiten dieser Lauf fuer die Messung zurueckgehalten hat.

    Drei Zustaende, die verschieden zu lesen sind, und `None` ist nicht `0`:
    `0` heisst „Registry befragt, sie hielt nichts zurueck", `None` heisst
    „niemand hat nachgesehen" (so stand es beim September-9B), und eine Zahl
    heisst, dass genau so viele Seiten dem Training fehlen.
    """
    progress = record.get("progress") or {}
    pages = progress.get("reserved_pages")
    return {
        "reserved_pages": pages,
        "source": progress.get("reserved_pages_source"),
        "fingerprint": (progress.get("heldout_fingerprint") or "")[:16] or None,
        "consulted": pages is not None,
    }


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
        # Only /scratch purges; the share keeps a job directory until someone
        # deletes it (#183), so a share row carries no purge date.
        purge = (mtime + SCRATCH_PURGE.total_seconds()
                 if mtime and summary["store"] == "scratch" else None)
        rows.append({
            "id": summary["id"], "host": summary["host"], "status": summary["status"],
            "model_id": summary["model_id"],
            "base_model": summary["base_model"], "granularity": summary["granularity"],
            "load_in_4bit": summary["load_in_4bit"], "created_cest": summary["created_cest"],
            "data_touched_cest": epoch_to_cest(mtime),
            "scratch_purge_cest": epoch_to_cest(purge),
            "artefact": summary["artefact"],
            "superseded_by": completed_models.get(summary["model_id"], []),
        })
    return {"rows": rows, "stores": store_states(paths),
            "note": "superseded_by lists completed jobs of the same model_id: a leftover, "
                    "not a next step; scratch_purge_cest is null for a job on the share, "
                    "which never purges"}


def _stage_log(job_dir: Path, stage: Optional[str]) -> Optional[Path]:
    """The log that says where a live job is: ``logs/<stage>.log`` when the stage
    runs a subprocess and has written to it, else ``logs/runner.log``. The same
    choice the runner makes for a failure (``_failure_log``)."""
    candidates = []
    if stage:
        candidates.append(job_dir / "logs" / ("%s.log" % stage))
    candidates.append(job_dir / "logs" / "runner.log")
    for path in candidates:
        try:
            if path.is_file() and path.stat().st_size:
                return path
        except OSError:
            continue
    return None


def live(paths: Paths, host: Optional[str] = None) -> Dict[str, Any]:
    """Records with a runner, or waiting for one, on any host, read from the record.

    This is the asteraix counterpart of ``queue``: there is no Slurm there, so
    the status, the stage and the progress counter in the stage log are the
    live state. The same reading works for a UBELIX record, and ``slurm_jobs``
    then names what ``slurm_job`` can ask about.

    A record stays live after its runner died until something closes it: on
    asteraix the service's next reconcile marks it failed, on UBELIX nobody
    does (``close_job`` by hand, #17). ``updated_cest`` and ``log_modified_cest``
    are what a reader has to judge that by.
    """
    rows = []
    for job_dir, record in iter_jobs(paths):
        if record.get("status") not in LIVE_STATUSES:
            continue
        if host and record["host"] != host:
            continue
        summary = summarise_job(job_dir, record, with_draw=False)
        progress = record.get("progress") or {}
        log_path = _stage_log(job_dir, record.get("stage"))
        counter = None
        log_modified = None
        if log_path is not None:
            edges = read_log_edges(log_path)
            counter = last_progress(edges["tail"], source=log_path.name)
            log_modified = epoch_to_cest(log_path.stat().st_mtime)
        rows.append({
            "id": summary["id"], "host": summary["host"], "store": summary["store"],
            "status": summary["status"], "stage": summary["stage"],
            "model_id": summary["model_id"], "engine": summary["engine"],
            "base_model": summary["base_model"], "granularity": summary["granularity"],
            "queued_reason": record.get("queued_reason"),
            "pid": record.get("pid"), "gpus": record.get("gpus") or [],
            "created_cest": summary["created_cest"], "started_cest": summary["started_cest"],
            "updated_cest": utc_to_cest(record.get("updated_at")),
            "epoch": progress.get("epoch"), "epochs": progress.get("epochs"),
            "total_steps": progress.get("total_steps"),
            "val_accuracy": progress.get("val_accuracy"),
            "peak_gpu_mib": progress.get("peak_gpu_mib"),
            "log": str(log_path) if log_path else None,
            "log_modified_cest": log_modified, "progress": counter,
            "slurm_jobs": summary["slurm_jobs"], "artefact": summary["artefact"],
        })
    rows.sort(key=lambda r: (r["host"] or "", r["started_cest"] or "", r["id"]))
    return {"host": host, "rows": rows, "stores": store_states(paths),
            "note": "live means the record says so; a runner that died leaves the record "
                    "live until the service (asteraix) or close_job (UBELIX) closes it, so "
                    "read updated_cest and log_modified_cest before believing a row"}


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
    disk_why = None
    try:
        line = run(["df", "-P", str(paths.home)]).splitlines()[-1].split()
        disk = {"filesystem": line[0], "use_percent": line[4],
                "avail_gib": round(int(line[3]) / 2 ** 20, 1), "mounted_on": line[5]}
    except (ProbeError, IndexError, ValueError) as exc:
        # Not measured is not "fine": the answer carries the reason (#165).
        disk_why = "df could not be read: %s" % (exc or exc.__class__.__name__)
    return {"now_cest": _label(now), "artefacts": artefacts, "scratch_purge": purge,
            "home": disk, "home_why": disk_why,
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
    notes.update(slurm_progress(paths, edges))
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
        entry.update(slurm_progress(paths, edges))
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


# ── research-storage ────────────────────────────────────────────────────────
def storage(run: Runner, paths: Paths) -> Dict[str, Any]:
    """Belegung des research-storage und, bei Unterschreitung, die Vorschläge.

    Ruft das ausgerollte ``research_storage.py --json`` auf statt dessen Regeln
    hier nachzubauen. Zwei Fassungen derselben Regeln wären eine Gabelung, und
    die Regeln sind der Teil, der sich ändert: welche Verzeichnisse
    wiederherstellbar sind, was nur die einzige lokale Kopie ist, was abgeleitet
    ist. Dieser Probe ist der Zusteller, nicht der Urheber.
    """
    script = paths.storage_script
    if not script.exists():
        return {"error": "%s fehlt — das Prüfskript ist auf diesem Host nicht "
                         "ausgerollt (scp scripts/research_storage.py "
                         "ubelix:~/ubelix/)" % script,
                "deployed": False}
    try:
        out = run(["python3", str(script), "--json"], timeout=120.0)
    except ProbeError as exc:
        return {"error": str(exc), "deployed": True}
    try:
        record = json.loads(out)
    except ValueError:
        return {"error": "%s hat kein JSON geliefert: %s" % (script, out[:200]),
                "deployed": True}
    record["script"] = str(script)
    return record


# ── entry point ─────────────────────────────────────────────────────────────
def dispatch(cmd: str, args: Dict[str, Any], run: Runner, paths: Paths) -> Dict[str, Any]:
    if cmd == "queue":
        return queue(run, paths)
    if cmd == "finished":
        return finished(run, paths, days=int(args.get("days", 2)))
    if cmd == "job":
        return job(paths, str(args["job_id"]), run=run)
    if cmd == "results":
        return results(paths, status=args.get("status"), granularity=args.get("granularity"),
                       base_model=args.get("base_model"))
    if cmd == "draw":
        return draw(paths, str(args["job_id"]))
    if cmd == "prepared":
        return prepared(paths)
    if cmd == "live":
        return live(paths, host=args.get("host"))
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
    if cmd == "storage":
        return storage(run, paths)
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

#!/usr/bin/env python3
"""Remove the working directories of line-dataset jobs that sacct says are over (#183).

    python ubelix/clean_line_workdirs.py            # list, delete nothing
    python ubelix/clean_line_workdirs.py --apply    # delete what sacct calls dead
    python ubelix/clean_line_workdirs.py --apply --unattributed
                                                    # also the directories no job can
                                                    # be matched to, when no line-dataset
                                                    # job of yours is queued or running

Run on a login node (it calls sacct and squeue), outside the container. Stdlib only.

Nine leftovers were 1.1 TB on 08.10.2026, one of them 380 GB, every one the
working directory of a job that ended by SIGKILL. The rule here is the one from
#165: the state is looked up, not inferred. A directory ``job-<id>`` is removed
when ``sacct`` reports that job in a terminal state; one whose job is RUNNING or
PENDING is kept; one sacct does not know (older than its window, or a name from
before #183: ``line-ds-<random>``, ``pid-<n>``) is **unknown**, which is not
dead, and goes only with ``--unattributed`` and only while ``squeue`` shows no
line-dataset job of yours at all — the one condition under which nothing can be
writing to it. Never by mtime: the top directory's mtime does not move while
``materialize`` writes below it, and a cleaner reading it would have deleted a
running job's 300 GB (measured in the issue).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

#: What sacct's State column starts with when a job is over. "CANCELLED by 123"
#: is matched on its first word.
TERMINAL = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "PREEMPTED",
            "NODE_FAIL", "DEADLINE", "BOOT_FAIL", "REVOKED")
#: Still going, or about to: never touched.
ALIVE = ("RUNNING", "PENDING", "REQUEUED", "SUSPENDED", "COMPLETING", "CONFIGURING",
         "RESIZING", "STAGE_OUT", "SIGNALING", "STOPPED")
JOB_DIR = re.compile(r"^job-(\d+)$")
#: The batch job's name in make_line_dataset.sbatch; what squeue is asked for.
JOB_NAME = "line-dataset"


@dataclass(frozen=True)
class Entry:
    path: Path
    job_id: str | None
    #: sacct's State for the job, None when sacct did not list it or there is no job.
    state: str | None
    end: str | None
    size_bytes: int | None

    @property
    def verdict(self) -> str:
        """``dead``, ``alive`` or ``unknown`` — three answers, never two."""
        if self.state is None:
            return "unknown"
        first = self.state.split()[0] if self.state.split() else ""
        if first.startswith(TERMINAL):
            return "dead"
        if first.startswith(ALIVE):
            return "alive"
        return "unknown"


def run(argv: list[str]) -> str:
    """stdout of a Slurm command; empty when it is missing or fails."""
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=60,
                              check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def sacct_states(job_ids: list[str]) -> dict[str, tuple[str, str]]:
    """``{job_id: (State, End)}`` for the ids sacct knows."""
    if not job_ids:
        return {}
    out = run(["sacct", "-j", ",".join(sorted(job_ids)), "-X", "-P", "-n",
               "-o", "JobID,State,End"])
    found: dict[str, tuple[str, str]] = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) >= 3 and parts[0] in job_ids:
            found[parts[0]] = (parts[1], parts[2])
    return found


def line_dataset_jobs_queued(user: str) -> list[str]:
    """Ids of this user's line-dataset jobs squeue shows, queued or running."""
    out = run(["squeue", "-u", user, "-h", "-o", "%i %j"])
    return [line.split()[0] for line in out.splitlines()
            if len(line.split()) >= 2 and line.split()[1] == JOB_NAME]


def du_bytes(path: Path) -> int | None:
    out = run(["du", "-sb", str(path)])
    try:
        return int(out.split()[0])
    except (IndexError, ValueError):
        return None


def entries(base: Path, *, sizes: bool = True) -> list[Entry]:
    """Every directory under *base*, with its job's state where it has one."""
    dirs = sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    ids = {p: m.group(1) for p in dirs if (m := JOB_DIR.match(p.name))}
    states = sacct_states(sorted(set(ids.values())))
    out = []
    for path in dirs:
        job = ids.get(path)
        state, end = states.get(job, (None, None)) if job else (None, None)
        out.append(Entry(path=path, job_id=job, state=state, end=end,
                         size_bytes=du_bytes(path) if sizes else None))
    return out


def _gb(size: int | None) -> str:
    return "?" if size is None else f"{size / 1e9:.1f} GB"


def plan(found: list[Entry], *, unattributed: bool, queued: list[str]) -> list[tuple[Entry, str, str]]:
    """``(entry, action, why)`` per directory: ``remove`` or ``keep``."""
    out = []
    for entry in found:
        verdict = entry.verdict
        if verdict == "dead":
            out.append((entry, "remove", f"job {entry.job_id} {entry.state} {entry.end or ''}".rstrip()))
        elif verdict == "alive":
            out.append((entry, "keep", f"job {entry.job_id} is {entry.state}"))
        elif not unattributed:
            why = (f"job {entry.job_id} is not in sacct's window" if entry.job_id
                   else "no job id in the name (from before #183)")
            out.append((entry, "keep", f"{why}; unknown is not dead (--unattributed)"))
        elif queued:
            out.append((entry, "keep", f"unknown, and line-dataset job(s) {', '.join(queued)} "
                                       "are queued or running — one of them may be writing here"))
        else:
            out.append((entry, "remove", "unknown to sacct, and no line-dataset job of yours "
                                         "is queued or running"))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, default=None,
                        help="the line-datasets directory (default: $ATR_TRAIN_SCRATCH or "
                             "/scratch/network/users/$USER/line-datasets)")
    parser.add_argument("--apply", action="store_true", help="delete; the default only lists")
    parser.add_argument("--unattributed", action="store_true",
                        help="also remove directories no job can be matched to, when no "
                             "line-dataset job of yours is queued or running")
    parser.add_argument("--no-size", action="store_true", help="skip du (slow on 300 GB)")
    args = parser.parse_args(argv)
    user = os.environ.get("USER", "unknown")
    base = args.base or Path(os.environ.get("ATR_TRAIN_SCRATCH")
                             or f"/scratch/network/users/{user}/line-datasets")
    if not base.is_dir():
        print(f"{base}: no such directory — nothing to clean")
        return 0

    found = entries(base, sizes=not args.no_size)
    queued = line_dataset_jobs_queued(user) if args.unattributed else []
    decisions = plan(found, unattributed=args.unattributed, queued=queued)
    if not decisions:
        print(f"{base}: empty")
        return 0

    width = max(len(e.path.name) for e, _, _ in decisions)
    freed = 0
    removed = kept = 0
    for entry, action, why in decisions:
        print(f"{action:6} {entry.path.name:{width}}  {_gb(entry.size_bytes):>9}  {why}")
        if action == "remove":
            if args.apply:
                shutil.rmtree(entry.path, ignore_errors=True)
                if entry.path.exists():
                    print(f"       {entry.path} could not be removed entirely")
                    continue
            freed += entry.size_bytes or 0
            removed += 1
        else:
            kept += 1
    verb = "removed" if args.apply else "would remove"
    print(f"\n{verb} {removed} director{'y' if removed == 1 else 'ies'} ({_gb(freed)}), "
          f"kept {kept}" + ("" if args.apply else " — re-run with --apply to delete"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

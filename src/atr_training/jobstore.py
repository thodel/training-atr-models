"""On-disk job store — the single source of truth for a training run.

A job lives entirely in its directory::

    <root>/<job_id>/
        job.json          the TrainJob record (atomically replaced)
        data/             materialized pages, manifests, .arrow datasets
        checkpoints/      ketos --output
        model/            promoted weights + metadata.json
        logs/<stage>.log  one log per stage

Nothing about a job is held only in the service's memory. The runner is a
**detached** child (``start_new_session=True``), so restarting ``atr-train`` must
reconcile against what is on disk and what is still running — never kill the run
and never assume it survived (:meth:`JobStore.reconcile`).

**A job belongs to one host** (#15). The store sits on the research share, which
idhefix, asteraix and UBELIX all mount, and a record's ``pid`` is a statement
about the machine that assigned it and no other. So every record names its host
(:meth:`JobStore.host_of`), and a store opened with a ``host_id`` spawns,
reconciles and signals only its own jobs. Two schedulers of one host are kept
apart by a claim file taken with ``O_CREAT|O_EXCL`` before a spawn
(:meth:`JobStore.claim`); on the share that call is exclusive across machines as
well — created on asteraix, the same call on idhefix raised ``FileExistsError``
(measured 16.09.2026).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from loguru import logger

from atr_training.contracts import (
    STAGE_STATUS,
    TERMINAL_STATUSES,
    JobStatus,
    TrainJob,
    TrainRequest,
    utcnow,
)

__all__ = ["JobStoreError", "IllegalTransition", "JobPaths", "JobStore", "SpawnClaim",
           "TRANSITIONS", "SLURM_HOST", "LEGACY_JOB_HOST", "CLAIM_ORPHAN_AFTER_S",
           "CLAIM_SPAWN", "CLAIM_CANCEL", "CLAIM_CLOSE"]

#: The host on records Slurm supervises (``ubelix/submit_job.py``, ``fanout.py``).
#: No trainer owns it, whatever its own ``host_id`` says: the runner of such a job
#: lives on a compute node, and its pid is that node's.
SLURM_HOST = "ubelix"
#: Whose a record without ``host`` is — see ``TrainerSettings.legacy_job_host``.
LEGACY_JOB_HOST = "idhefix"
#: How long a claim may stand without the runner pid it promises. Claim → spawn →
#: save pid takes well under a second, and the runner writes its own pid as its
#: first act. Ten minutes past the claim with no pid, the scheduler died in
#: between, and the job would otherwise sit queued for ever: a claimed job is
#: never offered to the queue again.
CLAIM_ORPHAN_AFTER_S = 600.0

#: What a claim was taken for. One file, one owner, whatever the purpose: a
#: scheduler about to start the job, a cancel of a job that has not started,
#: or an operator closing another host's record (:mod:`atr_training.close_job`).
#: Whoever creates the file decides the job's next state; everyone else stands
#: aside.
CLAIM_SPAWN = "spawn"
CLAIM_CANCEL = "cancel"
CLAIM_CLOSE = "close"


class JobStoreError(RuntimeError):
    """Raised on a job-store operation that cannot be satisfied."""


class IllegalTransition(JobStoreError):
    """Raised when a status change is not part of the job lifecycle."""


#: The lifecycle. Terminal statuses have no outgoing edges.
#:
#: ``training`` has a **self-edge**, and it is the only one. A job on a
#: preemptable queue can lose its node mid-training and be requeued by the
#: scheduler; when the runner starts again on the same job it is not beginning a
#: new attempt, it is continuing this one from the last checkpoint. Marking that
#: ``cancelled`` and starting over would throw away hours of GPU time and, on a
#: multi-day run, would never finish at all. Every other status stays exactly as
#: strict: a completed or failed job is still terminal, and a cancellation is
#: still a cancellation.
TRANSITIONS: dict[str, frozenset[str]] = {
    "queued": frozenset({"preparing", "cancelled", "failed"}),
    "preparing": frozenset({"compiling", "cancelled", "failed"}),
    "compiling": frozenset({"training", "cancelled", "failed"}),
    "training": frozenset({"testing", "cancelled", "failed", "training"}),
    "testing": frozenset({"registering", "cancelled", "failed"}),
    "registering": frozenset({"completed", "cancelled", "failed"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}

_JOB_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[a-z0-9._-]+$")


@dataclass(frozen=True)
class JobPaths:
    root: Path

    @property
    def job_json(self) -> Path:
        return self.root / "job.json"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def pages(self) -> Path:
        return self.data / "pages"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def model(self) -> Path:
        return self.root / "model"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def claim(self) -> Path:
        """Taken exclusively before the runner is spawned (:meth:`JobStore.claim`)."""
        return self.root / "spawn.claim"

    def log(self, stage: str) -> Path:
        return self.logs / f"{stage}.log"

    def mkdirs(self) -> None:
        for p in (self.data, self.pages, self.checkpoints, self.model, self.logs):
            p.mkdir(parents=True, exist_ok=True)


def _pid_state(pid: int, proc_root: str | Path = "/proc") -> str | None:
    """The process state letter from ``/proc/<pid>/stat``, or None if no entry.

    The ``comm`` field is parenthesised and may contain spaces — a defunct child
    of ours reads ``(python) Z``, and one named by the kernel can read worse — so
    the line is split on the **last** ``)`` rather than on whitespace. Splitting
    from the left is the bug this function exists to avoid introducing.
    """
    try:
        line = (Path(proc_root) / str(pid) / "stat").read_text(encoding="utf-8")
        return line.rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return None


def _pid_alive(pid: int, proc_root: str | Path = "/proc") -> bool:
    """Whether ``pid`` is a process still doing something.

    **A zombie is dead.** This is the whole reason the function does not simply
    use ``os.kill(pid, 0)``: a defunct child keeps its pid and its ``/proc`` entry
    until someone waits on it, so the signal probe succeeds and the process reads
    as alive for ever. On 2026-09-10 that left
    ``20260909T190659Z-qwen3vl-german-pages-v2`` at ``status: training`` after its
    trainer had died in a network outage — measured on the box:

        os.kill(2786095, 0)      -> no error
        /proc/2786095/stat       -> state Z

    and with ``max_concurrent: 1`` that one stale record meant no job could start
    again, on two idle GPUs, until the record was edited by hand (#118).

    ``/proc`` is Linux; elsewhere there is no zombie distinction to be had from
    the signal probe, so that is the fallback and it keeps the old behaviour.
    """
    if Path(proc_root).is_dir():
        # Where /proc exists it is authoritative, for both answers: a state letter
        # says what the process is doing, and a missing entry says it is gone.
        return _pid_state(pid, proc_root) not in (None, "Z")
    # No /proc at all (macOS, where the dev suite runs). There is no zombie
    # distinction to be had from a signal probe, so this keeps the old behaviour.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else
        return True
    return True


def reap_children() -> int:
    """Wait on any finished child, so it stops being a zombie. Returns how many.

    The runners are spawned detached and their ``Popen`` handles are discarded, so
    nothing ever waits on them and every finished run leaves a defunct entry
    behind. :func:`_pid_alive` no longer believes those, which is the fix that
    matters; this keeps them from piling up in the process table as well.

    Never raises and never blocks: ``ECHILD`` simply means there is nothing to
    reap, which is the normal case.
    """
    reaped = 0
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return reaped
        except OSError:
            return reaped
        if pid == 0:
            return reaped
        reaped += 1


@dataclass(frozen=True)
class SpawnClaim:
    """Who claimed a job for spawning, and when. Any field may be unknown: the
    file is created before its content is written, and a reader can land in
    between."""

    host: str | None
    pid: int | None
    at: datetime | None
    #: :data:`CLAIM_SPAWN` for a file without the key: that is all it was used
    #: for before the cancel took it too.
    purpose: str = CLAIM_SPAWN


class JobStore:
    """Directory-backed store for :class:`TrainJob` records.

    ``host_id`` is who is asking. The runner and the UBELIX scripts only load and
    save one record and open the store without it; everything that judges a job
    by its pid or starts one needs it, and refuses without it rather than
    guessing (:meth:`owns`).
    """

    def __init__(self, root: str | Path, host_id: str | None = None,
                 legacy_host: str = LEGACY_JOB_HOST,
                 claim_orphan_after_s: float = CLAIM_ORPHAN_AFTER_S) -> None:
        self.root = Path(root)
        self.host_id = host_id
        self.legacy_host = legacy_host
        self.claim_orphan_after_s = claim_orphan_after_s
        # Foreign jobs are listed on every scheduler tick; saying once per job
        # that they are left alone is information, saying it every 10 s is noise.
        self._foreign_noted: set[str] = set()

    # ── ownership (#15) ─────────────────────────────────────────────────────
    def host_of(self, job: TrainJob) -> str:
        """The host a job belongs to. The one place a missing ``host`` is read."""
        return job.host or self.legacy_host

    def owns(self, job: TrainJob) -> bool:
        """Whether this store's host may spawn, reconcile or signal ``job``."""
        if not self.host_id:
            raise JobStoreError(
                "this job store was opened without a host id, so it cannot tell its "
                "own jobs from another machine's — and a pid is only meaningful on "
                "the machine that assigned it (#15)")
        host = self.host_of(job)
        return host == self.host_id and host != SLURM_HOST

    def claim(self, job_id: str, now: datetime | None = None,
              purpose: str = CLAIM_SPAWN) -> None:
        """Claim ``job_id`` for spawning (or ``purpose``), or raise ``FileExistsError``.

        ``O_CREAT|O_EXCL`` is the whole mechanism: one caller creates the file and
        every other gets ``FileExistsError``. That is what keeps two schedulers of
        one host (#12) from both starting a job each has listed as unstarted. On
        the share the call is exclusive across machines too (measured
        16.09.2026). Nothing else there would do: chmod, symlink and hardlink all
        fail on that mount, and the tmp-file + ``os.replace`` the records use
        overwrites rather than refuses.

        The same file is the lock between starting a queued job and cancelling it
        before it starts: a cancel that only rewrote the record was undone by a
        tick that had listed the job earlier and then saved its own copy, and
        the job started anyway (#15 review). So a cancel takes the claim too.

        The file stays after the spawn; a terminal job's goes with its artefacts.
        """
        if not self.host_id:
            raise JobStoreError("a job can only be claimed by a store that knows its host")
        fd = os.open(self.paths(job_id).claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump({"host": self.host_id, "pid": os.getpid(),
                       "at": (now or utcnow()).isoformat(), "for": purpose}, out)

    def is_claimed(self, job_id: str) -> bool:
        return self.paths(job_id).claim.exists()

    def read_claim(self, job_id: str) -> SpawnClaim | None:
        """The claim on ``job_id``, or None if there is none."""
        path = self.paths(job_id).claim
        try:
            raw = path.read_text(encoding="utf-8")
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return None
        except OSError:
            # It is there and cannot be read (a share hiccup): claimed, by
            # someone, at a time nobody can vouch for — so never judged stale.
            return SpawnClaim(host=None, pid=None, at=None)
        try:
            data = json.loads(raw)
            return SpawnClaim(host=str(data["host"]), pid=int(data["pid"]),
                              at=datetime.fromisoformat(data["at"]),
                              purpose=str(data.get("for", CLAIM_SPAWN)))
        except (ValueError, KeyError, TypeError):
            # Created, content not written yet — or never, if the writer died in
            # between. The file's own time is the best there is.
            return SpawnClaim(host=None, pid=None,
                              at=datetime.fromtimestamp(mtime, tz=timezone.utc))

    # ── layout ──────────────────────────────────────────────────────────────
    def paths(self, job_id: str) -> JobPaths:
        if not _JOB_ID_RE.match(job_id):
            raise JobStoreError(f"malformed job id: {job_id!r}")
        return JobPaths(self.root / job_id)

    def new_job_id(self, model_id: str, now: str | None = None) -> str:
        """Sortable, readable, unique: ``<utc>-<model_id>``."""
        stamp = now or utcnow().strftime("%Y%m%dT%H%M%SZ")
        job_id = f"{stamp}-{model_id}"
        n = 2
        while (self.root / job_id).exists():
            job_id = f"{stamp}-{model_id}-{n}"
            n += 1
        return job_id

    # ── CRUD ────────────────────────────────────────────────────────────────
    def create(self, request: TrainRequest, job_id: str | None = None,
               host: str | None = None) -> TrainJob:
        """Write a new ``queued`` record, belonging to ``host`` (default: this store's).

        A new record without a host is refused: it would read as a legacy record
        and belong to ``legacy_host`` — a job idhefix never saw, attributed to it.
        """
        host = host or self.host_id
        if not host:
            raise JobStoreError(
                "a new job needs a host: open the store with host_id, or pass host= "
                "(a record without one is read as a legacy idhefix record, #15)")
        job_id = job_id or self.new_job_id(request.model_id)
        paths = self.paths(job_id)
        if paths.job_json.exists():
            raise JobStoreError(f"job {job_id} already exists")
        paths.mkdirs()
        # Which code accepted the job (#147). Imported here, not at module level:
        # codeversion imports contracts, and the store is imported by everything.
        from atr_training.codeversion import current_code
        job = TrainJob(id=job_id, request=request, status="queued", host=host,
                       code=current_code())
        self.save(job)
        return job

    def save(self, job: TrainJob) -> TrainJob:
        """Atomically replace ``job.json`` (tmp file + ``os.replace``).

        A half-written record read by a concurrent ``GET /jobs`` would look like
        a corrupt job; ``os.replace`` is atomic within a filesystem, so a reader
        sees either the old record or the new one.
        """
        paths = self.paths(job.id)
        paths.root.mkdir(parents=True, exist_ok=True)
        job.updated_at = utcnow()
        tmp = paths.job_json.with_suffix(".json.tmp")
        tmp.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, paths.job_json)
        return job

    def load(self, job_id: str) -> TrainJob:
        paths = self.paths(job_id)
        if not paths.job_json.exists():
            raise JobStoreError(f"no such job: {job_id}")
        try:
            return TrainJob.model_validate_json(paths.job_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError) as exc:
            raise JobStoreError(f"job {job_id} has an unreadable job.json: {exc}") from exc

    def list_ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        ids = [p.name for p in self.root.iterdir()
               if p.is_dir() and _JOB_ID_RE.match(p.name) and (p / "job.json").exists()]
        return sorted(ids, reverse=True)  # newest first (ids are timestamp-prefixed)

    def list(self, limit: int | None = None) -> list[TrainJob]:
        """Every job record, newest first; ``limit`` keeps only the N newest.

        The slice happens on ``list_ids()`` — **before** any ``job.json`` is
        loaded — because the ids are sorted, so the newest N are simply the
        first N of that list. Loading the rest of a store that has hundreds of
        jobs to answer a "what is the queue?" question is the cost this exists
        to remove (#38, serving-atr-inference#107).
        """
        jobs = []
        for job_id in self.list_ids()[:limit]:
            try:
                jobs.append(self.load(job_id))
            except JobStoreError:
                continue  # a corrupt record must not break the whole listing
        return jobs

    def delete(self, job_id: str, keep: Iterable[str] = ()) -> None:
        """Remove a job's artifacts. ``keep`` names top-level entries to spare."""
        import shutil

        paths = self.paths(job_id)
        if not paths.root.is_dir():
            raise JobStoreError(f"no such job: {job_id}")
        keep_set = set(keep)
        for child in paths.root.iterdir():
            if child.name in keep_set:
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
        if not keep_set:
            paths.root.rmdir()

    # ── lifecycle ───────────────────────────────────────────────────────────
    @staticmethod
    def can_transition(current: JobStatus, target: JobStatus) -> bool:
        return target in TRANSITIONS.get(current, frozenset())

    def advance(self, job: TrainJob, target: JobStatus) -> TrainJob:
        """Move a job to ``target``, refusing anything off the lifecycle.

        ``completed`` additionally requires a parsed CER: a run whose metrics we
        could not read is a failure, not a success with a blank score. This is
        the same rule as #21 on the recognition side — an empty result must never
        be indistinguishable from a real one.
        """
        if not self.can_transition(job.status, target):
            raise IllegalTransition(f"{job.id}: {job.status} → {target} is not a legal transition")
        if target == "completed" and (job.metrics is None
                or (job.metrics.cer is None and job.metrics.benchmark_cer is None)):
            raise JobStoreError(
                f"{job.id}: refusing to complete without a parsed CER — a run whose "
                "ketos test report could not be read is a failure, not a success"
            )
        job.status = target
        job.stage = next((s for s, st in STAGE_STATUS.items() if st == target), None)
        if job.started_at is None and target not in TERMINAL_STATUSES:
            job.started_at = utcnow()
        if target in TERMINAL_STATUSES:
            job.finished_at = utcnow()
        return self.save(job)

    def fail(self, job: TrainJob, error: str, log_tail: list[str] | None = None) -> TrainJob:
        """Terminate a job as ``failed`` with a non-empty reason."""
        if not error or not error.strip():
            raise JobStoreError("a failed job needs a reason")
        job.error = error.strip()
        if log_tail:
            job.log_tail = list(log_tail)[-50:]
        return self.advance(job, "failed")

    def reconcile(
        self, job: TrainJob, is_alive: Callable[[int], bool] = _pid_alive,
        now: datetime | None = None,
    ) -> TrainJob:
        """Bring a record in line with reality after a service restart.

        A non-terminal job whose runner process is gone did not finish — it was
        killed (OOM, reboot, ``systemctl restart`` of the wrong thing). Mark it
        failed rather than leaving it "training" forever.

        **Only this host's jobs.** Another host's job is returned untouched: its
        pid cannot be confirmed or refuted from here. Judging it anyway is how two
        trainers on one store declared each other's runs dead on every tick — or,
        where the number happened to be taken locally, kept a dead run alive (#15).

        A ``queued`` job is only reconciled once it has a **pid**: that is what
        distinguishes "waiting its turn" (nothing to reconcile — it is supposed to
        sit there) from "spawned, but the runner died before writing its first
        status". The latter would otherwise stay queued forever while the
        scheduler counted it as running. The one exception is a job claimed for
        spawning that never got a pid (:meth:`_reconcile_claim`).
        """
        if job.is_terminal:
            return job
        if not self.owns(job):
            if job.id not in self._foreign_noted:
                self._foreign_noted.add(job.id)
                logger.info("job {} ({}) belongs to host {}, not {}: not reconciled here",
                            job.id, job.status, self.host_of(job), self.host_id)
            return job
        if job.status == "queued" and job.pid is None:
            return self._reconcile_claim(job, now or utcnow())
        if job.pid is not None and is_alive(job.pid):
            return job
        # After the liveness check, not before: a runner writes its terminal
        # status and then exits, so a read taken once the pid is dead sees that
        # write. The copy passed in may be older — the scheduler lists every
        # record before it judges any — and failing it would replace the
        # `completed` (or `cancelled`) the runner wrote a moment ago with
        # "runner process N is gone": the false "failed" #15 is about, from the
        # host's own tick (#15 review, reproduced with a real child process).
        if (newer := self._newer_on_disk(job)) is not None:
            return newer
        reason = (
            f"runner process {job.pid} is gone while the job was {job.status}"
            if job.pid is not None
            else f"job was {job.status} but no runner pid was recorded"
        )
        return self.fail(job, f"{reason}; see logs/ in the job directory")

    def _newer_on_disk(self, job: TrainJob) -> TrainJob | None:
        """The record on disk if it changed since ``job`` was read, else None.

        Every save stamps ``updated_at``, so a different stamp is a later write.
        A record that cannot be read right now comes back as ``job`` itself:
        nothing is judged on a copy that cannot be confirmed, and the next tick
        asks again.
        """
        try:
            fresh = self.load(job.id)
        except JobStoreError:
            return job
        return fresh if fresh.updated_at != job.updated_at else None

    def _reconcile_claim(self, job: TrainJob, now: datetime) -> TrainJob:
        """Fail a queued job whose claim has stood too long without a pid.

        The scheduler claims, spawns, then saves the pid, and the runner saves its
        own pid first thing. A claim older than ``claim_orphan_after_s`` with no
        pid means the scheduler died in between, or the runner died before it
        could write. Such a job is never offered to the queue again, so without
        this it would stay ``queued`` for ever and hold a slot.

        A claim naming another host is left alone, for the reason foreign jobs
        are. One with no readable owner is this host's: only an owner claims.

        A **cancel** claim on a job still queued is a cancel whose write was lost
        (a tick saved its older copy over it, or the save failed after the
        claim). Nobody will start the job — the claim is taken — so the cancel is
        completed here, at once, rather than failed ten minutes later under a
        message about a scheduler that never touched it.
        """
        claim = self.read_claim(job.id)
        if claim is None or claim.at is None:
            return job
        if claim.host is not None and claim.host != self.host_id:
            return job
        if claim.purpose == CLAIM_CANCEL:
            if (newer := self._newer_on_disk(job)) is not None:
                return newer
            job.error = job.error or "cancelled before it started"
            return self.advance(job, "cancelled")
        if (now - claim.at).total_seconds() < self.claim_orphan_after_s:
            return job
        # The pid may have been saved after this copy was read (see reconcile).
        if (newer := self._newer_on_disk(job)) is not None:
            return newer
        by = (f"by {claim.host} (service pid {claim.pid})" if claim.host
              else "by this host (the claim file is empty)")
        return self.fail(job, (
            f"claimed for spawning {by} at {claim.at:%Y-%m-%d %H:%M:%S} UTC, but no "
            "runner pid was ever recorded: the scheduler stopped between claiming the "
            "job and starting its runner, or the runner died before its first write "
            "(see logs/runner.out, if it exists). A claimed job is never started "
            "again — resubmit it"))

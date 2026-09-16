"""Close another host's job record by hand, once that host no longer runs it.

    cd <checkout> && PYTHONPATH=src python -m atr_training.close_job <job_id> \\
        --reason "ps -p 1843 on idhefix: no such process"       # shows, writes nothing
    ... --yes                                                    # writes

Since #15 no trainer judges, starts or signals another host's job, and nothing
else in the code changes such a record. That rule is what stops two trainers
from declaring each other's runs dead, and it leaves one case to a person: the
owning host is gone. At the cutover the old trainer on idhefix is disabled for
good; a legacy record it left ``queued``, or a run of it that dies after the
stop, stays live for ever. Cancel and DELETE answer 409 "cancel it there", and a
resubmit of the model_id answers 409 "cancel that job first" — a circle, since
"there" no longer exists. A UBELIX job that was scancelled, or never ran, is the
same: on the preemptable template the runner reads SIGTERM as a preemption and
leaves the record in ``training`` (#15 review).

What it does: a job that never started (``queued``, no pid) becomes
``cancelled``, anything else ``failed``, and the error names who closed it, on
which host, and the reason given. What it never does: send a signal. The pid in
the record is the other machine's, and here it is a stranger's or nobody's.

It refuses:

* a job of THIS host — the service cancels those itself, and signals the runner;
* without ``ATR_TRAIN_HOST_ID`` set — "this host" would be the bare hostname,
  and the check above would pass for this host's own jobs. Run it from the
  checkout, whose ``.env`` names the host;
* a queued job another scheduler has claimed for spawning — its host is not gone;
* a record that changed while it ran — something is still writing it.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from pydantic import ValidationError

from atr_training.jobstore import (
    CLAIM_CANCEL,
    CLAIM_CLOSE,
    SLURM_HOST,
    JobStore,
    JobStoreError,
)
from atr_training.settings import REPO_ROOT, TrainerSettings

__all__ = ["close_command", "close_job", "main"]

#: Exit codes: done (or nothing to do), refused, bad invocation.
EXIT_OK, EXIT_REFUSED, EXIT_USAGE = 0, 1, 2


def close_command(job_id: str) -> str:
    """The command that closes ``job_id``, for a 409 that has to name a way out.

    Spelled with this interpreter and this checkout, so it can be pasted as it
    stands on this machine (the same reasoning as
    :func:`atr_training.registration.manual_registration`). The ``cd`` is not
    decoration: the host id comes from the checkout's ``.env``. Without
    ``--yes`` it only shows what it would do.
    """
    return (f"cd {shlex.quote(str(REPO_ROOT))} && PYTHONPATH=src "
            f"{shlex.quote(sys.executable)} -m atr_training.close_job {job_id} "
            "--reason '<how you know the host no longer runs it>'")


def close_job(store: JobStore, job_id: str, reason: str, *, write: bool,
              out=print) -> int:
    """Close ``job_id`` in ``store`` (or only describe it). Returns an exit code."""
    if not reason.strip():
        out("--reason must say how you know the host no longer runs the job")
        return EXIT_USAGE
    try:
        job = store.load(job_id)
    except JobStoreError as exc:
        out(str(exc))
        return EXIT_REFUSED
    host = store.host_of(job)
    if job.is_terminal:
        out(f"job {job_id} is already {job.status}; nothing to close")
        return EXIT_OK
    if store.owns(job):
        out(f"job {job_id} is this host's ({host}). Cancel it through the service — "
            f"POST /jobs/{job_id}/cancel — which signals its runner if it has one; its "
            "own scheduler fails it if the runner is gone.")
        return EXIT_REFUSED

    unstarted = job.status == "queued" and job.pid is None
    target = "cancelled" if unstarted else "failed"
    who = "the Slurm job" if host == SLURM_HOST else f"host {host}"
    error = (f"closed by hand on {store.host_id}: {who} no longer runs it "
             f"({reason.strip()}). Nothing was signalled.")
    legacy = " (no host in the record: a legacy record)" if job.host is None else ""
    out(f"job      {job.id}\n"
        f"host     {host}{legacy}\n"
        f"status   {job.status} (stage {job.stage}, pid {job.pid} on {host})\n"
        f"updated  {job.updated_at:%Y-%m-%d %H:%M:%S} UTC\n"
        f"becomes  {target}: {error}")
    if not write:
        out("Nothing written. Check that the host really no longer runs it, then add --yes.")
        return EXIT_OK

    claimed = False
    if unstarted:
        # The owner's scheduler claims before it spawns; taking the claim first
        # means it can no longer start the job while this closes it — and if it
        # already has, this is the place to find out.
        try:
            store.claim(job.id, purpose=CLAIM_CLOSE)
            claimed = True
        except FileExistsError:
            claim = store.read_claim(job.id)
            if claim is None or claim.purpose not in (CLAIM_CLOSE, CLAIM_CANCEL):
                by = f"{claim.host} at {claim.at}" if claim and claim.host else "someone"
                out(f"not closed: job {job.id} was claimed for spawning by {by}. A host "
                    "that is starting the job is not gone.")
                return EXIT_REFUSED
            # An earlier close or cancel of this job that did not finish: carry on.

    fresh = store.load(job.id)
    if fresh.updated_at != job.updated_at:
        if claimed:
            store.paths(job.id).claim.unlink(missing_ok=True)
        out(f"not closed: job {job.id} changed while this ran (now {fresh.status}, "
            f"updated {fresh.updated_at:%H:%M:%S} UTC). Something is still writing it.")
        return EXIT_REFUSED
    if target == "failed":
        closed = store.fail(fresh, error)
    else:
        fresh.error = error
        closed = store.advance(fresh, "cancelled")
    out(f"job {closed.id} is now {closed.status}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Close another host's job record once that host no longer runs it. "
                    "Sends no signal. Without --yes, only shows what it would write.")
    parser.add_argument("job_id")
    parser.add_argument("--reason", required=True,
                        help="how you know the host no longer runs it; goes into the record")
    parser.add_argument("--yes", action="store_true", help="write the record")
    parser.add_argument("--root", type=Path,
                        help="the job store (default: ATR_TRAIN_JOBS_ROOT)")
    args = parser.parse_args(argv)
    try:
        settings = TrainerSettings()
    except ValidationError as exc:
        print(f"the settings do not load: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if "host_id" not in settings.model_fields_set:
        print("ATR_TRAIN_HOST_ID is not set, so this host's own jobs cannot be told "
              "from another's. Run this from the checkout, whose .env names the host.",
              file=sys.stderr)
        return EXIT_USAGE
    store = JobStore(args.root or settings.jobs_root, host_id=settings.host_id,
                     legacy_host=settings.legacy_job_host)
    return close_job(store, args.job_id, args.reason, write=args.yes)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

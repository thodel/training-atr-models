"""A record for a measurement run by hand, so the daily report can see it (#385).

``scripts/measure_*.py`` run outside the trainer: on asteraix by hand, for hours
on a card at times, and until #385 they left nothing in any store. ``live()``
and ``results()`` of the atr-results MCP read job records, so such a run was
invisible while it ran and its output lived only in an issue comment.

Each measure script now wraps its ``main`` in :func:`measurement`, which writes
``<root>/<id>/measurement.json`` once at the start (``status: running``) and once
at the end (``completed``, ``failed`` or ``aborted``, with ``ended_at`` and the
exit code). The root sits next to the job store, not in it: a measurement has no
``request`` and no runner, and the service's reconcile must not try to judge it.

* asteraix: ``ATR_TRAIN_JOBS_ROOT`` is ``…/training_folder/jobs`` on the share,
  so the records land in ``…/training_folder/measurements``.
* UBELIX: the sbatch scripts set ``ATR_TRAIN_JOBS_ROOT`` to scratch, so a
  measurement there lands in ``/scratch/…/runs/measurements``.
* ``ATR_MEASUREMENTS_ROOT`` overrides both.

The scripts are run from a shell that has not sourced ``.env``, so the keys this
module reads fall back to the checkout's ``.env``: the first record on asteraix
(06.11.2026) named its host ``dhserver03`` and ``live(host="asteraix")`` missed it.

A record without ``ended_at`` is a run that is still going **or** one that was
killed too hard to say so (``kill -9``, a lost node): the reader shows it as
running with its start time, never as finished. SIGTERM is turned into an exit,
so a plain ``kill`` still ends the record as ``aborted``.

Writing the record never fails the measurement. A store that cannot be written
is said once on stderr, and the measurement runs as before.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping, Sequence

MEASUREMENTS_DIRNAME = "measurements"
RECORD_NAME = "measurement.json"

#: The asteraix job store when no environment says otherwise: the measure
#: scripts are often run without sourcing ``.env``.
DEFAULT_JOBS_ROOT = Path("/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/jobs")

#: The checkout's ``.env``, read for :data:`ENV_KEYS` when the shell lacks them.
DOTENV = Path(__file__).resolve().parents[2] / ".env"
ENV_KEYS = ("ATR_MEASUREMENTS_ROOT", "ATR_TRAIN_JOBS_ROOT", "ATR_TRAIN_HOST_ID")

#: The flags whose value is where a measure script writes its result.
OUTPUT_FLAGS = ("--out", "--output", "--report")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def dotenv_values(path: Path) -> dict[str, str]:
    """``KEY=VALUE`` lines of ``path`` for :data:`ENV_KEYS`; nothing if unreadable."""
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and key.strip() in ENV_KEYS:
            values[key.strip()] = value.strip().strip("'\"")
    return values


def process_env(dotenv: Path | None = None) -> dict[str, str]:
    """The environment, with the checkout's ``.env`` under it for :data:`ENV_KEYS`."""
    return {**dotenv_values(DOTENV if dotenv is None else dotenv), **os.environ}


def measurements_root(env: Mapping[str, str] | None = None) -> Path | None:
    """Where records go, or None when there is no store to write to."""
    env = process_env() if env is None else env
    if env.get("ATR_MEASUREMENTS_ROOT"):
        return Path(env["ATR_MEASUREMENTS_ROOT"]).expanduser()
    jobs = Path(env["ATR_TRAIN_JOBS_ROOT"]).expanduser() if env.get("ATR_TRAIN_JOBS_ROOT") \
        else DEFAULT_JOBS_ROOT
    return jobs.parent / MEASUREMENTS_DIRNAME if jobs.is_dir() else None


def host_id(env: Mapping[str, str] | None = None) -> str:
    """The host a record names: the trainer's own id where one is set."""
    env = process_env() if env is None else env
    return env.get("ATR_TRAIN_HOST_ID") or socket.gethostname().split(".")[0]


def output_from_argv(argv: Sequence[str]) -> str | None:
    """The value of the first output flag in ``argv``, made absolute, or None."""
    for i, arg in enumerate(argv):
        for flag in OUTPUT_FLAGS:
            value = None
            if arg == flag and i + 1 < len(argv):
                value = argv[i + 1]
            elif arg.startswith(flag + "="):
                value = arg[len(flag) + 1:]
            if value:
                return str(Path(value).expanduser().resolve())
    return None


def _write(path: Path, record: dict) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError as exc:
        print(f"measurement record not written ({path}): {exc}", file=sys.stderr)
        return False


def _exit_code(exc: BaseException | None) -> int:
    if exc is None:
        return 0
    if isinstance(exc, SystemExit):
        code = exc.code
        if code is None:
            return 0
        return code if isinstance(code, int) else 1
    if isinstance(exc, KeyboardInterrupt):
        return 130
    return 1


def _status(exc: BaseException | None, code: int) -> str:
    if isinstance(exc, KeyboardInterrupt) or code in (130, 128 + signal.SIGTERM):
        return "aborted"
    return "completed" if code == 0 else "failed"


def _sigterm_to_exit(signum, _frame) -> None:
    raise SystemExit(128 + signum)


@contextmanager
def measurement(name: str, argv: Sequence[str] | None = None,
                root: Path | None = None, env: Mapping[str, str] | None = None,
                now=utcnow) -> Iterator[dict]:
    """Record the run of measure script ``name`` around the ``with`` body.

    Yields the record; a script may add fields (``record["note"] = …``), which
    are written with the end stamp.
    """
    env = process_env() if env is None else env
    argv = list(sys.argv if argv is None else argv)
    root = measurements_root(env) if root is None else root
    started = now()
    record = {
        "kind": "measurement",
        "id": f"{started.strftime('%Y%m%dT%H%M%SZ')}-{name}-{uuid.uuid4().hex[:6]}",
        "name": name, "host": host_id(env), "pid": os.getpid(),
        "status": "running", "started_at": started.isoformat(), "ended_at": None,
        "gpus": env.get("CUDA_VISIBLE_DEVICES"),
        "command": " ".join(argv), "cwd": os.getcwd(),
        "output": output_from_argv(argv[1:]), "exit_code": None,
    }
    path = root / record["id"] / RECORD_NAME if root is not None else None
    if path is not None and not _write(path, record):
        path = None

    previous = None
    if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
        try:
            previous = signal.signal(signal.SIGTERM, _sigterm_to_exit)
        except ValueError:  # not the main thread
            previous = None
    error: BaseException | None = None
    try:
        yield record
    except BaseException as exc:
        error = exc
        raise
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
        code = _exit_code(error)
        record.update(status=_status(error, code), ended_at=now().isoformat(),
                      exit_code=code)
        if error is not None and not isinstance(error, (SystemExit, KeyboardInterrupt)):
            record["error"] = f"{type(error).__name__}: {error}"[:500]
        if path is not None:
            _write(path, record)

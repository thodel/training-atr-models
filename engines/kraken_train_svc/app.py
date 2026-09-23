"""Training service (:8204) — supervises every training backend.

Supervision only: it never trains in-process, and never imports an engine
package. Submitting a job writes a record, and a scheduler loop starts it as a
**detached** child *of that engine's interpreter* when the GPU is free and
nothing else is running. State lives in the job directory, so a restart of this
service reconciles against reality instead of losing (or killing) a run.

One service for both backends is a deliberate choice about the GPU rather than
about tidiness — see :mod:`atr_training.backends`. The package is still
named ``kraken_train_svc`` because kraken was the first backend; the service is
``atr-train`` and the API is engine-agnostic.

Endpoints mirror what the gateway proxies in #35:

    POST   /jobs              submit            → 202 {job_id}
    POST   /jobs/verify       check a dataset spec, queue nothing
    GET    /jobs              list
    GET    /jobs/{id}         one record
    GET    /jobs/{id}/log     tail a stage log
    GET    /jobs/{id}/curve   per-epoch metrics, live while training
    POST   /jobs/{id}/cancel  SIGTERM the process group (this host's jobs only)
    DELETE /jobs/{id}         drop artifacts (never the registered model)
    GET    /gpu               this machine's cards and who holds them
    GET    /health            the only route without a key

Every route but ``/health`` needs ``X-API-Key`` and every caller must be
loopback or in ``ATR_TRAIN_ALLOWED_CLIENTS`` (#13, :mod:`atr_training.access`).
Start it with ``python -m atr_training.serve``, which refuses an unsafe bind.

The job store is on the share and may hold other hosts' jobs (#15). They are
listed and readable here like any other; this service spawns, reconciles,
signals and counts only the jobs stamped with its own ``ATR_TRAIN_HOST_ID``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Literal, get_args

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from loguru import logger

from atr_training import gpu as gpu_probe
from atr_training.preflight import free_disk_gb
from atr_training.access import AccessGuard, key_refusal
from atr_training.shared_registry import RegistryUnavailable, load_shared_registry
from atr_training.base_models import BaseModelError, resolve_base_model
from atr_training.backends import BACKENDS, UnknownBackend, backend_for
from atr_training.contracts import TrainJob, TrainRequest
from atr_training.curves import (
    CURVE_FILENAME,
    curve_from_checkpoints,
    curve_payload,
    empty_curve,
)
from atr_training.hf_source import (
    DatasetSelectionError,
    VerificationUnavailable,
    verify_dataset_spec,
)
from atr_training.close_job import close_command
from atr_training.jobstore import (
    CLAIM_CANCEL,
    SLURM_HOST,
    JobStore,
    JobStoreError,
    reap_children,
)

from atr_training.preflight import (
    PreflightError,
    check_datasets_cache,
    check_disk,
    check_tmpdir,
    check_vram,
    query_gpus,
)
from atr_training.registration import (
    SUFFIX as REGISTRATION_SUFFIX,
    RegistrationError,
    is_registration_name,
    read_registration,
    trained_dir,
)
from atr_training.runner_base import tail
from atr_training.settings import TrainerSettings, get_settings

RUNNING_STATUSES = ("preparing", "compiling", "training", "testing", "registering")

#: The engines ``POST /jobs`` accepts, read off the request model itself so
#: ``/health`` cannot advertise a list of its own. The gateway builds its engine
#: check from this (serving#137) instead of importing ``BACKENDS``, which after
#: the split would be another repository's code.
ENGINES: tuple[str, ...] = tuple(sorted(get_args(TrainRequest.model_fields["engine"].annotation)))


# ── wiring (overridable in tests via app.state) ─────────────────────────────
def _settings() -> TrainerSettings:
    return getattr(app.state, "settings", None) or get_settings()


def _store() -> JobStore:
    store = getattr(app.state, "store", None)
    if store is None:
        settings = _settings()
        store = JobStore(settings.jobs_root, host_id=settings.host_id,
                         legacy_host=settings.legacy_job_host)
        app.state.store = store
    return store


def _registry():
    """``(registry, why_not)`` for base_model lookups and the curated-id check.

    The registry is the file the gateway publishes to the share (#5). Unreadable
    means ``(None, reason)``, not an error here: a job naming a DOI has no
    business being refused because the registry is missing. The reason travels
    on, so a job naming an *id* is told the registry was the problem rather
    than its spelling.

    Overridable on ``app.state`` so tests need no file on disk.
    """
    if (override := getattr(app.state, "registry", None)) is not None:
        return override, None
    try:
        return load_shared_registry(_settings().models_config), None
    except RegistryUnavailable as exc:
        logger.warning("curated registry unavailable: {}", exc)
        return None, str(exc)


def _spawn(settings: TrainerSettings, job: TrainJob) -> int:
    """Start the job's runner detached, in its engine's venv, and return its pid.

    ``start_new_session=True`` puts it in its own process group: it survives a
    restart of this service, and cancelling it kills the group (runner + the
    trainer subprocess it drives) rather than orphaning the child.
    """
    backend = backend_for(job.request.engine)
    python = settings.runner_python(job.request.engine)
    if not python.exists():
        # Better here than as a traceback inside a detached child: a box that only
        # trains kraken has no reason to have built the VLM venv, and the fix is
        # one documented command.
        raise PreflightError(
            f"no interpreter at {python} — the {backend.venv} venv has not been "
            f"built on this box. Run:  bash scripts/make_venvs.sh {backend.venv}"
        )
    cmd = [str(python), "-m", backend.runner_module,
           "--root", str(settings.jobs_root), "--job-id", job.id]
    env = {**os.environ, **settings.env_for_child()}
    log = _store().paths(job.id).logs / "runner.out"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as out:
        proc = subprocess.Popen(  # noqa: S603
            cmd, stdout=out, stderr=subprocess.STDOUT, env=env,
            cwd=str(Path(__file__).resolve().parents[1]), start_new_session=True,
        )
    logger.info("spawned runner pid={} for job {}", proc.pid, job.id)
    return proc.pid


def schedule_once(
    store: JobStore, settings: TrainerSettings, spawn=_spawn, vram_check=check_vram
) -> TrainJob | None:
    """Reconcile records, then start the oldest queued job if the box allows it.

    Returns the job that was started, or None. Reasons for *not* starting are
    written to ``queued_reason`` — a queued job is not a failed job, and the
    caller deserves to know whether it is waiting on the GPU or on another run.
    """
    # Before judging liveness: a finished runner stays defunct until someone waits
    # on it, and a defunct pid used to read as alive (#118). `_pid_alive` no longer
    # believes a zombie, so this is hygiene rather than correctness — but a process
    # table that fills with dead runners is its own problem.
    reaped = reap_children()
    if reaped:
        logger.debug("reaped {} finished runner(s)", reaped)

    jobs = [store.reconcile(j) for j in store.list()]
    # Only this host's jobs, for the count as much as for the queue (#15):
    # max_concurrent is about THIS box's card, and another host's run neither
    # occupies it nor is ours to start. Nor are their records ours to write —
    # hold() below saves every job it considers.
    mine = [j for j in jobs if store.owns(j)]
    unspawned = [j for j in mine if j.status == "queued" and j.pid is None]
    # Claimed but no pid yet: a scheduler of this host is starting it right now,
    # between claim and pid save. It holds a slot, and it is not offered again —
    # saving its record here could land after that pid save and erase it. A claim
    # that never gets its pid is failed by reconcile above, so this cannot hold a
    # slot for longer than CLAIM_ORPHAN_AFTER_S.
    starting = {j.id for j in unspawned if store.is_claimed(j.id)}
    # A job stays "queued" from the moment it is spawned until its detached runner
    # writes the first status — a window that a second submit lands in easily,
    # since submitting schedules immediately. A queued job with a live pid has
    # therefore already been started, and starting it again would put two runners
    # on one job directory and one GPU.
    running = [j for j in mine
               if j.status in RUNNING_STATUSES or (j.status == "queued" and j.pid is not None)
               or j.id in starting]
    queued = sorted([j for j in unspawned if j.id not in starting],
                    key=lambda j: j.created_at)
    if not queued:
        return None

    def hold(reason: str) -> None:
        # From a fresh read, never from the listing. `starting` covers only the
        # claims that existed when this tick listed; a scheduler of this host
        # that claimed, spawned and saved the pid since (the submit's pass runs on
        # the event loop while the tick runs in a thread) would otherwise have
        # that pid erased by this save, and ten minutes later the live run was
        # failed as "no runner pid was ever recorded" and a second job started
        # on the card (#15 review, reproduced). The same stale save turned a
        # cancel back into `queued`. The claim is checked AFTER the read. A claim
        # taken later still has to spawn and save the pid, so this save can only
        # overtake that one by milliseconds — and the runner saves its own pid as
        # its first act, seconds later, which repairs it.
        for listed in queued:
            if listed.queued_reason == reason:
                continue
            try:
                job = store.load(listed.id)
            except JobStoreError:
                continue
            if (job.status != "queued" or job.pid is not None
                    or job.queued_reason == reason or store.is_claimed(job.id)):
                continue
            job.queued_reason = reason
            store.save(job)

    if len(running) >= settings.max_concurrent:
        first = running[0]
        hold(f"waiting for {first.id} "
             f"({'starting' if first.id in starting else first.status})")
        return None

    # The oldest queued job goes first — no reordering to fit a smaller job into
    # the free VRAM, which would starve exactly the expensive runs the queue
    # exists for. How much VRAM is "enough" depends on the engine.
    job = queued[0]
    try:
        gpu = vram_check(settings.gpu, settings.min_free_vram_for(job.request.engine))
    except PreflightError as exc:
        hold(str(exc))
        return None

    # The claim is what makes "listed as unspawned" and "started by us" one
    # step: whoever creates the file spawns, everyone else stands aside.
    try:
        store.claim(job.id)
    except FileExistsError:
        logger.info("{} was claimed by another scheduler since this tick listed it; "
                    "leaving the start to that one", job.id)
        return None
    except OSError as exc:
        hold(f"could not claim {job.id} for spawning: {exc}")
        return None
    # The claim is ours: from here no scheduler or cancel of this host writes
    # the record. The listing may still predate a write made before the claim,
    # or without one — the old trainer on idhefix spawning or cancelling it.
    # Spawn from what is on disk, and only if it is still waiting to start.
    try:
        job = store.load(job.id)
    except JobStoreError as exc:
        logger.warning("{} was claimed but cannot be read back, not starting it: {}",
                       job.id, exc)
        return None
    if job.status != "queued" or job.pid is not None:
        logger.info("{} is {} (pid {}) since this tick listed it; not starting it",
                    job.id, job.status, job.pid)
        return None

    logger.info("starting {} ({} job; GPU {} has {} MB free)",
                job.id, job.request.engine, gpu.index, gpu.free_mb)
    job.queued_reason = None
    try:
        job.pid = spawn(settings, job)
    except (PreflightError, UnknownBackend, OSError) as exc:
        # A job that cannot be spawned will not spawn on the next tick either.
        # Failing it names the reason once, instead of logging it every 10 s
        # forever while the record still says "queued".
        logger.error("cannot start {}: {}", job.id, exc)
        return store.fail(job, f"could not start the {job.request.engine} runner: {exc}")
    # Record the pid on what is on disk now, not on the copy loaded before the
    # spawn. The runner is detached and saves its own pid and `preparing` as its
    # first act; saving the pre-spawn copy after that put the record back to
    # `queued` until the runner's next write — which in prepare can be an hour
    # away, and the bot shows `queued` for a run that is working (#15 review).
    try:
        current = store.load(job.id)
    except JobStoreError:
        current = job
    if current.pid is not None:
        return current              # the runner has written; its record stands
    current.pid, current.queued_reason = job.pid, None
    return store.save(current)


def _schedule() -> TrainJob | None:
    """Run one scheduling pass with whatever seams are installed on app.state."""
    return schedule_once(
        _store(), _settings(),
        spawn=getattr(app.state, "spawn", None) or _spawn,
        vram_check=getattr(app.state, "vram_check", None) or check_vram,
    )


async def _scheduler() -> None:  # pragma: no cover - timing loop
    while True:
        try:
            await asyncio.to_thread(_schedule)
        except Exception as exc:  # noqa: BLE001 — the loop must not die
            logger.error("scheduler tick failed: {}", exc)
        await asyncio.sleep(_settings().poll_interval_s)


def _newest_mtime(directory: Path) -> float:
    """The latest mtime of ``directory`` and its top-level entries.

    Both, and the maximum. Rewriting an entry in place moves the entry's mtime
    and not the directory's — a retried registration does exactly that
    (``mkdir(exist_ok=True)``, then ``copyfile`` over the old weights). Adding
    an entry moves the directory's, while a copy that keeps times leaves the
    entry old. An empty directory has only its own time. One level is enough:
    a registration writes at the top level.
    """
    newest = directory.stat().st_mtime
    with os.scandir(directory) as entries:
        for entry in entries:
            newest = max(newest, entry.stat(follow_symlinks=False).st_mtime)
    return newest


def _names_directory(job: TrainJob, directory: Path) -> bool:
    """Whether ``job`` will register into, or has registered into, ``directory``.

    Every runner registers into ``<trained_root>/<model_id>``; ``model_path`` is
    that directory or a file in it, depending on the engine.
    """
    if job.request.model_id == directory.name:
        return True
    if not job.model_path:
        return False
    path = Path(job.model_path)
    return path == directory or directory in path.parents


def _registered_weights(registry_root: Path | str) -> list[tuple[str, Path | None]] | None:
    """``(id, local_path)`` for every trained registration; None if unreadable.

    A registration that does not validate still names its directory through its
    file name — the id — so it keeps that one; its ``local_path`` cannot be
    trusted and is not used.
    """
    root = Path(registry_root)
    if not root.is_dir():
        return None
    try:
        names = [p.name for p in trained_dir(root).iterdir() if is_registration_name(p.name)]
    except FileNotFoundError:
        return []  # nothing trained has been registered yet
    except OSError as exc:
        logger.warning("the registry {} cannot be listed: {}", trained_dir(root), exc)
        return None
    found: list[tuple[str, Path | None]] = []
    for name in names:
        model_id = name[: -len(REGISTRATION_SUFFIX)]
        try:
            registration = read_registration(root, model_id)
        except RegistrationError as exc:
            logger.warning("registration {} is unusable ({}); keeping the directory named "
                           "after it all the same", model_id, exc)
            registration = None
        local = registration.local_path if registration is not None else None
        found.append((model_id, Path(local) if local else None))
    return found


def _registration_for(directory: Path, registered: list[tuple[str, Path | None]]) -> str | None:
    """The id of a registration that names ``directory``, by id or by local_path."""
    for model_id, local in registered:
        if model_id == directory.name:
            return model_id
        if local is not None and (local == directory or directory in local.parents):
            return model_id
    return None


def _cleanup_orphaned_weights(trained_root: Path | str, store: JobStore,
                              min_age_h: float, *, registry_root: Path | str,
                              now: float | None = None) -> int:
    """Remove weight directories a registration left behind unfinished.

    A registration writes ``metadata.json`` last, so a directory without it is
    either an orphan or a registration still under way — and ``trained_root`` is
    on the share, so the registration may be the other machine's (#15). Until
    then this removed every directory without the file on each DELETE, and
    would have deleted a model mid-registration on the other host with no word
    to anyone. A directory goes only if ALL hold:

    * it has no ``metadata.json``;
    * nothing in it changed for ``min_age_h`` hours — a registration in progress
      keeps writing, and this also covers one from a store this service does not
      read;
    * no job of ANY host in this store that is not terminal names it — a
      registration that stalls is still its job's, and a job that has not
      registered yet will write there;
    * no registration in ``registry_root`` names it, by id or by
      ``local_path``. A model registered by hand never gets ``metadata.json``:
      the curated-clash failure tells the operator to copy the weights to
      ``<trained_root>/<new_id>`` and run ``python -m atr_training.registration``,
      which writes only the YAML. Without this the next restart deleted those
      weights and left the gateway a registration pointing at nothing (#15
      review, reproduced).

    Every candidate kept is logged with the reason, and every removal. Anything
    that cannot be read keeps the directory: a wrong keep costs disk, a wrong
    removal costs a trained model.
    """
    trained_root = Path(trained_root)
    try:
        candidates = [entry for entry in trained_root.iterdir()
                      if entry.is_dir() and not (entry / "metadata.json").is_file()]
    except FileNotFoundError:
        return 0
    except OSError as exc:
        logger.warning("orphan cleanup skipped, {} is unreadable: {}", trained_root, exc)
        return 0
    if not candidates:
        return 0
    try:
        live = [job for job in store.list() if not job.is_terminal]
    except OSError as exc:
        logger.warning("orphan cleanup skipped, the job store is unreadable ({}); keeping {} "
                       "weights directories without metadata.json", exc, len(candidates))
        return 0
    registered = _registered_weights(registry_root)
    if registered is None:
        logger.warning("orphan cleanup skipped, the registry {} is unreadable; keeping {} "
                       "weights directories without metadata.json", registry_root,
                       len(candidates))
        return 0

    now = time.time() if now is None else now
    removed = 0
    for entry in candidates:
        owner = next((job for job in live if _names_directory(job, entry)), None)
        if owner is not None:
            logger.info("keeping {} (no metadata.json): job {} on host {} is {} and names it",
                        entry.name, owner.id, store.host_of(owner), owner.status)
            continue
        if (model_id := _registration_for(entry, registered)) is not None:
            logger.info("keeping {} (no metadata.json): the registration {} names it",
                        entry.name, model_id)
            continue
        try:
            age_h = (now - _newest_mtime(entry)) / 3600
        except OSError as exc:
            logger.info("keeping {} (no metadata.json): its age cannot be read: {}",
                        entry.name, exc)
            continue
        if age_h < min_age_h:
            logger.info("keeping {} (no metadata.json): changed {:.1f} h ago, under the "
                        "{:g} h a registration in progress is allowed", entry.name, age_h,
                        min_age_h)
            continue
        logger.warning("removing orphaned weights directory {}: no metadata.json, unchanged "
                       "for {:.1f} h, and no live job names it", entry.name, age_h)
        shutil.rmtree(entry, ignore_errors=True)
        removed += 1
    return removed


def _warn_without_gateway_key(settings: TrainerSettings) -> None:
    """Say at startup what an empty ``ATR_TRAIN_GATEWAY_API_KEY`` costs (#48).

    Nothing else does until a kraken run reaches ``register``: the trainer
    starts, ``/health`` is green, jobs are accepted, and the promotion gate then
    leaves the model disabled. On asteraix it was empty from the cutover on
    16.09 until 21.09 without anyone noticing."""
    if not settings.gateway_api_key:
        logger.warning("ATR_TRAIN_GATEWAY_API_KEY is not set: no promotion gate and no "
                       "evaluation through the gateway, so a trained kraken model registers "
                       "disabled (training-atr-models#48)")


async def lifespan(_app: FastAPI):  # pragma: no cover - process lifecycle
    settings = _settings()
    _warn_without_gateway_key(settings)
    settings.jobs_root.mkdir(parents=True, exist_ok=True)
    settings.trained_root.mkdir(parents=True, exist_ok=True)
    # A restart must not leave a killed job looking like it is still training.
    for job in _store().list():
        _store().reconcile(job)
    # Its docstring always said "on startup", but until #15 only DELETE called it,
    # so a registration that died stayed until someone deleted some job. It is
    # safe here only because of the conditions #15 added.
    _cleanup_orphaned_weights(settings.trained_root, _store(), settings.orphan_weights_min_age_h,
                              registry_root=settings.registry_root)
    task = asyncio.create_task(_scheduler())
    _app.state.scheduler = task
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


# /docs and /redoc off: Swagger UI fetches the schema without the key header, so
# behind the key they could only render an error. /openapi.json stays, keyed.
app = FastAPI(title="ATR Kraken Training Service", version="0.1.0", lifespan=lifespan,
              docs_url=None, redoc_url=None)
app.add_middleware(AccessGuard, settings=_settings)


# ── endpoints ───────────────────────────────────────────────────────────────
def _health_body(*, deep: bool = False) -> dict:
    settings = _settings()
    jobs = _store().list()
    try:
        gpus = [g.__dict__ for g in query_gpus()]
    except PreflightError as exc:
        gpus = [{"error": str(exc)}]

    body = {
        "status": "ok",
        # Which machine answered. The gateway on idhefix reads this across the
        # network now, and "ok" alone does not say from where.
        "host": socket.gethostname(),
        "engines": list(ENGINES),
        "available_engines": [e for e in ENGINES
                              if e in BACKENDS and settings.runner_python(e).exists()],
        "gpu": settings.gpu,
        "gpus": gpus,
        "jobs_root": str(settings.jobs_root),
        # Which backends this box can actually run, not which ones exist in code:
        # a venv that was never built is the difference between a job that trains
        # and a job that fails at spawn (cf. #30/#31 — never advertise what the
        # host cannot run).
        "backends": {
            engine: {
                "runner": backend.runner_module,
                "venv": str(settings.runner_python(engine).parents[1]),
                "available": settings.runner_python(engine).exists(),
                "min_free_vram_mb": settings.min_free_vram_for(engine),
            }
            for engine, backend in BACKENDS.items()
        },
        "jobs": {"total": len(jobs),
                 "running": len([j for j in jobs if j.status in RUNNING_STATUSES]),
                 "queued": len([j for j in jobs if j.status == "queued"])},
        "gateway_auth_configured": bool(settings.gateway_api_key),
    }

    if deep and settings.gateway_api_key:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(
                    f"{settings.gateway_url.rstrip('/')}/models",
                    headers={"X-API-Key": settings.gateway_api_key},
                )
            body["gateway_reachable"] = True
            body["gateway_models_status"] = resp.status_code
        except Exception as exc:
            body["gateway_reachable"] = False
            body["gateway_models_error"] = str(exc)

    return body


@app.get("/health")
async def health(request: Request, deep: bool = False) -> JSONResponse:
    """Liveness, and what this host can train. Open: the one route without a key.

    Off the event loop: it lists the job store on the share and shells out to
    nvidia-smi, and the gateway now asks it with a 5 s timeout (serving#137).

    ``?deep=1`` also checks that the gateway is reachable with the configured
    API key and reports the ``/models`` status code. Intended for operators who
    want to verify the promotion gate without running a full kraken job. It
    calls the gateway *with the gateway's key*, so unlike plain ``/health`` it
    needs the trainer's key (:func:`~atr_training.access.key_refusal`).
    """
    if deep and (refused := key_refusal(request.scope, _settings())) is not None:
        status, detail = refused
        return JSONResponse({"detail": detail}, status_code=status)
    return JSONResponse(await asyncio.to_thread(_health_body, deep=deep))


# HEAD is what `curl -I` and many probes send. Same answer, same exemption
# (atr_training.access), and not a second entry in the schema.
app.add_api_route("/health", health, methods=["HEAD"], include_in_schema=False)


def _job_pids(store: JobStore, jobs) -> dict[int, str]:
    """pid -> job id for this host's runs that the store says are live.

    Live only. The gateway's version took every job with a pid, which was
    harmless while the store had one writer. On the share it does not: after the
    cutover this store holds the records idhefix wrote, each with an idhefix pid,
    and a local process that happens to reuse one would be reported as that old
    job's and dropped from ``unaccounted_mib`` — the silent failure #13 names.

    This host's only, for the same reason: a live job of another host carries
    that host's pid, which here is a stranger's or nobody's (#15). A
    non-terminal job of this host has a pid its scheduler reconciled as alive.
    """
    return {job.pid: job.id for job in jobs
            if job.pid and not job.is_terminal and store.owns(job)}


def _host_body() -> dict:
    """Disk and RAM for this machine, used by GET /host and by the gateway."""
    settings = _settings()
    volumes = [
        {"name": "jobs_root",      "path": str(settings.jobs_root)},
        {"name": "checkpoint_root","path": str(settings.checkpoint_root)},
        {"name": "trained_root",   "path": str(settings.trained_root)},
        {"name": "artefact_cache", "path": str(settings.artefact_cache_root)},
    ]
    disk = []
    for vol in volumes:
        try:
            usage = shutil.disk_usage(vol["path"])
            disk.append({
                "name": vol["name"],
                "path": vol["path"],
                "total_gb": round(usage.total / (1024**3), 1),
                "free_gb":  round(usage.free  / (1024**3), 1),
            })
        except OSError as exc:
            disk.append({"name": vol["name"], "path": vol["path"],
                         "error": str(exc)})

    # RAM: read MemTotal and MemAvailable from /proc/meminfo directly so a
    # monkeypatched test can substitute a tmp_path.  A missing or unreadable
    # /proc/meminfo is treated as "unknown" (None) rather than a 500, because
    # partial data is better than no data for an operator watching this endpoint.
    ram: dict = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("MemTotal:", "MemAvailable:")):
                    key = line.split(":")[0].strip().lower()
                    # value is in kB; store as MiB
                    ram[key] = int(line.split()[1]) // 1024
    except OSError:
        ram = {"error": "/proc/meminfo unreadable"}

    return {
        "host": socket.gethostname(),
        "disk": disk,
        "ram_mib": ram,
    }


@app.get("/host")
async def host() -> JSONResponse:
    """Disk free and RAM for this machine, for the gateway's /train/host proxy.

    Returns per-volume free space (jobs_root, checkpoint_root, trained_root,
    artefact_cache_root) and RAM (MemTotal / MemAvailable from /proc/meminfo).
    A volume that cannot be read appears with an "error" string instead of
    numbers, so the caller can distinguish "this volume is fine but empty" from
    "this volume is not readable".  A missing /proc/meminfo returns an error
    rather than a 500 so monitoring can still see the disk numbers.
    """
    return JSONResponse(await asyncio.to_thread(_host_body))


@app.get("/gpu")
async def gpu() -> dict:
    """This machine's cards, every process holding memory, and whose it is.

    The shape the gateway's ``/train/gpu`` has always returned, plus ``host``;
    the gateway now proxies here (serving#137) instead of reading its own cards.
    See :mod:`atr_training.gpu` for what each field is for.
    """
    attribution = True
    try:
        jobs = await asyncio.to_thread(_store().list)
    except OSError as exc:
        # An unreadable store must not hide the cards. Everything is then
        # unregistered, and the flag below says the attribution is missing.
        logger.warning("job store unreadable for /gpu, reporting the cards "
                       "unattributed: {}", exc)
        jobs, attribution = [], False
    job_pids = _job_pids(_store(), jobs)
    try:
        cards = await asyncio.to_thread(gpu_probe.inspect, job_pids)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — a wedged driver, a timeout
        raise HTTPException(
            status_code=502, detail=f"nvidia-smi failed: {type(exc).__name__}: {exc}",
        ) from exc
    # services_expected=False: nothing of ours on this box holds a card except
    # through a job, so a stray in atr-train.service is a finished run's leftover.
    rows = gpu_probe.card_rows(cards, services_expected=False)
    return {"host": socket.gethostname(), "cards": rows,
            "job_attribution_available": attribution, "known_job_pids": len(job_pids)}


#: Where the RAM figures come from. A module constant, not a literal in the
#: handler, so a test can substitute a fixture — including one that cannot be
#: read, which is the case a real /proc never produces (#40).
MEMINFO_PATH = Path("/proc/meminfo")


@app.get("/host")
async def host() -> dict:
    """Disk free and RAM on this machine.

    Returns free bytes for the four volumes the trainer cares about, and the
    system's total and available RAM. Missing volumes (path does not exist or
    is not a mount point) are omitted. A caller that needs a guarantee should
    check that the expected keys are present.

    The four paths are taken from the settings rather than hard-coded, so they
    match what this box actually uses regardless of defaults.
    """
    settings = _settings()
    result: dict = {"host": socket.gethostname(), "volumes": {}}

    for label, attr in [
        ("jobs_root",      settings.jobs_root),
        ("checkpoint_root", settings.checkpoint_root),
        ("artefact_cache_root", settings.artefact_cache_root),
        ("trained_root",   settings.trained_root),
    ]:
        try:
            free_gb = free_disk_gb(attr)
            result["volumes"][label] = {"free_gb": round(free_gb, 2), "path": str(attr)}
        except OSError:
            pass  # path not readable or not a mount point

    # RAM from /proc/meminfo. Named so a test can point it at a fixture: the
    # issue asks for an injectable path and for partial data rather than a 500,
    # and a real /proc cannot produce the unreadable case on demand (#40).
    try:
        meminfo = MEMINFO_PATH.read_text(encoding="ascii")
        total_kb = free_kb = None
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                free_kb = int(line.split()[1])
                break
        if total_kb is not None:
            result["ram"] = {
                "total_gb": round(total_kb / 1e6, 2),
                "available_gb": round(free_kb / 1e6, 2) if free_kb is not None else None,
            }
    except (OSError, ValueError):
        pass  # /proc/meminfo not readable (not Linux?)

    return result


@app.post("/jobs", status_code=202)
async def submit(request: TrainRequest, response: Response,
                 verify_only: bool = Query(False)) -> dict:
    """Submit a training job, or — with ``verify_only=true`` — only report on it.

    ``verify_only`` is declared HERE and not only on the gateway proxy (#59).
    FastAPI silently drops query parameters a route does not declare, so
    ``POST :8204/jobs?verify_only=true`` used to be an ordinary submit: a flag
    whose entire purpose is "change nothing" queued a multi-day training run,
    and the caller could not tell from the request that it had been ignored.
    A safety flag that is dropped rather than refused is worse than no flag.
    """
    settings = _settings()
    store = _store()
    # Same rule as disk below: a venv that was never built will not build itself
    # while the job sits in the queue, so refuse now with the command that fixes
    # it rather than accepting a job that can only fail at spawn.
    python = settings.runner_python(request.engine)
    if not python.exists():
        backend = backend_for(request.engine)
        raise HTTPException(
            status_code=503,
            detail=(f"the {backend.venv} venv is not built on this box, so {request.engine} "
                    f"jobs cannot run. Build it:  bash scripts/make_venvs.sh {backend.venv}"),
        )
    # Disk is checked here because it will not fix itself; VRAM is checked when
    # the scheduler starts the job, because a busy GPU is what the queue is for.
    try:
        check_disk(settings.jobs_root, settings.min_free_disk_gb)
    except PreflightError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc
    # A base_model that names nothing loadable is knowable now. It used to fail in
    # the TRAIN stage - after prepare and compile - which on a large selection is
    # ten hours to learn that a registry id was spelled as a DOI (#76).
    registry, why_not = _registry()
    if request.base_model:
        try:
            resolve_base_model(request.base_model, engine=request.engine,
                               registry=registry, registry_error=why_not)
        except BaseModelError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    # A network TMPDIR breaks temp-dir cleanup mid-compile; catch it at submit.
    try:
        check_tmpdir(os.environ.get("TMPDIR", "/tmp"))
    except PreflightError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    # ...and a network datasets cache breaks the Arrow generation pass eleven
    # hours in (#60). Only when this run will actually cache: a streaming job
    # writes no Arrow cache, so its location cannot hurt it.
    if settings.cache_datasets:
        try:
            check_datasets_cache()
        except PreflightError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # The dataset is checked HERE, not in the gateway proxy, so the guard holds
    # for every caller — the proxy's own contract is that no training logic lives
    # in it (#35), and a check it owned would be one a direct call could skip.
    checked = _verify(request)
    if verify_only:
        # A dry run answers a question; it does not act. 200 even for an invalid
        # spec — "is this spec good?" and "did my request fail?" are different
        # questions, and the caller reads ``valid``. Same contract as the proxy.
        response.status_code = 200
        return checked
    if not checked["valid"]:
        raise HTTPException(status_code=400, detail=checked)

    # A model_id is a directory name AND a registry id, so two live jobs sharing
    # one means whichever registers last silently replaces the other's weights
    # (#56). The job ids stay distinct — JobStore de-duplicates those — which is
    # exactly why this is easy to miss until the models are already overwritten.
    # Every host's jobs count here: trained_root and the registry are shared (#15).
    clash = next((j for j in store.list()
                  if j.request.model_id == request.model_id and not j.is_terminal), None)
    if clash is not None:
        if store.owns(clash):
            way_out = "cancel that job first."
        else:
            # "Cancel that job first" alone sent the caller to a cancel that
            # answers 409 for another host's job — a circle once that host is
            # retired (#15 review).
            way_out = (f"cancel that job on host {store.host_of(clash)}, where it belongs. "
                       + _foreign_way_out(store, clash))
        raise HTTPException(
            status_code=409,
            detail=(f"job {clash.id} is already {clash.status} for model_id "
                    f"{request.model_id!r}. Two live jobs writing one model_id means the "
                    "second overwrites the first's registered weights — choose a different "
                    f"model_id, or {way_out}"),
        )
    # The gateway skips a trained registration whose id is curated, and its
    # promotion gate answers that id with the curated weights — so the run would
    # end "promoted" with a model nobody can reach (#14 review). Refused before
    # any GPU is spent; the runner checks again before it registers. A curated
    # file that cannot be read does not block the queue.
    if registry is not None and registry.get(request.model_id) is not None:
        raise HTTPException(
            status_code=409,
            detail=(f"model_id {request.model_id!r} is a curated id in "
                    f"{registry.path or 'the curated registry'}. The gateway never serves a "
                    "trained model under a curated id — it could not be told apart from the "
                    "curated one. Choose a different model_id."),
        )
    if registry is None:
        logger.warning("queuing {} without checking it against the curated ids: {}",
                       request.model_id, why_not)
    # An unreachable hub does NOT block the queue: the job downloads when it
    # starts, which may be hours from now, and refusing it would turn a hiccup
    # into a failed submission. The reason travels on the job instead.
    if not checked["checked"]:
        logger.warning("queuing {} unverified: {}",
                       request.model_id, checked["unverified_reason"])

    # Stamped here, at acceptance: the job belongs to the host that took it, and
    # only this host will ever start it (#15).
    job = store.create(request, host=settings.host_id)
    logger.info("queued job {} for model {} on host {}", job.id, request.model_id, job.host)
    _schedule()  # start immediately when the box allows it, rather than at the next tick
    job = store.load(job.id)
    return {"job_id": job.id, "status": job.status, "queued_reason": job.queued_reason,
            "dataset_verified": checked["checked"],
            **({"unverified_reason": checked["unverified_reason"]}
               if not checked["checked"] else {})}


def _verify(request: TrainRequest) -> dict:
    """Check the dataset spec against the hub. Never raises for a reachability
    problem — that is reported as ``checked: false`` and left to the caller.

    Three outcomes, deliberately distinct:

    ``{valid: true,  checked: true}``   the selection is really there
    ``{valid: false, checked: true}``   the spec is wrong, and ``errors`` says how
    ``{valid: true,  checked: false}``  the hub could not be reached; unknown
    """
    check = getattr(app.state, "verify_spec", None) or verify_dataset_spec
    errors: list[str] = []
    # Every dataset, not just the first (#40). Checking one of three and reporting
    # "valid" would be the same class of mistake the guard exists to prevent — and
    # each error is prefixed, because "project 'x' not found" is not actionable
    # when the job named three repos.
    for spec in request.datasets:
        try:
            found = check(spec, _settings(), chunk_capable=_chunk_capable(request.engine))
        except DatasetSelectionError as exc:
            return {"valid": False, "checked": True,
                    "errors": [f"{spec.hf_repo}: {exc}"]}
        except VerificationUnavailable as exc:
            return {"valid": True, "checked": False, "errors": [],
                    "unverified_reason": f"the hub could not be reached: {exc}"}
        errors += ([f"{spec.hf_repo}: {e}" for e in found]
                   if len(request.datasets) > 1 else found)
    return {"valid": not errors, "checked": True, "errors": errors}


def _chunk_capable(engine: str) -> bool:
    """Does this engine's backend actually implement chunked prepare?

    Only kraken does. The size guard used to read ``ATR_TRAIN_CHUNK_PAGES`` alone
    and cleared a 293 GB vllm corpus on the strength of a setting that backend
    ignores (#85).
    """
    from atr_training.backends import BACKENDS

    backend = BACKENDS.get(engine)
    return bool(backend and backend.supports_chunked_prepare)


@app.post("/jobs/verify", status_code=200)
async def verify(request: TrainRequest) -> dict:
    """Check a TrainRequest's dataset against the hub without queueing anything.

    Always 200 — the answer is in the body. A caller asking "would this run?"
    gets a report, not an exception, and an unreachable hub is reported as
    ``checked: false`` rather than as a bad spec.
    """
    return _verify(request)


#: What ``fields=summary`` returns per job (#38, serving-atr-inference#107):
#: enough to see what is running and why a queued job has not started, without
#: the full record — the body of GET /jobs ran to ~1 MB on a busy trainer.
_SUMMARY_FIELDS = ("id", "status", "stage", "created_at", "queued_reason", "error")


@app.get("/jobs")
async def list_jobs(
    limit: int | None = Query(None, ge=1,
                              description="keep only the N newest jobs"),
    fields: Literal["full", "summary"] = Query("full"),
) -> dict:
    """List jobs, newest first.

    Both parameters are opt-in and the default response is byte-identical to
    before: no query string still returns every job in full. ``limit`` slices
    the id list before anything is loaded, and ``fields=summary`` returns
    only :data:`_SUMMARY_FIELDS` per job — the shape the gateway's
    ``atr_status`` consumer needs, not the whole record.
    """
    jobs = _store().list(limit=limit)
    if fields == "summary":
        full = [j.model_dump(mode="json") for j in jobs]
        return {"jobs": [{k: j[k] for k in _SUMMARY_FIELDS} for j in full]}
    return {"jobs": [j.model_dump(mode="json") for j in jobs]}


def _load(job_id: str) -> TrainJob:
    try:
        return _store().load(job_id)
    except JobStoreError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    return _store().reconcile(_load(job_id)).model_dump(mode="json")


@app.get("/jobs/{job_id}/log")
async def get_log(job_id: str, stage: str = Query("train"), lines: int = Query(200, ge=1, le=5000)):
    job = _load(job_id)
    path = _store().paths(job.id).log(stage)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no {stage} log for job {job_id}")
    return {"job_id": job_id, "stage": stage, "lines": tail(path, lines)}


@app.get("/jobs/{job_id}/curve")
async def get_curve(job_id: str) -> dict:
    """The per-epoch record for a run (#38) — including while it is still running.

    Three sources, in order, and the shape is the same for all of them (#77):

    1. ``training.json``, written when the train stage ends — the final record.
    2. **The checkpoint directory, read live.** Lightning writes each epoch's
       metric into the filename as it goes, so a running job's progress is on
       disk long before the stage finishes. Reading it only at the end made the
       endpoint useless for the thing it is most wanted for: deciding, mid-run,
       whether a job is still improving or has plateaued and should be stopped.
    3. Neither, because the job has not reached training — an empty ``points``
       list and a note saying so, not a 404. Callers poll this; an answer that
       changes shape between "not yet" and "here you go" makes every caller
       handle two bodies to ask one question.
    """
    job = _store().reconcile(_load(job_id))
    path = _store().paths(job.id).root / CURVE_FILENAME
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    if job.checkpoint_dir and Path(job.checkpoint_dir).is_dir():
        curve = curve_from_checkpoints(job.checkpoint_dir)
        if curve.points:
            payload = curve_payload(curve, job.id)
            payload["live"] = True
            payload["note"] = (
                f"read live from the checkpoint directory while the job is {job.status}; "
                + curve.note
            )
            return payload

    return curve_payload(
        empty_curve(
            f"no checkpoints yet — job is {job.status}"
            + (f" in the {job.stage} stage" if job.stage else "")
            + ". Metrics appear once the train stage starts writing checkpoints."
        ),
        job.id,
    )


def _foreign_way_out(store: JobStore, job: TrainJob) -> str:
    """How a person closes another host's record once that host no longer runs it.

    Nothing in the service does: it never judges another host's job. A retired
    host (the old trainer on idhefix after the cutover) or a Slurm job that is
    gone would otherwise leave the record live for ever. The tool needs a shell
    on a machine that mounts the store, on purpose — a caller of this API
    cannot confirm that a process on another machine is gone.
    """
    host = store.host_of(job)
    gone = ("If the Slurm job is gone (scancelled, or it never ran)" if host == SLURM_HOST
            else f"If host {host} no longer runs it (its trainer is retired, or the "
                 "process is confirmed gone there)")
    return (f"{gone}, an operator can close the record without signalling anything: "
            f"{close_command(job.id)}")


def _refuse_foreign(store: JobStore, job: TrainJob) -> None:
    """409 for a live job of another host, naming the host and the way out.

    Its pid is that machine's: ``killpg`` on it here would SIGTERM whatever
    local process group happens to carry the number (#15). And a queued one is
    still that host's to start — marking it cancelled here would race its
    scheduler writing the same record.
    """
    if job.is_terminal or store.owns(job):
        return
    host = store.host_of(job)
    state = "is queued on" if job.status == "queued" else "runs on"
    where = "with scancel, on UBELIX" if host == SLURM_HOST else "there"
    raise HTTPException(
        status_code=409,
        detail=(f"job {job.id} {state} host {host}; cancel it {where}. "
                + _foreign_way_out(store, job)))


@app.post("/jobs/{job_id}/cancel")
async def cancel(job_id: str) -> dict:
    store = _store()
    job = store.reconcile(_load(job_id))
    if job.is_terminal:
        raise HTTPException(status_code=409, detail=f"job {job_id} is already {job.status}")
    _refuse_foreign(store, job)
    if job.status == "queued" and job.pid is None:
        return _cancel_unstarted(store, job_id)
    if job.pid is not None:
        try:
            os.killpg(os.getpgid(job.pid), signal.SIGTERM)
            logger.info("SIGTERM sent to process group of pid {}", job.pid)
        except ProcessLookupError:
            logger.warning("pid {} already gone for job {}", job.pid, job_id)
    # The runner marks itself cancelled on SIGTERM; a queued (or already dead)
    # job has nobody to do that, so record it here.
    job = store.load(job_id)
    if job.status == "queued" or job.pid is None:
        job.error = "cancelled before it started"
        job = store.advance(job, "cancelled")
    return job.model_dump(mode="json")


def _cancel_unstarted(store: JobStore, job_id: str) -> dict:
    """Cancel a job nobody has started, holding the claim a spawn would need.

    Rewriting the record alone lost the race: a tick that listed the job before
    this call saved its own copy after it — hold() rewrites a waiting job's
    reason on nearly every tick — or spawned it and saved the pid, and the job
    was queued again and started, while the caller had been told `cancelled`
    (#15 review, reproduced both ways). Once the claim is ours, no scheduler
    starts the job and none writes its record. If a scheduler holds it, the job
    is being started this instant; a moment later it has a pid and cancels like
    any running job.
    """
    try:
        store.claim(job_id, purpose=CLAIM_CANCEL)
    except FileExistsError:
        raise HTTPException(
            status_code=409,
            detail=(f"job {job_id} is being started right now; ask again in a few "
                    "seconds, and the cancel will stop its runner")) from None
    except OSError as exc:
        raise HTTPException(status_code=503,
                            detail=f"job {job_id} could not be claimed for the cancel: {exc}"
                            ) from exc
    # If this write is lost after all, the claim says what was meant, and the
    # next reconcile completes the cancel (JobStore._reconcile_claim).
    job = store.load(job_id)
    if job.status == "queued" and job.pid is None:
        job.error = "cancelled before it started"
        job = store.advance(job, "cancelled")
    return job.model_dump(mode="json")


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str) -> dict:
    """Drop a terminal job's artefacts.

    A terminal job of ANOTHER host may be deleted here too: its job directory is
    on the share and no process of it is left to signal, so nothing about it is
    that host's alone — except its checkpoints, which are on that host's local
    disk and are left for it. A live one is refused like a cancel (#15).
    """
    store = _store()
    settings = _settings()
    job = store.reconcile(_load(job_id))
    _refuse_foreign(store, job)
    if not job.is_terminal:
        raise HTTPException(
            status_code=409,
            detail=f"job {job_id} is {job.status}; cancel it before deleting its artifacts",
        )
    # job.json is kept so the record (and its metrics) survives; the registered
    # model lives outside the job directory and is never touched here.
    store.delete(job_id, keep=["job.json"])
    # Checkpoints live on local scratch outside the job dir, so the store cannot
    # reach them — clean them up here or they leak. Only this host's: another
    # host's checkpoint_dir names a directory on ITS disk, and the same path here
    # is not that directory.
    ckpt = Path(job.checkpoint_dir) if job.checkpoint_dir and store.owns(job) else None
    if job.checkpoint_dir and ckpt is None:
        logger.info("job {} ran on host {}; its checkpoints ({}) are left for that host",
                    job_id, store.host_of(job), job.checkpoint_dir)
    if ckpt is not None and ckpt.is_dir():
        shutil.rmtree(ckpt, ignore_errors=True)
    # Orphaned weights: a failed register stage may have left a weights directory
    # with no metadata.json (#50). Under the same three conditions as at startup,
    # so a leftover younger than orphan_weights_min_age_h waits for a later DELETE
    # or restart: from here it looks exactly like a registration in progress.
    orphaned = _cleanup_orphaned_weights(settings.trained_root, store,
                                         settings.orphan_weights_min_age_h,
                                         registry_root=settings.registry_root)
    return {"job_id": job_id, "deleted": True, "record_kept": True,
            "checkpoints_removed": ckpt is not None, "orphaned_weights_removed": orphaned}


if __name__ == "__main__":  # pragma: no cover
    # Through the launcher, so this entry point cannot bind what it would refuse.
    from atr_training.serve import main

    raise SystemExit(main())

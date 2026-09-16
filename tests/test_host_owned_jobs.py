"""A job belongs to the host that accepted it (#15).

The job store is on the research share, and idhefix, asteraix and UBELIX all
write into it. A record's pid is a statement about one machine: read against
another machine's /proc it declared live runs dead on every scheduler tick, and
two schedulers on one store would both start every queued job. These tests put
two hosts — and two schedulers of one host — on one directory and check that
each judges, starts, signals and counts only its own jobs.
"""

from __future__ import annotations

import importlib.util
import json
import os
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from loguru import logger
from pydantic import ValidationError

from atr_training import close_job
from atr_training.contracts import DatasetSpec, Metrics, TrainRequest, utcnow
from atr_training.jobstore import (
    CLAIM_CANCEL,
    CLAIM_ORPHAN_AFTER_S,
    JobStore,
    JobStoreError,
)
from atr_training.preflight import GpuInfo, PreflightError
from atr_training.registration import registration_path, trained_dir, write_registration
from atr_training.settings import TrainerSettings
from kraken_train_svc import app as app_module

REPO = "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi"
BODY = {
    "model_id": "kraken-thun-missiven-v1",
    "dataset": {"hf_repo": REPO, "train_projects": ["GT_Thun-Training_(TEST-DEMO)"],
                "eval_projects": ["GT_Thun-Test_(DEMO_TEST)"]},
}
LOOPBACK = ("127.0.0.1", 50000)
#: Not a pid on any machine (pid_t is int32; Linux pid_max is far below).
PID_NEVER = 2**31 - 1
UBELIX_DIR = Path(__file__).resolve().parent.parent / "ubelix"


def request(model_id: str = "kraken-thun-missiven-v1", engine: str = "kraken") -> TrainRequest:
    return TrainRequest(model_id=model_id, engine=engine,
                        dataset=DatasetSpec(hf_repo=REPO, train_projects=["P"]))


def free_gpu(gpu, min_free_mb):
    return GpuInfo(index=gpu, free_mb=40000, total_mb=46068)


class Spawns:
    """The detached runner, as far as the scheduler can tell. Thread-safe, because
    two schedulers call it at once below. Reports this process's pid, so a
    spawned job stays alive to reconcile."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def __call__(self, settings, job):
        with self._lock:
            self.calls.append((settings.host_id, job.id))
        return os.getpid()


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, UBELIX_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _running(store: JobStore, job_id: str, pid: int | None, status: str = "training"):
    job = store.load(job_id)
    for step in ("preparing", "compiling", "training", "testing", "registering"):
        job = store.advance(job, step)
        if step == status:
            break
    job.pid = pid
    return store.save(job)


def _strip_host(store: JobStore, job_id: str) -> None:
    """Rewrite a record the way the old trainer writes it: no ``host`` key."""
    path = store.paths(job_id).job_json
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["host"]
    path.write_text(json.dumps(data), encoding="utf-8")


def _age(path: Path, hours: float) -> None:
    """Backdate a directory and its top-level entries, all to the same moment.

    Both, because utime on a child never moves the directory's mtime (only
    adding, removing or renaming an entry does), and the cleanup reads both.
    Which of the two it reads is pinned by
    test_startup_cleanup_reads_the_newest_mtime_of_the_directory_and_its_entries."""
    then = time.time() - hours * 3600
    for child in path.iterdir():
        os.utime(child, (then, then))
    os.utime(path, (then, then))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "jobs"


@pytest.fixture
def asteraix(root: Path) -> JobStore:
    return JobStore(root, host_id="asteraix")


@pytest.fixture
def idhefix(root: Path) -> JobStore:
    """The same directory, seen from the other machine."""
    return JobStore(root, host_id="idhefix")


@pytest.fixture
def logs():
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(message.record["message"]), level="DEBUG")
    yield lines
    logger.remove(sink)


@pytest.fixture
def venvs(tmp_path: Path) -> Path:
    root = tmp_path / "venvs"
    for name in ("kraken-train", "vlm-train"):
        (root / name / "bin").mkdir(parents=True)
        (root / name / "bin" / "python").touch()
    return root


def trainer_settings(tmp_path: Path, venvs: Path, host_id: str, **kw) -> TrainerSettings:
    return TrainerSettings(jobs_root=tmp_path / "jobs", trained_root=tmp_path / "trained",
                           venvs_root=venvs, checkpoint_root=tmp_path / "ckpt",
                           min_free_disk_gb=0, host_id=host_id, **kw)


@pytest.fixture
def settings(tmp_path: Path, venvs: Path) -> TrainerSettings:
    return trainer_settings(tmp_path, venvs, "asteraix")


@pytest.fixture
def app(settings: TrainerSettings):
    """asteraix's service, its store built from its settings as in production."""
    app = app_module.app
    app.state.settings = settings
    app.state.spawn = Spawns()
    app.state.vram_check = free_gpu
    app.state.verify_spec = lambda spec, settings, **kw: []
    yield app
    for attr in ("settings", "store", "spawn", "vram_check", "verify_spec",
                 "gpu_claim", "gpu_claim_at"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


@pytest.fixture
def client(app, trainer_key):
    with TestClient(app, client=LOOPBACK, headers={"X-API-Key": trainer_key}) as c:
        yield c


def store_of(app) -> JobStore:
    """The store ``app`` built from its settings — not one a test handed it."""
    assert app is app_module.app
    return app_module._store()


# ── reconcile ───────────────────────────────────────────────────────────────
def test_reconcile_leaves_a_job_from_another_host_alone(asteraix, idhefix, logs):
    """idhefix's run, dead or alive, is not asteraix's to judge.

    The incident this prevents: asteraix looks for idhefix's pid in its own
    /proc, does not find it, and writes ``failed`` into a run that is still
    training — every tick. The pid is not even asked about.
    """
    job = idhefix.create(request())
    _running(idhefix, job.id, PID_NEVER)
    before = idhefix.paths(job.id).job_json.read_bytes()
    asked = []

    for _ in range(3):   # three ticks
        out = asteraix.reconcile(asteraix.load(job.id),
                                 is_alive=lambda pid: asked.append(pid) or False)
        assert out.status == "training" and out.error is None

    assert asked == []
    assert idhefix.paths(job.id).job_json.read_bytes() == before, "the record was rewritten"
    noted = [line for line in logs if job.id in line]
    assert len(noted) == 1, "a foreign job is logged once, not every tick"
    assert "belongs to host idhefix" in noted[0]

    # A pid that happens to be taken locally is no evidence either.
    alive_here = asteraix.reconcile(asteraix.load(job.id), is_alive=lambda pid: True)
    assert alive_here.status == "training"

    # Its own host still judges it — the rule narrowed, it did not disappear.
    failed = idhefix.reconcile(idhefix.load(job.id), is_alive=lambda pid: False)
    assert failed.status == "failed" and str(PID_NEVER) in failed.error


def test_a_store_that_does_not_know_its_host_refuses_to_judge_or_create(root, asteraix):
    """Guessing an identity is how a pid gets read on the wrong machine."""
    job = _running(asteraix, asteraix.create(request()).id, PID_NEVER)
    anonymous = JobStore(root)
    with pytest.raises(JobStoreError, match="without a host id"):
        anonymous.reconcile(job, is_alive=lambda pid: False)
    with pytest.raises(JobStoreError, match="needs a host"):
        anonymous.create(request("other-model"))
    # Loading and saving one record — all the runner does — needs no identity.
    anonymous.save(anonymous.load(job.id))
    assert asteraix.load(job.id).host == "asteraix"


def test_a_legacy_record_without_host_is_treated_as_the_legacy_host(
        root, asteraix, idhefix, tmp_path, venvs, monkeypatch):
    """The 48 records in the shared store predate the field and were all born on
    idhefix, by a trainer that will never write it."""
    job = idhefix.create(request())
    _running(idhefix, job.id, PID_NEVER)
    _strip_host(idhefix, job.id)

    legacy = asteraix.load(job.id)
    assert legacy.host is None
    assert asteraix.host_of(legacy) == idhefix.host_of(legacy) == "idhefix"
    assert not asteraix.owns(legacy) and idhefix.owns(legacy)
    assert asteraix.reconcile(legacy, is_alive=lambda pid: False).status == "training"
    assert idhefix.reconcile(idhefix.load(job.id),
                             is_alive=lambda pid: False).status == "failed"

    # The default is idhefix, and the service reads it from its settings.
    monkeypatch.delenv("ATR_TRAIN_LEGACY_JOB_HOST")
    assert TrainerSettings(_env_file=None).legacy_job_host == "idhefix"
    moved = trainer_settings(tmp_path, venvs, "asteraix", legacy_job_host="asteraix")
    app_module.app.state.settings = moved
    try:
        assert app_module._store().host_of(legacy) == "asteraix"
        assert app_module._store().owns(legacy)
    finally:
        for attr in ("settings", "store"):
            delattr(app_module.app.state, attr)


def test_the_host_id_is_a_setting_with_a_plain_name(monkeypatch):
    """The hostnames are no identity (idhefix says `srv`, asteraix `dhserver03`),
    so the name is configured — and checked, because it is compared byte for byte."""
    import socket

    monkeypatch.delenv("ATR_TRAIN_HOST_ID")
    assert TrainerSettings(_env_file=None).host_id == socket.gethostname()
    monkeypatch.setenv("ATR_TRAIN_HOST_ID", "asteraix")
    assert TrainerSettings(_env_file=None).host_id == "asteraix"
    for bad in ("", "aster aix", "asteraix\n", "idhefix/1"):
        with pytest.raises(ValidationError, match="ATR_TRAIN_HOST_ID must be non-empty"):
            TrainerSettings(_env_file=None, host_id=bad)
    with pytest.raises(ValidationError, match="ATR_TRAIN_LEGACY_JOB_HOST"):
        TrainerSettings(_env_file=None, legacy_job_host="")


# ── the service stamps what it accepts ──────────────────────────────────────
def test_a_job_records_the_host_that_spawned_it(client, app):
    resp = client.post("/jobs", json=BODY)
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    record = store_of(app).load(job_id)
    assert record.host == "asteraix"
    assert record.pid == os.getpid(), "accepted and started by the same host"
    assert app.state.spawn.calls == [("asteraix", job_id)]
    # Additive on the wire: the gateway proxy, the Discord bot and the MCP read
    # these records, and every field they had is still there.
    one = client.get(f"/jobs/{job_id}").json()
    assert one["host"] == "asteraix"
    assert {"id", "request", "status", "stage", "pid", "queued_reason", "error"} <= set(one)
    assert [j["host"] for j in client.get("/jobs").json()["jobs"]] == ["asteraix"]


# ── the scheduler ───────────────────────────────────────────────────────────
def test_two_schedulers_on_one_store_never_spawn_the_same_job(tmp_path, venvs, root):
    # Two hosts: each starts only its own.
    spawns = Spawns()
    on_asteraix = trainer_settings(tmp_path, venvs, "asteraix")
    on_idhefix = trainer_settings(tmp_path, venvs, "idhefix")
    asteraix = JobStore(root, host_id="asteraix")
    idhefix = JobStore(root, host_id="idhefix")
    job = asteraix.create(request())
    before = asteraix.paths(job.id).job_json.read_bytes()

    assert app_module.schedule_once(idhefix, on_idhefix, spawn=spawns,
                                    vram_check=free_gpu) is None
    assert spawns.calls == []
    assert asteraix.paths(job.id).job_json.read_bytes() == before, \
        "idhefix wrote a record that is not its own"
    started = app_module.schedule_once(asteraix, on_asteraix, spawn=spawns,
                                       vram_check=free_gpu)
    assert started.id == job.id
    for _ in range(2):
        app_module.schedule_once(idhefix, on_idhefix, spawn=spawns, vram_check=free_gpu)
        app_module.schedule_once(asteraix, on_asteraix, spawn=spawns, vram_check=free_gpu)
    assert spawns.calls == [("asteraix", job.id)]
    asteraix.fail(asteraix.load(job.id), "make room for the race")

    # One host, two schedulers (#12), racing: both have listed the job as
    # unstarted before either claims it. The claim decides.
    for n in range(10):
        spawns = Spawns()
        fresh = asteraix.create(request(f"race-{n}"))
        both_listed = threading.Barrier(2, timeout=5)

        def gpu_once_both_have_listed(gpu, min_free_mb):
            both_listed.wait()
            return free_gpu(gpu, min_free_mb)

        errors: list[BaseException] = []

        def tick():
            try:
                app_module.schedule_once(JobStore(root, host_id="asteraix"), on_asteraix,
                                         spawn=spawns, vram_check=gpu_once_both_have_listed)
            except BaseException as exc:  # noqa: BLE001 — reported below
                errors.append(exc)

        threads = [threading.Thread(target=tick) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert errors == []
        assert spawns.calls == [("asteraix", fresh.id)], f"round {n}"
        record = asteraix.load(fresh.id)
        assert record.pid == os.getpid() and record.status == "queued"
        asteraix.fail(record, "make room for the next round")


def test_the_running_count_is_per_host(asteraix, idhefix, tmp_path, venvs):
    """max_concurrent is about this box's card; idhefix's run is on idhefix's."""
    spawns = Spawns()
    settings = trainer_settings(tmp_path, venvs, "asteraix")   # max_concurrent 1
    theirs = _running(idhefix, idhefix.create(request("v5")).id, PID_NEVER)
    mine = asteraix.create(request("mine"))

    started = app_module.schedule_once(asteraix, settings, spawn=spawns, vram_check=free_gpu)
    assert started is not None and started.id == mine.id
    assert idhefix.load(theirs.id).status == "training", "a foreign run was judged"

    # ...and this host's own run does hold the queue, named as the reason.
    _running(asteraix, mine.id, os.getpid())
    waiting = asteraix.create(request("waiting"))
    assert app_module.schedule_once(asteraix, settings, spawn=spawns,
                                    vram_check=free_gpu) is None
    assert asteraix.load(waiting.id).queued_reason == f"waiting for {mine.id} (training)"
    assert spawns.calls == [("asteraix", mine.id)]


def test_another_host_s_run_does_not_claim_this_card(client, app, root):
    """/gpu-claim tells the gateway whether THIS card is spoken for."""
    other = JobStore(root, host_id="idhefix")
    job = other.create(request("v5"))
    _running(other, job.id, 1843)
    app_module.refresh_gpu_claim(store_of(app).list())
    assert client.get("/gpu-claim").json()["claimed"] is False

    mine = store_of(app).create(request("mine"))
    _running(store_of(app), mine.id, os.getpid())
    app_module.refresh_gpu_claim(store_of(app).list())
    body = client.get("/gpu-claim").json()
    assert body["claimed"] is True
    assert [j["id"] for j in body["jobs"]] == [mine.id]


def test_a_claim_without_a_pid_is_failed_after_the_window(asteraix, tmp_path, venvs):
    """Claimed, then the scheduler died before the pid was saved. The job is never
    offered to the queue again, so without this it would be queued for ever."""
    settings = trainer_settings(tmp_path, venvs, "asteraix")
    spawns = Spawns()
    stuck = asteraix.create(request("stuck"))
    nxt = asteraix.create(request("next"))
    asteraix.claim(stuck.id, now=utcnow() - timedelta(minutes=5))

    # Inside the window it is a start in progress: it holds the slot, and
    # nobody writes its record.
    before = asteraix.paths(stuck.id).job_json.read_bytes()
    assert app_module.schedule_once(asteraix, settings, spawn=spawns,
                                    vram_check=free_gpu) is None
    assert spawns.calls == []
    assert asteraix.paths(stuck.id).job_json.read_bytes() == before
    assert asteraix.load(nxt.id).queued_reason == f"waiting for {stuck.id} (starting)"

    # Past it, the job is failed, saying why — and the queue moves.
    asteraix.paths(stuck.id).claim.unlink()
    asteraix.claim(stuck.id, now=utcnow() - timedelta(seconds=CLAIM_ORPHAN_AFTER_S + 60))
    started = app_module.schedule_once(asteraix, settings, spawn=spawns, vram_check=free_gpu)
    failed = asteraix.load(stuck.id)
    assert failed.status == "failed"
    assert "claimed for spawning by asteraix" in failed.error
    assert "no runner pid was ever recorded" in failed.error
    assert started.id == nxt.id and spawns.calls == [("asteraix", nxt.id)]


def test_an_old_claim_is_judged_only_when_it_is_this_host_s(asteraix):
    """A claim naming another host is left alone, like that host's jobs. An empty
    one — the writer died between create and write — can only be an owner's."""
    theirs = asteraix.create(request("claimed-elsewhere"))
    asteraix.paths(theirs.id).claim.write_text(json.dumps(
        {"host": "idhefix", "pid": 1, "at": (utcnow() - timedelta(days=1)).isoformat()}))
    assert asteraix.reconcile(asteraix.load(theirs.id)).status == "queued"

    empty = asteraix.create(request("claimed-empty"))
    asteraix.paths(empty.id).claim.touch()
    assert asteraix.reconcile(asteraix.load(empty.id)).status == "queued", "fresh: in progress"
    old = time.time() - CLAIM_ORPHAN_AFTER_S - 60
    os.utime(asteraix.paths(empty.id).claim, (old, old))
    out = asteraix.reconcile(asteraix.load(empty.id))
    assert out.status == "failed" and "the claim file is empty" in out.error

    unclaimed = asteraix.create(request("waiting-its-turn"))
    assert asteraix.reconcile(asteraix.load(unclaimed.id)).status == "queued"


def test_ubelix_records_are_never_spawned_by_a_trainer(tmp_path, venvs, root, monkeypatch,
                                                       capsys):
    """Slurm supervises them; their pids are compute nodes'."""
    monkeypatch.setenv("ATR_TRAIN_JOBS_ROOT", str(root))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"engine": "vllm", "model_id": "qwen3vl-on-ubelix",
                                "dataset": {"hf_repo": REPO, "train_projects": ["P"]}}))
    assert _load_script("submit_job").main([str(spec)]) == 0
    job_id = capsys.readouterr().out.strip()

    trainer = JobStore(root, host_id="asteraix")
    submitted = trainer.load(job_id)
    assert submitted.host == "ubelix"
    spawns = Spawns()
    settings = trainer_settings(tmp_path, venvs, "asteraix")
    assert app_module.schedule_once(trainer, settings, spawn=spawns,
                                    vram_check=free_gpu) is None
    assert spawns.calls == []
    assert trainer.load(job_id).queued_reason is None, "a trainer wrote a Slurm record"

    # Running on a node: its pid is nothing here, alive or dead.
    _running(trainer, job_id, PID_NEVER)
    assert trainer.reconcile(trainer.load(job_id),
                             is_alive=lambda pid: False).status == "training"
    # No trainer can take the name — not through settings, and not by a store
    # opened with it directly.
    with pytest.raises(ValidationError, match="reserved"):
        TrainerSettings(_env_file=None, host_id="ubelix")
    assert not JobStore(root, host_id="ubelix").owns(trainer.load(job_id))

    # A fan-out clone is Slurm's too, whoever prepared the corpus.
    prepared = _running(trainer, trainer.create(request("corpus", engine="vllm")).id, None)
    data = trainer.paths(prepared.id).data
    (data / "crops").mkdir()
    for name in ("train.jsonl", "val.jsonl", "pages_train.lst", "pages_val.lst"):
        (data / name).write_text("", encoding="utf-8")
    (arm,) = _load_script("fanout").fan_out(str(root), prepared.id,
                                            [("qwen35-arm", "Qwen/Qwen3.5-2B")])
    assert trainer.load(arm).host == "ubelix"
    assert trainer.load(prepared.id).host == "asteraix"


# ── cancel and delete ───────────────────────────────────────────────────────
def test_cancel_never_signals_a_pid_from_another_host(client, app, root, monkeypatch):
    """The pid is this process's own — the worst case: a live local process
    group that a killpg would really have hit."""
    signalled = []
    monkeypatch.setattr(app_module.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(app_module.os, "killpg",
                        lambda pgid, sig: signalled.append((pgid, sig)))
    idhefix = JobStore(root, host_id="idhefix")

    stamped = _running(idhefix, idhefix.create(request("stamped")).id, os.getpid())
    legacy = _running(idhefix, idhefix.create(request("legacy")).id, os.getpid())
    _strip_host(idhefix, legacy.id)
    theirs_queued = idhefix.create(request("theirs-queued"))
    slurm = _running(idhefix, idhefix.create(request("slurm"), host="ubelix").id, os.getpid())

    for job, state in ((stamped, "runs on host idhefix"), (legacy, "runs on host idhefix"),
                       (theirs_queued, "is queued on host idhefix"),
                       (slurm, "runs on host ubelix")):
        before = idhefix.paths(job.id).job_json.read_bytes()
        for answer in (client.post(f"/jobs/{job.id}/cancel"), client.delete(f"/jobs/{job.id}")):
            assert answer.status_code == 409, answer.text
            detail = answer.json()["detail"]
            assert detail.startswith(f"job {job.id} {state}; "), detail
            assert close_job.close_command(job.id) in detail, "no way out is named"
        assert idhefix.paths(job.id).job_json.read_bytes() == before
        assert idhefix.paths(job.id).data.is_dir(), "a live job's artefacts were dropped"
    assert "cancel it there" in client.post(f"/jobs/{stamped.id}/cancel").json()["detail"]
    assert "scancel" in client.post(f"/jobs/{slurm.id}/cancel").json()["detail"]
    assert signalled == []

    # This host's own job is still signalled: the rule is about whose pid it is.
    mine = store_of(app).create(request("mine"))
    _running(store_of(app), mine.id, os.getpid())
    assert client.post(f"/jobs/{mine.id}/cancel").status_code == 200
    assert signalled and signalled[0][0] == os.getpid()


def test_a_finished_job_of_another_host_may_be_deleted_but_not_its_checkpoints(
        client, app, root, tmp_path):
    """Its directory is on the share and no process of it is left, so there is
    nothing host-bound to protect — except checkpoints, which live on that host's
    local disk. A directory at the same path here is not those checkpoints."""
    idhefix = JobStore(root, host_id="idhefix")
    job = idhefix.create(request("done-on-idhefix"))
    (idhefix.paths(job.id).pages / "p.jpg").write_bytes(b"x")
    local = tmp_path / "ckpt" / job.id
    local.mkdir(parents=True)
    job = idhefix.load(job.id)
    job.checkpoint_dir = str(local)
    idhefix.fail(job, "finished")

    answer = client.delete(f"/jobs/{job.id}")
    assert answer.status_code == 200, answer.text
    assert answer.json()["checkpoints_removed"] is False
    assert not idhefix.paths(job.id).data.exists()
    assert idhefix.load(job.id).host == "idhefix", "the record is kept as it was"
    assert local.is_dir()


# ── the runner ──────────────────────────────────────────────────────────────
def test_the_runner_keeps_the_host_when_it_saves(tmp_path):
    """The runner loads, advances and saves the record dozens of times; it runs
    under whatever settings its box has, and it must not restamp or drop the
    host — a record that lost it would read as idhefix's."""
    from test_train_svc_pipeline import FakeRunner, FakeSource, request_with

    from kraken_train_svc.runner import Pipeline

    (tmp_path / "registry").mkdir()
    settings = TrainerSettings(
        jobs_root=tmp_path / "training", trained_root=tmp_path / "trained",
        registry_root=tmp_path / "registry", checkpoint_root=tmp_path / "scratch",
        ketos=tmp_path / "ketos", min_free_disk_gb=0, artefact_cache=False,
        artefact_cache_root=tmp_path / "artefacts")
    assert settings.host_id == "test-trainer"
    # What the runner itself opens: no identity at all.
    store = JobStore(settings.jobs_root)

    for host, runner, status in (("asteraix", FakeRunner(), "completed"),
                                 ("asteraix", FakeRunner(fail_on="train"), "failed"),
                                 ("ubelix", FakeRunner(), "completed")):
        job = store.create(request_with(model_id=f"kept-{host}-{status}"), host=host)
        done = Pipeline(store, settings, runner=runner,
                        source=FakeSource({"train": 4, "eval": 2})).execute(job.id)
        assert done.status == status, done.error
        assert store.load(job.id).host == host


# ── orphaned weights on the shared trained_root ─────────────────────────────
def _weights_dir(settings: TrainerSettings, model_id: str, *, metadata: bool = False) -> Path:
    directory = settings.trained_root / model_id
    directory.mkdir(parents=True)
    (directory / f"{model_id}.mlmodel").write_bytes(b"WEIGHTS")
    if metadata:
        (directory / "metadata.json").write_text("{}", encoding="utf-8")
    return directory


def _start(app, trainer_key) -> None:
    """Run the service's startup and shutdown, nothing else."""
    with TestClient(app, client=LOOPBACK, headers={"X-API-Key": trainer_key}):
        pass


def test_startup_cleanup_leaves_a_registration_in_progress_alone(
        app, settings, root, trainer_key, logs):
    """The other machine is registering: weights copied, metadata.json not yet
    written. Old on purpose — only the job store can protect these two."""
    idhefix = JobStore(root, host_id="idhefix")
    theirs = _running(idhefix, idhefix.create(request("registering-there")).id, 1843,
                      status="registering")
    theirs_dir = _weights_dir(settings, "registering-there")
    # And one found through model_path rather than its model_id.
    asteraix = JobStore(root, host_id="asteraix")
    mine = _running(asteraix, asteraix.create(request("some-model")).id, os.getpid(),
                    status="registering")
    mine_dir = _weights_dir(settings, "named-by-model-path")
    mine.model_path = str(mine_dir / "named-by-model-path.mlmodel")
    asteraix.save(mine)
    for directory in (theirs_dir, mine_dir):
        _age(directory, hours=48)

    _start(app, trainer_key)

    assert (theirs_dir / "registering-there.mlmodel").read_bytes() == b"WEIGHTS"
    assert mine_dir.is_dir()
    assert any("keeping registering-there" in line and theirs.id in line
               and "host idhefix" in line for line in logs), logs


def test_startup_cleanup_leaves_a_young_directory_alone(app, settings, trainer_key, logs):
    """No job this store knows of — a registration from a store this service does
    not read looks exactly like this, until it writes metadata.json."""
    young = _weights_dir(settings, "just-copied")
    _age(young, hours=settings.orphan_weights_min_age_h - 1)

    _start(app, trainer_key)

    assert young.is_dir()
    assert any("keeping just-copied" in line and "changed 23.0 h ago" in line
               for line in logs), logs


def test_startup_cleanup_removes_an_old_orphan(app, settings, root, trainer_key, logs):
    """All three conditions hold: no metadata.json, unchanged for a day, and the
    only job naming it is finished."""
    store = JobStore(root, host_id="idhefix")
    store.fail(store.create(request("failed-registration")), "register failed")
    orphan = _weights_dir(settings, "failed-registration")
    registered = _weights_dir(settings, "registered-long-ago", metadata=True)
    for directory in (orphan, registered):
        _age(directory, hours=48)
    stray = settings.trained_root / "notes.txt"
    stray.write_text("not a model", encoding="utf-8")

    _start(app, trainer_key)

    assert not orphan.exists()
    assert registered.is_dir() and stray.is_file()
    assert any("removing orphaned weights directory failed-registration" in line
               for line in logs), logs
    # The window is the setting's, not a constant.
    later = _weights_dir(settings, "another-orphan")
    _age(later, hours=48)
    removed = app_module._cleanup_orphaned_weights(
        settings.trained_root, store, min_age_h=72, registry_root=settings.registry_root)
    assert removed == 0 and later.is_dir()
    assert app_module._cleanup_orphaned_weights(
        settings.trained_root, store, min_age_h=24, registry_root=settings.registry_root) == 1


# ── records written after the tick listed them (#15 review) ────────────────
def test_reconcile_never_fails_a_job_on_an_older_copy_of_its_record(asteraix):
    """The tick lists every record before it judges any. A runner that writes
    its last status and exits in between did not die: its `completed` stays."""
    for last in ("completed", "cancelled"):
        job = _running(asteraix, asteraix.create(request(f"ends-{last}")).id, PID_NEVER,
                       status="registering")
        listed = asteraix.load(job.id)

        def finishes_then_exits(pid, job_id=job.id, last=last):
            record = asteraix.load(job_id)
            record.metrics = Metrics(cer=0.05)
            asteraix.advance(record, last)
            return False  # by the time anyone looks, the process is gone

        out = asteraix.reconcile(listed, is_alive=finishes_then_exits)
        on_disk = asteraix.load(job.id)
        assert out.status == on_disk.status == last
        assert on_disk.error is None, on_disk.error

    # The same for an old claim: the pid was saved after the listing.
    claimed = asteraix.create(request("claimed-then-started"))
    asteraix.claim(claimed.id, now=utcnow() - timedelta(seconds=CLAIM_ORPHAN_AFTER_S + 60))
    listed = asteraix.load(claimed.id)
    started = asteraix.load(claimed.id)
    started.pid = PID_NEVER
    asteraix.save(started)
    out = asteraix.reconcile(listed)
    assert out.status == "queued" and out.pid == PID_NEVER
    assert asteraix.load(claimed.id).status == "queued"

    # A record nobody wrote since is judged as before.
    dead = asteraix.reconcile(asteraix.load(claimed.id), is_alive=lambda pid: False)
    assert dead.status == "failed" and str(PID_NEVER) in dead.error


def test_a_waiting_tick_never_erases_a_start_it_did_not_see(asteraix, tmp_path, venvs):
    """Two schedulers of one host — the submit's pass on the event loop, the tick
    in a thread. S has listed the job and waits on nvidia-smi; T claims, spawns
    and saves the pid; S reads the card as busy and holds. S used to save its
    old copy, without the pid: ten minutes later the live run was failed as
    never started, and a second job went onto the card."""
    settings = trainer_settings(tmp_path, venvs, "asteraix")
    other = JobStore(asteraix.root, host_id="asteraix")
    spawns = Spawns()
    job = asteraix.create(request("m1"))

    def busy_once_the_other_has_started_it(gpu, min_free_mb):
        started = app_module.schedule_once(other, settings, spawn=spawns, vram_check=free_gpu)
        assert started is not None and started.id == job.id
        raise PreflightError("GPU 1 has 1200 MB free, need 20000 MB")

    assert app_module.schedule_once(asteraix, settings, spawn=spawns,
                                    vram_check=busy_once_the_other_has_started_it) is None
    record = asteraix.load(job.id)
    assert record.pid == os.getpid() and record.status == "queued"
    assert record.queued_reason is None, "the waiting tick wrote a job it did not start"

    later = utcnow() + timedelta(seconds=CLAIM_ORPHAN_AFTER_S + 60)
    assert asteraix.reconcile(record, is_alive=lambda pid: True, now=later).error is None
    second = asteraix.create(request("m2"))
    assert app_module.schedule_once(asteraix, settings, spawn=spawns,
                                    vram_check=free_gpu) is None
    assert spawns.calls == [("asteraix", job.id)]
    assert asteraix.load(second.id).queued_reason == f"waiting for {job.id} (queued)"

    # Claimed since the listing, pid not saved yet: not written either, since
    # that save could land after the pid's. Other waiting jobs still are.
    roomier = trainer_settings(tmp_path, venvs, "asteraix", max_concurrent=2)
    third = asteraix.create(request("m3"))
    before = asteraix.paths(second.id).job_json.read_bytes()

    def busy_once_the_other_has_claimed_it(gpu, min_free_mb):
        other.claim(second.id)
        raise PreflightError("GPU 1 has 900 MB free, need 20000 MB")

    assert app_module.schedule_once(asteraix, roomier, spawn=spawns,
                                    vram_check=busy_once_the_other_has_claimed_it) is None
    assert asteraix.paths(second.id).job_json.read_bytes() == before
    assert asteraix.load(third.id).queued_reason.startswith("GPU 1 has 900 MB free")


def test_a_job_that_changed_after_the_listing_is_not_started(asteraix, tmp_path, venvs):
    """The spawn starts from the record on disk once the claim is held, not from
    the listing: a writer that takes no claim — the old trainer on idhefix, or
    a cancel on the old code — may have moved the job in between."""
    settings = trainer_settings(tmp_path, venvs, "asteraix")
    spawns = Spawns()
    cancelled = asteraix.create(request("cancelled-meanwhile"))

    def cancelled_meanwhile(gpu, min_free_mb):
        asteraix.advance(asteraix.load(cancelled.id), "cancelled")
        return free_gpu(gpu, min_free_mb)

    assert app_module.schedule_once(asteraix, settings, spawn=spawns,
                                    vram_check=cancelled_meanwhile) is None
    assert spawns.calls == []
    assert asteraix.load(cancelled.id).status == "cancelled"

    started = asteraix.create(request("started-elsewhere"))

    def started_meanwhile(gpu, min_free_mb):
        record = asteraix.load(started.id)
        record.pid = PID_NEVER
        asteraix.save(record)
        return free_gpu(gpu, min_free_mb)

    assert app_module.schedule_once(asteraix, settings, spawn=spawns,
                                    vram_check=started_meanwhile) is None
    assert spawns.calls == []
    assert asteraix.load(started.id).pid == PID_NEVER


@pytest.fixture
def bare_client(app, trainer_key):
    """The routes without the lifespan: no background tick, so a test can put a
    cancel inside a tick of its own."""
    return TestClient(app, client=LOOPBACK, headers={"X-API-Key": trainer_key})


def test_a_cancel_of_a_queued_job_is_never_undone_by_a_tick(app, bare_client, settings,
                                                            monkeypatch):
    """The caller is told `cancelled`; no tick that listed the job earlier may
    make it queued again, let alone start it."""
    store = store_of(app)
    tick = JobStore(store.root, host_id="asteraix")
    spawns = Spawns()

    # While the tick waits for VRAM: hold() rewrote the waiting reason from its
    # old copy — on nearly every tick, since the free MB change.
    waiting = store.create(request("waiting"))

    def cancelled_while_the_card_is_busy(gpu, min_free_mb):
        answer = bare_client.post(f"/jobs/{waiting.id}/cancel")
        assert answer.json()["status"] == "cancelled", answer.text
        raise PreflightError("GPU 1 has 812 MB free, need 20000 MB")

    assert app_module.schedule_once(tick, settings, spawn=spawns,
                                    vram_check=cancelled_while_the_card_is_busy) is None
    record = store.load(waiting.id)
    assert record.status == "cancelled" and record.queued_reason is None
    assert app_module.schedule_once(tick, settings, spawn=spawns, vram_check=free_gpu) is None

    # Between the listing and the claim: the cancel holds the claim now.
    listed = store.create(request("listed"))

    def cancelled_before_the_claim(gpu, min_free_mb):
        answer = bare_client.post(f"/jobs/{listed.id}/cancel")
        assert answer.json()["status"] == "cancelled", answer.text
        return free_gpu(gpu, min_free_mb)

    assert app_module.schedule_once(tick, settings, spawn=spawns,
                                    vram_check=cancelled_before_the_claim) is None
    assert store.load(listed.id).status == "cancelled"
    assert spawns.calls == []

    # While the tick is starting it — claim held, pid not saved yet: refused,
    # and a moment later an ordinary cancel of a running job.
    signalled = []
    monkeypatch.setattr(app_module.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(app_module.os, "killpg", lambda pgid, sig: signalled.append(pgid))
    starting = store.create(request("starting"))
    during_spawn = []

    def spawn_and_cancel(settings, job):
        during_spawn.append(bare_client.post(f"/jobs/{starting.id}/cancel"))
        return spawns(settings, job)

    started = app_module.schedule_once(tick, settings, spawn=spawn_and_cancel,
                                       vram_check=free_gpu)
    assert started is not None and started.id == starting.id
    assert during_spawn[0].status_code == 409
    assert "being started right now" in during_spawn[0].json()["detail"]
    assert store.load(starting.id).pid == os.getpid()
    again = bare_client.post(f"/jobs/{starting.id}/cancel")
    assert again.status_code == 200 and signalled == [os.getpid()]

    # A cancel whose write was lost after all: its claim says what was meant.
    lost = store.create(request("lost"))
    store.claim(lost.id, purpose=CLAIM_CANCEL)
    completed = store.reconcile(store.load(lost.id))
    assert completed.status == "cancelled"
    assert completed.error == "cancelled before it started"
    assert app_module.schedule_once(tick, settings, spawn=spawns, vram_check=free_gpu) is None
    assert spawns.calls == [("asteraix", starting.id)]


# ── another host's record, once nothing on that host watches it ────────────
def test_another_host_s_stuck_record_can_be_closed_by_hand(client, app, root, monkeypatch,
                                                           capsys):
    """After the cutover idhefix's trainer is disabled for good. A legacy record
    it left queued is never started, judged or cancelled here — by design — and
    held its model_id for ever, while the two 409s sent the caller to each other.
    The service names the way out, and the tool takes it without a signal."""
    idhefix = JobStore(root, host_id="idhefix")
    left = idhefix.create(request())  # BODY's model_id
    _strip_host(idhefix, left.id)
    for _ in range(3):
        app_module._schedule()
    assert store_of(app).load(left.id).status == "queued"
    assert app.state.spawn.calls == []

    for answer in (client.post(f"/jobs/{left.id}/cancel"), client.delete(f"/jobs/{left.id}")):
        assert answer.status_code == 409
        assert "is queued on host idhefix" in answer.json()["detail"]
    resubmit = client.post("/jobs", json=BODY)
    assert resubmit.status_code == 409
    detail = resubmit.json()["detail"]
    assert "cancel that job on host idhefix" in detail
    assert close_job.close_command(left.id) in detail, "the resubmit names no way out"

    monkeypatch.setenv("ATR_TRAIN_HOST_ID", "asteraix")
    args = [left.id, "--root", str(root), "--reason", "idhefix's trainer is disabled (cutover)"]
    before = idhefix.paths(left.id).job_json.read_bytes()
    assert close_job.main(args) == close_job.EXIT_OK
    assert "add --yes" in capsys.readouterr().out
    assert idhefix.paths(left.id).job_json.read_bytes() == before, "a dry run wrote"

    assert close_job.main([*args, "--yes"]) == close_job.EXIT_OK
    closed = store_of(app).load(left.id)
    assert closed.status == "cancelled"
    assert closed.host is None, "the owner was rewritten"
    assert "closed by hand on asteraix: host idhefix no longer runs it" in closed.error
    assert "(cutover)" in closed.error
    assert client.post("/jobs", json=BODY).status_code == 202, "the model_id is still taken"
    assert client.delete(f"/jobs/{left.id}").status_code == 200

    # A Slurm job scancelled while training — on the preemptable template the
    # runner took SIGTERM for a preemption and left `training`.
    slurm = _running(idhefix, idhefix.create(request("scancelled"), host="ubelix").id,
                     PID_NEVER)
    assert "with scancel" in client.post(f"/jobs/{slurm.id}/cancel").json()["detail"]
    assert close_job.main([slurm.id, "--root", str(root), "--reason", "sacct: CANCELLED",
                           "--yes"]) == close_job.EXIT_OK
    gone = store_of(app).load(slurm.id)
    assert gone.status == "failed" and "the Slurm job no longer runs it" in gone.error


def test_closing_by_hand_refuses_what_it_cannot_vouch_for(root, tmp_path, monkeypatch,
                                                          capsys):
    asteraix = JobStore(root, host_id="asteraix")
    idhefix = JobStore(root, host_id="idhefix")
    lines: list[str] = []

    def close(job_id, reason="checked on idhefix", out=lines.append):
        return close_job.close_job(asteraix, job_id, reason, write=True, out=out)

    # This host's own job: the service cancels those, and signals the runner.
    mine = _running(asteraix, asteraix.create(request("mine")).id, PID_NEVER)
    assert close(mine.id) == close_job.EXIT_REFUSED
    assert f"POST /jobs/{mine.id}/cancel" in lines[-1]
    assert asteraix.load(mine.id).status == "training"

    # A queued job its own scheduler is starting: that host is not gone.
    starting = idhefix.create(request("starting"))
    idhefix.claim(starting.id)
    assert close(starting.id) == close_job.EXIT_REFUSED
    assert "claimed for spawning by idhefix" in lines[-1]
    assert idhefix.load(starting.id).status == "queued"

    # A record that changes while the tool runs is still being written — and a
    # claim the tool took for it is given back.
    for busy in (_running(idhefix, idhefix.create(request("busy")).id, PID_NEVER),
                 idhefix.create(request("busy-queued"))):
        def written_meanwhile(line, job_id=busy.id):
            lines.append(line)
            if line.startswith("job      "):
                idhefix.save(idhefix.load(job_id))

        assert close(busy.id, out=written_meanwhile) == close_job.EXIT_REFUSED
        assert "changed while this ran" in lines[-1]
        assert idhefix.load(busy.id).status == busy.status
        assert not idhefix.is_claimed(busy.id)

    # No reason, no close. A finished job is left as it is.
    assert close(busy.id, reason="  ") == close_job.EXIT_USAGE
    done = idhefix.fail(idhefix.create(request("done")), "finished")
    assert close(done.id) == close_job.EXIT_OK and "already failed" in lines[-1]
    assert idhefix.load(done.id).error == "finished"

    # Without a configured host id, "this host" would be the bare hostname and
    # this host's own jobs would pass as another's.
    monkeypatch.delenv("ATR_TRAIN_HOST_ID")
    monkeypatch.chdir(tmp_path)  # no .env here
    assert close_job.main([mine.id, "--root", str(root), "--reason", "x",
                           "--yes"]) == close_job.EXIT_USAGE
    assert "ATR_TRAIN_HOST_ID is not set" in capsys.readouterr().err
    assert asteraix.load(mine.id).status == "training"


# ── the cleanup's other two guards (#15 review) ─────────────────────────────
def test_startup_cleanup_leaves_registered_weights_alone(app, settings, root, trainer_key, logs,
                                                         tmp_path):
    """The curated-clash failure tells the operator to copy the weights under a
    new id and register that by hand. That writes the YAML, never metadata.json."""
    rescued = _weights_dir(settings, "kraken-rescued-v1")
    write_registration(settings.registry_root, {
        "id": "kraken-rescued-v1", "engine": "kraken",
        "local_path": str(rescued / "kraken-rescued-v1.mlmodel")})
    # Registered under another id: found through local_path.
    copied = _weights_dir(settings, "copied-by-hand")
    write_registration(settings.registry_root, {
        "id": "kraken-other-v1", "engine": "kraken",
        "local_path": str(copied / "copied-by-hand.mlmodel")})
    # A registration that no longer validates still names its directory.
    broken = _weights_dir(settings, "kraken-broken-v1")
    registration_path(settings.registry_root, "kraken-broken-v1").write_text(
        "id: [unclosed\n", encoding="utf-8")
    orphan = _weights_dir(settings, "nobody-registered-this")
    for directory in (rescued, copied, broken, orphan):
        _age(directory, hours=48)

    _start(app, trainer_key)

    assert (rescued / "kraken-rescued-v1.mlmodel").read_bytes() == b"WEIGHTS"
    assert copied.is_dir() and broken.is_dir()
    assert not orphan.exists(), "the cleanup did not run at all"
    assert any("keeping kraken-rescued-v1" in line and "registration kraken-rescued-v1" in line
               for line in logs), logs
    assert any("keeping copied-by-hand" in line and "registration kraken-other-v1" in line
               for line in logs), logs

    # A registry that cannot be read protects nothing it names — so nothing goes.
    later = _weights_dir(settings, "another-orphan")
    _age(later, hours=48)
    store = JobStore(root, host_id="asteraix")
    assert app_module._cleanup_orphaned_weights(
        settings.trained_root, store, min_age_h=24, registry_root=tmp_path / "unmounted") == 0
    assert later.is_dir()
    assert trained_dir(settings.registry_root).is_dir()
    assert app_module._cleanup_orphaned_weights(
        settings.trained_root, store, min_age_h=24, registry_root=settings.registry_root) == 1


def test_startup_cleanup_reads_the_newest_mtime_of_the_directory_and_its_entries(settings,
                                                                                 root):
    """Either time alone is wrong. A retried registration rewrites the weights in
    place (mkdir(exist_ok=True), then copyfile): the file's mtime moves, the
    directory's does not. A directory that just received an old file is new
    while its entry is old. An empty directory has only its own time."""
    store = JobStore(root, host_id="asteraix")
    old = time.time() - 48 * 3600

    rewritten = _weights_dir(settings, "rewritten")
    weights = rewritten / "rewritten.mlmodel"
    os.utime(weights, (old, old))
    weights.write_bytes(b"RETRIED")
    os.utime(rewritten, (old, old))

    added = _weights_dir(settings, "added")
    os.utime(added / "added.mlmodel", (old, old))

    empty = settings.trained_root / "empty"
    empty.mkdir()

    stale = _weights_dir(settings, "stale")
    _age(stale, hours=48)

    removed = app_module._cleanup_orphaned_weights(
        settings.trained_root, store, min_age_h=24, registry_root=settings.registry_root)
    assert not stale.exists() and removed == 1
    assert rewritten.is_dir() and added.is_dir() and empty.is_dir()


def test_a_fast_runner_s_first_write_is_not_undone_by_the_pid_save(tmp_path, root, venvs):
    """The runner saves its pid and `preparing` before the scheduler saves the pid."""
    settings = trainer_settings(tmp_path, venvs, "asteraix")
    store = JobStore(root, host_id="asteraix")
    job = store.create(request())

    def fast_runner(settings, job):
        mine = store.load(job.id)
        mine.pid = os.getpid()
        store.advance(mine, "preparing")
        return mine.pid

    started = app_module.schedule_once(store, settings, spawn=fast_runner, vram_check=free_gpu)
    on_disk = store.load(job.id)
    assert on_disk.status == "preparing", "the pid save put a started run back to queued"
    assert on_disk.pid == os.getpid()
    assert started.status == "preparing"


def test_a_slow_runner_still_gets_its_pid_recorded(tmp_path, root, venvs):
    settings = trainer_settings(tmp_path, venvs, "asteraix")
    store = JobStore(root, host_id="asteraix")
    job = store.create(request())
    app_module.schedule_once(store, settings, spawn=lambda s, j: os.getpid(),
                             vram_check=free_gpu)
    on_disk = store.load(job.id)
    assert on_disk.status == "queued" and on_disk.pid == os.getpid()
    assert on_disk.host == "asteraix"

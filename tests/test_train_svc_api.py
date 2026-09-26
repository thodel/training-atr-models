"""Trainer service API + scheduler (#34).

The spawn and VRAM-check seams are replaced on ``app.state`` so nothing here
touches a GPU or starts a process.
"""

import os
import socket
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from atr_training.contracts import DatasetSpec, Metrics, TrainRequest
from atr_training.hf_source import VerificationUnavailable
from atr_training.jobstore import JobStore

from kraken_train_svc import app as app_module
from atr_training.preflight import GpuInfo, PreflightError
from atr_training.settings import TrainerSettings

REPO = "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi"
BODY = {
    "model_id": "kraken-thun-missiven-v1",
    "dataset": {
        "hf_repo": REPO,
        "train_projects": ["GT_Thun-Training_(TEST-DEMO)"],
        "eval_projects": ["GT_Thun-Test_(DEMO_TEST)"],
    },
}


#: A pid that cannot exist (pid_t is int32; Linux pid_max tops out far below this),
#: so reconcile() reliably sees the runner as gone.
PID_NEVER = 2**31 - 1


class FakeSpawn:
    """Stands in for the detached runner. Reports THIS process's pid, so the job
    looks alive to reconcile() — the tests that need a dead runner set PID_NEVER."""

    def __init__(self, pid: int | None = None) -> None:
        self.pid = pid or os.getpid()
        self.calls: list[str] = []

    def __call__(self, settings, job):
        self.calls.append(job.id)
        return self.pid


def free_gpu(gpu, min_free_mb):
    return GpuInfo(index=gpu, free_mb=40000, total_mb=46068)


def busy_gpu(gpu, min_free_mb):
    raise PreflightError(f"GPU {gpu} has 2000 MB free, need {min_free_mb} MB")


@pytest.fixture
def venvs(tmp_path: Path) -> Path:
    """Stand-in interpreters for both backends.

    Pointed at tmp_path rather than the real ``.venvs/``: submit refuses an engine
    whose venv is not built, and a test suite that only passes on a machine which
    happens to have provisioned the engines is not a test suite.
    """
    root = tmp_path / "venvs"
    for name in ("kraken-train", "vlm-train"):
        (root / name / "bin").mkdir(parents=True)
        (root / name / "bin" / "python").touch()
    return root


@pytest.fixture
def settings(tmp_path: Path, venvs: Path) -> TrainerSettings:
    return TrainerSettings(
        jobs_root=tmp_path / "training",
        trained_root=tmp_path / "trained",
        venvs_root=venvs,
        # Isolated deliberately: the default is ~/atr-cache/checkpoints, so a test
        # that writes checkpoints would land in the developer's home directory and
        # collide with any other test whose job id shares its second.
        checkpoint_root=tmp_path / "checkpoints",
        min_free_disk_gb=0.0,
    )


@pytest.fixture
def spawn() -> FakeSpawn:
    return FakeSpawn()


@pytest.fixture
def app():
    """The service module's app object, so a test can install its own seams."""
    return app_module.app


#: Starlette's test client reports the host "testclient", which the access
#: middleware refuses as it would any peer it cannot place (#13).
LOOPBACK = ("127.0.0.1", 50000)


@pytest.fixture
def client(settings: TrainerSettings, spawn: FakeSpawn, trainer_key: str):
    """The app behind its real middleware: a loopback caller with the key."""
    app = app_module.app
    app.state.settings = settings
    app.state.store = JobStore(settings.jobs_root, host_id=settings.host_id)
    app.state.spawn = spawn
    app.state.vram_check = free_gpu
    with TestClient(app, client=LOOPBACK, headers={"X-API-Key": trainer_key}) as c:
        yield c
    for attr in ("settings", "store", "spawn", "vram_check", "verify_spec"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


def store_of(client) -> JobStore:
    return client.app.state.store


# ── submit ──────────────────────────────────────────────────────────────────
def test_submit_queues_and_starts_a_job(client, spawn):
    resp = client.post("/jobs", json=BODY)
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    assert spawn.calls == [job_id]
    assert store_of(client).load(job_id).pid == spawn.pid


def test_submitted_job_is_readable(client):
    job_id = client.post("/jobs", json=BODY).json()["job_id"]
    body = client.get(f"/jobs/{job_id}").json()
    assert body["request"]["model_id"] == "kraken-thun-missiven-v1"
    assert body["request"]["params"]["batch_size"] == 256
    assert body["request"]["params"]["schedule"] == "cosine"


def test_invalid_request_is_rejected(client):
    bad = {**BODY, "model_id": "Not A Slug"}
    assert client.post("/jobs", json=bad).status_code == 422


def test_a_dataset_selecting_nothing_is_refused_at_submit(client):
    """Was accepted until #46, on the grounds that the pipeline would fail it
    loudly. It still would — but hours later, after the job queued and started
    downloading. The check is free and structural, so it happens here."""
    resp = client.post("/jobs", json={"model_id": "m", "dataset": {"hf_repo": REPO}})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["valid"] is False and detail["checked"] is True
    assert "train_projects" in detail["errors"][0]


def test_a_project_on_both_sides_of_the_split_is_refused_at_submit(client):
    resp = client.post("/jobs", json={
        "model_id": "m",
        "dataset": {"hf_repo": REPO, "train_projects": ["a"], "eval_projects": ["a"]},
    })
    assert resp.status_code == 400
    assert "both train and eval" in resp.json()["detail"]["errors"][0]


def test_a_spec_the_hub_rejects_is_refused_with_every_problem_at_once(client, app):
    app.state.verify_spec = lambda spec, settings, **kw: [
        "project 'GT_Thun-Trainig' not found under data/train/",
        "no .parquet files found",
    ]
    resp = client.post("/jobs", json=BODY)
    assert resp.status_code == 400
    assert len(resp.json()["detail"]["errors"]) == 2


def test_an_unreachable_hub_queues_the_job_rather_than_refusing_it(client, app):
    """"Could not check" is not "your spec is wrong". The job downloads when it
    starts, possibly hours later, so a hiccup now must not cost the submission —
    but the record says it went in unverified."""
    def unreachable(spec, settings, **kw):
        raise VerificationUnavailable("ConnectionError: hub unreachable")

    app.state.verify_spec = unreachable
    resp = client.post("/jobs", json=BODY)
    assert resp.status_code == 202
    assert resp.json()["dataset_verified"] is False
    assert "hub" in resp.json()["unverified_reason"]


def test_a_verified_submission_says_so(client, app):
    app.state.verify_spec = lambda spec, settings, **kw: []
    resp = client.post("/jobs", json=BODY)
    assert resp.status_code == 202
    assert resp.json()["dataset_verified"] is True


def test_verify_answers_without_queueing_anything(client, app):
    app.state.verify_spec = lambda spec, settings, **kw: ["project 'typo' not found"]
    resp = client.post("/jobs/verify", json=BODY)
    assert resp.status_code == 200          # an answered question, not a failed request
    assert resp.json()["valid"] is False
    assert client.get("/jobs").json()["jobs"] == []


def test_full_disk_refuses_the_submission(client, settings):
    settings.min_free_disk_gb = 10**9  # more than any disk
    resp = client.post("/jobs", json=BODY)
    assert resp.status_code == 507
    assert "free" in resp.json()["detail"]


# ── queueing ────────────────────────────────────────────────────────────────
def test_second_job_waits_for_the_first(client, spawn):
    first = client.post("/jobs", json=BODY).json()["job_id"]
    store = store_of(client)
    job = store.load(first)
    store.advance(job, "preparing")  # now running

    second = client.post("/jobs", json={**BODY, "model_id": "second-model"}).json()
    assert second["status"] == "queued"
    assert first in second["queued_reason"]
    assert spawn.calls == [first]  # not started


def test_a_busy_gpu_holds_the_queue_without_failing_it(client, spawn):
    client.app.state.vram_check = busy_gpu
    body = client.post("/jobs", json=BODY).json()
    assert body["status"] == "queued"
    assert "2000 MB free" in body["queued_reason"]
    assert spawn.calls == []

    # ...and it starts once the GPU frees up
    client.app.state.vram_check = free_gpu
    started = app_module.schedule_once(store_of(client), client.app.state.settings, spawn=spawn,
                                       vram_check=free_gpu)
    assert started.id == body["job_id"] and started.queued_reason is None


def test_queue_is_fifo(client, spawn):
    store = store_of(client)
    client.app.state.vram_check = busy_gpu
    first = client.post("/jobs", json=BODY).json()["job_id"]
    second = client.post("/jobs", json={**BODY, "model_id": "second-model"}).json()["job_id"]
    assert (first, second) != (None, None)
    app_module.schedule_once(store, client.app.state.settings, spawn=spawn, vram_check=free_gpu)
    assert spawn.calls == [first]


# ── listing, logs ───────────────────────────────────────────────────────────
def test_list_is_newest_first(client):
    a = client.post("/jobs", json=BODY).json()["job_id"]
    store = store_of(client)
    store.advance(store.load(a), "preparing")
    b = client.post("/jobs", json={**BODY, "model_id": "second-model"}).json()["job_id"]
    assert [j["id"] for j in client.get("/jobs").json()["jobs"]] == sorted([a, b], reverse=True)


def test_list_limit_keeps_newest_first(client):
    a = client.post("/jobs", json=BODY).json()["job_id"]
    store = store_of(client)
    store.advance(store.load(a), "preparing")
    b = client.post("/jobs", json={**BODY, "model_id": "second-model"}).json()["job_id"]
    newest_first = sorted([a, b], reverse=True)
    body = client.get("/jobs", params={"limit": 1}).json()
    assert [j["id"] for j in body["jobs"]] == newest_first[:1]
    body = client.get("/jobs", params={"limit": 5}).json()
    assert [j["id"] for j in body["jobs"]] == newest_first


def test_list_fields_summary_returns_only_the_summary_keys(client):
    client.post("/jobs", json=BODY)
    jobs = client.get("/jobs", params={"fields": "summary"}).json()["jobs"]
    assert len(jobs) == 1
    assert set(jobs[0]) == {"id", "status", "stage", "created_at", "queued_reason", "error"}


def test_list_fields_summary_and_limit_combine(client):
    a = client.post("/jobs", json=BODY).json()["job_id"]
    b = client.post("/jobs", json={**BODY, "model_id": "second-model"}).json()["job_id"]
    jobs = client.get("/jobs", params={"limit": 1, "fields": "summary"}).json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["id"] == sorted([a, b], reverse=True)[0]
    assert set(jobs[0]) == {"id", "status", "stage", "created_at", "queued_reason", "error"}


def test_list_without_params_returns_the_full_shape(client):
    job_id = client.post("/jobs", json=BODY).json()["job_id"]
    default = client.get("/jobs").json()["jobs"][0]
    full = client.get("/jobs", params={"fields": "full"}).json()["jobs"][0]
    assert default == full
    assert default["id"] == job_id
    assert "request" in default and "stages" in default and "metrics" in default


def test_list_limit_below_one_is_rejected(client):
    assert client.get("/jobs", params={"limit": 0}).status_code == 422
    assert client.get("/jobs", params={"fields": "nope"}).status_code == 422


def test_log_tail(client):
    job_id = client.post("/jobs", json=BODY).json()["job_id"]
    log = store_of(client).paths(job_id).log("train")
    log.write_text("\n".join(f"epoch {i}" for i in range(300)), encoding="utf-8")
    body = client.get(f"/jobs/{job_id}/log", params={"stage": "train", "lines": 5}).json()
    assert body["lines"] == [f"epoch {i}" for i in range(295, 300)]


def test_missing_log_is_a_404(client):
    job_id = client.post("/jobs", json=BODY).json()["job_id"]
    assert client.get(f"/jobs/{job_id}/log", params={"stage": "train"}).status_code == 404


def test_unknown_job_is_a_404(client):
    assert client.get("/jobs/20260806T120000Z-nope").status_code == 404
    assert client.post("/jobs/20260806T120000Z-nope/cancel").status_code == 404


def test_health_reports_the_queue(client):
    client.post("/jobs", json=BODY)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["gpu"] == 1
    assert body["jobs"]["total"] == 1




def test_host_reports_free_disk_for_the_four_volumes_and_ram(
        client, settings, monkeypatch, tmp_path):
    """#40: disk free per volume, RAM from an injectable /proc/meminfo."""
    import kraken_train_svc.app as app_module

    for attr in ("jobs_root", "checkpoint_root", "artefact_cache_root", "trained_root"):
        path = tmp_path / attr
        path.mkdir(parents=True, exist_ok=True)
        setattr(settings, attr, path)
    client.app.state.settings = settings
    monkeypatch.setattr(app_module, "free_disk_gb", lambda p: 100.0)

    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       32000000 kB\n"
                       "MemFree:         1000000 kB\n"
                       "MemAvailable:    8000000 kB\n", encoding="ascii")
    monkeypatch.setattr(app_module, "MEMINFO_PATH", meminfo)

    body = client.get("/host").json()

    assert body["host"] == socket.gethostname()
    assert sorted(body["volumes"]) == ["artefact_cache_root", "checkpoint_root",
                                       "jobs_root", "trained_root"]
    for volume in body["volumes"].values():
        assert volume["free_gb"] == 100.0
        assert volume["path"].startswith(str(tmp_path))
    assert body["ram"] == {"total_gb": 32.0, "available_gb": 8.0}


def test_host_answers_without_ram_when_proc_cannot_be_read(
        client, settings, monkeypatch, tmp_path):
    """Partial data, not a 500 (#40). A real /proc never produces this case,
    which is why the path is a module constant a test can point elsewhere."""
    import kraken_train_svc.app as app_module

    monkeypatch.setattr(app_module, "free_disk_gb", lambda p: 12.5)
    monkeypatch.setattr(app_module, "MEMINFO_PATH", tmp_path / "not-here")

    response = client.get("/host")

    assert response.status_code == 200
    body = response.json()
    assert "ram" not in body            # unknown is absent, never invented
    assert body["volumes"], "the volumes still answer"


def test_host_omits_a_volume_it_cannot_measure(client, settings, monkeypatch, tmp_path):
    """One unreadable path must not cost the other three."""
    import kraken_train_svc.app as app_module

    for attr in ("jobs_root", "checkpoint_root", "artefact_cache_root", "trained_root"):
        setattr(settings, attr, tmp_path / attr)
    client.app.state.settings = settings

    def measure(path):
        if str(path).endswith("trained_root"):
            raise OSError("no such volume")
        return 7.0

    monkeypatch.setattr(app_module, "free_disk_gb", measure)
    monkeypatch.setattr(app_module, "MEMINFO_PATH", tmp_path / "not-here")

    body = client.get("/host").json()

    assert "trained_root" not in body["volumes"]
    assert sorted(body["volumes"]) == ["artefact_cache_root", "checkpoint_root",
                                       "jobs_root"]

def test_host_omits_unreadable_volumes(client, settings, monkeypatch, tmp_path):
    """Volumes that raise OSError in free_disk_gb are omitted from the response."""
    import unittest.mock
    from atr_training.preflight import free_disk_gb as _real_free_disk_gb

    settings.jobs_root = tmp_path / "jobs"
    settings.checkpoint_root = tmp_path / "checkpoints"
    settings.artefact_cache_root = tmp_path / "nonexistent"
    settings.trained_root = tmp_path / "also-nonexistent"
    for d in (settings.jobs_root, settings.checkpoint_root):
        d.mkdir(parents=True)
    client.app.state.settings = settings

    FAILED = {"nonexistent", "also-nonexistent"}

    def fake_free_disk(path):
        # free_disk_gb traverses up to the nearest existing parent before calling
        # disk_usage.  Check the original path name to decide failure — both
        # non-existent subdirs resolve to the same existing parent.
        if Path(path).name in FAILED:
            raise OSError(f"not accessible: {path}")
        return _real_free_disk_gb(path)

    with unittest.mock.patch("kraken_train_svc.app.free_disk_gb", side_effect=fake_free_disk):
        resp = client.get("/host")

    assert resp.status_code == 200
    body = resp.json()
    # Only the two whose free_disk_gb succeeded survive
    assert set(body["volumes"].keys()) == {"jobs_root", "checkpoint_root"}


def test_host_returns_ram_from_proc_meminfo(client, settings, monkeypatch, tmp_path):
    import unittest.mock

    for d in (settings.jobs_root, settings.checkpoint_root,
              settings.artefact_cache_root, settings.trained_root):
        d.mkdir(parents=True, exist_ok=True)
    client.app.state.settings = settings

    fake_meminfo = "MemTotal:       64000000 kB\nMemAvailable:   48000000 kB\n"
    with unittest.mock.patch("pathlib.Path.read_text", return_value=fake_meminfo):
        resp = client.get("/host")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ram"]["total_gb"] == 64.0
    assert body["ram"]["available_gb"] == 48.0


def test_host_fails_gracefully_when_proc_meminfo_unreadable(client, settings, tmp_path, monkeypatch):
    for d in (settings.jobs_root, settings.checkpoint_root,
              settings.artefact_cache_root, settings.trained_root):
        d.mkdir(parents=True, exist_ok=True)
    client.app.state.settings = settings

    def read_text_that_fails(*args, **kwargs):
        raise OSError("no /proc/meminfo")
    import unittest.mock
    with unittest.mock.patch("pathlib.Path.read_text", side_effect=read_text_that_fails):
        resp = client.get("/host")

    assert resp.status_code == 200
    body = resp.json()
    assert "ram" not in body  # gracefully absent when unreadable


# ── cancel / delete ─────────────────────────────────────────────────────────
def test_cancel_a_queued_job(client, spawn):
    client.app.state.vram_check = busy_gpu
    job_id = client.post("/jobs", json=BODY).json()["job_id"]
    body = client.post(f"/jobs/{job_id}/cancel").json()
    assert body["status"] == "cancelled"
    assert "before it started" in body["error"]


def test_cancel_signals_the_process_group(client, monkeypatch):
    job_id = client.post("/jobs", json=BODY).json()["job_id"]
    sent = {}
    monkeypatch.setattr(app_module.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(app_module.os, "killpg", lambda pgid, sig: sent.update(pgid=pgid, sig=sig))
    client.post(f"/jobs/{job_id}/cancel")
    assert sent["pgid"] == os.getpid()


def test_cancelling_a_finished_job_is_a_conflict(client):
    job_id = client.post("/jobs", json=BODY).json()["job_id"]
    store = store_of(client)
    job = store.load(job_id)
    store.fail(job, "already failed")
    assert client.post(f"/jobs/{job_id}/cancel").status_code == 409


def test_delete_refuses_a_running_job(client):
    job_id = client.post("/jobs", json=BODY).json()["job_id"]
    store = store_of(client)
    store.advance(store.load(job_id), "preparing")
    assert client.delete(f"/jobs/{job_id}").status_code == 409


def test_delete_drops_artifacts_but_keeps_the_record(client):
    job_id = client.post("/jobs", json=BODY).json()["job_id"]
    store = store_of(client)
    (store.paths(job_id).pages / "p.jpg").write_bytes(b"x")
    job = store.load(job_id)
    job.metrics = Metrics(cer=0.05)
    store.save(job)
    store.fail(store.load(job_id), "done enough")

    assert client.delete(f"/jobs/{job_id}").status_code == 200
    assert not store.paths(job_id).data.exists()
    assert store.load(job_id).metrics.cer == 0.05


# ── restart reconciliation ──────────────────────────────────────────────────
def test_a_job_whose_runner_died_is_failed_not_left_training(client, spawn):
    job_id = client.post("/jobs", json=BODY).json()["job_id"]
    store = store_of(client)
    store.advance(store.load(job_id), "preparing")
    store.advance(store.load(job_id), "compiling")
    job = store.advance(store.load(job_id), "training")
    job.pid = PID_NEVER  # the runner was killed (OOM, reboot, wrong restart)
    store.save(job)

    body = client.get(f"/jobs/{job_id}").json()
    assert body["status"] == "failed"
    assert str(PID_NEVER) in body["error"] and "training" in body["error"]


def test_reconcile_frees_the_queue_for_the_next_job(client, spawn):
    dead = client.post("/jobs", json=BODY).json()["job_id"]
    store = store_of(client)
    job = store.advance(store.load(dead), "preparing")
    job.pid = PID_NEVER
    store.save(job)
    # Submitting schedules straight away, so this is where the dead job is
    # reconciled and the next one starts — no second pass needed.
    nxt = client.post("/jobs", json={**BODY, "model_id": "second-model"}).json()["job_id"]

    assert store.load(dead).status == "failed"
    assert spawn.calls == [dead, nxt]


# ── two backends, one supervisor ────────────────────────────────────────────
VLM_BODY = {
    "engine": "vllm",
    "model_id": "qwen3vl-thun-v1",
    "dataset": {"hf_repo": REPO, "train_projects": ["GT_Thun-Training_(TEST-DEMO)"]},
}


def test_a_vlm_job_is_accepted_by_the_same_endpoint(client, spawn):
    resp = client.post("/jobs", json=VLM_BODY)
    assert resp.status_code == 202
    job = store_of(client).load(resp.json()["job_id"])
    assert job.request.engine == "vllm"
    assert job.request.params.granularity == "line"
    assert job.request.base_model == "Qwen/Qwen3-VL-8B-Instruct"


def test_both_backends_share_one_queue(client, spawn):
    """One GPU, so one job at a time — regardless of which engine each job is for.
    Two services would each think they were the only one training."""
    kraken = client.post("/jobs", json=BODY).json()["job_id"]
    store = store_of(client)
    store.advance(store.load(kraken), "preparing")

    vlm = client.post("/jobs", json=VLM_BODY).json()
    assert vlm["status"] == "queued"
    assert kraken in vlm["queued_reason"]
    assert spawn.calls == [kraken]


def test_a_vlm_job_needs_more_free_vram_than_a_kraken_job(client, settings):
    """A card with room for a kraken run has not necessarily got room for a
    QLoRA fine-tune of an 8B; the gate is per engine."""
    seen = []

    def record(gpu, min_free_mb):
        seen.append(min_free_mb)
        return GpuInfo(index=gpu, free_mb=40000, total_mb=46068)

    client.app.state.vram_check = record
    client.post("/jobs", json=VLM_BODY)
    assert seen == [settings.vlm_min_free_vram_mb]
    assert settings.vlm_min_free_vram_mb > settings.min_free_vram_mb


def test_a_job_is_spawned_with_its_own_engine_s_interpreter(client, settings, monkeypatch):
    """kraken and the VLM trainer cannot share a dependency tree, so they must not
    share an interpreter — the supervisor imports neither."""
    launched: list[list[str]] = []

    class FakeProc:
        pid = 4242

    monkeypatch.setattr(app_module.subprocess, "Popen",
                        lambda cmd, **kw: launched.append(cmd) or FakeProc())
    delattr(client.app.state, "spawn")  # exercise the real _spawn

    store = store_of(client)
    vlm = client.post("/jobs", json=VLM_BODY).json()["job_id"]
    assert launched[0][0] == str(settings.venvs_root / "vlm-train" / "bin" / "python")
    assert launched[0][2] == "vlm_train_svc.runner"

    store.fail(store.load(vlm), "make room for the next one")
    client.post("/jobs", json=BODY)
    assert launched[1][0] == str(settings.venvs_root / "kraken-train" / "bin" / "python")
    assert launched[1][2] == "kraken_train_svc.runner"


def test_a_spawned_job_is_not_spawned_again_before_its_runner_reports(client, settings, spawn):
    """A job stays 'queued' until the detached runner writes its first status;
    scheduling again in that window would put two runners on one GPU."""
    store = store_of(client)
    job_id = client.post("/jobs", json=BODY).json()["job_id"]
    assert spawn.calls == [job_id]
    assert store.load(job_id).status == "queued" and store.load(job_id).pid is not None

    app_module.schedule_once(store, settings, spawn=spawn, vram_check=free_gpu)
    assert spawn.calls == [job_id]  # not started twice


def test_a_runner_that_died_before_reporting_does_not_block_the_queue(client, settings, spawn):
    """The other half of the same rule: 'queued with a pid' means spawned, so a
    dead pid there is a dead run, not a job politely waiting its turn."""
    store = store_of(client)
    dead = client.post("/jobs", json=BODY).json()["job_id"]
    job = store.load(dead)
    job.pid = PID_NEVER
    store.save(job)
    nxt = client.post("/jobs", json={**BODY, "model_id": "second-model"}).json()["job_id"]

    assert store.load(dead).status == "failed"
    assert spawn.calls == [dead, nxt]


def test_a_backend_whose_venv_is_missing_is_refused_at_submit(client, settings):
    """Named now, with the command that fixes it — not as a traceback inside a
    detached child two ticks later."""
    (settings.venvs_root / "vlm-train" / "bin" / "python").unlink()
    resp = client.post("/jobs", json=VLM_BODY)
    assert resp.status_code == 503
    assert "make_venvs.sh vlm-train" in resp.json()["detail"]
    assert client.post("/jobs", json=BODY).status_code == 202  # kraken unaffected


def test_health_reports_which_backends_this_box_can_actually_run(client, settings):
    (settings.venvs_root / "vlm-train" / "bin" / "python").unlink()
    backends = client.get("/health").json()["backends"]
    assert backends["kraken"]["available"] is True
    assert backends["vllm"]["available"] is False
    assert backends["vllm"]["runner"] == "vlm_train_svc.runner"


def test_a_job_that_cannot_be_spawned_fails_instead_of_queueing_forever(client, settings):
    """The scheduler would otherwise log the same error every 10 s while the
    record still claimed the job was queued."""
    store = store_of(client)
    job_id = client.post("/jobs", json=VLM_BODY).json()["job_id"]
    store.advance(store.load(job_id), "preparing")  # occupy the queue
    second = client.post("/jobs", json={**VLM_BODY, "model_id": "second-vlm"}).json()["job_id"]

    (settings.venvs_root / "vlm-train" / "bin" / "python").unlink()
    delattr(client.app.state, "spawn")
    store.fail(store.load(job_id), "make room")
    app_module.schedule_once(store, settings, vram_check=free_gpu)

    failed = store.load(second)
    assert failed.status == "failed"
    assert "could not start the vllm runner" in failed.error


def test_request_defaults_survive_the_wire(client):
    """A minimal body still carries the agreed kraken+ recipe."""
    job_id = client.post("/jobs", json={
        "model_id": "minimal", "dataset": {"hf_repo": REPO, "train_projects": ["P"]},
    }).json()["job_id"]
    params = store_of(client).load(job_id).request.params
    assert params.spec.startswith("[256,64,0,1 Cr4,2,8,4,2")
    assert (params.lrate, params.quit, params.weights_format) == (1e-4, "fixed", "coreml")
    assert TrainRequest(model_id="x", dataset=DatasetSpec(hf_repo=REPO)).params == params


# ── verify_only must not act (#59) ───────────────────────────────────────────
def test_verify_only_does_not_queue_anything(client, spawn):
    """The incident: `POST :8204/jobs?verify_only=true` queued a multi-day run.
    FastAPI drops query params a route does not declare, so the flag whose whole
    purpose is 'change nothing' was silently discarded."""
    resp = client.post("/jobs", params={"verify_only": "true"}, json=BODY)
    assert resp.status_code == 200
    assert "job_id" not in resp.json()
    assert resp.json()["valid"] is True
    assert store_of(client).list() == []      # nothing created
    assert spawn.calls == []                  # nothing started


def test_verify_only_reports_an_invalid_spec_without_failing_the_request(client):
    """'Is this spec good?' and 'did my request fail?' are different questions."""
    resp = client.post("/jobs", params={"verify_only": "true"},
                       json={**BODY, "dataset": {**BODY["dataset"], "train_projects": []}})
    assert resp.status_code in (200, 400)
    assert store_of(client).list() == []


# ── one live job per model_id (#56) ──────────────────────────────────────────
def test_a_second_live_job_for_the_same_model_id_is_refused(client, spawn):
    """Job ids are de-duplicated; model_ids were not — and the model_id is the
    directory and registry name, so the later run overwrites the earlier one's
    registered weights."""
    first = client.post("/jobs", json=BODY).json()["job_id"]
    resp = client.post("/jobs", json=BODY)
    assert resp.status_code == 409
    assert first in resp.json()["detail"]
    assert BODY["model_id"] in resp.json()["detail"]
    assert len(store_of(client).list()) == 1


def test_the_model_id_is_free_again_once_the_job_is_terminal(client):
    """A finished run must not block retraining the same model."""
    first = client.post("/jobs", json=BODY).json()["job_id"]
    store = store_of(client)
    store.fail(store.load(first), "earlier attempt failed")
    assert client.post("/jobs", json=BODY).status_code == 202


def test_a_different_model_id_is_unaffected(client):
    client.post("/jobs", json=BODY)
    assert client.post("/jobs", json={**BODY, "model_id": "other-model"}).status_code == 202


# ── a curated id is not a trained model_id (#14) ────────────────────────────
def test_a_curated_model_id_is_refused_before_a_job_exists(client, app, spawn):
    """The gateway skips trained/<id>.yaml when the id is curated, and its
    promotion gate answers that id with the curated weights: a day of GPU for a
    model nobody could reach, recorded as promoted. The overlay's merge() used
    to refuse the shadowing; submit is the cheapest place to refuse it now."""
    from atr_training.shared_registry import BaseEntry, SharedRegistry

    app.state.registry = SharedRegistry(
        [BaseEntry(id=BODY["model_id"], engine="kraken", zenodo_id="10.5281/zenodo.1")],
        path=Path("/share/registry/models.yaml"))
    try:
        resp = client.post("/jobs", json=BODY)
    finally:
        del app.state.registry
    assert resp.status_code == 409
    assert "curated id" in resp.json()["detail"]
    assert "/share/registry/models.yaml" in resp.json()["detail"]
    assert store_of(client).list() == [] and spawn.calls == []


def test_an_unreadable_curated_registry_does_not_block_the_queue(client, settings):
    """No models.yaml under the registry root here: the check is skipped, as the
    base_model lookup is when it does not need the file."""
    assert not settings.models_config.exists()
    assert client.post("/jobs", json=BODY).status_code == 202


# ── a cached run refuses a network datasets cache (#60) ─────────────────────
def test_a_cached_run_is_refused_when_the_datasets_cache_is_on_the_share(
        client, settings, monkeypatch, tmp_path):
    """The 11½-hour failure, caught at submit instead."""
    settings.cache_datasets = True
    monkeypatch.setattr(app_module, "check_datasets_cache",
                        lambda *a, **k: (_ for _ in ()).throw(
                            PreflightError("the datasets cache /mnt/... is on a cifs filesystem")))
    resp = client.post("/jobs", json=BODY)
    assert resp.status_code == 500
    assert "cifs" in resp.json()["detail"]
    assert store_of(client).list() == []


def test_a_streaming_run_does_not_care_where_the_cache_lives(client, settings, monkeypatch):
    """streaming=True writes no Arrow cache, so refusing over its location would
    block a job that cannot be hurt by it."""
    settings.cache_datasets = False
    called = []
    monkeypatch.setattr(app_module, "check_datasets_cache",
                        lambda *a, **k: called.append(1))
    assert client.post("/jobs", json=BODY).status_code == 202
    assert called == []


# ── the per-epoch record (#38) ──────────────────────────────────────────────
def test_the_curve_answers_before_the_train_stage_writes_it(client):
    """Was a 404 until #77. A caller polling a running job should not have to
    handle one body for "not yet" and another for "here you go"."""
    job_id = client.post("/jobs", json=BODY).json()["job_id"]
    resp = client.get(f"/jobs/{job_id}/curve")
    assert resp.status_code == 200
    assert resp.json()["points"] == []


def test_the_curve_is_served_once_written(client):
    from atr_training.curves import CURVE_FILENAME, curve_from_checkpoints, write_training_json

    job_id = client.post("/jobs", json=BODY).json()["job_id"]
    ckpt = Path(store_of(client).paths(job_id).root) / "ckpt"
    ckpt.mkdir(parents=True)
    for epoch, metric in ((48, "0.9000"), (50, "0.9200")):
        (ckpt / f"checkpoint_{epoch}-{metric}.ckpt").write_bytes(b"CKPT")
    write_training_json(Path(store_of(client).paths(job_id).root) / CURVE_FILENAME,
                        curve_from_checkpoints(ckpt), job_id)

    body = client.get(f"/jobs/{job_id}/curve").json()
    assert body["best"] == {"epoch": 50, "val_metric": 0.92}
    assert body["still_improving"] is True
    assert body["complete"] is False        # top-10 only; never claims otherwise


def test_every_dataset_is_verified_not_only_the_first(client, app):
    """Checking one of three and answering "valid" is the same class of mistake
    the guard exists to prevent."""
    seen = []

    def check(spec, settings, **kw):
        seen.append(spec.hf_repo)
        return ["project 'typo' not found"] if spec.hf_repo.endswith("second") else []

    app.state.verify_spec = check
    body = {**BODY, "datasets": [{"hf_repo": "dh-unibe/first", "train_projects": ["a"]},
                                 {"hf_repo": "dh-unibe/second", "train_projects": ["b"]}]}
    body.pop("dataset", None)
    resp = client.post("/jobs/verify", json=body)

    assert seen == ["dh-unibe/first", "dh-unibe/second"]
    assert resp.json()["valid"] is False
    assert "dh-unibe/second" in resp.json()["errors"][0]   # which one, not just what


# ── base_model is checked at submit, not in the train stage (#76) ───────────
class TestBaseModelAtSubmit:
    """A run was lost to `kraken-medieval_generic_b is not a valid DOI` raised in
    the TRAIN stage — after prepare and compile. Everything needed to refuse it
    was in the request."""

    @pytest.fixture(autouse=True)
    def registry(self, app):
        from atr_training.shared_registry import BaseEntry, SharedRegistry

        app.state.registry = SharedRegistry([
            BaseEntry(id="kraken-late_medieval_german", engine="kraken",
                      zenodo_id="10.5281/zenodo.15366732"),
        ])
        yield
        if hasattr(app.state, "registry"):
            delattr(app.state, "registry")

    def test_an_unknown_base_model_is_refused_before_a_job_exists(self, client):
        resp = client.post("/jobs", json={**BODY, "base_model": "kraken-nope"})
        assert resp.status_code == 400
        assert "kraken-late_medieval_german" in resp.json()["detail"]
        assert client.get("/jobs").json()["jobs"] == []   # nothing was queued

    def test_a_registry_id_is_accepted(self, client):
        resp = client.post("/jobs", json={
            **BODY, "base_model": "kraken-late_medieval_german",
            "params": {"batch_size": 16, "epochs": 30, "resize": "union"}})
        assert resp.status_code == 202

    def test_a_zenodo_doi_is_still_accepted(self, client):
        resp = client.post("/jobs", json={
            **BODY, "base_model": "10.5281/zenodo.15366732",
            "params": {"batch_size": 16, "epochs": 30}})
        assert resp.status_code == 202

    def test_from_scratch_is_unaffected(self, client):
        """base_model is optional for kraken; absent means from scratch."""
        assert client.post("/jobs", json=BODY).status_code == 202


# ── the curve is for watching a RUNNING job (#77) ───────────────────────────
class TestCurveWhileRunning:
    """`training.json` is written when the train stage ends, so the endpoint had
    nothing to say while a job was training — which is when it is most wanted.
    Lightning writes each epoch's metric into the checkpoint filename as it goes,
    so the data was on disk the whole time."""

    def _training_job(self, client, settings, checkpoints: dict[int, float]):
        job_id = client.post("/jobs", json=BODY).json()["job_id"]
        store = store_of(client)
        store.advance(store.load(job_id), "preparing")
        store.advance(store.load(job_id), "compiling")
        job = store.advance(store.load(job_id), "training")
        ckpt = settings.checkpoint_root / job_id
        ckpt.mkdir(parents=True, exist_ok=True)
        for epoch, metric in checkpoints.items():
            (ckpt / f"checkpoint_{epoch:02d}-{metric:.4f}.ckpt").touch()
        job.checkpoint_dir = str(ckpt)
        store.save(job)
        return job_id

    def test_a_running_job_reports_the_epochs_written_so_far(self, client, settings):
        job_id = self._training_job(client, settings, {5: 0.71, 6: 0.73, 7: 0.75})
        body = client.get(f"/jobs/{job_id}/curve").json()

        assert body["live"] is True
        assert [p["epoch"] for p in body["points"]] == [5, 6, 7]
        assert body["best"]["epoch"] == 7
        assert body["still_improving"] is True
        assert "while the job is training" in body["note"]

    def test_val_error_is_given_alongside_the_accuracy(self, client, settings):
        job_id = self._training_job(client, settings, {5: 0.75})
        point = client.get(f"/jobs/{job_id}/curve").json()["points"][0]
        assert point["val_metric"] == 0.75
        assert point["val_error"] == pytest.approx(0.25)

    def test_a_job_that_has_not_trained_answers_in_the_same_shape(self, client):
        """Not a 404: callers poll this, and an answer that changes shape between
        'not yet' and 'here you go' makes every caller handle two bodies."""
        job_id = client.post("/jobs", json=BODY).json()["job_id"]
        resp = client.get(f"/jobs/{job_id}/curve")

        assert resp.status_code == 200
        body = resp.json()
        assert body["points"] == []          # iterable, not null — the #77 complaint
        assert body["best"] is None
        assert "no checkpoints yet" in body["note"]
        assert body["job_id"] == job_id

    def test_the_written_record_wins_once_the_stage_has_finished(self, client, settings):
        """A finished stage's training.json is the authority; the checkpoint dir
        gets pruned and would give a poorer answer."""
        import json as _json

        job_id = self._training_job(client, settings, {5: 0.71})
        (store_of(client).paths(job_id).root / "training.json").write_text(
            _json.dumps({"job_id": job_id, "points": [{"epoch": 42, "val_metric": 0.9}],
                         "best": {"epoch": 42, "val_metric": 0.9}, "complete": False,
                         "source": "file", "note": "final", "last_epoch": 42,
                         "still_improving": True}),
            encoding="utf-8")
        body = client.get(f"/jobs/{job_id}/curve").json()
        assert [p["epoch"] for p in body["points"]] == [42]
        assert "live" not in body

    def test_an_unknown_job_is_still_a_404(self, client):
        assert client.get("/jobs/20260101T000000Z-nope/curve").status_code == 404


# ── the GPU coordination is gone (#139) ─────────────────────────────────────

def test_the_trainer_has_no_gpu_claim_route(client):
    """404 *with* the key, so this is the route being absent and not the guard.

    The claim coordinated one card between a gateway and a trainer that shared
    it. Since 16.09.2026 the gateway is on idhefix and this service on asteraix;
    an answer from here would describe a machine the asker does not run on.
    """
    assert client.get("/gpu-claim").status_code == 404
    assert "/gpu-claim" not in client.get("/openapi.json").json()["paths"]
    # GET /gpu, this host's own cards, is not part of the coordination and stays.
    assert "/gpu" in client.get("/openapi.json").json()["paths"]
    for name in ("compute_gpu_claim", "refresh_gpu_claim", "GPU_STAGES"):
        assert not hasattr(app_module, name), name

# ── gateway key health (#48) ─────────────────────────────────────────────────

def test_health_reports_gateway_auth_configured_false(client, settings):
    """gateway_auth_configured is False when the key is empty."""
    # settings fixture has no gateway_api_key by default
    body = client.get("/health").json()
    assert body["gateway_auth_configured"] is False


def test_health_reports_gateway_auth_configured_true(client, settings, monkeypatch):
    """gateway_auth_configured is True when a key is set."""
    monkeypatch.setattr(settings, "gateway_api_key", "a" * 32)
    monkeypatch.setattr(settings, "gateway_url", "http://127.0.0.1:8200")
    body = client.get("/health").json()
    assert body["gateway_auth_configured"] is True


def test_health_deep_check_reports_gateway_reachable(client, settings, monkeypatch):
    """/health?deep=1 includes gateway_reachable when key is set and gateway up."""
    import unittest.mock
    monkeypatch.setattr(settings, "gateway_api_key", "a" * 32)
    monkeypatch.setattr(settings, "gateway_url", "http://127.0.0.1:8200")
    # Pretend gateway is reachable
    with unittest.mock.patch("httpx.Client") as mck:
        mck.return_value.__enter__.return_value.get.return_value.status_code = 200
        body = client.get("/health", params={"deep": "1"}).json()
    assert body["gateway_auth_configured"] is True
    assert body["gateway_reachable"] is True
    assert body["gateway_models_status"] == 200


def test_health_deep_omit_when_key_missing(client, settings):
    """/health?deep=1 omits gateway_reachable when key is not configured."""
    # No key set — deep check must not be attempted
    body = client.get("/health", params={"deep": "1"}).json()
    assert body["gateway_auth_configured"] is False
    assert "gateway_reachable" not in body



def test_the_deep_check_needs_the_trainer_key(client):
    """Plain /health stays open; ?deep=1 calls the gateway with the gateway's key,
    so a caller without the trainer's key gets the same 401 as on any other route."""
    keyless = TestClient(app_module.app, client=LOOPBACK)
    assert keyless.get("/health").status_code == 200
    refused = keyless.get("/health", params={"deep": "1"})
    assert refused.status_code == 401
    assert "gateway_reachable" not in refused.text


def test_the_gateway_key_is_in_no_health_answer_and_no_log(client, settings, monkeypatch):
    """Neither the deep check's answer nor its failure path carries the key."""
    import httpx
    import unittest.mock
    from loguru import logger

    secret = "gateway-key-" + "x" * 32
    monkeypatch.setattr(settings, "gateway_api_key", secret)
    monkeypatch.setattr(settings, "gateway_url", "http://127.0.0.1:8200")
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)), level="DEBUG",
                      format="{level} {message} {extra} {exception}")
    try:
        with unittest.mock.patch("httpx.Client") as mck:
            mck.return_value.__enter__.return_value.get.return_value.status_code = 200
            ok = client.get("/health", params={"deep": "1"})
            mck.return_value.__enter__.return_value.get.side_effect = httpx.ConnectError(
                "connection refused")
            down = client.get("/health", params={"deep": "1"})
    finally:
        logger.remove(sink)
    assert ok.json()["gateway_models_status"] == 200
    assert down.json()["gateway_reachable"] is False
    for text in (ok.text, down.text, *lines):
        assert secret not in text


def test_startup_says_when_the_gateway_key_is_missing(settings, monkeypatch):
    from loguru import logger

    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)), level="DEBUG",
                      format="{level} {message}")
    try:
        monkeypatch.setattr(settings, "gateway_api_key", "")
        app_module._warn_without_gateway_key(settings)
        missing = list(lines)
        lines.clear()
        monkeypatch.setattr(settings, "gateway_api_key", "set-" + "y" * 32)
        app_module._warn_without_gateway_key(settings)
    finally:
        logger.remove(sink)
    assert len(missing) == 1 and missing[0].startswith("WARNING")
    assert "ATR_TRAIN_GATEWAY_API_KEY" in missing[0]
    assert lines == []


def test_every_route_is_registered_once(client):
    """Two handlers for one path: the first wins and the second is dead code.

    #64 and #69 each brought a `GET /host`. Both landed, the shutil-based one
    was registered first, and five of the six /host tests went red on main while
    each PR had been green on its own.
    """
    from collections import Counter

    seen = Counter((route.path, method)
                   for route in client.app.routes
                   for method in getattr(route, "methods", ()) or ())
    duplicates = {key: n for key, n in seen.items() if n > 1}
    assert not duplicates, f"registered more than once: {duplicates}"


# ── GET /bases (#39): the hand before the century ───────────────────────────
#
# The route ranks rather than filters on script, which is the part worth testing:
# a caller asking for Kurrent must not be handed an empty list when the registry
# has none, because "[]" and "no registry" would then look the same.

BASES = [
    {"id": "kraken-catmus_caroline", "engine": "kraken",
     "zenodo_id": "10.5281/zenodo.5468665",
     "scripts": ["Caroline minuscule"], "languages": ["la"],
     "centuries": [9, 10, 11, 12]},
    {"id": "kraken-catmus_medieval", "engine": "kraken",
     "zenodo_id": "10.5281/zenodo.7516057",
     "scripts": ["Caroline minuscule"], "languages": ["la"],
     "centuries": [14, 15, 16]},
    {"id": "kraken-kurrent_early_modern", "engine": "kraken",
     "zenodo_id": "10.5281/zenodo.1",
     "scripts": ["Kurrent (chancery)"], "languages": ["de"],
     "centuries": [16, 17]},
    {"id": "kraken-textura_late_medieval", "engine": "kraken",
     "zenodo_id": "10.5281/zenodo.2",
     "scripts": ["Textura"], "languages": ["de"], "centuries": [14, 15, 16]},
    {"id": "vllm-qwen3vl", "engine": "vllm", "local_path": "/models/qwen",
     "scripts": [], "languages": ["de"], "centuries": []},
]


@pytest.fixture
def bases_client(client, app):
    from atr_training.shared_registry import BaseEntry, SharedRegistry
    app.state.registry = SharedRegistry([BaseEntry(**entry) for entry in BASES])
    yield client
    del app.state.registry


def ids_of(resp) -> list[str]:
    return [base["id"] for base in resp.json()["bases"]]


def test_bases_lists_only_kraken_entries(bases_client):
    """A vLLM entry is not a kraken fine-tuning base, whatever its metadata says."""
    assert "vllm-qwen3vl" not in ids_of(bases_client.get("/bases"))


def test_a_script_match_outranks_a_closer_century(bases_client):
    """§9c, as a test: Kurrent (16-17) beats Textura (14-16) for 15th-century
    material, because the hand outranks the date."""
    resp = bases_client.get("/bases", params={"script": "kurrent", "century": 15})
    assert ids_of(resp)[0] == "kraken-kurrent_early_modern"
    assert resp.json()["matched_script"] is True


def test_the_century_orders_within_the_script(bases_client):
    """Both Caroline bases match the script, so the century decides between them."""
    ids = ids_of(bases_client.get("/bases", params={"script": "caroline", "century": 15}))
    assert ids[:2] == ["kraken-catmus_medieval", "kraken-catmus_caroline"]


def test_a_script_nothing_matches_falls_back_to_the_closest_centuries(bases_client):
    """Not empty, and it says so: an empty list is indistinguishable from a
    registry that could not be read."""
    resp = bases_client.get("/bases", params={"script": "Beneventan", "century": 17})
    assert resp.json()["matched_script"] is False
    assert ids_of(resp)[0] == "kraken-kurrent_early_modern"   # 16-17 is nearest


def test_language_removes_a_base_rather_than_ranking_it(bases_client):
    """A base that cannot read the language is not a candidate at all."""
    ids = ids_of(bases_client.get("/bases", params={"language": "DE"}))
    assert ids == ["kraken-kurrent_early_modern", "kraken-textura_late_medieval"]


def test_the_order_is_the_same_on_every_call(bases_client):
    """Ties break by id, so a caller (or a diff) sees one order, not dict order."""
    first = ids_of(bases_client.get("/bases"))
    second = ids_of(bases_client.get("/bases"))
    assert first == second == sorted(first)


def test_bases_carries_the_fields_a_caller_ranks_on(bases_client):
    entry = next(b for b in bases_client.get("/bases").json()["bases"]
                 if b["id"] == "kraken-kurrent_early_modern")
    assert entry["scripts"] == ["Kurrent (chancery)"]
    assert entry["languages"] == ["de"]
    assert entry["centuries"] == [16, 17]
    assert entry["zenodo_id"] == "10.5281/zenodo.1"


def test_bases_without_a_registry_is_503_not_an_empty_list(client, app):
    """The one case where empty would be a lie about the registry."""
    if hasattr(app.state, "registry"):
        del app.state.registry
    resp = client.get("/bases")
    assert resp.status_code == 503
    assert "registry" in resp.json()["detail"]


def test_bases_needs_the_key(settings, app, spawn):
    """The access middleware covers this route like every other one but /health."""
    app.state.settings = settings
    app.state.store = JobStore(settings.jobs_root, host_id=settings.host_id)
    with TestClient(app, client=LOOPBACK) as unauthenticated:
        assert unauthenticated.get("/bases").status_code in (401, 403)

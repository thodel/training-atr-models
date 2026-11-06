"""A preemptable card: the service says when a job of ours waits for VRAM (#396).

Since #12 the service trains on both A40s, and its only gate is free memory on a
card no job of ours holds. Anything else on the box — a weightspace pass, an
ad-hoc script — is invisible to it except as memory that is missing, and
nothing told that other process a job was waiting behind it. The proposal is
cooperative: the service writes down who is waiting and how much it needs, and
the other process steps aside between two units of its work.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from atr_training import gpu
from atr_training.contracts import DatasetSpec, TrainRequest
from atr_training.jobstore import JobStore
from atr_training.preflight import GpuInfo, PreflightError
from atr_training.settings import TrainerSettings

from kraken_train_svc import app as app_module
from tests.test_trainer_gpu import client, settings, venvs  # noqa: F401 — fixtures

REPO = "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi"


def request(model_id: str, engine: str = "kraken") -> TrainRequest:
    return TrainRequest(model_id=model_id, engine=engine,
                        dataset=DatasetSpec(hf_repo=REPO, train_projects=["P"]))


def full(gpu_index, min_free_mb):
    """Both cards held by something that is not a job of ours."""
    raise PreflightError(f"GPU {gpu_index} has 9000 MB free, {min_free_mb} needed")


def free(gpu_index, min_free_mb):
    return GpuInfo(index=gpu_index, free_mb=40000, total_mb=46068)


def spawn(settings, job):  # noqa: F811
    return os.getpid()


@pytest.fixture
def box(tmp_path: Path, venvs) -> tuple[JobStore, TrainerSettings]:  # noqa: F811
    settings = TrainerSettings(jobs_root=tmp_path / "jobs", trained_root=tmp_path / "trained",  # noqa: F811
                               venvs_root=venvs, checkpoint_root=tmp_path / "ckpt",
                               gpus=[0, 1])
    return JobStore(settings.jobs_root, host_id=settings.host_id), settings


# ── the acceptance of #396 ──────────────────────────────────────────────────
@pytest.mark.parametrize("engine, needs", [("kraken", 12000), ("vllm", 24000)])
def test_a_job_refused_on_every_card_is_waiting_with_its_threshold(box, engine, needs):
    store, cfg = box
    job = store.create(request("waits", engine))

    assert app_module.schedule_once(store, cfg, spawn=spawn, vram_check=full) is None

    assert app_module.waiting_for_card() == {
        "job_id": job.id, "engine": engine, "needs_mib": needs}


def test_nothing_queued_is_nobody_waiting(box):
    store, cfg = box
    app_module.schedule_once(store, cfg, spawn=spawn, vram_check=full)

    assert app_module.waiting_for_card() is None


def test_a_free_card_ends_the_wait(box):
    """The job starts; the next reader must not be told to step aside."""
    store, cfg = box
    store.create(request("waits"))
    app_module.schedule_once(store, cfg, spawn=spawn, vram_check=full)
    assert app_module.waiting_for_card() is not None

    started = app_module.schedule_once(store, cfg, spawn=spawn, vram_check=free)

    assert started is not None
    assert app_module.waiting_for_card() is None


def test_a_job_that_waits_on_our_own_run_is_not_waiting_for_a_card(box):
    """Every slot is held by a job of this host: a foreign process letting go of
    memory would not start anything, so it is not asked to."""
    store, cfg = box
    cfg.max_concurrent = 1
    store.create(request("first"))
    assert app_module.schedule_once(store, cfg, spawn=spawn, vram_check=free) is not None
    store.create(request("second"))

    assert app_module.schedule_once(store, cfg, spawn=spawn, vram_check=full) is None
    assert app_module.waiting_for_card() is None


def test_a_cancelled_job_stops_waiting(box):
    store, cfg = box
    job = store.create(request("waits"))
    app_module.schedule_once(store, cfg, spawn=spawn, vram_check=full)
    store.advance(store.load(job.id), "cancelled")

    app_module.schedule_once(store, cfg, spawn=spawn, vram_check=full)

    assert app_module.waiting_for_card() is None


# ── where it is read ────────────────────────────────────────────────────────
def test_gpu_reports_the_waiting_job(client, monkeypatch):  # noqa: F811
    monkeypatch.setattr(gpu, "inspect_all", lambda job_pids: ([], []))
    waiting = {"job_id": "20261106T000000Z-x", "engine": "kraken", "needs_mib": 12000}
    monkeypatch.setattr(app_module, "_waiting", waiting)

    body = client.get("/gpu").json()

    assert body["waiting"] == waiting


def test_gpu_says_null_when_nobody_waits(client, monkeypatch):  # noqa: F811
    monkeypatch.setattr(gpu, "inspect_all", lambda job_pids: ([], []))
    monkeypatch.setattr(app_module, "_waiting", None)

    body = client.get("/gpu").json()

    assert "waiting" in body and body["waiting"] is None


def test_health_says_it_without_a_key(client, monkeypatch):  # noqa: F811
    """The other process should not need a secret to learn it is asked to yield."""
    waiting = {"job_id": "20261106T000000Z-x", "engine": "vllm", "needs_mib": 24000}
    monkeypatch.setattr(app_module, "_waiting", waiting)

    answer = client.get("/health", headers={"X-API-Key": ""})

    assert answer.status_code == 200
    assert answer.json()["waiting"] == waiting

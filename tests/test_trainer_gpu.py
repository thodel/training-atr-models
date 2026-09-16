"""``GET /gpu`` and ``GET /health`` as the gateway reads them across the network (#13).

After the split the gateway on idhefix can no longer read the training cards:
its nvidia-smi describes idhefix, and matching asteraix's job pids against
idhefix's ``/proc`` marks a local stranger as a foreign job's. So the trainer
reads its own cards, and ``/health`` names the engines the gateway checks a
submit against instead of importing ``BACKENDS``.

Offline throughout: nvidia-smi and ``/proc`` are stubbed the way the gateway's
tests stub them (serving tests/test_train_gpu_inspection.py). The response
shapes are pinned against ``tests/fixtures/trainer_contract/``, byte-identical
copies of the gateway's fixtures (serving-atr-inference#137), which feeds the
same files through its proxy.
"""

from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path
from typing import get_args

import pytest
from fastapi.testclient import TestClient

from atr_training import gpu
from atr_training.backends import BACKENDS
from atr_training.contracts import DatasetSpec, TrainEngine, TrainRequest
from atr_training.jobstore import JobStore
from atr_training.preflight import GpuInfo
from atr_training.settings import TrainerSettings
from kraken_train_svc import app as app_module

FIXTURES = Path(__file__).parent / "fixtures" / "trainer_contract"
HEALTH = json.loads((FIXTURES / "health.json").read_text(encoding="utf-8"))
GPU = json.loads((FIXTURES / "gpu.json").read_text(encoding="utf-8"))

LOOPBACK = ("127.0.0.1", 50000)
REPO_ID = "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi"
#: v5, the run training on asteraix while this was written.
V5 = "20260916T061502Z-qwen3vl-german-pages-v5"

CARDS = [["0", "NVIDIA A40", "46068", "24591", "20986", "97", "61"],
         ["1", "NVIDIA A40", "46068", "0", "45589", "0", "0"]]
UUIDS = [["GPU-aaa"], ["GPU-bbb"]]


def assert_same_keys(actual, expected, where: str = "$") -> None:
    """Same keys at every level; list items against the fixture's first item.

    Values are the machine's and differ; the structure is the contract.
    """
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{where}: {type(actual).__name__}, not an object"
        assert set(actual) == set(expected), \
            f"{where}: extra {sorted(set(actual) - set(expected))}, " \
            f"missing {sorted(set(expected) - set(actual))}"
        for key, value in expected.items():
            assert_same_keys(actual[key], value, f"{where}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{where}: {type(actual).__name__}, not a list"
        if expected:
            for i, item in enumerate(actual):
                assert_same_keys(item, expected[0], f"{where}[{i}]")


def _smi_stub(apps):
    def fake(query, *, per_app):
        if per_app:
            return apps
        return UUIDS if query == "uuid" else CARDS
    return fake


@pytest.fixture
def procs(monkeypatch):
    """A settable /proc: {pid: (user, age, command, ppid[, unit])}."""
    table: dict = {}

    def info(pid):
        row = table.get(pid)
        return (None, None, None) if row is None else (row[0], row[1], row[2])

    def ancestors(pid, limit=32):
        chain, seen = [], set()
        cur = pid
        while cur in table and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            cur = table[cur][3]
        return chain

    def unit(pid):
        row = table.get(pid)
        return row[4] if row and len(row) > 4 else None

    monkeypatch.setattr(gpu, "_proc_info", info)
    monkeypatch.setattr(gpu, "_ancestors", ancestors)
    monkeypatch.setattr(gpu, "_unit_of", unit)
    return table


@pytest.fixture
def venvs(tmp_path: Path) -> Path:
    """kraken and vlm built, trocr not — the state of asteraix on 16.09.2026."""
    root = tmp_path / "venvs"
    for name in ("kraken-train", "vlm-train"):
        (root / name / "bin").mkdir(parents=True)
        (root / name / "bin" / "python").touch()
    return root


@pytest.fixture
def settings(tmp_path: Path, venvs: Path) -> TrainerSettings:
    return TrainerSettings(jobs_root=tmp_path / "jobs", trained_root=tmp_path / "trained",
                           venvs_root=venvs, checkpoint_root=tmp_path / "ckpt")


@pytest.fixture
def client(settings, trainer_key, monkeypatch):
    app = app_module.app
    app.state.settings = settings
    app.state.store = JobStore(settings.jobs_root, host_id=settings.host_id)
    monkeypatch.setattr(app_module, "query_gpus",
                        lambda: [GpuInfo(0, 20986, 46068), GpuInfo(1, 45589, 46068)])
    yield TestClient(app, client=LOOPBACK, headers={"X-API-Key": trainer_key})
    for attr in ("settings", "store"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


def _job(store: JobStore, job_id: str, status: str, pid: int | None,
         engine: str = "vllm", host: str | None = None, legacy: bool = False) -> None:
    """``host`` defaults to the store's own; ``legacy`` writes no host at all, as
    the old trainer on idhefix did."""
    request = TrainRequest(model_id=job_id.split("-", 1)[1], engine=engine,
                           dataset=DatasetSpec(hf_repo=REPO_ID, train_projects=["P"]))
    job = store.create(request, job_id=job_id, host=host)
    job.status, job.pid = status, pid
    if legacy:
        job.host = None
    store.save(job)


# ── /health ─────────────────────────────────────────────────────────────────
def test_health_names_every_backend_this_host_can_actually_run(client, settings):
    """Built venvs, not declared backends: trocr exists in code and not on this box."""
    body = client.get("/health").json()
    assert body["host"] == socket.gethostname()
    assert body["available_engines"] == ["kraken", "vllm"]
    assert body["available_engines"] == sorted(
        engine for engine, backend in body["backends"].items() if backend["available"])
    assert_same_keys(body, HEALTH)

    (settings.venvs_root / "vlm-train" / "bin" / "python").unlink()
    assert client.get("/health").json()["available_engines"] == ["kraken"]


def test_health_needs_no_key(client):
    """Liveness must not need a secret; the gateway's engine check reads it too."""
    assert client.get("/health", headers={"X-API-Key": ""}).status_code == 200


def test_the_engine_list_agrees_with_the_request_model(client):
    """Three lists kept in agreement by hand until now, untested: the request
    model's Literal, the backend registry, and what /health tells the gateway."""
    literal = sorted(get_args(TrainEngine))
    assert literal == sorted(BACKENDS)
    assert client.get("/health").json()["engines"] == literal
    assert list(app_module.ENGINES) == literal

    # The gateway refuses an unknown engine with a message it builds from that
    # list; it must read like the trainer's own refusal (serving#137).
    answer = client.post("/jobs/verify", json={
        "engine": "nope", "model_id": "x",
        "dataset": {"hf_repo": REPO_ID, "train_projects": ["P"]}})
    assert answer.status_code == 422
    error = next(e for e in answer.json()["detail"] if e["loc"] == ["body", "engine"])
    assert error["type"] == "literal_error"
    assert error["msg"] == "Input should be 'kraken', 'trocr' or 'vllm'"


def test_health_keeps_every_field_it_had(client):
    """The gateway and install_user_unit.sh read the old fields; #13 only adds."""
    body = client.get("/health").json()
    assert {"status", "gpu", "gpus", "jobs_root", "backends", "jobs"} <= set(body)
    assert body["gpus"] == HEALTH["gpus"]


# ── /gpu ────────────────────────────────────────────────────────────────────
def test_gpu_reports_the_cards_of_this_machine(client, procs, monkeypatch):
    """The reading happens where the job pids mean something."""
    monkeypatch.setattr(gpu, "_smi", _smi_stub([["1843", "24570", "GPU-aaa"]]))
    procs[1843] = ("tobias", 30512.4,
                   "/home/tobias/Repo/training-atr-models/.venvs/vlm-train/bin/python "
                   "-m vlm_train_svc.runner train", 1, "atr-train.service")
    _job(client.app.state.store, V5, "training", 1843)

    answer = client.get("/gpu")
    assert answer.status_code == 200
    body = answer.json()
    assert body["host"] == socket.gethostname()
    assert [c["index"] for c in body["cards"]] == [0, 1]
    run = body["cards"][0]["processes"][0]
    assert (run["registered"], run["job_id"], run["own_service"]) == (True, V5, True)
    assert body["cards"][0]["unaccounted_mib"] == 0
    assert body["cards"][0]["service_mib"] == 24570
    assert body["cards"][1]["processes"] == []
    assert body["job_attribution_available"] is True
    assert body["known_job_pids"] == 1
    assert_same_keys(body, GPU)
    # Not just the same keys: the gateway fixture is what this machine would say.
    assert {k: v for k, v in body.items() if k != "host"} == \
        {k: v for k, v in GPU.items() if k != "host"}


def test_a_pid_from_another_host_is_never_attributed_locally(client, procs, monkeypatch,
                                                             tmp_path):
    """pid 1843 here is a stranger's gunicorn. Several records name that pid, and
    none may claim it: idhefix's store (another host's pids, never read here),
    finished jobs in THIS store — on the share it holds the records idhefix
    wrote before the split, each with an idhefix pid — and live jobs in this
    store that belong to idhefix, stamped or legacy (#15). Only live runs of
    this host count, which is what #414 needs to stay true: the stranger's
    memory stays in ``unaccounted_mib``."""
    monkeypatch.setattr(gpu, "_smi", _smi_stub([["1843", "10440", "GPU-aaa"],
                                                ["4242", "24570", "GPU-bbb"]]))
    procs[1843] = ("change", 2251639.0, "gunicorn: worker", 1, "gunicorn.service")
    procs[4242] = ("tobias", 900.0, "python -m vlm_train_svc.runner", 1, "atr-train.service")

    elsewhere = JobStore(tmp_path / "jobs-idhefix", host_id="idhefix")
    _job(elsewhere, "20260910T080000Z-idhefix-running", "training", 1843)
    here = client.app.state.store
    _job(here, "20260908T101611Z-written-by-idhefix", "failed", 1843, legacy=True)
    _job(here, "20260912T120000Z-cancelled-here", "cancelled", 1843)
    _job(here, "20260915T070000Z-legacy-live-on-idhefix", "training", 1843, legacy=True)
    _job(here, "20260916T090000Z-live-on-idhefix", "training", 1843, host="idhefix")
    _job(here, V5, "training", 4242)

    body = client.get("/gpu").json()
    stranger = body["cards"][0]["processes"][0]
    assert stranger["pid"] == 1843
    assert stranger["registered"] is False and stranger["job_id"] is None
    assert body["cards"][0]["unaccounted_mib"] == 10440
    ours = body["cards"][1]["processes"][0]
    assert ours["registered"] is True and ours["job_id"] == V5
    assert body["known_job_pids"] == 1


def test_a_leftover_of_a_finished_run_is_unaccounted(client, procs, monkeypatch):
    """A failed kraken job's runner is gone from the live pids, but its ketos
    still holds card 1 — in atr-train.service, where every runner stays
    (start_new_session does not leave the cgroup; KillMode=process keeps it
    past a restart). Nothing of ours holds a card on this box except through a
    job, so that memory is what a queued job is waiting for. Under the gateway's
    rule it read 0 unaccounted and showed only in service_mib."""
    monkeypatch.setattr(gpu, "_smi", _smi_stub([["1843", "24570", "GPU-aaa"],
                                                ["7777", "27530", "GPU-bbb"]]))
    procs[1843] = ("tobias", 30512.4, "python -m vlm_train_svc.runner train", 1,
                   "atr-train.service")
    procs[4242] = ("tobias", 108010.0, "python -m kraken_train_svc.runner", 1,
                   "atr-train.service")
    procs[7777] = ("tobias", 108000.0, "ketos train -f page", 4242, "atr-train.service")
    store = client.app.state.store
    _job(store, V5, "training", 1843)
    _job(store, "20260915T090000Z-kraken-failed", "failed", 4242, engine="kraken")

    body = client.get("/gpu").json()
    live, stale = body["cards"]
    assert live["unaccounted_mib"] == 0 and live["service_mib"] == 24570
    leftover = stale["processes"][0]
    assert (leftover["registered"], leftover["own_service"], leftover["orphaned"]) == \
        (False, True, False)
    assert stale["unaccounted_mib"] == 27530
    assert stale["service_mib"] == 27530, "still one of ours, and still said so"
    assert body["known_job_pids"] == 1


def test_gpu_asks_the_probe_with_this_store_s_live_pids_only(client, monkeypatch):
    """The seam itself: what the route hands to the inspection."""
    seen = []
    monkeypatch.setattr(gpu, "inspect", lambda job_pids: seen.append(job_pids) or [])
    store = client.app.state.store
    _job(store, V5, "training", 4242)
    _job(store, "20260916T070000Z-queued-spawned", "queued", 4343)
    _job(store, "20260916T080000Z-queued-waiting", "queued", None)
    _job(store, "20260901T000000Z-old", "failed", 1843)
    assert client.get("/gpu").status_code == 200
    assert seen == [{4242: V5, 4343: "20260916T070000Z-queued-spawned"}]


def test_gpu_without_nvidia_smi_says_so(client, monkeypatch):
    def missing(job_pids):
        raise FileNotFoundError("nvidia-smi is not on PATH")
    monkeypatch.setattr(gpu, "inspect", missing)
    answer = client.get("/gpu")
    assert answer.status_code == 503 and "nvidia-smi" in answer.json()["detail"]


def test_a_failing_nvidia_smi_is_a_502(client, monkeypatch):
    def wedged(job_pids):
        raise subprocess.TimeoutExpired(["nvidia-smi"], gpu.TIMEOUT_S)
    monkeypatch.setattr(gpu, "inspect", wedged)
    answer = client.get("/gpu")
    assert answer.status_code == 502
    assert answer.json()["detail"].startswith("nvidia-smi failed: TimeoutExpired")


def test_an_unreadable_store_still_reports_the_cards(client, procs, monkeypatch):
    """Losing attribution must not lose the memory figures — the share can hang."""
    monkeypatch.setattr(gpu, "_smi", _smi_stub([["1843", "24570", "GPU-aaa"]]))
    procs[1843] = ("tobias", 1.0, "python", 1, "atr-train.service")

    def unreadable():
        raise OSError("Host is down")
    monkeypatch.setattr(client.app.state.store, "list", unreadable)
    body = client.get("/gpu").json()
    assert body["job_attribution_available"] is False
    assert body["known_job_pids"] == 0
    assert body["cards"][0]["memory_used_mib"] == 24591


def test_the_inspection_is_the_gateway_s(procs, monkeypatch):
    """The duplicate behaves as the original: orphans, children, strangers —
    and, with ``services_expected=True``, an engine of ours is not a stray
    (the gateway's own suite pins that with the same atr-trocr row)."""
    monkeypatch.setattr(gpu, "_smi", _smi_stub([["2743851", "27530", "GPU-aaa"],
                                                ["5001", "3000", "GPU-bbb"],
                                                ["7777", "6000", "GPU-bbb"],
                                                ["6100", "1600", "GPU-bbb"]]))
    procs[4242] = ("tobias", 900.0, "ketos train", 1)
    procs[5001] = ("tobias", 890.0, "python -c from multiprocessing", 4242)
    procs[7777] = ("tobias", 108000.0, "ketos train -f page", 1)
    procs[6100] = ("tobias", 5000.0, "trocr engine", 1, "atr-trocr.service")
    cards = gpu.inspect({4242: "job-a"})
    idle, busy = gpu.card_rows(cards, services_expected=True)

    orphan = idle["processes"][0]
    assert orphan["orphaned"] is True and orphan["registered"] is False
    assert idle["orphaned_mib"] == idle["unaccounted_mib"] == 27530

    child, stray, engine = busy["processes"]
    assert child["registered"] is True and child["job_id"] == "job-a"
    assert stray["registered"] is False and stray["age_s"] == 108000.0
    assert (engine["registered"], engine["own_service"]) == (False, True)
    assert busy["unaccounted_mib"] == 6000
    assert busy["service_mib"] == 1600

    # The trainer's rule differs in that one row and nowhere else.
    _, trainer_busy = gpu.card_rows(cards, services_expected=False)
    assert trainer_busy["unaccounted_mib"] == 6000 + 1600
    assert trainer_busy["service_mib"] == 1600


def test_the_contract_fixtures_carry_no_secret(trainer_key):
    """Both repositories publish these files."""
    for path in FIXTURES.iterdir():
        text = path.read_text(encoding="utf-8").lower()
        assert "key" not in text and trainer_key.lower() not in text, path.name

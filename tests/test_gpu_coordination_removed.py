"""The trainer no longer coordinates a GPU with the gateway (serving#139).

serving#129 taught a gateway and a trainer to share one card: the trainer
published a claim at ``GET /gpu-claim``, and at the start of ``train`` asked the
gateway to unload its models through ``POST /admin/release-gpu``. Both only
make sense on one machine. Since 16.09.2026 the gateway runs on idhefix and this
trainer on asteraix, the serving side has removed its half (the gateway neither
reads a claim nor answers a release), and this side's half is gone with it.

What stays is this host's own guard: the VRAM preflight before a spawn
(``check_vram``, ``min_free_vram_*``) and ``GET /gpu``. The gateway settings
stay too — the promotion gate and eval/ still call idhefix.

The route itself is checked in tests/test_train_svc_api.py, where the client is.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import httpx
import pytest

from atr_training.jobstore import JobStore
from atr_training.settings import TrainerSettings
from kraken_train_svc.runner import Pipeline
from test_train_svc_pipeline import FakeRunner, FakeSource, request_with

REPO = Path(__file__).resolve().parents[1]

#: Everything that is not a test and could carry the names: the code, the
#: launchers and units, and the documents an operator reads (docs/ since #16).
ROOTS = ("src", "engines", "scripts", "deploy", "config", "ubelix", ".github", "docs")
FILES = (".env.example", "README.md", "pyproject.toml")
TEXT_SUFFIXES = {".py", ".md", ".sh", ".service", ".yaml", ".yml", ".toml",
                 ".txt", ".json", ".cfg", ".ini", ".example"}
NAMES = ("release_gpu", "release-gpu", "gpu_release", "gateway_is_local",
         "gpu_claim", "gpu-claim", "GPU_STAGES")


class WatchesTheCard(FakeRunner):
    """A kraken runner that notes every HTTP call made before ``train`` began."""

    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self._calls = calls
        self.before_train: list[str] | None = None

    def run(self, cmd, log_path: Path, env=None):
        if "train" in cmd and self.before_train is None:
            self.before_train = list(self._calls)
        return super().run(cmd, log_path, env)


@pytest.fixture
def http_calls(monkeypatch) -> list[str]:
    """Every URL httpx is asked for, sync or async; nothing leaves the process.

    Patched at ``send``, below ``httpx.post`` and ``Client.request``, so a
    release coming back under any spelling is seen. The refusal is a
    ConnectError because both callers the runner has ever had — the promotion
    gate and the old release — treat that as a result, not a crash.
    """
    calls: list[str] = []

    def send(self, request, *args, **kwargs):
        calls.append(str(request.url))
        raise httpx.ConnectError("refused by the test", request=request)

    async def asend(self, request, *args, **kwargs):
        return send(self, request)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "send", asend)
    return calls


def test_the_runner_never_asks_a_gateway_to_release_a_card(tmp_path, http_calls):
    """A gateway on loopback with a key — the one case the old code did ask.

    Asking would now evict idhefix's recognition models to free memory on a
    card this run never touches, and the gateway no longer has the route.
    """
    (tmp_path / "registry").mkdir()
    settings = TrainerSettings(
        jobs_root=tmp_path / "training",
        trained_root=tmp_path / "trained",
        registry_root=tmp_path / "registry",
        checkpoint_root=tmp_path / "checkpoints",
        ketos=tmp_path / "ketos",
        min_free_disk_gb=0.0,
        artefact_cache=False,
        gateway_url="http://127.0.0.1:8200",
        gateway_api_key="a-gateway-key-for-the-promotion-gate",
    )
    store = JobStore(settings.jobs_root, host_id=settings.host_id)

    # Not vacuous: a call to that gateway is seen by the probe.
    with pytest.raises(httpx.ConnectError):
        httpx.post(f"{settings.gateway_url}/probe")
    assert http_calls == [f"{settings.gateway_url}/probe"]
    http_calls.clear()

    runner = WatchesTheCard(http_calls)
    job = store.create(request_with())
    job = Pipeline(store, settings, runner=runner,
                   source=FakeSource({"train": 4, "eval": 2})).execute(job.id)

    assert "train" in [s.name for s in job.stages], job.error
    assert runner.before_train == [], "the gateway was called before the run took the card"
    # Across the whole run, the gateway is asked to read a page and nothing else.
    assert all(url.endswith("/ocr") for url in http_calls), http_calls


def test_nothing_mentions_release_gpu_or_gpu_claim():
    """No module, route, setting, unit or operator note names the coordination.

    Tests are not scanned: the ones that prove the absence have to name it.
    """
    assert importlib.util.find_spec("atr_training.gpu_release") is None
    files = [REPO / name for name in FILES]
    for root in ROOTS:
        base = REPO / root
        if base.is_dir():
            files += [p for p in base.rglob("*")
                      if p.is_file() and p.suffix in TEXT_SUFFIXES
                      and "__pycache__" not in p.parts]
    assert any(p.name == "app.py" for p in files), "the scan found no source; it is vacuous"

    offenders = [
        f"{path.relative_to(REPO)}:{number}  {line.strip()}"
        for path in files if path.is_file()
        for number, line in enumerate(path.read_text(encoding="utf-8",
                                                     errors="replace").splitlines(), 1)
        if any(name in line for name in NAMES)
    ]
    assert not offenders, "the GPU coordination is mentioned:\n  " + "\n  ".join(offenders)

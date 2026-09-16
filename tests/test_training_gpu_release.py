"""Asking the gateway for the card back (#129)."""

from __future__ import annotations

import pytest


from atr_training.gpu_release import ReleaseResult, release_gpu


class FakeResponse:
    def __init__(self, body): self._body = body
    def raise_for_status(self): pass
    def json(self): return self._body


def test_it_reports_what_the_gateway_released(monkeypatch):
    seen = {}

    def post(url, headers=None, timeout=None):
        seen.update(url=url, headers=headers)
        return FakeResponse({"dropped": ["qwen3vl-8b-hebrew"], "kept": ["lightonocr"]})

    import httpx
    monkeypatch.setattr(httpx, "post", post)
    result = release_gpu("http://127.0.0.1:8200/", "k")
    assert result.reached and result.dropped == ["qwen3vl-8b-hebrew"]
    assert result.kept == ["lightonocr"]
    assert seen["url"] == "http://127.0.0.1:8200/admin/release-gpu"
    assert seen["headers"] == {"X-API-Key": "k"}


def test_an_unreachable_gateway_is_not_fatal(monkeypatch):
    """The VRAM preflight is what actually refuses to start a job.

    On 15.09 it did exactly that: "GPU 1 has 12660 MB free, need 24000 MB".
    Abandoning a run because the gateway did not answer would be strictly worse.
    """
    import httpx

    def post(*a, **kw):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", post)
    result = release_gpu("http://127.0.0.1:8200", "k")
    assert not result.reached
    assert "ConnectError" in result.detail


def test_the_summary_names_what_stayed(monkeypatch):
    """A pinned model the run then fails to fit around has to be nameable."""
    r = ReleaseResult(reached=True, dropped=[], kept=["lightonocr-catmus-caroline"])
    assert "nothing resident to release" in str(r)
    assert "kept pinned: lightonocr-catmus-caroline" in str(r)


def test_an_unreached_gateway_says_so_rather_than_looking_empty():
    r = ReleaseResult(reached=False, detail="ConnectError: refused")
    assert str(r).startswith("gateway not reached")


# ── the two faults found by the first job on asteraix (16.09.2026) ──────────

def test_a_venv_without_httpx_does_not_fail_the_job(monkeypatch):
    """trocr-train does not install httpx, and the import sat outside the try."""
    import builtins

    real_import = builtins.__import__

    def no_httpx(name, *args, **kwargs):
        if name == "httpx":
            raise ModuleNotFoundError("No module named 'httpx'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_httpx)
    result = release_gpu("http://127.0.0.1:8200", "k")
    assert not result.reached
    assert "ModuleNotFoundError" in result.detail


@pytest.mark.parametrize("url, local", [
    ("http://127.0.0.1:8200", True),
    ("http://localhost:8200/", True),
    ("http://[::1]:8200", True),
    ("http://130.92.59.240:8200", False),     # idhefix, seen from asteraix
    ("http://idhefix:8200", False),
])
def test_only_a_gateway_on_this_host_counts_as_local(url, local):
    from atr_training.gpu_release import gateway_is_local
    assert gateway_is_local(url) is local


def test_a_remote_gateway_is_never_asked_to_release(monkeypatch, tmp_path):
    """Asking would free nothing here and evict idhefix's recognition models."""
    import httpx

    from atr_training import runner_base

    calls = []
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: calls.append(a) or None)

    from types import SimpleNamespace

    pipeline = SimpleNamespace(settings=SimpleNamespace(
        gateway_url="http://130.92.59.240:8200", gateway_api_key="k"))
    runner_base.BasePipeline._release_gpu(pipeline)
    assert calls == []

    # …and the same call with a local gateway does reach httpx, so the empty list
    # above is the rule working, not the probe being unable to see a call.
    pipeline.settings.gateway_url = "http://127.0.0.1:8200"
    runner_base.BasePipeline._release_gpu(pipeline)
    assert len(calls) == 1

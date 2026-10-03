"""`eval/` came over with #11 — and brought two things worth pinning (#1, #11).

The harness measures the deployed system through `/recognize`, so it is the one
piece of this repo that talks to idhefix on purpose. Both assertions here are
about that: that it needs nothing from the serving *code*, and that it cannot
quietly talk to the wrong machine.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path

import pytest


def test_eval_runs_without_the_serving_package(monkeypatch):
    """`eval/` had zero serving imports even before the move — which is why E4
    (`eval/` moves with the trainer) was decided on the code rather than against
    it. Pinned on the import machinery, not only on the source: an
    `atr_serving` reached through `importlib` would pass a grep.
    """
    real_import = builtins.__import__

    def refuse_serving(name, *args, **kwargs):
        if name == "atr_serving" or name.startswith("atr_serving."):
            raise AssertionError(f"eval/ imported {name}")
        return real_import(name, *args, **kwargs)

    for module in [m for m in sys.modules if m == "eval" or m.startswith("eval.")]:
        del sys.modules[module]
    monkeypatch.setattr(builtins, "__import__", refuse_serving)

    metrics = importlib.import_module("eval.metrics")
    run_eval = importlib.import_module("eval.run_eval")

    # And it really did get the metrics it claims to share with the trainer.
    assert metrics.cer("hallo", "hello") == 1 / 5
    assert run_eval.build_record("m", Path("a.png"), {"text": "hallo"}, 1, "hello")["cer"] == 1 / 5


def test_the_gateway_url_is_configurable_and_not_loopback_by_default(monkeypatch, tmp_path, capsys):
    """Until the move, `http://127.0.0.1:8200` was the gateway. Here it is
    asteraix's own port, where nothing listens — and the harness would have
    written one connection error per image into the `error` column and printed a
    table of failures that says nothing about any model. Refusing is the only
    answer that cannot be misread.
    """
    from eval import run_eval

    monkeypatch.delenv(run_eval.GATEWAY_ENV, raising=False)
    monkeypatch.setattr(sys, "argv",
                        ["run_eval.py", "--images-dir", str(tmp_path), "--models", "m"])
    assert run_eval.main() == 2
    message = capsys.readouterr().err
    assert run_eval.GATEWAY_ENV in message and "130.92.59.240" in message
    assert not run_eval.build_parser().parse_args(
        ["--images-dir", str(tmp_path), "--models", "m"]).gateway, "no default at all"

    # Configurable through the same variable the promotion gate reads, so the two
    # edges into idhefix are set in one place.
    monkeypatch.setenv(run_eval.GATEWAY_ENV, "http://130.92.59.240:8200")
    argv = ["--images-dir", str(tmp_path), "--models", "m"]
    assert run_eval.build_parser().parse_args(argv).gateway == "http://130.92.59.240:8200"
    overridden = run_eval.build_parser().parse_args([*argv, "--gateway", "http://127.0.0.1:8299"])
    assert overridden.gateway == "http://127.0.0.1:8299", "--gateway must still win"


@pytest.mark.parametrize("env,expected", [
    ({"ATR_TRAIN_GATEWAY_API_KEY": "trainer-side"}, "trainer-side"),
    ({"ATR_API_KEY": "gateway-side"}, "gateway-side"),
    ({"ATR_TRAIN_GATEWAY_API_KEY": "trainer-side", "ATR_API_KEY": "gateway-side"},
     "trainer-side"),
    ({}, ""),
])
def test_the_key_is_the_gateways_under_either_name(monkeypatch, env, expected):
    """`ATR_TRAIN_GATEWAY_API_KEY` is what `.env` sets here; `ATR_API_KEY` is the
    same secret's name on idhefix, kept so a run started by hand over there still
    works. #9 split the two directions, and this is the one into idhefix.
    """
    from eval import run_eval

    for name in run_eval.KEY_ENVS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert run_eval._key_from_env() == expected

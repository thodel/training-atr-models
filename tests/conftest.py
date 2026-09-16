"""Test-wide safety rails.

Both exist because a setting that points at a real place makes the suite write
there.

The artefact cache (#109) defaults to a directory under ``$HOME``, and a
pipeline test that runs the whole lifecycle will happily write there. A test
suite that leaves 40 GB — or, as it did the first time, five stray directories —
in the developer's home is a bug in the suite, not a quirk of it.

The shared registry (#14) defaults to the LIVE directory on the research share,
and ``TrainerSettings`` also reads ``.env`` from the working directory — the
checkout on asteraix, whose ``.env`` names that directory on purpose. Every
pipeline test that reaches the register stage would then put a fake model into
the registry the gateway on idhefix serves from. The gateway's suite had the same
hole and closed it the same way (serving-atr-inference#138).
"""

from __future__ import annotations

import os

import pytest

# An environment variable beats .env in pydantic-settings. Set at import, before
# any test module is collected, to a directory that does not exist: registering
# into it fails (trained/ is created, its parents never are), so a settings object
# built at import time cannot reach the share either. The fixture below replaces
# it with a real, empty directory per test.
os.environ["ATR_TRAIN_REGISTRY_ROOT"] = "/nonexistent/atr-training-test-registry"
os.environ["ATR_TRAIN_MODELS_CONFIG"] = ""


@pytest.fixture(autouse=True)
def _artefact_cache_never_touches_home(tmp_path_factory, monkeypatch):
    """Point every ``TrainerSettings`` in the suite at a throwaway cache root.

    Set through the environment rather than the fixtures, so it holds for the
    settings objects constructed inside the code under test as well as the ones
    the tests build themselves.
    """
    root = tmp_path_factory.mktemp("artefact-cache")
    monkeypatch.setenv("ATR_TRAIN_ARTEFACT_CACHE_ROOT", str(root))


@pytest.fixture(autouse=True)
def _registry_never_touches_the_share(tmp_path_factory, monkeypatch):
    """A fresh, empty registry root per test, again through the environment.

    Per test rather than once: a test that deletes the variable gets it back
    instead of handing the live default to every test after it. ``models_config``
    is emptied too, so a ``.env`` that still names the published file cannot pin
    it to the share while the root moves.
    """
    root = tmp_path_factory.mktemp("registry")
    monkeypatch.setenv("ATR_TRAIN_REGISTRY_ROOT", str(root))
    monkeypatch.setenv("ATR_TRAIN_MODELS_CONFIG", "")

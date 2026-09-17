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

The access settings (#13) are pinned the same way, for the same ``.env``: on
asteraix it holds the real ``ATR_TRAIN_API_KEY``, which a test would otherwise
read, send, and — on a failure — print. Every settings object in the suite gets
:data:`TEST_API_KEY` instead, so the app's routes are exercised behind the real
middleware; a test about an unconfigured trainer empties it explicitly.

The host identity (#15) too: that ``.env`` says ``ATR_TRAIN_HOST_ID=asteraix``,
and a suite whose idea of "this host" changes with the machine it runs on would
pass or fail by where it ran. :data:`TEST_HOST_ID` is a name no real host has.
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
# Inside a Slurm job the register stage leaves the registry alone (#17). The suite
# run on a UBELIX node must still exercise the ordinary path; the tests for the
# Slurm path set the variable themselves.
os.environ.pop("SLURM_JOB_ID", None)

#: Long enough for the launcher's 32-character floor, and recognisable in output.
TEST_API_KEY = "test-trainer-key-0123456789abcdef-not-a-secret"
os.environ["ATR_TRAIN_API_KEY"] = TEST_API_KEY
os.environ["ATR_TRAIN_REQUIRE_AUTH"] = "true"
os.environ["ATR_TRAIN_ALLOWED_CLIENTS"] = ""

#: "This host" for every settings object in the suite.
TEST_HOST_ID = "test-trainer"
os.environ["ATR_TRAIN_HOST_ID"] = TEST_HOST_ID
os.environ["ATR_TRAIN_LEGACY_JOB_HOST"] = "idhefix"


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


@pytest.fixture(autouse=True)
def _access_settings_never_come_from_env_files(monkeypatch):
    """Per test, for the reason the registry root is: a test that changes one of
    these gets the suite's values back instead of leaking its own."""
    monkeypatch.setenv("ATR_TRAIN_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("ATR_TRAIN_REQUIRE_AUTH", "true")
    monkeypatch.setenv("ATR_TRAIN_ALLOWED_CLIENTS", "")


@pytest.fixture(autouse=True)
def _host_identity_never_comes_from_env_files(monkeypatch):
    """Per test, like the access settings."""
    monkeypatch.setenv("ATR_TRAIN_HOST_ID", TEST_HOST_ID)
    monkeypatch.setenv("ATR_TRAIN_LEGACY_JOB_HOST", "idhefix")


@pytest.fixture
def trainer_key() -> str:
    """The key every ``TrainerSettings`` in the suite carries."""
    return TEST_API_KEY

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
import sys

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

# The image ceiling (#145) cuts the train arrows after compile, and the fake
# compiles of the pipeline tests write b"ARROW", not a corpus. Off for the suite;
# the tests of the cut turn it on themselves.
os.environ["ATR_TRAIN_IMAGE_CEILING"] = "0"

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


@pytest.fixture(autouse=True)
def _the_write_roots_never_point_at_a_real_place(tmp_path_factory, monkeypatch):
    """The other three roots a settings object can be asked to write to.

    The artefact cache above was pinned because a pipeline test would leave
    tens of GB under ``$HOME``. The job store, the trained weights and the
    checkpoints are the same kind of path and were left open, so a settings
    object nobody pinned writes to ``~/atr-cache/{training,trained,checkpoints}``
    here — and on asteraix, whose ``.env`` names the research share, into the
    live job store.

    Nothing in the suite reaches that today: every path installs its own
    settings, and the fallback was measured as never taken (see
    :func:`_the_settings_cache_is_never_a_previous_test_s` below). This is the
    floor under the day one test forgets, which is the whole reason the four
    fixtures above exist.

    Explicit constructor arguments still win, so the fixtures that build their
    own ``TrainerSettings`` are unaffected.
    """
    root = tmp_path_factory.mktemp("write-roots")
    monkeypatch.setenv("ATR_TRAIN_JOBS_ROOT", str(root / "training"))
    monkeypatch.setenv("ATR_TRAIN_TRAINED_ROOT", str(root / "trained"))
    monkeypatch.setenv("ATR_TRAIN_CHECKPOINT_ROOT", str(root / "checkpoints"))


@pytest.fixture(autouse=True)
def _the_settings_cache_is_never_a_previous_test_s():
    """``get_settings()`` keeps its answer in a module global, forever.

    Whichever test filled it would pin *its* environment for the whole session:
    the per-test registry root and write roots above are set with monkeypatch,
    so the cached object would carry one test's tmp directories into every
    later caller — and the fixtures that pin them would silently stop working.

    Measured as never filled across the full suite, in three file orders, so
    this is a net and not a fix. It is here because #252 removed the leak that
    *hid* it: with ``app.state.settings`` now cleared after every test, a test
    that installs none falls through to ``get_settings()`` rather than to the
    previous test's object. Both are wrong; the new one reads the real ``.env``.
    """
    from atr_training import settings as settings_module

    settings_module._settings = None
    yield
    settings_module._settings = None


@pytest.fixture
def trainer_key() -> str:
    """The key every ``TrainerSettings`` in the suite carries."""
    return TEST_API_KEY


#: The seams the service reads off ``app.state``. ``kraken_train_svc.app`` is a
#: module, so what a test installs there outlives the test — and the next one
#: gets it. A single test leaving ``store`` behind made
#: ``test_host_owned_jobs.py`` fail whenever it ran after
#: ``test_train_svc_api.py``, while the full suite stayed green because that
#: order does not happen to arise: the red appeared for whoever ran a subset,
#: and looked like their fault (#252).
#:
#: ``scheduler`` is deliberately not here. It is not a test seam — the app's own
#: lifespan sets it and cancels it — and a fixture that deleted it would be
#: clearing up after the code under test.
APP_STATE_SEAMS = ("settings", "store", "spawn", "vram_check", "verify_spec", "registry")

#: Both import paths are in use across the suite, and they are *different*
#: modules with different ``app`` objects. The net clears whichever ones exist.
APP_MODULES = ("kraken_train_svc.app", "engines.kraken_train_svc.app")


def clear_app_state() -> list[str]:
    """Remove every installed seam from every loaded service module.

    Returns what it removed, so a test can check the net rather than trust it.
    """
    removed = []
    for name in APP_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue                       # nothing imported it; nothing to clear
        for attr in APP_STATE_SEAMS:
            if hasattr(module.app.state, attr):
                delattr(module.app.state, attr)
                removed.append(f"{name}.{attr}")
    return removed


@pytest.fixture(autouse=True)
def _app_state_never_leaks_into_the_next_test():
    """Per test, after it, whether or not it cleaned up itself.

    The fixtures that install seams do delete them again; this is the net under
    the tests that set ``app.state`` directly, which is easy to do and gives no
    sign of having done it. Autouse and last to tear down, so a fixture's own
    cleanup still runs first and sees what it expects.
    """
    yield
    clear_app_state()

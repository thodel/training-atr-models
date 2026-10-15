"""#252: what a test installs on `app.state` must not become the next test's.

`kraken_train_svc.app` is a module, so `app` is one object for the whole
session. Every seam the suite installs — settings, store, spawn, vram_check —
is written onto that object, and a test that forgets to remove one has changed
the service for everything collected after it.

It happened:

    $ pytest tests/test_train_svc_api.py::test_bases_needs_the_key \
             tests/test_host_owned_jobs.py
    FAILED test_a_legacy_record_without_host_is_treated_as_the_legacy_host

    $ pytest tests/test_host_owned_jobs.py
    26 passed

The full run was green throughout, because that order does not arise there.
So the red appeared only for whoever ran a subset — and looked like their fault.
That is the expensive part: not the failure, but which failure a person is shown.

Two defects, one each below, and they are not the same one:

* the **leak** — a test that set `app.state` directly and did not unset it;
* the **cache** — `_store()` answered "already have one" without asking which
  settings that one was built from, so the leak came back as a *plausible*
  store rather than as an error.

Either alone is survivable. Together they turn a forgotten `delattr` into a
wrong `host_id`, three test modules later, with nothing in the output pointing
at the cause.

Two things here are deliberately **not** pinned, because the fixes overlap and
each one alone already fixes the observed failure:

* whether the net clears up *before* or *after* each test. Both leave every
  test with clean state; the net clears after, so the invariant it enforces is
  "no test leaves state behind" rather than "no test finds any".
* that `test_bases_needs_the_key` removes its own seams. The net catches it
  either way — the `finally` there is for the reader, not for the suite.

Both are redundancy on purpose. A test asserting one of them would be pinning
an arbitrary half of a belt-and-braces pair, and would read like a requirement.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from atr_training.jobstore import JobStore
from atr_training.settings import TrainerSettings
from kraken_train_svc import app as app_module
from tests.conftest import APP_MODULES, APP_STATE_SEAMS, clear_app_state

APP_SOURCE = Path(app_module.__file__)


@pytest.fixture
def venvs(tmp_path: Path) -> Path:
    root = tmp_path / "venvs"
    for name in ("kraken-train", "vlm-train"):
        (root / name / "bin").mkdir(parents=True)
        (root / name / "bin" / "python").touch()
    return root


def _settings(tmp_path: Path, venvs: Path, host_id: str, **kw) -> TrainerSettings:
    return TrainerSettings(jobs_root=tmp_path / host_id, trained_root=tmp_path / "trained",
                           venvs_root=venvs, checkpoint_root=tmp_path / "ckpt",
                           min_free_disk_gb=0.0, host_id=host_id, **kw)


# ── the cache knows what it was built from ──────────────────────────────────

def test_replacing_the_settings_replaces_the_store(tmp_path, venvs):
    """The defect itself. Nothing here is about tests: a cache that cannot say
    which inputs produced its answer gives a stale answer that reads as fresh."""
    app_module.app.state.settings = _settings(tmp_path, venvs, "asteraix")
    first = app_module._store()
    assert first.host_id == "asteraix"

    app_module.app.state.settings = _settings(tmp_path, venvs, "idhefix")
    second = app_module._store()

    assert second is not first
    assert second.host_id == "idhefix", (
        "the store was built from the settings that have been replaced — same "
        "jobs_root and host_id as before, and no error anywhere")


def test_the_legacy_host_travels_with_the_settings_too(tmp_path, venvs):
    """The field the original failure turned on: `host_of` reads `legacy_host`,
    which is a constructor argument of the store and not re-read per call."""
    app_module.app.state.settings = _settings(tmp_path, venvs, "asteraix")
    assert app_module._store().legacy_host == "idhefix"

    app_module.app.state.settings = _settings(tmp_path, venvs, "asteraix",
                                              legacy_job_host="asteraix")
    assert app_module._store().legacy_host == "asteraix"


def test_unchanged_settings_are_still_answered_from_the_cache(tmp_path, venvs):
    """The counter-test. Opening a store touches the filesystem, and every
    request asks for one — rebuilding per call would be the other mistake."""
    app_module.app.state.settings = _settings(tmp_path, venvs, "asteraix")

    assert app_module._store() is app_module._store() is app_module._store()


def test_an_equal_but_distinct_settings_object_still_rebuilds(tmp_path, venvs):
    """Identity, not equality, and the choice is deliberate.

    Two settings objects with the same fields would give a store that behaves
    identically, so comparing field by field would also be correct — just a
    deep compare of a large model on a path every request takes. Identity is
    the cheap side of a choice that cannot be wrong: a different object rebuilds,
    which costs one `JobStore`, and the suite is the only caller that ever swaps
    one in.
    """
    first = _settings(tmp_path, venvs, "asteraix")
    app_module.app.state.settings = first
    built = app_module._store()

    same_values = _settings(tmp_path, venvs, "asteraix")
    assert same_values == first, "the premise: equal in value, different objects"
    app_module.app.state.settings = same_values

    rebuilt = app_module._store()
    assert rebuilt is not built
    assert rebuilt.host_id == built.host_id == "asteraix", "and it answers the same"


def test_a_store_installed_from_outside_is_never_replaced(tmp_path, venvs):
    """A store on `app.state` is a seam, not a cache: five fixtures install one
    deliberately, with a host_id that need not match the settings beside it.
    Rebuilding over it would silently take the seam away."""
    app_module.app.state.settings = _settings(tmp_path, venvs, "asteraix")
    installed = JobStore(tmp_path / "elsewhere", host_id="ubelix")
    app_module.app.state.store = installed

    assert app_module._store() is installed
    app_module.app.state.settings = _settings(tmp_path, venvs, "idhefix")
    assert app_module._store() is installed, "the seam was taken away by a rebuild"


def test_the_mark_is_on_the_store_and_not_on_app_state(tmp_path, venvs):
    """Why it is on the object: a flag beside it on `app.state` would survive the
    store being swapped, and then say the wrong thing about the new one."""
    app_module.app.state.settings = _settings(tmp_path, venvs, "asteraix")
    built = app_module._store()

    assert getattr(built, app_module._BUILT_FROM, None) is app_module.app.state.settings
    assert not hasattr(app_module.app.state, app_module._BUILT_FROM)


# ── the net under every test ────────────────────────────────────────────────

def test_the_net_removes_every_seam_it_finds(tmp_path, venvs):
    app_module.app.state.settings = _settings(tmp_path, venvs, "asteraix")
    app_module.app.state.store = app_module._store()
    app_module.app.state.spawn = lambda settings, job: 1
    app_module.app.state.vram_check = lambda gpu, min_free_mb: None

    removed = clear_app_state()

    assert set(removed) >= {"kraken_train_svc.app.settings", "kraken_train_svc.app.store",
                            "kraken_train_svc.app.spawn", "kraken_train_svc.app.vram_check"}
    for attr in APP_STATE_SEAMS:
        assert not hasattr(app_module.app.state, attr)


def test_the_net_is_harmless_when_there_is_nothing_to_clear():
    assert clear_app_state() == []
    assert clear_app_state() == []


def test_the_net_is_autouse_so_a_test_cannot_leave_state_behind():
    """The load-bearing part of the net, and the part that is invisible.

    A net a test has to ask for protects only the tests that already knew. This
    reads the fixture's own marker rather than trusting the decorator to have
    said what it meant.
    """
    from tests import conftest

    net = conftest._app_state_never_leaks_into_the_next_test
    # pytest keeps this on a private attribute whose name it has already changed
    # once (`_pytestfixturefunction` through 8.x, `_fixture_function_marker` in
    # 9.x). Both are tried, and a third name fails with that sentence rather
    # than an AttributeError that reads like a broken test.
    marker = next((m for m in (getattr(net, "_fixture_function_marker", None),
                               getattr(net, "_pytestfixturefunction", None))
                   if m is not None), None)
    assert marker is not None, (
        "pytest has renamed the fixture marker again — update this probe; it is "
        "the only check that the net applies to every test")
    assert marker.autouse is True, "the net no longer applies to every test"
    assert marker.scope == "function", (
        "a net that is not per test leaves the tests inside its scope exposed")


def _state_attributes(path: Path) -> set[str]:
    """Every `app.state.<name>` the service reads or writes, from its AST.

    Parsed rather than grepped: `app.state.store` appears in this file's own
    prose and in that module's comments, and a text search cannot tell an
    explanation from a use. The same shape has cost three wrong answers in this
    project (a `ProtectSystem=strict` that was prose, a `top.score` that matched
    a substring, a comment quoting the engine name its own test forbade).
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        # getattr(app.state, "spawn", None)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
                and _is_app_state(node.args[0])):
            found.add(node.args[1].value)
        # app.state.store = ... / _app.state.scheduler = ...
        elif isinstance(node, ast.Attribute) and _is_app_state(node.value):
            found.add(node.attr)
    return found


def _is_app_state(node: ast.AST) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "state"
            and isinstance(node.value, ast.Name) and node.value.id in ("app", "_app"))


def _modules_that_touch_app_state() -> dict[Path, set[str]]:
    """Every shipped module that reads or writes `app.state`, and which names.

    Over `src/` and `engines/` rather than over the one app module: the check
    below says "every seam the service reads", and a version that looked only
    at `app.py` would have claimed that while measuring less — this project's
    most expensive recurring shape (#641, #646, and the first draft of the
    import test below). Measured today: only `app.py`. `access.py` names
    `app.state` in a docstring, which the AST does not see, and that is the
    point of using one.
    """
    found = {}
    for path in [*Path("src").rglob("*.py"), *Path("engines").rglob("*.py")]:
        names = _state_attributes(path)
        if names:
            found[path.resolve()] = names
    return found


def test_the_net_knows_every_seam_the_service_reads():
    """The check that keeps the net honest.

    A list of names copied once is the failure this repository has twice paid
    for: a rule pinned only in a hardcoded list does not travel to new code
    (#641 SEC-18, #646 SEC-19). So the names come out of the shipped modules'
    own AST, and a seam added there without being added to the net fails here.

    `scheduler` is the one exception and it is listed by name with a reason: the
    lifespan owns it, so clearing it would be tidying up after the code under
    test rather than after a test.
    """
    touching = _modules_that_touch_app_state()
    assert touching, "nothing found — the parse is matching the wrong shape"
    assert APP_SOURCE in touching, (
        "the service module itself no longer appears — the scan is looking in "
        "the wrong place")
    reads = set().union(*touching.values())

    unprotected = reads - set(APP_STATE_SEAMS) - {"scheduler"}
    assert not unprotected, (
        f"the service reads {sorted(unprotected)} off app.state, and the net in "
        f"conftest does not clear it: a test installing one leaks it into every "
        f"test collected after. Found in: "
        f"{sorted(str(p) for p, n in touching.items() if n & unprotected)}")


#: `pythonpath` holds both `engines` and `.`, so every service module can be
#: imported under two dotted names — and Python then builds **two module
#: objects**, each with its own globals and, for the app, its own `app.state`.
_ENGINE_PREFIXES = ("engines.kraken_train_svc.", "kraken_train_svc.")


def _engine_imports(path: Path) -> set[tuple[str, str]]:
    """`(prefix, submodule)` for every service module *path* imports.

    From the AST, because the first version of this check was a substring search
    for `engines.kraken_train_svc` — and it flagged three files, two of them for
    the *comment* explaining this very fix and one for importing `runner` rather
    than `app`. A text search cannot tell a use from an explanation; that is the
    third time this shape has given a wrong answer here.
    """
    found: set[tuple[str, str]] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module, *(f"{node.module}.{a.name}" for a in node.names)]
        else:
            continue
        for name in names:
            for prefix in _ENGINE_PREFIXES:
                if name.startswith(prefix):
                    found.add((prefix, name[len(prefix):].split(".")[0]))
                    break
    return found


def test_each_service_module_is_imported_under_one_name():
    """Two names for one file are two modules, and the suite then holds two of
    whatever that file defines.

    For `app` it is `app.state`: a seam installed through one name is invisible
    to code reached through the other, which falls back to `get_settings()` and
    the real `.env`. For `runner` it was two `Pipeline` classes — eight test
    files, six under one name and two under the other, which is the kind of
    split that shows up as an `isinstance` that is simply false.

    Both were latent when found: `test_two_cards` only called `schedule_once`,
    a pure function handed everything explicitly. Latent is where this belongs
    — it goes live at the first `_store()` or `isinstance`, in a test nobody is
    reading at the time.
    """
    paths: dict[str, set[str]] = {}
    for path in sorted(Path("tests").glob("test_*.py")):
        if path.name == Path(__file__).name:
            continue                      # names both prefixes on purpose
        for prefix, submodule in _engine_imports(path):
            paths.setdefault(submodule, set()).add(prefix)

    assert paths, "no service imports found — the parse is matching the wrong shape"
    split = {name: sorted(prefixes) for name, prefixes in paths.items() if len(prefixes) > 1}
    assert not split, (
        f"imported under two names, so loaded twice: {split} — pick one prefix "
        f"per module; the suite's is `kraken_train_svc.`")

    assert APP_MODULES[0] == "kraken_train_svc.app", (
        "the net's first module is no longer the path the suite uses")


# ── the state that outlives a test without being on app.state ───────────────
#
# Same class, one level down. `app.state` was the seam a test could see; these
# two are a module global and an environment default, and a test that trips
# either leaves nothing behind that points at it.

def test_the_settings_cache_is_empty_inside_a_test():
    """`get_settings()` answers from a module global that nothing ever reset.

    Whichever test filled it would pin *its* environment for the session, and
    the four fixtures that pin a root per test would quietly stop working from
    there on. Reset on both sides of every test — before as well, because a
    module-level `get_settings()` anywhere would fill it at **collection**,
    before any fixture has run.
    """
    from atr_training import settings as settings_module

    assert settings_module._settings is None


def test_the_cache_is_reset_on_both_sides_of_the_test():
    """Structural, because the before-reset is the half no behaviour can show:
    a cache filled during collection is already there when the first test
    starts, and only a reset *before* that test clears it."""
    from tests import conftest

    source = inspect.getsource(conftest._the_settings_cache_is_never_a_previous_test_s
                               ._get_wrapped_function())
    before, _, after = source.partition("yield")
    assert "_settings = None" in before, (
        "nothing is reset before the test — a cache filled at collection time "
        "survives into the first test")
    assert "_settings = None" in after, (
        "nothing is reset after the test — the last test's cache outlives the "
        "session it was built in")


def test_the_fallback_settings_write_nowhere_real(tmp_path):
    """What `get_settings()` hands back when a test installs none.

    Four roots, each one a place a run writes tens of GB to: here they default
    under `$HOME`, and on asteraix — whose `.env` this reads — the job store is
    the research share. Pinned per test, so the day a test forgets its own
    settings it writes into a tmp directory rather than into the live store.
    """
    from atr_training.settings import get_settings

    fallback = get_settings()
    roots = {
        "jobs_root": fallback.jobs_root,
        "trained_root": fallback.trained_root,
        "checkpoint_root": fallback.checkpoint_root,
        "artefact_cache_root": fallback.artefact_cache_root,
        "registry_root": fallback.registry_root,
    }
    base = tmp_path.parent.parent          # pytest's per-session tmp root
    for name, root in roots.items():
        assert Path.home() not in Path(root).parents, (
            f"{name} is {root} — a pipeline test would write into $HOME")
        assert str(base) in str(root), (
            f"{name} is {root}, which is not under this session's tmp root: it "
            f"is a real place, and on asteraix it is the research share")


def test_the_fallback_still_carries_the_test_identity_and_key():
    """The counter-test: pinning the roots must not cost the rails that were
    already there — the fake host id and the key the suite's routes expect."""
    from atr_training.settings import get_settings
    from tests.conftest import TEST_API_KEY, TEST_HOST_ID

    fallback = get_settings()
    assert fallback.host_id == TEST_HOST_ID
    assert fallback.api_key == TEST_API_KEY
    assert fallback.legacy_job_host == "idhefix"

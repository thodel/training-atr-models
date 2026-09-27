"""The scripts run against the training package alone (#6).

Twelve scripts came over from serving-atr-inference. The point of moving them is
not tidiness: while they imported `atr_serving.training`, running any of them
needed a checkout of the *serving* repository on the path, on a box whose job is
training. A script that still reaches back is a script that works here by
accident, and the accident holds until somebody deletes the other checkout.

So each one is imported in a subprocess whose path holds this repository and
nothing else, and the static check says what the dynamic one cannot: that no
`atr_serving` import is hiding behind a conditional or an `importlib` call.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

#: The nine that imported `atr_serving.training` and needed the package move
#: first, plus the three that moved on their own (`make_split`, `smoke_trocr`,
#: `check_env.sh` — the shell one is checked statically only).
MOVED = [
    "artefact_cache.py",
    "audit_eval_material.py",
    "check_line_geometry.py",
    "compare_eval_reports.py",
    "eval_granularity.py",
    "make_split.py",
    "plan_corpus.py",
    "publish_to_hub.py",
    "smoke_trocr.py",
    "stratified_eval_set.py",
    "tei_edition_to_hf.py",
]

ALL_MOVED = MOVED + ["check_env.sh"]


@pytest.mark.parametrize("name", ALL_MOVED)
def test_the_script_is_here(name: str):
    assert (SCRIPTS / name).is_file(), f"{name} did not come across"


@pytest.mark.parametrize("name", ALL_MOVED)
def test_no_script_reaches_back_into_the_serving_package(name: str):
    """Static, because an import inside a function or behind a try would pass
    the run below and still fail on a box without the other checkout."""
    text = (SCRIPTS / name).read_text(encoding="utf-8")

    assert "atr_serving" not in text, (
        f"{name} still names atr_serving; rewrite it to atr_training, or say "
        "which serving module it genuinely needs and why it moved at all")


@pytest.mark.parametrize("name", MOVED)
def test_every_moved_script_runs_with_the_training_package_alone(name: str):
    """Import each script in a subprocess that has only this repository.

    `-I` isolates: no user site-packages, no inherited `PYTHONPATH`. What is left
    is this checkout's `src` and the installed dependencies — which is what a
    training box has.
    """
    code = (
        "import importlib.util, sys;"
        f"sys.path.insert(0, {str(ROOT / 'src')!r});"
        f"spec = importlib.util.spec_from_file_location('moved', {str(SCRIPTS / name)!r});"
        "m = importlib.util.module_from_spec(spec);"
        "spec.loader.exec_module(m)"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code],
                            capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, (
        f"{name} does not import with this repository alone:\n{result.stderr[-2000:]}")


def test_merge_loras_did_not_come_along():
    """It belongs to serving: it turns an adapter into something the gateway can
    serve, and every path it touches is a gateway setting. Moving it here would
    put serving work on the training box (#6)."""
    assert not (SCRIPTS / "merge_loras.py").exists()

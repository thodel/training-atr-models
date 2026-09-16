"""What the scaffold promises, held to it.

These are not placeholders waiting for the real suite. Each one pins a decision
that `pyproject.toml` states in prose and that nothing else would catch if it
quietly stopped being true — and the second one was paid for in the serving repo
before it was a rule here.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


@pytest.fixture(scope="module")
def project() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


#: Imported inside functions, never at module level, because the trainer service
#: spawns its runners in per-engine venvs and must not carry their pins.
ML_STACK = ("torch", "transformers", "peft", "datasets", "accelerate",
            "bitsandbytes", "kraken", "trl")


def test_the_orchestration_dependencies_carry_no_ml_stack(project):
    """A test that starts needing torch is a design regression, not a missing pin.

    The trainer service imports fastapi, loguru, pydantic, pydantic-settings and
    pyyaml at module level and nothing else; torch, transformers, peft, datasets
    and PIL are imported inside the functions that use them. That is what lets
    CI run in 15 minutes on a runner with no GPU, and what keeps the engine venvs
    the only place their versions are decided.
    """
    declared = " ".join(project["project"]["dependencies"]).lower()
    found = [name for name in ML_STACK if name in declared]
    assert not found, (
        f"{found} declared as an orchestration dependency. These belong in the "
        "per-engine venvs (.venvs/{kraken,trocr,vlm}-train), which exist so their "
        "pins never enter this tree."
    )


def test_ruff_is_pinned_not_floored(project):
    """`ruff>=x` let CI judge a tree against rules the author never ran.

    In the serving repo a floated ruff installed 0.16.2 in CI while the tree had
    been cleaned against 0.15.20, and 0.16 selects rules 0.15 does not: 125
    errors in the first run against a tree that was clean locally. A linter whose
    rule set changes between the machine you fix it on and the machine that
    judges it cannot be a gate.
    """
    dev = " ".join(project["project"]["optional-dependencies"]["dev"])
    assert "ruff==" in dev, f"ruff must be pinned with ==, got: {dev}"


def test_the_test_paths_match_the_layout(project):
    """Three roots, and each one is load-bearing.

    `src` for the package, `.` for scripts/, `engines` for the three train
    services — which are imported as top-level modules (`kraken_train_svc.app`),
    not as part of the package. Dropping any one of them makes `pytest` behave
    differently from what a developer runs locally.
    """
    paths = project["tool"]["pytest"]["ini_options"]["pythonpath"]
    assert paths == ["src", ".", "engines"], paths

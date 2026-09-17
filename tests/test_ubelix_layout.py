"""ubelix/ after the move from serving-atr-inference (#7).

The job scripts run on UBELIX from a checkout of THIS repository. Two things
went wrong there before the move and must not come back with it: a path that
still names the old repository, and a helper run from a stale copy in
~/ubelix instead of from the checkout the job was submitted with.
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

UBELIX = Path(__file__).resolve().parents[1] / "ubelix"
SCRIPTS = sorted([*UBELIX.glob("*.sbatch"), *UBELIX.glob("*.sh")])
CODE = sorted([*SCRIPTS, *UBELIX.glob("*.py"), *UBELIX.glob("*.def")])


def _ids(paths):
    return [p.name for p in paths]


def test_the_scripts_are_here():
    names = {p.name for p in SCRIPTS}
    assert {"prepare.sbatch", "train.sbatch", "submit.sh", "status.sh",
            "chain_train_score.sbatch", "score_federal_minutes.sbatch"} <= names


@pytest.mark.parametrize("path", CODE, ids=_ids(CODE))
def test_no_sbatch_file_points_at_the_old_repo_path(path):
    text = path.read_text(encoding="utf-8")
    assert "serving-atr-inference" not in text
    assert "atr_serving" not in text


@pytest.mark.parametrize("path", SCRIPTS, ids=_ids(SCRIPTS))
def test_helpers_and_chained_jobs_run_from_the_checkout(path):
    # ~/ubelix holds images, logs and specs. A helper copied there goes stale:
    # prepare.sbatch once ran a copy of submit_job.py that imported the old package.
    stale = re.findall(r"\$HOME/ubelix/[\w-]+\.(?:py|sbatch|sh)", path.read_text(encoding="utf-8"))
    assert stale == []


@pytest.mark.parametrize("path", SCRIPTS, ids=_ids(SCRIPTS))
def test_repo_is_defined_before_it_is_used(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    uses = [i for i, line in enumerate(lines) if "$REPO" in line and not line.lstrip().startswith("#")]
    if not uses:
        return
    defs = [i for i, line in enumerate(lines) if re.match(r"\s*REPO=", line)]
    assert defs and defs[0] < uses[0], f"$REPO used on line {uses[0] + 1} before it is set"


@pytest.mark.parametrize("path", SCRIPTS, ids=_ids(SCRIPTS))
def test_every_script_parses(path):
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0


def test_the_repo_default_is_this_repository_and_overridable():
    for path in SCRIPTS:
        for line in path.read_text(encoding="utf-8").splitlines():
            if re.match(r"\s*REPO=", line):
                assert line.strip() == "REPO=${ATR_TRAIN_REPO:-$HOME/training-atr-models}", path.name


def test_the_preflight_checks_this_repository(monkeypatch):
    monkeypatch.delenv("ATR_TRAIN_REPO", raising=False)
    spec = importlib.util.spec_from_file_location("pf_layout", UBELIX / "preflight.py")
    pf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pf)
    assert pf.REPO.endswith("/training-atr-models")

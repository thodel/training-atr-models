"""ubelix/submit.sh pins the code it submits (#147, part 2) — with a stand-in sbatch."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

UBELIX = Path(__file__).resolve().parents[1] / "ubelix"

JOB = """#!/bin/bash
#SBATCH --job-name=t
#SBATCH --qos=job_gratis
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
echo hi
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def env(tmp_path: Path):
    """A clean, up-to-date checkout holding ubelix/, and an sbatch that reports what it got."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    repo = tmp_path / "training-atr-models"
    subprocess.run(["git", "clone", "-q", str(origin), str(repo)], check=True,
                   capture_output=True)
    _git(repo, "config", "user.email", "t@example.org")
    _git(repo, "config", "user.name", "t")
    _git(repo, "checkout", "-q", "-b", "main")
    shutil.copytree(UBELIX, repo / "ubelix", ignore=shutil.ignore_patterns("__pycache__"))
    (repo / "ubelix" / "t.sbatch").write_text(JOB)
    _git(repo, "add", "ubelix")
    _git(repo, "commit", "-q", "-m", "ubelix")
    _git(repo, "push", "-q", "origin", "main")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "sbatch"
    fake.write_text('#!/bin/bash\necho "ARGS=$*"\necho "PIN=${ATR_CODE_COMMIT:-}"\n')
    fake.chmod(0o755)

    e = {k: v for k, v in os.environ.items() if k not in ("ATR_CODE_COMMIT", "ATR_UNPINNED")}
    e.update(PATH=f"{bin_dir}:{e['PATH']}", ATR_TRAIN_REPO=str(repo), HOME=str(tmp_path))
    return repo, e


def _submit(repo: Path, env: dict, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(repo / "ubelix" / "submit.sh"),
                           str(repo / "ubelix" / "t.sbatch"), *extra],
                          env=env, capture_output=True, text=True)


def _out(done: subprocess.CompletedProcess, key: str) -> str:
    return next(line for line in done.stdout.splitlines() if line.startswith(key + "="))[len(key) + 1:]


def test_a_submission_is_pinned_to_head(env):
    repo, e = env
    done = _submit(repo, e)
    assert done.returncode == 0, done.stderr
    assert _out(done, "PIN") == _git(repo, "rev-parse", "HEAD")


def test_an_explicit_pin_is_kept(env):
    repo, e = env
    e["ATR_CODE_COMMIT"] = "a" * 40
    assert _out(_submit(repo, e), "PIN") == "a" * 40


def test_unpinned_is_an_explicit_choice(env):
    repo, e = env
    e["ATR_UNPINNED"] = "1"
    done = _submit(repo, e)
    assert done.returncode == 0, done.stderr
    assert _out(done, "PIN") == ""
    assert "UNPINNED" in done.stdout


def test_sbatch_options_after_the_separator_reach_sbatch(env):
    repo, e = env
    done = _submit(repo, e, "--", "--cpus-per-task=2", "--time=00:30:00")
    assert "--cpus-per-task=2 --time=00:30:00" in _out(done, "ARGS")


def test_an_over_cap_request_never_reaches_sbatch(env):
    repo, e = env
    done = _submit(repo, e, "--", "--cpus-per-task=16", "--time=24:00:00")
    assert done.returncode != 0
    assert "ARGS=" not in done.stdout
    assert "CPU-minutes" in done.stderr


def test_a_dirty_checkout_never_reaches_sbatch(env):
    repo, e = env
    (repo / "ubelix" / "t.sbatch").write_text(JOB + "# edited\n")
    done = _submit(repo, e)
    assert done.returncode != 0
    assert "ARGS=" not in done.stdout

"""ubelix/pin_code.sh — a batch job runs the commit it was submitted with (#147, part 2)."""
from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

PIN = Path(__file__).resolve().parents[1] / "ubelix" / "pin_code.sh"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path: Path):
    """A checkout with two commits: `old` (submitted) and `new` (pulled later)."""
    repo = tmp_path / "training-atr-models"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.org")
    _git(repo, "config", "user.name", "t")
    (repo / "code.py").write_text("VERSION = 'old'\n")
    _git(repo, "add", "code.py")
    _git(repo, "commit", "-q", "-m", "old")
    old = _git(repo, "rev-parse", "HEAD")
    (repo / "code.py").write_text("VERSION = 'new'\n")
    _git(repo, "commit", "-qam", "new")
    return repo, old, _git(repo, "rev-parse", "HEAD")


def _pin(repo: Path, root: Path, commit: str | None) -> subprocess.CompletedProcess:
    """Source pin_code.sh as a batch file would, and report where REPO ended up."""
    env = {k: v for k, v in os.environ.items() if k != "ATR_CODE_COMMIT"}
    env.update(ATR_CODE_ROOT=str(root))
    if commit is not None:
        env["ATR_CODE_COMMIT"] = commit
    script = (f'set -uo pipefail; REPO="{repo}"; source "{PIN}"; '
              'pin_code || exit 7; echo "REPO=$REPO"')
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)


def _repo_of(done: subprocess.CompletedProcess) -> Path:
    return Path(next(line for line in done.stdout.splitlines() if line.startswith("REPO="))[5:])


def test_a_pinned_job_runs_the_submitted_commit_not_the_pulled_one(checkout, tmp_path):
    repo, old, _new = checkout
    done = _pin(repo, tmp_path / "wt", old)
    assert done.returncode == 0, done.stderr
    code = _repo_of(done)
    assert code != repo
    assert _git(code, "rev-parse", "HEAD") == old
    assert (code / "code.py").read_text() == "VERSION = 'old'\n"
    assert "pinned to" in done.stdout


def test_a_requeued_job_reuses_the_same_tree(checkout, tmp_path):
    repo, old, _ = checkout
    first = _repo_of(_pin(repo, tmp_path / "wt", old))
    (first / "marker").write_text("from the first start")
    second = _repo_of(_pin(repo, tmp_path / "wt", old))
    assert second == first
    assert (second / "marker").exists()


def test_the_worktree_is_a_real_checkout_so_the_commit_can_be_recorded(checkout, tmp_path):
    # atr_training.codeversion reads `git rev-parse --show-toplevel` and HEAD.
    repo, old, _ = checkout
    code = _repo_of(_pin(repo, tmp_path / "wt", old))
    assert Path(_git(code, "rev-parse", "--show-toplevel")).resolve() == code.resolve()
    assert _git(code, "status", "--porcelain", "--untracked-files=no") == ""


def test_an_unknown_commit_stops_the_job(checkout, tmp_path):
    repo, _, _ = checkout
    done = _pin(repo, tmp_path / "wt", "0" * 40)
    assert done.returncode == 7
    assert "cannot create a worktree" in done.stderr


def test_a_tree_at_the_wrong_commit_is_refused(checkout, tmp_path):
    repo, old, new = checkout
    root = tmp_path / "wt"
    code = _repo_of(_pin(repo, root, old))
    _git(code, "checkout", "-q", "--detach", new)    # someone moved it by hand
    done = _pin(repo, root, old)
    assert done.returncode == 7
    assert "refusing to run" in done.stderr


def test_without_a_pin_the_checkout_runs_as_it_is_and_says_so(checkout, tmp_path):
    repo, _, _ = checkout
    done = _pin(repo, tmp_path / "wt", None)
    assert done.returncode == 0, done.stderr
    assert _repo_of(done) == repo
    assert "UNPINNED" in done.stdout
    assert not (tmp_path / "wt").exists()


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock not installed (it is on UBELIX and in CI)")
def test_fan_out_arms_starting_together_share_one_tree(checkout, tmp_path):
    repo, old, _ = checkout
    with ThreadPoolExecutor(max_workers=6) as pool:
        runs = list(pool.map(lambda _: _pin(repo, tmp_path / "wt", old), range(6)))
    assert all(r.returncode == 0 for r in runs), [r.stderr for r in runs]
    assert len({_repo_of(r) for r in runs}) == 1


def test_every_batch_file_that_sets_repo_pins_it_immediately():
    for path in sorted(PIN.parent.glob("*.sbatch")):
        lines = path.read_text(encoding="utf-8").splitlines()
        sets = [i for i, line in enumerate(lines) if line.startswith("REPO=")]
        if not sets:
            continue
        following = [line for line in lines[sets[0] + 1:] if not line.startswith("#")]
        assert following[0] == 'source "$REPO/ubelix/pin_code.sh"; pin_code || exit 1', path.name

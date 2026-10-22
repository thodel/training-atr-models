"""#150: a requeue Slurm does not accept during a preemption is not a job that ends.

Fifteen preemptions of 16298912 printed "(requeue refused; job will end)" and
the job ran on every time, because `--requeue` had already put it back before
the trap's `scontrol requeue` ran. The trap has one fact to tell a preemption
from the wall: the wall's TERM arrives 120 s before the limit. These tests run
the trap's function with a fake `scontrol` and `squeue` on PATH, so each of the
three answers is produced, not read off the script.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

UBELIX = Path(__file__).resolve().parents[1] / "ubelix"
HELPER = UBELIX / "requeue_trap.sh"


def _fake_slurm(tmp_path: Path, *, requeue_exit: int, requeue_stderr: str = "",
                time_left: str | None = "23:40:00") -> Path:
    """`scontrol` and `squeue` as the trap will find them, and a log of calls."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    calls = tmp_path / "calls.log"
    (bin_dir / "scontrol").write_text(
        '#!/usr/bin/env bash\necho "scontrol $*" >> "%s"\n'
        'if [ "$1" = requeue ]; then [ -n "%s" ] && echo "%s" >&2; exit %d; fi\nexit 0\n'
        % (calls, requeue_stderr, requeue_stderr, requeue_exit), encoding="utf-8")
    squeue = '#!/usr/bin/env bash\necho "squeue $*" >> "%s"\n' % calls
    squeue += ("exit 1\n" if time_left is None else 'echo "%s"\n' % time_left)
    (bin_dir / "squeue").write_text(squeue, encoding="utf-8")
    for name in ("scontrol", "squeue"):
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IEXEC)
    return bin_dir


def _run(bin_dir: Path, job_id: str = "16298912") -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", '. "%s" && requeue_once; echo "rc=$?"' % HELPER],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "SLURM_JOB_ID": job_id})


def test_a_requeue_the_trap_got_accepted_says_so(tmp_path):
    done = _run(_fake_slurm(tmp_path, requeue_exit=0))
    assert "requeued 16298912 by this trap" in done.stdout
    assert done.stdout.strip().endswith("rc=0")


def test_a_preemption_is_not_called_refused_and_the_job_goes_on(tmp_path):
    """The fifteen lines of 16298912: requeue not accepted, 23 h 40 of wall left."""
    done = _run(_fake_slurm(tmp_path, requeue_exit=1,
                            requeue_stderr="Job is pending execution", time_left="23:40:00"))
    out = done.stdout
    assert "refused" not in out.lower() and "will end" not in out, out
    assert "preemption" in out and "the job goes on" in out
    assert "Slurm requeues 16298912 itself" in out
    assert "Job is pending execution" in out, "what scontrol said is kept, not hidden"


def test_a_refusal_at_the_wall_is_the_one_case_that_ends(tmp_path):
    """Within the signal window the TERM is the wall's; a requeue Slurm does not
    accept there really does end the job — and the line says so, with what
    scontrol said and what to do."""
    done = _run(_fake_slurm(tmp_path, requeue_exit=1, requeue_stderr="Requested operation is "
                            "presently disabled", time_left="1:58"))
    out = done.stdout
    assert "REFUSED at the wall" in out and "will end as TIMEOUT" in out and "resubmit" in out
    assert "presently disabled" in out


def test_without_an_answer_from_squeue_nothing_is_claimed(tmp_path):
    done = _run(_fake_slurm(tmp_path, requeue_exit=1, time_left=None))
    out = done.stdout
    assert "not known from here" in out and "sacct -j 16298912" in out
    assert "refused" not in out.lower() and "will end" not in out and "goes on" not in out


def test_the_function_never_ends_the_shell(tmp_path):
    """A trap body that exits non-zero would end the batch script on the
    requeue's status; the function returns 0 whatever it found."""
    for exit_code, left in ((0, "1:00:00"), (1, "1:00:00"), (1, "1:00"), (1, None)):
        done = _run(_fake_slurm(tmp_path / f"{exit_code}-{left}", requeue_exit=exit_code, time_left=left))
        assert done.stdout.strip().endswith("rc=0"), done.stdout


@pytest.mark.parametrize("text, seconds", [
    ("1-02:03:04", 93784), ("02:03:04", 7384), ("03:04", 184), ("4", 4), ("23:40:00", 85200),
    ("UNLIMITED", ""), ("", ""), ("N/A", ""), ("1:x", ""),
])
def test_slurm_time_left_is_parsed_like_the_probe(text, seconds):
    done = subprocess.run(["bash", "-c", '. "%s" && slurm_seconds "%s"' % (HELPER, text)],
                          capture_output=True, text=True, timeout=30)
    assert done.stdout.strip() == str(seconds)


# ── both batch files use it, in the trap and after an exit 75 ───────────────

@pytest.mark.parametrize("name", ["train.sbatch", "train_resumable.sbatch"])
def test_the_batch_files_source_the_helper_and_call_it_from_the_trap(name):
    text = (UBELIX / name).read_text(encoding="utf-8")
    active = [line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    assert any(re.match(r'^\. "\$REPO/ubelix/requeue_trap\.sh"$', line.strip()) for line in active), \
        f"{name} does not source the helper"
    trap = re.search(r"trap '(.*?)' TERM", text, re.S).group(1)
    assert "requeue_once" in trap
    assert "scontrol requeue" not in trap, "the trap calls scontrol itself again"
    assert "refused" not in trap and "will end" not in trap


def test_the_exit_75_path_uses_the_same_words():
    """The asymmetry the issue names: the branch after exit 75 had the right
    message and a bare `|| true`; now it is the same function, once."""
    text = (UBELIX / "train_resumable.sbatch").read_text(encoding="utf-8")
    active = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    assert not any("scontrol requeue" in line for line in active), "a bare scontrol requeue remains"
    assert any(line == '[ -n "$REQUEUED" ] || requeue_once' for line in active)

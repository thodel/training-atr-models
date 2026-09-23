"""#21: the stage log the child writes must survive the share going away.

On 09.09. the network went while a run was training. `train.log` of
20260909T190659Z-qwen3vl-german-pages-v2 ends mid-progress-line at step 628,
with no traceback: the failure that killed the run also erased its description,
because the child's stdout was a file handle on the share.

The child now writes into a pipe, a reader thread appends to a log beside the
checkpoint root, and mirrors to the share. These tests pin the three properties
the issue names: the local copy is complete, the share copy is live, and the
share half is optional.

Offline: real subprocesses, no GPU, no network.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from atr_training.runner_base import SubprocessRunner


def _runner(tmp_path: Path) -> tuple[SubprocessRunner, Path, Path]:
    local_root, share_root = tmp_path / "nvme", tmp_path / "share"
    runner = SubprocessRunner(local_root=local_root, share_root=share_root)
    share_log = share_root / "20260923T000000Z-m" / "logs" / "train.log"
    local_log = local_root / "20260923T000000Z-m" / "logs" / "train.log"
    return runner, share_log, local_log


def _echo(text: str) -> list[str]:
    return [sys.executable, "-c", f"print({text!r})"]


#: Prints PAYLOAD without the word appearing in the command, so a test can count
#: occurrences in the log without also counting the header.
_PAYLOAD = [sys.executable, "-c", "print(chr(80) + 'AY' + 'LOAD')"]


def test_both_copies_carry_the_output_and_the_command(tmp_path):
    runner, share_log, local_log = _runner(tmp_path)

    assert runner.run(_echo("step 628"), share_log) == 0

    for log in (local_log, share_log):
        body = log.read_text(encoding="utf-8")
        assert "step 628" in body
        assert "$ " in body and "print" in body, "the header lost the command"


def test_the_header_is_the_command_not_the_expression_that_builds_it(tmp_path):
    """An f-string with doubled braces wrote `{' '.join(cmd)}` into the log."""
    runner, share_log, local_log = _runner(tmp_path)
    runner.run(_echo("x"), share_log)
    assert "join(cmd)" not in local_log.read_text(encoding="utf-8")


def test_the_local_copy_is_complete_when_the_share_cannot_be_written(tmp_path):
    """The case the issue is about: the share is what disappears."""
    runner, share_log, local_log = _runner(tmp_path)
    share_log.parent.mkdir(parents=True, exist_ok=True)

    lines = [sys.executable, "-c",
             "import sys\nfor i in range(200): print('line', i)\nsys.exit(3)"]

    real_open = Path.open

    def failing_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        if self == share_log:
            def gone(*_a, **_kw):
                raise OSError(5, "Input/output error")   # the share, mid-run
            handle.write = gone
        return handle

    Path.open = failing_open
    try:
        exit_code = runner.run(lines, share_log)
    finally:
        Path.open = real_open

    assert exit_code == 3, "the child's exit code must survive the mirror"
    body = local_log.read_text(encoding="utf-8")
    assert "line 0" in body and "line 199" in body, "the local copy lost output"


def test_the_share_copy_is_written_while_the_job_still_runs(tmp_path):
    """Live tailing: /train/jobs/{id} reads the share log of a running job.

    The child prints, then waits for a sentinel file. If the share copy only
    appeared at the end of the stage, this would time out.
    """
    runner, share_log, _ = _runner(tmp_path)
    sentinel = tmp_path / "go"
    child = [sys.executable, "-c",
             "import sys, time, pathlib\n"
             "print('started', flush=True)\n"
             f"p = pathlib.Path({str(sentinel)!r})\n"
             "for _ in range(200):\n"
             "    if p.exists(): break\n"
             "    time.sleep(0.05)\n"]

    import threading
    result: list[int] = []
    worker = threading.Thread(target=lambda: result.append(runner.run(child, share_log)))
    worker.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if share_log.exists() and "started" in share_log.read_text(encoding="utf-8"):
                break
            time.sleep(0.05)
        else:
            raise AssertionError("nothing reached the share log while the job ran")
    finally:
        sentinel.write_text("go", encoding="utf-8")
        worker.join(timeout=10)
    assert result == [0]


def test_a_box_without_a_separate_local_root_writes_one_file(tmp_path):
    """Both roots local: mirroring a file onto itself would double every line."""
    log = tmp_path / "jobs" / "j" / "logs" / "train.log"
    runner = SubprocessRunner(local_root=tmp_path / "jobs", share_root=tmp_path / "jobs")

    assert runner.run(_PAYLOAD, log) == 0

    assert log.read_text(encoding="utf-8").count("PAYLOAD") == 1


def test_a_log_outside_the_share_root_is_written_once(tmp_path):
    runner = SubprocessRunner(local_root=tmp_path / "nvme", share_root=tmp_path / "share")
    log = tmp_path / "elsewhere" / "train.log"

    assert runner.run(_PAYLOAD, log) == 0

    assert log.read_text(encoding="utf-8").count("PAYLOAD") == 1
    assert not (tmp_path / "nvme").exists()


def test_a_local_root_without_a_share_root_writes_one_file(tmp_path):
    """Half a configuration must not double the log.

    The first version of this defaulted the local root and left the share root
    unset, so both handles opened the same file and the mirror wrote every chunk
    to it twice — the feature inert and the log corrupted.
    """
    log = tmp_path / "jobs" / "j" / "logs" / "train.log"
    runner = SubprocessRunner(local_root=tmp_path / "nvme")

    assert runner.run(_PAYLOAD, log) == 0

    assert log.read_text(encoding="utf-8").count("PAYLOAD") == 1

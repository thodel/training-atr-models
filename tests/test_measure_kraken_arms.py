"""`measure_kraken_arms` — the two places it can lie, and the one it can leak (#163).

The measurement itself needs a card and the kraken stack. These cover what does
not: whether a stopped run is labelled as the floor it is, and whether stopping
one actually ends it. A ketos run forks dataloader workers under a new session,
so signalling only the child leaves them reparented to init, still holding a
card — a "stopped" run that keeps 43 GB busy.
"""
from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "measure_kraken_arms", REPO / "scripts" / "measure_kraken_arms.py")
mka = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mka
_SPEC.loader.exec_module(mka)


# ── the label ───────────────────────────────────────────────────────────────
def test_a_completed_run_is_a_peak_and_a_stopped_one_is_a_floor():
    """The distinction #163 insists on: a lower bound may be reported, not
    reported as a limit.

    The first run of this script finished a whole epoch over the arrow in 215 s
    and was nonetheless labelled "time-bounded", which would have put a complete
    measurement in the table as if it were a truncated one.
    """
    source = (REPO / "scripts" / "measure_kraken_arms.py").read_text(encoding="utf-8")
    assert "full epoch over the arrow" in source
    assert "arrow head, stopped after" in source
    # Both branches hang off the exit code, not off whether the timeout fired:
    # a run can exit non-zero for its own reasons and must not then claim
    # completeness.
    assert 'proc.returncode == 0' in source


def test_the_manifest_is_built_not_the_arrow_passed():
    """`ketos train -t` wants a manifest of .arrow paths, one per line.

    Handing it the arrow earns "File … is not a text file" after 2.8 seconds,
    which is how the first attempt died.
    """
    source = (REPO / "scripts" / "measure_kraken_arms.py").read_text(encoding="utf-8")
    assert "binary_manifest" in source, "use the pipeline's own helper"
    assert "train_bin.lst" in source


# ── stopping a run really stops it ──────────────────────────────────────────
def test_stop_ends_the_whole_session_not_only_the_child(tmp_path):
    """A grandchild is the shape ketos arrives in: sh -c spawning python."""
    marker = tmp_path / "grandchild.pid"
    # Written to a file rather than passed with -c: the nesting of sh's quotes
    # around python's around a repr'd path is how the first version of this test
    # handed sh an unquoted path and got a SyntaxError.
    child_py = tmp_path / "child.py"
    child_py.write_text(
        "import os, pathlib, time\n"
        f"pathlib.Path(r{str(marker)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n", encoding="utf-8")
    # `& wait` rather than a bare command: a POSIX shell is allowed to exec a
    # single simple command and replace itself, and then there is no grandchild
    # and the test proves nothing. Backgrounding forces the fork.
    proc = subprocess.Popen(["/bin/sh", "-c", f"{sys.executable} {child_py} & wait"],
                            start_new_session=True)
    for _ in range(200):
        if marker.exists():
            break
        time.sleep(0.05)
    grandchild = int(marker.read_text())
    assert grandchild != proc.pid, (
        "the shell replaced itself, so this is one generation and the test would "
        "pass without ever exercising the process group")

    mka.stop(proc)

    assert proc.poll() is not None, "the child we spawned is gone"
    for _ in range(100):
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(grandchild, signal.SIGKILL)
        raise AssertionError(
            f"grandchild {grandchild} survived stop() — reparented to init and "
            "still holding whatever it held")


def test_stopping_an_already_dead_process_is_not_an_error(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    proc.wait()

    mka.stop(proc)  # must not raise

    assert proc.poll() == 0

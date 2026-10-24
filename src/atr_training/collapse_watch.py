"""Stop a run from scratch that has collapsed, while it runs (#131).

The step-count guard judges a configuration before anything is trained, and on
the German sweep corpus it let through two runs that learned nothing: 2,778
steps for CER 1.0000 in 83 minutes, and 10,186 steps for CER 0.985 in 4.9 hours.
No floor fixes that, because what decides is the material, and the guard cannot
see the material. The run can: Lightning writes each epoch's validation accuracy
into the checkpoint's file name, so the checkpoint directory says, epoch by
epoch, whether a single character has been read yet.

:class:`CollapseWatch` polls that directory beside the ketos process and asks
:func:`~atr_training.convergence.collapse_reason`. When the answer is a reason,
it ends the process group — SIGTERM, then SIGKILL after a grace period — and
keeps the reason, which the stage turns into the job's error instead of
"ketos exited -15". It stops nothing else and never on its own judgement of a
number: the rule is the function's, and ``force`` on the request disarms it.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from pathlib import Path
from typing import Callable

from loguru import logger

from atr_training.convergence import COLLAPSE_PATIENCE, FLOOR_FROM_SCRATCH, collapse_reason
from atr_training.curves import curve_from_checkpoints

__all__ = ["CollapseWatch", "POLL_S", "TERM_GRACE_S"]

#: An epoch on the corpora here takes minutes; a minute's delay costs nothing.
POLL_S = 60.0
#: Lightning shuts down on SIGTERM within seconds; after this, SIGKILL.
TERM_GRACE_S = 120.0


class CollapseWatch:
    """Watch ``checkpoint_dir`` while the process ``pid`` trains; stop it on collapse."""

    def __init__(self, checkpoint_dir: Path, steps_per_epoch: int, *,
                 floor: int = FLOOR_FROM_SCRATCH, patience: int = COLLAPSE_PATIENCE,
                 poll_s: float = POLL_S, grace_s: float = TERM_GRACE_S,
                 killpg: Callable[[int, int], None] = os.killpg,
                 alive: Callable[[int], bool] | None = None) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.steps_per_epoch = steps_per_epoch
        self.floor = floor
        self.patience = patience
        self.poll_s = poll_s
        self.grace_s = grace_s
        self._killpg = killpg
        self._alive = alive or _alive
        #: Set once the run was stopped as collapsed; the stage's error.
        self.reason: str | None = None
        self._pid: int | None = None
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, pid: int) -> "CollapseWatch":
        """Begin watching ``pid``, the leader of ketos' own session."""
        self._pid = pid
        self._thread = threading.Thread(target=self._loop, name="collapse-watch", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._done.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_s + 5)

    def check(self) -> str | None:
        """One look at the checkpoints; the reason if the run has collapsed."""
        points = curve_from_checkpoints(self.checkpoint_dir).points
        return collapse_reason(points, self.steps_per_epoch, self.floor, self.patience)

    def _loop(self) -> None:
        while not self._done.wait(self.poll_s):
            try:
                reason = self.check()
            except Exception as exc:   # noqa: BLE001 — a watcher must not end a run by failing
                logger.warning("collapse watch: could not read {}: {}", self.checkpoint_dir, exc)
                continue
            if reason is None:
                continue
            self.reason = reason
            logger.warning("collapse watch: stopping pid {} — {}", self._pid, reason)
            self._end()
            return

    def _end(self) -> None:
        pid = self._pid
        if pid is None:
            return
        try:
            self._killpg(pid, signal.SIGTERM)
        except OSError:
            return
        deadline = time.monotonic() + self.grace_s
        while time.monotonic() < deadline:
            if not self._alive(pid):
                return
            if self._done.wait(1.0):
                return
        try:
            self._killpg(pid, signal.SIGKILL)
        except OSError:
            pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

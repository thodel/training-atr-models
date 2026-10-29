"""Stop a run from scratch that has collapsed, while it runs (#131).

The step-count guard judges a configuration before anything is trained, and on
the German sweep corpus it let through two runs that learned nothing: 2,778
steps for CER 1.0000 in 83 minutes, and 10,186 steps for CER 0.985 in 4.9 hours.
No floor fixes that, because what decides is the material, and the guard cannot
see the material. The run can: Lightning writes a checkpoint every epoch, and a
checkpoint can be asked how much it writes.

:class:`CollapseWatch` polls the checkpoint directory beside the ketos process.
Each checkpoint past the floor is measured once, by :class:`LengthProbe` — its
length ratio on a sample of the validation pages — and the measurements go to
:func:`~atr_training.convergence.collapse_reason`. When the answer is a reason,
the watch ends the process group — SIGTERM, then SIGKILL after a grace period —
and keeps the reason, which the stage turns into the job's error instead of
"ketos exited -15". Once a checkpoint reads, the watch stands down: nothing it
could measure later would stop the run. It stops nothing else and never on its
own judgement of a number: the rule is the function's, and ``force`` on the
request disarms it.

Why not the validation accuracy in the checkpoint's name, which costs nothing?
That was the rule until the probe for #131: a collapse that creeps reads
0.0001, 0.0007 … 0.0083 and never shows the exact 0 the rule waited for.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Mapping

from loguru import logger

from atr_training.convergence import (
    COLLAPSE_LENGTH_RATIO,
    COLLAPSE_PATIENCE,
    FLOOR_FROM_SCRATCH,
    collapse_reason,
    past_the_floor,
)
from atr_training.curves import parse_checkpoint
from atr_training.ketos_cmd import compile_cmd, convert_cmd, evaluate_cmd, parse_test_report
from atr_training.manifests import binary_manifest, read_manifest, write_manifest

__all__ = ["CollapseWatch", "LengthProbe", "POLL_S", "TERM_GRACE_S", "SAMPLE_PAGES"]

#: An epoch on the corpora here takes minutes; a minute's delay costs nothing.
POLL_S = 60.0
#: Lightning shuts down on SIGTERM within seconds; after this, SIGKILL.
TERM_GRACE_S = 120.0
#: Validation pages the length ratio is measured on. 58 pages of the #131 probe
#: were 53,331 characters, and a checkpoint took 24 s to convert and read on an
#: A40 beside nothing else; the whole validation set took 4 min 19 s.
SAMPLE_PAGES = 60
#: One conversion or one ``ketos test`` on the sample; far above the 17 s measured.
PROBE_TIMEOUT_S = 900.0


class LengthProbe:
    """The length ratio a kraken checkpoint reads on a sample of the validation pages.

    ``ketos test`` cannot read a checkpoint on kraken 7.0.2, so each one is
    converted to a coreml file first, read, and the file removed again. The
    sample is compiled on first use, from every n-th page of ``pages_val`` so it
    spans the set rather than its first documents. Any failure is a ``None`` and
    a warning, never an exception: a probe that cannot measure must not end a
    run, and a sample that cannot be built is not built again every minute.
    """

    def __init__(self, ketos: str | Path, pages_val: Path, workdir: Path, *,
                 device: str = "cuda:0", workers: int = 8, normalization: str | None = "NFD",
                 env: Mapping[str, str] | None = None, sample_pages: int = SAMPLE_PAGES,
                 timeout_s: float = PROBE_TIMEOUT_S,
                 run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> None:
        self.ketos = ketos
        self.pages_val = Path(pages_val)
        self.workdir = Path(workdir)
        self.device = device
        self.workers = workers
        self.normalization = normalization
        #: Overrides on the service's environment, as the training stage gets them.
        self.env = {**os.environ, **env} if env is not None else None
        self.sample_pages = sample_pages
        self.timeout_s = timeout_s
        self._run = run
        self._sample: Path | None = None
        self._broken: str | None = None

    def __call__(self, checkpoint: Path) -> float | None:
        sample = self._sample_manifest()
        if sample is None:
            return None
        weights = self.workdir / f"{checkpoint.stem}.mlmodel"
        log = self.workdir / f"{checkpoint.stem}.log"
        try:
            if not self._ok(convert_cmd(self.ketos, checkpoint=checkpoint, output=weights),
                            log, f"convert {checkpoint.name}"):
                return None
            if not self._ok(evaluate_cmd(self.ketos, model=weights, manifest=sample,
                                         device=self.device, workers=self.workers,
                                         normalization=self.normalization),
                            log, f"test {checkpoint.name}", append=True):
                return None
            metrics = parse_test_report(log.read_text(encoding="utf-8", errors="replace"))
        finally:
            weights.unlink(missing_ok=True)
        if metrics.length_ratio is None:
            logger.warning("collapse watch: {} gave no length ratio — see {}", checkpoint.name, log)
        return metrics.length_ratio

    def _sample_manifest(self) -> Path | None:
        if self._sample is not None or self._broken is not None:
            return self._sample
        try:
            pages = read_manifest(self.pages_val)
        except OSError as exc:
            self._broken = f"cannot read {self.pages_val}: {exc}"
        else:
            if not pages:
                self._broken = f"{self.pages_val} lists no pages"
        if self._broken is None:
            self.workdir.mkdir(parents=True, exist_ok=True)
            step = max(1, math.ceil(len(pages) / self.sample_pages))
            manifest = write_manifest(self.workdir / "sample_pages.lst", pages[::step])
            arrow = self.workdir / "sample.arrow"
            if self._ok(compile_cmd(self.ketos, manifest=manifest, output=arrow,
                                    device=self.device, workers=self.workers),
                        self.workdir / "compile.log", "compile the sample") and arrow.exists():
                self._sample = binary_manifest(self.workdir / "sample_bin.lst", arrow)
            else:
                self._broken = f"the sample did not compile — see {self.workdir / 'compile.log'}"
        if self._broken is not None:
            logger.warning("collapse watch: no length sample, so no checkpoint is measured: {}",
                           self._broken)
        return self._sample

    def _ok(self, cmd: list[str], log: Path, what: str, *, append: bool = False) -> bool:
        try:
            with log.open("a" if append else "w", encoding="utf-8") as fh:
                fh.write("$ " + " ".join(cmd) + "\n")
                fh.flush()
                code = self._run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=self.env,
                                 timeout=self.timeout_s).returncode
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("collapse watch: {} failed: {}", what, exc)
            return False
        if code != 0:
            logger.warning("collapse watch: {} exited {} — see {}", what, code, log)
        return code == 0


class CollapseWatch:
    """Watch ``checkpoint_dir`` while the process ``pid`` trains; stop it on collapse."""

    def __init__(self, checkpoint_dir: Path, steps_per_epoch: int,
                 measure: Callable[[Path], float | None], *,
                 floor: int = FLOOR_FROM_SCRATCH, patience: int = COLLAPSE_PATIENCE,
                 threshold: float = COLLAPSE_LENGTH_RATIO,
                 poll_s: float = POLL_S, grace_s: float = TERM_GRACE_S,
                 killpg: Callable[[int, int], None] = os.killpg,
                 alive: Callable[[int], bool] | None = None) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.steps_per_epoch = steps_per_epoch
        self.measure = measure
        self.floor = floor
        self.patience = patience
        self.threshold = threshold
        self.poll_s = poll_s
        self.grace_s = grace_s
        self._killpg = killpg
        self._alive = alive or _alive
        #: Set once the run was stopped as collapsed; the stage's error.
        self.reason: str | None = None
        #: Epoch → length ratio, for every checkpoint past the floor measured so far.
        self.ratios: dict[int, float] = {}
        #: Set once a checkpoint read; the watch then stands down.
        self.reading = False
        self._tried: set[int] = set()
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
        """Measure the checkpoints not yet measured; the reason if the run has collapsed."""
        found = []
        if self.checkpoint_dir.is_dir():
            for path in self.checkpoint_dir.glob("checkpoint_*.ckpt"):
                point = parse_checkpoint(path.name)
                if (point is not None and point.epoch not in self._tried
                        and past_the_floor(point.epoch, self.steps_per_epoch, self.floor)):
                    found.append((point.epoch, path))
        for epoch, path in sorted(found):
            if self._done.is_set():
                break
            self._tried.add(epoch)
            try:
                ratio = self.measure(path)
            except Exception as exc:   # noqa: BLE001 — a measurement must not end a run by failing
                logger.warning("collapse watch: could not measure {}: {}", path.name, exc)
                ratio = None
            if ratio is None:
                continue
            self.ratios[epoch] = ratio
            logger.info("collapse watch: epoch {} ({:,} steps) length ratio {:.4f}",
                        epoch, (epoch + 1) * self.steps_per_epoch, ratio)
            if ratio > self.threshold:
                self.reading = True
                return None
            reason = collapse_reason(self.ratios, self.steps_per_epoch, self.floor,
                                     self.patience, self.threshold)
            if reason is not None:
                return reason
        return None

    def _loop(self) -> None:
        while not self._done.wait(self.poll_s):
            try:
                reason = self.check()
            except Exception as exc:   # noqa: BLE001 — a watcher must not end a run by failing
                logger.warning("collapse watch: could not read {}: {}", self.checkpoint_dir, exc)
                continue
            if self.reading:
                logger.info("collapse watch: the run reads (length ratio above {:.2f}) — standing down",
                            self.threshold)
                return
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

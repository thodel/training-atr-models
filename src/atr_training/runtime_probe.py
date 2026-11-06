"""What a run looks like from outside, right now (#250, agentic_historian#366).

Every diagnosis in agentic_historian#366 was made over ssh with ``/proc`` and
``du``, and that is not an accident of habit: for a kraken run the trainer knows
almost nothing from the inside. ``ketos`` renders its metrics through rich and
the log keeps none of them (#51), so ``Progress.epoch`` is written **once, after
training returns**, from the checkpoint filenames that survived. During the nine
hours #366 describes, the record had nothing to say.

What the trainer *can* do is look at the process, because it runs on the same
box. That is what this module is: a sample taken at read time.

Why this is not part of :class:`~atr_training.contracts.Progress`
----------------------------------------------------------------
``Progress`` is what the run **recorded** — written by the runner at stage
boundaries and true of a moment in the past. These values are what is **true
now**, and they stop being true a second later. Mixing the two is how
``progress.epoch`` came to read as a live counter while meaning "the highest
epoch among the surviving top-ten checkpoints", which is a different statement
and a worse one to confuse. So an observation carries its own ``at`` and lives
under its own key.

What this deliberately does **not** report
------------------------------------------
#366 also asks for ``it/s``, the early-stopping counter and the previous epoch's
metric for a trend arrow. None of those is honestly available today, and
inventing them is the failure this repository keeps removing:

* **``cer_previous``** — the curve is a *selection*, not a series: kraken keeps
  the top ten checkpoints, so the point before the last one is the next-best
  surviving epoch and not the previous epoch. A trend arrow computed from it
  would point whichever way the selection happened to fall.
  :attr:`TrainingCurve.still_improving` is the correctly scoped version of that
  diagnostic and already exists.
* **the early-stopping counter** — needs the epoch currently training, which
  nothing knows (see above).
* **``it/s``** — ``ketos`` prints it and keeps it nowhere; #51 is the
  prerequisite, and log scraping was already ruled out there.

A field holding ``None`` because nobody has got round to it reads the same as a
field holding ``None`` because the number cannot exist. These are absent instead.

The compile phase is the one of #366's asks that *is* measurable, and it was
nearly refused here for the wrong reason. A chunked compile writes a record
beside each finished chunk's arrow — not for an observer, but so a resumed run
knows what not to rebuild (#72). Counting them is reading what the run wrote, so
:func:`compile_chunks` reports it. What stays unavailable is the position
*inside* one ``ketos compile``, which is a single invocation that says nothing
until it exits.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from atr_training.chunking import CHUNK_PLAN_FILENAME, finished_chunk, read_plan

#: Below this, two samples count as the same reading. Disk and RSS both jitter —
#: a page cache flush, an allocator returning a block — and a "growing" that
#: flickers on noise is worse than no arrow at all, because #366's whole point is
#: that flat and growing mean opposite things.
TREND_FLOOR_PCT = 2.0

#: A sample older than this is not compared against: the gap between two polls
#: tells you nothing about a rate if the second came half an hour late.
TREND_MAX_GAP_S = 900.0


@dataclass(frozen=True)
class Sample:
    """One observation, with the clock reading that makes it comparable."""

    at: float
    rss_mib: int | None
    job_dir_mib: int | None


def rss_mib(pid: int | None, *, proc: Path | None = None) -> int | None:
    """Resident set size of *pid* in MiB, or None when it cannot be read.

    None covers every reason at once on purpose — no pid on the record, the
    process has exited, another user's process, not Linux. The caller's question
    is "can I show a number", and a 0 would answer it wrongly: a run whose
    process is gone is the case where 0 MiB reads as "idle" and means "dead".

    ``VmRSS`` rather than ``statm``: it is already in kB and labelled, so there
    is no page size to multiply by and get wrong.
    """
    if not pid:
        return None
    root = proc or Path("/proc")
    try:
        status = (root / str(int(pid)) / "status").read_text(encoding="ascii")
    except (OSError, ValueError):
        return None
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1]) // 1024
                except ValueError:
                    return None
    return None


def dir_mib(path: Path | str | None) -> int | None:
    """Total size under *path* in MiB, or None when it cannot be walked.

    This is the "temp arrow growing 1 GB every five minutes" signal, which is
    what told #366 that a nine-hour ``compile`` was healthy rather than wedged.
    Walked rather than ``statvfs``'d: the question is how much *this run* has
    written, and free space on the volume answers a different one (that is
    ``GET /host``).
    """
    if path is None:
        return None
    total = 0
    try:
        # ``onerror=_raise`` is what makes a missing or unreadable directory a
        # None rather than a 0: os.walk swallows the opening scandir error by
        # default and yields nothing, which is indistinguishable from an empty
        # directory — and those two must not read alike.
        for dirpath, _dirnames, filenames in os.walk(Path(path), onerror=_raise):
            for name in filenames:
                try:
                    total += (Path(dirpath) / name).stat().st_size
                except OSError:
                    continue          # a file written away under us is not an error
    except OSError:
        return None
    return total // (1024 * 1024)


def _raise(error: OSError) -> None:
    raise error


#: How long a reader waits for a walk before answering with the last finished
#: one. Long enough that a local job directory is answered fresh, short enough
#: that a reader never waits on the share.
DIR_WALK_WAIT_S = 2.0

#: A job directory is walked at most this often, however often it is asked
#: about. The bot polls every 60 s and a sweep driver as often; two readers must
#: not mean two walks over the same twenty thousand files.
DIR_WALK_EVERY_S = 60.0


class DirWalker:
    """Job-directory sizes, walked beside the request rather than inside it.

    :func:`dir_mib` stats every file, and a job directory on the research share
    holds one image per page during ``prepare`` — 21,062 of them for sweep 2's
    first cell, where ``GET /jobs/<id>`` took over two minutes and the sweep
    driver, with a 30 s timeout, died of it twice (#402). A size that costs that
    much cannot be taken at read time.

    So a reader starts a walk when the last one is older than ``every_s`` and
    none is running, waits for it ``wait_s`` at most, and is answered with the
    last *finished* walk — which carries its own ``at``, because a size from
    three minutes ago is not a size from now. Before the first walk finishes
    the answer is no reading at all, not a zero.
    """

    def __init__(self, *, wait_s: float = DIR_WALK_WAIT_S,
                 every_s: float = DIR_WALK_EVERY_S,
                 walk: Callable[[Path | str | None], int | None] = dir_mib,
                 clock: Callable[[], float] = time.time) -> None:
        self.wait_s = wait_s
        self.every_s = every_s
        self._walk_fn = walk
        self._clock = clock
        self._lock = threading.Lock()
        #: key -> (the walk before the last, the last finished walk)
        self._done: dict[str, tuple[Sample | None, Sample]] = {}
        self._running: dict[str, threading.Event] = {}

    def read(self, key: str, path: Path | str | None) -> tuple[Sample | None, Sample | None]:
        """The last two finished walks of *path*, oldest first; None for each
        that has not happened."""
        with self._lock:
            pair = self._done.get(key)
            latest = pair[1] if pair else None
            event = self._running.get(key)
            if event is None and (latest is None
                                  or self._clock() - latest.at >= self.every_s):
                event = threading.Event()
                self._running[key] = event
                threading.Thread(target=self._walk, args=(key, path, event),
                                 name=f"dir-walk-{key}", daemon=True).start()
        if event is not None:
            event.wait(self.wait_s)
        with self._lock:
            return self._done.get(key, (None, None))

    def forget(self, key: str) -> None:
        """Drop what is known of *key*; a walk still running is discarded when
        it ends. A service that runs for months must not keep every job."""
        with self._lock:
            self._done.pop(key, None)
            self._running.pop(key, None)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._done or key in self._running

    def _walk(self, key: str, path: Path | str | None, event: threading.Event) -> None:
        try:
            mib = self._walk_fn(path)
            sample = Sample(at=self._clock(), rss_mib=None, job_dir_mib=mib)
            with self._lock:
                if self._running.get(key) is event:      # not forgotten meanwhile
                    previous = self._done.get(key, (None, None))[1]
                    self._done[key] = (previous, sample)
        finally:
            with self._lock:
                if self._running.get(key) is event:
                    del self._running[key]
            event.set()


def compile_chunks(data_dir: Path | str | None) -> dict | None:
    """``{"done": n, "expected": m | None}``, or None when there is nothing to count.

    This is the answer to "is the nine-hour compile moving?" that #366 had to get
    over ssh. It is not an estimate: a chunk counts as done on exactly the
    evidence the resume loop uses — :func:`atr_training.chunking.finished_chunk`
    — so a chunk reported here is one the runner would not rebuild.

    None rather than ``{"done": 0}`` when the run is not chunked: an unchunked
    compile is one ``ketos`` invocation with no inner structure, and a zero would
    read as a chunked compile that has finished nothing. ``expected`` is None
    when the plan has no page cap, because the length of a streamed selection is
    not known in advance — the count still climbs, which is the signal.
    """
    if data_dir is None:
        return None
    data = Path(data_dir)
    plan_path = data / CHUNK_PLAN_FILENAME
    try:
        plan = read_plan(plan_path)
    except (OSError, ValueError, KeyError):
        return None
    try:
        arrows = sorted(data.glob("train_[0-9]*.arrow"))
    except OSError:
        return None
    return {"done": sum(1 for arrow in arrows if finished_chunk(arrow) is not None),
            "expected": plan.expected_chunks}


def trend(previous: Sample | None, current: Sample, field: str) -> str | None:
    """``"growing"`` / ``"flat"`` / ``"falling"``, or None when it cannot be said.

    None is the answer in four cases, and each of them is a real one rather than
    a placeholder: there is no previous sample (the first poll, or the service
    restarted), the previous one is too old to compare against, either reading
    is missing, or the two are equal in time. A direction asserted from one
    sample is the thing #366 complains about from the other side — a single
    value cannot tell "learning slowly" from "flat on a plateau".
    """
    if previous is None:
        return None
    before = getattr(previous, field, None)
    after = getattr(current, field, None)
    if before is None or after is None:
        return None
    gap = current.at - previous.at
    if gap <= 0 or gap > TREND_MAX_GAP_S:
        return None
    if before == 0:
        return "growing" if after > 0 else "flat"
    change_pct = (after - before) / before * 100.0
    if abs(change_pct) < TREND_FLOOR_PCT:
        return "flat"
    return "growing" if change_pct > 0 else "falling"

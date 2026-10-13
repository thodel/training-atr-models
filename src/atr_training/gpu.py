"""What the GPUs on *this* machine are doing, and who is doing it.

**A deliberate duplicate** of ``serving-atr-inference/src/atr_serving/gpu.py``
(serving#137, T2.7). The gateway still needs its copy for its own vLLM budget;
this one exists because after the split the gateway's ``/train/gpu`` could only
read idhefix's cards and match asteraix's job pids against idhefix's ``/proc`` —
where a pid collision marks a local stranger as ``registered`` to a foreign job
and drops it from ``unaccounted_mib``, silently (#13). A pid means something
only on the host that issued it, so the reading has to happen here. Importing
the gateway's module is not an option either: this repo never imports the
serving half (tests/test_isolation.py).

``inspect`` and its helpers are the gateway's, unchanged, so the two files diff
cleanly. Not copied: ``card_memory``, the gateway's launch-sizing helper — the
trainer sizes a start with :func:`atr_training.preflight.check_vram`. Added:
:func:`card_rows`, the totals the gateway's route computed inline — with one
divergence, made explicit there: on this host an unregistered process in one of
our units counts as unaccounted, because here no unit of ours holds memory
except through a job.

The gateway's own account of why this module exists:

The scheduler's view and the machine's state drift apart, and nothing surfaces
the gap. Three incidents in two weeks, each found by hand and late:

- a PyTorch data-loader worker outlived its training process, kept
  ``/dev/nvidia1`` mapped and so kept a dead parent's CUDA context alive. It held
  27 530 MiB under the process name ``[Not Found]`` — a pid with no ``/proc``
  entry — for sixteen hours while both cards reported 0 % utilisation and a
  queued job waited for memory nobody was using.
- a three-day run was "cancelled on request" by nobody in that session.
- a 30-hour ``ketos`` run appeared on a card belonging to no job at all.

This module answers, for every process holding GPU memory: who owns it, how old
it is, what it is running, and — the part that matters — whether it belongs to a
job the trainer knows about.

`nvidia-smi` is shelled out to rather than binding NVML: the service stays free
of ML dependencies (pyproject.toml), and the two queries used here are stable CSV.
"""

from __future__ import annotations

import os
import pwd
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field

#: Give up rather than hang a request on a wedged driver.
TIMEOUT_S = 8

#: Unit-name prefixes that mark a service as this deployment's own. Everything
#: else on the card belongs to somebody else and stays in the unaccounted total —
#: a foreign process displacing a training run is precisely what #414 is about.
OWN_UNIT_PREFIXES = ("atr-",)

CARD_QUERY = ("index,name,memory.total,memory.used,memory.free,"
              "utilization.gpu,utilization.memory")
APP_QUERY = "pid,used_gpu_memory,gpu_uuid"


@dataclass
class Process:
    pid: int
    used_mib: int
    #: No ``/proc`` entry. The process is gone but its memory is not: this is the
    #: ``[Not Found]`` row, and it is the one worth waking someone for.
    orphaned: bool = False
    #: It is, or descends from, a pid the trainer recorded for a job. Descent
    #: matters: a data-loader worker of a healthy run is a child, not a stray, and
    #: flagging every one of them would bury the signal.
    registered: bool = False
    job_id: str | None = None
    #: The systemd unit the process belongs to, from its cgroup. Answers the
    #: question a bare command line does not: whose process is this.
    service: str | None = None
    #: One of *our* services rather than someone else's. A run of ours holding
    #: memory is expected; anyone else's process on the card is not, and the two
    #: must not be summed into one number. (The case this was written for was the
    #: neighbours' RAG service on the serving box; asteraix has no such tenant
    #: today, which is why unaccounted memory here is a leftover of our own run.)
    own_service: bool = False
    user: str | None = None
    age_s: float | None = None
    command: str | None = None


@dataclass
class Card:
    index: int
    name: str
    memory_total_mib: int
    memory_used_mib: int
    memory_free_mib: int
    utilisation_pct: int
    memory_utilisation_pct: int
    processes: list = field(default_factory=list)


def _smi(query: str, *, per_app: bool) -> list[list[str]]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise FileNotFoundError("nvidia-smi is not on PATH")
    flag = "--query-compute-apps" if per_app else "--query-gpu"
    out = subprocess.run(
        [exe, f"{flag}={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=TIMEOUT_S, check=True).stdout
    return [[c.strip() for c in line.split(",")]
            for line in out.splitlines() if line.strip()]


def _int(value: str) -> int:
    """nvidia-smi writes '[N/A]' and '[Not Supported]' where a number belongs."""
    try:
        return int(float(value))
    except ValueError:
        return 0


def _float(value: str) -> float:
    """Same, for `power.draw` — which is one of the fields that answers '[N/A]'.

    A card or driver that does not report wattage yields 0.0, and an energy total
    of 0 Wh is then the honest reading: nothing was measured. The caller must not
    read it as "drew no power" — `Footprint` refuses to publish a zero.
    """
    try:
        return float(value)
    except ValueError:
        return 0.0


def _age_seconds(pid: int) -> float | None:
    """Seconds since the process started, from its own start time.

    Field 22 of ``/proc/<pid>/stat`` is the start time in clock ticks since boot;
    with ``/proc/uptime`` that gives an age that does not depend on anything
    having touched the directory. The mtime of ``/proc/<pid>`` is the obvious
    shortcut and is not the same thing — and "sixteen hours" is the number the
    whole endpoint exists to report, so it should be the real one.
    """
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            fields = fh.read().rsplit(") ", 1)[-1].split()
        starttime_ticks = int(fields[19])       # field 22, minus pid/comm/state
        with open("/proc/uptime", encoding="utf-8") as fh:
            uptime = float(fh.read().split()[0])
        hz = os.sysconf("SC_CLK_TCK") or 100
        return max(0.0, uptime - starttime_ticks / hz)
    except (FileNotFoundError, IndexError, ValueError, PermissionError, OSError):
        return None


def _proc_info(pid: int) -> tuple[str | None, float | None, str | None]:
    """(user, age in seconds, command) — all None when /proc has no such pid."""
    base = f"/proc/{pid}"
    try:
        st = os.stat(base)
        with open(f"{base}/cmdline", "rb") as fh:
            raw = fh.read().replace(b"\0", b" ").strip()
        command = raw.decode("utf-8", "replace") or None
        if command is None:                     # kernel thread: comm is all there is
            with open(f"{base}/comm", encoding="utf-8") as fh:
                command = f"[{fh.read().strip()}]"
        try:
            user = pwd.getpwuid(st.st_uid).pw_name
        except KeyError:                        # uid with no passwd entry
            user = str(st.st_uid)
        return user, _age_seconds(pid), command
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None, None, None


def _unit_of(pid: int) -> str | None:
    """The systemd unit owning this pid, from ``/proc/<pid>/cgroup``.

    Read from the cgroup rather than asked of systemd, for two reasons: it needs
    no privileges and works for other users' processes, and it covers *children*.
    ``systemctl show -p MainPID`` names one process; an engine's workers and a
    trainer's data loaders are not it, and those are exactly the rows that would
    otherwise look unexplained.

        0::/user.slice/.../app.slice/atr-trocr.service   -> atr-trocr.service
        0::/system.slice/gunicorn.service                -> gunicorn.service
    """
    try:
        with open(f"/proc/{pid}/cgroup", encoding="utf-8") as fh:
            path = fh.read().strip().rsplit(":", 1)[-1]
    except (FileNotFoundError, PermissionError, OSError):
        return None
    for part in reversed(path.split("/")):
        if part.endswith(".service"):
            return part
    return None


def _ancestors(pid: int, limit: int = 32) -> list:
    """pid and its ancestors, nearest first. Empty when /proc has no such pid."""
    chain, seen = [], set()
    current = pid
    for _ in range(limit):
        if current in seen or current <= 1:
            break
        seen.add(current)
        chain.append(current)
        try:
            with open(f"/proc/{current}/stat", encoding="utf-8") as fh:
                fields = fh.read().rsplit(") ", 1)[-1].split()
            current = int(fields[1])            # ppid, after state
        except (FileNotFoundError, IndexError, ValueError, PermissionError):
            break
    return chain


def inspect(job_pids: dict | None = None) -> list:
    """Every card with the processes holding memory on it.

    ``job_pids`` maps a trainer pid to its job id. A process counts as registered
    when it is one of those pids or descends from one.
    """
    job_pids = job_pids or {}
    cards, by_uuid = [], {}
    for row in _smi(CARD_QUERY, per_app=False):
        index, name, total, used, free, util, mem_util = (row + [""] * 7)[:7]
        cards.append(Card(index=_int(index), name=name,
                          memory_total_mib=_int(total), memory_used_mib=_int(used),
                          memory_free_mib=_int(free), utilisation_pct=_int(util),
                          memory_utilisation_pct=_int(mem_util)))
    # A card's uuid is not in the gpu query above; map by index order, which
    # nvidia-smi keeps consistent between the two queries.
    uuids = [r[0] for r in _smi("uuid", per_app=False)]
    for card, uuid in zip(cards, uuids):
        by_uuid[uuid] = card

    for row in _smi(APP_QUERY, per_app=True):
        pid_s, used_s, uuid = (row + [""] * 3)[:3]
        pid = _int(pid_s)
        user, age, command = _proc_info(pid)
        chain = _ancestors(pid)
        job_id = next((job_pids[p] for p in chain if p in job_pids), None)
        unit = _unit_of(pid)
        process = Process(
            pid=pid, used_mib=_int(used_s), orphaned=user is None,
            registered=job_id is not None, job_id=job_id,
            service=unit,
            own_service=bool(unit and unit.startswith(OWN_UNIT_PREFIXES)),
            user=user, age_s=None if age is None else round(age, 1),
            command=command)
        card = by_uuid.get(uuid)
        if card is not None:
            card.processes.append(process)
    return cards


def card_rows(cards: list, *, services_expected: bool) -> list[dict]:
    """The cards as JSON rows, each with the three totals the route reports.

    The arithmetic of the gateway's ``/train/gpu`` local reading
    (``train_routes._local_reading``), with one rule made a parameter, because
    the two hosts differ in exactly that rule — and required, so no caller
    inherits the other host's answer:

    ``services_expected=True`` is the gateway's: memory held by one of our units
    is explained even when no job owns it. On idhefix, where the trainer shared
    the cards with the inference engines, most such memory is an engine's, held
    by design (atr-trocr.service, 1.6 GB on card 1).

    ``services_expected=False`` is the trainer's: on asteraix no unit of ours
    holds GPU memory except through a job. The service itself never touches a
    card (deploy/systemd/atr-train.service), and a runner stays in that unit's
    cgroup — ``start_new_session`` does not leave it, and ``KillMode=process``
    keeps it there past a restart. So an unregistered process in
    atr-train.service is the leftover of a run the store calls finished: a
    data-loader worker that outlived its runner, a ``ketos`` a cancelled job
    left behind. Under the gateway's rule those land in ``service_mib`` only and
    ``unaccounted_mib`` reads 0 — the stray this endpoint exists to show, hidden
    by the field meant to show it, for as long as ``/proc`` still has it (the
    ``[Not Found]`` row has no readable cgroup and counts either way).
    ``service_mib`` is the same under both rules.
    """
    rows = []
    for card in cards:
        procs = [vars(p) for p in card.processes]
        row = {k: v for k, v in vars(card).items() if k != "processes"}
        row["processes"] = procs
        # What nobody here can explain: not a training job, not (where one is
        # expected) one of our services. An engine holding memory is expected
        # and must not be summed with a stray, or the number stops meaning
        # anything and the row that matters gets read past — which is how a
        # sixteen-hour orphan stays invisible.
        row["unaccounted_mib"] = sum(
            p["used_mib"] for p in procs
            if not p["registered"] and not (services_expected and p["own_service"]))
        row["service_mib"] = sum(
            p["used_mib"] for p in procs if p["own_service"])
        row["orphaned_mib"] = sum(
            p["used_mib"] for p in procs if p["orphaned"])
        rows.append(row)
    return rows


# ── peak over time, for an engine that holds no torch handle (#163) ─────────
@dataclass
class Peak:
    """The high-water mark of one subprocess tree, and how trustworthy it is.

    Two numbers per card, because they answer different questions and only one
    of them is this job's:

    * ``own_mib`` sums the compute apps whose ancestor chain contains the pid we
      spawned. That is this run's footprint, and on a shared card it is the only
      honest figure.
    * ``card_mib`` is what the card reported in use at the same instant,
      neighbours included. It is the number an OOM measures itself against, and
      the one comparable to the 45,486 MiB an A40 makes usable.

    Three outcomes, not two, and the third is the one that bites. ``own_seen``
    means at least one compute app was traced back to our pid, so ``own_mib`` is
    measured. ``apps_seen`` without ``own_seen`` means processes were using the
    card and none of them could be traced to us — a PID namespace would do that —
    so ``own_mib`` reads 0 and means "could not look". Neither flag set means no
    process was on the card at all, and then 0 is a measured zero.

    Collapsing the last two is #165's mistake: a reading that failed must not
    arrive looking like a number. The distinction was found to be missing by
    running this against a real idle A40 (asteraix, 07.10.2026): it reported
    "unknown" where 0 was a fact.

    Measured on the same box, same day, against a grandchild — ``sh -c`` spawning
    a python that held 2 GiB, which is the shape ``ketos`` arrives in::

        gpu0: 2354 MiB own / 2363 MiB on the card; gpu1: 0 MiB own / 4 MiB on the card

    Two things that matters: the ancestor chain carries across two generations,
    and the 9 MiB between ``own`` and ``card`` on gpu0 is the CUDA context — the
    card sees it, the per-process figure does not. gpu1's 4 MiB belonged to a
    neighbour and was correctly not attributed.
    """

    own_mib: dict[int, int] = field(default_factory=dict)
    card_mib: dict[int, int] = field(default_factory=dict)
    #: Energy drawn by each card over the sampled interval, in watt-hours. The
    #: integral of `power.draw` against the real time between samples, not
    #: against the nominal interval — a sampler that falls behind must not
    #: under-report. This is the CARD's energy, which is the job's only when the
    #: job has the card to itself; `cards_shared` says when it did not.
    energy_wh: dict[int, float] = field(default_factory=dict)
    #: Another process was seen holding memory on a card we were using, so part
    #: of `energy_wh` is somebody else's.
    cards_shared: bool = False
    readings: int = 0
    failures: int = 0
    #: One of our own processes was found holding memory.
    own_seen: bool = False
    #: Some process — anyone's — was found holding memory.
    apps_seen: bool = False

    @property
    def own_is_unknown(self) -> bool:
        """The card was busy and none of it could be attributed to us."""
        return self.apps_seen and not self.own_seen

    def summary(self) -> str:
        if not self.readings:
            return f"no GPU reading taken ({self.failures} failure(s))"
        if not self.apps_seen:
            return (f"no process on any card over {self.readings} reading(s) — "
                    "a measured zero, not a missing measurement")
        cards = sorted(set(self.own_mib) | set(self.card_mib))
        parts = [f"gpu{i}: {self.own_mib.get(i, 0)} MiB own / "
                 f"{self.card_mib.get(i, 0)} MiB on the card" for i in cards]
        tail = (" — the card was busy but nothing traced back to our pid, so 'own' "
                "is unknown rather than zero") if self.own_is_unknown else ""
        if self.total_energy_wh:
            shared = ", shared with another process" if self.cards_shared else ""
            tail += (f"; {self.total_energy_wh:.1f} Wh drawn by the card(s)"
                     f"{shared}")
        return f"peak over {self.readings} reading(s): " + "; ".join(parts) + tail

    @property
    def total_energy_wh(self) -> float:
        return sum(self.energy_wh.values())


class PeakSampler:
    """Poll nvidia-smi for as long as a subprocess runs, and keep the maximum.

    For engines that never touch torch from Python: ``kraken`` drives ``ketos``
    as an external CLI (``ketos_cmd.train_cmd``), so there is no allocator to ask
    and no number in the log either — ketos renders through ``rich``. The peak
    #163 asks for can only be sampled from outside, which is what this does.

    The attribution is the module's own idiom, from :func:`inspect`: a GPU
    process belongs to this job when the pid we spawned appears in its ancestor
    chain. ``start_new_session=True`` (see ``ProcessRunner.run``) makes that pid a
    session leader, so every descendant ketos forks stays reachable through
    ``/proc``.

    Never raises into the caller: a stage must not fail because a measurement
    could not be taken. Failures are counted and surfaced in
    :meth:`Peak.summary`.
    """

    def __init__(self, pid: int, *, interval_s: float = 2.0) -> None:
        self.pid = pid
        self.interval_s = interval_s
        self.peak = Peak()
        self._last_watts: dict[int, float] = {}
        self._last_at: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "PeakSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> Peak:
        self._stop.set()
        if self._thread is not None:
            # Bounded: one nvidia-smi call has TIMEOUT_S to answer, and a stage
            # must not be held open by a sampler that cannot finish.
            self._thread.join(timeout=TIMEOUT_S + 2)
        return self.peak

    # The caller spawns the process and only then knows the pid, so start/stop are
    # the primary interface; `with` is for the measurement scripts, which do both
    # in one place.
    def __enter__(self) -> "PeakSampler":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._once()
            self._stop.wait(self.interval_s)
        # One last look: a short stage can finish inside the first interval, and
        # a peak of nothing would be the worst kind of measurement — plausible.
        self._once()

    def _once(self) -> None:
        try:
            apps = _smi(APP_QUERY, per_app=True)
            uuids = [r[0] for r in _smi("uuid", per_app=False)]
            used = [_int(r[0]) for r in _smi("memory.used", per_app=False)]
            watts = [_float(r[0]) for r in _smi("power.draw", per_app=False)]
        except Exception:  # noqa: BLE001 — a reading that failed is not a stage that failed
            self.peak.failures += 1
            # The clock is NOT carried across a failed reading: integrating the
            # last known wattage over a gap we did not observe would invent
            # energy. The interval is dropped instead, which under-reports by
            # the length of the outage and says so in `failures`.
            self._last_at = None
            return

        # Trapezoid against the REAL elapsed time: a sampler that falls behind
        # (a loaded box, a slow nvidia-smi) must not under-report by assuming its
        # nominal interval. The first reading only arms the integral.
        now = time.monotonic()
        if self._last_at is not None:
            hours = (now - self._last_at) / 3600.0
            for index, w in enumerate(watts):
                previous = self._last_watts.get(index, w)
                mean_w = (previous + w) / 2.0
                self.peak.energy_wh[index] = (
                    self.peak.energy_wh.get(index, 0.0) + mean_w * hours)
        self._last_at = now
        self._last_watts = dict(enumerate(watts))

        index_of = {uuid: i for i, uuid in enumerate(uuids)}
        own: dict[int, int] = {}
        for row in apps:
            pid_s, used_s, uuid = (row + [""] * 3)[:3]
            index = index_of.get(uuid)
            if index is None:
                continue
            if self.pid in _ancestors(_int(pid_s)):
                own[index] = own.get(index, 0) + _int(used_s)

        self.peak.readings += 1
        if apps:
            self.peak.apps_seen = True
        if own:
            self.peak.own_seen = True
            # Somebody else on a card we are using means part of that card's
            # wattage is theirs, and the energy cannot be attributed to this job.
            for row in apps:
                pid_s, _used, uuid = (row + [""] * 3)[:3]
                index = index_of.get(uuid)
                if index in own and self.pid not in _ancestors(_int(pid_s)):
                    self.peak.cards_shared = True
        for index, mib in own.items():
            self.peak.own_mib[index] = max(self.peak.own_mib.get(index, 0), mib)
        for index, mib in enumerate(used):
            self.peak.card_mib[index] = max(self.peak.card_mib.get(index, 0), mib)

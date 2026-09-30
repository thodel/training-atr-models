"""The consumer `rungs.py` has been waiting three weeks for (#114).

`plan_rungs` and `promote` have been correct and unused since 08.09.2026: one
importer, their own test. Everything around them — the manifest (#113), the
guards in `runner_base`, the job store — was built for a sweep nobody could
start, because the loop that turns a manifest into jobs and job records back into
a ranking did not exist. This is that loop.

Per rung: take this rung's configurations, submit one job each with the rung's
step budget, wait, read the metric out of each job record, hand the scores to
`promote()`, and carry the survivors into the next rung.

Four things it does **not** do, each because the issue or the history says so.

**It never converts a budget into epochs by itself.** `convergence.StepBudget`
already knows what a configuration costs — ``ceil(train_lines / (batch_size ×
accumulate_grad_batches))`` optimizer steps per epoch — and it is the number the
convergence guard judges against. :func:`epochs_for` inverts *that* formula
rather than writing a second one. The first sweep counted micro-batches instead
of optimizer steps and handed the large configurations a quarter of their budget;
two formulas for one quantity is how that happens.

**It never sets ``force``.** In ``runner_base`` that flag overrides both the
convergence guard and the line-geometry guard, recording an override on the job.
A driver that sets it so a sweep runs to completion suspends those guards for
every run, which is exactly what #K7 exists to prevent. A configuration the
guards refuse is a configuration the sweep learned something about.

**It never mixes metrics.** One metric is chosen for the whole sweep, and a job
that does not carry it counts as *unscored* — which `promote()` already handles
correctly: never promoted, never silently dropped. Falling back from
``benchmark_cer`` to ``cer`` per job would rank a held-out number against a
validation number that overlaps training.

**It keeps its state beside the manifest, not inside it.** The issue suggests
writing progress into the sweep file; the manifest refuses unknown top-level keys
(#113), so state written there would make the file unreadable on the next run.
A sidecar also keeps the definition of the experiment and the record of it apart,
which is what lets a manifest be edited — a fourth learning rate added — without
the results of the first three becoming suspect.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from loguru import logger

from atr_training.convergence import epochs_for as _epochs_for
from atr_training.rungs import DEFAULT_ETA, Promotion, plan_rungs, promote
from atr_training.sweep_manifest import SweepConfig, SweepManifest

__all__ = [
    "SweepError", "Preempted", "Metric", "METRICS", "SweepState", "SweepDriver",
    "TrainerClient", "epochs_for", "steps_at_rung", "YIELD_POLICIES", "IDLE_S",
]

#: Job statuses from which no further progress is possible.
TERMINAL = frozenset({"completed", "failed", "cancelled"})

#: A job in one of these holds, or is about to hold, the card.
ACTIVE = frozenset({"queued", "preparing", "compiling", "training", "testing",
                    "registering"})

#: How a sweep gets out of the way when somebody asks for the card (#117).
#:
#: ``cancel`` stops the running cell at once. ``finish_cell`` lets it end and
#: submits nothing more. Which is right depends on what a cell costs: at rung 0
#: it is under an hour and cancelling is cheap, at the last rung it is many hours
#: and cancelling throws all of them away — but waiting them out is exactly the
#: block this issue exists to prevent. The default is ``cancel``, because a sweep
#: that can make a requested run wait hours is not the last tenant of the card,
#: whatever it is called.
YIELD_POLICIES = ("cancel", "finish_cell")

#: How long a serving driver waits before looking again — for a freed card, or
#: for a manifest that gained a configuration.
IDLE_S = 300.0

STATE_VERSION = 1

#: How often a waiting driver asks. A kraken rung is hours; a minute of latency
#: on noticing costs nothing and a tighter loop is a poll storm on a box whose
#: job is training.
POLL_S = 60.0


def _minutes(started: str | None, finished: str | None) -> float | None:
    """Wall-clock minutes between two ISO timestamps, or None.

    #116 asks for the runtime on the table because it changes what a result
    means: h256 costs about twelve times h48 per step, so a gain of 0.0021 for
    70 % more compute is a different statement from a gain of 0.0021.
    """
    if not started or not finished:
        return None
    try:
        a = datetime.fromisoformat(str(started))
        b = datetime.fromisoformat(str(finished))
    except ValueError:
        return None
    return round((b - a).total_seconds() / 60, 1)


class SweepError(RuntimeError):
    """The sweep cannot proceed without a person deciding something."""


class Preempted(RuntimeError):
    """This pass gave the card back. Not a failure — the cell simply did not
    happen, and the next pass will try it again."""


@dataclass(frozen=True)
class Metric:
    """Which number ranks this sweep, and where it lives in a job record.

    ``promote()`` ranks high-first, so an error rate is turned into ``1 - e``
    before it is ranked. The raw value is kept beside it: a leaderboard (#K4)
    reporting 0.8 when the corpus talks in CER 0.2 would be read wrong.
    """

    name: str
    #: Dotted path inside the job record.
    path: str
    higher_is_better: bool

    def raw_of(self, job: Mapping[str, Any]) -> float | None:
        node: Any = job
        for step in self.path.split("."):
            if not isinstance(node, Mapping):
                return None
            node = node.get(step)
        return float(node) if isinstance(node, (int, float)) else None

    def rank_of(self, job: Mapping[str, Any]) -> float | None:
        raw = self.raw_of(job)
        if raw is None:
            return None
        return raw if self.higher_is_better else 1.0 - raw


METRICS: dict[str, Metric] = {
    # The primary quality claim: measured on held-out material (#112), which is
    # what #111's table compares.
    "benchmark_cer": Metric("benchmark_cer", "result.benchmark_cer", False),
    # The validation split, which overlaps the training projects. Secondary, and
    # never mixed with the above in one ranking.
    "cer": Metric("cer", "result.cer", False),
    # What kraken prints while training. Cheapest, and the only one available
    # before a test stage runs.
    "val_accuracy": Metric("val_accuracy", "progress.val_accuracy", True),
}


def epochs_for(steps: int, train_lines: int, effective_batch: int) -> int:
    """:func:`convergence.epochs_for`, with the sweep's own way of saying why.

    The arithmetic is not repeated here: the driver and the convergence guard
    judging its jobs have to agree, and two functions with one formula is how
    they stop agreeing.
    """
    if steps < 1:
        raise SweepError(f"a rung budget must be at least one step, got {steps}")
    if train_lines < 1:
        raise SweepError(
            f"cannot size a rung against {train_lines} training lines. The count "
            "comes from a completed job's progress.train_lines, or from "
            "--train-lines for the first rung.")
    return _epochs_for(steps, train_lines, effective_batch)


def steps_at_rung(base_steps: int, rung: int, eta: int = DEFAULT_ETA) -> int:
    """Successive halving multiplies the budget as the field narrows.

    ``budget.steps`` in the manifest is rung 0's budget — the cheap screening
    pass — not the whole sweep's.
    """
    return base_steps * (eta ** rung)


class TrainerClient(Protocol):
    """What the driver needs of the trainer. Four calls, so a test can be one."""

    def verify(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def submit(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def job(self, job_id: str) -> Mapping[str, Any]: ...
    def jobs(self) -> Sequence[Mapping[str, Any]]: ...
    def cancel(self, job_id: str) -> Mapping[str, Any]: ...


@dataclass
class SweepState:
    """What a restart needs to not do the same work twice."""

    sweep: str
    data_digest: str
    base_steps: int
    metric: str
    train_lines: int | None = None
    #: What the noise floor was measured on, carried so a reader of this file
    #: never has to trust a bare number (#115).
    noise_floor_provenance: dict = field(default_factory=dict)
    #: rung (as a string, because JSON) -> config_id -> {job_id, status, score, raw}.
    #: Rung-major, because a configuration runs at several rungs with different
    #: budgets and therefore different scores. Keyed by configuration alone, a
    #: rung-1 result overwrote its own rung-0 result: the record of what the
    #: cheap screening pass measured was lost, and a restart no longer
    #: recognised rung 0 as finished and ran the whole field again.
    results: dict[str, dict[str, dict]] = field(default_factory=dict)
    promotions: list[dict] = field(default_factory=list)
    path: Path | None = None

    @classmethod
    def for_manifest(cls, manifest: SweepManifest, metric: str,
                     path: Path | None = None) -> "SweepState":
        return cls(sweep=manifest.name, data_digest=manifest.data_digest,
                   base_steps=manifest.steps, metric=metric, path=path,
                   noise_floor_provenance=dict(manifest.noise_floor_provenance))

    @classmethod
    def load(cls, path: Path, manifest: SweepManifest, metric: str) -> "SweepState":
        if not path.is_file():
            return cls.for_manifest(manifest, metric, path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        state = cls(sweep=raw["sweep"], data_digest=raw["data_digest"],
                    base_steps=raw["base_steps"], metric=raw["metric"],
                    train_lines=raw.get("train_lines"),
                    noise_floor_provenance=(raw.get("noise_floor_provenance")
                                            or dict(manifest.noise_floor_provenance)),
                    results=raw.get("results", {}),
                    promotions=raw.get("promotions", []), path=path)
        state.check_against(manifest, metric)
        return state

    def check_against(self, manifest: SweepManifest, metric: str) -> None:
        """Refuse to continue a sweep whose terms changed underneath it.

        Adding a value to an axis is fine and is why `config_id` is what it is
        (#113) — the new configurations simply have no results yet. Changing the
        data, the metric or the base budget is not: earlier rungs were decided on
        the old terms and the ladder would compare across two experiments.
        """
        for field_name, was, now in (("data.digest", self.data_digest, manifest.data_digest),
                                     ("metric", self.metric, metric),
                                     ("budget.steps", self.base_steps, manifest.steps)):
            if was != now:
                raise SweepError(
                    f"{self.path}: this sweep was run with {field_name} = {was!r} "
                    f"and the manifest now says {now!r}. Earlier rungs were "
                    "decided on the old terms. Start a new sweep — a new state "
                    "file — rather than continuing this one.")

    def observe_train_lines(self, lines: int | None) -> None:
        """Record the corpus size, and notice if it ever changes.

        The data digest pins what was *meant* to be trained on; this notices if
        what arrived is a different size, which is the cheapest possible check
        that the material did not move under a multi-day sweep.
        """
        if not lines:
            return
        if self.train_lines is None:
            self.train_lines = lines
            return
        if self.train_lines != lines:
            raise SweepError(
                f"a job reported {lines:,} training lines and this sweep has been "
                f"sizing its budgets against {self.train_lines:,}. The data "
                f"changed under digest {self.data_digest}; every rung so far "
                "measured something else.")

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": STATE_VERSION, "sweep": self.sweep,
            "data_digest": self.data_digest, "base_steps": self.base_steps,
            "metric": self.metric, "train_lines": self.train_lines,
            "noise_floor_provenance": self.noise_floor_provenance,
            "results": self.results, "promotions": self.promotions,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".part")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)       # a killed driver never leaves half a record

    def at(self, rung: int, config_id: str) -> dict:
        return self.results.get(str(rung), {}).get(config_id, {})

    def put(self, rung: int, config_id: str, entry: dict) -> None:
        self.results.setdefault(str(rung), {})[config_id] = entry

    def scored(self, config_id: str, rung: int) -> bool:
        """Finished *and* holding a number.

        A preempted cell is terminal and is not scored: it ran on part of its
        budget, and half a budget is not a measurement (#117). Leaving it out of
        this means the next pass submits it again, which is the whole point.
        """
        entry = self.at(rung, config_id)
        if entry.get("status") not in TERMINAL:
            return False
        return not entry.get("preempted")

    def spent_steps(self) -> int:
        """Optimizer steps this sweep has spent, preempted cells included.

        Included because they cost the card the same, and a ceiling that only
        counts successful work is not a ceiling on what the sweep consumes.
        """
        return sum(int(entry.get("steps") or 0)
                   for rung in self.results.values() for entry in rung.values())


class SweepDriver:
    """A manifest in, a ladder of promotions out."""

    def __init__(self, manifest: SweepManifest, state: SweepState,
                 client: TrainerClient, *, eta: int = DEFAULT_ETA,
                 poll_s: float = POLL_S,
                 sleep: Callable[[float], None] = time.sleep,
                 yield_policy: str = "cancel") -> None:
        self.manifest = manifest
        self.state = state
        self.client = client
        self.eta = eta
        self.poll_s = poll_s
        self.sleep = sleep
        if yield_policy not in YIELD_POLICIES:
            raise SweepError(f"yield policy must be one of {YIELD_POLICIES}, "
                             f"got {yield_policy!r}")
        self.yield_policy = yield_policy
        self.metric = METRICS[state.metric]

    # ── the ladder ──────────────────────────────────────────────────────────
    def ladder(self, n_configs: int) -> list[int]:
        """How many configurations enter each rung.

        The manifest may state it; otherwise `plan_rungs` derives it, which is
        what gives that function its first caller outside its own test.
        """
        if self.manifest.rungs:
            return list(self.manifest.rungs)
        return [rung.configs for rung in plan_rungs(n_configs, eta=self.eta)]

    def request_for(self, config: SweepConfig, rung: int) -> dict:
        """A `TrainRequest` body. Note what is *not* here: `force`."""
        steps = steps_at_rung(self.state.base_steps, rung, self.eta)
        params = dict(config.params)
        batch = int(params.get("batch_size", 1) or 1)
        accumulate = int(params.get("accumulate_grad_batches", 1) or 1)
        params["epochs"] = epochs_for(steps, self.state.train_lines or 0, batch * accumulate)
        return {
            "engine": self.manifest.engine,
            "model_id": f"{self.manifest.name}-r{rung}-{config.config_id}",
            "datasets": [dict(spec) for spec in self.manifest.datasets],
            "base_model": self.manifest.base_model,
            "params": params,
            "notes": (f"sweep {self.manifest.name} rung {rung}, config "
                      f"{config.config_id}, budget {steps} optimizer steps, data "
                      f"{self.manifest.data_digest}"),
        }

    # ── the sweep is the last tenant of the card ────────────────────────────
    def our_job_ids(self) -> set[str]:
        return {str(entry.get("job_id")) for rung in self.state.results.values()
                for entry in rung.values() if entry.get("job_id")}

    def requested_jobs(self) -> list[str]:
        """Active jobs this sweep did not submit.

        Identified by absence from our own state rather than by a marker on the
        job: a marker can be forgotten, and the state file already holds every id
        we asked for. The consequence worth naming is that a *second* sweep's
        jobs read as requested here and would preempt this one — which is why one
        sweep at a time, and why this says so rather than guessing at names.
        """
        mine = self.our_job_ids()
        try:
            jobs = self.client.jobs()
        except Exception as exc:                      # noqa: BLE001 — see below
            # Never let a failed poll look like a free card. A driver that
            # cannot see the queue must assume somebody is in it, because the
            # failure this yields to killed a run with 841 MiB free (#129).
            logger.warning("could not list the trainer's jobs ({}) — assuming the "
                           "card is wanted", exc)
            return ["<unknown>"]
        return [str(j.get("id")) for j in jobs
                if str(j.get("status")) in ACTIVE and str(j.get("id")) not in mine]

    def _preempt(self, config: SweepConfig, rung: int, job_id: str,
                 wanted_by: list[str]) -> None:
        """Get out of the way, and leave no half measurement behind."""
        if self.yield_policy == "cancel":
            try:
                self.client.cancel(job_id)
            except Exception as exc:                  # noqa: BLE001
                logger.warning("could not cancel {} ({}) — it will finish on its "
                               "own; nothing is recorded for it either way",
                               job_id, exc)
        logger.warning(
            "rung {}: yielding the card to {} — {} is {} and its budget is "
            "forfeit. Half a budget is not a measurement, so this configuration "
            "stays unscored and will be run again.",
            rung, ", ".join(wanted_by[:3]), config.config_id,
            "cancelled" if self.yield_policy == "cancel" else "left to finish")
        entry = dict(self.state.at(rung, config.config_id))
        entry.update(status="cancelled", score=None, raw=None, preempted=True,
                     preempted_for=wanted_by[:3])
        self.state.put(rung, config.config_id, entry)
        self.state.save()

    # ── one configuration ───────────────────────────────────────────────────
    def _await(self, job_id: str, config: SweepConfig | None = None,
               rung: int = 0) -> Mapping[str, Any] | None:
        """Wait for a job, unless somebody asks for the card first.

        ``None`` means this cell was given up. It is not an error and not a
        result: the configuration stays unscored and comes back on a later pass.
        """
        while True:
            job = self.client.job(job_id)
            if str(job.get("status")) in TERMINAL:
                return job
            if config is not None:
                wanted_by = self.requested_jobs()
                if wanted_by:
                    self._preempt(config, rung, job_id, wanted_by)
                    return None
            self.sleep(self.poll_s)

    def _record(self, config: SweepConfig, rung: int, job: Mapping[str, Any]) -> None:
        self.state.observe_train_lines(
            (job.get("progress") or {}).get("train_lines"))
        # Merged, not replaced: the budget this cell was given is written at
        # submit, and a fresh dict here dropped it — `spent_steps()` then counted
        # zero and the ceiling never held. Found by its own test.
        self.state.put(rung, config.config_id, {
            **self.state.at(rung, config.config_id),
            "job_id": job.get("id"),
            "status": str(job.get("status")),
            "score": self.metric.rank_of(job),
            "raw": self.metric.raw_of(job),
            "metric": self.metric.name,
            # For the leaderboard (#116), and all of it from the job record so
            # nothing is retyped: the cost, the code, and whether a guard was
            # overridden. That last one matters even though this driver never
            # sets `force` — a CER from a run known not to converge must never be
            # read as an ordinary one, and the table is where it would be.
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
            "minutes": _minutes(job.get("started_at"), job.get("finished_at")),
            "commit": (job.get("code") or {}).get("commit"),
            # A guard that refused this configuration said why, and that sentence
            # is the result (#119). Without it the leaderboard shows an absent
            # CER, which reads like a crash rather than like a configuration the
            # material cannot support.
            "error": job.get("error"),
            "reserved_pages": (job.get("progress") or {}).get("reserved_pages"),
            "reserved_pages_source": (job.get("progress") or {}).get("reserved_pages_source"),
            "overrides": [name for name in ("convergence_override", "geometry_override")
                          if job.get(name)],
        })
        self.state.save()

    def _run_config(self, config: SweepConfig, rung: int) -> None:
        if self.state.scored(config.config_id, rung):
            return
        previous = self.state.at(rung, config.config_id)
        # A preempted cell keeps the id of the job that was given up. Re-attaching
        # to it would find a cancelled job, record no score, and do the same again
        # on every later pass — the configuration would never be measured and
        # nothing would say why. It is submitted afresh instead.
        job_id = None if previous.get("preempted") else previous.get("job_id")
        if job_id:
            # A driver killed mid-rung left a job running; re-attach rather than
            # submit a second one for the same configuration.
            logger.info("rung {}: re-attaching to {} for {}", rung, job_id,
                        config.config_id)
        else:
            wanted_by = self.requested_jobs()
            if wanted_by:
                logger.info("rung {}: not submitting {} — {} has the card",
                            rung, config.config_id, ", ".join(wanted_by[:3]))
                raise Preempted(config.config_id)
            answer = self.client.submit(self.request_for(config, rung))
            job_id = str(answer["job_id"])
            self.state.put(rung, config.config_id, {
                "job_id": job_id, "status": str(answer.get("status", "queued")),
                "score": None, "raw": None, "metric": self.metric.name,
                # What this cell is allowed to cost, recorded at submit so the
                # ceiling counts a preempted cell too: it spent the card whether
                # or not it produced a number.
                "steps": steps_at_rung(self.state.base_steps, rung, self.eta),
            })
            self.state.save()
            logger.info("rung {}: submitted {} as {}", rung, config.config_id, job_id)
        finished = self._await(job_id, config, rung)
        if finished is None:
            raise Preempted(config.config_id)
        self._record(config, rung, finished)

    # ── the sweep ───────────────────────────────────────────────────────────
    def verify_all(self, configs: list[SweepConfig]) -> None:
        """Ask the trainer about every configuration before submitting any.

        `POST /jobs/verify` costs a round trip and catches a bad spec across the
        whole sweep at once, rather than at rung 0 configuration by
        configuration, hours apart.
        """
        complaints: list[str] = []
        for config in configs:
            answer = self.client.verify(self.request_for(config, 0))
            if not answer.get("valid", True):
                complaints.append(f"  {config.config_id}: "
                                  f"{'; '.join(answer.get('errors') or ['refused'])}")
        if complaints:
            raise SweepError("the trainer refuses these configurations:\n"
                             + "\n".join(complaints))

    def over_ceiling(self) -> str | None:
        """Why this sweep must stop, or ``None`` while it may continue.

        A continuously running sweep without an upper bound is not a process, it
        is a leak (#117). Checked before each cell rather than after, because a
        ceiling crossed by the cell that crossed it has not held.
        """
        ceiling = self.manifest.max_total_steps
        if ceiling is None:
            return None
        spent = self.state.spent_steps()
        if spent < ceiling:
            return None
        return (f"{self.manifest.name} has spent {spent:,} optimizer steps of its "
                f"{ceiling:,} ceiling. Raise budget.max_total_steps if this sweep "
                "is worth more, but raise it deliberately.")

    def serve(self, *, reload: Callable[[], SweepManifest] | None = None,
              idle_s: float = IDLE_S, passes: int | None = None) -> int:
        """Keep the sweep going: the standing queue of #117.

        One pass is one attempt at the ladder. A pass that is preempted, or that
        finds the whole manifest already measured, sleeps and tries again — so a
        configuration added to the file days later is picked up, and a card taken
        by a requested run is simply waited out.

        Returns the number of passes made, which is what a test can assert on.
        Stops for two reasons only: the ceiling, or ``passes`` running out.
        """
        made = 0
        while passes is None or made < passes:
            made += 1
            if reload is not None:
                self.manifest = reload()
                self.state.check_against(self.manifest, self.state.metric)
            reason = self.over_ceiling()
            if reason:
                logger.warning("sweep stopped: {}", reason)
                return made
            try:
                self.run()
            except Preempted as gave_way:
                logger.info("pass {} gave the card back at {}; waiting {:.0f}s",
                            made, gave_way, idle_s)
                self.sleep(idle_s)
                continue
            if self.outstanding():
                continue          # more to do, and nobody is in the way
            self.sleep(idle_s)
        return made

    def outstanding(self) -> list[str]:
        """Configurations of rung 0 with no score yet. The queue, in other words."""
        return [c.config_id for c in self.manifest.configs()
                if not self.state.scored(c.config_id, 0)]

    def run(self) -> list[Promotion]:
        configs = {c.config_id: c for c in self.manifest.configs()}
        if self.state.train_lines is None:
            raise SweepError(
                "this sweep does not know its corpus size yet, so it cannot turn "
                "a step budget into epochs. Pass --train-lines for the first run; "
                "afterwards it is read from the jobs themselves.")

        ladder = self.ladder(len(configs))
        self.verify_all(list(configs.values()))

        # Rung 0 is the whole field — the manifest refuses any other width, so
        # no configuration can be left unrun by a ladder that quietly starts short.
        entrants = list(configs)
        promotions: list[Promotion] = []
        for rung, width in enumerate(ladder):
            entrants = entrants[:width]
            logger.info("rung {}: {} configurations, {} steps each", rung,
                        len(entrants), steps_at_rung(self.state.base_steps, rung, self.eta))
            for config_id in entrants:
                reason = self.over_ceiling()
                if reason:
                    logger.warning("sweep stopped mid-rung: {}", reason)
                    return promotions
                self._run_config(configs[config_id], rung)

            if rung + 1 >= len(ladder):
                break
            scores = {cid: self.state.at(rung, cid).get("score") for cid in entrants}
            decision = promote(scores, eta=self.eta, keep=ladder[rung + 1], rung=rung,
                               noise_floor=self.manifest.noise_floor)
            promotions.append(decision)
            self._log_promotion(decision)
            self.state.promotions.append({
                "rung": decision.rung, "promoted": decision.promoted,
                "eliminated": decision.eliminated, "unscored": decision.unscored,
                # The number the rung turned on, and whether the material can
                # resolve it (#115). Without these two in the file the mark lives
                # only in a log line that scrolls past, and the leaderboard (#116)
                # has nothing to print — which is the state the first architecture
                # search published its ranking in.
                "boundary_margin": decision.boundary_margin,
                "decided_within_noise": decision.decided_within_noise,
                "noise_floor": self.manifest.noise_floor,
                "anomalies": [{"config_id": a.config_id, "score": a.score,
                               "lower_fence": a.lower_fence} for a in decision.anomalies],
            })
            self.state.save()
            entrants = list(decision.promoted)
        return promotions

    def _log_promotion(self, decision: Promotion) -> None:
        logger.info("{}", decision)
        if decision.decided_within_noise:
            # Not an error: a rung has to narrow. But the record has to say that
            # repeating the same configuration would have moved it further than
            # the difference this cut rests on.
            logger.warning(
                "rung {}: the cut fell on {:.4f}, inside the measured noise floor "
                "{} — this rung was decided by noise, not by the configurations",
                decision.rung, decision.boundary_margin, self.manifest.noise_floor)
        for flag in decision.anomalies:
            # Not a low score: a training outcome that failed. Logged distinctly
            # so a post-run audit can tell "needs more data" from "collapsed".
            logger.warning(
                "rung {}: {} scored {:.4f}, below the Tukey fence {:.4f} — "
                "treated as a collapse, eliminated and NOT promoted",
                decision.rung, flag.config_id, flag.score, flag.lower_fence)
        for config_id in decision.unscored:
            logger.warning("rung {}: {} produced no {} — not promoted",
                           decision.rung, config_id, self.metric.name)

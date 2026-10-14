#!/usr/bin/env python3
"""Measure a VLM sweep's noise floor through the trainer (#138, #115).

    scripts/measure_vlm_noise_floor.py config/sweeps/vlm-xix-effective-batch-01.yaml \
        --seeds 42 43 44 45 --write-to config/sweeps/vlm-xix-effective-batch-01.yaml

One cell of the manifest — by default the first, which in the XIX sweep is the
baseline's own setting — submitted N times at rung 0's budget, differing only in
``seed``. The spread of the N scores is the resolution of the sweep: no gap
between two cells narrower than it was decided by the hyperparameters.

Why a second script and not ``--engine`` on ``measure_noise_floor.py``: that one
calls ``ketos`` itself on arrows it is handed, because what it measures is the
variance of training and nothing else. A VLM run has no such shortcut — the
adapter is trained by an engine in its own venv, behind the trainer's prepare
and test stages — so this goes through the trainer the way the sweep itself
will. The request is built by the sweep driver's own ``request_for``, so the
floor is measured on exactly the configuration the cells run, at exactly their
budget (``max_steps``), with the seed as the only difference.

What it shares with the kraken measurement is how a result is read
(``noise_floor.is_collapsed``: a spread over empty hypotheses is no resolution)
and how the number reaches the manifest (``noise_floor.write_floor``: with its
provenance, and without erasing the comments beside it).

It is resumable. The job ids are kept in ``<manifest>.noise.state.json`` and a
restart re-attaches to them rather than submitting the seeds again — the runs
cost hours each. The trainer runs as many at once as it has cards.

Run it BEFORE the sweep, not beside it: the driver treats any active job it did
not submit as a request for the card and yields to it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_training.codeversion import current_code  # noqa: E402
from atr_training.noise_floor import is_collapsed, write_floor  # noqa: E402
from atr_training.sweep_driver import (  # noqa: E402
    METRICS, TERMINAL, SweepDriver, SweepState, TrainerClient, _minutes,
)
from atr_training.sweep_manifest import (  # noqa: E402
    ManifestError, SweepConfig, SweepManifest, load_manifest,
)

POLL_S = 60.0


def pick_cell(manifest: SweepManifest, wanted: str | None) -> SweepConfig:
    """The configuration every seed runs: the first cell, or the one named.

    The first because the manifests put the baseline's own setting there — the
    XIX sweep says so in its `axes` comment — and a floor measured on the
    configuration the others are compared against is the floor the comparison
    needs. A prefix of the config id is enough, the way git takes one.
    """
    configs = manifest.configs()
    if wanted is None:
        return configs[0]
    hits = [c for c in configs if c.config_id.startswith(wanted)]
    if len(hits) != 1:
        raise SystemExit(f"--cell {wanted!r} matches {len(hits)} of "
                         f"{[c.config_id for c in configs]}")
    return hits[0]


def requests_for(manifest: SweepManifest, config: SweepConfig, seeds: list[int],
                 metric: str) -> dict[int, dict]:
    """One request per seed, built by the driver's own `request_for`.

    Built there and not here so that the floor cannot drift from the cells: the
    budget, the epoch cover and the datasets come from the same function that
    will submit the sweep. Only the seed, the model id and the note differ.
    """
    state = SweepState.for_manifest(manifest, metric)
    driver = SweepDriver(manifest, state, _NoTrainer())
    if not driver.budgets_in_steps:
        raise SystemExit(
            f"engine {manifest.engine!r} does not take a step budget; this "
            "measurement is for the VLM engine (scripts/measure_noise_floor.py "
            "is the kraken one)")
    base = driver.request_for(config, rung=0)
    # One draw for every seed, or the spread is the model's variance plus the
    # draw's: the first measurement (12.–13.10.2026) scored its four seeds on four
    # different 200-line draws, because the draw was taken with the training
    # seed. The cell's own seed names the draw; only `seed` varies.
    base["params"]["eval_seed"] = int(base["params"].get("seed", 42))
    out: dict[int, dict] = {}
    for seed in seeds:
        request = deepcopy(base)
        request["params"]["seed"] = seed
        request["model_id"] = f"{manifest.name}-noise-s{seed}"
        request["notes"] = (f"noise floor of sweep {manifest.name} (#115): cell "
                            f"{config.config_id} at seed {seed}, budget "
                            f"{base['params']['max_steps']} optimizer steps, data "
                            f"{manifest.data_digest}")
        out[seed] = request
    return out


class _NoTrainer:
    """`request_for` needs a driver and the driver needs a client; this one
    refuses every call, so a request can be built without a trainer."""

    def __getattr__(self, name: str):
        raise RuntimeError(f"no trainer here: {name} must not be called while "
                           "building requests")


class NoiseState:
    """Which seed became which job — kept beside the manifest, like the sweep's
    own state, so an interrupted measurement re-attaches instead of paying twice."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.jobs: dict[int, str] = {}
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.jobs = {int(k): str(v) for k, v in (raw.get("jobs") or {}).items()}
            self.cell = raw.get("cell")
            self.data_digest = raw.get("data_digest")
        else:
            self.cell = None
            self.data_digest = None

    def check_against(self, config: SweepConfig, data_digest: str) -> None:
        if self.jobs and (self.cell, self.data_digest) != (config.config_id, data_digest):
            raise SystemExit(
                f"{self.path} holds jobs for cell {self.cell} on data "
                f"{self.data_digest}, and this run asks for {config.config_id} on "
                f"{data_digest}. A floor belongs to one configuration on one "
                "corpus; move the old state file aside to measure another.")

    def save(self, config: SweepConfig, data_digest: str) -> None:
        self.path.write_text(json.dumps({
            "cell": config.config_id, "data_digest": data_digest,
            "jobs": {str(k): v for k, v in sorted(self.jobs.items())},
        }, indent=2) + "\n", encoding="utf-8")


def submit_missing(client: TrainerClient, requests: Mapping[int, dict],
                   state: NoiseState, config: SweepConfig, data_digest: str,
                   *, dry_run: bool = False) -> None:
    for seed, request in requests.items():
        if seed in state.jobs:
            print(f"seed {seed}: re-attaching to {state.jobs[seed]}", flush=True)
            continue
        answer = client.verify(request)
        if answer.get("valid") is False:
            # `/jobs/verify` answers 200 with the refusal inside; only the HTTP
            # errors raise. The driver reads it the same way (`verify_all`).
            raise SystemExit(f"seed {seed}: the trainer refuses the request: "
                             f"{answer.get('errors')}")
        if dry_run:
            print(f"seed {seed}: verified, would submit {request['model_id']}", flush=True)
            continue
        answer = client.submit(request)
        state.jobs[seed] = str(answer["job_id"])
        state.save(config, data_digest)
        print(f"seed {seed}: submitted {request['model_id']} as {state.jobs[seed]}",
              flush=True)


def await_all(client: TrainerClient, jobs: Mapping[int, str], *,
              poll_s: float = POLL_S,
              sleep: Callable[[float], None] = time.sleep) -> dict[int, Mapping]:
    done: dict[int, Mapping] = {}
    while len(done) < len(jobs):
        for seed, job_id in jobs.items():
            if seed in done:
                continue
            job = client.job(job_id)
            if str(job.get("status")) in TERMINAL:
                done[seed] = job
                print(f"seed {seed}: {job_id} {job.get('status')}", flush=True)
        if len(done) < len(jobs):
            sleep(poll_s)
    return done


def read_runs(jobs: Mapping[int, Mapping], metric: str) -> list[dict]:
    """One row per seed from the job records, nothing retyped."""
    reader = METRICS[metric]
    rows = []
    for seed, job in sorted(jobs.items()):
        result = job.get("metrics") or {}
        score = reader.raw_of(job)
        ratio = result.get("length_ratio")
        rows.append({
            "seed": seed, "job_id": job.get("id"), "status": str(job.get("status")),
            metric: score,
            "length_ratio": float(ratio) if isinstance(ratio, (int, float)) else None,
            "collapsed": is_collapsed(score, ratio if isinstance(ratio, (int, float)) else None),
            "minutes": _minutes(job.get("started_at"), job.get("finished_at")),
            "commit": (job.get("code") or {}).get("commit"),
            "error": job.get("error"),
        })
    return rows


def summarise(manifest: SweepManifest, config: SweepConfig, requests: Mapping[int, dict],
              runs: list[dict], metric: str) -> dict:
    first = next(iter(requests.values()))
    code = current_code()
    summary = {
        "sweep": manifest.name, "engine": manifest.engine,
        "data_digest": manifest.data_digest,
        "commit": code.commit, "dirty": code.dirty,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # The configuration, because the floor is a property of one (#115).
        "cell": config.config_id, "axes": config.axes,
        "base_model": config.base_model,
        "params": {k: v for k, v in first["params"].items() if k != "seed"},
        "eval_seed": first["params"]["eval_seed"],
        "steps": first["params"]["max_steps"],
        "metric": metric,
        "seeds": [r["seed"] for r in runs], "runs": runs,
    }
    scored = [r[metric] for r in runs if r[metric] is not None and not r["collapsed"]]
    if len(scored) >= 2:
        summary["noise_floor"] = {
            "spread": max(scored) - min(scored), "n": len(scored),
            "min": min(scored), "mean": statistics.fmean(scored), "max": max(scored),
        }
    collapsed = [r["seed"] for r in runs if r["collapsed"]]
    if collapsed:
        summary["collapsed_seeds"] = collapsed
    return summary


def manifest_block(summary: dict) -> dict:
    """The shape `sweep_manifest._noise_floor` reads, with what a VLM floor is
    a property of: the cell, its base and its step budget."""
    return {
        "value": summary["noise_floor"]["spread"],
        "measured_on": summary["data_digest"],
        "seeds": summary["seeds"],
        "commit": summary["commit"],
        "steps": summary["steps"],
        "n": summary["noise_floor"]["n"],
        "measured_at": summary["measured_at"],
        "metric": summary["metric"],
        "eval_seed": summary["eval_seed"],
        "cell": summary["cell"],
        "base_model": summary["base_model"],
    }


def report(summary: dict, metric: str) -> None:
    print(f"\nnoise floor of {summary['sweep']}, cell {summary['cell']} "
          f"({summary['axes']}), {summary['steps']} optimizer steps, metric {metric}:")
    for r in summary["runs"]:
        value = "—" if r[metric] is None else f"{r[metric]:.4f}"
        flag = "  COLLAPSED" if r["collapsed"] else ""
        mins = f"{r['minutes']:.0f} min" if r["minutes"] is not None else "—"
        print(f"  seed {r['seed']:<4} {value:>8}  length_ratio "
              f"{r['length_ratio'] if r['length_ratio'] is not None else '—'}  "
              f"{r['status']}  {mins}  {r['job_id']}{flag}")
    floor = summary.get("noise_floor")
    if floor:
        print(f"\nspread {floor['spread']:.4f} over {floor['n']} runs "
              f"(min {floor['min']:.4f}, mean {floor['mean']:.4f}, max {floor['max']:.4f})")
        print("This is the floor: no ranking in this sweep may be read finer than it.")
    else:
        usable = sum(1 for r in summary["runs"] if r[metric] is not None and not r["collapsed"])
        print(f"\nonly {usable} usable run(s) — no floor can be stated")
    if summary.get("collapsed_seeds"):
        print(f"\n{len(summary['collapsed_seeds'])} run(s) COLLAPSED (empty hypothesis): "
              f"seeds {summary['collapsed_seeds']}. A spread over those is not a "
              "resolution, it is the absence of one.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    ap.add_argument("--cell", default=None,
                    help="config id (or a prefix) of the cell to repeat; default: "
                         "the first, which the manifests reserve for the baseline")
    ap.add_argument("--metric", choices=sorted(METRICS), default="cer",
                    help="the number the sweep will rank by (default: %(default)s — "
                         "a VLM sweep's validation partition is drawn across its "
                         "corpora and benchmark_cer cannot be filled)")
    ap.add_argument("--trainer", default="http://127.0.0.1:8204")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--state", type=Path, default=None,
                    help="default: <manifest>.noise.state.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="where noise_floor.json goes (default: ~/atr-cache/noise/<sweep>)")
    ap.add_argument("--poll", type=float, default=POLL_S)
    ap.add_argument("--write-to", type=Path, default=None,
                    help="splice the floor into this manifest (normally the same file)")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify every request against the trainer, submit nothing")
    args = ap.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        print(exc, file=sys.stderr)
        return 1
    if len(set(args.seeds)) < 2:
        ap.error("a floor is a spread; give at least two distinct seeds")
    config = pick_cell(manifest, args.cell)
    requests = requests_for(manifest, config, sorted(set(args.seeds)), args.metric)

    from atr_training.trainer_http import HttpTrainer
    client = HttpTrainer(args.trainer, args.api_key)
    state = NoiseState(args.state or args.manifest.with_suffix(
        args.manifest.suffix + ".noise.state.json"))
    state.check_against(config, manifest.data_digest)
    print(f"cell {config.config_id} {config.axes} of {manifest.name}, "
          f"{requests[args.seeds[0]]['params']['max_steps']} optimizer steps, "
          f"seeds {sorted(set(args.seeds))}", flush=True)
    submit_missing(client, requests, state, config, manifest.data_digest,
                   dry_run=args.dry_run)
    if args.dry_run:
        return 0

    jobs = await_all(client, state.jobs, poll_s=args.poll)
    runs = read_runs(jobs, args.metric)
    summary = summarise(manifest, config, requests, runs, args.metric)
    report(summary, args.metric)

    out = args.out or (Path.home() / "atr-cache" / "noise" / manifest.name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "noise_floor.json").write_text(json.dumps(summary, indent=2) + "\n",
                                          encoding="utf-8")
    print(f"written to {out / 'noise_floor.json'}")

    if args.write_to and "noise_floor" in summary:
        try:
            write_floor(args.write_to, manifest_block(summary))
        except ValueError as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 1
        print(f"and into {args.write_to}")
    return 0 if "noise_floor" in summary else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run a sweep manifest over all its rungs, and survive being interrupted (#114).

    scripts/run_sweep.py config/sweeps/<name>.yaml --train-lines 120000
    scripts/run_sweep.py config/sweeps/<name>.yaml            # resumed

State lives in `<manifest>.state.json`, beside the manifest rather than inside
it: the manifest refuses unknown top-level keys (#113), and keeping the
definition of the experiment apart from the record of it is what lets a fourth
learning rate be added without the first three's results becoming suspect.

A restart re-reads that file, skips every configuration already scored at its
rung, and re-attaches to a job it had submitted but not yet seen finish — so an
interruption costs nothing at all, not even the running job.

`--train-lines` is needed only the first time: the budget is in optimizer steps
and turning it into the epoch count `ketos` wants needs the corpus size. After
the first job completes the number comes from `progress.train_lines`, and the
driver refuses to continue if a later job reports a different one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_training.sweep_driver import (  # noqa: E402
    METRICS, SweepDriver, SweepError, SweepState,
)
from atr_training.sweep_manifest import ManifestError, load_manifest  # noqa: E402


class HttpTrainer:
    """The trainer over HTTP. Three calls, all of them already in its API."""

    def __init__(self, base_url: str, api_key: str | None = None,
                 timeout: float = 30.0) -> None:
        headers = {"X-API-Key": api_key} if api_key else {}
        self._client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers,
                                    timeout=timeout)

    def verify(self, request: dict) -> dict:
        response = self._client.post("/jobs/verify", json=request)
        response.raise_for_status()
        return response.json()

    def submit(self, request: dict) -> dict:
        response = self._client.post("/jobs", json=request)
        response.raise_for_status()
        return response.json()

    def job(self, job_id: str) -> dict:
        response = self._client.get(f"/jobs/{job_id}")
        response.raise_for_status()
        return response.json()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--trainer", default="http://127.0.0.1:8204",
                        help="the training service (default: %(default)s)")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--metric", choices=sorted(METRICS), default="benchmark_cer",
                        help="which number ranks the sweep (default: %(default)s)")
    parser.add_argument("--train-lines", type=int, default=None,
                        help="corpus size, needed for the first rung only")
    parser.add_argument("--state", type=Path, default=None,
                        help="default: <manifest>.state.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="verify every configuration and print the ladder, "
                             "submit nothing")
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        print(exc, file=sys.stderr)
        return 1

    state_path = args.state or args.manifest.with_suffix(args.manifest.suffix + ".state.json")
    client = HttpTrainer(args.trainer, args.api_key)
    try:
        state = SweepState.load(state_path, manifest, args.metric)
        if args.train_lines:
            state.observe_train_lines(args.train_lines)
        driver = SweepDriver(manifest, state, client)
        configs = manifest.configs()

        if args.dry_run:
            driver.verify_all(configs)
            print(f"{manifest.name}: {len(configs)} configurations, "
                  f"ladder {driver.ladder(len(configs))}")
            print(f"data {manifest.data_digest}, metric {args.metric}")
            print("every configuration was accepted by the trainer; nothing submitted")
            return 0

        driver.run()
    except SweepError as exc:
        print(exc, file=sys.stderr)
        return 1

    state.save()
    print(f"\n{manifest.name} finished. State in {state_path}.")
    for entry in state.promotions:
        print(f"  rung {entry['rung']}: promoted {', '.join(entry['promoted'])}")
        for flag in entry["anomalies"]:
            print(f"    anomaly {flag['config_id']} at {flag['score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

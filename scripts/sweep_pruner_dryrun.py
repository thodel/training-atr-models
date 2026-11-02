#!/usr/bin/env python3
"""Replay a median + floor pruner over the finished arms of a sweep (#334).

No GPU: everything is read from what the arms left behind — the Trainer's
``trainer_state.json`` in the last checkpoint (loss every 25 steps), the
progress bar in ``logs/train.log`` (wall time per step), the sweep's state file
(cells, the shared-draw metric, the promotion) and the job records (measured
train-stage energy). On asteraix::

    .venvs/kraken-train/bin/python scripts/sweep_pruner_dryrun.py \\
        --checkpoints ~/atr-cache/checkpoints \\
        --state config/sweeps/vlm-xix-effective-batch-01.yaml.state.json \\
        --records records.json --rung 0

``records.json`` is ``{job_id: record}`` as the results MCP's ``job`` tool
returns it (``energy.per_stage_wh.train`` is what is read); without it every
arm's energy is estimated from wall time and the table says so.

The arms of rung 0 are the ones in the state file; the seeds of the noise-floor
cell are found by the ``-noise-s<seed>`` suffix of the cell's job id and are
the floor. Runs of a cell that were cancelled or superseded (the steps-budget
attempt of 14.10.2026) are not in the state and are not read.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atr_training.pruner_dryrun import Arm, dry_run, markdown_table  # noqa: E402

BAR = re.compile(r"(\d+)/(\d+) \[(\d+):(\d+)(?::(\d+))?<")


def read_losses(run_dir: Path) -> tuple[list[tuple[int, float]], int]:
    """``(step, loss)`` from the last checkpoint's trainer_state, and its step."""
    states = sorted(run_dir.glob("checkpoint-*/trainer_state.json"),
                    key=lambda p: int(p.parent.name.split("-")[1]))
    if not states:
        raise FileNotFoundError(f"no checkpoint-*/trainer_state.json under {run_dir}")
    state = json.loads(states[-1].read_text(encoding="utf-8"))
    losses = [(int(e["step"]), float(e["loss"])) for e in state["log_history"] if "loss" in e]
    return losses, int(state["global_step"])


def read_log(run_dir: Path) -> dict:
    """Effective batch, elapsed seconds per step and the Trainer's train_runtime."""
    text = (run_dir / "logs" / "train.log").read_text(encoding="utf-8", errors="replace")
    max_steps = int(re.search(r"--max-steps (\d+)", text).group(1))
    batch = int(re.search(r"--batch-size (\d+)", text).group(1))
    accumulate = int(re.search(r"--accumulate-grad-batches (\d+)", text).group(1))
    elapsed: dict[int, float] = {}
    for m in BAR.finditer(text):
        if int(m.group(2)) != max_steps:  # the evaluation has its own bar
            continue
        h, mi, s = m.group(3), m.group(4), m.group(5)
        secs = (int(h) * 3600 + int(mi) * 60 + int(s)) if s is not None else int(h) * 60 + int(mi)
        elapsed[int(m.group(1))] = float(secs)
    runtime = re.search(r"'train_runtime': ([0-9.]+)", text)
    eval_runtime = re.search(r"'eval_runtime': ([0-9.]+)", text)
    return {"effective_batch": batch * accumulate, "elapsed": elapsed,
            "train_seconds": float(runtime.group(1)) if runtime else None,
            "eval_seconds": float(eval_runtime.group(1)) if eval_runtime else None}


def arms_of(state: dict, rung: str, checkpoints: Path, records: dict) -> tuple[list[Arm], str | None]:
    arms: list[Arm] = []
    cells = state["results"][rung]
    for cell, result in cells.items():
        job = result["job_id"]
        noise = re.match(r"(.*)-noise-s\d+$", job)
        run_ids = (sorted(p.name for p in checkpoints.glob(f"*{noise.group(1).split('Z-', 1)[1]}-noise-s*")
                          if p.is_dir()) if noise else [job])
        # the noise seeds share the cell's metric only as a mean; each seed's
        # own shared-draw value is in the provenance
        seed_values = (state.get("noise_floor_provenance") or {}).get("values") or []
        for i, run in enumerate(run_ids):
            run_dir = next(iter(checkpoints.glob(f"*{run.split('Z-', 1)[1]}")), None) \
                if not (checkpoints / run).is_dir() else checkpoints / run
            if run_dir is None:
                print(f"skip {run}: no checkpoint directory", file=sys.stderr)
                continue
            losses, _ = read_losses(run_dir)
            log = read_log(run_dir)
            rec = records.get(run_dir.name) or {}
            wh = ((rec.get("energy") or {}).get("per_stage_wh") or {}).get("train")
            metric = (seed_values[i] if noise and i < len(seed_values) else result.get("raw"))
            arms.append(Arm(name=run_dir.name, cell=cell, effective_batch=log["effective_batch"],
                            losses=losses, final_metric=metric, train_wh=wh,
                            wh_estimated=False, train_seconds=log["train_seconds"],
                            elapsed=log["elapsed"], is_floor=bool(noise)))
    promoted = next((p["promoted"] for p in state.get("promotions", []) if str(p["rung"]) == rung), [])
    winner = next((a.name for a in arms if a.cell in promoted), None)
    return arms, winner


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", type=Path, required=True, help="directory holding one run directory per arm")
    ap.add_argument("--state", type=Path, required=True, help="the sweep's .state.json")
    ap.add_argument("--records", type=Path, help="{job_id: record} from the results MCP's job tool")
    ap.add_argument("--rung", default="0")
    ap.add_argument("--budget", type=int, help="samples per arm at this rung (default: from the state)")
    ap.add_argument("--json", action="store_true", help="print the per-grid curves and floors as JSON instead")
    args = ap.parse_args()

    state = json.loads(args.state.read_text(encoding="utf-8"))
    records = json.loads(args.records.read_text(encoding="utf-8")) if args.records else {}
    arms, winner = arms_of(state, args.rung, args.checkpoints, records)
    budget = args.budget or next(iter(state["results"][args.rung].values()))["samples"]
    run = dry_run(arms, budget, winner=winner)
    if args.json:
        from atr_training.pruner_dryrun import on_grid
        print(json.dumps({"budget": budget, "floors": run.floors,
                          "curves": {a.name: on_grid(a, budget) for a in arms}}, indent=1))
        return 0
    print(markdown_table(run))
    evals = [(a.name, read_log(args.checkpoints / a.name)["eval_seconds"]) for a in arms]
    print()
    for name, secs in evals:
        if secs:
            print(f"end-of-run evaluation of `{name}`: {secs / 60:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

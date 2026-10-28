#!/usr/bin/env python3
"""Rewrite ``steps_per_epoch`` / ``total_steps`` of page runs in samples (#264).

On UBELIX, from the checkout, inside the container — ``atr_training`` needs
pydantic, and the login node's ``python3`` has none::

    apptainer exec --bind /storage/research --bind /scratch --bind /rs_scratch \
      --env PYTHONPATH=$HOME/training-atr-models/src:$HOME/training-atr-models/engines \
      $HOME/ubelix/vlm-train.sif python3 scripts/recount_step_budget.py \
      --store /scratch/network/users/th19c587/runs/jobs \
      --store /storage/research/wbkolleg_dh_1/Textrecognition_Training/training_folder/jobs \
      --cache /scratch/network/users/th19c587/expA/artefacts

That is the dry run: what each record says, and what it should say. ``--apply``
writes the rewrites (terminal job records only, sidecar first).

See :mod:`atr_training.step_budget_recount` for what is rewritten, from what
evidence, and what is never touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from atr_training.step_budget_recount import (  # noqa: E402
        apply_to_job,
        recount_job,
    )
except ModuleNotFoundError as exc:          # the login node: no pydantic
    raise SystemExit(
        f"cannot import atr_training ({exc}). Run this inside the container — "
        "see the command at the top of scripts/recount_step_budget.py. Nothing "
        "was read or written.") from None


def default_caches() -> list:
    from atr_results_mcp.probe import Paths
    from atr_training.settings import TrainerSettings

    roots = [Paths().artefacts_root, Path(TrainerSettings().artefact_cache_root)]
    return list(dict.fromkeys(root.expanduser() for root in roots))


def default_stores() -> list:
    from atr_results_mcp.probe import Paths
    from atr_training.settings import TrainerSettings

    roots = [root for _, root in Paths().job_roots] + [Path(TrainerSettings().jobs_root)]
    return list(dict.fromkeys(root.expanduser() for root in roots))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", action="append", type=Path, default=None,
                        help="an artefact-cache root; repeatable. Default: this box's.")
    parser.add_argument("--store", action="append", type=Path, default=None,
                        help="a job store root; repeatable. Default: this box's.")
    parser.add_argument("--apply", action="store_true", help="write the rewrites")
    parser.add_argument("--json", action="store_true", help="machine-readable")
    args = parser.parse_args(argv)

    caches_by_key = {}
    for root in args.cache if args.cache is not None else default_caches():
        if not root.is_dir():
            print(f"cache {root} is not readable from here — its entries are absent",
                  file=sys.stderr)
            continue
        for entry in sorted(p for p in root.iterdir() if (p / "artefact.json").is_file()):
            caches_by_key.setdefault(entry.name[:12], entry)

    jobs = []
    for root in args.store if args.store is not None else default_stores():
        if not root.is_dir():
            print(f"store {root} is not readable from here — its runs are absent, "
                  "not correct", file=sys.stderr)
            continue
        jobs += sorted(p for p in root.iterdir() if (p / "job.json").is_file())
    jobs_by_id = {}
    for job_dir in jobs:
        jobs_by_id.setdefault(job_dir.name, job_dir)

    rows = [(job_dir, recount_job(job_dir, jobs_by_id, caches_by_key)) for job_dir in jobs]

    if args.apply:
        for path, recount in rows:
            if recount.verdict == "correct":
                apply_to_job(path, recount)

    # Line runs and runs without a budget are counted below, not listed.
    shown = [r for _, r in rows if r.verdict != "line"]
    if args.json:
        print(json.dumps([vars(r) | {"agrees": r.agrees} for r in shown], indent=2))
        return 0
    print("| | gran. | samples | batch × epochs | steps/epoch was → now | "
          "total was → now | trainer samples | trainer steps/epoch | agrees | from |")
    print("|---|---|---:|---|---:|---:|---:|---:|---|---|")
    for r in shown:
        print(f"| `{r.ident}` {r.verdict} | {r.granularity or '—'} | {_n(r.samples)} | "
              f"{_n(r.effective_batch)} × {_n(r.epochs)}"
              f"{'' if r.max_steps is None else f' (max {r.max_steps:,})'} | "
              f"{_n(r.steps_before)} → {_n(r.steps_after)} | "
              f"{_n(r.total_before)} → {_n(r.total_after)} | {_n(r.trainer_samples)} | "
              f"{_n(r.trainer_steps_per_epoch)} | {_agree(r.agrees)} | "
              f"{r.source or r.why} |")
    counts = {verdict: sum(r.verdict == verdict for _, r in rows)
              for verdict in ("correct", "already", "skip", "line")}
    print(f"\n{len(rows)} records: {counts['correct']} to rewrite, "
          f"{counts['already']} already right, {counts['skip']} skipped, "
          f"{counts['line']} line runs or without a budget"
          + (" — written." if args.apply else " — dry run, nothing written."))
    return 0


def _n(value) -> str:
    return "—" if value is None else f"{value:,}"


def _agree(value) -> str:
    return "—" if value is None else ("yes" if value else "**no**")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

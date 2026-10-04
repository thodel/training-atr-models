"""Score a finished run's adapter on **another** run's evaluation draw.

Two arms of one ladder share a corpus and a page-level split by symlink, and are
still scored on different samples: `_eval_subset` (vlm_train_svc/runner.py:313)
stratifies the draw by `job.progress.dataset_counts`, which belongs to the job and
not to the corpus. An arm that adopted a cached artefact records those counts
differently, so `plan_eval_subset` picks different rows from the same pool and the
two CERs are not comparable — measured on 2026-10-04, when
`20260926T080902Z-ladder-xix-gemma4-e4b` turned out to hold a `val_eval.jsonl`
whose md5 differed from the one its four siblings share byte for byte.

This puts both numbers on one draw. It changes nothing about the model: the
adapter, the base, the pixel budget and the sequence budget are the ones the run
trained with, taken from its own record, and only `--val-jsonl`, `--data-root` and
`--report` point at the other job.

    python ubelix/rescore_on_shared_draw.py --job <id-to-score> --draw-from <id>

Writes `data/eval_report.<draw-from>.json` in the scored job's directory and
prints the CER beside the one the run recorded. Touches no job record.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from atr_training.jobstore import JobStore
from atr_training.settings import get_settings
from atr_training.vlm_cmd import evaluate_cmd, find_adapter, parse_eval_report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", required=True, help="the run whose adapter is scored")
    p.add_argument("--draw-from", required=True,
                   help="the run whose data/val_eval.jsonl is used as the draw")
    p.add_argument("--adapter", default=None,
                   help="override the adapter directory (default: the job's checkpoints)")
    args = p.parse_args(argv)

    settings = get_settings()
    store = JobStore(settings.jobs_root)
    job = store.load(args.job)
    params = job.request.params

    draw = Path(settings.jobs_root) / args.draw_from / "data" / "val_eval.jsonl"
    if not draw.is_file():
        print(f"no draw at {draw}", file=sys.stderr)
        return 2
    adapter = Path(args.adapter) if args.adapter else find_adapter(
        Path(settings.checkpoint_root) / args.job)
    if adapter is None:
        print(f"no adapter for {args.job}", file=sys.stderr)
        return 2

    report = store.paths(args.job).data / f"eval_report.{args.draw_from}.json"
    cmd = evaluate_cmd(sys.executable, params=params,
                       base_model=job.request.base_model, adapter_dir=adapter,
                       val_jsonl=draw, data_root=draw.parent.parent, report=report)
    print("== scoring", args.job, "on the draw of", args.draw_from, flush=True)
    print("==", " ".join(str(c) for c in cmd), flush=True)
    code = subprocess.run(cmd).returncode
    if code != 0 or not report.is_file():
        print(f"evaluation exited {code} and wrote {report.is_file()}", file=sys.stderr)
        return 1

    fresh = parse_eval_report(report.read_text(encoding="utf-8"))
    # `metrics` sits on the job, not on `progress` — `Progress` has no such field —
    # and it is a `Metrics` model, not a dict. Both mistakes cost nothing here and
    # an AttributeError at the end of a GPU job if they are made.
    own = getattr(job.metrics, "cer", None)
    print("\n== %s" % args.job)
    print("   on its own draw   : CER %s" % own)
    print("   on %s's draw: CER %.4f over %s samples"
          % (args.draw_from, fresh.cer, fresh.samples))
    print("   report: %s" % report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    python ubelix/rescore_on_shared_draw.py --job <id> --draw-file lines.jsonl \
        --data-root <corpus root> --label <name>

The second form scores a draw that is no run's ``val_eval.jsonl`` — the line crops
of another run's page draw, for instance (#74): ``scripts/lines_of_pages.py``
writes it, and ``--data-root`` is where its ``data/…`` paths resolve.

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


def corpus_root_of(store: JobStore, settings, job_id: str) -> Path | None:
    """Where a draw's ``data/…`` image paths resolve.

    The job's own directory when it compiled the corpus — and the cache entry it
    adopted otherwise. A run that adopted an artefact materializes nothing of
    its own (`_adopt_cached`), so ``draw.parent.parent`` holds the draw and not
    one of its images: three of six rescorings on 14.10.2026 died on the first
    ``Image.open`` for exactly that. The record carries the entry's key prefix
    in ``progress.artefact`` ("<12 hex> reused, built by …"); it is resolved
    against the cache root the way the runner's own test stage was.
    """
    data = store.paths(job_id).data
    if (data / "train.jsonl").is_file():
        return data.parent
    note = store.load(job_id).progress.artefact or ""
    prefix = note.split(" ", 1)[0]
    hits = sorted(Path(settings.artefact_cache_root).glob(f"{prefix}*")) if prefix else []
    return hits[0] if len(hits) == 1 and (hits[0] / "data").is_dir() else None


def params_for_draw_file(params, draw: Path):
    """The run's params, with the sample cap lifted to the whole draw file.

    A ``--draw-file`` *is* the selection — every line of a page draw (#74) — so
    the run's own ``eval_samples`` must not cap it. With the cap, the evaluator
    drew 200 of 3,890 lines at random (19709423, 23.10.2026), and the per-page
    comparison had nothing to join for most pages.
    """
    with draw.open(encoding="utf-8") as handle:
        rows = sum(1 for line in handle if line.strip())
    return params.model_copy(update={"eval_samples": max(rows, 1)})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", required=True, help="the run whose adapter is scored")
    p.add_argument("--draw-from", default=None,
                   help="the run whose data/val_eval.jsonl is used as the draw")
    p.add_argument("--draw-file", type=Path, default=None,
                   help="a draw as a file instead (needs --data-root and --label)")
    p.add_argument("--data-root", type=Path, default=None,
                   help="where --draw-file's data/… paths resolve")
    p.add_argument("--label", default=None,
                   help="names the report: data/eval_report.<label>.json")
    p.add_argument("--adapter", default=None,
                   help="override the adapter directory (default: the job's checkpoints)")
    args = p.parse_args(argv)
    if (args.draw_from is None) == (args.draw_file is None):
        print("give exactly one of --draw-from and --draw-file", file=sys.stderr)
        return 2
    if args.draw_file is not None and (args.data_root is None or not args.label):
        print("--draw-file needs --data-root and --label", file=sys.stderr)
        return 2

    settings = get_settings()
    store = JobStore(settings.jobs_root)
    job = store.load(args.job)
    params = job.request.params

    if args.draw_file is not None:
        draw, data_root, label = args.draw_file, args.data_root, args.label
        if draw.is_file():
            params = params_for_draw_file(params, draw)
    else:
        draw = Path(settings.jobs_root) / args.draw_from / "data" / "val_eval.jsonl"
        data_root, label = corpus_root_of(store, settings, args.draw_from), args.draw_from
    if not draw.is_file():
        print(f"no draw at {draw}", file=sys.stderr)
        return 2
    if data_root is None:
        print(f"the draw of {args.draw_from} names images under data/ and neither its "
              "job directory nor a cache entry it adopted holds them", file=sys.stderr)
        return 2
    # The record carries an absolute `checkpoint_dir`; prefer it. Deriving the path
    # from `settings.checkpoint_root` needs ATR_TRAIN_CHECKPOINT_ROOT to be exported
    # too, and when it is not the default is `$HOME/atr-cache/checkpoints`, which
    # holds nothing — job 17300890 died in five seconds reporting "no adapter" for a
    # run whose adapter was on disk the whole time.
    root = (Path(args.adapter) if args.adapter
            else Path(job.checkpoint_dir) if job.checkpoint_dir
            else Path(settings.checkpoint_root) / args.job)
    adapter = root if args.adapter else find_adapter(root)
    if adapter is None:
        print(f"no adapter under {root} for {args.job}", file=sys.stderr)
        return 2

    report = store.paths(args.job).data / f"eval_report.{label}.json"
    cmd = evaluate_cmd(sys.executable, params=params,
                       base_model=job.request.base_model, adapter_dir=adapter,
                       val_jsonl=draw, data_root=data_root, report=report)
    print("== scoring", args.job, "on the draw", label, flush=True)
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
    print("   on the draw %s: CER %.4f over %s samples"
          % (label, fresh.cer, fresh.samples))
    print("   report: %s" % report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

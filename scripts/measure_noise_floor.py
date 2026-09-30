#!/usr/bin/env python3
"""Measure what a sweep's material and budget can resolve (#115).

    scripts/measure_noise_floor.py --train sweep_train.arrow \
        --benchmark german_test.arrow --steps 2000 --seeds 42 43 44 45

The **same** configuration, N times, differing only in ``--seed``. The spread of
those N numbers is the resolution of everything measured on this corpus at this
budget, and no ranking may be read finer than it.

Why this is the first thing a sweep needs, and not a refinement of it: the first
architecture search produced a full ranking over seven heights and four LSTM
widths, then four cells were repeated with a second seed. The matched pair —
0.0148 apart — flipped sign. The spread of one configuration across two seeds was
0.0085, and h256 moved 0.1924, from 0.7515 to 0.5591, which is below h64. The
ranking was not noisy, it was uninformative, and nothing in it said so.

Two things it does not do:

* **It does not go through the trainer.** No prepare, no compile, no job record —
  the corpus is given, and what is being measured is the variance of training
  itself. It calls the same ``train_cmd`` / ``evaluate_cmd`` the pipeline calls,
  from ``atr_training.ketos_cmd``, so the runs are the runs a job would do.
* **It does not average.** min, mean, max and the full spread are reported, and a
  single summary number would hide the shape — a collapse is not a low value
  around a mean, it is a different outcome (``rungs.detect_anomaly`` exists for
  exactly that).

The score is the **benchmark** CER: one held-out set, identical for every seed,
document-disjoint from the corpus. Scoring each run on its own validation split
would add the split's variance to the model's and measure the sum.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_training.contracts import KrakenTrainParams  # noqa: E402
from atr_training.codeversion import current_code  # noqa: E402
from atr_training.convergence import (  # noqa: E402
    epochs_for, floor_for, plan_steps,
)
from atr_training.ketos_cmd import (  # noqa: E402
    evaluate_cmd,
    find_best_weights,
    parse_test_report,
    train_cmd,
)
from atr_training.manifests import binary_manifest  # noqa: E402


def run(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as handle:
        handle.write((" ".join(cmd) + "\n\n").encode())
        handle.flush()
        return subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT).wait()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True, help="compiled training arrow")
    ap.add_argument("--benchmark", type=Path, required=True,
                    help="compiled held-out arrow every seed is scored on")
    ap.add_argument("--train-lines", type=int, required=True,
                    help="transcribed lines in --train, for the step arithmetic")
    ap.add_argument("--steps", type=int, default=None,
                    help="optimizer steps per run; default = the convergence floor")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    ap.add_argument("--ketos", type=Path,
                    default=Path.home() / "Repo/training-atr-models/.venvs/kraken-train/bin/ketos")
    ap.add_argument("--out", type=Path, default=Path.home() / "atr-cache" / "noise")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base-model", default=None,
                    help="fine-tune from these weights instead of from scratch")
    ap.add_argument("--data-digest", required=True,
                    help="the sweep's data version (#112) — a floor belongs to the "
                         "corpus it was measured on and to nothing else")
    ap.add_argument("--write-to", type=Path, default=None,
                    help="write the measured floor into this sweep manifest, "
                         "instead of retyping it")
    args = ap.parse_args(argv)

    from_scratch = args.base_model is None
    steps = args.steps or floor_for("kraken", from_scratch)
    params = KrakenTrainParams(device=args.device, workers=args.workers, quit="fixed")
    effective = params.effective_batch_size
    epochs = epochs_for(steps, args.train_lines, effective)
    actual = plan_steps(args.train_lines, effective, epochs).total_steps

    args.out.mkdir(parents=True, exist_ok=True)
    train_lst = binary_manifest(args.out / "train_bin.lst", args.train)
    bench_lst = binary_manifest(args.out / "bench_bin.lst", args.benchmark)

    print(f"{len(args.seeds)} runs of one configuration, seeds {args.seeds}")
    print(f"  {args.train_lines} lines, effective batch {effective}, "
          f"{epochs} epoch(s) = {actual} steps (asked for {steps}, "
          f"floor {floor_for('kraken', from_scratch)})")
    print(f"  scored on {args.benchmark.name}\n", flush=True)

    results: list[dict] = []
    for seed in args.seeds:
        seeded = params.model_copy(update={"seed": seed, "epochs": epochs})
        work = args.out / f"seed{seed}"
        started = time.time()
        code = run(train_cmd(args.ketos, params=seeded, training_manifest=train_lst,
                            evaluation_manifest=bench_lst, checkpoint_dir=work,
                            load=args.base_model),
                   work / "train.log")
        weights = find_best_weights(work, params.weights_format)
        entry: dict = {"seed": seed, "train_exit": code,
                       "minutes": round((time.time() - started) / 60, 1),
                       "weights": str(weights) if weights else None}
        if code == 0 and weights is not None:
            run(evaluate_cmd(args.ketos, model=weights, manifest=bench_lst,
                             device=args.device, workers=args.workers,
                             normalization=params.normalization),
                work / "test.log")
            report = parse_test_report(
                (work / "test.log").read_text(encoding="utf-8", errors="replace"))
            entry.update(cer=report.cer, chars=report.chars, errors=report.errors)
        results.append(entry)
        print(f"  seed {seed}: cer={entry.get('cer')} "
              f"({entry['minutes']} min, exit {code})", flush=True)

    scored = [r["cer"] for r in results if r.get("cer") is not None]
    code = current_code()
    summary = {
        "train": str(args.train), "benchmark": str(args.benchmark),
        # What #115 asks to be noted beside the number: the seeds, the commit and
        # the data version. Without the last two a floor is a number nobody can
        # attribute, and a floor from other material licenses a ranking it never
        # earned — which is the failure the whole measurement exists to prevent.
        "data_digest": args.data_digest,
        "commit": code.commit, "dirty": code.dirty,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_bytes": args.train.stat().st_size if args.train.exists() else None,
        "benchmark_bytes": (args.benchmark.stat().st_size
                            if args.benchmark.exists() else None),
        "train_lines": args.train_lines, "effective_batch": effective,
        "epochs": epochs, "steps": actual, "from_scratch": from_scratch,
        "base_model": args.base_model, "seeds": args.seeds, "runs": results,
    }
    if len(scored) >= 2:
        summary["noise_floor"] = {
            "min": min(scored), "max": max(scored),
            "mean": statistics.fmean(scored), "spread": max(scored) - min(scored),
            "stdev": statistics.stdev(scored), "n": len(scored),
        }
        print(f"\nspread {max(scored) - min(scored):.4f} over {len(scored)} runs "
              f"(min {min(scored):.4f}, mean {statistics.fmean(scored):.4f}, "
              f"max {max(scored):.4f})")
        print("This is the floor: no ranking on this corpus at this budget may be "
              "read finer than it.")
    else:
        print(f"\nonly {len(scored)} run(s) produced a CER — no floor can be stated")
    (args.out / "noise_floor.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"written to {args.out / 'noise_floor.json'}")

    if args.write_to and "noise_floor" in summary:
        try:
            write_into_manifest(args.write_to, summary)
        except ValueError as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 1
        print(f"and into {args.write_to}")
    return 0 if len(scored) >= 2 else 1


def write_into_manifest(path: Path, summary: dict) -> None:
    """Put the measured floor into a sweep manifest, with what it was measured on.

    Retyping the number is how it ends up stale or attached to the wrong corpus,
    so the manifest's own data version is checked against the measurement's
    first. The written block is the shape `sweep_manifest` validates: a bare
    number would parse, but nobody reading it later could say where it came from.
    """
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    digest = str(((raw.get("data") or {}).get("digest") or "")).strip()
    if digest != summary["data_digest"]:
        raise ValueError(
            f"{path} runs on data {digest!r} and this floor was measured on "
            f"{summary['data_digest']!r}. A floor belongs to the corpus and the "
            "budget it was measured at; writing it here would license a ranking "
            "it never earned.")
    raw["noise_floor"] = {
        "value": summary["noise_floor"]["spread"],
        "measured_on": summary["data_digest"],
        "seeds": summary["seeds"],
        "commit": summary["commit"],
        "steps": summary["steps"],
        "n": summary["noise_floor"]["n"],
        "measured_at": summary["measured_at"],
    }
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

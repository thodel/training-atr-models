#!/usr/bin/env python3
"""Peak VRAM of a short `ketos train`, for the row kraken never had (#163).

The first branch of `docs/WHERE_A_RUN_RUNS.md` asks whether a run fits on one
A40. For `vllm` that question has an answer; for `kraken` the doc said "no such
figure at all", and for a good reason: kraken drives `ketos` as an external CLI,
imports no torch, and its log carries no numbers because ketos renders through
`rich`. There is no allocator to ask.

So this watches from outside. `gpu.PeakSampler` polls nvidia-smi and attributes a
compute app to this run when the pid we spawned is in its ancestor chain — which
holds across the `sh`/`ketos` generations, measured on asteraix 07.10.2026.

**Time-bounded, and the figure is a floor.** `--epochs` does not bound a ketos run
under `--quit early` (see `ketos_cmd.train_cmd`), and a full epoch over a 30 GB
arrow is not what a peak measurement needs: the per-step high-water mark is
reached in the first steps at full batch. So the run is stopped after
`--seconds`. What that cannot see is the length distribution — ketos reads the
arrow in the order compile wrote it, and this script does not reorder it the way
`measure_vlm_arms.py --worst-case` can. The output says so in `selection`, and
#163 allows a lower bound as long as it is labelled one.

    .venvs/kraken-train/bin/python scripts/measure_kraken_arms.py \\
        --ketos .venvs/kraken-train/bin/ketos \\
        --train-arrow ~/atr-cache/arrows/german_val.arrow \\
        --out kraken_peak.json --seconds 240

Run it with a venv that has this repo importable; only `ketos` itself needs the
kraken stack. The argv comes from `ketos_cmd.train_cmd`, so the command measured
is the command the pipeline would run.

**What it found on asteraix, 07.10.2026** — the first peak kraken has ever had.
`german_val.arrow` (2.9 GB) as training data, `german_test.arrow` as evaluation,
`KRAKEN_PLUS_SPEC`, batch 256 (the project default), one full epoch in ~215 s:

    own 43,536 MiB   card 43,579 MiB   of 45,486 MiB usable on an A40

That is **95.7 % of the card, with 1,950 MiB of headroom** — and it is a peak over
every line of that arrow, not a head. Four independent runs returned 43,536 MiB
to the megabyte, so the measurement is deterministic.

Two things that number settles. The first branch of `docs/WHERE_A_RUN_RUNS.md`
can stop guessing for kraken: at the default batch a run fits one A40 and very
nearly does not, which means the limit for this engine is the batch geometry and
not the model size — a kraken model is ~16 MB. And a second process on the card
is then fatal: while a colleague's job held 740 MiB of gpu0, `card_mib` rose to
44,319 and `own_mib` stayed at 43,536, which is exactly why the two are reported
apart.

Still open: the same point at batch 128, 64, 32. The first attempt measured four
times at 256 because the sweep forgot to pass `--batch-size`, and the identical
figures read like batch-invariance when they were the same configuration.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from atr_training.contracts import KrakenTrainParams  # noqa: E402
from atr_training.gpu import PeakSampler  # noqa: E402
from atr_training.ketos_cmd import train_cmd  # noqa: E402
from atr_training.manifests import binary_manifest  # noqa: E402


def stop(proc: subprocess.Popen) -> None:
    """End the whole session, not just the process we can see.

    `start_new_session=True` makes our child a session leader, and ketos forks
    dataloader workers under it. Signalling the child alone leaves those
    reparented to init, still holding the cards — which is how a "stopped" run
    keeps a GPU busy.
    """
    for sig, grace in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
        if proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def measure(args) -> dict:
    params = KrakenTrainParams(batch_size=args.batch_size, epochs=1, quit="fixed")
    with tempfile.TemporaryDirectory(prefix="kraken-peak-") as tmp:
        # `ketos train -t` takes a MANIFEST of binary datasets, one .arrow path per
        # line — not an .arrow. Handing it the arrow earns
        # "File … is not a text file", which is how the first run of this script
        # ended after 2.8 s. Built with the pipeline's own helper so the file is
        # the file kraken would have read.
        train_lst = binary_manifest(Path(tmp) / "train_bin.lst", args.train_arrow)
        val_lst = (binary_manifest(Path(tmp) / "val_bin.lst", args.val_arrow)
                   if args.val_arrow else None)
        cmd = train_cmd(args.ketos, params=params,
                        training_manifest=train_lst,
                        evaluation_manifest=val_lst,
                        checkpoint_dir=Path(tmp) / "ckpt")
        log = Path(tmp) / "train.log"
        print("$ " + " ".join(cmd), flush=True)
        started = time.perf_counter()
        with log.open("wb") as handle:
            proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            sampler = PeakSampler(proc.pid, interval_s=args.interval).start()
            try:
                proc.wait(timeout=args.seconds)
                ended = "finished on its own"
            except subprocess.TimeoutExpired:
                stop(proc)
                ended = f"stopped after {args.seconds} s"
            finally:
                peak = sampler.stop()
        wall = time.perf_counter() - started
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]

    print(peak.summary(), flush=True)
    return {
        "engine": "kraken",
        "cmd": cmd,
        "batch_size": params.batch_size,
        "spec": params.spec,
        "train_arrow": str(args.train_arrow),
        "wall_seconds": round(wall, 1),
        "outcome": ended,
        "exit_code": proc.returncode,
        # Where the samples came from, and how much of them the peak saw. A run
        # that finished has been over every line of the arrow, which is a peak and
        # not a floor; one that was stopped has seen the arrow's head in compile
        # order, which is. #163 allows a lower bound, but not an unlabelled one.
        "selection": ("full epoch over the arrow" if proc.returncode == 0
                      else f"arrow head, stopped after {args.seconds} s"),
        "peak_own_mib": {f"gpu{i}": v for i, v in sorted(peak.own_mib.items())},
        "peak_card_mib": {f"gpu{i}": v for i, v in sorted(peak.card_mib.items())},
        "own_is_unknown": peak.own_is_unknown,
        "readings": peak.readings,
        "reading_failures": peak.failures,
        "summary": peak.summary(),
        "last_lines": tail,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ketos", required=True, type=Path)
    ap.add_argument("--train-arrow", required=True, type=Path)
    ap.add_argument("--val-arrow", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=KrakenTrainParams().batch_size,
                    help="the kraken default is what the table wants first")
    ap.add_argument("--seconds", type=float, default=240.0,
                    help="wall-clock bound; the per-step peak arrives long before")
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args(argv)

    result = measure(args)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

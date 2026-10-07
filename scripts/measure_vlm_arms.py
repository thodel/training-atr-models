#!/usr/bin/env python3
"""Wall time and peak VRAM of a short VLM run, 4-bit against bf16, one card and two (#137).

The expensive half of #137: the footprint script answers what the weights cost,
this answers what a *step* costs. Same data, same step budget, one variable per
arm — and the peak sampled from nvidia-smi, because that is the number that
decides whether a long run dies, not what torch admits to allocating.

**What it found on asteraix, 03.10.2026**: 96 page samples from the medieval
corpus, batch 1 x accumulate 16, one epoch = 6 optimizer steps, page budget
2,097,152 px (~2048 visual tokens), `Qwen/Qwen3-VL-8B-Instruct`:

    arm                 train_runtime   samples/s   peak gpu0   peak gpu1
    4-bit, one card         301.5 s       0.318     21,675 MiB      --
    bf16,  one card         247.6 s       0.388     27,763 MiB      --
    bf16,  both cards       249.7 s       0.384     13,761 MiB  18,625 MiB

Two results worth having. **bf16 on one card is 22 % faster than 4-bit** — nf4
dequantises every weight on every pass, and an A40 runs bf16 matmuls natively;
the quantisation was buying memory at the cost of time all along. And **the
second card buys no speed** (249.7 against 247.6 s is noise): `device_map="auto"`
is naive model parallelism, so one card waits while the other computes. It
spreads the peak, which is what a model too large for one card needs, and nothing
else.

Peak is a **lower bound** for a full corpus: 96 of 9,441 samples is a 1 % draw,
and on this project OOMs have twice been the distribution's tail rather than the
batch size. 17.3 GiB were still free in the bf16 arm, which is headroom, not
proof.

`--worst-case` removes exactly that caveat (#163). It replaces the head of the
file with the most expensive samples — the longest transcriptions and the largest
crops — so the peak comes from the tail that decides whether a long run dies.
Every row of the report now carries `train_selection`, naming `head` or
`worst-case`, because the two numbers answer different questions and only one of
them may be read as a ceiling. The default stays `head` so the figures above
remain reproducible.

    .venvs/kraken-train/bin/python scripts/measure_vlm_arms.py \\
        --root <job dir> --out report.json

Run it with the REPO venv: it only builds the argv (`vlm_cmd.train_cmd`, so the
command is the one the pipeline would run) and spawns the engine's venv for the
work.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from atr_training.contracts import VLM_BASE_MODEL, VlmTrainParams  # noqa: E402
from atr_training.vlm_cmd import train_cmd  # noqa: E402
from atr_training.vlm_dataset import read_jsonl, worst_case_samples  # noqa: E402

STEP_RE = re.compile(r"'(?:train_runtime|loss)':")


def sample_cards(stop: threading.Event, every: float, into: list[dict[int, int]]) -> None:
    """Poll nvidia-smi until told to stop. MiB in use per physical card."""
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=True).stdout
            into.append({int(i): int(m) for i, m in
                         (line.split(", ") for line in out.strip().splitlines())})
        except Exception:                                   # noqa: BLE001
            pass
        stop.wait(every)


def take_lines(src: Path, dest: Path, count: int, worst_case: bool) -> dict:
    """Write ``count`` samples from ``src`` to ``dest``, and say which ones.

    Two selections, and the difference is the whole point of #163. The head of
    the file is what #137 measured — cheap, reproducible, and a **lower bound**,
    which it said so itself. The worst case is the tail that actually decides
    whether a long run dies: the fp32 logit tensor scales with the transcription,
    the image tokens with the crop, and on this project an OOM was twice the edge
    of the distribution rather than the batch size.

    The returned mapping travels into the report so no row can be read as a
    ceiling when it is a floor.
    """
    if not worst_case:
        with src.open(encoding="utf-8") as fh:
            lines = [next(fh) for _ in range(count)]
        dest.write_text("".join(lines), encoding="utf-8")
        return {"selection": "head", "written": len(lines)}

    picked = worst_case_samples(read_jsonl(src), count)
    dest.write_text(
        "".join(sample.to_json() + "\n" for sample in picked.samples), encoding="utf-8")
    return {"selection": "worst-case", "written": len(picked.samples),
            "considered": picked.considered, "longest_chars": picked.longest_chars,
            "widest_pixels": picked.widest_pixels, "page_samples": picked.page_samples}


def run_arm(name: str, *, cards: str, four_bit: bool, root: Path, out_root: Path,
            samples: int, val_samples: int, python: Path, epochs: int,
            worst_case: bool = False) -> dict:
    work = out_root / name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    train_jsonl = work / "train.jsonl"
    val_jsonl = work / "val.jsonl"
    # `data_root` is the JOB directory, not its `data/`: the `image` paths in the
    # JSONL are already relative to the job ("data/pages/…"), which is what keeps
    # a compiled sample set portable. Passing `data/` doubles the segment, and the
    # first run of this script died on `…/data/data/pages/…` after loading the
    # model three times.
    # The training set carries the selection under test. The validation set stays
    # on the head: its loop is not what #163 is measuring, and changing both at
    # once would leave the peak unattributable to either.
    selection = take_lines(root / "data" / "train.jsonl", train_jsonl, samples, worst_case)
    take_lines(root / "data" / "val.jsonl", val_jsonl, val_samples, worst_case=False)

    params = VlmTrainParams(granularity="page", load_in_4bit=four_bit, epochs=epochs,
                            max_epochs=epochs, save_steps=10_000)
    cmd = train_cmd(python, params=params, base_model=VLM_BASE_MODEL,
                    train_jsonl=train_jsonl, val_jsonl=val_jsonl,
                    data_root=root, output_dir=work / "adapter")

    # The same two the service sets for a real runner, plus the import path the
    # unit provides through PYTHONPATH and its WorkingDirectory: `vlm_train_svc`
    # is a top-level package under engines/, not a module of this one.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": cards,
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
           "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'engines'}"}
    readings: list[dict[int, int]] = []
    stop = threading.Event()
    sampler = threading.Thread(target=sample_cards, args=(stop, 2.0, readings), daemon=True)

    log = (work / "train.log").open("w", encoding="utf-8")
    t0 = time.perf_counter()
    sampler.start()
    proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
    stop.set()
    sampler.join(timeout=15)
    log.close()
    wall = time.perf_counter() - t0

    peak = {}
    for index in sorted({i for r in readings for i in r}):
        peak[f"gpu{index}"] = max(r.get(index, 0) for r in readings)

    text = (work / "train.log").read_text(encoding="utf-8", errors="replace")
    # The trainer's own runtime, which excludes the model load — and the load is
    # most of the wall clock here: the HF cache is a symlink onto CIFS and the
    # four shards take ~42 s each, the same on every arm.
    runtime = re.search(r"'train_runtime':\s*([0-9.]+)", text)
    per_second = re.search(r"'train_samples_per_second':\s*([0-9.]+)", text)
    return {
        "arm": name,
        "cards": cards,
        "four_bit": four_bit,
        "exit_code": proc.returncode,
        "wall_seconds": round(wall, 1),
        "train_runtime_s": float(runtime.group(1)) if runtime else None,
        "train_samples_per_second": float(per_second.group(1)) if per_second else None,
        "samples": samples,
        "effective_batch": params.batch_size * params.accumulate_grad_batches,
        "peak_mib": peak,
        # Where the samples came from. #163: a peak from the head of the file is a
        # lower bound and must not be read as a ceiling, so every row says which
        # it is rather than leaving the reader to remember.
        "train_selection": selection,
        "readings": len(readings),
        "placement": next((ln for ln in text.splitlines() if "[placement]" in ln), None),
        "last_lines": text.strip().splitlines()[-6:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="job (or artefact) directory holding data/train.jsonl")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--work", type=Path, default=Path.home() / "atr-cache" / "137" / "arms")
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--val-samples", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    # NOT resolved: a venv's `bin/python` is a symlink to the system interpreter,
    # and following it leaves the venv behind — which is how the first run of this
    # script ended up in /usr/bin/python3.12 without the stack.
    ap.add_argument("--python", type=Path,
                    default=REPO / ".venvs" / "vlm-train" / "bin" / "python")
    ap.add_argument("--only", default="", help="comma-separated arm names")
    ap.add_argument("--worst-case", action="store_true",
                    help="train on the most expensive samples (longest text, largest "
                         "crops) instead of the head of the file — the peak #163 asks "
                         "for. Off by default so the 03.10.2026 figures above stay "
                         "reproducible.")
    args = ap.parse_args()

    plan = [("fourbit_one_card", "0", True),
            ("bf16_one_card", "0", False),
            ("bf16_two_cards", "0,1", False)]
    wanted = {a for a in args.only.split(",") if a}
    results = []
    for name, cards, four_bit in plan:
        if wanted and name not in wanted:
            continue
        print(f"== {name} ==", flush=True)
        result = run_arm(name, cards=cards, four_bit=four_bit, root=args.root,
                         out_root=args.work, samples=args.samples,
                         val_samples=args.val_samples, python=args.python,
                         epochs=args.epochs, worst_case=args.worst_case)
        print(json.dumps(result, indent=2), flush=True)
        results.append(result)
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("== ALL ARMS DONE ==", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

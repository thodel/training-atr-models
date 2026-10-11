#!/usr/bin/env python3
"""How close a corpus runs to `max_seq_len` — without training it (#138).

`max_seq_len` truncates nothing: the collator passes an over-budget sample
through whole, because truncating a multimodal sequence severs the image tokens
from the placeholders indexing them and yields an invalid sample rather than a
short one (#86). It reaches exactly one place, the collator's threshold, and the
numbers it produces say how much more memory some samples cost than the budget
planned for.

Which left #138's second question — raise the budget, or write down why it stays
— answerable only from a finished run. And until 11.10.2026 not even then: the
collator ran in the DataLoader's worker processes and the parent recorded its own
zeros, so seven runs reported `longest = 0` (`SequenceTally`, #138).

This measures the same thing in minutes instead of hours, because the answer does
not need a trained model: tokenising is the measurement. It loads the
**processor** only — no weights, no GPU — runs the real `HTRCollator` over a draw
of the compiled samples, and reports the distribution rather than the maximum. A
maximum says whether the budget was crossed; percentiles say whether it is the
right budget.

    .venvs/vlm-train/bin/python scripts/measure_sequence_lengths.py \\
        --jsonl <job>/data/train.jsonl --data-root <job> \\
        --granularity page --samples 200 --out lengths.json

The draw is **half worst-case, half random** (`worst_case_samples`, #163): the
tail is where a budget is crossed, and a random draw of 200 from 400,000 will
miss it, which is the limitation #137's peak measurement named about itself.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "engines"))

from atr_training.contracts import VlmTrainParams  # noqa: E402
from atr_training.vlm_dataset import read_jsonl, worst_case_samples  # noqa: E402


def draw(jsonl: Path, count: int, seed: int) -> tuple[list, int, int, str]:
    """``(samples, tail_count, considered, how)`` — half the costly tail, half random.

    The two halves must stay distinguishable in the report. A draw that
    deliberately over-samples the tail cannot state a corpus rate: "48 of 200 are
    over budget" would be read as 24 % of the corpus when the tail half was
    chosen *for* being expensive. The random half is the unbiased estimate; the
    tail half says how far the worst case goes. Reporting one number for both is
    the mistake #137 named about its own peak and this one would repeat.
    """
    pool = list(read_jsonl(jsonl))
    if not pool:
        raise SystemExit(f"{jsonl} has no samples")
    if count >= len(pool):
        return pool, 0, len(pool), "the whole set (no draw)"
    tail = worst_case_samples(pool, max(1, count // 2))
    taken = {id(s) for s in tail.samples}
    rest = [s for s in pool if id(s) not in taken]
    random.Random(seed).shuffle(rest)
    picked = [*tail.samples, *rest[: count - len(tail.samples)]]
    return (picked, len(tail.samples), len(pool),
            f"{len(tail.samples)} worst-case + {count - len(tail.samples)} random")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--granularity", default="line",
                    choices=["line", "block", "page", "mixed"])
    ap.add_argument("--base-model", default=None, help="default: the contract's")
    ap.add_argument("--max-seq-len", type=int, default=None,
                    help="default: the granularity's budget")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    params = VlmTrainParams(granularity=args.granularity)
    budget = args.max_seq_len or params.sequence_budget()
    base = args.base_model or "Qwen/Qwen3-VL-8B-Instruct"

    from transformers import AutoProcessor
    from vlm_train_svc.train_qlora import HTRCollator

    processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
    # The same budget the training path applies, per sample kind (#59): a line
    # crop must not be measured at a page's pixel budget.
    collator = HTRCollator(processor, params.prompt, budget, params.kind_budgets())

    picked, tail_count, considered, how = draw(args.jsonl, args.samples, args.seed)
    print(f"{len(picked)} of {considered} samples ({how}), budget {budget}", flush=True)

    per_sample = _lengths(collator, picked, args.data_root)
    # The tail half comes first, by construction of `draw`.
    tail, rnd = per_sample[:tail_count], per_sample[tail_count:]
    report = {
        "jsonl": str(args.jsonl), "granularity": args.granularity,
        "base_model": base, "sequence_budget": budget,
        "samples": len(per_sample), "considered": considered, "draw": how,
        "longest": max(per_sample),
        # How far past the budget the worst case goes, as a share of the budget.
        "longest_over_budget_pct": round(100 * (max(per_sample) / budget - 1), 1),
        # The honest corpus estimate: the RANDOM half only. The tail half was
        # chosen for being expensive and cannot speak for the corpus.
        "random_half": _half(rnd, budget),
        "worst_case_half": _half(tail, budget),
    }
    text = json.dumps(report, indent=2)
    print(text, flush=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


def _lengths(collator, picked, data_root: Path) -> list[int]:
    """Per-sample token counts.

    One sample per call, with a fresh tally each time: the batch dimension pads
    to the longest member, so a batch would report what a batch costs and the
    question is what each sample costs. The tally keeps a running maximum, which
    is right for a run and useless for a distribution — hence the reset.
    """
    from atr_training.vlm_dataset import SequenceTally

    out: list[int] = []
    for n, sample in enumerate(picked, 1):
        collator.tally = SequenceTally()
        collator([{"image": str(data_root / sample.image), "text": sample.text,
                   "source_type": sample.source_type}])
        out.append(collator.tally.longest)
        if n % 50 == 0:
            print(f"  {n}/{len(picked)} …", flush=True)
    return out


def _half(values: list[int], budget: int) -> dict:
    """What one half of the draw says on its own."""
    if not values:
        return {}
    return {"samples": len(values),
            "median": int(statistics.median(values)),
            "p90": _pct(values, 90), "longest": max(values),
            "over_budget": sum(1 for n in values if n > budget),
            "over_budget_share": round(sum(1 for n in values if n > budget) / len(values), 3)}


def _pct(values: list[int], q: int) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q / 100 * len(ordered)) - 1))
    return ordered[index]


if __name__ == "__main__":
    raise SystemExit(main())

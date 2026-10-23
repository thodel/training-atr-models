#!/usr/bin/env python3
"""A page arm against line arms, on the same pages and the same reference (#74).

    python scripts/compare_page_and_lines.py \\
        --pages <page draw>/val_eval.jsonl --lines lines.jsonl \\
        --page-arm qwen3.5-4b-page=<page job>/data/eval_report.<draw>.raw.jsonl \\
        --line-arm 4bit=<line job>/data/eval_report.<label>.raw.jsonl \\
        --line-arm bf16=<line job>/data/eval_report.<label>.raw.jsonl \\
        --json out.json

Every arm is scored per **page**, against the page draw's own reference text, with
``textmetrics.score_pairs`` — the function the evaluator itself uses. A page arm's
output is its page. A line arm's output for a page is its outputs for that page's
lines, in the line corpus's order, joined with newlines, as the page reference
joins them.

The line arm reads ground-truth line crops: no segmentation error reaches it, so
its figure is an **upper bound** for a line pipeline. ``coverage`` says how much
of each page's reference text the line corpus holds at all; a page the line
corpus covers only in part costs the line arm characters it was never shown.

The comparison is **paired**: per page, line CER minus page CER, with a seeded
bootstrap interval over pages. Two means over the same pages are not the claim;
the per-page difference is.
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

from atr_training.textmetrics import score_pairs  # noqa: E402

BOOTSTRAP = 2000
SEED = 42


def stem(path: str) -> str:
    return Path(path).stem


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def page_outputs_of_lines(line_rows: list[dict], raw_rows: list[dict]) -> dict[str, str]:
    """Each page's text as a line arm read it: its lines' outputs, in order."""
    raw = {r["image"]: (r.get("raw") or "").strip() for r in raw_rows}
    missing = [r["image"] for r in line_rows if r["image"] not in raw]
    if missing:
        raise ValueError(f"{len(missing)} line(s) of the draw have no output, e.g. "
                         f"{missing[:2]} — a raw report of another draw?")
    pages: dict[str, list[str]] = {}
    for row in line_rows:
        pages.setdefault(stem(row["page"]), []).append(raw[row["image"]])
    return {page: "\n".join(lines) for page, lines in pages.items()}


def coverage(line_rows: list[dict], page_rows: list[dict]) -> dict[str, float]:
    """Per page: characters of the line references over those of the page reference."""
    lines: dict[str, int] = {}
    for row in line_rows:
        key = stem(row["page"])
        lines[key] = lines.get(key, 0) + len((row.get("text") or "").replace("\n", ""))
    out = {}
    for row in page_rows:
        ref = len((row.get("text") or "").replace("\n", ""))
        out[stem(row["image"])] = (lines.get(stem(row["image"]), 0) / ref) if ref else 0.0
    return out


def per_page(outputs: dict[str, str], page_rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in page_rows:
        key = stem(row["image"])
        if key not in outputs:
            raise ValueError(f"no output for page {key}")
        ref, hyp = row["text"], outputs[key]
        score = score_pairs([(hyp, ref)])
        result[key] = {"cer": score.cer, "length_ratio": len(hyp) / max(len(ref), 1)}
    return result


def summarise(outputs: dict[str, str], page_rows: list[dict]) -> dict:
    pages = per_page(outputs, page_rows)
    pooled = score_pairs([(outputs[stem(r["image"])], r["text"]) for r in page_rows])
    cers = [p["cer"] for p in pages.values()]
    ratios = sorted(p["length_ratio"] for p in pages.values())
    return {"pages": len(pages), "cer_pooled": pooled.cer,
            "cer_page_mean": statistics.fmean(cers), "cer_page_median": statistics.median(cers),
            "length_ratio_median": statistics.median(ratios),
            "pages_over_1_5x": sum(r > 1.5 for r in ratios),
            "pages_under_third": sum(r < 1 / 3 for r in ratios),
            "per_page": pages}


def paired(a: dict[str, dict], b: dict[str, dict], *, seed: int = SEED,
           rounds: int = BOOTSTRAP) -> dict:
    """Per page ``a − b`` in CER, its mean, and a percentile bootstrap interval."""
    keys = sorted(a)
    diffs = [a[k]["cer"] - b[k]["cer"] for k in keys]
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(rounds))
    lo, hi = means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]
    return {"mean_diff": statistics.fmean(diffs), "ci95": [lo, hi],
            "a_better": sum(d < 0 for d in diffs), "b_better": sum(d > 0 for d in diffs),
            "ties": sum(d == 0 for d in diffs)}


def _arm(spec: str) -> tuple[str, Path]:
    name, _, path = spec.partition("=")
    if not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {spec!r}")
    return name, Path(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pages", type=Path, required=True)
    p.add_argument("--lines", type=Path, required=True)
    p.add_argument("--page-arm", type=_arm, required=True)
    p.add_argument("--line-arm", type=_arm, action="append", required=True)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    page_rows, line_rows = _read(args.pages), _read(args.lines)
    page_name, page_raw = args.page_arm
    page_outputs = {stem(r["image"]): (r.get("raw") or "").strip() for r in _read(page_raw)}
    arms = {page_name: summarise(page_outputs, page_rows)}
    for name, path in args.line_arm:
        arms[name] = summarise(page_outputs_of_lines(line_rows, _read(path)), page_rows)
    cover = coverage(line_rows, page_rows)
    comparisons = {f"{name} − {page_name}": paired(arms[name]["per_page"],
                                                    arms[page_name]["per_page"])
                   for name, _ in args.line_arm}

    print(f"{len(page_rows)} pages, {len(line_rows)} lines; line coverage of the page "
          f"text: median {statistics.median(cover.values()):.3f}, "
          f"min {min(cover.values()):.3f}\n")
    print(f"{'arm':<22}{'CER pooled':>12}{'page mean':>11}{'median':>9}"
          f"{'len ratio':>11}{'>1.5x':>7}{'<1/3':>6}")
    for name, s in arms.items():
        print(f"{name:<22}{s['cer_pooled']:>12.4f}{s['cer_page_mean']:>11.4f}"
              f"{s['cer_page_median']:>9.4f}{s['length_ratio_median']:>11.3f}"
              f"{s['pages_over_1_5x']:>7}{s['pages_under_third']:>6}")
    print()
    for label, c in comparisons.items():
        print(f"{label}: mean per-page ΔCER {c['mean_diff']:+.4f}, 95 % CI "
              f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]; line arm better on "
              f"{c['a_better']}, page arm on {c['b_better']}, tied {c['ties']}")
    if args.json:
        args.json.write_text(json.dumps({"arms": arms, "coverage": cover,
                                         "comparisons": comparisons}, indent=1,
                                        ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

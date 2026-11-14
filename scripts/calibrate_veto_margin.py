#!/usr/bin/env python3
"""L3 of agentic_historian#491: the margin the fusion veto cuts at, from L2's data.

``fusion.fuse`` in the historian vetoes every candidate that lies more than a
margin (bits/char) above the **best candidate of the same page** under one
scorer. The margin was set provisionally from the #298 trace (1.5). This script
sets it from measured material: the same lines, readings and scorer that
``measure_scorer_tau.py`` reads, and for a grid of margins it counts what the
veto would have done —

* how many of the **plausible** readings (CER below ``--plausible``) it would
  have thrown out: the cost;
* how many of the **garbage** readings (CER at or above ``--garbage``) it would
  have removed: the gain;
* on how many lines it would have vetoed the reading with the **lowest CER**:
  the fatal case, the one #298 is about, in the other direction;
* on how many lines one reading survives alone, and how its CER compares with
  the best CER on that line.

The recommendation line names the smallest margin whose cost stays under
``--max-plausible-lost`` and whose fatal rate stays under ``--max-fatal``; the
table is what to read, the line is a suggestion.

    scripts/calibrate_veto_margin.py val.jsonl <job>/data/eval_report.raw.jsonl ... \\
        --scorer ~/atr-cache/scorers/de.scorer.json --json margin.json

Pure Python, like ``measure_scorer_tau.py``, whose loaders it reuses.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from atr_training.scorer import CharNgramScorer  # noqa: E402
from atr_training.textmetrics import cer  # noqa: E402

_spec = importlib.util.spec_from_file_location("measure_scorer_tau", HERE / "measure_scorer_tau.py")
_mst = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mst)
load_references, load_predictions = _mst.load_references, _mst.load_predictions

DEFAULT_MARGINS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)


def score_lines(references: dict[str, str], runs: dict[str, dict[str, str]],
                scorer: CharNgramScorer) -> dict[str, list[dict]]:
    """image → [{model, bits, cer}] for every line at least two models read.
    An empty reading has infinite bits and CER 1.0."""
    lines: dict[str, list[dict]] = {}
    for model, preds in runs.items():
        for img, pred in preds.items():
            if img not in references:
                continue
            bits = scorer.bits_per_char(pred) if pred.strip() else float("inf")
            lines.setdefault(img, []).append({"model": model, "bits": bits,
                                              "cer": cer(pred, references[img]) if pred.strip() else 1.0})
    return {img: cands for img, cands in lines.items() if len(cands) >= 2}


def apply_margin(lines: dict[str, list[dict]], margin: float, *, plausible: float = 0.3,
                 garbage: float = 0.5) -> dict:
    n_cands = n_vetoed = plausible_total = plausible_lost = garbage_total = garbage_removed = 0
    fatal = single = 0
    single_cer: list[float] = []
    single_best: list[float] = []
    for cands in lines.values():
        best = min(c["bits"] for c in cands)
        lowest_cer = min(c["cer"] for c in cands)
        kept = []
        for c in cands:
            n_cands += 1
            is_plausible, is_garbage = c["cer"] < plausible, c["cer"] >= garbage
            plausible_total += is_plausible
            garbage_total += is_garbage
            if c["bits"] - best > margin:
                n_vetoed += 1
                plausible_lost += is_plausible
                garbage_removed += is_garbage
            else:
                kept.append(c)
        if lowest_cer < min(c["cer"] for c in kept):
            fatal += 1
        if len(kept) == 1:
            single += 1
            single_cer.append(kept[0]["cer"])
            single_best.append(lowest_cer)
    n_lines = len(lines)
    return {
        "margin": margin, "lines": n_lines, "candidates": n_cands, "vetoed": n_vetoed,
        "vetoed_share": n_vetoed / n_cands if n_cands else None,
        "plausible": plausible_total, "plausible_lost": plausible_lost,
        "plausible_lost_share": plausible_lost / plausible_total if plausible_total else None,
        "garbage": garbage_total, "garbage_removed": garbage_removed,
        "garbage_removed_share": garbage_removed / garbage_total if garbage_total else None,
        "fatal": fatal, "fatal_share": fatal / n_lines if n_lines else None,
        "single_survivor": single,
        "single_survivor_mean_cer": sum(single_cer) / single if single else None,
        "single_survivor_best_cer": sum(single_best) / single if single else None,
    }


def calibrate(references, runs, scorer, *, margins=DEFAULT_MARGINS, plausible=0.3, garbage=0.5,
              max_plausible_lost=0.05, max_fatal=0.01) -> dict:
    lines = score_lines(references, runs, scorer)
    rows = [apply_margin(lines, m, plausible=plausible, garbage=garbage) for m in margins]
    ok = [r for r in rows
          if (r["plausible_lost_share"] or 0) <= max_plausible_lost and (r["fatal_share"] or 0) <= max_fatal]
    return {"scorer": scorer.cell.key(), "lines": len(lines), "models": sorted(runs),
            "plausible_cer_below": plausible, "garbage_cer_at_least": garbage,
            "max_plausible_lost": max_plausible_lost, "max_fatal": max_fatal,
            "rows": rows, "recommended_margin": min(r["margin"] for r in ok) if ok else None}


def _pct(v) -> str:
    return "—" if v is None else f"{v:.1%}"


def _num(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def render(table: dict) -> str:
    out = [f"scorer `{table['scorer']}`, {table['lines']} lines read by at least two of "
           f"{len(table['models'])} models; plausible = CER < {table['plausible_cer_below']}, "
           f"garbage = CER ≥ {table['garbage_cer_at_least']}", "",
           "| margin | vetoed | plausible lost | garbage removed | fatal lines | one survivor | survivor CER / best CER |",
           "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in table["rows"]:
        out.append(f"| {r['margin']:.2f} | {r['vetoed']} ({_pct(r['vetoed_share'])}) | "
                   f"{r['plausible_lost']} of {r['plausible']} ({_pct(r['plausible_lost_share'])}) | "
                   f"{r['garbage_removed']} of {r['garbage']} ({_pct(r['garbage_removed_share'])}) | "
                   f"{r['fatal']} ({_pct(r['fatal_share'])}) | {r['single_survivor']} | "
                   f"{_num(r['single_survivor_mean_cer'])} / {_num(r['single_survivor_best_cer'])} |")
    m = table["recommended_margin"]
    if m is None:
        out.append(f"\nno margin keeps plausible loss ≤ {table['max_plausible_lost']:.0%} and fatal lines "
                   f"≤ {table['max_fatal']:.0%}: the garbage edge and the plausible band overlap under this scorer")
    else:
        row = next(r for r in table["rows"] if r["margin"] == m)
        out.append(f"\nrecommended margin: {m:.2f} bits/char — loses {_pct(row['plausible_lost_share'])} of the "
                   f"plausible readings, removes {_pct(row['garbage_removed_share'])} of the garbage, "
                   f"vetoes the best reading on {_pct(row['fatal_share'])} of the lines")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_set", type=Path, help="val.jsonl with image and text (the reference)")
    ap.add_argument("reports", nargs="+", metavar="[NAME=]REPORT", help="eval_report files, as measure_scorer_tau")
    ap.add_argument("--scorer", type=Path, required=True, help="the one scorer the fusion will veto with")
    ap.add_argument("--margins", default=",".join(str(m) for m in DEFAULT_MARGINS),
                    help="comma-separated margins in bits/char")
    ap.add_argument("--plausible", type=float, default=0.3)
    ap.add_argument("--garbage", type=float, default=0.5)
    ap.add_argument("--max-plausible-lost", type=float, default=0.05)
    ap.add_argument("--max-fatal", type=float, default=0.01)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    references = load_references(args.eval_set)
    runs = {}
    for spec in args.reports:
        name, sep, path = spec.partition("=")
        label, preds = load_predictions(Path(path if sep else spec), name if sep else None)
        if label in runs:
            raise SystemExit(f"two reports under the label {label!r}; name one with NAME=PATH")
        runs[label] = preds
    margins = [float(m) for m in args.margins.split(",") if m.strip()]
    table = calibrate(references, runs, CharNgramScorer.load(args.scorer), margins=margins,
                      plausible=args.plausible, garbage=args.garbage,
                      max_plausible_lost=args.max_plausible_lost, max_fatal=args.max_fatal)
    print(render(table))
    if args.json:
        args.json.write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

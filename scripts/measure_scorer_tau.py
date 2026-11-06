#!/usr/bin/env python3
"""L2 of agentic_historian#491: does perplexity order candidates the way CER does?

The measurement that decides whether the scorer matrix (#110) is worth building
or only a veto at the garbage edge. Material with a *real* external reference —
the Bundesratsprotokolle lines, ``escript_test`` — scored by several models
(``evaluate_qlora`` writes ``<report>.raw.jsonl`` with one reading per line; the
reference stands in the eval set under the same image path). For every line and
model this computes the CER against the reference and the bits/char under one
or more scorers, then:

* **Kendall τ-b within each model** between bits/char and CER over its lines —
  does a worse reading cost more bits?
* **The share of correctly ordered pairs across models, per line** — of two
  readings of the same line, does the one with more bits have the higher CER?
  This is the question the fusion veto (L3) and the routing prior (L4) ask.
* Both again on the **plausible** candidates only (CER below ``--plausible``),
  because the pre-experiment showed garbage is easy and the plausible band is
  where ranking failed.

Every scorer given is measured side by side, so a cell can be read against the
generic cell. **The decision rule from the epic:** τ < 0.3 on the plausible band
→ L4 and the matrix fall, L3 stays.

    scripts/measure_scorer_tau.py val.jsonl \\
        --scorer xix=jobs/<id>/data/de-Swiss-protocol-19-20.scorer.json \\
        --scorer generic=jobs/<id>/data/de.scorer.json \\
        <job>/data/eval_report.<label>.json ... --json tau.json

Pure Python and the repo's own ``textmetrics.cer``; no scipy. τ-b is O(n²)
per model, which is seconds for a few thousand lines.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atr_training.scorer import CharNgramScorer
from atr_training.textmetrics import cer


def kendall_tau_b(x: list[float], y: list[float]) -> float | None:
    """Kendall's τ-b (ties corrected) between two equally long sequences.
    ``None`` when fewer than two pairs or one side is constant."""
    n = len(x)
    if n < 2 or len(y) != n:
        return None
    concordant = discordant = ties_x = ties_y = 0
    for i, j in combinations(range(n), 2):
        dx = (x[i] > x[j]) - (x[i] < x[j])
        dy = (y[i] > y[j]) - (y[i] < y[j])
        if dx == 0 and dy == 0:
            continue
        if dx == 0:
            ties_x += 1
        elif dy == 0:
            ties_y += 1
        elif dx == dy:
            concordant += 1
        else:
            discordant += 1
    denom = ((concordant + discordant + ties_x) * (concordant + discordant + ties_y)) ** 0.5
    if denom == 0:
        return None
    return (concordant - discordant) / denom


def load_references(eval_set: Path) -> dict[str, str]:
    refs = {}
    for line in eval_set.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            refs[row["image"]] = row["text"]
    return refs


def load_predictions(report: Path) -> tuple[str, dict[str, str]]:
    """(model label, image -> raw reading) from ``<report>.raw.jsonl``."""
    label = report.stem
    try:
        label = json.loads(report.read_text(encoding="utf-8")).get("model", label) or label
    except (OSError, ValueError):
        pass
    preds = {}
    for line in report.with_suffix(".raw.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            preds[row["image"]] = row["raw"] if isinstance(row["raw"], str) else str(row["raw"])
    return str(label), preds


def measure(references: dict[str, str], runs: dict[str, dict[str, str]],
            scorers: dict[str, CharNgramScorer], *, plausible: float = 0.3) -> dict:
    """The table: per scorer, per model τ (all / plausible) and the share of
    correctly ordered cross-model pairs per line (all / plausible)."""
    out: dict = {"n_lines": len(references), "plausible_cer_below": plausible, "scorers": {}}
    # CER once per model and line; bits once per scorer, model and line
    cers: dict[str, dict[str, float]] = {
        m: {img: cer(pred, references[img]) for img, pred in preds.items() if img in references}
        for m, preds in runs.items()}
    for name, scorer in scorers.items():
        per_model = {}
        bits: dict[str, dict[str, float]] = {}
        for m, preds in runs.items():
            bits[m] = {img: scorer.bits_per_char(pred) if pred.strip() else float("inf")
                       for img, pred in preds.items() if img in references}
            imgs = sorted(bits[m])
            xs = [bits[m][i] for i in imgs]
            ys = [cers[m][i] for i in imgs]
            keep = [k for k in range(len(imgs)) if ys[k] < plausible]
            per_model[m] = {
                "n": len(imgs),
                "tau": kendall_tau_b(xs, ys),
                "n_plausible": len(keep),
                "tau_plausible": kendall_tau_b([xs[k] for k in keep], [ys[k] for k in keep]),
                "mean_cer": sum(ys) / len(ys) if ys else None,
                "mean_bits": (sum(x for x in xs if x != float("inf")) / len(xs)) if xs else None,
            }
        # cross-model pairs, per line
        agree = total = agree_pl = total_pl = 0
        models = sorted(runs)
        for img in references:
            for a, b in combinations(models, 2):
                if img not in cers[a] or img not in cers[b]:
                    continue
                dc = cers[a][img] - cers[b][img]
                db = bits[a][img] - bits[b][img]
                if dc == 0 or db == 0:
                    continue
                total += 1
                ok = (dc > 0) == (db > 0)
                agree += ok
                if cers[a][img] < plausible and cers[b][img] < plausible:
                    total_pl += 1
                    agree_pl += ok
        out["scorers"][name] = {
            "cell": scorer.cell.key(),
            "train_chars": scorer.train_chars,
            "per_model": per_model,
            "pairs": {"n": total, "ordered_like_cer": agree / total if total else None,
                      "n_plausible": total_pl,
                      "ordered_like_cer_plausible": agree_pl / total_pl if total_pl else None},
        }
    return out


def verdict(table: dict, threshold: float = 0.3) -> str:
    """The decision rule of #491 in one line per scorer."""
    lines = []
    for name, s in table["scorers"].items():
        taus = [v["tau_plausible"] for v in s["per_model"].values() if v["tau_plausible"] is not None]
        if not taus:
            lines.append(f"{name} ({s['cell']}): no plausible band to measure on")
            continue
        worst, best = min(taus), max(taus)
        if best < threshold:
            lines.append(f"{name} ({s['cell']}): τ on the plausible band {worst:.2f}…{best:.2f} < "
                         f"{threshold} — veto only; L4 and the matrix fall")
        else:
            lines.append(f"{name} ({s['cell']}): τ on the plausible band {worst:.2f}…{best:.2f} — "
                         f"orders plausible candidates; L4 may be built, measured against this")
    return "\n".join(lines)


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def render(table: dict) -> str:
    rows = ["| scorer | cell | model | n | τ | n plausible | τ plausible | mean CER | mean bits |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, s in table["scorers"].items():
        for m, v in s["per_model"].items():
            rows.append(f"| {name} | `{s['cell']}` | {m} | {v['n']} | {_fmt(v['tau'])} | "
                        f"{v['n_plausible']} | {_fmt(v['tau_plausible'])} | {_fmt(v['mean_cer'])} | "
                        f"{_fmt(v['mean_bits'])} |")
        p = s["pairs"]
        rows.append(f"| {name} | `{s['cell']}` | *pairs across models* | {p['n']} | "
                    f"{_fmt(p['ordered_like_cer'])} | {p['n_plausible']} | "
                    f"{_fmt(p['ordered_like_cer_plausible'])} | | |")
    return "\n".join(rows) + "\n\n" + verdict(table)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_set", type=Path, help="val.jsonl with image and text (the reference)")
    ap.add_argument("reports", type=Path, nargs="+", help="eval_report.<label>.json files")
    ap.add_argument("--scorer", action="append", required=True, metavar="NAME=PATH",
                    help="a scorer artefact; repeat to read cells side by side")
    ap.add_argument("--plausible", type=float, default=0.3,
                    help="CER below which a reading counts as plausible (default 0.3)")
    ap.add_argument("--json", type=Path, help="also write the table as JSON")
    args = ap.parse_args(argv)

    references = load_references(args.eval_set)
    runs = dict(load_predictions(p) for p in args.reports)
    scorers = {}
    for spec in args.scorer:
        name, _, path = spec.partition("=")
        if not path:
            name, path = Path(spec).stem, spec
        scorers[name] = CharNgramScorer.load(path)
    table = measure(references, runs, scorers, plausible=args.plausible)
    print(render(table))
    if args.json:
        args.json.write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

**Across engines** (agentic_historian#875, the L4 rule): with readings of the
same lines by different engines — the VLM's ``evaluate_qlora``, TrOCR's
``evaluate_trocr``, kraken's ``read_lines_kraken.py`` — the cross-model pairs are
also counted **per engine pair** (``pairs_by_engine``: ``kraken|vlm`` apart from
``trocr|vlm``) and over every pair of two different engines
(``pairs_across_engines``). A report's engine is its JSON's ``engine`` field,
else read off its shape (``adapter`` → vlm, ``checkpoint``/``beam_size`` →
trocr); ``--engine NAME=ENGINE`` sets it. **The rule:** pairs across different
engines ordered like CER less than 0.7 of the time → L4 stays unbuilt.

    scripts/measure_scorer_tau.py $E/data/val.jsonl \
        vlm-qwen=$E/readings/qwen35-4b-xix.json trocr-xix=$E/readings/trocr-xix.json \
        kraken-xix=$E/readings/kraken-xix.json \
        --scorer generic=<scorers>/de.scorer.json --scorer xix=<scorers>/de-Swiss-protocol-19-20.scorer.json \
        --json tau-engines.json

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


def load_predictions(report: Path, label: str | None = None) -> tuple[str, dict[str, str]]:
    """(model label, image -> raw reading) from ``<report>.raw.jsonl``.

    ``report`` is the ``eval_report.<label>.json`` or the ``.raw.jsonl`` itself.
    The label is, in this order: the one given (``NAME=PATH`` on the command
    line), the ``model`` in the report JSON, the job directory the file lies in
    (``<job>/data/eval_report.raw.jsonl`` — every job's bare report has the same
    stem, and two runs under one label would silently overwrite each other),
    else the file stem. A reading is ``raw`` (``evaluate_qlora``) or
    ``prediction`` (the TrOCR trainer's report).
    """
    raw = report if report.name.endswith(".raw.jsonl") else report.with_suffix(".raw.jsonl")
    if label is None:
        try:
            label = json.loads(report.read_text(encoding="utf-8")).get("model") or None
        except (OSError, ValueError):
            label = None
    if label is None:
        job = raw.resolve().parent
        label = job.parent.name if job.name == "data" and job.parent.name else raw.stem
    preds = {}
    for line in raw.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            text = row.get("raw", row.get("prediction", ""))
            preds[row["image"]] = text if isinstance(text, str) else str(text)
    return str(label), preds


def load_engine(report: Path) -> str | None:
    """The engine that wrote ``report`` (``.json`` or ``.raw.jsonl``), or None.

    The report JSON's ``engine`` when it names one (``read_lines_kraken.py``),
    else its shape: ``evaluate_qlora`` records the ``adapter``, ``evaluate_trocr``
    the ``checkpoint`` and its ``beam_size``. A bare ``.raw.jsonl`` says nothing
    about who wrote it; ``--engine`` names it then.
    """
    path = report.with_name(report.name[:-len(".raw.jsonl")] + ".json") \
        if report.name.endswith(".raw.jsonl") else report
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("engine"), str) and data["engine"]:
        return data["engine"]
    if "adapter" in data:
        return "vlm"
    if "checkpoint" in data or "beam_size" in data:
        return "trocr"
    return None


def _engine_pair(a: str | None, b: str | None) -> str:
    return "|".join(sorted((a or "?", b or "?")))


def _share(agree: int, total: int) -> float | None:
    return agree / total if total else None


def measure(references: dict[str, str], runs: dict[str, dict[str, str]],
            scorers: dict[str, CharNgramScorer], *, plausible: float = 0.3,
            engines: dict[str, str | None] | None = None) -> dict:
    """The table: per scorer, per model τ (all / plausible) and the share of
    correctly ordered cross-model pairs per line (all / plausible).

    With ``engines`` (model label → engine), each scorer also gets
    ``pairs_by_engine`` (the same counts per sorted engine pair, ``kraken|vlm``;
    an unknown engine is ``?``) and ``pairs_across_engines`` (every pair of two
    different known engines), and the table names the engines it was given.
    """
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
        # cross-model pairs, per line; [agree, total, agree plausible, total plausible]
        counts: dict[str, list[int]] = {}
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
                both_plausible = cers[a][img] < plausible and cers[b][img] < plausible
                if both_plausible:
                    total_pl += 1
                    agree_pl += ok
                if engines is not None:
                    c = counts.setdefault(_engine_pair(engines.get(a), engines.get(b)), [0, 0, 0, 0])
                    c[0] += ok
                    c[1] += 1
                    c[2] += ok and both_plausible
                    c[3] += both_plausible
        out["scorers"][name] = {
            "cell": scorer.cell.key(),
            "train_chars": scorer.train_chars,
            "per_model": per_model,
            "pairs": {"n": total, "ordered_like_cer": agree / total if total else None,
                      "n_plausible": total_pl,
                      "ordered_like_cer_plausible": agree_pl / total_pl if total_pl else None},
        }
        if engines is not None:
            def block(c: list[int]) -> dict:
                return {"n": c[1], "ordered_like_cer": _share(c[0], c[1]),
                        "n_plausible": c[3], "ordered_like_cer_plausible": _share(c[2], c[3])}
            across = [0, 0, 0, 0]
            for key, c in counts.items():
                first, second = key.split("|")
                if first != second and "?" not in (first, second):
                    across = [x + y for x, y in zip(across, c)]
            out["scorers"][name]["pairs_by_engine"] = {k: block(c) for k, c in sorted(counts.items())}
            out["scorers"][name]["pairs_across_engines"] = block(across)
    if engines is not None:
        out["engines"] = {m: engines.get(m) for m in sorted(runs)}
    return out


def verdict(table: dict, threshold: float = 0.3) -> str:
    """The decision rule of #491 in one line per scorer.

    Judged on the **median** τ over the models, not the best one: the first
    run (14.11.2026, eight arms) had 0.21…0.44 with two arms under 0.3, and the
    best arm alone would have carried the verdict. The share of cross-model
    pairs ordered like CER is printed beside it, because that — not the
    within-model τ — is the question L4 asks, and 0.57 there reads as a near
    coin flip whatever the τ says.
    """
    lines = []
    for name, s in table["scorers"].items():
        taus = sorted(v["tau_plausible"] for v in s["per_model"].values() if v["tau_plausible"] is not None)
        if not taus:
            lines.append(f"{name} ({s['cell']}): no plausible band to measure on")
            continue
        worst, best = taus[0], taus[-1]
        median = taus[len(taus) // 2] if len(taus) % 2 else (taus[len(taus) // 2 - 1] + taus[len(taus) // 2]) / 2
        pairs = (s.get("pairs") or {}).get("ordered_like_cer_plausible")
        pairs_txt = f"; cross-model pairs ordered like CER {pairs:.2f}" if pairs is not None else ""
        span = f"{worst:.2f}…{best:.2f} (median {median:.2f})"
        if median < threshold:
            lines.append(f"{name} ({s['cell']}): τ on the plausible band {span} < {threshold} — "
                         f"veto only; L4 and the matrix fall{pairs_txt}")
        else:
            lines.append(f"{name} ({s['cell']}): τ on the plausible band {span} — orders plausible "
                         f"candidates within a model; L4 only if the cross-model pairs do too{pairs_txt}")
    return "\n".join(lines)


def engine_verdict(table: dict, threshold: float = 0.7) -> str:
    """The L4 rule of #875, one line per scorer and engine pair: pairs across
    *different* engines on the same lines ordered like CER less than
    ``threshold`` of the time → L4 stays unbuilt.

    Judged on the **plausible** band (both readings below ``--plausible``), as
    the τ verdict is: a garbage reading is the veto's business, and what L4 has
    to do is tell two usable engines apart. The share over all pairs is printed
    beside it, so a scorer that only separates garbage shows as such.
    """
    lines = []
    for name, s in table["scorers"].items():
        if "pairs_by_engine" not in s or not s["pairs_across_engines"]["n"]:
            continue        # one engine only: the τ verdict above is all there is
        blocks = [("across engines", s["pairs_across_engines"])] + [
            (k, v) for k, v in s["pairs_by_engine"].items()
            if len(set(k.split("|"))) == 2 and "?" not in k.split("|")]
        for label, p in blocks:
            pl, al = p["ordered_like_cer_plausible"], p["ordered_like_cer"]
            head = f"{name} ({s['cell']}), {label}: "
            if pl is None:
                lines.append(head + f"no plausible pair to judge on (all pairs: {_fmt(al)}, "
                             f"n={p['n']})")
            elif pl < threshold:
                lines.append(head + f"pairs ordered like CER {pl:.2f} on the plausible band "
                             f"(n={p['n_plausible']}; all {_fmt(al)}, n={p['n']}) < {threshold} "
                             "— L4 stays unbuilt")
            else:
                lines.append(head + f"pairs ordered like CER {pl:.2f} on the plausible band "
                             f"(n={p['n_plausible']}; all {_fmt(al)}, n={p['n']}) ≥ {threshold} "
                             "— L4 passes this rule")
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
        for key, p in (s.get("pairs_by_engine") or {}).items():
            rows.append(f"| {name} | `{s['cell']}` | *pairs {key.replace('|', '–')}* | {p['n']} | "
                        f"{_fmt(p['ordered_like_cer'])} | {p['n_plausible']} | "
                        f"{_fmt(p['ordered_like_cer_plausible'])} | | |")
        if "pairs_across_engines" in s:
            p = s["pairs_across_engines"]
            rows.append(f"| {name} | `{s['cell']}` | *pairs across engines* | {p['n']} | "
                        f"{_fmt(p['ordered_like_cer'])} | {p['n_plausible']} | "
                        f"{_fmt(p['ordered_like_cer_plausible'])} | | |")
    text = "\n".join(rows) + "\n\n" + verdict(table)
    l4 = engine_verdict(table)
    return text + ("\n\n" + l4 if l4 else "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_set", type=Path, help="val.jsonl with image and text (the reference)")
    ap.add_argument("reports", nargs="+", metavar="[NAME=]REPORT",
                    help="eval_report.<label>.json or .raw.jsonl files, optionally labelled NAME=PATH")
    ap.add_argument("--scorer", action="append", required=True, metavar="NAME=PATH",
                    help="a scorer artefact; repeat to read cells side by side")
    ap.add_argument("--plausible", type=float, default=0.3,
                    help="CER below which a reading counts as plausible (default 0.3)")
    ap.add_argument("--json", type=Path, help="also write the table as JSON")
    ap.add_argument("--engine", action="append", default=[], metavar="NAME=ENGINE",
                    help="the engine of the report labelled NAME (vlm, trocr, kraken), "
                         "where its JSON does not say; any engine known adds the "
                         "per-engine-pair counts")
    args = ap.parse_args(argv)

    references = load_references(args.eval_set)
    runs = {}
    engines: dict[str, str | None] = {}
    for spec in args.reports:
        name, sep, path = spec.partition("=")
        report = Path(path if sep else spec)
        label, preds = load_predictions(report, name if sep else None)
        if label in runs:
            raise SystemExit(f"two reports under the label {label!r}; name one with NAME=PATH")
        runs[label] = preds
        engines[label] = load_engine(report)
    for spec in args.engine:
        name, sep, engine = spec.partition("=")
        if not sep or name not in runs:
            raise SystemExit(f"--engine {spec!r}: give NAME=ENGINE for one of {sorted(runs)}")
        engines[name] = engine
    scorers = {}
    for spec in args.scorer:
        name, _, path = spec.partition("=")
        if not path:
            name, path = Path(spec).stem, spec
        scorers[name] = CharNgramScorer.load(path)
    table = measure(references, runs, scorers, plausible=args.plausible,
                    engines=engines if any(engines.values()) else None)
    print(render(table))
    if args.json:
        args.json.write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    from atr_training.measurement_record import measurement  # noqa: E402

    # A record next to the job store, so live() and results() see this run (#385).
    with measurement(Path(__file__).stem):
        raise SystemExit(main())

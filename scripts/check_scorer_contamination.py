#!/usr/bin/env python3
"""Could a scorer have read the lines it is measured on? (agentic_historian#875)

The τ measurement on an external reference (the Federal Council lines) is only a
lower bound for the scorer if the scorer's training text does not contain those
lines. The scorers are built by ``build_scorer_from_draw.py`` from the
validation draws of dh-unibe corpora (XIX, XVI–XIX), and two of those corpora are
Swiss federal records (``parlamentsdienste-protokolle``,
``nr-sr-vereinigte-bundesversammlung-xix``; ``docs/EVAL_SETS.md``). Three checks,
each saying what it can and cannot see:

* **documents** — the eval set's Transkribus ``docId`` (the Federal Council set
  names it in every file name, ``…_docId_<n>``; other sets through
  ``heldout.document_of``) against the units the scorer artefact records in
  ``documents`` (page names or docIds, ``--exclude-level``). Exact, but only as
  good as the names: a document copied into another collection under a new id
  passes.
* **lines** (``--draw``, the draw files the scorer was built from, if they are
  still there) — reference lines that stand verbatim, whitespace-normalised, in
  the draw's text. Lines shorter than ``--min-line-chars`` are left out:
  ``Der Bundesrat`` stands everywhere.
* **perplexity** — the mean z of the reference lines against the scorer's own
  held-out distribution. Memorised text scores far *below* the held-out pages
  (z well under 0); text of the same register it has not seen lies near or above
  0. A population signal, not a per-line proof.

    scripts/check_scorer_contamination.py $E/data/val.jsonl \\
        --scorer xix=<scorers>/de-Swiss-protocol-19-20.scorer.json \\
        --scorer generic=<scorers>/de.scorer.json \\
        [--draw <the val.jsonl files the scorer was built from> ...] --json contamination.json

Pure Python; reads the artefacts, never trains anything.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atr_training.heldout import document_of  # noqa: E402
from atr_training.scorer import CharNgramScorer  # noqa: E402

_DOC_ID = re.compile(r"docId_(\d+)")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def row_document(row: dict) -> str | None:
    """The Transkribus document of an eval-set row: ``docId_<n>`` in its page or
    image name, else ``document_of`` on the page."""
    for key in ("page", "image"):
        match = _DOC_ID.search(row.get(key) or "")
        if match:
            return match.group(1)
    return document_of(row["page"]) if row.get("page") else None


def unit_document(unit: str) -> str:
    """A scorer unit as a document: a page name gives its docId, a bare docId
    (``--exclude-level document``) is one already."""
    match = _DOC_ID.search(unit)
    if match:
        return match.group(1)
    return document_of(unit) or unit


def _norm(text: str) -> str:
    return " ".join(text.split())


def check(references: list[dict], scorer: CharNgramScorer, draw_lines: set[str] | None,
          *, min_line_chars: int = 25) -> dict:
    eval_docs = {d for d in (row_document(r) for r in references) if d}
    scorer_docs = {unit_document(u) for u in scorer.documents}
    shared_docs = sorted(eval_docs & scorer_docs)

    lines = None
    if draw_lines is not None:
        long_refs = [_norm(r["text"]) for r in references if len(_norm(r["text"])) >= min_line_chars]
        hits = sorted({t for t in long_refs if t in draw_lines})
        lines = {"checked": len(long_refs), "verbatim_in_draw": len(hits), "examples": hits[:10]}

    zs = []
    for r in references:
        text = (r.get("text") or "").strip()
        if text and scorer.heldout:
            zs.append(scorer.z(text)[0])
    return {
        "cell": scorer.cell.key(), "sources": scorer.sources, "train_chars": scorer.train_chars,
        "documents": {"eval_set": len(eval_docs), "scorer_units": len(scorer.documents),
                      "shared": shared_docs},
        "lines": lines,
        "perplexity": {"lines": len(zs), "mean_z": mean(zs) if zs else None,
                       "share_below_minus_2": (sum(z < -2 for z in zs) / len(zs)) if zs else None},
    }


def render(name: str, r: dict) -> str:
    d, p, lines = r["documents"], r["perplexity"], r["lines"]
    out = [f"{name} ({r['cell']}), {r['train_chars']:,} training characters from "
           f"{', '.join(r['sources']) or 'sources not recorded'}"]
    out.append(f"  documents: {len(d['shared'])} of the eval set's {d['eval_set']} also among the "
               f"scorer's {d['scorer_units']} units" + (f" ({', '.join(d['shared'][:10])})"
                                                        if d["shared"] else ""))
    if lines is None:
        out.append("  lines: not checked (no --draw: the scorer artefact keeps counts, not text)")
    else:
        out.append(f"  lines: {lines['verbatim_in_draw']} of {lines['checked']} reference lines "
                   "stand verbatim in the draw")
    if p["mean_z"] is None:
        out.append("  perplexity: not calibrated, no held-out distribution to compare with")
    else:
        out.append(f"  perplexity: mean z {p['mean_z']:+.2f} against the scorer's held-out pages, "
                   f"{p['share_below_minus_2']:.1%} of the lines below z = -2")
    clean = not d["shared"] and (lines is None or not lines["verbatim_in_draw"]) and \
        (p["mean_z"] is None or p["mean_z"] > -1)
    out.append("  → no sign of the eval lines in the training text" if clean else
               "  → the scorer may have read these lines: the τ on them is not a lower bound")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_set", type=Path, help="val.jsonl with image, text and page")
    ap.add_argument("--scorer", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--draw", action="append", type=Path, default=[],
                    help="a draw the scorers were built from; repeat for several")
    ap.add_argument("--min-line-chars", type=int, default=25)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    references = _rows(args.eval_set)
    draw_lines = None
    if args.draw:
        draw_lines = {_norm(row.get("text") or "") for path in args.draw for row in _rows(path)}
    table = {}
    for spec in args.scorer:
        name, _, path = spec.partition("=")
        if not path:
            name, path = Path(spec).stem, spec
        table[name] = check(references, CharNgramScorer.load(path), draw_lines,
                            min_line_chars=args.min_line_chars)
        print(render(name, table[name]))
    if args.json:
        args.json.write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

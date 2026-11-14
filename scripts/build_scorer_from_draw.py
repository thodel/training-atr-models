#!/usr/bin/env python3
"""L1 of agentic_historian#491: a cell's scorer from a job's validation draw.

The line manifests ``prepare`` wrote (``lines_train.jsonl``) are gone from the
job store; what the VLM jobs kept is ``data/val.jsonl`` — ground-truth lines of
the same corpora, one row per line crop, with the page each crop came from
(``005533_083620_0201_….xml``: collection, **document**, page). That is edition
text and corrected ground truth, never an engine's output, so a scorer may be
trained on it (#491, the one construction constraint).

What this script guards is the other constraint, #23: the scorer must not have
seen the lines it will later score. The L2 material is the readings in
``eval_report.raw.jsonl`` of the same jobs, which name their crops in the same
draw. So every **unit** that holds a scored crop leaves the training text
altogether (``--exclude-scored``) — the unit being the **page** by default
(``--exclude-level page``: the evaluation pages, as #491 words the rule) or the
whole Transkribus **document** (``--exclude-level document``, stricter). The
rest is split by unit, deterministically, into the training text and the
held-out units the cell is calibrated on.

Why the page is the default, measured on the XIX draw (14.11.2026): 93 280
lines lie in about 40 documents, and the 746 crops the six VLM arms scored
touch every one of them — by document, nothing is left to train on. By page the
scorer has seen *other pages of the same hands*, which is what a register
scorer is for, but it also means the τ it yields is an upper bound on what a
scorer meets on a document it has never seen; the report says which level was
used, and the artefact's ``documents`` are the units it trained on.

    scripts/build_scorer_from_draw.py --draw $J/<xix-job>/data/val.jsonl \\
        --exclude-scored $J/<arm>/data/eval_report.raw.jsonl ... \\
        --lang de --region Swiss --document-type protocol --century-band 19-20 \\
        --source nr-sr-vereinigte-bundesversammlung-xix --gt-commit <sha> \\
        --out ~/atr-cache/scorers/de-Swiss-protocol-19-20.scorer.json

Several ``--draw`` files make one text (the generic ``de`` cell is the XIX draw
and the XVI–XIX draw together). CPU, seconds to a minute. ``--curve`` also
prints the learning curve on the held-out pages, the measurement that sets the
cell's floor (``MIN_CHARS`` is an anchor, not a number).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atr_training.heldout import document_of
from atr_training.scorer import BOUNDARY, Cell, CharNgramScorer, TooLittleText, learning_curve


def read_draw(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def document_of_row(row: dict, level: str = "page") -> str:
    """The unit a draw row belongs to. ``level="document"``: the Transkribus
    document in its ``page`` name, else the page name, else the crop itself.
    ``level="page"``: the page name, else the crop. Never None — a row that
    cannot be placed is its own unit, so it can still be excluded by name."""
    page = row.get("page") or ""
    if level == "document":
        doc = document_of(page) if page else None
        return doc or Path(page).stem or row["image"]
    if level == "page":
        return Path(page).stem or row["image"]
    raise ValueError(f"unknown exclusion level {level!r}")


def pages_by_document(rows: list[dict], level: str = "page") -> dict[str, list[str]]:
    """unit → its lines, in draw order. The scorer only needs the unit
    boundary to count the text and refuse contamination."""
    out: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        text = (row.get("text") or "").strip()
        if text:
            out[document_of_row(row, level)].append(text)
    return out


def scored_images(reports: list[Path]) -> set[str]:
    imgs: set[str] = set()
    for p in reports:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                imgs.add(json.loads(line)["image"])
    return imgs


def split_documents(docs: list[str], *, calib_every: int = 10) -> tuple[list[str], list[str]]:
    """Deterministic: sorted, every ``calib_every``-th document is held out."""
    ordered = sorted(docs)
    calib = [d for i, d in enumerate(ordered) if i % calib_every == calib_every - 1]
    train = [d for i, d in enumerate(ordered) if i % calib_every != calib_every - 1]
    return train, calib


def build(draws: list[Path], cell: Cell, *, exclude: list[Path] = (), order: int = 6,
          calib_every: int = 10, min_chars: int | None = None, sources: list[str] = (),
          gt_commit: str = "", curve: bool = False, level: str = "page") -> tuple[CharNgramScorer, dict]:
    rows = [r for p in draws for r in read_draw(p)]
    by_doc = pages_by_document(rows, level)
    scored = scored_images(list(exclude))
    excluded = {document_of_row(r, level) for r in rows if r["image"] in scored}
    usable = [d for d in by_doc if d not in excluded]
    train_docs, calib_docs = split_documents(usable, calib_every=calib_every)
    train_pages = [(d, BOUNDARY.join(by_doc[d])) for d in train_docs]
    calib_pages = [(d, BOUNDARY.join(by_doc[d])) for d in calib_docs]
    kwargs = {} if min_chars is None else {"min_chars": min_chars}
    scorer = CharNgramScorer(cell, order=order).train(train_pages, sources=sources,
                                                      gt_commit=gt_commit, **kwargs)
    heldout = scorer.calibrate(calib_pages)
    report = {
        "cell": cell.key(), "order": order, "rows": len(rows), "unit": level, "units": len(by_doc),
        "excluded_units": len(excluded), "scored_crops": len(scored),
        "train_units": len(train_docs), "train_chars": scorer.train_chars,
        "calibration_units": len(calib_docs), "heldout": heldout,
    }
    if curve:
        total = scorer.train_chars
        steps = [n for n in (16_000, 50_000, 150_000, 500_000, 1_500_000, 5_000_000) if n < total] + [total]
        report["learning_curve"] = learning_curve(train_pages, calib_pages, steps, cell=cell, order=order)
    return scorer, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draw", type=Path, action="append", required=True,
                    help="val.jsonl of a job (image, text, page); repeat to pool draws")
    ap.add_argument("--exclude-scored", type=Path, action="append", default=[],
                    help="eval_report.raw.jsonl whose crops' units leave the training text (#23)")
    ap.add_argument("--exclude-level", choices=("page", "document"), default="page",
                    help="the unit a scored crop takes with it: its page (default) or its whole document")
    ap.add_argument("--lang", default="de")
    ap.add_argument("--region")
    ap.add_argument("--document-type")
    ap.add_argument("--century-band")
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--calib-every", type=int, default=10, help="every n-th document is held out")
    ap.add_argument("--min-chars", type=int, default=None, help="override MIN_CHARS (tests only)")
    ap.add_argument("--source", action="append", default=[], help="corpus the draw comes from; repeat")
    ap.add_argument("--gt-commit", default="")
    ap.add_argument("--curve", action="store_true", help="also print the learning curve")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cell = Cell(lang=args.lang, region=args.region, document_type=args.document_type,
                century_band=args.century_band)
    try:
        scorer, report = build(args.draw, cell, exclude=args.exclude_scored, order=args.order,
                               calib_every=args.calib_every, min_chars=args.min_chars,
                               sources=args.source, gt_commit=args.gt_commit, curve=args.curve,
                               level=args.exclude_level)
    except (TooLittleText, ValueError) as exc:
        raise SystemExit(f"not built: {exc}") from exc
    scorer.save(args.out)
    report["out"] = str(args.out)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

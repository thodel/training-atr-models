#!/usr/bin/env python3
"""The line crops of a page draw: a line arm's draw on the same pages (#74).

    python scripts/lines_of_pages.py --pages <page job>/data/val_eval.jsonl \\
        --lines <line corpus>/data/val.jsonl --out lines.jsonl

A page arm is scored on whole pages, a line arm on line crops. To compare the two
on the same material, the line arm reads every line of the pages the page arm
read: each row of ``--lines`` whose ``page`` names a page of ``--pages`` is kept,
in the order the line corpus has it (the order its compile wrote, page by page).

Pages are matched by file stem (``000048_8716099_0003_97836934``): a page draw
names the ``.jpg``, a line row the ``.xml`` it was cut from. That is only sound
when both corpora materialized the same selection under the same names — check
that their ``pages_val.lst`` agree before trusting the match; this script reports
how many draw pages found lines, and refuses when any found none.

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def stem(path: str) -> str:
    return Path(path).stem


def lines_of_pages(page_rows: list[dict], line_rows: list[dict]) -> tuple[list[dict], dict]:
    """The line rows of the draw's pages, and a count per page stem."""
    wanted = [stem(r["image"]) for r in page_rows]
    wanted_set = set(wanted)
    kept = [r for r in line_rows if r.get("page") and stem(r["page"]) in wanted_set]
    per_page: dict[str, int] = {s: 0 for s in wanted}
    for row in kept:
        per_page[stem(row["page"])] += 1
    return kept, per_page


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pages", type=Path, required=True, help="a page arm's val_eval.jsonl")
    p.add_argument("--lines", type=Path, required=True, help="the line corpus's val.jsonl")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    kept, per_page = lines_of_pages(_read(args.pages), _read(args.lines))
    empty = [s for s, n in per_page.items() if n == 0]
    print(f"{len(per_page)} page(s) in the draw, {len(kept)} line(s) found, "
          f"{len(per_page) - len(empty)} page(s) with lines")
    if empty:
        print(f"refusing: {len(empty)} page(s) of the draw have no line in the line "
              f"corpus, e.g. {empty[:3]} — the two corpora do not hold the same pages",
              file=sys.stderr)
        return 1
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
                        encoding="utf-8")
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

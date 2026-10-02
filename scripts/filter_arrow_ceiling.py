#!/usr/bin/env python3
"""Cut a compiled corpus down to the lines kraken can actually batch (#145).

    scripts/filter_arrow_ceiling.py --in sweep_train.arrow --dry-run
    scripts/filter_arrow_ceiling.py --in sweep_train.arrow --out sweep_train-60.arrow

`scripts/apply_line_ceiling.py` applies the same ceiling to a **page pool**, and
on the German pool it removes nothing: 0 of 345,359 lines, widest box 60.0:1
before and after — measured 02.10.2026. The pool is already clean *for the
quantity that tool measures*, which is the `Coords` box.

What kraken pads a batch to is the **extracted image**, and that is a different
quantity: 11.4 median, 81.1 at p99, **177:1** at the worst, 4.97 % above the
ceiling. Four fine-tunes died of CUDA OOM on this corpus, three to thirteen
minutes in, mid-epoch — the signature of a peak that hangs on one outlier.

So this cuts where the quantity binds: after the compile, where it can be
measured exactly instead of predicted from the PageXML. Neither the box width nor
the baseline's arc length predicts it (10.3 and 10.2 median against the image's
11.4; 60.0 and 71.3 worst against 177.0), so there is nothing to predict it
*with* — the discrepancy sits in the height ketos normalises to, and that is not
yet explained.

The output is a **new data version** and needs its own digest: a sweep manifest
pins `data.digest` and refuses to load without one (#113), and every number
measured on the uncut corpus keeps pointing at the uncut corpus.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_training.line_ceiling import MAX_LINE_ASPECT, cut_arrow  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="source", type=Path, required=True,
                    help="compiled .arrow to read")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write the cut corpus; never the input")
    ap.add_argument("--ceiling", type=float, default=MAX_LINE_ASPECT,
                    help=f"width-to-height limit on the extracted image "
                         f"(default {MAX_LINE_ASPECT}, the same figure "
                         f"pagexml applies to the box)")
    ap.add_argument("--dry-run", action="store_true",
                    help="measure and report, write nothing")
    args = ap.parse_args(argv)

    if not args.source.is_file():
        ap.error(f"{args.source} does not exist")
    if not args.dry_run and args.out is None:
        ap.error("--out is required unless --dry-run is given")

    try:
        cut = cut_arrow(args.source, args.out, ceiling=args.ceiling,
                        dry_run=args.dry_run)
    except RuntimeError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    print(cut)
    if cut.removed == 0:
        print("Nothing to cut: every line is already under the ceiling. For this "
              "corpus that would be the finding, not the expected outcome — the "
              "page pool reads that way and the images do not (#145).")
    if cut.written:
        print(f"\nNew data version. Record its digest before any sweep uses it:\n"
              f"  sha256sum {cut.written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

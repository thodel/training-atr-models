#!/usr/bin/env python3
"""Which published numbers cross the held-out boundary? (#247, #98)

    # on asteraix, or any box with the research share mounted
    python scripts/heldout_boundary.py --markdown   # the Results table, to paste
    python scripts/heldout_boundary.py --json | jq '.summary'
    python scripts/heldout_boundary.py --store /scratch/.../jobs --all

Applies **today's** held-out registry (``config/heldout_eval_documents.json``)
to the manifest each completed run actually trained on, and says per run whether
its number stands. No GPU, no network, no model — minutes.

Both job stores by default, because they do not hold the same runs: #247
measured 61 in the share against 58 in scratch, and a run purged from one may
still be whole in the other. A run found in both is reported from both; the
rows carry the store, so a disagreement is visible rather than averaged.

``progress.reserved_pages`` is printed beside the measurement rather than
trusted. It is a corpus fingerprint, not a boundary marker, and for a chunked or
line-level run it recorded ``0`` — the same value a real check that found
nothing records. Where the record and the material disagree, the material wins
and the row shows both.

Four verdicts, and the two that are not verdicts about the number:

    clean        entries name documents, none reserved today
    crossed      trained on material reserved today — the number is optimistic
    uncheckable  nothing in the manifest names a document (line corpus of
                 foreign scans, or a chunked run whose train side was
                 materialised per chunk and discarded)
    gone         no manifest left (scratch purge) — undecidable, not clean

This refuses to guess. A run it cannot check is never counted as unaffected,
which is the one way a census like this gets quietly optimistic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atr_training.heldout_boundary import (  # noqa: E402
    Verdict,
    check_store,
    load_today,
    resolve_adoptions,
    summarize,
)
from atr_training.settings import TrainerSettings  # noqa: E402

HEADER = ("| Lauf | Store | CER | `reserved_pages` | Verdikt | gefallen | Grund |\n"
          "|---|---|---:|---:|---|---:|---|")


def default_stores(settings: TrainerSettings) -> list[tuple[str, Path]]:
    """``(label, root)`` for every job store this repo knows about.

    From ``atr_results_mcp.probe.Paths.job_roots`` rather than from a list here:
    that is already the repo's answer to "which stores exist", labelled scratch
    and share, and a second copy of the layout would drift from it (#252 is the
    same lesson one floor down). This box's own ``jobs_root`` is added because on
    asteraix the ``.env`` points it at the store the service actually writes.

    Every store is *named* even when unreadable — the caller prints that. A
    missing store that silently contributes no rows reads as "no runs affected",
    which is the failure mode this whole script exists to prevent.
    """
    from atr_results_mcp.probe import Paths

    stores: list[tuple[str, Path]] = [("local", Path(settings.jobs_root))]
    stores += [(label, Path(root)) for label, root in Paths().job_roots]
    seen: set[Path] = set()
    unique: list[tuple[str, Path]] = []
    for label, root in stores:
        resolved = root.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append((label, resolved))
    return unique


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", action="append", type=Path, default=None,
                        help="a job store root; repeatable. Default: this box's stores.")
    parser.add_argument("--registry", type=Path, default=None,
                        help="held-out registry (default: this checkout's)")
    parser.add_argument("--all", action="store_true",
                        help="every run, not only those with a CER in the record — a CER "
                             "recovered by re-scoring lives outside it")
    parser.add_argument("--markdown", action="store_true", help="the table for the docs page")
    parser.add_argument("--json", action="store_true", help="machine-readable")
    args = parser.parse_args(argv)

    held = load_today(args.registry)
    if not held:
        print("the held-out registry holds nothing — every run would read as clean, "
              "which is the one answer this script must not give", file=sys.stderr)
        return 2

    settings = TrainerSettings()
    stores = ([(str(p), p) for p in args.store] if args.store
              else default_stores(settings))

    checks = []
    for label, root in stores:
        if not Path(root).is_dir():
            print(f"store {label} ({root}) is not readable from here — its runs are "
                  f"absent from this table, not clean", file=sys.stderr)
            continue
        checks += check_store(Path(root), store=label, held=held,
                              completed_only=not args.all)
    # After every store is read: a builder may sit in a store other than its
    # adopter's, and it need not have a CER of its own.
    checks = resolve_adoptions(checks, [(label, Path(root)) for label, root in stores],
                               held=held)

    counts = summarize(checks)
    if args.json:
        print(json.dumps({"summary": counts,
                          "registry_documents": len(held.documents),
                          "runs": [vars(c) | {"verdict": c.verdict.value} for c in checks]},
                         indent=2, ensure_ascii=False, default=str))
        return 0

    if args.markdown:
        # The table and its count only: the page around it is prose, and
        # redirecting this into the page would replace that prose.
        print(f"Gemessen gegen die heutige Registry ({len(held.documents)} Dokumente, "
              f"{', '.join(sorted(held.sets))}).\n")
        print(HEADER)
        for check in checks:
            print(check.row)
        print()
    else:
        for check in checks:
            print(f"{check.verdict.value:12s} {check.job_id:48s} {check.why}")

    crossed = [c for c in checks if c.verdict is Verdict.CROSSED]
    unknown = [c for c in checks if c.verdict in (Verdict.UNCHECKABLE, Verdict.GONE)]
    print(f"\n{counts['runs']} Läufe: {counts['clean']} unberührt, "
          f"{counts['crossed']} betroffen, {counts['uncheckable']} nicht prüfbar, "
          f"{counts['gone']} unentscheidbar.")
    if crossed:
        print(f"{counts['reserved_pages']} Trainingsseiten liegen heute im Held-out.")
    if unknown:
        print(f"{len(unknown)} Läufe sind KEINE Freigabe — sie sind ungeprüft.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

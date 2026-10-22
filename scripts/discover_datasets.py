#!/usr/bin/env python3
"""What changed in an org's training material since the last run (#32).

    .venvs/kraken-train/bin/python scripts/discover_datasets.py            # dh-unibe
    .venvs/kraken-train/bin/python scripts/discover_datasets.py --dry-run  # write nothing
    .venvs/kraken-train/bin/python scripts/discover_datasets.py \\
        --period 1300 1600 --period 1800 1900                             # two targets

Runs daily from ``deploy/systemd/atr-discovery.timer`` on asteraix, where
``hf auth login`` is done and the gated dh-unibe datasets are readable.

State lives in ``--state-dir`` (default ``~/atr-cache/discovery``, local disk):

    snapshot.json          the memory — what the last plausible run saw
    reports/YYYY-MM-DD.md  written only when something changed
    runs.jsonl             one line per run, changes or not

The first run records a baseline and reports nothing: an inventory is not a
discovery. A run whose listing is implausibly small keeps the old memory and
exits 1, so an outage is never remembered as a mass deletion.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_training.corpus_plan import MEDIEVAL_GERMAN, Target  # noqa: E402
from atr_training.discovery import Snapshot, diff, implausible, render  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--org", default="dh-unibe")
    p.add_argument("--state-dir", type=Path,
                   default=Path.home() / "atr-cache" / "discovery")
    p.add_argument("--period", nargs=2, type=int, action="append", metavar=("FROM", "TO"),
                   help="a target span to score against; repeatable "
                        "(default: 1300 1600, the medieval German profile)")
    p.add_argument("--catalogue", type=Path, default=None,
                   help="read a catalogue JSON (as plan_corpus.py --cache writes, or "
                        "{listed, rows, unreadable}) instead of the Hub")
    p.add_argument("--dry-run", action="store_true", help="print the report, keep no state")
    args = p.parse_args(argv)

    targets = ([Target(period=(a, b)) for a, b in args.period] if args.period
               else [MEDIEVAL_GERMAN])
    current = _fetch(args)
    state = args.state_dir
    snapshot_path = state / "snapshot.json"
    previous = (Snapshot.from_json(json.loads(snapshot_path.read_text(encoding="utf-8")))
                if snapshot_path.is_file() else None)

    problem = implausible(previous, current)
    if problem:
        print(f"discovery: not trusting this run — {problem}; the memory from "
              f"{previous.taken_at if previous else 'nowhere'} is kept", file=sys.stderr)
        _log(state, current, changes=None, note=problem, dry=args.dry_run)
        return 1

    if previous is None:
        print(f"discovery: baseline — {len(current.datasets)} readable dataset(s), "
              f"{len(current.unreadable)} unreadable, in {current.org}. Nothing to "
              "compare yet; the next run reports changes.")
        if not args.dry_run:
            _write(snapshot_path, current.dumps())
        _log(state, current, changes=[], note="baseline", dry=args.dry_run)
        return 0

    changes = diff(previous, current, targets)
    report = render(changes, previous, current)
    if report:
        print(report)
        if not args.dry_run:
            day = time.strftime("%Y-%m-%d")
            _write(state / "reports" / f"{day}.md", report)
    else:
        print(f"discovery: no change in {current.org} since {previous.taken_at}")
    if not args.dry_run:
        _write(snapshot_path, current.carry_forward(previous).dumps())
    _log(state, current, changes=changes, note="", dry=args.dry_run)
    return 0


def _fetch(args) -> Snapshot:
    if args.catalogue:
        data = json.loads(args.catalogue.read_text(encoding="utf-8"))
        if isinstance(data, list):  # plan_corpus.py --cache: readable rows only
            data = {"listed": [r["repo"] for r in data], "rows": data, "unreadable": {}}
        return Snapshot.from_catalogue(args.org, data["listed"], data["rows"],
                                       data.get("unreadable"))
    from atr_training.hub_catalogue import fetch_catalogue  # noqa: PLC0415

    catalogue = fetch_catalogue(args.org)
    return Snapshot.from_catalogue(args.org, catalogue.listed, catalogue.rows,
                                   catalogue.unreadable)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        out.write(text)
    os.replace(tmp, path)


def _log(state: Path, current: Snapshot, *, changes, note: str, dry: bool) -> None:
    if dry:
        return
    state.mkdir(parents=True, exist_ok=True)
    line = {"at": current.taken_at, "org": current.org,
            "readable": len(current.datasets), "unreadable": len(current.unreadable),
            "changes": None if changes is None else len(changes), "note": note}
    with (state / "runs.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Energy against CER on a ladder: what a CER point costs (#333).

    # on UBELIX (login node is fine: no pydantic needed), or any box with the
    # research share mounted
    python3 scripts/energy_per_cer.py --markdown
    python3 scripts/energy_per_cer.py --markdown --filter load_in_4bit=true
    python3 scripts/energy_per_cer.py --json | jq '.[0].steps'
    python3 scripts/energy_per_cer.py --store /scratch/.../runs/jobs --floor 0.3

No GPU, no network, no model: it reads ``job.json`` and ``data/val_eval.jsonl``
of every run in the job stores and groups the completed ones by draw and corpus,
the two conditions under which two CER values compare at all (#108, #98). See
:mod:`atr_training.energy_per_cer` for what a frontier step is and why a run can
be dominated.

``--floor`` is the draw's noise floor in CER points (#115). Without it the table
says so; with it, a step inside the floor is printed as infinite cost rather than
as a number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atr_training.energy_per_cer import as_json, ladders, load_store, markdown  # noqa: E402


def default_stores() -> list[tuple[str, Path]]:
    """``(label, root)`` for every job store this box knows about; unreadable ones included."""
    stores: list[tuple[str, Path]] = []
    try:
        from atr_results_mcp.probe import Paths
        stores += [(label, Path(root)) for label, root in Paths().job_roots]
    except Exception:  # noqa: BLE001 — the probe's own imports are optional here
        pass
    try:
        from atr_training.settings import TrainerSettings
        stores.append(("local", Path(TrainerSettings().jobs_root).expanduser()))
    except Exception:  # noqa: BLE001 — pydantic is absent on the login node
        pass
    seen: dict[Path, tuple[str, Path]] = {}
    for label, root in stores:
        seen.setdefault(root, (label, root))
    return list(seen.values())


def parse_filters(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--filter wants name=value, got {item!r}")
        name, value = item.split("=", 1)
        out[name.strip()] = value.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", action="append", type=Path, default=None,
                        help="a job store root; repeatable. Default: this box's stores.")
    parser.add_argument("--filter", action="append", default=[], metavar="NAME=VALUE",
                        help="keep only runs whose field matches, e.g. load_in_4bit=true, "
                             "granularity=line; repeatable")
    parser.add_argument("--floor", type=float, default=None,
                        help="noise floor of the draw in CER points (#115)")
    parser.add_argument("--markdown", action="store_true", help="the tables for the docs")
    parser.add_argument("--json", action="store_true", help="machine-readable")
    args = parser.parse_args(argv)

    stores = ([(str(p), p) for p in args.store] if args.store else default_stores())
    runs = []
    for label, root in stores:
        if not root.is_dir():
            print(f"store {label} ({root}) is not readable from here — its runs are absent, "
                  "not measured", file=sys.stderr)
            continue
        runs += load_store(root, store=label)
    found = ladders(runs, floor_points=args.floor, filters=parse_filters(args.filter))
    if args.json:
        print(json.dumps([as_json(lad) for lad in found], indent=2))
        return 0
    if not found:
        print("no draw with two measured runs — nothing to put side by side", file=sys.stderr)
        return 1
    for lad in found:
        print(markdown(lad, floor_points=args.floor))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Write a sweep's leaderboard, derived rather than retyped (#116).

    scripts/sweep_leaderboard.py config/sweeps/<name>.yaml
    scripts/sweep_leaderboard.py config/sweeps/<name>.yaml --stdout

Reads the manifest and the state file the driver keeps beside it, and writes
`<manifest>.leaderboard.md`. Markdown, versioned next to the sweep: a table that
exists only as a rendered page cannot be checked in six months, and comparing two
sweeps is then not a diff.

An unfinished sweep produces a valid table with a note saying so — no error and
no empty file. A sweep stopped in the middle of rung 1 is the normal state of a
multi-day run, and it is exactly when somebody wants to look.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_training.leaderboard import render  # noqa: E402
from atr_training.sweep_driver import SweepError, SweepState  # noqa: E402
from atr_training.sweep_manifest import ManifestError, load_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--state", type=Path, default=None,
                        help="default: <manifest>.state.json")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <manifest>.leaderboard.md")
    parser.add_argument("--stdout", action="store_true", help="print instead of writing")
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        print(exc, file=sys.stderr)
        return 1

    state_path = args.state or args.manifest.with_suffix(args.manifest.suffix + ".state.json")
    if not state_path.is_file():
        print(f"no state at {state_path} — this sweep has not run yet, so there is "
              "nothing to rank. `scripts/run_sweep.py` writes it.", file=sys.stderr)
        return 1
    try:
        state = SweepState.load(state_path, manifest, _metric_of(state_path))
    except SweepError as exc:
        print(exc, file=sys.stderr)
        return 1

    table = render(manifest, state)
    if args.stdout:
        print(table, end="")
        return 0
    out = args.out or args.manifest.with_suffix(args.manifest.suffix + ".leaderboard.md")
    out.write_text(table, encoding="utf-8")
    print(f"written to {out}")
    return 0


def _metric_of(state_path: Path) -> str:
    """The metric the sweep actually ran with.

    Read from the state rather than taken as an argument: `SweepState.load`
    refuses a metric that differs from the recorded one, and a leaderboard has no
    business changing what a sweep measured — it reports it.
    """
    import json

    return str(json.loads(state_path.read_text(encoding="utf-8")).get("metric", "benchmark_cer"))


if __name__ == "__main__":
    raise SystemExit(main())

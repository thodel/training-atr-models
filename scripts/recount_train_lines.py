#!/usr/bin/env python3
"""Take the held-out lines back out of ``train_lines`` written before #256.

    # dry run: what each payload and record says, and what it should say
    python scripts/recount_train_lines.py
    # write the corrections (cache payloads first, then terminal job records)
    python scripts/recount_train_lines.py --apply

The cache payloads are the reason this exists: an adopting job copies
``train_lines`` out of them, and the step-count guard divides by it. See
:mod:`atr_training.train_lines_recount` for what is corrected, from what
evidence, and what is never touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atr_training.train_lines_recount import (  # noqa: E402
    Recount,
    apply_to_cache,
    apply_to_job,
    recount_cache_entry,
    recount_job,
)

ADOPTED_PREFIX = " reused, built by "


def default_caches() -> list[Path]:
    from atr_results_mcp.probe import Paths
    from atr_training.settings import TrainerSettings

    roots = [Paths().artefacts_root, Path(TrainerSettings().artefact_cache_root)]
    return list(dict.fromkeys(root.expanduser() for root in roots))


def default_stores() -> list[Path]:
    from atr_results_mcp.probe import Paths
    from atr_training.settings import TrainerSettings

    roots = [root for _, root in Paths().job_roots] + [Path(TrainerSettings().jobs_root)]
    return list(dict.fromkeys(root.expanduser() for root in roots))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", action="append", type=Path, default=None,
                        help="an artefact-cache root; repeatable. Default: this box's.")
    parser.add_argument("--store", action="append", type=Path, default=None,
                        help="a job store root; repeatable. Default: this box's.")
    parser.add_argument("--apply", action="store_true", help="write the corrections")
    parser.add_argument("--json", action="store_true", help="machine-readable")
    args = parser.parse_args(argv)

    rows: list[tuple[Path, Recount]] = []
    by_key: dict[str, Recount] = {}
    for root in args.cache or default_caches():
        if not root.is_dir():
            print(f"cache {root} is not readable from here — its entries are absent, "
                  "not correct", file=sys.stderr)
            continue
        for entry in sorted(p for p in root.iterdir() if (p / "artefact.json").is_file()):
            recount = recount_cache_entry(entry)
            rows.append((entry, recount))
            by_key.setdefault(recount.key, recount)

    jobs: list[Path] = []
    for root in args.store or default_stores():
        if not root.is_dir():
            print(f"store {root} is not readable from here — its runs are absent, "
                  "not correct", file=sys.stderr)
            continue
        jobs += sorted(p for p in root.iterdir() if (p / "job.json").is_file())
    # Builders before adopters: an adopter takes its builder's recount, and the
    # builder of an entry that has left the cache is the only place it exists.
    adopters = [j for j in jobs if ADOPTED_PREFIX in _artefact_of(j)]
    for job_dir in [j for j in jobs if j not in adopters] + adopters:
        recount = recount_job(job_dir, by_key)
        rows.append((job_dir, recount))
        if recount.key and recount.verdict == "correct":
            by_key.setdefault(recount.key, recount)

    if args.apply:
        for path, recount in rows:
            if recount.verdict != "correct":
                continue
            (apply_to_cache if recount.kind == "cache" else apply_to_job)(path, recount)

    # The many runs that reserved nothing are counted below, not listed.
    shown = [r for _, r in rows if r.why != "nothing was reserved"]
    if args.json:
        print(json.dumps([vars(r) | {"agrees": r.agrees} for r in shown], indent=2))
        return 0
    print("| | was | removed | now | pages_train | train.jsonl | agrees | why |")
    print("|---|---:|---:|---:|---:|---:|---|---|")
    for recount in shown:
        print(f"| {recount.kind} `{recount.ident}` {recount.verdict} | {_n(recount.before)} | "
              f"{_n(recount.removed)} | {_n(recount.after)} | {_n(recount.train_pages_lines)} | "
              f"{_n(recount.train_jsonl_rows)} | {_agree(recount.agrees)} | {recount.why} |")
    counts = {verdict: sum(r.verdict == verdict for _, r in rows)
              for verdict in ("correct", "already", "skip")}
    print(f"\n{len(rows)} payloads and records: {counts['correct']} to correct, "
          f"{counts['already']} already right, {counts['skip']} skipped"
          + (" — written." if args.apply else " — dry run, nothing written."))
    return 0


def _artefact_of(job_dir: Path) -> str:
    try:
        record = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str((record.get("progress") or {}).get("artefact") or "")


def _n(value: int | None) -> str:
    return "—" if value is None else f"{value:,}"


def _agree(value: bool | None) -> str:
    return "—" if value is None else ("yes" if value else "**no**")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Recover which image produced each record's number, from the Slurm logs (#206).

    # dry run: what each record carries, what its log says, what stays unknown
    python scripts/runtime_provenance.py --store … --logs ~/ubelix/logs
    # write the recovered runtimes (terminal records only, sidecar first)
    python scripts/runtime_provenance.py --store … --logs ~/ubelix/logs --apply

See :mod:`atr_training.runtime_provenance` for what is written, from what
evidence, and what is left unknown rather than guessed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atr_training.runtime_provenance import apply_to_job, recover_job  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", action="append", type=Path, required=True,
                        help="a job store root; repeatable")
    parser.add_argument("--logs", type=Path, default=Path.home() / "ubelix" / "logs",
                        help="where the Slurm logs are (default: ~/ubelix/logs)")
    parser.add_argument("--apply", action="store_true", help="write the recovered runtimes")
    args = parser.parse_args(argv)

    rows = []
    for root in args.store:
        if not root.is_dir():
            print("store %s is not readable from here — its runs are absent, not unknown"
                  % root, file=sys.stderr)
            continue
        for job_dir in sorted(p for p in root.iterdir() if (p / "job.json").is_file()):
            rows.append((job_dir, recover_job(job_dir, args.logs)))

    if args.apply:
        for job_dir, recovery in rows:
            if recovery.verdict == "recover":
                apply_to_job(job_dir, recovery)

    print("| record | verdict | image | transformers | stages | from |")
    print("|---|---|---|---|---|---|")
    for _job_dir, r in rows:
        if r.verdict == "skip" and "no GPU stage" in r.why:
            continue
        image = Path(r.image).name if r.image else ("—" if r.verdict != "recover" else "no container")
        print("| `%s` | %s | %s | %s | %s | %s |" % (
            r.ident, r.verdict, image, r.transformers or "—", ", ".join(r.stages) or "—",
            Path(r.log).name if r.log else r.why))
    counts = {v: sum(r.verdict == v for _, r in rows) for v in ("recover", "already", "unknown", "skip")}
    print("\n%d records: %d recovered, %d already carry a runtime, %d unknown, %d skipped%s"
          % (len(rows), counts["recover"], counts["already"], counts["unknown"], counts["skip"],
             " — written." if args.apply else " — dry run, nothing written."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

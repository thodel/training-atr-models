#!/usr/bin/env python
"""Draw a pinned page excerpt of the medieval Leuven registers (Itinera Nova).

``dh-unibe/image-text_medieval-scripts_xiv-xv-xvi`` holds 663 ``SAL…`` register
folders, 1350–1800. Only the medieval ``_full`` ones are listed in
``docs/medieval_page_sources/leuven_registers.tsv`` (``period == medieval``);
how they were dated is in ``docs/MEDIEVAL_PAGE_SOURCES.md`` §4. The excerpt
takes ``--per-register`` pages from each, drawn with a fixed seed from the
register's ``filename`` column, plus a few spares because ``materialize`` skips
a page without a transcribed line and ``max_pages`` then stops at the cap.

Writes one ``datasets`` entry per register (``train_projects``, ``pages``,
``max_pages``). With ``--base-spec`` the entries are appended to that spec's
datasets and the whole TrainRequest is written, under ``--model-id``. Reads only
the ``filename`` column of each shard, from the local hub snapshot where there is
one; the hub allows 1,000 requests per five minutes, so hub reads are paced.

    .venvs/kraken-train/bin/python scripts/draw_leuven_excerpt.py \\
        --exclude $(cat docs/medieval_page_sources/leuven_heldout_registers.txt) \\
        --base-spec ubelix/specs/medieval-german-page-qwen35-v1.json \\
        --model-id qwen3.5-4b-medieval-page-leuven-v1 \\
        --out ubelix/specs/medieval-page-qwen35-leuven-v1.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

REPO = "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi"
REVISION = "729e9b2721ba2a31901f776368d8aacf0a5e9406"
REGISTERS = Path(__file__).resolve().parents[1] / "docs/medieval_page_sources/leuven_registers.tsv"


def medieval_registers(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as fh:
        return [r["project"] for r in csv.DictReader(fh, delimiter="\t")
                if r["period"] == "medieval"]


SNAPSHOT = (Path.home() / ".cache/huggingface/hub"
            / f"datasets--{REPO.replace('/', '--')}/snapshots/{REVISION}")


def filenames(project: str, fs, pq) -> list[str]:
    # The local snapshot first (h1 mirrors the full dataset; there the shards are
    # named ``<timestamp>-0000.parquet``), the hub only for a register not there.
    local = sorted((SNAPSHOT / "data/train" / project).glob("*.parquet"))
    if local:
        return [n for f in local
                for n in pq.read_table(f, columns=["filename"]).column(0).to_pylist()]
    path = f"datasets/{REPO}@{REVISION}/data/train/{project}/a-0000.parquet"
    with fs.open(path) as fh:
        return pq.ParquetFile(fh).read(columns=["filename"]).column(0).to_pylist()


def draw(names: list[str], k: int, seed: int, project: str) -> list[str]:
    # One RNG per register, seeded by seed and name: adding or dropping a register
    # leaves every other register's draw as it was.
    rng = random.Random(f"{seed}:{project}")
    unique = sorted(set(names))
    return sorted(rng.sample(unique, min(k, len(unique))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-register", type=int, default=9)
    ap.add_argument("--spares", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="registers held out (whole) for evaluation")
    ap.add_argument("--pause", type=float, default=1.0)
    ap.add_argument("--base-spec", type=Path,
                    help="TrainRequest whose datasets the excerpt is appended to")
    ap.add_argument("--model-id", help="model_id of the written spec (with --base-spec)")
    args = ap.parse_args(argv)

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    entries = []
    projects = [p for p in medieval_registers(REGISTERS) if p not in set(args.exclude)]
    for i, project in enumerate(projects, 1):
        for attempt in range(5):
            try:
                names = filenames(project, fs, pq)
                break
            except Exception as exc:  # rate limit or transient hub error
                print(f"{project}: {exc!r}, retry {attempt + 1}", file=sys.stderr)
                time.sleep(30 * (attempt + 1))
        else:
            print(f"{project}: skipped after 5 attempts", file=sys.stderr)
            continue
        pages = draw(names, args.per_register + args.spares, args.seed, project)
        entries.append({"hf_repo": REPO, "revision": REVISION, "granularity": "page",
                        "train_projects": [project], "pages": pages,
                        "max_pages": args.per_register, "seed": args.seed})
        print(f"[{i}/{len(projects)}] {project}: {len(names)} pages, drew {len(pages)}",
              file=sys.stderr)
        if not (SNAPSHOT / "data/train" / project).is_dir():
            time.sleep(args.pause)

    doc = entries
    if args.base_spec:
        doc = json.loads(args.base_spec.read_text(encoding="utf-8"))
        if args.model_id:
            doc["model_id"] = args.model_id
        doc["datasets"] = doc["datasets"] + entries
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    cap = sum(e["max_pages"] for e in entries)
    print(f"{len(entries)} registers, at most {cap} pages -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Build a line-level dataset from a page-shaped one, and push it private.

The datasets this project trains on are page-shaped: one row per scan, with the
PageXML beside it. The training pipeline cuts the lines itself, so a line-shaped
copy buys the *trainer* nothing — `data-towerbooks-textlines` is read through
`source: line` and the rest through materialize-and-crop, and both routes work.
What a line-shaped copy does buy is everything outside this pipeline: a dataset
anyone can open with `load_dataset` and see one crop with one transcription,
without a PageXML parser, a segmenter, or this repository.

**The crops are this project's own.** The chain here is the chain the prepare
stage runs — `HFPageSource.stream` → `prepare.materialize` → `samples_for` →
`cropping.write_crops` — so a line in the published dataset is byte-identical to
the line the trainer sees. A second implementation would have been a second set of
numbers.

    python ubelix/make_line_dataset.py --repo dh-unibe/image-text_aaeb-xiv-xvii-part-2 \
        --revision ce2739fce41c0b8f8673127525839842411020b0 \
        --out dh-unibe/image-text_aaeb-xiv-xvii-part-2-lines --max-pages 200

`--dry-run` stops before the push and leaves the parquet on disk to inspect.

**Schema.** `image` (JPEG bytes), `text`, `page`, `line_index`, `project`,
`source_repo`, `source_revision`. It is deliberately narrower than
`data-towerbooks-textlines`, which also carries `line_id`, `region_id` and the two
reading orders: those come from the PageXML and this project's `Sample` does not
keep them, so publishing them would mean inventing them.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from loguru import logger

from atr_training.contracts import DatasetSpec
from atr_training.cropping import write_crops
from atr_training.hf_source import data_files_for, expand_all_projects
from atr_training.prepare import HFPageSource, materialize
from atr_training.vlm_dataset import samples_for


def build(repo: str, revision: str, max_pages: int, workdir: Path) -> tuple[list[dict], str]:
    """Crop every transcribed line of ``repo`` and return the rows to publish."""
    spec = DatasetSpec(hf_repo=repo, granularity="line", all_projects=True,
                       max_pages=max_pages, revision=revision)
    spec = expand_all_projects(spec)
    files = data_files_for(spec)
    train_files = files.get("train") or next(iter(files.values()))
    logger.info("{}: {} data file glob(s)", repo, len(train_files))

    rows = HFPageSource(cache=True).stream(repo, train_files, revision=revision)
    pages = materialize(rows, workdir, role="pool", max_pages=max_pages)
    logger.info("{}: {}", repo, pages.summary)
    if not pages.xml_paths:
        return [], pages.summary

    samples = samples_for(pages.xml_paths, "line", root=workdir)
    logger.info("{}: {} line sample(s) before cropping", repo, len(samples))
    crops_dir = workdir / "crops"
    cropped = write_crops(samples, workdir, crops_dir)

    out: list[dict] = []
    for index, sample in enumerate(cropped):
        path = workdir / sample.image
        if not path.is_file():
            continue
        out.append({
            "image": {"bytes": path.read_bytes(), "path": Path(sample.image).name},
            "text": sample.text,
            "page": sample.page or "",
            "line_index": index,
            "project": sample.source or "",
            "source_repo": repo,
            "source_revision": revision,
        })
    return out, pages.summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="the page-shaped source dataset")
    p.add_argument("--revision", required=True, help="full 40-hex SHA of the source")
    p.add_argument("--out", required=True, help="the dataset repo to create, e.g. owner/name-lines")
    p.add_argument("--max-pages", type=int, required=True,
                   help="bound on pages read; DatasetSpec refuses all_projects without one")
    p.add_argument("--dry-run", action="store_true", help="build and report, do not push")
    # The pages and crops land here before they are packed. /tmp on the login node
    # has ~18 GB and `materialize` wants 50 free, so the default is scratch.
    p.add_argument("--workdir", default=None,
                   help="where to materialize (default: $ATR_TRAIN_SCRATCH or "
                        "/scratch/network/users/$USER/line-datasets)")
    args = p.parse_args(argv)

    if len(args.revision) != 40:
        print("--revision must be a full 40-character SHA", file=sys.stderr)
        return 2

    base = Path(args.workdir or os.environ.get("ATR_TRAIN_SCRATCH")
                or "/scratch/network/users/%s/line-datasets" % os.environ.get("USER", "unknown"))
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="line-ds-", dir=str(base)) as tmp:
        rows, summary = build(args.repo, args.revision, args.max_pages, Path(tmp))
        if not rows:
            print(f"no transcribed lines in {args.repo} — nothing to publish ({summary})",
                  file=sys.stderr)
            return 1
        chars = sum(len(r["text"]) for r in rows)
        print(f"\n== {args.repo} -> {args.out}")
        print(f"   {len(rows)} line(s), {chars} characters, "
              f"{sum(len(r['image']['bytes']) for r in rows) / 1e6:.1f} MB of crops")
        print(f"   source: {summary}")
        print(f"   example: {rows[0]['text'][:70]!r}")

        from datasets import Dataset, Features, Image, Value
        features = Features({
            "image": Image(decode=False),
            "text": Value("string"),
            "page": Value("string"),
            "line_index": Value("int32"),
            "project": Value("string"),
            "source_repo": Value("string"),
            "source_revision": Value("string"),
        })
        ds = Dataset.from_list(rows, features=features)
        if args.dry_run:
            local = Path.cwd() / (args.out.split("/")[-1] + ".parquet")
            ds.to_parquet(str(local))
            print(f"   dry run: wrote {local} ({local.stat().st_size / 1e6:.1f} MB), nothing pushed")
            return 0
        ds.push_to_hub(args.out, private=True)
        print(f"   pushed as PRIVATE to https://hf.co/datasets/{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

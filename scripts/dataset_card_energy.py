#!/usr/bin/env python3
"""Render the materialisation-energy section of a dataset card (#331, point 4).

Reads every job store this box knows about (or the ones given) and prints, for
one HuggingFace dataset, the section ``atr_training.dataset_energy`` renders.
Nothing is uploaded unless asked twice::

    # the section alone
    .venvs/kraken-train/bin/python scripts/dataset_card_energy.py \\
        --repo dh-unibe/image-text_kurrent-xix

    # the hub README with the section spliced in, as a diff — writes nothing
    ... --repo dh-unibe/image-text_kurrent-xix --hub

    # upload that README (needs a token with write access to the dataset)
    ... --repo dh-unibe/image-text_kurrent-xix --hub --upload

Publishing to the hub is a public act; ``--upload`` is the one flag that does
it, and it shows the diff first. ``pagexml-hf`` wrote the README's front matter
and summary; the section goes after them between HTML markers and a later run
replaces it.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atr_training.dataset_energy import (  # noqa: E402
    dataset_energy,
    load_jobs,
    markdown_section,
    splice,
)


def default_stores() -> list[tuple[str, Path]]:
    """``(label, root)`` for every job store this box knows about."""
    try:
        from atr_results_mcp.probe import Paths
        return [(label, Path(root)) for label, root in Paths().job_roots]
    except Exception:  # noqa: BLE001 — off the boxes there is no default
        return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="the dataset, e.g. dh-unibe/image-text_kurrent-xix")
    ap.add_argument("--store", action="append", default=[], metavar="LABEL=PATH",
                    help="a job store root; repeatable; default: the box's stores")
    ap.add_argument("--readme", type=Path, help="a local README.md to splice into (printed, not written)")
    ap.add_argument("--hub", action="store_true", help="fetch the dataset's README from the hub and show the diff")
    ap.add_argument("--upload", action="store_true", help="with --hub: upload the spliced README")
    args = ap.parse_args(argv)

    stores = [tuple(s.split("=", 1)) for s in args.store] or default_stores()
    if not stores:
        print("no job store given and none known on this box: pass --store LABEL=PATH", file=sys.stderr)
        return 2
    jobs = []
    for label, root in stores:
        jobs += load_jobs(Path(root), store=str(label))
    energy = dataset_energy(jobs, args.repo)
    section = markdown_section(energy, generated=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    if not energy.rows:
        print(f"no materialisation of {args.repo} in {len(jobs)} job records", file=sys.stderr)

    if args.readme:
        print(splice(args.readme.read_text(encoding="utf-8"), section))
        return 0
    if not args.hub:
        print(section)
        return 0

    from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415 — optional, engine-venv only
    api = HfApi()
    current = Path(hf_hub_download(args.repo, "README.md", repo_type="dataset")).read_text(encoding="utf-8")
    spliced = splice(current, section)
    diff = difflib.unified_diff(current.splitlines(), spliced.splitlines(),
                                "hub/README.md", "spliced/README.md", lineterm="")
    print("\n".join(diff) or "(no change)")
    if not args.upload:
        print("\n-- dry run: nothing uploaded; pass --upload to write this README to the hub", file=sys.stderr)
        return 0
    if spliced == current:
        return 0
    api.upload_file(path_or_fileobj=spliced.encode("utf-8"), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset",
                    commit_message="Energy of materialising this dataset (training-atr-models #331)")
    print(f"uploaded README.md to {args.repo}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

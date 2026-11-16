#!/usr/bin/env python3
"""A published benchmark project as an evaluation set on disk (agentic_historian#875).

The Federal Council lines (Hodel & Schoch 2021, doi:10.5281/zenodo.4746342,
``dh-unibe/image-text_federal-minutes-testset`` / ``TEST_federal_minutes``) were
scored from ``$SCRATCH/runs/jobs/evalset-federal-minutes``: ``data/val.jsonl`` and
the crops it names, with ``DATA_ROOT`` the directory itself. That directory was
**hand-built** (serving-atr-inference/docs/UBELIX_PLAN.md §21: "Lines were cut
with the service's own ``samples_for`` + ``write_crops``"; §F: "a hand-built
evaluation directory, a one-off sbatch"), never tracked, and lost to the 30-day
scratch purge in November 2026.

This rebuilds it the way the trainer builds a benchmark beside a run
(``BasePipeline._benchmark_pages`` + ``vlm_train_svc.runner.Pipeline._benchmark_jsonl``):
``materialize`` the project's pages, ``samples_for`` at line granularity,
``write_crops`` at native pixels, ``write_jsonl``. The same functions, not a
second pipeline, so a CER on it is comparable with ``benchmark_cer`` on a card.
No length filter, as there: dropping the hardest lines would flatter the score.

    python scripts/materialize_benchmark.py \\
        --hf-repo dh-unibe/image-text_federal-minutes-testset --project TEST_federal_minutes \\
        --out /storage/research/wbkolleg_dh_1/Textrecognition_Training/eval_sets/federal-minutes \\
        --expect-lines 2751

Layout under ``--out`` (what ``DRAW_FILE=$E/data/val.jsonl DATA_ROOT=$E`` expects):

* ``data/val.jsonl`` — one ``{image, text, source_type, page}`` row per line,
  ``image`` relative to ``--out``;
* ``data/crops/benchmark_<project>/`` — the line crops;
* ``data/benchmarks/<project>/`` — the materialised pages (image + PageXML);
* ``data/pages_benchmark_<project>.lst`` — the page manifest;
* ``evalset.json`` — where it came from: repo, project, split, the resolved
  revision, line and character counts, the md5 of ``val.jsonl``, the code commit.

Needs ``datasets`` (the vlm-train image; ``ubelix/materialize_benchmark.sbatch``).
Refuses an ``--out`` that already holds a ``val.jsonl``: an evaluation set other
reports were scored on is moved aside by hand, not overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atr_training.cropping import write_crops  # noqa: E402
from atr_training.hf_source import project_glob  # noqa: E402
from atr_training.manifests import write_manifest  # noqa: E402
from atr_training.prepare import materialize  # noqa: E402
from atr_training.vlm_dataset import samples_for, write_jsonl  # noqa: E402

FEDERAL_MINUTES_REPO = "dh-unibe/image-text_federal-minutes-testset"
FEDERAL_MINUTES_PROJECT = "TEST_federal_minutes"
#: The dataset's own file names carry the Transkribus document:
#: ``<n>_page_<pageId>_line_<lineId>_docId_<docId>.jpg`` (dataset card).
_DOC_ID = re.compile(r"docId_(\d+)")


def resolve_revision(hf_repo: str, revision: str | None) -> str | None:
    """The commit a branch name or ``None`` points at today, so the record names
    what was read. ``None`` when the hub cannot be asked — then the record says
    the revision is unknown rather than guessing."""
    if revision and re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    try:
        from huggingface_hub import HfApi  # lazy: the vlm-train image has it, CI need not
        return HfApi().dataset_info(hf_repo, revision=revision).sha
    except Exception as exc:  # noqa: BLE001 — any failure means "not known"
        print(f"!! could not resolve the revision of {hf_repo} ({exc}); "
              "evalset.json records it as unknown", file=sys.stderr)
        return None


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_benchmark(source, out: Path, *, hf_repo: str, project: str,
                          split: str = "train", revision: str | None = None,
                          min_free_disk_gb: float = 5.0) -> dict:
    """Build the evaluation set under ``out`` from ``source`` (a
    :class:`atr_training.prepare.PageSource`) and return the record written to
    ``out/evalset.json``."""
    out = Path(out)
    data = out / "data"
    val = data / "val.jsonl"
    if val.exists():
        raise SystemExit(f"{val} exists: move the old set aside before building it again")

    files = [project_glob(split, project)]
    prepared = materialize(source.stream(hf_repo, files, revision),
                           data / "benchmarks" / project, role="eval",
                           min_free_disk_gb=min_free_disk_gb)
    pages = [str(p) for p in prepared.xml_paths]
    write_manifest(data / f"pages_benchmark_{project}.lst", pages)
    samples = samples_for(pages, "line", root=out)
    samples = write_crops(samples, out, data / "crops" / f"benchmark_{project}")
    lines = write_jsonl(val, samples)
    if not lines:
        raise SystemExit(f"{hf_repo}/{project}: {prepared.pages_written} page(s) and no "
                         "usable line — nothing to evaluate on")

    documents = {m.group(1) for s in samples if (m := _DOC_ID.search(s.page or ""))}
    try:
        from atr_training.codeversion import current_code
        code = current_code().commit
    except Exception:  # noqa: BLE001 — provenance is best effort, never fatal
        code = None
    record = {
        "hf_repo": hf_repo, "project": project, "split": split, "revision": revision,
        "data_files": files,
        "pages_written": prepared.pages_written, "pages_skipped": prepared.pages_skipped,
        "wide_lines_dropped": prepared.wide_lines,
        "lines": lines, "chars": sum(len(s.text) for s in samples),
        "documents": len(documents) or None,
        "val_jsonl": "data/val.jsonl", "val_jsonl_md5": _md5(val),
        "granularity": "line", "crop_scale": None,
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": code,
        "use": "DRAW_FILE=<this>/data/val.jsonl DATA_ROOT=<this>",
    }
    (out / "evalset.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    return record


def main(argv: list[str] | None = None, source=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-repo", default=FEDERAL_MINUTES_REPO)
    ap.add_argument("--project", default=FEDERAL_MINUTES_PROJECT)
    ap.add_argument("--split", default="train",
                    help="the split the project lies under, data/<split>/<project>/")
    ap.add_argument("--revision", default=None,
                    help="branch, tag or commit; resolved to a commit before reading")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expect-lines", type=int, default=None,
                    help="fail unless exactly this many lines were written "
                         "(2751 for the Federal Council set)")
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    args = ap.parse_args(argv)

    if source is None:
        from atr_training.prepare import HFPageSource
        source = HFPageSource(cache=True)
        revision = resolve_revision(args.hf_repo, args.revision)
    else:
        revision = args.revision
    record = materialize_benchmark(source, args.out, hf_repo=args.hf_repo, project=args.project,
                                   split=args.split, revision=revision,
                                   min_free_disk_gb=args.min_free_gb)
    print(json.dumps(record, indent=2, ensure_ascii=False))
    if args.expect_lines is not None and record["lines"] != args.expect_lines:
        print(f"!! {record['lines']} lines, expected {args.expect_lines}: the set is not the "
              "published one — compare evalset.json with the dataset card before scoring",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

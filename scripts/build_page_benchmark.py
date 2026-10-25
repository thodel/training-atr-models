#!/usr/bin/env python3
"""A held-out page set as a VLM draw: proven against the registry, then frozen (#301).

`german-medieval-v1` exists as two kraken arrows and as page lists in the pool
job; a VLM page arm can read neither. It reads a draw — one JSON sample per page,
``image``/``text``/``source_type``/``page`` (``vlm_dataset.Sample``) — with the
images under a root it is given. This writes that draw for one side of a set,
and refuses to do so unless the pages are the set's:

* every **document** of the list is one the registry records for that side, and
  none is missing (``config/heldout_eval_documents.json``);
* the list's **page digest** equals the recorded one. The digest is pinned for
  `german-medieval-v1` (``7a3f839ff97da4fd`` for test), so a list that holds the
  right documents but other pages of them is refused too
  (``split_identity`` explains why the two differ).

Then the pages are **copied**: XML and JPEG side by side under
``<out>/data/pages/``, the draw as ``<out>/pages.jsonl`` with paths relative to
``<out>``, and ``<out>/benchmark.json`` with what was checked and counted. The
copy is the point. The page pool is a job directory, and job directories are
cleaned (#183) — which is how the arrows' first copy was lost. Put ``--out``
under ``eval_sets/``, outside the jobs tree.

Scoring is the next step and a different script:

    python ubelix/rescore_on_shared_draw.py --job <id> --heldout \\
        --draw-file <out>/pages.jsonl --data-root <out> --label german-medieval-v1-test

``--heldout`` is what turns that into a held-out number: it refuses when the
scored run trained on any document of the draw.

    python scripts/build_page_benchmark.py --set german-medieval-v1 --side test \\
        --pages-list <pool job>/data/split/pages_test.lst \\
        --pages-dir  <pool job>/data/pages \\
        --out        <share>/eval_sets/german-medieval-v1/vlm-pages-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_training.split_identity import (  # noqa: E402
    documents_of, load_identity, page_digest, page_key,
)
from atr_training.vlm_dataset import page_sample, write_jsonl  # noqa: E402

SIDES = ("test", "val")


class BenchmarkError(RuntimeError):
    """The list is not the set it claims to be, or its pages are not all there."""


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def check_identity(pages: list[str], *, name: str, side: str,
                   registry: Path | None = None) -> dict:
    """Documents and page digest of *pages* against the registry's *side*.

    Raises unless both match. A set without a recorded digest is refused rather
    than accepted on documents alone: the frozen draw is what every later number
    cites, and a draw that only half matches would carry the name anyway.
    """
    identity = load_identity(name, registry)
    expected_docs = identity.test_documents if side == "test" else identity.val_documents
    expected_digest = identity.test_page_digest if side == "test" else identity.val_page_digest

    got_docs = documents_of(pages)
    missing = sorted(expected_docs - got_docs)
    extra = sorted(got_docs - expected_docs)
    problems = []
    if missing:
        problems.append(f"{len(missing)} recorded {side} documents are not in the list "
                        f"({', '.join(missing[:5])}{', …' if len(missing) > 5 else ''})")
    if extra:
        problems.append(f"{len(extra)} documents of the list were never {side} documents "
                        f"of {name} ({', '.join(extra[:5])}{', …' if len(extra) > 5 else ''})")
    digest = page_digest(pages)
    if not expected_digest:
        problems.append(f"{name} records no {side} page digest, so the pages cannot be "
                        "proven; pin one with scripts/restore_eval_split.py --record first")
    elif digest != expected_digest:
        problems.append(f"page digest {digest} is not the recorded {expected_digest}: "
                        "the same documents, other pages of them")
    if problems:
        raise BenchmarkError(f"{name}/{side}:\n  " + "\n  ".join(problems))
    return {"documents": len(got_docs), "pages": len({page_key(p) for p in pages}),
            "page_digest": digest}


def copy_pages(pages: list[str], pages_dir: Path, out: Path) -> list[Path]:
    """Copy each page's XML and its sibling JPEG to ``<out>/data/pages/``.

    Every page must be there with both files: a draw with a page silently
    missing is a different set under the right name. Already-copied files of the
    same size are kept, so an interrupted copy resumes.
    """
    dest = out / "data" / "pages"
    dest.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    copied: list[Path] = []
    for entry in sorted({page_key(p) for p in pages}):
        xml = pages_dir / entry
        jpg = xml.with_suffix(".jpg")
        if not xml.is_file() or not jpg.is_file():
            missing.append(entry)
            continue
        for src in (xml, jpg):
            target = dest / src.name
            if not (target.is_file() and target.stat().st_size == src.stat().st_size):
                shutil.copy2(src, target)
        copied.append(dest / xml.name)
    if missing:
        raise BenchmarkError(
            f"{len(missing)} pages of the list have no XML+JPEG pair under {pages_dir} "
            f"({', '.join(missing[:5])}{', …' if len(missing) > 5 else ''})")
    return copied


def build(*, name: str, side: str, pages_list: Path, pages_dir: Path, out: Path,
          registry: Path | None = None) -> dict:
    pages = read_list(pages_list)
    identity = check_identity(pages, name=name, side=side, registry=registry)
    xmls = copy_pages(pages, pages_dir, out)

    samples, empty = [], []
    for xml in xmls:
        sample = page_sample(xml, root=out)
        if sample is None:
            empty.append(xml.name)
        else:
            samples.append(sample)
    draw = out / "pages.jsonl"
    write_jsonl(draw, samples)

    record = {
        "set": name,
        "side": side,
        **identity,
        "samples": len(samples),
        # A page with no transcription is not a sample (page_sample's rule, the
        # same one prepare applies). Named, so 695 pages and fewer samples is
        # an explained difference rather than a lost page.
        "pages_without_text": empty,
        "reference_chars": sum(len(s.text) for s in samples),
        "draw": draw.name,
        "draw_md5": hashlib.md5(draw.read_bytes()).hexdigest(),
        "source_list": str(pages_list),
        "source_pages": str(pages_dir),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "benchmark.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", required=True, dest="name", help="a set in the registry")
    p.add_argument("--side", required=True, choices=SIDES)
    p.add_argument("--pages-list", required=True, type=Path,
                   help="the side's page list, e.g. <pool job>/data/split/pages_test.lst")
    p.add_argument("--pages-dir", required=True, type=Path,
                   help="where the listed pages' XML and JPEG are")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--registry", type=Path, default=None)
    args = p.parse_args(argv)
    try:
        record = build(name=args.name, side=args.side, pages_list=args.pages_list,
                       pages_dir=args.pages_dir, out=args.out, registry=args.registry)
    except BenchmarkError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(f"{record['set']}/{record['side']}: {record['documents']} documents, "
          f"{record['pages']} pages, {record['samples']} samples, "
          f"{record['reference_chars']} reference chars, digest {record['page_digest']}")
    if record["pages_without_text"]:
        print(f"  {len(record['pages_without_text'])} pages without text, not in the draw")
    print(f"  draw: {args.out / record['draw']} (md5 {record['draw_md5']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

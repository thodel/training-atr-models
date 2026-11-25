#!/usr/bin/env python3
"""The lines of a frozen page set, as a line arm's draw (agentic_historian#903, #59).

A frozen held-out set (``eval_sets/<set>/vlm-pages-<side>/``, written by
``scripts/build_page_benchmark.py``) holds whole pages. A line arm scored on it
through ``ubelix/rescore_on_shared_draw.py`` is handed each **page** at its line
pixel budget — the evaluator reads every row's ``image`` as it is — and the CER
that comes back measures nothing the arm was trained to do. Its own corpus cannot
help: the set's documents are reserved, so no line of them is in it, and
``scripts/lines_of_pages.py`` finds nothing to join.

This cuts the lines of the set's own pages, by the rule ``granularity: line``
trains on (``vlm_dataset.line_samples``: padded boxes clamped to the scan,
implausible and empty lines dropped) — the rule ``scripts/eval_granularity.py
--page-set`` cuts its line level by, so the two read the same lines — and writes
them the way a run's compile does (``cropping.write_crops`` at native pixels):

* ``<out>/data/crops/NNNNNNN.jpg`` — the line crops;
* ``<out>/lines.jsonl`` — one ``{image, text, source_type, page}`` row per line,
  ``image`` relative to ``<out>``, ``page`` the PageXML's path inside the page set
  (``data/pages/<name>.xml``): its name carries the document, which is what
  ``rescore_on_shared_draw.py --heldout`` compares with the run's training side;
* ``<out>/benchmark.json`` — the page set it was cut from (path, ``draw_md5``,
  page digest), what was cut and what was not, the draw's md5, the code commit.

It refuses a page set whose draw is not the one its ``benchmark.json`` froze, an
``--out`` inside the page set (that set is frozen), and an ``--out`` that already
holds a ``lines.jsonl``: a draw reports were scored on is moved aside by hand,
not overwritten. Nothing about the page set is changed.

    python scripts/build_line_benchmark.py \\
        --page-set <share>/eval_sets/german-kurrent-xvi-xix-v1/vlm-pages-test \\
        --out      <share>/eval_sets/german-kurrent-xvi-xix-v1/vlm-lines-test

and then, for a line arm:

    JOB=<id> DRAW_FILE=<out>/lines.jsonl DATA_ROOT=<out> \\
        LABEL=german-kurrent-xvi-xix-v1-test-lines HELDOUT=1 \\
        ubelix/submit.sh ubelix/rescore_on_shared_draw.sbatch -- --time=<…>

Needs PIL and the trainer's package (the vlm-train image on UBELIX).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from atr_training.cropping import write_crops  # noqa: E402
from atr_training.vlm_dataset import DEFAULT_LINE_PAD, line_samples, write_jsonl  # noqa: E402
from scripts.eval_granularity import PageSetError, read_page_set  # noqa: E402

DRAW = "lines.jsonl"


class LineBenchmarkError(RuntimeError):
    """The page set or the output directory is not one this may write from or to."""


def _code_commit() -> str | None:
    try:
        from atr_training.codeversion import current_code  # noqa: PLC0415
        return current_code().commit
    except Exception:  # noqa: BLE001 — a record without the commit beats no record
        return None


def cut_lines(page_set: Path, pages: list[str]) -> tuple[list, list[str]]:
    """The line samples of ``pages``, in page order, and the pages that gave none.

    The page size is the scan's own, read from the JPEG — as
    ``eval_granularity.py`` does — not the one the PageXML declares: a page that
    overstates its dimensions would otherwise keep boxes past the real edge.
    Images are absolute here so the crops can go outside the page set; ``page``
    is the path inside it.
    """
    from PIL import Image  # noqa: PLC0415 — the image has it, the gateway venv too

    samples, without = [], []
    for page in pages:
        xml = (page_set / page).resolve()
        with Image.open(xml.with_suffix(".jpg")) as im:
            size = im.size
        cut = line_samples(xml, root=None, page_size=size)
        if not cut:
            without.append(page)
        samples.extend(dataclasses.replace(s, page=page) for s in cut)
    return samples, without


def build(*, page_set: Path, out: Path) -> dict:
    page_set = Path(page_set)
    out = Path(out)
    try:
        pages, identity = read_page_set(page_set)
    except PageSetError as exc:
        raise LineBenchmarkError(str(exc)) from exc
    if out.resolve().is_relative_to(page_set.resolve()):
        raise LineBenchmarkError(f"{out} is inside the page set {page_set}, which is frozen: "
                                 "put the lines beside it (vlm-lines-<side>)")
    draw = out / DRAW
    if draw.exists():
        raise LineBenchmarkError(f"{draw} exists: move the old draw aside before building "
                                 "it again — reports may have been scored on it")
    page_record = json.loads((page_set / "benchmark.json").read_text(encoding="utf-8"))

    samples, without = cut_lines(page_set, pages)
    crops = write_crops(samples, out, out / "data" / "crops")
    write_jsonl(draw, crops)

    page_chars = page_record.get("reference_chars")
    line_chars = sum(len(s.text) for s in crops)
    record = {
        "set": identity.get("set"),
        "side": identity.get("side"),
        "level": "line",
        "rule": f"vlm_dataset.line_samples (pad {DEFAULT_LINE_PAD}, page size from the scan), "
                "cropping.write_crops at native pixels",
        "page_set": {"path": str(page_set), **identity},
        "pages": len(pages),
        "pages_with_lines": len(pages) - len(without),
        # A page with text and no line that survives the rule is named, so a
        # smaller line count than the page set suggests is an explained one.
        "pages_without_lines": without,
        "samples": len(crops),
        "lines_dropped_at_crop": len(samples) - len(crops),
        "reference_chars": line_chars,
        # Line references leave out the breaks and the lines the rule drops; the
        # page set's own count is beside it so the coverage is visible.
        "page_reference_chars": page_chars,
        "draw": DRAW,
        "draw_md5": hashlib.md5(draw.read_bytes()).hexdigest(),
        "code_commit": _code_commit(),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "benchmark.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--page-set", type=Path, required=True,
                   help="a frozen page set: eval_sets/<set>/vlm-pages-<side>")
    p.add_argument("--out", type=Path, default=None,
                   help="default: vlm-lines-<side> beside the page set")
    args = p.parse_args(argv)
    out = args.out or args.page_set.parent / args.page_set.name.replace("vlm-pages-", "vlm-lines-", 1)
    if out == args.page_set:
        print("refused: give --out; the page set's name does not start with vlm-pages-",
              file=sys.stderr)
        return 1
    try:
        record = build(page_set=args.page_set, out=out)
    except LineBenchmarkError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(f"{record['set']}/{record['side']} lines: {record['samples']} lines from "
          f"{record['pages_with_lines']} of {record['pages']} pages, "
          f"{record['reference_chars']} reference chars "
          f"(the pages hold {record['page_reference_chars']})")
    if record["pages_without_lines"]:
        print(f"  {len(record['pages_without_lines'])} pages gave no line: "
              f"{', '.join(record['pages_without_lines'][:5])}")
    if record["lines_dropped_at_crop"]:
        print(f"  {record['lines_dropped_at_crop']} lines dropped at the crop (box off the scan)")
    print(f"  draw: {out / DRAW} (md5 {record['draw_md5']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

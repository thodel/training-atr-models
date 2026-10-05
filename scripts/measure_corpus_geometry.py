#!/usr/bin/env python3
"""Line geometry of a page corpus, before a run pays for it (#164, #145).

Two questions, one parse, no GPU and no full download:

* **Does the aspect ceiling bind?** kraken pads every batch to its widest line,
  so one outlier is paid for by every line in its batch — the mechanism behind
  four CUDA OOMs and the 11,318 lines cut out of the medieval sweep corpus
  (#145, #113). `summarise` answers it in the form `apply_line_ceiling.py` uses.
* **Does a VGSL spec leave CTC enough frames?** `aspect_per_char` is
  width/(height*characters) at the p10, scale-free, so one measurement serves
  every candidate input height (#91). The medieval p10 is 0.246 and the default
  everywhere; a corpus of 19th-century protocol hands has no reason to match it.

Run it in the engine venv (`datasets` is there, not in the service venv):

    .venvs/vlm-train/bin/python scripts/measure_corpus_geometry.py \
        --repo dh-unibe/image-text_zh-regierungsratsprotokolle \
        --all-projects --pages 300

    .venvs/vlm-train/bin/python scripts/measure_corpus_geometry.py \
        --repo dh-unibe/image-text_kurrent-xix --pages 300 \
        --projects TRAIN_CITlab_Steiner,TRAIN_CITlab_Suppes

Streaming, never cached: a sample of 300 pages must not pull shards of a 15.6 TB
dataset onto the disk. `--cache` turns that off when a corpus is wanted locally
anyway.

The process may die at teardown with a `PyGILState_Release` fatal error from
`datasets`/pyarrow — observed on `kurrent-xix`, after the report was printed and
written. `"complete": true` in the JSON is what says the measurement finished;
the exit code of that process does not.

The boxes are the PageXML ones, which is the honest half of the measurement:
#145 is open precisely because kraken pads the *extracted* line image, whose
aspect exceeded the box's 60:1 at 177:1 on the medieval corpus. A box tail that
is already over the ceiling settles the question; one that is under it does not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_training.contracts import KRAKEN_PLUS_SPEC, DatasetSpec  # noqa: E402
from atr_training.hf_source import data_files_for  # noqa: E402
from atr_training.line_ceiling import summarise  # noqa: E402
from atr_training.pagexml import MAX_LINE_ASPECT, line_boxes, line_texts  # noqa: E402
from atr_training.prepare import HFPageSource  # noqa: E402
from atr_training.vgsl_geometry import (  # noqa: E402
    MEDIEVAL_ASPECT_PER_CHAR_P10,
    aspect_per_char,
    check_line_geometry,
)


def samples(repo: str, files: list[str], pages: int, cache: bool, revision: str | None):
    """``(aspects, per_char_samples, pages_read, untranscribed_lines)``.

    The untranscribed count is informational and deliberately not in the tail:
    those lines have geometry and land in a batch like any other, but kraken's
    compile drops them (`--skip-empty-lines`), so they cost nothing at training
    time and would overstate the ceiling's reach if counted.
    """
    source = HFPageSource(cache=cache)
    aspects: list[float] = []
    per_char: list[tuple[float, float, int]] = []
    read = 0
    untranscribed = 0
    for row in source.stream(repo, files, revision=revision):
        xml = row.get("xml_content") or row.get("xml")
        if not xml:
            continue
        read += 1
        # `line_boxes` already yields only TRANSCRIBED lines, each carrying its
        # own text — so the text comes off the box rather than from a second,
        # differently filtered list. Zipping `line_texts` against it by position
        # would pair box i with the i-th of ALL TextLines, which is the wrong
        # line as soon as one is untranscribed.
        boxes = line_boxes(xml)
        untranscribed += sum(1 for text in line_texts(xml) if not text.strip())
        for box in boxes:
            width = float(box.width)
            height = float(box.height)
            if width <= 0 or height <= 0:
                continue
            aspects.append(width / height)
            per_char.append((width, height, len(box.text)))
        if read >= pages:
            break
    return aspects, per_char, read, untranscribed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--projects", default="", help="comma-separated project names")
    ap.add_argument("--all-projects", action="store_true")
    ap.add_argument("--split", default="train")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--pages", type=int, default=300)
    ap.add_argument("--ceiling", type=float, default=MAX_LINE_ASPECT)
    ap.add_argument("--cache", action="store_true", help="keep the shards (default: stream)")
    ap.add_argument("--spec", action="append", default=[],
                    help="VGSL spec to judge against this material (repeatable)")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    projects = [p for p in args.projects.split(",") if p]
    if bool(projects) == args.all_projects:
        print("Pass exactly one of --projects or --all-projects: the contract "
              "forbids both together, and neither selects nothing.", file=sys.stderr)
        return 2
    spec = DatasetSpec(hf_repo=args.repo, split=args.split, max_pages=args.pages,
                       all_projects=args.all_projects,
                       train_projects=projects or [])
    files = data_files_for(spec)
    chosen = files.get(args.split) or next(iter(files.values()))

    aspects, per_char, pages, untranscribed = samples(
        args.repo, chosen, args.pages, args.cache, args.revision)
    if not aspects:
        print(f"no line geometry in the first {args.pages} page(s) of {args.repo}",
              file=sys.stderr)
        return 1

    tail = summarise(aspects, ceiling=args.ceiling)
    p10 = aspect_per_char(per_char, percentile=10)
    p50 = aspect_per_char(per_char, percentile=50)
    verdicts = [check_line_geometry(s, p10)
                for s in (args.spec or [KRAKEN_PLUS_SPEC])]

    report = {
        # Written before the interpreter exits, and marked, because `datasets`
        # streaming has died at teardown here with
        # "PyGILState_Release: thread state must be current when releasing" —
        # AFTER the report was complete. A consumer must be able to tell a
        # finished measurement from a truncated one without trusting the exit
        # code of a process that crashed on the way out.
        "complete": True,
        "repo": args.repo,
        "projects": projects or "all",
        "pages_read": pages,
        "lines": tail.lines,
        "lines_without_text": untranscribed,
        "box_aspect": {"median": round(tail.median, 2), "p99": round(tail.p99, 2),
                       "max": round(tail.maximum, 2),
                       "over_ceiling": tail.over_ceiling,
                       "share_over": round(tail.share_over, 5),
                       "ceiling": tail.ceiling, "state": tail.state},
        "aspect_per_char": {"p10": round(p10, 4), "p50": round(p50, 4),
                            "medieval_p10": MEDIEVAL_ASPECT_PER_CHAR_P10},
        "specs": [{"input_height": v.input_height, "width_stride": v.width_stride,
                   "frames_per_char": round(v.frames_per_char, 2),
                   "severity": v.severity, "reason": v.reason} for v in verdicts],
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text, flush=True)
    print(f"\n{tail.describe()}", flush=True)
    print(f"aspect_per_char p10 {p10:.4f} (medieval {MEDIEVAL_ASPECT_PER_CHAR_P10}), "
          f"p50 {p50:.4f}", flush=True)
    for verdict in verdicts:
        print(f"[{verdict.severity}] {verdict.reason}", flush=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

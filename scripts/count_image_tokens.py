#!/usr/bin/env python3
"""Count the image tokens a real line and a real page cost, per base and budget (#278).

    # in the venv or container the run uses — processors only, no weights
    python scripts/count_image_tokens.py
    python scripts/count_image_tokens.py Qwen/Qwen3-VL-4B-Instruct google/gemma-4-E4B-it
    python scripts/count_image_tokens.py --line my_line.jpg --page my_page.jpg

For every model: a fresh processor per granularity, set with
``apply_visual_budget`` to that granularity's entry in ``VLM_PIXEL_BUDGET``, then
the two fixture images (``tests/fixtures/image_tokens/``: a line crop cut by
``cropping.write_crops`` and a page fitted to the page budget, both from
``dh-unibe/image-text_rats-und-richtebuecher_xv-xvi``) encoded the way the
training collator encodes them, and the image tokens counted twice — the
placeholders in ``input_ids`` and, where the processor reports a grid, what the
grid implies. ``assumed`` is what ``apply_visual_budget`` said the budget buys:
a ceiling for Qwen-style processors, the price of the step for Gemma.

Measured 23.10.2026 with transformers 5.17.0 (CPU, no weights):

    model                     budget  image  fitted     assumed  tokens
    Qwen/Qwen3-VL-4B-Instruct line    line   631x178    256      120
    Qwen/Qwen3-VL-4B-Instruct line    page   455x576    256      252
    Qwen/Qwen3-VL-4B-Instruct block   page   910x1152   1024     1008
    Qwen/Qwen3-VL-4B-Instruct page    page   1287x1629  2048     2040
    Qwen/Qwen3.5-4B           (identical to Qwen3-VL-4B: same image processor)
    allenai/olmOCR-2-7B-1025  line    line   631x178    334      138
    allenai/olmOCR-2-7B-1025  line    page   455x576    334      320
    allenai/olmOCR-2-7B-1025  block   page   910x1152   1337     1312
    allenai/olmOCR-2-7B-1025  page    page   1287x1629  2674     2668
    google/gemma-4-E4B-it     line    line   631x178    140      132
    google/gemma-4-E4B-it     line    page   455x576    140      130
    google/gemma-4-E4B-it     block   line   631x178    560      528
    google/gemma-4-E4B-it     block   page   910x1152   560      546
    google/gemma-4-E4B-it     page    line   631x178    1120     1071
    google/gemma-4-E4B-it     page    page   1287x1629  1120     1073

What the deviations are:

* Qwen-style: the budget is honoured as a **ceiling** — the page lands within
  1 % of it at every granularity (252/256, 1008/1024, 2040/2048) because the
  resize rounds both edges down to a multiple of the 32 px cell. A line crop of
  112 000 pixels is under the line budget and costs 120, not 256: ``fit_pixels``
  never scales up, so the budget is the most a sample may cost, not what it does.
* olmOCR-2 (Qwen2.5-VL) has a 28 px cell, not 32: the **same pixels cost 30 %
  more tokens** (334 for the line budget, not 256). ``apply_visual_budget``
  reads that off the processor; the 256 in ``contracts.py`` is Qwen3's figure.
* Gemma 4 charges the **step, whatever the image**: a line at the line budget
  costs 132, a page 130, and the same line at the page budget costs 1071. The
  pooled grid is a few tokens under the step because its two edges are rounded
  down separately. Per-sample fitting does not make a line cheaper on Gemma —
  see the ``stepped`` note on ``AppliedBudget``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_training.image_tokens import measure  # noqa: E402

DEFAULT_MODELS = (
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3.5-4B",
    "allenai/olmOCR-2-7B-1025",
    "google/gemma-4-E4B-it",
)
FIXTURES = ROOT / "tests" / "fixtures" / "image_tokens"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="*", default=list(DEFAULT_MODELS),
                        help="Hub ids or local checkpoints (processor files suffice)")
    parser.add_argument("--line", type=Path, default=FIXTURES / "line.jpg")
    parser.add_argument("--page", type=Path, default=FIXTURES / "page.jpg")
    parser.add_argument("--prompt", default="Transcribe.")
    parser.add_argument("--local-files-only", action="store_true",
                        help="never download: use the cache or fail")
    args = parser.parse_args(argv)

    import transformers
    from PIL import Image
    from transformers import AutoProcessor

    images = {"line": Image.open(args.line), "page": Image.open(args.page)}
    for label, image in images.items():
        print(f"{label}: {image.size[0]}x{image.size[1]} = {image.size[0] * image.size[1]} px")
    print(f"transformers {transformers.__version__}\n")
    print("| model | budget | image | fitted | assumed | tokens | from grid | patches | seq |")
    print("|---|---|---|---|---|---|---|---|---|")
    failures = 0
    for model in args.models:
        def load(model=model):
            return AutoProcessor.from_pretrained(model, local_files_only=args.local_files_only)
        try:
            rows = measure(load, images, prompt=args.prompt)
        except Exception as exc:  # noqa: BLE001 — one base missing must not hide the rest
            print(f"| {model} | — | — | — | — | not measured: {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:80]} | | | |")
            failures += 1
            continue
        for row in rows:
            count = row.count
            flags = []
            if not row.within_budget:
                flags.append("OVER BUDGET")
            if not count.agrees:
                flags.append("grid disagrees")
            grid = "" if count.from_grid is None else str(count.from_grid)
            print(f"| {model} | {row.kind} | {row.image} | {row.fitted[0]}x{row.fitted[1]} | "
                  f"{row.budget.visual_tokens}{' (step)' if row.budget.stepped else ''} | "
                  f"{count.placeholders}{' ' + ', '.join(flags) if flags else ''} | {grid} | "
                  f"{'' if count.patches is None else count.patches} | {count.sequence_len} |")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

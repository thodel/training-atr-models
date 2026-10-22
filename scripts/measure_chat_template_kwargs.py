#!/usr/bin/env python3
"""Does ``apply_chat_template`` warn for these models, and what reaches the template? (#274)

    # in the venv or container the run uses — the answer is per transformers version
    python scripts/measure_chat_template_kwargs.py allenai/olmOCR-2-7B-1025 Qwen/Qwen3.5-4B

For every model: the processor's template kwargs as :func:`template_kwargs`
decides them, then the two renders training and evaluation make — with the
assistant's answer (``add_generation_prompt=False``) and the generation prompt
(``True``) — each with the transformers logger captured, so a warning shows up
as ``warned=True`` beside the call that caused it, not somewhere in a 1 500-step
log. Also run with ``--raw`` to pass ``CHAT_TEMPLATE_KWARGS`` unfiltered, which
is what the code did before #274 and what the log of 21.10.2026 shows.

Measured 22.10.2026 with transformers 5.17.0 and 4.57.6 (CPU, no weights):
under 5.17 the unfiltered flag warns on olmOCR-2 and Qwen3-VL at every call
and never on Qwen3.5-4B; filtered, nothing warns and Qwen3.5-4B still renders
``<think>\\n\\n</think>`` instead of an open ``<think>``. Under 4.57 nothing
warns and every render is byte-identical to 5.17's.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atr_training.vlm_dataset import CHAT_TEMPLATE_KWARGS, chat_example, template_kwargs  # noqa: E402


def _captured(call):
    """``(result, warning text)`` — what the transformers logger said during *call*."""
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setLevel(logging.WARNING)
    root = logging.getLogger("transformers")
    root.addHandler(handler)
    try:
        result = call()
    finally:
        root.removeHandler(handler)
    return result, buffer.getvalue().strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+", help="Hub ids or local checkpoints")
    parser.add_argument("--raw", action="store_true",
                        help="pass CHAT_TEMPLATE_KWARGS unfiltered, as before #274")
    parser.add_argument("--prompt", default="Transcribe.")
    args = parser.parse_args(argv)

    import transformers
    from transformers import AutoProcessor
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_warning()
    print(f"transformers {transformers.__version__}")

    warned_anywhere = False
    for model in args.models:
        processor = AutoProcessor.from_pretrained(model)
        kwargs = dict(CHAT_TEMPLATE_KWARGS) if args.raw else template_kwargs(processor)
        for generation_prompt in (False, True):
            conversation = chat_example(args.prompt, None if generation_prompt else "ANSWER")
            rendered, warning = _captured(lambda: processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=generation_prompt, **kwargs))
            warned_anywhere |= bool(warning)
            print(f"{model:32s} kwargs={kwargs!s:28s} add_generation_prompt={generation_prompt!s:5s} "
                  f"warned={bool(warning)!s:5s} tail=…{rendered[-48:]!r}")
            if warning:
                print(f"    {warning.splitlines()[-1]}")
    return 1 if warned_anywhere else 0


if __name__ == "__main__":
    raise SystemExit(main())

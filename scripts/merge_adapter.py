#!/usr/bin/env python
"""Fold a LoRA adapter into its base and save a complete model (#176, package 3).

GGUF cannot carry our adapters: llama.cpp's vision tower has no LoRA machinery
(``tools/mtmd/clip.cpp`` has no ``lora``), so a VLM arm reaches llama-server only
as a full model. This is that step, as a script rather than a handful of
commands, so the GGUF beside an adapter can say exactly how it was made.

Three things it does on purpose:

* **The base is loaded in bf16.** For an arm trained in bf16 the merge is exact.
  For a 4-bit arm it is not the weights the adapter was fitted against; pass
  ``--dequantize-from-4bit`` to load the base with the training's bnb config and
  dequantise it first, so the adapter lands on ``dequant(nf4(W))`` (#176 comment
  of 07.10.2026).
* **``LoraModel.merge_and_unload``**, reached as ``peft_model.base_model``. In
  peft 0.20 ``PeftModel`` itself has no such method.
* **The visual budget is written down.** ``--max-pixels`` adds
  ``min_image_tokens`` / ``max_image_tokens`` to the processor config. On
  07.10.2026 llama.cpp's Qwen3-VL converter turned them into
  ``vision_min/max_pixels`` in the projector GGUF; **in b11868 (24.10.2026) only
  the GLM-5 converter still does**, so for Qwen the keys do not reach the GGUF
  and the budget has to be given to llama-server as ``--image-min-tokens`` /
  ``--image-max-tokens``. They are written anyway, for the converters that read
  them and for the record: without a budget llama.cpp uses its own caps, which
  differ from the one every CER here was measured at.

Usage::

    python scripts/merge_adapter.py --base Qwen/Qwen3.5-4B \\
        --adapter dh-unibe/qwen3.5-4b-german-xix-v2 --out merged/qwen3.5-4b-german-xix-v2 \\
        --max-pixels 262144
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base", required=True, help="hub id or path of the base model")
    p.add_argument("--adapter", required=True, help="hub id or path of the peft adapter")
    p.add_argument("--adapter-revision", default=None, help="hub revision of the adapter")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-pixels", type=int, default=0,
                   help="visual budget the arm was trained and measured at; 0 bakes nothing")
    p.add_argument("--dequantize-from-4bit", action="store_true",
                   help="the arm trained over a 4-bit base: merge into its dequantised weights")
    return p.parse_args(argv)


def token_budget(preprocessor: dict, max_pixels: int) -> dict[str, int]:
    """``min_image_tokens`` / ``max_image_tokens`` for a pixel budget.

    One token is one merged patch, ``(patch_size * merge_size)²`` pixels — 1 024 for
    Qwen3-VL and Qwen3.5. The minimum is the processor's own ``shortest_edge``
    (a pixel count in these configs, despite the name), so the converter writes
    the floor the processor already applied rather than llama.cpp's.
    """
    patch = int(preprocessor.get("patch_size") or 16)
    merge = int(preprocessor.get("merge_size") or 2)
    per_token = (patch * merge) ** 2
    size = preprocessor.get("size") or {}
    out = {"max_image_tokens": max_pixels // per_token}
    if size.get("shortest_edge"):
        out["min_image_tokens"] = max(1, int(size["shortest_edge"]) // per_token)
    return out


def bake_budget(out: Path, max_pixels: int) -> dict[str, int]:
    """Write the token bounds where llama.cpp's converter will read them.

    transformers 4 saved ``preprocessor_config.json``; transformers 5 saves
    ``processor_config.json`` with the image settings nested under
    ``image_processor``. The converter reads both and lifts that section to the
    top (``conversion/base.py``, ``ModelBase`` of b11868), so the keys go to the
    top level of whichever file exists — inside ``image_processor`` they would
    reach the image processor's constructor as unknown arguments.
    """
    legacy = out / "preprocessor_config.json"
    path = legacy if legacy.is_file() else out / "processor_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    settings = config.get("image_processor", config)
    baked = token_budget(settings, max_pixels)
    config.update(baked)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return baked


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import peft
    import torch
    import transformers
    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    if args.out.exists() and any(args.out.iterdir()):
        print(f"{args.out} exists and is not empty; refusing to mix two merges", file=sys.stderr)
        return 2

    if args.dequantize_from_4bit:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        base = AutoModelForImageTextToText.from_pretrained(
            args.base, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0})
        base = base.dequantize().to(torch.bfloat16)
    else:
        base = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16)

    model = PeftModel.from_pretrained(base, args.adapter, revision=args.adapter_revision)
    merged = model.base_model.merge_and_unload()
    merged.save_pretrained(args.out, safe_serialization=True)

    # The processor the adapter was trained with, when training saved one beside
    # it — the same rule evaluate_qlora applies.
    try:
        processor = AutoProcessor.from_pretrained(args.adapter, revision=args.adapter_revision)
        processor_source = args.adapter
    except Exception:  # noqa: BLE001 — no processor beside the adapter: the base's is the one
        processor = AutoProcessor.from_pretrained(args.base)
        processor_source = args.base
    processor.save_pretrained(args.out)

    baked = bake_budget(args.out, args.max_pixels) if args.max_pixels > 0 else {}

    adapter_sha = None
    if not Path(args.adapter).exists():
        adapter_sha = HfApi().model_info(args.adapter, revision=args.adapter_revision).sha
    provenance = {
        "base": args.base,
        "adapter": args.adapter,
        "adapter_sha": adapter_sha,
        "processor_from": processor_source,
        "dequantized_from_4bit": args.dequantize_from_4bit,
        "max_pixels": args.max_pixels or None,
        "baked_budget": baked or None,
        "versions": {"transformers": transformers.__version__, "peft": peft.__version__,
                     "torch": torch.__version__},
    }
    (args.out / "merge_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

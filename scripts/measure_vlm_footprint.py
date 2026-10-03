#!/usr/bin/env python3
"""Static VRAM footprint of a VLM base, 4-bit against bf16, one card and two (#137).

Not a training run: it loads the model exactly as `train_qlora.build_model` does,
applies the same LoRA configuration, and reports what sits on each card. The
question #137 asks — does a fine-tune without 4-bit fit — has a cheap half and an
expensive half, and this is the cheap half: weights plus adapters plus optimiser
state are fixed, and only activations depend on the batch and the sample.

Run it in the engine's venv, which is the only place the stack is installed, and
with the cards the measurement is about:

    CUDA_VISIBLE_DEVICES=0   .venvs/vlm-train/bin/python scripts/measure_vlm_footprint.py \
        --out ~/atr-cache/137/bf16_one_card.json
    CUDA_VISIBLE_DEVICES=0   .venvs/vlm-train/bin/python scripts/measure_vlm_footprint.py \
        --four-bit --out ~/atr-cache/137/fourbit_one_card.json
    CUDA_VISIBLE_DEVICES=0,1 .venvs/vlm-train/bin/python scripts/measure_vlm_footprint.py \
        --device-map auto --out ~/atr-cache/137/bf16_two_cards.json

**What it found on asteraix, 03.10.2026**, for `Qwen/Qwen3-VL-8B-Instruct` with
the contract's default adapters (r=64, the seven projections, vision tower
excluded) and gradient checkpointing, as `in_use_gib` — what nvidia-smi shows,
CUDA context included:

    4-bit, one card      9.25 of 44.42 GiB
    bf16,  one card     17.31 of 44.42 GiB      <- fits, 27.1 GiB left
    bf16,  both cards    7.84 + 9.79 GiB        (3968M + 4974M parameters)

Three things worth keeping from that run. An A40 here is **44.42 GiB usable**, so
two are 88.8 and not the 92 the epic's arithmetic assumed. `device_map="auto"`
really does shard Qwen3-VL — the class declares
`_no_split_modules = ['Qwen3VLTextDecoderLayer', 'Qwen3VLVisionBlock']`, checked
in the venv rather than read off the documentation. And
`prepare_model_for_kbit_training` costs ~3 GiB of the 4-bit arm (5.94 -> 8.92
allocated), because it upcasts the norms and the embeddings.

Loading is slow for a reason that is not the GPU: the HF cache is a symlink onto
the CIFS share, and the four shards take ~42 s each.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

BASE_DEFAULT = "Qwen/Qwen3-VL-8B-Instruct"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EXCLUDE = r"(?:^|.*\.)(vision_tower|audio_tower|visual)\..*"
GIB = 2 ** 30


def cards() -> dict[str, dict[str, float]]:
    out = {}
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        out[f"gpu{i}"] = {
            "allocated_gib": round(torch.cuda.memory_allocated(i) / GIB, 2),
            "reserved_gib": round(torch.cuda.memory_reserved(i) / GIB, 2),
            # What nvidia-smi would show: the CUDA context and anything torch
            # does not own counts here and not above.
            "in_use_gib": round((total - free) / GIB, 2),
            "total_gib": round(total / GIB, 2),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=BASE_DEFAULT)
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--device-map", default="single", choices=["single", "auto"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    quantization = None
    if args.four_bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    device_map = {"": 0} if args.device_map == "single" else "auto"
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, quantization_config=quantization, dtype=torch.bfloat16,
        device_map=device_map, trust_remote_code=True)
    load_s = time.perf_counter() - t0
    after_weights = cards()

    model.config.use_cache = False
    if args.four_bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
        target_modules=TARGETS, exclude_modules=EXCLUDE))
    model.gradient_checkpointing_enable()
    after_adapters = cards()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    # AdamW keeps two fp32 moments per trainable parameter, plus the fp32 grad.
    adam_gib = round(trainable * (4 + 4 + 4) / GIB, 2)
    placement: dict[str, int] = {}
    for name, param in model.named_parameters():
        dev = str(param.device)
        placement[dev] = placement.get(dev, 0) + param.numel()

    report = {
        "base_model": args.base_model,
        "four_bit": args.four_bit,
        "device_map": args.device_map,
        "load_seconds": round(load_s, 1),
        "params_total_m": round(total / 1e6, 1),
        "params_trainable_m": round(trainable / 1e6, 2),
        "adamw_state_gib_estimate": adam_gib,
        "params_per_device_m": {d: round(n / 1e6, 1) for d, n in sorted(placement.items())},
        "after_weights": after_weights,
        "after_adapters_and_checkpointing": after_adapters,
    }
    text = json.dumps(report, indent=2)
    print(text, flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

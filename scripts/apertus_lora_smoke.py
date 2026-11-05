"""Will the LoRA probe train? One batch of Apertus 1.5 through the real trainer (#316).

Run on asteraix, in the venv the job will run in, before the first job:

    .venvs/vlm-train-apertus/bin/python scripts/apertus_lora_smoke.py \\
        [--image strip1.jpg --text "Zeile eins\\nZeile zwei" ...] [--report smoke.json]

Everything a job would find out an hour in, this finds out in minutes, through
the trainer's own code (``vlm_train_svc.train_qlora``), not a copy of it:

1. the processor loads and takes the probe's pixel budget (``apply_visual_budget``);
2. the chat template renders the answer, and the assistant header the collator
   masks up to is found (Apertus' template takes the answer as a string only);
3. a batch of enlarged strips tokenizes, with the loss on the answer alone;
4. the LoRA attaches to the language model, and to nothing in the vision or
   audio tokenizer;
5. one forward and backward pass fits on the card, with the peak it needed;
6. generation from the prompt returns text (the evaluate stage's path).

Without ``--image`` it draws two synthetic strips, which proves the plumbing and
nothing about reading. Exit 0 only if every step passed.

**One card, named, with room on it.** A job gets exactly one card through
``CUDA_VISIBLE_DEVICES``; the smoke test has to see the same, or it measures
something no job will run. The first run (05.11.2026) saw both cards, spread the
model over them with ``device_map=auto``, and ran out of memory after three
minutes of loading on a card a hand-started embedding server held 17.7 GB of.
So it refuses to start without ``CUDA_VISIBLE_DEVICES`` naming one card, and
before loading it reads that card's free memory and refuses below
``--min-free-gib`` (default: the probe's estimate), saying who holds the rest.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "engines")]

SPEC = ROOT / "ubelix" / "specs" / "apertus-1.5-8b-kurrent-xix-strip4-x2-probe.json"

#: What one training step of the probe needs, in GiB: ~17 bf16 weights of the
#: language model, ~1 for the fp32 tokenizers, the LoRA with its optimizer state,
#: and the activations of a batch whose logits span a 266 752-entry vocabulary.
#: An estimate until the first passing run reports its peak; then set it to that.
MIN_FREE_GIB = 32.0


def card_problem(visible: str | None, free_gib: float | None, min_free_gib: float) -> str | None:
    """Why the smoke test must not start on these cards — None if it may.

    Pure, so the rule is tested without a GPU: ``visible`` is the value of
    ``CUDA_VISIBLE_DEVICES``, ``free_gib`` what the one visible card has free.
    """
    if visible is None or not visible.strip():
        return ("CUDA_VISIBLE_DEVICES is not set: the test would see every card and "
                "spread the model over them, which no job does. Name one free card, "
                "e.g. CUDA_VISIBLE_DEVICES=1")
    cards = [c for c in visible.split(",") if c.strip()]
    if len(cards) != 1:
        return (f"CUDA_VISIBLE_DEVICES={visible} names {len(cards)} cards; a job gets one. "
                "Name exactly one")
    if free_gib is not None and free_gib < min_free_gib:
        return (f"card {cards[0]} has {free_gib:.1f} GiB free, the probe needs about "
                f"{min_free_gib:.0f}. Free the card or lower --min-free-gib if you know better")
    return None


def holders(card: str) -> str:
    """Who holds memory on ``card``, from nvidia-smi — for the refusal message."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_bus_id",
             "--format=csv,noheader", "-i", card],
            capture_output=True, text=True, timeout=20, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "(nvidia-smi not available)"
    return out or "(no compute processes listed)"


def _synthetic(tmp: Path) -> list[dict]:
    """Two strips of four lines, drawn small and enlarged 2x as the probe's crops are."""
    from PIL import Image, ImageDraw

    from atr_training.cropping import enlarge

    texts = ["Hochgeehrter Herr Baron!\nIch habe die Ehre, Ihnen\nden Empfang Ihres Briefes\nzu bestätigen.",
             "Mit vorzüglicher Hochachtung\nIhr ergebenster Diener\nJ. Schmid\nBern, den 12. Juli 1832"]
    out = []
    for i, text in enumerate(texts):
        img = Image.new("RGB", (620, 120), "white")
        ImageDraw.Draw(img).multiline_text((8, 6), text, fill="black", spacing=10)
        path = tmp / f"strip{i}.jpg"
        enlarge(img, 2.0, 1_000_000).save(path)
        out.append({"image": str(path), "text": text, "source_type": "block"})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", type=Path, default=SPEC)
    ap.add_argument("--image", action="append", default=[], help="a strip crop (repeatable)")
    ap.add_argument("--text", action="append", default=[], help="its transcription, \\n between lines")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--tmp", type=Path, default=Path("apertus-smoke"))
    ap.add_argument("--min-free-gib", type=float, default=MIN_FREE_GIB,
                    help="refuse to start below this much free memory on the card")
    args = ap.parse_args(argv)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    problem = card_problem(visible, None, args.min_free_gib)
    if problem:
        print(f"[REFUSED] {problem}", file=sys.stderr)
        return 2

    import torch

    free, total = torch.cuda.mem_get_info(0)
    free_gib = free / 2**30
    problem = card_problem(visible, free_gib, args.min_free_gib)
    if problem:
        print(f"[REFUSED] {problem}\n  on that card: {holders(visible.strip())}", file=sys.stderr)
        return 2
    print(f"[card] CUDA_VISIBLE_DEVICES={visible}: {free_gib:.1f} of {total / 2**30:.1f} GiB "
          "free", flush=True)
    from transformers import AutoProcessor

    from atr_training.contracts import TrainRequest
    from atr_training.vlm_cmd import train_cmd
    from atr_training.vlm_dataset import apply_visual_budget, example_for
    from vlm_train_svc.train_qlora import (
        HTRCollator, assistant_header_ids, build_model, parse_args)

    raw = json.loads(args.spec.read_text(encoding="utf-8"))
    request = TrainRequest(**{k: v for k, v in raw.items() if k != "notes"})
    params = request.params
    report: dict = {"spec": str(args.spec), "base_model": request.base_model,
                    "card": {"visible": visible, "free_gib": round(free_gib, 2)}, "steps": {}}

    def step(name, **facts):
        report["steps"][name] = facts
        print(f"[ok] {name}: {json.dumps(facts, ensure_ascii=False)}", flush=True)

    args.tmp.mkdir(parents=True, exist_ok=True)
    if args.image:
        if len(args.image) != len(args.text):
            ap.error("one --text per --image")
        samples = [{"image": i, "text": t.replace("\\n", "\n"), "source_type": "block"}
                   for i, t in zip(args.image, args.text)]
    else:
        samples = _synthetic(args.tmp)

    # The trainer's own argv, from the spec, as the runner would build it.
    argv_job = train_cmd("python", params=params, base_model=request.base_model,
                         train_jsonl="t.jsonl", val_jsonl="v.jsonl", data_root=".",
                         output_dir=str(args.tmp / "out"))[3:]
    job = parse_args(argv_job)

    try:
        t0 = time.time()
        processor = AutoProcessor.from_pretrained(job.base_model, trust_remote_code=True)
        budget = apply_visual_budget(processor, job.max_pixels)
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        step("processor", cls=type(processor).__name__, budget=str(budget),
             seconds=round(time.time() - t0, 1))

        header = assistant_header_ids(processor, job.prompt)
        render = processor.apply_chat_template(example_for(processor, job.prompt, "ANSWER"),
                                               tokenize=False, add_generation_prompt=False)
        step("template", header=processor.tokenizer.decode(header),
             answer_rendered="ANSWER" in render)

        collator = HTRCollator(processor, job.prompt, job.max_seq_len,
                               {"block": job.max_pixels})
        batch = collator(samples)
        labelled = int((batch["labels"] != -100).sum())
        answer = processor.tokenizer.decode(
            batch["labels"][0][batch["labels"][0] != -100], skip_special_tokens=True)
        step("batch", input_ids=list(batch["input_ids"].shape), labelled_tokens=labelled,
             first_answer_starts=answer[:40], keys=sorted(batch.keys()))
        if samples[0]["text"].split("\n")[0][:10] not in answer:
            raise SystemExit("the loss does not cover the transcription: masking is wrong")

        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        model = build_model(job, processor)
        adapted = [n for n, _ in model.named_parameters() if "lora_" in n]
        stray = [n for n in adapted if "vision_tokenizer" in n or "audio_tokenizer" in n]
        if stray:
            raise SystemExit(f"LoRA reached the tokenizers: {stray[:3]}")
        step("model", lora_tensors=len(adapted), example=adapted[0] if adapted else None,
             seconds=round(time.time() - t0, 1),
             loaded_gib=round(torch.cuda.memory_allocated() / 2**30, 2))

        model.train()
        device = next(model.parameters()).device
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        t0 = time.time()
        loss = model(**inputs).loss
        loss.backward()
        step("train_step", loss=round(float(loss), 4), seconds=round(time.time() - t0, 1),
             peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
             card_gib=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))

        model.eval()
        from PIL import Image
        prompt = processor.apply_chat_template(example_for(processor, job.prompt),
                                               tokenize=False, add_generation_prompt=True)
        gen_in = processor(text=[prompt], images=[[Image.open(samples[0]["image"]).convert("RGB")]],
                           return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**gen_in, max_new_tokens=48, do_sample=False)
        text = processor.tokenizer.decode(out[0][gen_in["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
        step("generate", text=text[:200])
    except SystemExit as exc:
        report["failed"] = str(exc)
    except Exception as exc:  # noqa: BLE001 — the report is the point
        report["failed"] = f"{type(exc).__name__}: {exc}"

    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if "failed" in report:
        print(f"[FAILED] {report['failed']}", file=sys.stderr)
        return 1
    print("all steps passed; the probe can be submitted", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

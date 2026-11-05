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
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "engines")]

SPEC = ROOT / "ubelix" / "specs" / "apertus-1.5-8b-kurrent-xix-strip4-x2-probe.json"


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
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoProcessor

    from atr_training.contracts import TrainRequest
    from atr_training.vlm_cmd import train_cmd
    from atr_training.vlm_dataset import apply_visual_budget, example_for
    from vlm_train_svc.train_qlora import (
        HTRCollator, assistant_header_ids, build_model, parse_args)

    raw = json.loads(args.spec.read_text(encoding="utf-8"))
    request = TrainRequest(**{k: v for k, v in raw.items() if k != "notes"})
    params = request.params
    report: dict = {"spec": str(args.spec), "base_model": request.base_model, "steps": {}}

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

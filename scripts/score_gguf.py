#!/usr/bin/env python
"""Score a GGUF arm served by llama-server on a line set, like evaluate_qlora (#176, package 4).

The point is a CER that stands beside the transformers figure of the same
adapter, so everything that can be held equal is:

* **the image** — fitted to ``--max-pixels`` with the collator's own
  :func:`atr_training.vlm_dataset.fit_pixels` and sent as PNG, so no second lossy
  encoding happens on the way;
* **the conversation** — :func:`chat_example` (image first, then the prompt) and
  :data:`CHAT_TEMPLATE_KWARGS`, passed as ``chat_template_kwargs``. Before
  anything is scored, the server's render of one sample (``/apply-template``) is
  compared with the processor's, and a difference other than the image
  placeholder stops the run: a prompt that drifts is a silent distribution
  shift, and the number would not mean what its label says;
* **decoding** — greedy, ``--max-new-tokens`` 256 as on the benchmark;
* **the score** — :func:`atr_training.textmetrics.score_pairs`, plus the
  whitespace-flat and convention-normalised views evaluate_qlora reports.

What cannot be held equal is the image pre-processing inside llama.cpp. Its
token bounds are part of the server's command line, so the caller passes them
here (``--server-flags``) and the report keeps them.

Usage::

    python scripts/score_gguf.py --server http://127.0.0.1:8401 \\
        --val-jsonl evalset/data/val.jsonl --data-root evalset \\
        --processor merged/qwen3.5-4b-german-xix-v2 --label Q8_0 \\
        --report evalset/report_gguf-Q8_0.json --server-flags "..."
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_training.churro_xml import flatten_whitespace, normalize_convention  # noqa: E402
from atr_training.textmetrics import score_pairs  # noqa: E402
from atr_training.vlm_dataset import (  # noqa: E402
    chat_example,
    fit_pixels,
    template_kwargs,
)

PROMPT = "Transcribe the handwritten text in this image exactly as written."
#: What the processor's render has where llama-server's has its media marker.
QWEN_IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
MEDIA_MARKER = re.compile(r"<__media(?:_[A-Za-z0-9]*)?__>")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--server", required=True)
    p.add_argument("--val-jsonl", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--processor", required=True, type=Path,
                   help="the merged model directory; its processor renders the reference prompt")
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--label", required=True, help="e.g. BF16, Q8_0")
    p.add_argument("--server-flags", default="", help="recorded verbatim in the report")
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--max-pixels", type=int, default=262144)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    return p.parse_args(argv)


def post(url: str, body: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def messages_for(png: bytes, prompt: str) -> list[dict]:
    """:func:`chat_example` in the OpenAI shape llama-server reads."""
    out = []
    for turn in chat_example(prompt):
        content = []
        for part in turn["content"]:
            if part["type"] == "image":
                content.append({"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(png).decode()}})
            else:
                content.append(part)
        out.append({"role": turn["role"], "content": content})
    return out


def png_of(path: Path, max_pixels: int) -> bytes:
    from PIL import Image

    with Image.open(path) as raw:
        image = fit_pixels(raw.convert("RGB"), max_pixels)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()


def check_render(args, processor, kwargs: dict, png: bytes) -> dict:
    """The server's prompt against the processor's, image placeholders aside."""
    ours = processor.apply_chat_template(chat_example(args.prompt), tokenize=False,
                                         add_generation_prompt=True, **kwargs)
    theirs = post(args.server + "/apply-template", {
        "messages": messages_for(png, args.prompt),
        "chat_template_kwargs": kwargs})["prompt"]
    # llama-server marks the image with a random ``<__media_…__>`` token, and mtmd
    # wraps the embeddings it stands for in <|vision_start|> … <|vision_end|> for
    # Qwen3-VL (tools/mtmd/mtmd.cpp, b11868) — the block the processor writes out.
    comparable_theirs = MEDIA_MARKER.sub(QWEN_IMAGE, theirs)
    return {"processor": ours, "server": theirs,
            "equal_but_image": ours == comparable_theirs}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.processor)
    kwargs = template_kwargs(processor)
    samples = [json.loads(line) for line in args.val_jsonl.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if args.max_samples:
        samples = samples[:args.max_samples]

    first_png = png_of(args.data_root / samples[0]["image"], args.max_pixels)
    render = check_render(args, processor, kwargs, first_png)
    if not render["equal_but_image"]:
        print("the server renders a different prompt than the processor:\n"
              f"processor: {render['processor']!r}\nserver:    {render['server']!r}",
              file=sys.stderr)
        return 3

    def one(sample: dict) -> tuple[str, int]:
        png = png_of(args.data_root / sample["image"], args.max_pixels)
        out = post(args.server + "/v1/chat/completions", {
            "messages": messages_for(png, args.prompt),
            "chat_template_kwargs": kwargs,
            "temperature": 0.0, "top_k": 1, "seed": 42,
            "max_tokens": args.max_new_tokens})
        choice = out["choices"][0]
        return (choice["message"].get("content") or "").strip(), int(
            choice.get("finish_reason") == "length")

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(one, samples))
    elapsed = time.monotonic() - started

    pairs = [(hyp, s["text"]) for (hyp, _), s in zip(results, samples)]
    report = score_pairs(pairs).as_report()
    flat = score_pairs([(flatten_whitespace(h), flatten_whitespace(r)) for h, r in pairs])
    report["whitespace_flat"] = {k: v for k, v in flat.as_report().items() if k != "examples"}
    normalized = score_pairs([(normalize_convention(h), normalize_convention(r)) for h, r in pairs])
    report["convention_normalized"] = {
        k: v for k, v in normalized.as_report().items() if k != "examples"}
    report.update({
        "label": args.label,
        "engine": "llama-server",
        "server_flags": args.server_flags,
        "processor": str(args.processor),
        "prompt": args.prompt,
        "chat_template_kwargs": kwargs,
        "rendered_prompt": render["server"],
        "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "truncated_at_cap": sum(cap for _, cap in results),
        "samples": len(samples),
        "seconds": round(elapsed, 1),
        "lines_per_second": round(len(samples) / elapsed, 2),
    })
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    with args.report.with_suffix(".raw.jsonl").open("w", encoding="utf-8") as fh:
        for (hyp, cap), s in zip(results, samples):
            fh.write(json.dumps({"image": s["image"], "reference": s["text"],
                                 "prediction": hyp, "at_cap": bool(cap)},
                                ensure_ascii=False) + "\n")
    print(f"{args.label}: CER {report['cer']:.4f} over {len(samples)} lines, "
          f"{elapsed:.0f} s, at cap {report['truncated_at_cap']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# A VLM arm as GGUF for llama-server (#176)

Our VLM arms are LoRA adapters. llama.cpp cannot apply a LoRA to the vision
tower (`tools/mtmd/clip.cpp` has no LoRA path), so an arm reaches
`llama-server` as a complete model: merge, convert, quantise, measure. This page
records the path and the first measurement.

## The path

| step | tool | note |
|---|---|---|
| merge | `scripts/merge_adapter.py` | base in bf16, `LoraModel.merge_and_unload`; exact for an arm trained in bf16, `--dequantize-from-4bit` for a 4-bit arm |
| text GGUF | `convert_hf_to_gguf.py <merged> --outtype bf16` | llama.cpp **b11868** (`78a8ae1`, 24.10.2026) |
| projector | `convert_hf_to_gguf.py <merged> --mmproj --outtype f16` | stays F16 at every text quantisation |
| quantise | `llama-quantize … Q8_0 / Q6_K / Q5_K_M / Q4_K_M` | |
| serve | `llama-server -m … --mmproj … --jinja --image-min-tokens 64 --image-max-tokens 256` | the budget **must** be given here, see below |
| measure | `scripts/score_gguf.py` | the benchmark's prompt, image fit, greedy decoding and `score_pairs` |

**The visual budget does not reach the GGUF for Qwen.** On 07.10.2026 the Qwen3-VL
converter wrote `vision_min/max_pixels` into the projector when the processor
config carried `min_image_tokens` / `max_image_tokens`. In b11868 only the GLM-5
converter still does. `merge_adapter.py` writes the keys anyway, and the
projector reads back without them. Without the two server flags llama.cpp
applies its own token bounds, which differ from the 64–256 tokens (65 536 to
262 144 px) every figure below was measured at.

**The prompt is checked, not assumed.** Before scoring, `score_gguf.py` compares
the server's render (`/apply-template`) with the processor's. Apart from the
image placeholder they must be identical. llama-server marks the image with a
random `<__media_…__>` token, and mtmd wraps its embeddings in
`<|vision_start|> … <|vision_end|>` itself. The check exists because a
thinking block in the wrong place cost a Gemma arm thirteen points.

## First measurement: `qwen3.5-4b-german-xix-v2`, 24.10.2026

The adapter is `dh-unibe/qwen3.5-4b-german-xix-v2` at revision `5ee0c4e7`. The
UBELIX checkpoint that produced the 6.80 % reference was emptied on 23.10., so
the published adapter stands in for it. It was measured on the
Federal Council benchmark (doi:10.5281/zenodo.4746342), all 2 751 lines, on one
A40 of asteraix with `--parallel 8`:

| | CER | WER | ins / del / sub | length ratio | lines identical to BF16 | Δ vs BF16, 95 % bootstrap | file |
|---|---:|---:|---|---:|---:|---|---:|
| transformers, bf16 (reference, 18.09.) | 6.80 % | 23.42 % | 1437 / 1388 / 4995 | 1.000 | — | — | — |
| GGUF BF16 | **6.72 %** | 23.17 % | 1412 / 1383 / 4926 | 1.000 | — | — | 7.85 GiB |
| Q8_0 | 6.74 % | 23.21 % | 1417 / 1386 / 4942 | 1.000 | 2 631 | +0.021 [−0.017, +0.057] | 4.17 GiB |
| Q6_K | 6.74 % | 23.17 % | 1444 / 1376 / 4934 | 0.999 | 2 500 | +0.029 [−0.020, +0.079] | 3.23 GiB |
| Q5_K_M | 6.75 % | 23.17 % | 1412 / 1383 / 4970 | 1.000 | 2 393 | +0.038 [−0.021, +0.097] | 2.86 GiB |
| Q4_K_M | 6.69 % | 23.05 % | 1426 / 1355 / 4913 | 0.999 | 2 208 | −0.023 [−0.111, +0.064] | 2.52 GiB |

The projector (mmproj, F16) adds 0.63 GiB to every row. No line hit the
256-token cap in any arm. Each arm took 9.5–12 minutes, about 4–5 lines per
second.

**What it shows.** The llama.cpp path reproduces the transformers figure: 6.72
against 6.80 %, with an equally balanced error profile. **No quantisation level
costs anything measurable.** Every difference to BF16 is under a tenth of a
point, and every interval includes zero. The threshold this project uses is the
measured draw noise of 0.6 points. Q4_K_M, which lost 15 points on printed
synthetic pages in the only published measurement (#176, comment of
07.10.2026), reads these lines as well as BF16 at a third of the size.

**What it does not show.** It is one arm, one base and one corpus of *lines*,
with outputs of a few dozen tokens. Long outputs such as whole pages are where
quantisation error accumulates, and where the published measurement broke down.
The projector was never quantised. The other two VLM arms named in #176 are
not measured yet.

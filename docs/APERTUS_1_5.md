# Apertus 1.5 for handwriting: what was measured, and the LoRA probe

`swiss-ai/Apertus-v1.5-8B` (October 2026) is the first multimodal Apertus. This
document records what it took to serve it, what four experiments showed about
reading handwriting with it, and how the LoRA probe that follows from them is
built and run. Issues: #314 (tokenizer round trip), #315 (zero-shot on the GT
pages), #316 (LoRA probe), #317 (strips). Serving side:
`serving-atr-inference/docs/APERTUS_1_5.md`.

## 1. The one fact everything else follows from

Apertus 1.5 does not look at an image through a vision transformer. It turns the
image into **discrete codes, one per 16 × 16 pixels**, from a frozen VQ tokenizer
(a port of BAAI's Emu3.5 vision tokenizer, fp32, 256 channels, codebook 131 072),
and the language model reads those codes like words. The processor caps an image
at **1.96 MP** (`max_pixels: 1960000`), about 7 700 codes.

A letter that is smaller than a cell has no code of its own. On a letter page
scanned at working resolution, Laßberg's hand is 8–12 px high. No amount of
training of the language model can recover what the tokenizer has already
thrown away, and no LoRA reaches the tokenizer.

## 2. What was measured

### 2.1 Round trip through the tokenizer (#314)

`serving-atr-inference/scripts/apertus_vq_roundtrip.py` encodes a page exactly
as Apertus does and decodes it with BAAI's decoder. Apertus' encoder weights and
Emu3.5's are identical (247/247 tensors) and the codes agree 100 %, so the
decoder shows what the model sees.

| page | seen | codes | PSNR | Qwen reading of the reconstruction vs. of the original |
|---|---|---:|---:|---:|
| Basel 1781 / 062 | 1248×1568 | 7 644 | 28.6 dB | 42.7 % CER |
| Basel 1781 / 167 | 784×1152 | 3 528 | 26.4 dB | 62.3 % |
| Basel 2263 / 117 | 1248×1568 | 7 644 | 28.6 dB | 343.6 % (runaway output) |

The best Kurrent reader available (`qwen3.5-4b-german-kurrent-xvi-xix-page-v1`)
loses 40–60 % of the characters on the reconstruction. This is an upper bound
on the loss — Qwen has never seen the decoder's artefacts — but the 1:1 crops
show word shapes that are actually destroyed.

### 2.2 Zero-shot on the 276 GT pages, five ways of cutting (#315, #317)

Run `apertus-gt-page-zeroshot` on tei, scored with `score-gt-20261105T022118`
over the **166 pages whose identity is settled**, every reading against the same
ground-truth page (so pages where a model invents text count with their real
error).

| reading | median CER | weighted | p90 | pages < 20 % | pages ≥ 60 % (invented) |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B Kurrent fine-tune (reference) | **11.2 %** | 12.3 % | 22.6 % | 140 | 1 |
| Apertus, 4 lines, **2× enlarged** | **22.4 %** | 24.4 % | 41.7 % | 69 | 3 |
| Apertus, 4 lines, native | 30.4 % | 32.8 % | 56.4 % | 31 | 10 |
| Apertus, whole page | 31.1 % | 34.3 % | 53.4 % | 20 | 9 |
| Apertus, single lines | 42.4 % | 41.8 % | 59.4 % | 11 | 16 |

By holding (median CER; Qwen / 2× / native / page / line):

| holding | n | Qwen | 2× | native | page | line |
|---|---:|---:|---:|---:|---:|---:|
| Basel (small scans, ~1300×1600) | 59 | 6.7 | **16.1** | 36.3 | 29.0 | 45.7 |
| St. Gallen KB | 36 | 15.4 | 29.3 | 29.7 | 34.3 | 42.6 |
| Briefe UB Freiburg | 26 | 9.4 | 17.5 | 20.1 | 24.3 | 32.0 |
| BLB Laßberg | 12 | 10.9 | 20.6 | 24.7 | 35.2 | 32.9 |
| KB Appenzell | 11 | 11.9 | 26.4 | 24.0 | 28.2 | 32.9 |
| Donaueschingen | 7 | 4.9 | 10.4 | 11.2 | 23.8 | 15.8 |

Pairwise on the same pages: 2× beats native on 121 of 166 (median −3.0
points) and the whole page on 150 of 166 (−8.6). Native strips beat the page on
only 87 of 166.

### 2.3 What the numbers say

1. **Pixels per letter are the lever, not context.** Native strips are no better
   than the page. The same strips enlarged 2× halve the error on Basel's small
   scans, where 2× really doubles the cells a letter gets; on large scans the
   1 MP budget caps the enlargement and native and 2× sit close together.
2. **Single lines are worst**: a line alone loses the context, and on small
   scans a line crop has no more pixels per letter than the page.
3. **Enlarged strips nearly stop the invention** (3 pages against 9 and 16).
4. **The gap to the fine-tune stays at 2×.** Zero-shot is not enough; whether a
   LoRA closes it is the probe's question.

Caveat: the first 27 pages of the 2× strips (almost all Basel) ran before the
strip budget was lowered from 1.96 to 1 MP and had slightly more pixels.

## 3. The LoRA probe (#316)

### 3.1 Design

| | value | why |
|---|---|---|
| base | `swiss-ai/Apertus-v1.5-8B` | |
| unit | `granularity: block`, `block_lines: 4` | the strip of §2.2 |
| scale | `crop_scale: 2.0` within `max_pixels: 1000000` | the best zero-shot scale; 1 MP is what serving can afford (serving#251) |
| adapters | q/k/v/o and up/down projections of the language model | Apertus' MLP has no `gate_proj` |
| excluded | `vision_tokenizer`, `audio_tokenizer` (regex) | the tokenizer is frozen; it has `q`, `k`, `v` modules of its own |
| precision | bf16, no 4-bit | an 8B fits an A40 in bf16 (#137) |
| batch | 1 × 16 accumulated | a 1 MP strip is ~3 900 codes |
| corpus | `dh-unibe/image-text_kurrent-xix`, ≤ 1 500 pages, the train/eval projects of `german-xix-block-v1` | a probe, not a corpus run |
| prompt | the serving strip prompt | the prompt travels with the model |

The spec is `ubelix/specs/apertus-1.5-8b-kurrent-xix-strip4-x2-probe.json`.
The 276 GT pages (Transkribus Laßberg exports) are not in that corpus, so they
stay a clean comparison after the probe.

### 3.2 What changed in this repo for it

- `VlmTrainParams.crop_scale` and `cropping.enlarge`: crops are enlarged when
  they are cut, by the same rule serving uses (`min(scale, sqrt(budget/area))`,
  serving#241/#243), so the model trains at the scale it is served at. Refused
  at `granularity: page`.
- `backends.venv_for`: an Apertus 1.5 base runs in `.venvs/vlm-train-apertus`,
  built by `scripts/make_venvs.sh vlm-train-apertus` with swiss-ai's
  transformers fork at the commit serving runs (`3797303`).
- `vlm_dataset.example_for`: Apertus' chat template accepts the assistant's
  answer only as a string (a list of parts raises `Invalid assistant content`);
  the form is asked of the template, so Qwen and Gemma keep their renders.
- `vlm_load.load_vlm`: Apertus' model card loads it through
  `AutoModelForMultimodalLM`; the image-text auto class every other base uses has
  no mapping for `apertus1p5`. Training, evaluation and `merge_adapter.py` try the
  second class only when the first says it does not know the model.
- `registration.vllm_launch_defaults`: a trained Apertus registers with
  `vllm_venv: vllm-apertus`, the gateway's Apertus `vllm_args` and a 900 s start
  timeout; a block model registers `lines_per_crop` and `crop_scale` so the
  gateway cuts the same strips.
- `scripts/apertus_lora_smoke.py`: one batch through the trainer's own code.

### 3.3 Runbook

On **asteraix**, card 1 free:

```bash
cd ~/Repo/training-atr-models && git pull
bash scripts/make_venvs.sh vlm-train-apertus          # once, ~10 min
nvidia-smi --query-compute-apps=pid,used_memory,gpu_bus_id --format=csv   # who holds what
CUDA_VISIBLE_DEVICES=1 .venvs/vlm-train-apertus/bin/python scripts/apertus_lora_smoke.py \
    --report ~/apertus-smoke.json
```

The smoke test refuses to start without `CUDA_VISIBLE_DEVICES` naming exactly one
card, and below ~32 GiB free on it (`--min-free-gib`), naming the processes that
hold the rest. Its first run (05.11.2026) saw both cards, split the model over them
and ran out of memory beside a hand-started embedding server
(`Qwen3-Embedding-4B`, `--data-parallel-size 2`) that held 40 % of each card.

The smoke test must end with `all steps passed`. It reports the assistant header,
the labelled tokens, the LoRA tensors (none in the tokenizers), the peak memory of
one training step, and a generation. Then submit the spec as any other VLM job
(`ubelix/submit_job.py` or the trainer API). The runner picks the venv from the
base model.

After training: merge the adapter into its base (`scripts/merge_adapter.py`, in the
`vlm-train-apertus` venv), copy the merged model to the gateway's
`vllm_merged_dir`, enable the registration, and read the 276 GT pages through
the gateway as in #315. The registration already carries the strip geometry and
the Apertus launch flags.

### 3.4 What would count as a result

- Held-out CER on the kurrent-xix eval projects, against the Qwen block run on
  the same projects (`german-xix-block-v1`).
- On the 276 GT pages through serving: below the zero-shot 22.4 % median, and
  how far towards Qwen's 11.2 %.
- The count of invented pages (≥ 60 %).

If the LoRA does not move the 2× strips clearly, the tokenizer is the limit and
the next step is a larger scale (2-line strips at 2×, or 3×), not more data.

## 4. Serving it: what failed, in order

Each of these was a crash on idhefix first; all are in serving-atr-inference.

| PR | failure | fix |
|---|---|---|
| #234 | upstream vLLM/transformers do not know `apertus1p5`; swiss-ai's image is CUDA 13 | own venv with the fork over precompiled cu129 kernels |
| #237 | cold start > 300 s; an orphaned EngineCore held 30 GB | process groups, `startup_timeout_s: 900`, `--safetensors-load-strategy prefetch` |
| #238 | an image split across prefill chunks | `--max-num-batched-tokens 16384` |
| #239 | OOM in the fp32 vision encoder on a full page | `--kv-cache-memory-bytes 3 GiB` |
| #241 | — | strips: `lines_per_crop`, `crop_scale` |
| #243 | 2× strips of Aarau's 5000 px scans crashed the engine | strips capped at the model's pixel budget |
| #251 | still OOM at 1.96 MP once the engines beside it grew | strip budget 1 MP |

Agentic historian: `agentic_historian#760` — the batch preflight now counts a
lazy resident the gateway would evict as free memory.

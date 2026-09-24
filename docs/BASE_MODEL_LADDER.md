# Base models and sizes — a plan for the next runs

Every model this project has fine-tuned sits on one of two 4B bases:
`Qwen/Qwen3-VL-4B-Instruct` or `Qwen/Qwen3.5-4B`. Two questions have never been
separated from each other, and neither has been asked above 4B:

1. **Does more capacity read better?** If a 27B halves the CER, the corpus
   campaign is worth less than the model campaign. If it does not, 4B is the
   answer and the remaining work is all data.
2. **Does another family read better?** Qwen is a choice nobody in this project
   ever justified — it was the first thing that worked. A second architecture is
   the only way to find out whether our findings are about handwriting or about
   Qwen.

This document plans the runs that answer them. It assumes
[INFRASTRUCTURE.md](INFRASTRUCTURE.md) for the queue rules and
`serving-atr-inference/docs/ARCHITECTURE_SEARCH.md` §1 for the search method —
neither is restated here.

Facts below are marked **measured** where a command produced them on
2026-09-24, and **assumed** where they are extrapolation. The distinction is the
point: this project has twice spent GPU-weeks on a plausible cause that
measurement then killed.

## 1. What is already measured

One corpus (964 472 lines, 19th century), one seed, one page-level split, one
epoch, line granularity. CER on the held-out [Federal Council
benchmark](https://doi.org/10.5281/zenodo.4746342), 2 751 lines:

| Base | Params | Benchmark CER |
|---|---:|---:|
| `Qwen/Qwen3.5-0.8B` | 0.87 B | 11.15 % |
| `Qwen/Qwen3.5-2B` | 2.27 B | 8.95 % |
| `Qwen/Qwen3.5-4B` | 4.66 B | **6.80 %** |
| `Qwen/Qwen3-VL-4B-Instruct` | 4.44 B | 7.65 % |

Two readings, and they pull in different directions:

**Size still pays, and pays evenly.** 0.8 → 2 → 4 B is 2.20 and 2.15 points per
step, each step being roughly a doubling. Nothing in the curve has flattened
yet. A naive log-linear extrapolation gives 9 B ≈ 5.7 % and 27 B ≈ 4.2 %
(**assumed** — this is the hypothesis the ladder tests, not a result).

**Generation is worth as much as size.** At an identical 4 B, changing the
family generation moved the CER 0.85 points — most of a doubling. So "which
base" is not a second-order question to be settled once and forgotten; a newer
base of the same size may be worth more than a bigger base of the old one.

## 2. Why a matrix is affordable

Three properties of the existing code, all verified today:

- **The artefact cache key does not contain the base model.**
  `_cache_key` ([engines/vlm_train_svc/runner.py:113](../engines/vlm_train_svc/runner.py))
  keys on the dataset specs, the engine and the granularity knobs. Compiling the
  corpus is the expensive half (the run in flight on 2026-09-24 took 5 h 25 to
  materialise and compile, then ~1 h 15 to copy 48 GB into the cache); the base
  model is a train-stage input that stays out of it. **One prepare, N arms.**
- **`fanout.py` already clones one prepared job per base**, sharing one corpus
  *and one seeded split*, with `fanout_submit.sbatch` queueing the arms behind
  the prepare with `--dependency=afterok`. It was written for experiment C
  (2B/4B/8B) and needs nothing added for this.
- **Evaluation does not go through vLLM.** `evaluate_qlora.py` loads
  `AutoModelForImageTextToText` + `AutoProcessor` and calls `generate`, so any
  family `transformers` supports can be *measured* whether or not it can be
  *served*. Serving is a separate gate (§8).

Measured cost of one 4 B training on this corpus, one H100: **6 h 31**
(`qwen3vl-german-xix-block-v1`, Slurm 16056580) and **4 h 51**
(`…-page-v1`, 16056584).

## 3. The candidates

All verified on the hub on 2026-09-24: parameter counts from the safetensors
index, image processor from `preprocessor_config.json` / `processor_config.json`,
vLLM support from `ModelRegistry.get_supported_archs()` in
`.venvs/vllm-next` (vLLM 0.29.0) on idhefix.

| Model | Params | Arch | Image processor | In vLLM 0.29 | Licence |
|---|---:|---|---|---|---|
| `Qwen/Qwen3.5-0.8B` … `-4B` | 0.87–4.66 B | `qwen3_5` | `Qwen2VLImageProcessorFast` | ✅ | Apache-2.0 |
| `Qwen/Qwen3.5-9B` | 9.65 B | `qwen3_5` | identical | ✅ | Apache-2.0 |
| `Qwen/Qwen3.5-27B` | 27.78 B | `qwen3_5` | identical | ✅ | Apache-2.0 |
| `Qwen/Qwen3.8-27B` | 27.78 B | `qwen3_5` | identical | ✅ | Apache-2.0 |
| `Qwen/Qwen3.5-35B-A3B` | 35.95 B (≈3 B active) | `qwen3_5` MoE | identical | ✅ | Apache-2.0 |
| `google/gemma-4-12B-it` | 11.96 B | `gemma4_unified` | `Gemma4UnifiedImageProcessor` | ✅ | Apache-2.0 |
| `google/gemma-4-31B-it` | 31.27 B | `gemma4` | `Gemma4ImageProcessor` | ✅ | Apache-2.0 |
| `google/gemma-4-26B-A4B-it` | 25.81 B (≈4 B active) | `gemma4` MoE | `Gemma4ImageProcessor` | ✅ | Apache-2.0 |

Two things this table settles that were worth checking rather than assuming:

- **Qwen3.5-9B/27B are a drop-in swap.** `preprocessor_config.json` is
  byte-identical to the 4B's — `patch_size: 16`, `merge_size: 2`,
  `size.longest_edge` — so every pixel budget we measured converts to the same
  number of visual tokens, and `apply_visual_budget` needs no change.
- **Qwen3.8-27B has exactly the same parameter count as Qwen3.5-27B**
  (27 781 427 952) and the same processor and vocabulary. It is the same model
  shape, one generation later. That makes it the cleanest generation control
  available anywhere in this comparison: one number changes, and it is not size.

**Gemma 4, not Gemma 3.** Gemma 3 repositories are gated — `google/gemma-3-4b-it`
answers `Access to model … is restricted` to an anonymous request — while the
whole Gemma 4 line is ungated and Apache-2.0. One less thing to arrange.

## 4. Gemma is not a drop-in, and that is the point

Verified from `processor_config.json`:

```
"image_processor": { "image_processor_type": "Gemma4ImageProcessor",
                     "max_soft_tokens": 280, "patch_size": 16,
                     "pooling_kernel_size": 3, ... },
"image_seq_length": 280
```

Gemma 4 spends a **fixed 280 soft tokens per image**, whatever the image is.
There is no `size.longest_edge` and no `max_pixels`. Two consequences:

**A Gemma arm fails at startup today, loudly.** `apply_visual_budget`
([src/atr_training/vlm_dataset.py:562](../src/atr_training/vlm_dataset.py))
raises `VisualBudgetError` when a processor has neither knob. That is the
behaviour #86 and #128 were fixed into — refusing beats silently training at the
model's default — so the failure is correct and the work is to extend the
function with a branch that recognises a fixed-budget processor and reports
`280` rather than pretending to set something.

**Above the line, the comparison is not fair, and the unfairness is measurable.**
Our budgets are 256 visual tokens for a line, 1 024 for a block of six, 2 048 for
a page. Gemma gives 280 at every level. On a line that is comparable — slightly
generous, even. On a page it is **seven times less** than Qwen sees.

So the prediction is explicit, and falsifiable: *Gemma 4 should be competitive
with a Qwen of similar size on lines, and clearly worse on blocks and pages,
unless a page is presented as several images.* If it holds, "a model reads the
unit it was trained on" survives contact with a second architecture and the page
problem is partly a token-budget problem after all. If Gemma reads pages well at
280 tokens, then our page budget of 2 048 was never the binding constraint and
the block/page/mixed results need rereading. Either answer is worth a run.

## 5. The confound to avoid before it is created

The current specs train at `load_in_4bit: false` — bf16, not QLoRA, despite the
module names. At 27 B that is 55.6 GB of weights on an 80 GB H100 before
activations; 4-bit NF4 is ≈ 15.6 GB and comfortable.

The trap is obvious once stated: if the ladder switches to 4-bit somewhere in
the middle, the "size curve" becomes a size-and-quantisation curve, and no later
analysis can separate them. The three 4B/2B/0.8B points already measured are
bf16.

**Decision: run the whole new ladder at `load_in_4bit: true`, and re-run
`Qwen3.5-4B` at 4-bit as well.** That last arm costs one 4B training (~6 h) and
buys a measured bf16→4-bit delta at a point whose bf16 number we already know —
which is what makes the old and the new points comparable at all.

## 6. The order of runs, and when to stop

`ARCHITECTURE_SEARCH.md` §1 already argues the method from this project's own
data: large gaps settle at a small budget, small gaps need a big one. Applied
here:

**Rung 1 — screen, on a subsampled corpus.** Line granularity, one epoch, a
corpus capped at roughly a sixth of the full one, so an arm is ~1 h rather than
~6. One prepare, then `fanout_submit.sbatch` with every arm in §3. Purpose:
rank, and catch the arms that fail for mechanical reasons (Gemma's budget, a
chat template whose assistant header cannot be derived, an OOM at 27 B) while
each failure costs an hour.

Note that the subsampled corpus is a *different* cache key, so rung 1 pays its
own prepare. It is the cheaper prepare of the two and it is paid once.

**Rung 2 — full corpus, survivors only.** Everything within ~1.5 points of the
best rung-1 arm, plus `Qwen3.5-4B` at 4-bit as the anchor. Scored on the Federal
Council benchmark, which is the only number in this project that has held up.

**Rung 3 — granularity, for the winner only.** Blocks, pages and the mix are a
four-way measurement per model (`scripts/eval_granularity.py` in the serving
repo). Crossing that axis with eight bases is 32 measurements to answer a
question we already know the shape of. Do it once, on whatever rung 2 selects,
and once on the best Gemma — because §4's prediction is precisely about this
axis.

**Stop rule.** Stop climbing when a doubling of parameters buys less than 0.5
points, or when the arm that wins cannot be served (§8) and its margin over the
best servable arm is under 1 point. Both thresholds are judgements, not
measurements; they are written down so that the decision is made before the
numbers arrive rather than after.

## 7. Queue and cost

Free UBELIX resources: `job_gratis` gives one H100 for up to 96 h;
`job_gpu_preemptable` gives four H100s for 24 h and can be interrupted at any
time. `train_resumable.sbatch` and `ATR_TRAIN_PREEMPTABLE=1` make an interrupted
job requeue and resume from its last checkpoint (`save_steps: 200`).

**Rung 1 belongs on `job_gpu_preemptable`** — it is four arms at a time, each
short enough that a preemption costs little, and it is the only way this project
uses more than one GPU at once. **Rung 2's big arms belong on `job_gratis`**,
which is where a 30 h job can finish without being interrupted.

Estimated GPU hours (**assumed**, scaled from the measured 4 B runs by parameter
count at 4-bit; the MoE arms are scaled by *active* parameters, which is the
optimistic reading and may be wrong):

| Rung | Arms | Each | Total |
|---|---|---:|---:|
| 1 — screen, ⅙ corpus | 8 | 1–4 h | ~15 h |
| 2 — full corpus | 3–4 | 6–30 h | ~60 h |
| 3 — granularity eval | 2 | <1 h | ~2 h |

Plus two prepares: ~2 h and ~7 h on 8 CPUs. The whole programme is therefore
roughly **80 GPU-hours**, which on one `job_gratis` H100 is a fortnight of
wall-clock and on the preemptable queue considerably less. The binding
constraint is not the GPU budget, it is that `job_gratis`'s CPU-minute cap is
shared with every other job in the account — a 12 h, 8-CPU prepare blocks the
next GPU job from starting, which is exactly what happened on 2026-09-24.

## 8. Serving is a separate gate, and it is lower than the training gate

idhefix has two A40s at **45 GB** each (measured). That sets a bf16 ceiling of
roughly **12 B** for a served model on one card, and card 1 is usually most of
the way full already.

- 9 B bf16 ≈ 19 GB — fits.
- `gemma-4-12B-it` ≈ 24 GB — fits.
- 27 B bf16 ≈ 56 GB — **does not fit.** It needs FP8 (`Qwen/Qwen3.8-27B-FP8`
  exists, ≈ 28 GB) or INT4, or tensor parallelism across both cards, which would
  take the box's second GPU away from everything else.

vLLM 0.29 in `.venvs/vllm-next` supports every architecture in §3 —
`Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForConditionalGeneration`,
`Gemma4ForConditionalGeneration`, `Gemma4UnifiedForConditionalGeneration`
(measured). So the serving question is memory and the LoRA-on-quantised-base
path, not architecture support.

**Consequence for the plan: a 27 B arm may be measurable and not deliverable.**
That is an acceptable outcome — it answers question 1 — but it must not be
promised as a production model before an FP8 or INT4 adapter path has actually
been served once. The ladder's deliverable is a number; the deliverable *model*
is the best arm that fits on an A40.

## 9. What would make this wrong

- **The benchmark is 2 751 isolated lines.** A base that is better at holding a
  page together would show nothing here. Rung 3 exists for that reason, but the
  selection at rung 2 is still made on a line-level number, and that is a known
  bias, not an oversight.
- **One seed, one epoch, no error bar.** This project has never measured its own
  run-to-run variance, and the differences at the top of the ladder may be
  smaller than it. Before reading a 9 B-vs-27 B gap as real, repeat one 4 B arm
  at a second seed. Roughly 6 GPU-hours for the right to interpret everything
  above it.
- **The extrapolation in §1 is a straight line through three points on a log
  axis.** Scaling curves bend. If 9 B lands at 6.5 % rather than 5.7 %, the
  honest conclusion is that this corpus saturates around 4–9 B, and the ladder
  stops there rather than being pushed to 27 B to see.
- **`load_in_4bit` at 27 B is untested in this repo.** Every measured run is
  bf16. The 4 B arm in §5 is the control that turns that from an assumption into
  a number.

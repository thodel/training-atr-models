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

Both are asked on **both corpora** — 19th-century German and medieval German —
because a base that wins on one century and loses on the other is a different
result from a base that wins.

This document assumes [INFRASTRUCTURE.md](INFRASTRUCTURE.md) for the queue rules
and `serving-atr-inference/docs/ARCHITECTURE_SEARCH.md` §1 for the search method.

Facts below are marked **measured** where a command produced them on 2026-09-24
or 2026-09-25, and **assumed** where they are extrapolation. The distinction is
the point: this project has twice spent GPU-weeks on a plausible cause that
measurement then killed — and §4 of this document was itself wrong for a day.

## 1. What is already measured

One seed, one page-level split, one epoch, line granularity, `load_in_4bit:
false`.

**19th century** (964 472 lines), CER on the held-out [Federal Council
benchmark](https://doi.org/10.5281/zenodo.4746342), 2 751 lines:

| Base | Params | Benchmark CER |
|---|---:|---:|
| `Qwen/Qwen3.5-0.8B` | 0.87 B | 11.15 % |
| `Qwen/Qwen3.5-2B` | 2.27 B | 8.95 % |
| `Qwen/Qwen3.5-4B` | 4.66 B | **6.80 %** |
| `Qwen/Qwen3-VL-4B-Instruct` | 4.44 B | 7.65 % |

**Medieval** (four corpora, 1300–1600, 306 582 lines): `qwen3vl-medieval-german-v3`
reads held-out pages at **11.1 %**. The smaller Qwen3.5 arms of that generation
were trained before the PageXML fix and are deliberately unmeasured.

Two readings, and they pull in different directions:

**Size still pays, and pays evenly.** 0.8 → 2 → 4 B is 2.20 and 2.15 points per
step, each step roughly a doubling. Nothing has flattened yet. A naive
log-linear extrapolation gives 9 B ≈ 5.7 % and 27 B ≈ 4.2 % (**assumed** — the
hypothesis the ladder tests, not a result).

**Generation is worth as much as size.** At an identical 4 B, changing the family
generation moved the CER 0.85 points — most of a doubling. "Which base" is not a
second-order question to be settled once and forgotten.

## 2. Why this costs no prepare at all

The expensive half of a run is building the corpus. This ladder does not have to:

- **The artefact cache key does not contain the base model.** `_cache_key`
  ([engines/vlm_train_svc/runner.py:113](../engines/vlm_train_svc/runner.py))
  keys on the dataset specs, the engine and the granularity knobs.
- **`fanout.py` clones a finished job into one arm per base**, sharing one corpus
  *and one seeded split*, by symlinking the six shared inputs. Its
  `SOURCE_STATUSES` accepts `completed`, not only a stage-1 prepare.
- **Both line corpora are still on disk** (measured 2026-09-25):

  | Corpus | Job | Samples | State |
  |---|---|---:|---|
  | 19th century | `20260916T090417Z-qwen3vl-german-xix-v2` | 873 468 | `completed`, all six inputs present |
  | medieval | `20260915T055400Z-qwen3vl-medieval-german-v3` | 306 582 | `completed`, all six inputs present |

That is better than a fresh prepare, not merely cheaper: an arm fanned out from
these trains on **the exact split that produced 7.65 % and 11.1 %**, so the
comparison against those numbers is exact rather than approximate. It also means
the ladder never competes for `job_gratis`'s CPU-minute cap, which is what
serialises everything else on this cluster.

- **Evaluation does not go through vLLM.** `evaluate_qlora.py` loads
  `AutoModelForImageTextToText` + `AutoProcessor` and calls `generate`, so any
  family `transformers` supports can be *measured* whether or not it can be
  *served*. Serving is a separate gate (§8).

Measured cost of one 4 B training on the 19th-century corpus, one H100: **6 h 31**
(`qwen3vl-german-xix-block-v1`, Slurm 16056580) and **4 h 51** (`…-page-v1`,
16056584).

## 3. The candidates

Verified on the hub on 2026-09-24/25: parameter counts from the safetensors
index, processors from the repositories' own config, vLLM support from
`ModelRegistry.get_supported_archs()` in `.venvs/vllm-next` (vLLM 0.29.0) on
idhefix.

| Model | Params | Arch | Image processor | In vLLM 0.29 | Licence |
|---|---:|---|---|---|---|
| `Qwen/Qwen3.5-0.8B` … `-4B` | 0.87–4.66 B | `qwen3_5` | `Qwen2VLImageProcessorFast` | ✅ | Apache-2.0 |
| `Qwen/Qwen3.5-9B` | 9.65 B | `qwen3_5` | identical | ✅ | Apache-2.0 |
| `Qwen/Qwen3.5-27B` | 27.78 B | `qwen3_5` | identical | ✅ | Apache-2.0 |
| `Qwen/Qwen3.8-27B` | 27.78 B | `qwen3_5` | identical | ✅ | Apache-2.0 |
| `Qwen/Qwen3.5-35B-A3B` | 35.95 B (≈3 B active) | `qwen3_5` MoE | identical | ✅ | Apache-2.0 |
| `google/gemma-4-E4B-it` | 8.00 B | `gemma4` | `Gemma4ImageProcessor` | ✅ | Apache-2.0 |
| `google/gemma-4-12B-it` | 11.96 B | `gemma4_unified` | `Gemma4UnifiedImageProcessor` | ✅ | Apache-2.0 |
| `google/gemma-4-31B-it` | 31.27 B | `gemma4` | `Gemma4ImageProcessor` | ✅ | Apache-2.0 |

Three things this settles that were worth checking rather than assuming:

- **Qwen3.5-9B/27B are a drop-in swap.** `preprocessor_config.json` is
  byte-identical to the 4B's — `patch_size: 16`, `merge_size: 2`,
  `size.longest_edge` — so every pixel budget we measured converts to the same
  number of visual tokens.
- **Qwen3.8-27B has exactly the same parameter count as Qwen3.5-27B**
  (27 781 427 952), the same processor and the same vocabulary. It is the same
  model shape, one generation later: the cleanest generation control in the set.
- **There is no dense 4B in Gemma 4.** The line is E2B/E4B, 12B, 26B-A4B, 31B, so
  there is no arm that matches our Qwens on size and architecture at once. E4B
  (8.00 B total) is the size-honest comparison and 12B the capability-honest one;
  the ladder runs both rather than pretending one of them is "the 4B".

**Gemma 4, not Gemma 3.** Gemma 3 repositories are gated —
`google/gemma-3-4b-it` answers `Access to model … is restricted` to an anonymous
request — while the whole Gemma 4 line is ungated and Apache-2.0. Gemma 3 is also
a different family with a genuinely fixed 896×896 / 256-token processor; do not
carry a finding from one to the other.

## 4. Gemma's budget is stepped, not absent

An earlier version of this section said Gemma 4 spends a fixed 280 soft tokens
per image, has no budget knob at all, and would therefore see seven times less
than Qwen on a page. **That was wrong**, and it was wrong in the way this project
keeps catching itself: it was read off a config file instead of a loaded
processor. Corrected against transformers 5.17.0 in `vlm-train-tf5.sif`
(measured 2026-09-25):

- `max_soft_tokens` is a real knob and it is honoured. It accepts exactly five
  values — `(70, 140, 280, 560, 1120)` — and raises `ValueError` for anything
  else. There is no rounding: a request lands on a step or the run does not
  start.
- The grid is `patch_size` 16 × `pooling_kernel_size` 3, so one soft token covers
  a **48 px cell** against Qwen3-VL's 32. At step 280 the patch count before
  pooling is exactly 2520 = 280 × 3².
- A step therefore carries a fixed pixel area: 161 k / 323 k / 645 k / 1.29 M /
  2.58 M.

Matching on **pixels** — both models shown the same detail, Gemma spending fewer
tokens on it because its pooling is coarser — every level we use is reachable:

| Level | our budget | Qwen tokens | Gemma step | verified cost of a real image |
|---|---:|---:|---:|---|
| line | 262 144 px | 256 | **140** | 96 (a 2000×120 strip cannot fill the grid) |
| block of 6 | 1 048 576 px | 1024 | **560** | 560 (2000×700) |
| page | 2 097 152 px | 2048 | **1120** | 1092 (2400×3400) |

So Gemma reads the same pixels for a little over half the visual tokens. That is
the real hypothesis this arm tests, and it is an efficiency claim rather than a
handicap.

**What does not change is the cost model, and it has a consequence.** Gemma
charges its step per image whatever the image is: at step 280 a 2000×120 line
strip costs 272 soft tokens and a 2400×3400 page costs 266. A line is not cheap
on Gemma. Mixed granularity, where one processor budget serves every kind, is
therefore the expensive case on this family and not the cheap one — so a Gemma
arm belongs on a single granularity first. `train_qlora` now says so and stops
pre-scaling per kind when the budget is stepped: shrinking a crop before a
processor that resizes to its own grid anyway only removes detail.

**And half the token count does not buy half the cost.** Measured on two idle
RTX 4090s in the same fifteen minutes, same corpus, same split, same effective
batch of 16 as micro-batch 2 × accumulate 8, line granularity, 4-bit:

| | 200 optimizer steps | s/step | GPU memory |
|---|---:|---:|---:|
| `Qwen/Qwen3.5-4B` (4.66 B) | 873 s | **4.37** | 9 876 MiB |
| `google/gemma-4-E4B-it` (8.00 B) | 903 s | **4.52** | 21 624 MiB |

Gemma is 3.4 % *slower*, not faster. At nearly double the parameters that is
still a good showing and consistent with 140 visual tokens against 256 — but the
efficiency claim above is about tokens, and it does not carry over to wall-clock.

The number to plan with is the third column: **2.2× the memory at an identical
micro-batch.** A Gemma arm at micro-batch 16 on an 80 GB H100 is not obviously
safe, and that has to be measured before thirty GPU-hours are booked against it.

(An earlier reading of these runs had Gemma *faster* per sample. It compared
micro-batch 2 against micro-batch 16, and the batch-16 figure came from a run
that died of OOM seconds later — almost certainly already thrashing. Both numbers
are discarded.)

## 5. The confound to avoid before it is created

`VlmTrainParams.load_in_4bit` defaults to `True`, but both corpora's specs set it
to `false` — the four measured points in §1 are **bf16**. At 27 B that would be
55.6 GB of weights on an 80 GB H100 before activations; 4-bit NF4 is ≈ 15.6 GB.

If the ladder switches quantisation halfway up, the size curve becomes a
size-and-quantisation curve and no later analysis can separate them.

**Decision: run the new ladder at `load_in_4bit: true` throughout, and include a
`Qwen3.5-4B` arm at 4-bit on each corpus.** That arm costs ~6 h and buys a
measured bf16→4-bit delta at a point whose bf16 number we already know, which is
what makes the old and the new points comparable at all. `fanout.py` takes the
override per arm, so one corpus still serves every arm.

## 6. The runs, in order

**Medieval first, and it is the screen.** The two `train.jsonl` files are
**306 582** and **873 468** samples (measured), so a medieval arm costs about 35 %
of a 19th-century one. Running the ladder there first ranks the bases on a corpus
we want measured anyway — a better screen than a subsample, because its result is
a result rather than a proxy.

```bash
# one fan-out per corpus, on the login node, inside the tf5 container
apptainer exec --bind /storage/research --bind /scratch --bind /rs_scratch \
  --env PYTHONPATH=$HOME/training-atr-models/src:$HOME/training-atr-models/engines \
  $HOME/ubelix/vlm-train-tf5.sif \
  /opt/vlm-train/bin/python $HOME/training-atr-models/ubelix/fanout.py \
  /scratch/network/users/$USER/runs/jobs \
  20260915T055400Z-qwen3vl-medieval-german-v3 \
  ladder-med-qwen35-4b-nf4=Qwen/Qwen3.5-4B,load_in_4bit=true \
  ladder-med-qwen35-9b=Qwen/Qwen3.5-9B,load_in_4bit=true \
  ladder-med-qwen35-27b=Qwen/Qwen3.5-27B,load_in_4bit=true \
  ladder-med-qwen38-27b=Qwen/Qwen3.8-27B,load_in_4bit=true \
  ladder-med-gemma4-e4b=google/gemma-4-E4B-it,load_in_4bit=true \
  ladder-med-gemma4-12b=google/gemma-4-12B-it,load_in_4bit=true
```

Each printed job id is then a stage-2 submission, and every Gemma or Qwen3.5 arm
needs the transformers 5.x container:

```bash
JOB_ID=<one of them> SIF=$HOME/ubelix/vlm-train-tf5.sif \
  ubelix/submit.sh ubelix/train.sbatch -- \
  --partition=gpu --qos=job_gratis --cpus-per-task=12 --time=15:00:00
```

The same two commands with `20260916T090417Z-qwen3vl-german-xix-v2` and
`ladder-xix-…` ids run the 19th-century half, for the arms that survive.

**Then granularity, for the winner only.** Blocks, pages and the mix are a
four-way measurement per model (`scripts/eval_granularity.py` in the serving
repo). Crossing that axis with eight bases and two corpora is 64 measurements to
answer a question whose shape we already know. Do it once on whatever the ladder
selects, and once on the best Gemma — because §4's claim is precisely about that
axis.

**Stop rule.** Stop climbing when a doubling of parameters buys less than 0.5
points, or when the winning arm cannot be served (§8) and its margin over the
best servable arm is under 1 point. Both thresholds are judgements, not
measurements; they are written down so the decision is made before the numbers
arrive rather than after.

## 7. Queue and cost

`job_gratis` gives **one** H100 for up to 96 h; `job_gpu_preemptable` gives four
for 24 h and can be interrupted at any time, which `train_resumable.sbatch` and
`ATR_TRAIN_PREEMPTABLE=1` make survivable (checkpoints every 200 steps).

The ladder is twelve GPU jobs and one H100, so the queue, not the GPU budget, is
the constraint. Estimated GPU hours (**assumed**, scaled from the measured 4 B
runs by parameter count at 4-bit):

| Corpus | Arms | Each | Total |
|---|---|---:|---:|
| medieval (≈⅓ the lines) | 6 | 2–10 h | ~30 h |
| 19th century | survivors, 3–4 | 6–30 h | ~60 h |

No prepare, no CPU-minute cap, and the arms of one corpus are independent — so
the medieval six are the natural candidates for `job_gpu_preemptable`, four at a
time.

## 8. Serving is a separate gate, and it is lower than the training gate

idhefix has two A40s at **45 GB** each (measured), which caps a served bf16 model
near **12 B** on one card — and card 1 is usually most of the way full already.

- 9 B bf16 ≈ 19 GB, `gemma-4-12B-it` ≈ 24 GB — both fit.
- 27 B bf16 ≈ 56 GB — **does not fit.** It needs FP8 (`Qwen/Qwen3.8-27B-FP8`
  exists, ≈ 28 GB) or INT4, or tensor parallelism across both cards, which would
  take the box's second GPU away from everything else.

vLLM 0.29 in `.venvs/vllm-next` supports every architecture in §3 —
`Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForConditionalGeneration`,
`Gemma4ForConditionalGeneration`, `Gemma4UnifiedForConditionalGeneration`
(measured). The serving question is memory and the LoRA-on-quantised-base path,
not architecture support.

**A 27 B arm may therefore be measurable and not deliverable.** That is an
acceptable outcome — it answers question 1 — but it must not be promised as a
production model before an FP8 or INT4 adapter path has been served once. The
ladder's deliverable is a number; the deliverable *model* is the best arm that
fits on an A40.

## 9. What would make this wrong

- **The 19th-century benchmark is 2 751 isolated lines.** A base that is better
  at holding a page together would show nothing there. §6's last step exists for
  that, but the selection is still made on a line-level number.
- **The micro-batch is a free variable nobody has pinned.** The measured
  baselines ran at batch 16 × accumulate 1; the mixed run and the arms above run
  at 2 × 8. The effective batch is the same, so the *result* should be, but the
  cost is not: 4.37 s/step at micro-batch 2 is better per sample than the batch-16
  figure it replaced — on a card where batch 16 did not fit, so the comparison is
  not clean. On an H100 both fit, and one 200-step probe would settle it for the
  whole ladder. Worth doing before the big arms, not after.
- **Watching a run needs the checkpoint directory, not the log.**
  `logs/train.log` is only flushed when the subprocess ends, so a running job is
  invisible in it — six minutes of silence looks like a hang and is not.
  `runs/checkpoints/<job>/checkpoint-*/trainer_state.json` carries `global_step`,
  `max_steps` and the loss history, and the timestamps of two checkpoint
  directories are the cheapest honest throughput measurement available.
- **One seed, one epoch, no error bar.** This project has never measured its own
  run-to-run variance, and the differences at the top of the ladder may be
  smaller than it. Before reading a 9 B-vs-27 B gap as real, repeat one 4 B arm
  at a second seed. Roughly 6 GPU-hours for the right to interpret everything
  above it.
- **The extrapolation in §1 is a straight line through three points on a log
  axis.** Scaling curves bend. If 9 B lands at 6.5 % rather than 5.7 %, the
  honest conclusion is that this corpus saturates around 4–9 B and the ladder
  stops there rather than being pushed to 27 B to see.
- **`load_in_4bit` at 27 B is untested in this repo.** Every measured run is
  bf16. The 4 B arm in §5 is the control that turns that from an assumption into
  a number.
- **Gemma trains.** As of 2026-09-25 the first arm is at step 1 200 of 19 162 on
  the medieval corpus, loss 6.26 → 4.44 → 2.3 over the first 1 200 steps. Four
  things had to be fixed to get there, and each of them was invisible until the
  one before it was out of the way (#92, #94, #97, #98). What is still open is
  whether it *converges* to something competitive, which only the run answers.
- **The earlier form of this caveat, kept because the order was the lesson.**
  Two of the three things that could have stopped Gemma were settled off the GPU:
  the budget can be set (§4), and the assistant header is derivable — after a fix. Asked for a generation
  prompt, Gemma 4 emits `<|turn>model\n<|channel>thought\n<channel|>`, an empty
  thinking channel that vanishes once the assistant turn has content, so the
  header the collator looked for occurred in no training sample and the guard
  would have refused the first batch of a queued 12B job. `assistant_header_ids`
  now derives it from the render the loss is computed over instead; verified
  against five bases, and the Qwen headers are byte-identical to what the old
  code produced. What is still untested is convergence, and whether the default
  LoRA targets are the right modules for this family. The first medieval Gemma
  arm remains as much a smoke test as a measurement.

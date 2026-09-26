# Tuning Gemma for handwriting — a plan, and what it rests on

`gemma-4-E4B-it` reads our medieval corpus at **16.88 %** CER where
`Qwen3-VL-4B-Instruct` reads the same 200 samples at **13.65 %**
([BASE_MODEL_LADDER.md](BASE_MODEL_LADDER.md) §4a). Selection noise is 0.6 points,
so the gap is real. Every hyperparameter in both runs was chosen for Qwen, so the
gap measured "worse as a drop-in", which is a narrower claim than "worse".

This plan tests the narrower claim. It is ordered by expected effect over cost,
one factor at a time, because the ladder has already shown what happens when two
things change together.

**Every number in §1 was measured here on 2026-09-25/26.** Every claim in §2 comes
from an external source, is attributed, and is a *claim* until one of the
experiments in §3 turns it into a number. Where the two disagree, §1 wins — it was
measured on this stack.

## 1. What we already know, from our own runs

| | measured | how |
|---|---|---|
| E4B is **~4.5 B effective**, not 8 B | `num_hidden_layers` 42, `vocab_size_per_layer_input` 262 144, `hidden_size_per_layer_input` 256 → ≈2.8 B in per-layer embedding tables alone | its `config.json` |
| Our pipeline gave Gemma **140 soft tokens** per line | `apply_visual_budget` maps our 262 144 px line budget onto the cheapest legal step whose capacity covers it | end-to-end against the real processor |
| Gemma's own default is **280** | `processor_config.json`, `max_soft_tokens: 280` | ditto |
| The budget **binds at every step** | a 2000×120 line strip costs 96 tokens at step 140, 272 at 280, 480 at 560, 1088 at 1120 | ditto |
| …but above 280 it is **upscaling** | step 280 carries 645 120 px; a line crop is ≈240 000 px | arithmetic on the measured grid (patch 16 × pooling 3) |
| `k_proj`/`v_proj` exist on only **24 of 42** layers | `num_kv_shared_layers = 18`; module count 24 against 42 for `q_proj`/`o_proj`/MLP. Qwen3-VL-4B: 36 of 36 everywhere | empty-weights load, counted |
| **No KV-sharing/`use_cache` regression** on transformers 5.17.0 | logits with `use_cache=True` and `False` differ by **0.0**; same top token, same 0.8321 probability | one forward pass on an RTX 4090 |
| The run converged normally | loss 6.26 → 4.44 → 2.3 → **1.117** over 19 162 steps, `truncated_at_cap` 0, length ratio 0.9915 | its own trainer state and report |
| **The evaluation prompt contains a block training never saw** | `transcribe` uses `add_generation_prompt=True`; for E4B that render ends `<\|turn>model\n<\|channel>thought\n<channel\|>`, while the training render is plain `<\|turn>model\n` | both renders, side by side |
| Memory 2.2× Qwen at the same micro-batch, 3 % slower per step | 21 624 vs 9 876 MiB; 4.52 vs 4.37 s/step | two 4090s, same fifteen minutes |

Two of those are self-inflicted and are the reason this plan exists: Gemma was
given **half the visual tokens it ships with**, and it was **prompted at
evaluation with a sequence absent from its training**. Neither is a Gemma defect.

## 2. What the external guidance says

Collected 2026-09-26. Treated as claims, not settings.

**The official recipe** — `ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora`
(updated 2026-06-19), which names the Gemma 4 sizes explicitly — differs from ours
in six places: `r=16` with `alpha=16` (we use 64/128), `target_modules="all-linear"`
(we use Qwen's seven names), `modules_to_save=["lm_head","embed_tokens"]` (we train
neither), `lr_scheduler_type="constant"` (we use cosine with 5 % warmup),
`max_grad_norm=0.3` (we use 1.0), and `adamw_torch_fused` (we use
`paged_adamw_8bit`). It sets no `attn_implementation` and no `max_soft_tokens`.

**The budget is the largest documented lever.** The Gemma 4 technical report
(arXiv:2607.02770) gives OmniDocBench edit distance — lower is better — at two
budgets:

| base | @1120 | @280 |
|---|---:|---:|
| E2B | 0.290 | 0.496 |
| **E4B** | **0.181** | **0.307** |
| 12B (encoder-free) | 0.164 | 0.408 |
| 31B | 0.131 | 0.201 |

For E4B that is a 41 % relative reduction. The report does not explain why and
does not recommend a budget per task; no official source says "use 1120 for OCR".
Corroborated from the other side by ollama issue #17152 (13 Jul 2026), where a
screenshot at 280 yields "no discernible text" and reads correctly at 1120.

**Train and infer at the same budget** — asserted by a Gemma-4-specific but
uncited blog post (datature.io, 8 Apr 2026). Worth treating as a hypothesis to
test rather than a rule, precisely because it is uncited.

**"Gemma needs eager attention" is Gemma 2 legacy.** The warning was removed from
Gemma 3 in transformers PR #40744 (8 Sep 2025); `Gemma4TextConfig.final_logit_softcapping`
defaults to `None`, so the reason is gone. Keep sdpa.

**Excluding the towers is endorsed everywhere.** peft issue #3129 documents
`Gemma4ClippableLinear`; Unsloth's Gemma 4 guide says start with
`finetune_vision_layers=False`. Our regex exclusion is the recommended posture,
not a workaround.

**`<bos>` may be duplicated** — the template renders it and the tokenizer's
post-processor can prepend another. Cheap to check, so §3.0 checks it.

**No published Gemma HTR fine-tune with CER exists.** The only Gemma OCR write-up
is LaTeX with loss only, and its author reports the model learning the dataset's
rendering style rather than the task. CHURRO, the obvious comparison, is built on
Llava-1.5 and does not evaluate Gemma.

## 3. The experiments, in order

The anchor for every one of them is **13.65 % (Qwen) and 16.88 % (Gemma)** on the
same 200 samples of the medieval corpus. **A change is a result only if it moves
the CER by more than 0.6 points.**

### 3.0 Free checks, no training (an afternoon)

- **Score the existing adapter without the empty thought block.** It was trained
  without one and evaluated with one. If this alone moves the number, the 16.88 %
  is partly an artefact of our own prompt.
- **Score the existing adapter at 280 and 560 soft tokens.** It trained at 140.
  Two outcomes, both informative: better means the budget was the handicap and
  train/inference matching is not sacred; worse means matching matters and the
  retraining in §3.1 is required rather than optional.
- **Assert the token ids do not begin `[2, 2, …]`** and that the image precedes
  the instruction in the rendered prompt.

Each is one 4090 job of ten minutes against an adapter already on disk. Nothing
here needs the H100 queue.

### 3.1 The budget, retrained (one factor)

Retrain E4B on the medieval corpus at **280** — its own default, and the step our
pixel-matching took away. Then, only if 280 helps, at **560**, which is the first
step that upscales a line crop well past its own resolution and therefore tests
whether the report's document-level gain survives at line level at all.

This is the single change with cited evidence *and* a measured self-inflicted
cause. It is first.

### 3.2 The adapter scope (one factor)

`target_modules="all-linear"`, towers still excluded. Two independent reasons,
which is unusual enough to note: the official recipe uses it, and our own count
shows Qwen's seven names leave **18 of 42 layers without k/v adapters** on this
architecture. Rank stays at 64 so that this measures scope and not capacity.

### 3.3 Rank and scaling (one factor)

`r=16, alpha=16` against our `r=64, alpha=128`. Cited as the recommended pair;
no source compares the two on a Gemma vision model, so this is a genuine A/B
rather than a correction.

### 3.4 Trained embeddings (one factor, expensive)

`modules_to_save=["lm_head","embed_tokens"]`. The official recipe's most
surprising element and the costliest: 262 144 × 2 560 is ~0.67 B parameters in
fp32 master weights. Our own contract already warns about exactly this for Qwen's
151 k vocab. Worth one run because the recipe is official, and worth measuring the
memory before booking a GPU for it.

### 3.5 Schedule and clipping (one cheap factor, bundled)

`lr_scheduler_type="constant"` and `max_grad_norm=0.3` together, because both come
from the same recipe and neither has an independent rationale here. If the pair
moves nothing, they stay as they are.

### 3.6 The base, if E4B stays behind

If E4B remains more than a point behind Qwen after §3.1–3.3, the family question
moves to **`gemma-4-12B-it`**, which is encoder-free and the most
budget-sensitive model in the line (0.408 at 280, 0.164 at 1120) — meaning it must
be tried *with* the raised budget or not at all. An arm is already training at the
old 140-token budget; read it as a control, not as 12B's answer.

## 4. What this plan will not do

- **It will not tune the vision tower.** No source recommends it first, it needs a
  `Gemma4ClippableLinear` monkeypatch, and no evidence was found that it helps OCR.
- **It will not touch the per-layer embedding tables.** ~2.8 B parameters that no
  LoRA target reaches and that no guidance addresses in either direction.
- **It will not chase the loss.** An external guide calls a training loss of 13–15
  "normal" for this family; ours ran 6.26 → 1.117 and the KV-sharing regression
  that produces such losses is measurably absent here. A loss number from another
  stack is not a target.
- **It will not change two things at once**, and it will not report a move under
  0.6 points as a result.

## 5. Cost

§3.0 is three ten-minute jobs on an idle consumer GPU. Each experiment in
§3.1–3.5 is one medieval-corpus arm: ~24 h on an RTX 4090 with requeues, or ~8 h
on an H100. Five experiments is therefore one H100-day or a handful of 4090-days,
and the 4090s have been empty every time this was checked — which is also what
makes §3.0 worth doing first rather than reasoning about.

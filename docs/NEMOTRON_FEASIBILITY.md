# Llama-Nemotron: the feasibility package (#135, F5)

**Neither size can be trained in the container we have today, and the reasons
differ.** `Llama-3.1-Nemotron-Nano-VL-8B-V1` cannot even have its configuration
constructed: that step reaches for `timm`, which the image does not carry.
`NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` gets much further — config, processor, chat
template, assistant header and stop token all work — and then fails to import its
language model, which requires `mamba-ssm`. Both refusals are hard, immediate, and
measured below.

Beyond the container there is a second finding, and it is the one that decides the
shape of the epic: **this family has no pixel budget.** It tiles. The only control
is `max_num_tiles`, a *ceiling* on a grid the vendor's own aspect-ratio heuristic
picks inside. For pages that lands within 12.5 % of our budget. For line strips it
does not land at all: a 481×202 crop costs 256 visual tokens only if it is squashed
into a square, 768 at the next reachable grid, and 2 816 at the vendor's default.
So **F6 (lines) cannot be run under the epic's "same pixel budget, change only
`base_model`" contract**, and F7 (pages) can.

Everything here was measured on 2026-10-01 against `vlm-train-tf5.sif`
(transformers 5.17.0, torch 2.8.0+cu128) on UBELIX, with no weights downloaded —
17 GB for the 8B and 27 GB for the 12B, and the answer arrived without them.

## How to repeat it

```bash
bash ~/ubelix/prefetch_code.sh nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 \
    nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base nvidia/C-RADIOv2-H
bash ~/ubelix/run_preflight_nemotron.sh nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16
OFFLINE=1 bash ~/ubelix/run_preflight_nemotron.sh nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16
```

[`ubelix/preflight_nemotron.py`](../ubelix/preflight_nemotron.py) is stage A of the
eight-point pre-flight: the six points that need nothing but the small files. Points
5 (`target_modules` against the real module tree) and 8 (a 4-bit forward and an
evaluation pass) need the weights and are stage B — worth paying for only if stage A
comes back clean, which it does not.

[`ubelix/prefetch_code.sh`](../ubelix/prefetch_code.sh) exists because
`prefetch_bases.sh` — which lives on UBELIX and has never been committed — cannot
prepare a `trust_remote_code` model at all: its
`allow_patterns` are `['*.json','*.txt','*.jinja','*.model','*.safetensors']`, with
no `*.py`. It would fetch the configuration and leave the vendor code behind, and
the gap would surface only on an offline compute node.

## 1. The container refuses both, for different reasons

The 8B dies inside `AutoConfig.from_pretrained`:

```
  File ".../Llama_..._8B_V1/configuration.py", line 35, in __init__
    vision_auto_config = get_class_from_dynamic_module(*vision_config["auto_map"]["AutoConfig"].split("--")[::-1])
ImportError: This modeling file requires the following packages that were not found
in your environment: timm. Run `pip install timm`
```

Its `configuration.py` resolves the vision tower's class *while building the config
object*, and the tower is `nvidia/C-RADIOv2-H`, a `timm` model in a different repo.
There is no code path to a Nemotron 8B config in this image.

The 12B's config builds, because it vendored the vision config instead of
resolving it: its `configuration.py` does `from .configuration_radio import
RADIOConfig`, where the 8B reaches across repos for the same class. That buys the
config, not the tower — `modeling.py` still calls `AutoModel.from_config(
config.vision_config, trust_remote_code=True)`, so the 12B needs `timm` too, one
step later. What it fails on first is its language model:

```
== 2. the resolved model class: can our trainer call its forward?
   -> ImportError: mamba-ssm is required by the Mamba model but cannot be imported
```

Inventory of the image against what this family wants:

| package | in `vlm-train-tf5.sif` | needed by |
|---|---|---|
| `torchvision` 0.23.0 | yes | 12B image processor |
| `timm` | **absent** | `C-RADIOv2-H` → both sizes |
| `einops` | **absent** | RADIO's modules import it — inferred, not yet named by a failure |
| `mamba_ssm`, `causal_conv1d` | **absent** | 12B language model |
| `flash_attn` | **absent** | both configs ask for `flash_attention_2` by default |

`timm` and `einops` are ordinary wheels: a cheap edit to
[`ubelix/vlm-train-tf5.def`](../ubelix/vlm-train-tf5.def) and one rebuild.
`mamba-ssm` and `causal-conv1d` are CUDA extensions that compile against the pinned
torch, which is a different order of work and can fail. **So the container cost is
asymmetric: the 8B needs the cheap rebuild, the 12B the expensive one.**

## 2. Our loader cannot name either model

```
   auto_map:      {'AutoConfig': 'configuration.NemotronH_Nano_VL_V2_Config',
                   'AutoModel': 'modeling.NemotronH_Nano_VL_V2',
                   'AutoModelForCausalLM': 'modeling.NemotronH_Nano_VL_V2'}
   AutoModelForImageTextToText in auto_map: False
   resolves in the static mapping: NOT IN MAPPING
```

`train_qlora.py:570` and `evaluate_qlora.py:123` both load through
`AutoModelForImageTextToText`. Neither Nemotron registers for it, under either
route. A second loader path — `AutoModel`/`AutoModelForCausalLM` with
`trust_remote_code=True` — would be needed in both places, and it is the first time
this pipeline would carry two loaders.

The 8B has no `AutoProcessor` at all; only `AutoImageProcessor` and
`AutoTokenizer`. Our collator's single call is
`self.processor(text=texts, images=[[image] for image in images], ...)`
(`train_qlora.py:268`), and the vendor's own example builds the prompt by calling
`model.chat(...)`, which is a generation-only path that expands the image tokens
inside the model. There is no training-shaped entry point for the 8B.

## 3. The training forward is not callable from our trainer

Measured against the resolved class for the 8B and the cached source for the 12B:

```
   forward accepts: pixel_values, input_ids, attention_mask, position_ids,
                    image_flags, past_key_values, labels, ...
     image_flags    yes
     num_patches    NO
   forward dereferences image_flags unconditionally: True
   forward calls torch.distributed.get_rank(): True
```

Both are true of both sizes. Each is fatal on a single-GPU QLoRA run:

- **`image_flags` is required and nothing produces it.** The processor returns
  `pixel_values` and `num_patches`; `forward` dereferences `image_flags.squeeze(-1)`
  with no `None` guard. Our collator would hand it `None`.
- **`num_patches` is returned and not accepted.** The collator passes the
  processor's output through, so it would arrive as an unexpected keyword.
- **`torch.distributed.get_rank()` is called unconditionally**, inside `forward`,
  to print a debug line. With no process group initialised it raises. Note that
  `generate()` avoids both — so inference works and training does not, which is
  exactly the shape of the Gemma 12B dtype bug: it strikes where a quick check does
  not look.

A fix is small (a collator branch, and a one-process group or a patched copy of the
vendor file) but it is *our* fix to a vendor file we do not control — see §7.

## 4. There is a knob, and it is a ceiling, not a budget

```
   line     262144 px -> VisualBudgetError: NemotronNanoVLV2ImageProcessor has none of
            size['longest_edge'], max_pixels or max_soft_tokens; there is no knob here
            to bound visual tokens with
```

`apply_visual_budget` refuses, correctly. What the processor does have is
`image_size=512`, `max_num_tiles=12`, `use_thumbnail=True`. An image is resized to a
whole number of 512×512 tiles on a grid chosen by aspect ratio, plus one thumbnail
tile whenever the grid is larger than 1×1.

The arithmetic is kinder than it looks. One tile is 512² = 262 144 px and carries
`(512/16)² × 0.5² = 256` tokens, so **one visual token carries 1 024 px — an
effective cell of 32 px, the same as Qwen3-VL** (olmOCR is 28). Same pixels really
is same tokens here, which is more than could be said for the olmOCR comparison.

What breaks it is that `max_num_tiles` bounds the grid without choosing it. The
vendor heuristic picks the grid, and the two sizes use different heuristics — the
8B also doubles every image before tiling.

## 5. Tokens per granularity: measured, not computed

Real samples: a 481×202 line crop from the medieval line corpus, and a 2775×4190
page fitted to the page budget at 1178×1779.

| image | `max_num_tiles` | tiles (incl. thumbnail) | visual tokens | our arms |
|---|---:|---:|---:|---:|
| line, line budget | 1 | 1 | **256** (squashed to 512×512) | 256 |
| line, line budget | 2 / 4 / 8 | 3 | 768 | 256 |
| line, line budget | 12 (default) | 11 | **2 816** | 256 |
| page, page budget | 4 | 3 (8B: 5) | 768 (8B: 1 280) | 2 048 |
| page, page budget | 8 / 12 | 7 | **1 792** | 2 048 |

**Pages are comparable.** At `max_num_tiles=8` a page costs 1 792 tokens against
our 2 048, 12.5 % less, and a 2:3 page is exactly a (2,3) grid, so nothing is
distorted. F7 can be run as a like-for-like arm with the token count reported beside
the result, as #142 established.

**Lines are not.** One tile is a square, so the only grid that hits the line budget
turns the sampled 2.4:1 strip into a 1:1 square — and a longer line into a worse
one. The next reachable grid is 3× the budget, and
the vendor default is 11×. There is no setting at which this family reads a line
strip at our line budget without distortion. Running F6 anyway would measure a
different experiment and report it as the same one — the error #77 already taught
this project to avoid.

## 6. What already works

Not everything is bad news. For the 12B, three of the eight points pass today —
3 (cell size), 4 (assistant header) and 7 (stop tokens):

```
   rendered prompt contains '<image>': True
   render: '<s><SPECIAL_10>System\n\n<SPECIAL_11>User\n<image>\nTranscribe the
            handwritten text in this image exactly as written.\n<SPECIAL_11>Assistant\n<think></think>'
   header ids: [11, 102897, 1010] -> '<SPECIAL_11>Assistant\n'
   present in a real training render: True
   stop ids: [12] -> ['<SPECIAL_12>']
```

The chat template renders, emits `<image>` where the processor expects it, and the
assistant header our loss mask depends on is present in a real training render.
Stop tokens resolve. Offline, all of this still works from the pre-downloaded cache,
which means the download recipe in `prefetch_code.sh` is complete.

Two caveats inside the good news:

- **The tokenizer warns that it will tokenise incorrectly.** `The tokenizer you are
  loading ... with an incorrect regex pattern ... This will lead to incorrect
  tokenization. You should set the 'fix_mistral_regex=True' flag`. For a
  transcription task that is not cosmetic, and nothing in our code sets that flag.
- **The tokenizer has no pad token, and our fallback is the wrong one here.**
  `padding=True` fails outright: `Asking to pad but the tokenizer does not have a
  padding token`. In a real run it would not fail — `train_qlora.py:696` sets
  `pad_token = eos_token` when pad is missing — and that is worse. The collator
  masks the pad id out of the loss (`labels[labels == pad_token_id] =
  ignore_index`), so with pad and eos the same token it masks the real end of
  every answer, and the model is never trained to stop. Measured: Nemotron 12B
  `pad=None`, `eos='<SPECIAL_12>'`; both Qwen bases carry distinct tokens
  (`<|endoftext|>` against `<|im_end|>`), so this fallback has never fired in a
  run so far. It is the same shape as the Gemma dtype bug — silent until
  generation, after the training is paid for.
- The template opens and closes `<think></think>`. This is a reasoning model, and
  the same property already cost us effort on the serving side.

## 7. The vendor code is unpinned

```
[transformers] A new version of the following files was downloaded from
https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16:
- modeling_nemotron_h.py
- evs.py
. ... To avoid downloading new versions of the code file, you can pin a revision.
```

For every other family in this project, pinning a revision (#143) protects the
*data*. Here it would protect the *code that runs*: the 12B pulls executable files
from three repos — itself, `nvidia/C-RADIOv2-H` and
`nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base` — and two of those are named nowhere in
its own file listing. A run whose results we intend to publish must pin all three,
and our `DatasetSpec.revision` has no counterpart for a base model.

## 8. What a run would cost, and what it would buy

Before a single training step:

1. Container: `timm` + `einops` (cheap) for the 8B; `mamba-ssm` + `causal-conv1d`
   (CUDA build, uncertain) for the 12B. One rebuild either way.
2. A second loader path in `train_qlora.py` and `evaluate_qlora.py`.
3. A collator branch: build `image_flags`, drop `num_patches`, and add a pad token
   that is **not** the eos token.
4. A patched or guarded copy of the vendor `forward`, for `get_rank()`.
5. A fourth branch in `apply_visual_budget` for `max_num_tiles`, documented as a
   ceiling rather than a budget.
6. Base-model revision pinning across three repos.
7. `fix_mistral_regex=True` for the 12B.

Against that: neither model is pre-trained on document OCR the way olmOCR is, which
was the substantive reason to run olmOCR first. The 12B is a 12 B reasoning model
and would be compared against 4 B arms, so the honest claim would again be "bigger
**and** different", not "better".

## Recommendation

**Do not run F6.** The line arm cannot be made comparable; say so in the epic rather
than producing a number that invites the wrong comparison.

**Hold F7 behind olmOCR's results.** The page arm is technically sound — 1 792
tokens against 2 048, no distortion — and the work above is a day or two, most of it
reusable. But it is only worth spending once the olmOCR arms have said whether a
different family moves the number at all. If olmOCR's pages land near
qwen3.5-4b's 0.4160, a seventh family is a cost without a question behind it.

**If F7 does go ahead, take the 12B, not the 8B,** despite the heavier container
work: it has a real `AutoProcessor`, a working chat template, a locatable assistant
header and resolvable stop tokens, where the 8B has none of those and would need a
processor we write ourselves.

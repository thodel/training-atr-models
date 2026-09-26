"""Training wire contracts — **pydantic only, zero heavy deps**.

Shared by the gateway proxy (#35) and the trainer service (#34), so this module
follows the same rule as :mod:`atr_serving.contracts`: no yaml, no httpx, no ML.

The envelope is deliberately engine-agnostic (``engine`` + ``dataset`` + ``params``)
so TrOCR and VLM-LoRA jobs can reuse the store, the API and the prepare stage —
only ``params`` and the stage commands differ per engine. ``vllm`` (QLoRA
fine-tuning of a Qwen3-VL base) is the second backend to take that route; it
shares the job store, the state machine, the five stage names and the whole
prepare stage with kraken.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ── the default architecture (serving-atr-inference/docs/TRAINING_PLAN.md §3a) ─────────────────────
# "kraken+". Input block = batch 256, line height 64, variable width, grayscale.
# NOTE: kraken parses the leading 256 only into ``example_input_array`` — the real
# batch size comes from ``-B``. KrakenTrainParams keeps the two in sync.
# NOTE: kraken appends its own output layer (``O1c<codec+1>``) on a from-scratch
# run, so the trailing ``Cr255,1,85,1,1`` is a hidden layer of width 85, NOT an
# 85-symbol alphabet.
KRAKEN_PLUS_SPEC = (
    "[256,64,0,1 Cr4,2,8,4,2 Cr4,2,32,1,1 Mp4,2,4,2 Cr3,3,64,1,1 Mp1,2,1,2 "
    "S1(1x0)1,3 Lbx256 Do0.5 Lbx256 Do0.5 Lbx256 Do0.5 Cr255,1,85,1,1]"
)

# ── the VLM defaults ─────────────────────────────────────────────────────────
# Qwen3-VL-8B is what the serving box already *serves* (three fine-tunes of it are
# in its config/models.yaml at 18 GB resident), and what its scripts/merge_loras.py
# knows how to bake an adapter into. Training the model that machine can serve keeps
# the loop closed; the 30B-A3B MoE that lassberg/vlm_training targets is selectable
# but has nowhere to be served — vLLM 0.11 would need a whole card there.
VLM_BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

#: TrOCR fine-tunes always start from a pretrained encoder-decoder; there is no
#: from-scratch case. Mirrors VLM_BASE_MODEL so the two backends read alike.
TROCR_BASE_MODEL = "microsoft/trocr-base-handwritten"

#: Instruction given to the VLM for every training and evaluation example. It is
#: stored on the trained ModelSpec (``prompt``) so serving replays exactly the
#: wording the model was tuned on — a different prompt at inference is a silent
#: distribution shift.
VLM_PROMPT = "Transcribe the handwritten text in this image exactly as written."

#: Visual-token budget per sample kind, in pixels. A processor divides by the area
#: of one merged patch to get visual tokens, and **that area is model-specific**:
#: 28² is Qwen2-VL (patch 14 x merge 2), while Qwen3-VL is patch 16 x merge 2 = 32².
#: These figures carry the intended *token* counts — 256 for a line, 2048 for a page,
#: as in lassberg/vlm_training's collator — against Qwen3-VL's grid, because that is
#: what ``VlmTrainParams.base_model`` points at. The runtime does not trust this: it
#: re-derives the cap from the processor's own patch_size/merge_size and reports it
#: (``vlm_dataset.apply_visual_budget``), so a base with a different grid cannot
#: quietly train at another budget than the one written here (#86).
#: ``block`` (#57) — up to ``DEFAULT_BLOCK_LINES`` consecutive lines — sits between
#: the two. Measured on the 19th-c. corpus on 2026-09-22: a line crop is ~119 px
#: tall at its native size (crops are never upscaled, so they rarely fill their
#: 256 tokens); six such lines of ~2000 px fit 1024 tokens at ~0.85 of that
#: height, and a whole page fits 2048 at ~0.75.
VLM_PIXEL_BUDGET: dict[str, int] = {"line": 256 * 32 * 32, "block": 1024 * 32 * 32,
                                    "page": 2048 * 32 * 32}
#: Token budget per sample kind (prompt + image + transcription).
VLM_MAX_SEQ_LEN: dict[str, int] = {"line": 512, "block": 2048, "page": 4096}
#: Tokens the model may *generate* at evaluation, per sample kind. This has to
#: scale with granularity for the same reason the input budget does, and it did
#: not: a flat 256 was right for a line and cut a page in half (#92).
#:
#: The failure is invisible, which is what makes it dangerous — it surfaces as a
#: bad CER, not as an error. `qwen3vl-sg-missiven-v1` was recorded at CER 0.5921
#: with `length_ratio` 0.515; re-scored at 1536 tokens the same adapter gives
#: **0.2785** at `length_ratio` 1.027. Half the reference was never generated.
#:
#: A St. Gallen missive page averages 967 reference characters at roughly 2
#: characters per token in this orthography, so ~500 tokens; 1536 leaves room for
#: the long ones without inviting a runaway generation.
#:
#: A block of six lines is ~90 tokens at the 19th-c. corpus's ~3 characters per
#: token (p90 of a page is ~560); 768 leaves the long ones room.
VLM_MAX_NEW_TOKENS: dict[str, int] = {"line": 256, "block": 768, "page": 1536}
#: Transcription length past which a sample is dropped at compile rather than
#: trained on (#110). **This is not `VLM_MAX_SEQ_LEN` in other units** — the two
#: answer different questions, and conflating them is what cost eleven hours.
#:
#: `VLM_MAX_SEQ_LEN` is the budget the visual sizing targets; a sample over it is
#: reported and trained anyway, and samples at 4–8 k tokens trained fine.
#: This is the point where one sample's loss tensor stops being affordable at
#: all: cross-entropy upcasts the logits to fp32, so a sequence costs
#: ``tokens × 151,936 × 4`` bytes in a single allocation.
#: `20260908T101611Z-qwen3vl-german-pages-v1` died in its eval loop 11 h 24 m in,
#: at step 785 of 2355, on **one page of 32,477 characters** — 88× the median of
#: 383 — which tokenized to 14,411 tokens and asked for **8.16 GiB** at once.
#:
#: 8,000 was chosen from that corpus's own distribution (13,953 pages: median 383,
#: p90 ~2,200, p99 ~5,000, max 32,477). It drops **24 samples, 0.17 %**, and caps
#: the loss allocation at ~3.4 GiB. The next threshold down, 6,000, saves 0.5 GiB
#: and costs three times as many pages; 12,000 keeps 18 more pages and gives back
#: a third of the headroom.
#:
#: A "line" longer than 1,000 characters is not a line — it is a mis-segmented
#: block, and it was never going to train usefully.
VLM_MAX_SAMPLE_CHARS: dict[str, int] = {"line": 1000, "block": 3000, "page": 8000}
#: The kinds a sample can be. ``mixed`` is not one of them: it is a *job* that
#: draws from all three (#59), and every sample still carries one of these.
VLM_SAMPLE_KINDS: tuple[str, ...] = ("line", "block", "page")
#: Shares of a mixed corpus when the job does not say otherwise. Line-heavy on
#: purpose: the line is what the CER is measured against on the serving side, and
#: `block-v1` showed that a middle unit already carries much of the range
#: (training-atr-models#60).
DEFAULT_GRANULARITY_MIX: dict[str, float] = {"line": 0.5, "block": 0.3, "page": 0.2}

# A model id doubles as a directory name and a registry id — keep it boring.
MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

TrainEngine = Literal["kraken", "trocr", "vllm"]

JobStatus = Literal[
    "queued", "preparing", "compiling", "training", "testing", "registering",
    "completed", "failed", "cancelled",
]
#: The five stages every backend goes through. ``compile`` is the point where a
#: backend turns materialized pages into whatever its trainer eats: for kraken
#: that is ``ketos compile`` → ``.arrow``; for the VLM backend it is line
#: cropping / page assembly → ``.jsonl``. Deliberately the same name, because it
#: is the same step in the same state machine — a job's status means the same
#: thing whichever engine is running.
JobStage = Literal["prepare", "compile", "train", "test", "register"]

#: Stage → the status a job carries while that stage runs.
STAGE_STATUS: dict[str, JobStatus] = {
    "prepare": "preparing",
    "compile": "compiling",
    "train": "training",
    "test": "testing",
    "register": "registering",
}

TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── request ─────────────────────────────────────────────────────────────────

# ── exceptions ──────────────────────────────────────────────────────────────

class DatasetNotOnHub(LookupError):
    """The repo (or the pinned revision) is not there."""


class MultipleDatasets(AttributeError):
    """Raised when single-dataset code meets a job that has several (#40).

    An AttributeError subclass so it reads naturally where ``.dataset`` used to
    be an attribute, and so ``getattr(req, "dataset", None)`` still degrades.
    """


class DatasetSelectionError(ValueError):
    """Raised when a DatasetSpec selects nothing, or something unsafe."""


class ProjectListingError(RuntimeError):
    """Raised when the hub cannot be reached to enumerate projects."""


class DatasetSpec(BaseModel):
    """Which slice of a HuggingFace dataset to train on.

    Selection is by **project directory**, because
    ``dh-unibe/image-text_medieval-scripts_xiv-xv-xvi`` is ~6.6 TB laid out as
    ``data/<split>/<project>/*.parquet`` over 694 projects. Resolving this to
    explicit ``data_files`` globs (see :mod:`atr_training.hf_source`) is
    what keeps a job from pulling the whole repo onto a disk that cannot hold it
    — neither asteraix's ``/`` nor the share the HF cache lives on.
    """

    model_config = ConfigDict(validator=False)

    hf_repo: str
    split: str = "train"
    #: Project directories under ``data/<split>/`` used for training.
    #: Optional — a dataset with no project directories is selected whole.
    train_projects: list[str] = Field(default_factory=list)
    #: Project directories used for evaluation. Empty → split ``train_projects``
    #: by ``partition`` instead.
    eval_projects: list[str] = Field(default_factory=list)
    #: Cap on materialized pages (per role). None = everything selected.
    max_pages: int | None = Field(default=None, ge=1)
    #: Train fraction of the seeded page-level split, used only when
    #: ``eval_projects`` is empty. Mirrors ketos' ``-p/--partition``.
    partition: float = Field(default=0.9, gt=0.0, lt=1.0)
    seed: int = 42
    revision: str | None = None
    #: Source shape. ``page`` — the standard dh-unibe layout with PageXML columns —
    #: goes through materialization and cropping. ``line`` — a dataset that already
    #: has one row per line crop with a plain text column — skips cropping entirely
    #: (there is nothing to crop). The declared value is validated against the
    #: actual dataset schema on the hub before any data is downloaded, so a mismatch
    #: between what the user wrote and what the dataset actually has is an error,
    #: not a silent failure that surfaces 47 rows in, as the original lassberg
    #: config did.
    granularity: Literal["page", "line"] = "page"
    #: Select all project directories under ``data/<split>/`` on the hub.
    #: Requires ``max_pages`` to be set (a bounded selection is always deliberate;
    #: an unbounded one defeats the purpose of the guard). Mutually exclusive with
    #: ``train_projects``.
    all_projects: bool = False
    #: When set, pages are materialized in chunks of this many pages each:
    #: materialize → compile → discard → next chunk. Keeps peak disk bounded.
    #: Must be ≥ 1. None (default) disables chunking.
    chunk_size: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_all_projects_guard(self):
        if not self.all_projects:
            return self
        if self.max_pages is None:
            raise ValueError(
                "all_projects=True requires max_pages to be set. "
                "Pass max_pages to bound the selection; without it, all_projects "
                "would select all 694 projects (~6.6 TB), which is larger than the disk."
            )
        if self.train_projects:
            raise ValueError(
                "all_projects=True and train_projects cannot both be set. "
                "Use one or the other, not both."
            )
        return self


#: Alias so that callers can refer to the schema by name rather than as
#: ``DatasetSpec`` everywhere.
DatasetSelection = DatasetSpec


class DatasetCounts(BaseModel):
    """Per-dataset materialisation counts written to the job record."""

    hf_repo: str
    pages_written: int = 0
    pages_skipped: int = 0
    lines: int = 0
    chars: int = 0
    samples_written: int = 0
    #: Lines whose aspect ratio marks them as probably mis-segmented (#90).
    wide_lines: int = 0
    max_aspect: float = 0.0


class KrakenTrainParams(BaseModel):
    """Hyperparameters for a kraken recognition run (defaults = §3a of the plan)."""

    model_config = ConfigDict(protected_namespaces=())

    spec: str = KRAKEN_PLUS_SPEC
    batch_size: int = Field(default=256, ge=1)
    #: ``cosine``, not ``1cycle``: kraken 7.0.2 hands ``OneCycleLR`` a
    #: ``steps_per_epoch`` counted in **samples**, so the cycle is ``batch_size``
    #: times too long and no run ever leaves the warmup — every kraken run this
    #: project did before 2026-09-15 trained at a near-constant ``lrate / 25``
    #: (#96). ``1cycle`` stays in the type so job records written then still load;
    #: :func:`ketos_cmd.train_cmd` refuses to build a new run with it.
    schedule: Literal[
        "constant", "1cycle", "exponential", "step", "reduceonplateau", "cosine"
    ] = "cosine"
    lrate: float = Field(default=1e-4, gt=0.0)
    quit: Literal["early", "fixed"] = "fixed"
    epochs: int = Field(default=50, ge=1)
    min_epochs: int | None = Field(default=None, ge=0)
    lag: int = Field(default=10, ge=1)
    augment: bool = True
    normalization: Literal["NFD", "NFKD", "NFC", "NFKC"] | None = "NFD"
    normalize_whitespace: bool = True
    # coreml until #36 switches kraken_svc to kraken.models.load_models —
    # kraken 7.0.2 serves only CoreML through load_any, though ketos writes
    # safetensors by default.
    weights_format: Literal["safetensors", "coreml"] = "coreml"
    #: Codec handling when fine-tuning (``--load``); irrelevant from scratch.
    resize: Literal["add", "union", "both", "new", "fail"] = "fail"
    freeze_backbone: int | None = Field(default=None, ge=0)
    accumulate_grad_batches: int = Field(default=1, ge=1)
    pad: int | None = Field(default=None, ge=0)
    warmup: int | None = Field(default=None, ge=0)
    seed: int = 42
    workers: int = Field(default=8, ge=0)
    #: The unit sets CUDA_VISIBLE_DEVICES=1, so physical GPU 1 is cuda:0 here.
    device: str = "cuda:0"

    @model_validator(mode="after")
    def _one_cycle_needs_a_full_cycle(self) -> "KrakenTrainParams":
        """kraken derives the 1cycle length from ``--epochs``, so early stopping
        can cut the cycle off mid-ramp and leave the LR nowhere near its annealed
        value. If someone asks for both anyway, hold the run to the full cycle by
        defaulting ``min_epochs`` to ``epochs``.

        This was written believing OneCycleLR is stepped per batch. It is stepped
        per batch, but kraken sizes the cycle in **samples** (#96), so the cycle
        is ``batch_size`` times too long and cutting it short is the smaller of
        the two problems by far. The guard is kept for the day the sizing is
        fixed upstream; until then ``train_cmd`` does not let a 1cycle run start.
        """
        if self.schedule == "1cycle" and self.quit == "early" and self.min_epochs is None:
            object.__setattr__(self, "min_epochs", self.epochs)
        return self

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accumulate_grad_batches


#: Never adapt the vision or audio encoder. A regex, because peft matches an
#: exclusion *list* by suffix and would silently exclude nothing; see
#: :attr:`VlmTrainParams.exclude_modules`.
DEFAULT_EXCLUDE_MODULES = r".*\.(vision_tower|audio_tower)\..*"


class VlmTrainParams(BaseModel):
    """Hyperparameters for a QLoRA fine-tune of a Qwen3-VL base.

    Defaults follow ``lassberg/vlm_training`` (the pipeline these numbers were
    tuned in) except where this machine forces a different choice — each such
    deviation is noted on the field. The memory arguments below were made when
    training shared a card with the serving engines; since 16.09.2026 a run owns
    its card, and none of them has been re-measured.
    """

    model_config = ConfigDict(protected_namespaces=())

    #: ``line`` crops every transcribed ``TextLine`` out of the page by its
    #: PageXML ``Coords``; ``page`` trains on whole pages with the lines joined by
    #: newlines; ``block`` (#57) on runs of ``block_lines`` consecutive lines of
    #: one region, cropped as one image. Line is the default because it is what the
    #: CER is measured against on the serving side for these models, and because a
    #: page sample costs 8× the visual tokens for one training signal.
    #:
    #: A model trained at ``line`` reads lines only: whole pages came back at CER
    #: 0.98-1.00 (serving-atr-inference#165). Train at ``block`` or ``page`` for a
    #: model that reads more than one line per call.
    #: ``mixed`` trains on all three at once, in the shares of
    #: ``granularity_mix`` — the answer to a model that reads one unit and
    #: collapses (line-trained) or over-generates (page-trained) on the others
    #: (#59, measured in #60).
    granularity: Literal["line", "block", "page", "mixed"] = "line"
    #: Lines per sample at ``granularity: block`` and in a mix; ignored otherwise.
    block_lines: int = Field(default=6, ge=1, le=16)
    #: Shares per sample kind at ``granularity: mixed``; None takes
    #: :data:`DEFAULT_GRANULARITY_MIX`. Refused for any other granularity, so a
    #: mix that is silently ignored cannot happen.
    granularity_mix: dict[str, float] | None = None
    prompt: str = VLM_PROMPT

    # ── QLoRA ────────────────────────────────────────────────────────────────
    #: 4-bit NF4 + double quant. False = LoRA on a bf16 base, which did not fit an
    #: 8B beside the serving engines on the shared box. Not re-measured on a card
    #: this run owns alone.
    load_in_4bit: bool = True
    lora_r: int = Field(default=64, ge=1)
    lora_alpha: int = Field(default=128, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"]
    )
    #: Modules the adapters must not touch, as a **regular expression** matched
    #: against the full module path. A list would be the obvious type and is the
    #: wrong one: peft matches a list by suffix, so ``["vision_tower"]`` excludes
    #: nothing at all and does it silently (measured against peft 0.20.0) — the
    #: same shape of failure as the visual budget that looked set and was not
    #: (#86). A regex is the only form that can name a subtree.
    #:
    #: The default freezes the vision and audio encoders, which is what every run
    #: so far has done without saying so: ``target_modules`` matches by suffix
    #: too, and in Qwen3-VL-4B and Qwen3.5-4B all 252 and 128 matches are inside
    #: ``model.language_model`` — their towers do not use these names. Gemma 4's
    #: do: E4B has 112 in ``model.vision_tower`` and 36 in ``model.audio_tower``,
    #: wrapped in ``Gemma4ClippableLinear``, which peft refuses outright
    #: (training-atr-models#95). So this default changes nothing for any measured
    #: run and makes the families comparable rather than accidentally different.
    exclude_modules: str = DEFAULT_EXCLUDE_MODULES
    #: lassberg trains ``lm_head`` as well, which helps when the ground truth has
    #: characters the tokenizer rarely saw. It is off here by default: at Qwen3-VL's
    #: 151 k vocab that single module is ~620 M trainable parameters, whose fp32
    #: master weights and optimizer state add several GB. That was decisive on the
    #: shared card; on asteraix a run owns its card, so this is now a question about
    #: the run's own budget — worth trying when the ground truth has such characters.
    modules_to_save: list[str] = Field(default_factory=list)

    # ── optimisation ─────────────────────────────────────────────────────────
    #: Minimum epochs. With ``max_epochs`` set this is a floor, not a count.
    epochs: int = Field(default=3, ge=1)
    #: Ceiling for the continuation policy (#88). None = train exactly ``epochs``
    #: and stop, the old behaviour. Set it and the run keeps going while the
    #: validation loss still improves, the way kraken's ``--quit early`` does —
    #: ``kraken-medieval-shard00-std`` ran to epoch 66 under a 30-epoch schedule
    #: and peaked at 21, which a fixed count would have missed by 45 epochs.
    max_epochs: int | None = Field(default=None, ge=1)
    #: Evaluations without a real improvement before stopping.
    patience: int = Field(default=2, ge=1)
    #: How much better counts as better. Without it a loss that improves in the
    #: fifth decimal reads as improvement and the run never stops on its own.
    min_delta: float = Field(default=1e-4, ge=0.0)
    #: Page samples can exceed 4 k tokens, so >1 risks OOM; scale with grad accum.
    batch_size: int = Field(default=1, ge=1)
    accumulate_grad_batches: int = Field(default=16, ge=1)
    lrate: float = Field(default=2e-4, gt=0.0)
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_ratio: float = Field(default=0.05, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    optim: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True

    # ── budgets ──────────────────────────────────────────────────────────────
    #: None = the granularity's entry in VLM_PIXEL_BUDGET / VLM_MAX_SEQ_LEN.
    #: Checkpoint every N optimizer steps. 0 keeps the per-epoch strategy, which
    #: is right when an epoch is minutes. It is useless on a preemptable queue at
    #: corpus scale: one epoch over 10 M line crops is days, the walltime is 24 h,
    #: and a job preempted at hour 23 with epoch-only checkpoints resumes from
    #: nothing. Set it to something that costs a few minutes of redone work.
    save_steps: int = Field(default=0, ge=0)
    max_pixels: int | None = Field(default=None, ge=32 * 32)
    max_seq_len: int | None = Field(default=None, ge=32)
    #: Drop training samples whose transcription is shorter than this. 0 = off.
    #: **Training only** — validation keeps every sample, so a CER stays
    #: comparable with runs made before the filter existed.
    #:
    #: For the reason it exists, see the medieval corpus: 20.9 % of its lines are
    #: 1-3 characters (folio numbers, column figures, marginalia), against 0.8 %
    #: in the 19th-century one. A model trained on them learns to emit
    #: end-of-turn early and then writes only the first word of every real line -
    #: output/reference length ran 1.70 at 1-3 chars, 0.67 at 4-15, 0.14 at
    #: 16-40, for a corpus-wide length_ratio of 0.48 and a CER near 0.55 that had
    #: nothing to do with the hyperparameters.
    min_train_chars: int = Field(default=0, ge=0)

    # ── evaluation ───────────────────────────────────────────────────────────
    #: Generating a transcription per sample is ~1 s; a full validation split of
    #: 20 k lines would take longer than the training. Capped, and the cap is
    #: recorded in the report so a CER is never quietly measured on a subset the
    #: reader did not know about.
    eval_samples: int = Field(default=200, ge=1)
    #: None = the granularity's entry in VLM_MAX_NEW_TOKENS. An explicit value
    #: overrides it, the same contract ``max_pixels`` and ``max_seq_len`` follow.
    max_new_tokens: int | None = Field(default=None, ge=1)

    # ── run ──────────────────────────────────────────────────────────────────
    seed: int = 42
    workers: int = Field(default=4, ge=0)
    #: The unit sets CUDA_VISIBLE_DEVICES=1, so physical GPU 1 is cuda:0 here.
    device: str = "cuda:0"
    #: Weights & Biases run name; None = reporting off (the box has no wandb key).
    wandb_run: str | None = None

    @model_validator(mode="after")
    def _check_epoch_bounds(self):
        if self.max_epochs is not None and self.max_epochs < self.epochs:
            raise ValueError(
                f"max_epochs={self.max_epochs} is below epochs={self.epochs}. "
                "`epochs` is the floor and `max_epochs` the ceiling (#88)."
            )
        return self

    # ── benchmarks ──────────────────────────────────────────────────────────
    #: Optional held-out benchmarks evaluated alongside the validation split.
    #: Each entry is a (dataset, project) pair from which a JSONL is built in
    #: the ``prepare`` stage and evaluated in ``test``. The score surfaces on
    #: the model card as the primary quality claim; the split CER is preserved
    #: below it.
    benchmarks: list[BenchmarkSpec] = Field(default_factory=list)

    @property
    def continuation(self):
        """The policy this run trains under, or None for a fixed epoch count."""
        from atr_training.continuation import ContinuationPolicy

        if self.max_epochs is None:
            return None
        return ContinuationPolicy(
            min_epochs=self.epochs, max_epochs=self.max_epochs,
            patience=self.patience, min_delta=self.min_delta,
            greater_is_better=False,          # the metric is eval_loss
        )

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accumulate_grad_batches

    def kinds(self) -> tuple[str, ...]:
        """The sample kinds this job trains on, in a stable order."""
        if self.granularity != "mixed":
            return (self.granularity,)
        return tuple(k for k in VLM_SAMPLE_KINDS if k in self.mix())

    def mix(self) -> dict[str, float]:
        """The shares per kind, normalised to 1. A single granularity is its own mix."""
        if self.granularity != "mixed":
            return {self.granularity: 1.0}
        raw = self.granularity_mix or DEFAULT_GRANULARITY_MIX
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}

    def pixel_budget(self) -> int:
        """The ceiling for the processor. Each kind is fitted to its own budget
        before it reaches the processor (``kind_budgets``), so this is the largest
        of them — a smaller ceiling would shrink the page samples a second time."""
        if self.max_pixels:
            return self.max_pixels
        return max(VLM_PIXEL_BUDGET[k] for k in self.kinds())

    def kind_budgets(self) -> dict[str, int]:
        """Pixels per sample kind, which is what a mixed batch needs."""
        if self.max_pixels:
            return {k: self.max_pixels for k in self.kinds()}
        return {k: VLM_PIXEL_BUDGET[k] for k in self.kinds()}

    def sequence_budget(self) -> int:
        if self.max_seq_len:
            return self.max_seq_len
        return max(VLM_MAX_SEQ_LEN[k] for k in self.kinds())

    def generation_budget(self) -> int:
        """How many tokens evaluation may generate for one sample."""
        if self.max_new_tokens:
            return self.max_new_tokens
        return max(VLM_MAX_NEW_TOKENS[k] for k in self.kinds())

    @model_validator(mode="after")
    def _check_mix(self) -> "VlmTrainParams":
        if self.granularity_mix is not None:
            if self.granularity != "mixed":
                raise ValueError(
                    f"granularity_mix is only meaningful at granularity: mixed, "
                    f"not {self.granularity!r}"
                )
            unknown = sorted(set(self.granularity_mix) - set(VLM_SAMPLE_KINDS))
            if unknown:
                raise ValueError(f"granularity_mix: unknown kind(s) {unknown}; "
                                 f"pick from {list(VLM_SAMPLE_KINDS)}")
            if not self.granularity_mix or any(v <= 0 for v in self.granularity_mix.values()):
                raise ValueError("granularity_mix: every share must be greater than 0 "
                                 "(leave a kind out instead of asking for none of it)")
        return self


class TrOCRTrainParams(BaseModel):
    """Hyperparameters for a TrOCR fine-tune (microsoft/trocr-* or dh-unibe/*)."""

    model_config = ConfigDict(protected_namespaces=())

    # ── model ────────────────────────────────────────────────────────────────
    #: TrOCR is a fine-tune only — a base model is always required. Copied up to
    #: ``TrainRequest.base_model`` at validation, which is what the runner and the
    #: step-count guard read.
    base_model: str = TROCR_BASE_MODEL

    # ── seq2seq optimisation ─────────────────────────────────────────────────
    epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=1, ge=1)
    accumulate_grad_batches: int = Field(default=8, ge=1)
    lrate: float = Field(default=5e-5, gt=0.0)
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    optim: str = "adamw_torch"
    gradient_checkpointing: bool = True

    # ── generation at eval time ──────────────────────────────────────────────
    #: Tokens to generate at most per sample during evaluation.
    max_new_tokens: int = Field(default=256, ge=1)
    #: Beam width for beam search at eval. 1 = greedy.
    beam_size: int = Field(default=1, ge=1)
    #: Length penalty passed to ``model.generate``. Positive values encourage
    #: longer sequences; negative encourage shorter.
    length_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)

    # ── evaluation ───────────────────────────────────────────────────────────
    eval_samples: int = Field(default=200, ge=1)

    # ── run ──────────────────────────────────────────────────────────────────
    seed: int = 42
    workers: int = Field(default=4, ge=0)
    #: The unit sets CUDA_VISIBLE_DEVICES=1, so physical GPU 1 is cuda:0 here.
    device: str = "cuda:0"
    #: Weights & Biases run name; None = reporting off.
    wandb_run: str | None = None
    #: Precision. TrOCR's ViT backbone benefits from amp (bf16); fp32 is slower.
    precision: Literal["fp32", "fp16", "bf16"] = "bf16"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accumulate_grad_batches


#: Which params model each engine's ``params`` block is parsed as.
PARAMS_BY_ENGINE: dict[str, type[BaseModel]] = {
    "kraken": KrakenTrainParams,
    "trocr": TrOCRTrainParams,
    "vllm": VlmTrainParams,
}


class TrainRequest(BaseModel):
    """A submitted training job."""

    #: Store this run's compiled corpus in the artefact cache? None = whatever
    #: the box is set to (``TrainerSettings.artefact_cache``); set is set (#22).
    #:
    #: A per-run answer because the cost and the benefit are per run. Storing the
    #: corpus of ``qwen3vl-german-xix-v2`` took **14 of the job's 16.5 hours** on
    #: GPFS — 966,748 files, one metadata operation each — for a copy no run has
    #: ever read, and which that corpus will very likely never need again. The
    #: global switch could not say that: it is the box's policy, and a box serves
    #: runs that want the cache and runs that do not.
    artefact_cache: bool | None = None

    model_config = ConfigDict(protected_namespaces=())

    engine: TrainEngine = "kraken"
    model_id: str
    #: The datasets to train on. At least one; each entry selects a HuggingFace
    #: repo and a list of projects (or ``all_projects``).
    #: For backwards compatibility a single ``dataset`` field is also accepted
    #: and normalised to a one-element list.
    datasets: list[DatasetSpec] = Field(min_length=1)

    @property
    def dataset(self) -> DatasetSpec:
        """The single dataset — for the paths that only make sense with one.

        Reading this on a multi-dataset job **raises** rather than quietly
        returning the first. Most of the subsystem was written when a job had
        exactly one dataset, and silently handing back ``datasets[0]`` would turn
        every un-migrated call site into a wrong answer instead of an error: a
        model card naming one corpus for a model trained on three, a spec
        verified while two others were not. Loud is the whole point — this repo's
        recurring failure is a plausible number from a path nobody checked.
        """
        if len(self.datasets) != 1:
            raise MultipleDatasets(
                f"this job has {len(self.datasets)} datasets, so `.dataset` is "
                "ambiguous. The caller needs to handle `.datasets` explicitly — "
                "verifying, reporting or publishing only the first would be wrong "
                "in a way that reads as correct."
            )
        return self.datasets[0]
    #: kraken: registry id or raw Zenodo DOI to fine-tune from, None = from
    #: scratch. vllm: the HF base checkpoint the LoRA adapts — never None, since
    #: there is no such thing as training a VLM from scratch here; it defaults to
    #: :data:`VLM_BASE_MODEL`.
    base_model: str | None = None
    params: KrakenTrainParams | TrOCRTrainParams | VlmTrainParams = Field(
        default_factory=KrakenTrainParams
    )
    #: Run even when the step-count guard says the configuration cannot converge
    #: (#72). For a deliberate smoke test; the override is recorded on the job so
    #: the resulting CER is never read as an ordinary one.
    force: bool = False
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalise_datasets_and_params(cls, data):
        """Accept both the old ``dataset`` field and the new ``datasets`` list.

        ``dataset`` (singular) is normalised to a one-element ``datasets`` list so
        that every code path downstream only has to handle ``datasets``.
        """
        if not isinstance(data, dict):
            return data

        # Normalise singular ``dataset`` → ``datasets`` for backwards compat.
        if "datasets" not in data and "dataset" in data:
            data = {**data, "datasets": [data["dataset"]]}

        # Parse ``params`` as the model belonging to ``engine``.
        engine = data.get("engine", "kraken")
        model = PARAMS_BY_ENGINE.get(engine)
        if model is None:
            return data  # unknown engine: let the Literal produce the error
        params = data.get("params")
        if params is None:
            return {**data, "params": model()}
        if isinstance(params, dict):
            return {**data, "params": model(**params)}
        if not isinstance(params, model):
            raise ValueError(
                f"engine {engine!r} takes {model.__name__}, got {type(params).__name__}"
            )
        return data

    @model_validator(mode="after")
    def _check_model_id(self) -> "TrainRequest":
        if not MODEL_ID_RE.match(self.model_id):
            raise ValueError(
                f"model_id {self.model_id!r} must match {MODEL_ID_RE.pattern} "
                "(it becomes a directory name and a registry id)"
            )
        if self.engine == "vllm" and not self.base_model:
            object.__setattr__(self, "base_model", VLM_BASE_MODEL)
        if self.engine == "trocr" and not self.base_model:
            # TrOCR is a fine-tune only, and its base sat on the *params* model
            # while the runner and the convergence guard both read
            # ``request.base_model``. Left unfilled, a submitted job passed
            # ``--base-model None`` to the training script, and #72 judged it
            # "from scratch" — the 2,000-step floor rather than the 500 a
            # fine-tune needs. One field is the source of truth; params supplies
            # the default.
            object.__setattr__(self, "base_model",
                               getattr(self.params, "base_model", TROCR_BASE_MODEL))
        return self


# ── job record ──────────────────────────────────────────────────────────────


class BenchmarkSpec(BaseModel):
    """A named held-out evaluation set evaluated alongside the validation split.

    A benchmark differs from the validation split in one crucial respect: its
    documents do not appear in training at all. A value computed over material
    that overlaps training is not a held-out score and must not be published as
    one.

    This docstring used to say the held-out registry enforced that. It does not:
    :func:`atr_training.runner_base._reserve_eval_documents` drops only the
    ``docId``s hand-written into ``config/heldout_eval_documents.json``, and a
    benchmark named in a request never reaches that file — so for two months the
    evaluation ran unconditionally and ``measured_on`` was set regardless (#23).
    What enforces it is the check before the benchmark evaluation itself,
    ``vlm_train_svc.runner.Pipeline._refuse_contaminated_benchmark``, which
    compares the two manifests by document and fails the stage rather than
    producing a number. Naming a benchmark here does **not** hold its documents
    out of training; it asks for them to be checked.

    Benchmarks are optional: a run without benchmarks produces a split-only
    ``cer`` as before. A run with benchmarks produces both and ``publish.py``
    surfaces the benchmark score as the primary claim.
    """

    model_config = ConfigDict(protected_namespaces=())

    #: HF dataset id, e.g. ``dh-unibe/image-text_federal-minutes-testset``.
    hf_repo: str
    #: Project directory name inside that dataset, e.g. ``TEST_federal_minutes``.
    project: str
    #: Split the project lives under, as in ``data/<split>/<project>/``. Named
    #: rather than assumed: a set built to be held out is often published under
    #: ``test``, and a wrong guess here reads as "the benchmark has no pages".
    split: str = "train"
    #: Human-readable label for the model card. None = ``"<repo> / <project>"``.
    label: str | None = None


class Metrics(BaseModel):
    """The evaluation result, whichever backend produced it.

    kraken fills it from the ``ketos test`` report, which states *accuracies*;
    ``cer``/``wer`` are the error rates derived from them (fractions, not
    percent), so a lower number is always better. The VLM backend computes the
    same two directly from generated vs. reference text and leaves kraken's
    accuracy/edit-op fields empty — ``cer`` is the field both agree on, and the
    one the job store insists on before a job may complete.
    """

    chars: int | None = None
    errors: int | None = None
    char_accuracy: float | None = None      # percent, as kraken prints it
    char_accuracy_ci: float | None = None   # case-insensitive, percent
    word_accuracy: float | None = None      # percent
    cer: float | None = None                # 1 - char_accuracy/100
    wer: float | None = None                # 1 - word_accuracy/100
    insertions: int | None = None
    deletions: int | None = None
    substitutions: int | None = None
    #: Ratio of hypothesis chars / reference chars. 1.0 = no over-generation.
    #: > 1.0 = autoregressive model emitting past the reference; < 1.0 = premature
    #: stopping. Primary signal for the stopping vs. reading distinction (#55).
    length_ratio: float | None = None
    #: Truncated CER: hypothesis clipped to reference length before scoring.
    #: Isolates reading ability from stopping ability (#55).
    truncated_cer: float | None = None
    #: How many samples the score is over. The VLM backend caps evaluation
    #: (``VlmTrainParams.eval_samples``), so a CER without this number would hide
    #: that it was measured on a subset.
    samples: int | None = None
    #: Held-out benchmark scores. Present when the request included benchmarks.
    #: A benchmark score is the primary quality claim; ``cer`` is a secondary
    #: check on the validation split, which overlaps training projects.
    benchmark_cer: float | None = None
    benchmark_wer: float | None = None
    benchmark_samples: int | None = None
    #: The benchmark these scores were measured on. None when no benchmark ran.
    measured_on: str | None = None


class Progress(BaseModel):
    epoch: int | None = None
    epochs: int | None = None
    val_accuracy: float | None = None
    pages_written: int | None = None
    lines_written: int | None = None
    #: Training lines after the split — what the step-count guard divides by.
    #: Distinct from ``lines_written``, which counts every transcribed line found,
    #: evaluation included.
    train_lines: int | None = None
    #: What the configuration will actually cost, computed once prepare knows the
    #: line count (#72).
    steps_per_epoch: int | None = None
    total_steps: int | None = None
    #: VLM backend: training examples built in ``compile`` (one per cropped line,
    #: or one per page at ``granularity: page``). Distinct from ``lines_written``,
    #: which counts transcribed lines found while materializing — the two differ
    #: whenever a line is dropped for being unusable as a crop.
    samples_written: int | None = None
    #: Per-dataset materialisation counts (pages, skipped, lines, chars).
    #: Supersedes the flat counters above when multiple datasets are used.
    dataset_counts: list[DatasetCounts] = Field(default_factory=list)
    #: Training pages dropped because their document is reserved for evaluation
    #: (#98). 0 means the registry was consulted and matched nothing — not that
    #: nothing was checked, which is what the log line says.
    reserved_pages: int = 0
    #: Aspect ratio per character over the prepared lines — what the line-geometry
    #: guard compares against the VGSL spec. Recorded so it can travel with a
    #: cached artefact (#109), whose pages are deleted once it is stored.
    aspect_per_char: float | None = None
    #: VLM: samples dropped at compile for a transcription past
    #: ``VLM_MAX_SAMPLE_CHARS``, and the longest one seen *before* the drop (#110).
    #: Measured before so the record shows what the corpus contained, not what
    #: survived — the outlier is the finding.
    long_samples: int | None = None
    #: VLM: training samples dropped at compile for being shorter than
    #: ``min_train_chars``. Validation is never filtered, so this counts the
    #: training split alone.
    short_samples: int | None = None
    max_sample_chars: int | None = None
    #: The cached artefact (#109) this run's compiled corpus lives in, and
    #: whether this job built it or reused one. Set on both paths, because after
    #: compile the arrows are in the cache rather than in the job directory anyone
    #: would look in first — "which corpus did this run actually train on" has to
    #: stay answerable from the job record alone.
    artefact: str | None = None


class CodeVersion(BaseModel):
    """Which code did something: the git commit of the checkout it was imported from.

    A job record says what was trained and what it scored, and until #147 not
    with which code — so a CER could not be tied to the evaluator that measured
    it. On UBELIX that mattered twice: the container imports the training code from
    a checkout via ``PYTHONPATH``, a job runs whatever that checkout holds when it
    **starts**, and two runs started on a checkout a day behind ``main``.

    ``commit`` is ``None`` when the code does not run from a git checkout (an
    installed wheel) or git is unavailable: unknown is recorded as unknown, never
    guessed. ``dirty`` means tracked files differed from ``commit``.
    """

    commit: str | None = None
    dirty: bool | None = None

    def short(self) -> str:
        if not self.commit:
            return "unknown"
        return self.commit[:12] + ("+dirty" if self.dirty else "")


class StageRecord(BaseModel):
    name: JobStage
    status: Literal["pending", "running", "completed", "failed", "cancelled"] = "pending"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None
    log: str | None = None  # path, relative to the job dir
    #: The code this stage actually ran with, which on a resumed or requeued job
    #: need not be the code the job was created with (#147).
    code: CodeVersion | None = None


class TrainJob(BaseModel):
    """The on-disk record (``job.json``) and the API's job representation."""

    model_config = ConfigDict(protected_namespaces=())

    id: str
    request: TrainRequest
    #: The code the job was created with (#147). Each stage records its own in
    #: ``StageRecord.code``; see :meth:`code_summary`.
    code: CodeVersion | None = None
    status: JobStatus = "queued"
    stage: JobStage | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    #: The host whose service accepted the job (``TrainerSettings.host_id``), or
    #: ``ubelix`` for a record Slurm supervises. Only that host may spawn,
    #: reconcile or signal it, because ``pid`` below means nothing anywhere else
    #: (#15). None on the records written before the field existed; read those
    #: through :meth:`JobStore.host_of`, never directly.
    host: str | None = None
    #: PID of the detached runner process group leader — on ``host``.
    pid: int | None = None
    #: Why a queued job has not started yet (e.g. another job is running, or the
    #: GPU is busy). Never a failure — a queued job is still going to run.
    queued_reason: str | None = None
    progress: Progress = Field(default_factory=Progress)
    metrics: Metrics | None = None
    stages: list[StageRecord] = Field(default_factory=list)
    #: Set on failure — a human-readable reason, plus the log tail (never empty
    #: on a failed job; a job that fails silently is the bug we are avoiding).
    error: str | None = None
    log_tail: list[str] = Field(default_factory=list)
    #: Populated by the register stage.
    model_path: str | None = None
    #: The promotion gate (#36): True once a real transcription came back through
    #: the gateway for this model. False with a reason is a normal outcome, not a
    #: failure — the model is trained and registered, it is simply not advertised.
    promoted: bool | None = None
    #: What auto-publish did, in words — including why it did nothing (#88). The
    #: job record is where anyone looks for why a model is or is not on the hub.
    published: str | None = None
    #: What the register stage did with the shared registry, in words: where it
    #: wrote, or why it did not. A Slurm job on UBELIX never writes the registry
    #: (#17) — asteraix registers after the job ends — and this says so, with the
    #: steps to register by hand until that exists.
    registration: str | None = None
    promotion_reason: str | None = None
    #: Set when the step-count guard refused the configuration and ``force`` ran it
    #: anyway (#72) — so a CER from a run that was known not to converge is never
    #: mistaken for an ordinary one.
    convergence_override: str | None = None
    #: Set when the line-geometry guard refused the spec and ``force`` ran it
    #: anyway (#91, S10). Kept separate from ``convergence_override``: they refuse
    #: for unrelated reasons — too few optimizer steps versus too few CTC
    #: timesteps — and a record that conflated them could not say which.
    geometry_override: str | None = None
    #: Local scratch holding this run's checkpoints (outside the job directory —
    #: see TrainerSettings.checkpoint_root). Recorded so it is discoverable and
    #: can be cleaned up with the job.
    checkpoint_dir: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def code_summary(self) -> dict[str, dict[str, Any] | None]:
        """``{"created": …, "<stage>": …}`` — the code behind each part of this job.

        Written into ``metadata.json`` by the register stage, so a published CER
        names the commit of the evaluator that measured it (the ``test`` entry),
        not only the one the job was submitted with.
        """
        out: dict[str, dict[str, Any] | None] = {
            "created": self.code.model_dump() if self.code else None}
        for record in self.stages:
            out[record.name] = record.code.model_dump() if record.code else None
        return out

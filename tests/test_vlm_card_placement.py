"""Where a VLM job's weights go, and what the second A40 is actually for (#12, #137).

#137 asks that a multi-card job be either supported or refused, and that a test
say which. It is **supported**: the record has carried a list of cards since #136
(`TrainJob.gpus`), the scheduler excludes every card a running job holds, and the
child gets them all through `CUDA_VISIBLE_DEVICES`. What was missing was the last
step — the model was pinned to the first card whatever it had been given.

Measured on asteraix, 03.10.2026, Qwen3-VL-8B with the default adapters and
gradient checkpointing, before any training step:

    4-bit, one card      9.25 GiB of 44.42
    bf16,  one card     17.31 GiB of 44.42      <- fits, with 27.1 GiB to spare
    bf16,  both cards    7.84 + 9.79 GiB

So the epic's premise — 92 GB makes an 8B trainable without 4-bit — is answered
one level down: **44 does.** Two cards buy activation headroom, not feasibility,
and for this model the better use of the second card is a second job.
"""

from __future__ import annotations

import pytest

from atr_training.contracts import DatasetSpec, TrainJob, TrainRequest, VlmTrainParams
from atr_training.settings import TrainerSettings
from vlm_train_svc.train_qlora import device_map_for

#: What the probe found on the box. Not a tolerance: these are the numbers the
#: comments and the issue quote, and a test that restates them is what makes a
#: later "it does not fit" check them rather than repeat them.
MEASURED_GIB = {"4bit_one_card": 9.25, "bf16_one_card": 17.31,
                "bf16_two_cards": (7.84, 9.79)}
A40_GIB = 44.42

#: And during 96 page samples at batch 1 x accumulate 16, one epoch — the
#: expensive half of the same question (`scripts/measure_vlm_arms.py`). Peak is
#: what nvidia-smi showed, sampled every 2 s.
MEASURED_STEPS = {
    "4bit_one_card": {"runtime_s": 301.5, "samples_per_s": 0.318, "peak_mib": (21675,)},
    "bf16_one_card": {"runtime_s": 247.6, "samples_per_s": 0.388, "peak_mib": (27763,)},
    "bf16_two_cards": {"runtime_s": 249.7, "samples_per_s": 0.384, "peak_mib": (13761, 18625)},
}
A40_MIB = 45486
#: The draw the step numbers come from, and the corpus it was drawn from.
DRAW_SAMPLES = 96
CORPUS_SAMPLES = 9441


def _request() -> TrainRequest:
    return TrainRequest(engine="vllm", model_id="qwen3vl-placement",
                        datasets=[DatasetSpec(hf_repo="dh-unibe/image-text_x")],
                        params=VlmTrainParams())


# ── the placement rule ───────────────────────────────────────────────────────
@pytest.mark.parametrize("visible", [0, 1])
def test_one_card_is_pinned_explicitly(visible):
    """`{"": 0}` rather than `auto`, which would reach the same placement: the
    explicit form names the card in the log, and a job handed one card behaves
    exactly as it did before #137."""
    assert device_map_for(visible) == {"": 0}


@pytest.mark.parametrize("visible", [2, 4])
def test_more_than_one_card_shards_the_model(visible):
    assert device_map_for(visible) == "auto"


def test_the_placement_follows_the_job_and_not_the_box():
    """The cards reach the child as CUDA_VISIBLE_DEVICES, which renumbers them —
    so a job on card 1 sees one card and places at `cuda:0`, and a job on both
    sees two. Two jobs on one box must not read the same box-wide constant."""
    both = TrainerSettings(gpus=[0, 1])

    one = both.env_for_child([1])["CUDA_VISIBLE_DEVICES"]
    two = both.env_for_child([0, 1])["CUDA_VISIBLE_DEVICES"]

    assert one == "1" and two == "0,1"
    assert device_map_for(len(one.split(","))) == {"": 0}
    assert device_map_for(len(two.split(","))) == "auto"


# ── the record can hold what the placement needs ─────────────────────────────
def test_a_job_record_carries_every_card_it_holds():
    """The reason the field is a list even though nothing requests two: a job
    over two cards that recorded one would hold one card by the record and two in
    fact, and the next job would be put on a card it is already using."""
    job = TrainJob(id="j", request=_request(), status="training", gpus=[0, 1])

    assert job.gpus == [0, 1]
    assert {c for c in job.gpus} == {0, 1}, "both cards are excluded from the next claim"


def test_a_job_that_holds_no_card_is_not_read_as_holding_the_default_one():
    job = TrainJob(id="j", request=_request(), status="queued")
    assert job.gpus == []


# ── the measurement, so a later claim has to check rather than repeat ────────
def test_an_eight_b_in_bf16_fits_on_one_card():
    assert MEASURED_GIB["bf16_one_card"] < A40_GIB
    headroom = A40_GIB - MEASURED_GIB["bf16_one_card"]
    assert headroom > 25, f"only {headroom:.1f} GiB left for activations"


def test_four_bit_buys_headroom_not_feasibility():
    """The question #137 asked was whether bf16 fits. It does — what 4-bit buys is
    8 GiB of activation budget, which is a batch-size question (#138), not a
    quantisation one."""
    saved = MEASURED_GIB["bf16_one_card"] - MEASURED_GIB["4bit_one_card"]
    assert 7.5 < saved < 8.5, saved
    assert MEASURED_GIB["4bit_one_card"] < A40_GIB


def test_two_cards_hold_about_what_one_does():
    """Naive model parallelism splits the weights and nothing else, so the sum is
    the single-card figure plus one more CUDA context."""
    first, second = MEASURED_GIB["bf16_two_cards"]
    assert first + second == pytest.approx(MEASURED_GIB["bf16_one_card"], abs=0.5)
    assert max(first, second) < MEASURED_GIB["bf16_one_card"], "neither card holds it all"


# ── and what a step costs ────────────────────────────────────────────────────
def test_bf16_is_faster_than_four_bit_not_slower():
    """The assumption worth refuting: 4-bit is cheaper. It is cheaper in memory
    and dearer in time — nf4 dequantises every weight on every pass, and an A40
    runs bf16 matmuls natively."""
    four = MEASURED_STEPS["4bit_one_card"]["runtime_s"]
    bf16 = MEASURED_STEPS["bf16_one_card"]["runtime_s"]

    assert bf16 < four
    assert (four - bf16) / four > 0.15, "a smaller margin would be within noise"


def test_the_bf16_peak_still_leaves_room_on_one_card():
    peak, = MEASURED_STEPS["bf16_one_card"]["peak_mib"]
    assert peak < A40_MIB
    assert (A40_MIB - peak) / 1024 > 15, "under 15 GiB spare is not headroom here"


def test_the_second_card_buys_headroom_and_not_speed():
    """Naive model parallelism serialises: one card waits while the other
    computes. Whoever reads "two cards" as "twice as fast" should fail here."""
    one = MEASURED_STEPS["bf16_one_card"]["runtime_s"]
    two = MEASURED_STEPS["bf16_two_cards"]["runtime_s"]

    assert abs(two - one) / one < 0.05, "within noise — no speed-up"
    assert max(MEASURED_STEPS["bf16_two_cards"]["peak_mib"]) < \
        MEASURED_STEPS["bf16_one_card"]["peak_mib"][0]


def test_the_peak_was_drawn_from_a_fraction_of_the_corpus():
    """The honest limit of the number above: 96 of the 9,441 samples, taken from
    the head of the file. On this project an OOM has twice been the
    distribution's tail rather than the batch size, so the 17 GiB margin is
    headroom and not proof. Pinned so a later "it fits" has to name the draw."""
    assert DRAW_SAMPLES == 96
    assert CORPUS_SAMPLES == 9441
    assert DRAW_SAMPLES / CORPUS_SAMPLES < 0.02, "a 1 % draw cannot bound the tail"


def test_the_default_is_still_four_bit_until_a_run_says_otherwise():
    """Fitting is not the same as training better or faster. The default moves on
    a trained model's CER, not on a footprint."""
    assert VlmTrainParams().load_in_4bit is True

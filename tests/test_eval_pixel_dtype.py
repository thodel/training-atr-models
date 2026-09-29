"""The pixels handed to generate() must be in the dtype the model computes in.

Transformers casts them itself — but only where it can see a floating point
weight to cast to. Gemma 4's encoder-free image path reads
``patch_dense.weight.dtype``, which under 4-bit is ``uint8``, so the cast is
skipped and fp32 pixels reach a bf16 LayerNorm. That is a RuntimeError in the
first forward of the test stage, after the training has already finished
(20260926T080903Z-ladder-med-gemma4-12b, 25 h, adapter intact).

No torch here: the helper only asks the model for its dtype and the tensor for
``.to``, so a stand-in for each is enough to pin the contract.

    pytest tests/test_eval_pixel_dtype.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engines"))

from vlm_train_svc.evaluate_qlora import _pixels_in_model_dtype  # noqa: E402


class FakeDtype:
    def __init__(self, name: str, floating: bool = True) -> None:
        self.name, self.is_floating_point = name, floating

    def __repr__(self) -> str:  # pragma: no cover — only for failure messages
        return self.name


BF16 = FakeDtype("bfloat16")
UINT8 = FakeDtype("uint8", floating=False)


class FakeTensor:
    """Records what it was cast to, and to nothing else."""

    def __init__(self, dtype: FakeDtype) -> None:
        self.dtype, self.casts = dtype, []

    def to(self, dtype):
        self.casts.append(dtype)
        cast = FakeTensor(dtype)
        cast.casts = self.casts
        return cast


class FakeModel:
    def __init__(self, dtype) -> None:
        self.dtype = dtype


def test_pixels_are_cast_to_the_models_dtype():
    pixels = FakeTensor(FakeDtype("float32"))
    out = _pixels_in_model_dtype({"pixel_values": pixels}, FakeModel(BF16))
    assert out["pixel_values"].dtype is BF16
    assert pixels.casts == [BF16]


def test_nothing_else_is_touched():
    """``input_ids`` and the masks are integer tensors; casting them to a float
    dtype would be a different bug, one that reads as a tokenizer failure."""
    ids, mask = FakeTensor(FakeDtype("int64", floating=False)), FakeTensor(UINT8)
    out = _pixels_in_model_dtype(
        {"input_ids": ids, "attention_mask": mask, "pixel_values": FakeTensor(BF16)},
        FakeModel(BF16))
    assert out["input_ids"] is ids and out["attention_mask"] is mask
    assert ids.casts == [] and mask.casts == []


def test_a_text_only_batch_passes_through():
    inputs = {"input_ids": FakeTensor(UINT8)}
    assert _pixels_in_model_dtype(inputs, FakeModel(BF16)) is inputs


@pytest.mark.parametrize("dtype", [None, UINT8], ids=["no dtype", "not floating point"])
def test_a_model_without_a_float_dtype_is_left_alone(dtype):
    """Nothing sensible to cast to, so nothing is cast — the model's own path
    decides, exactly as before this helper existed."""
    pixels = FakeTensor(FakeDtype("float32"))
    out = _pixels_in_model_dtype({"pixel_values": pixels}, FakeModel(dtype))
    assert out["pixel_values"] is pixels
    assert pixels.casts == []


def test_the_original_mapping_is_not_mutated():
    """``transcribe`` rebinds the result; a helper that edited in place would
    still work there and surprise the next caller."""
    pixels = FakeTensor(FakeDtype("float32"))
    inputs = {"pixel_values": pixels}
    _pixels_in_model_dtype(inputs, FakeModel(BF16))
    assert inputs["pixel_values"] is pixels

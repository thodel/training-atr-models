"""TrOCR trains on aligned labels, not on the token after next (#37).

transformers 4.57 resolves an encoder-decoder's loss through ``loss_function``
and falls back to ``ForCausalLMLoss`` without a ``loss_type``, which shifts
labels the decoder input has already shifted. The fakes below resolve the loss
the way ``PreTrainedModel.loss_function`` does, so the test pins the mechanism
and not just the attribute. The real model was checked on asteraix: loss 27.6
before, 2.3 after, on the same 16 lines.
"""
from __future__ import annotations

import pytest

from trocr_train_svc.train_trocr import ALIGNED_LOSS, use_aligned_loss


def ForCausalLMLoss(**kwargs):  # noqa: N802 — named like transformers' functions
    return "shifted"


def ForMaskedLMLoss(**kwargs):  # noqa: N802
    return "aligned"


MAPPING = {"ForCausalLM": ForCausalLMLoss, "ForMaskedLM": ForMaskedLMLoss}


class _Config:
    loss_type = None


class _Model:
    """Resolves ``loss_function`` from ``loss_type`` with the causal fallback."""

    def __init__(self) -> None:
        self.config = _Config()
        self.loss_type = None

    @property
    def loss_function(self):
        return MAPPING.get(self.loss_type or "", ForCausalLMLoss)


class _StubbornModel(_Model):
    """A version that ignores ``loss_type`` and always shifts."""

    @property
    def loss_function(self):
        return ForCausalLMLoss


def test_the_default_resolution_is_the_shifted_loss():
    """The premise: untouched, the fallback is the causal loss."""
    assert _Model().loss_function is ForCausalLMLoss


def test_the_encoder_decoder_trains_with_the_aligned_loss():
    model = _Model()

    use_aligned_loss(model)

    assert model.loss_function is ForMaskedLMLoss
    assert model.loss_type == model.config.loss_type == ALIGNED_LOSS


def test_a_version_that_still_shifts_stops_the_run():
    with pytest.raises(SystemExit, match="shift the labels twice"):
        use_aligned_loss(_StubbornModel())

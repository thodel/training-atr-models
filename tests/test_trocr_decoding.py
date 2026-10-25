"""TrOCR decodes with beam search by default, and the settings reach generate (#37).

On the first real run greedy decoding ran on after short lines and beam search
with early stopping halved the CER of the same checkpoint. Two things are pinned
here: the default, and that ``--length-penalty`` arrives at ``generate`` — it
was accepted and dropped before.
"""
from __future__ import annotations

import contextlib
import sys
import types

import pytest
from PIL import Image

from atr_training.contracts import TrOCRTrainParams
from atr_training.trocr_cmd import evaluate_cmd


@pytest.fixture
def fake_torch(monkeypatch):
    """`transcribe` imports torch for `no_grad` and nothing else."""
    monkeypatch.setitem(sys.modules, "torch",
                        types.SimpleNamespace(no_grad=contextlib.nullcontext))


class _Model:
    device = "cpu"

    def __init__(self) -> None:
        self.kwargs: dict = {}

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return [[0]]


class _Processor:
    def __call__(self, **kwargs):
        return {"pixel_values": object()}

    def batch_decode(self, ids, skip_special_tokens=True):
        return [" Von Gott. "]


def _transcribe(tmp_path, **kwargs):
    from trocr_train_svc.evaluate_trocr import transcribe

    image = tmp_path / "line.png"
    Image.new("RGB", (120, 20)).save(image)
    model = _Model()
    text = transcribe(model, _Processor(), image, 256, **kwargs)
    return text, model.kwargs


def test_beam_search_is_the_default():
    assert TrOCRTrainParams().beam_size == 4
    assert TrOCRTrainParams().length_penalty == 1.0
    assert TrOCRTrainParams().no_repeat_ngram_size == 3


def test_beam_settings_reach_generate(tmp_path, fake_torch):
    text, kwargs = _transcribe(tmp_path, num_beams=4, length_penalty=0.8)

    assert text == "Von Gott."
    assert kwargs["num_beams"] == 4
    assert kwargs["length_penalty"] == 0.8
    assert kwargs["early_stopping"] is True


def test_the_ngram_block_reaches_generate_and_zero_turns_it_off(tmp_path, fake_torch):
    _, on = _transcribe(tmp_path, num_beams=4, no_repeat_ngram_size=3)
    _, off = _transcribe(tmp_path, num_beams=4, no_repeat_ngram_size=0)

    assert on["no_repeat_ngram_size"] == 3
    assert "no_repeat_ngram_size" not in off


def test_greedy_gets_no_beam_only_settings(tmp_path, fake_torch):
    """transformers rejects length_penalty with num_beams=1."""
    _, kwargs = _transcribe(tmp_path, num_beams=1, length_penalty=0.8)

    assert kwargs["num_beams"] == 1
    assert "length_penalty" not in kwargs and "early_stopping" not in kwargs


def test_the_runner_passes_the_defaults_to_the_evaluator(tmp_path):
    cmd = evaluate_cmd("python", params=TrOCRTrainParams(),
                       base_model="microsoft/trocr-base-handwritten",
                       checkpoint=tmp_path / "ck", val_manifest=tmp_path / "val.jsonl",
                       data_root=tmp_path, report=tmp_path / "report.json")

    assert cmd[cmd.index("--beam-size") + 1] == "4"
    assert cmd[cmd.index("--length-penalty") + 1] == "1.0"
    assert cmd[cmd.index("--no-repeat-ngram-size") + 1] == "3"

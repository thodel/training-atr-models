"""Keeping the adapters out of the vision tower (#95).

``target_modules`` matches by suffix, so ``q_proj`` means every ``q_proj`` in the
model — decoder, vision encoder, audio encoder alike. For Qwen that happens to be
only the decoder: measured on 2026-09-25, all 252 matches in Qwen3-VL-4B and all
128 in Qwen3.5-4B are inside ``model.language_model``, because its towers do not
reuse those names. Gemma 4's do — E4B has 112 in ``model.vision_tower`` and 36 in
``model.audio_tower`` — and they are ``Gemma4ClippableLinear``, which peft refuses
outright, so the run dies at adapter injection.

The exclusion is a **regex, not a list**, and that is the whole point of these
tests. peft matches an exclusion list by suffix: ``["vision_tower"]`` reads like
"exclude the tower" and excludes nothing, silently. Measured against peft 0.20.0
with a toy model:

    exclude_modules=["vision_tower", "audio_tower"]  -> 3 modules adapted
    exclude_modules=".*(vision_tower|audio_tower).*" -> 1 module adapted
"""

from __future__ import annotations

import pathlib
import re
import tempfile

import pytest

from atr_training.contracts import DEFAULT_EXCLUDE_MODULES, VlmTrainParams
from atr_training.vlm_cmd import train_cmd
from vlm_train_svc.train_qlora import modules_matching

_TMP = pathlib.Path(tempfile.mkdtemp(prefix="atr-collator-"))

#: Module paths as the three families actually spell them (measured).
QWEN3VL = ["model.language_model.layers.0.self_attn.q_proj",
           "model.language_model.layers.0.mlp.gate_proj",
           "model.visual.blocks.0.attn.qkv",
           "model.visual.merger.linear_fc1"]
GEMMA4 = ["model.language_model.layers.0.self_attn.q_proj",
          "model.language_model.layers.0.mlp.gate_proj",
          "model.vision_tower.timm_model.blocks.0.attn.q_proj",
          "model.audio_tower.encoder.layers.0.self_attn.v_proj"]

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class FakeModel:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def named_modules(self):
        return [(name, object()) for name in self._names]


def adapted(names: list[str], pattern: str) -> list[str]:
    """What would still be adapted after the exclusion."""
    excluded = set(modules_matching(FakeModel(names), TARGETS, pattern))
    return [n for n in names if n.rsplit(".", 1)[-1] in set(TARGETS) and n not in excluded]


def test_the_default_spares_gemmas_towers():
    left = adapted(GEMMA4, DEFAULT_EXCLUDE_MODULES)
    assert left == ["model.language_model.layers.0.self_attn.q_proj",
                    "model.language_model.layers.0.mlp.gate_proj"]


def test_the_default_changes_nothing_for_qwen():
    """The measured runs must stay comparable: their towers never matched anyway.

    If this ever fails, the default has started excluding something a published
    CER was measured with, and the number and the model no longer belong to each
    other."""
    assert adapted(QWEN3VL, DEFAULT_EXCLUDE_MODULES) == adapted(QWEN3VL, "")


def test_a_suffix_list_would_have_excluded_nothing():
    """Why the field is a regex. Spelled out because the list form reads correct.

    peft matches an exclusion list by suffix, so the tower's *children* — which
    are what carries q_proj — never match the tower's own name."""
    as_a_list_would = "|".join(["vision_tower", "audio_tower"])
    assert not [n for n in GEMMA4 if re.match(as_a_list_would, n)]
    assert modules_matching(FakeModel(GEMMA4), TARGETS, as_a_list_would) == []


def test_it_excludes_nothing_it_was_not_asked_to():
    """A tower is not excluded by having the word in a longer name elsewhere."""
    names = ["model.language_model.layers.0.self_attn.q_proj",
             "model.language_model.vision_tower_adapter_notreally.q_proj"]
    assert modules_matching(FakeModel(names), TARGETS, DEFAULT_EXCLUDE_MODULES) == []


def test_only_targeted_modules_are_counted():
    """The count in the log is "modules this spared", not "modules that match"."""
    names = ["model.vision_tower.blocks.0.attn.q_proj",
             "model.vision_tower.blocks.0.layernorm"]
    assert modules_matching(FakeModel(names), TARGETS, DEFAULT_EXCLUDE_MODULES) == [
        "model.vision_tower.blocks.0.attn.q_proj"]


def test_an_empty_pattern_excludes_nothing():
    assert modules_matching(FakeModel(GEMMA4), TARGETS, "") == []


def test_a_broken_regex_is_refused_before_training():
    with pytest.raises(SystemExit, match="not a valid regex"):
        modules_matching(FakeModel(GEMMA4), TARGETS, "(unclosed")


class TestItReachesTheTrainer:
    def _cmd(self, **kwargs) -> list[str]:
        return train_cmd("python", params=VlmTrainParams(**kwargs),
                         base_model="google/gemma-4-E4B-it",
                         train_jsonl="t.jsonl", val_jsonl="v.jsonl",
                         output_dir="out", data_root=".")

    def test_the_default_is_passed(self):
        cmd = self._cmd()
        assert "--exclude-modules" in cmd
        assert cmd[cmd.index("--exclude-modules") + 1] == DEFAULT_EXCLUDE_MODULES

    def test_it_can_be_turned_off(self):
        assert "--exclude-modules" not in self._cmd(exclude_modules="")


# ── one list of images per text (#95, second blocker) ───────────────────────
class RecordingProcessor:
    """Enough of a processor to see what the collator hands it."""

    class _Tok:
        pad_token_id = 0

        def __call__(self, text, add_special_tokens=True):
            # One id per character, so the header derivation has a real diff.
            return type("E", (), {"input_ids": [ord(c) for c in text]})()

    def __init__(self) -> None:
        self.tokenizer = self._Tok()
        self.seen = None

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, **kwargs):
        answer = next((m["content"][0]["text"] for m in messages
                       if m["role"] == "assistant"), None)
        # The assistant marker appears only when there is an assistant turn —
        # as a real template does, and as the header derivation relies on.
        return "<u>T." + (f"<a>{answer}" if answer is not None else "")

    def __call__(self, text, images, return_tensors=None, padding=None):
        self.seen = images
        raise _Stop


class _Stop(Exception):
    pass


def test_the_collator_passes_one_image_list_per_text():
    """Gemma 4 reads a flat list as one sample's images and refuses the batch:

        ValueError: Received inconsistently sized batches of images (1) and text (2)

    Qwen accepts either form and produces byte-identical output — measured, 2x81
    ids for Qwen3-VL-4B and 2x85 for Qwen3.5-4B — so nested is right everywhere.
    """
    from PIL import Image

    from vlm_train_svc.train_qlora import HTRCollator

    processor = RecordingProcessor()
    collator = HTRCollator(processor, "T.", max_seq_len=1024)
    batch = []
    for name in ("a.jpg", "b.jpg"):
        path = _TMP / name
        Image.new("RGB", (40, 12)).save(path)
        batch.append({"image": str(path), "text": "x", "source_type": "line"})

    with pytest.raises(_Stop):
        collator(batch)
    assert processor.seen == [[processor.seen[0][0]], [processor.seen[1][0]]]
    assert len(processor.seen) == 2 and all(len(g) == 1 for g in processor.seen)

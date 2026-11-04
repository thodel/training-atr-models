"""#361 — "Using it" on a VLM card says what the adapter's own config says.

`dh-unibe/olmocr2-7b-german-kurrent-xvi-xix-blb-page-px4m-native-v1` (03.11.2026)
carried two sentences its adapter did not bear out: no peft version, though peft
0.19.1 had failed to read the adapter 0.20.0 wrote; and "it refuses LoRA on the
vision tower", for an adapter whose ``exclude_modules`` leaves the vision tower out.
"""

from __future__ import annotations

import json
from pathlib import Path

from atr_training.publish import ADAPTER_CONFIG_FILENAME, model_card, scan_trained
from test_training_publish import VLM_META, trained_dir

# The two fields that matter, as the olmOCR BLB adapter has them.
OLMOCR_CONFIG = {
    "peft_version": "0.20.0",
    "base_model_name_or_path": "allenai/olmOCR-2-7B-1025",
    "exclude_modules": "(?:^|.*\\.)(vision_tower|audio_tower|visual)\\..*",
    "target_modules": ["up_proj", "o_proj", "gate_proj", "down_proj", "v_proj", "q_proj", "k_proj"],
}


def _card(tmp_path: Path, config: dict | None, *, raw: str | None = None) -> str:
    directory = trained_dir(tmp_path, VLM_META, weights="adapter_model.safetensors")
    for name in ("tokenizer.json", "training_summary.json"):
        (directory / name).write_text("{}", encoding="utf-8")
    if raw is not None:
        (directory / ADAPTER_CONFIG_FILENAME).write_text(raw, encoding="utf-8")
    elif config is not None:
        (directory / ADAPTER_CONFIG_FILENAME).write_text(json.dumps(config), encoding="utf-8")
    return model_card(scan_trained(tmp_path).models[0], "dh-unibe/x")


def _using_it(card: str) -> str:
    return card.split("## Using it", 1)[1]


# ── the peft version ────────────────────────────────────────────────────────

def test_the_card_names_the_peft_that_wrote_the_adapter(tmp_path):
    usage = _using_it(_card(tmp_path, OLMOCR_CONFIG))
    assert "pip install 'peft>=0.20.0'" in usage
    assert "an older peft does not read its configuration" in usage
    assert usage.index("pip install") < usage.index("PeftModel.from_pretrained")


def test_without_a_recorded_version_no_version_is_invented(tmp_path):
    config = {k: v for k, v in OLMOCR_CONFIG.items() if k != "peft_version"}
    usage = _using_it(_card(tmp_path, config))
    assert "pip install" not in usage and "PeftModel.from_pretrained" in usage


def test_an_unreadable_config_says_less_and_guesses_nothing(tmp_path):
    usage = _using_it(_card(tmp_path, None, raw="{not json"))
    assert "pip install" not in usage
    assert "refuses LoRA on the vision tower" in usage, "the sentence for the general case"


# ── what the adapter adapts ─────────────────────────────────────────────────

def test_an_adapter_that_leaves_the_vision_tower_out_says_so(tmp_path):
    usage = _using_it(_card(tmp_path, OLMOCR_CONFIG))
    assert "language model only" in usage
    assert "refuses LoRA on the vision tower" not in usage
    assert "merge_loras.py" in usage, "the gateway still serves it merged"


def test_an_adapter_that_trains_the_vision_tower_keeps_the_old_sentence(tmp_path):
    config = {**OLMOCR_CONFIG, "exclude_modules": None}
    usage = _using_it(_card(tmp_path, config))
    assert "refuses LoRA on the vision tower" in usage
    assert "language model only" not in usage


def test_a_list_of_excluded_names_is_not_taken_as_evidence(tmp_path):
    config = {**OLMOCR_CONFIG, "exclude_modules": ["visual"]}
    assert "refuses LoRA on the vision tower" in _using_it(_card(tmp_path, config))


def test_a_qwen3vl_style_vision_tower_is_recognised_too(tmp_path):
    config = {**OLMOCR_CONFIG, "exclude_modules": ".*vision_tower.*"}
    assert "language model only" in _using_it(_card(tmp_path, config))


# ── the weights row ─────────────────────────────────────────────────────────

def test_the_weights_row_lists_weights_and_the_rest_beside_them(tmp_path):
    card = _card(tmp_path, OLMOCR_CONFIG)
    weights_row = next(line for line in card.splitlines() if line.startswith("| weights |"))
    assert weights_row == "| weights | `adapter_model.safetensors` |"
    with_them = next(line for line in card.splitlines() if line.startswith("| with them |"))
    assert "`training_summary.json`" in with_them and "`adapter_config.json`" in with_them

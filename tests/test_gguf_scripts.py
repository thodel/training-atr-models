"""The two GGUF scripts of #176: merge with a baked budget, score against llama-server.

Neither test touches torch, a model or a server. What they pin is the part
that silently changes a number: where the visual budget is written, and when
the server's prompt counts as the processor's.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


merge = _load("merge_adapter")
score = _load("score_gguf")

QWEN_IMAGE_PROCESSOR = {"patch_size": 16, "merge_size": 2,
                        "size": {"longest_edge": 262144, "shortest_edge": 65536}}


# ── the budget, in tokens ───────────────────────────────────────────────────
def test_the_benchmark_budget_is_256_tokens_with_the_processors_floor():
    assert merge.token_budget(QWEN_IMAGE_PROCESSOR, 262144) == {
        "max_image_tokens": 256, "min_image_tokens": 64}


def test_transformers_5_layout_gets_the_keys_at_the_top(tmp_path):
    """llama.cpp lifts ``image_processor`` to the top; inside it the keys would
    reach the image processor's constructor as unknown arguments."""
    path = tmp_path / "processor_config.json"
    path.write_text(json.dumps({"image_processor": QWEN_IMAGE_PROCESSOR}))

    merge.bake_budget(tmp_path, 262144)

    written = json.loads(path.read_text())
    assert written["max_image_tokens"] == 256 and written["min_image_tokens"] == 64
    assert "max_image_tokens" not in written["image_processor"]


def test_transformers_4_layout_is_written_where_it_lies(tmp_path):
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(QWEN_IMAGE_PROCESSOR))
    (tmp_path / "processor_config.json").write_text(json.dumps({"processor_class": "x"}))

    merge.bake_budget(tmp_path, 262144)

    assert json.loads((tmp_path / "preprocessor_config.json").read_text())[
        "max_image_tokens"] == 256
    assert "max_image_tokens" not in json.loads(
        (tmp_path / "processor_config.json").read_text())


# ── the prompt check ────────────────────────────────────────────────────────
PROCESSOR_RENDER = ("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                    "Transcribe it.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


@pytest.mark.parametrize("marker", ["<__media__>", "<__media_0dtRVSUySrowh77Fdz0XA0irbn0f5obz__>"])
def test_the_servers_media_marker_stands_for_the_vision_block(marker):
    """Measured against b11868: the marker carries a random suffix, and mtmd
    wraps the embeddings in <|vision_start|> … <|vision_end|> itself."""
    server = PROCESSOR_RENDER.replace(
        "<|vision_start|><|image_pad|><|vision_end|>", marker)

    assert score.MEDIA_MARKER.sub(score.QWEN_IMAGE, server) == PROCESSOR_RENDER


def test_a_prompt_that_differs_elsewhere_is_not_equal():
    """A thinking block the processor did not render is the 22.94 % failure."""
    server = PROCESSOR_RENDER.replace(
        "<|vision_start|><|image_pad|><|vision_end|>", "<__media__>").replace(
        "<think>\n\n</think>\n\n", "<think>\n")

    assert score.MEDIA_MARKER.sub(score.QWEN_IMAGE, server) != PROCESSOR_RENDER


def test_messages_put_the_image_before_the_prompt_like_training():
    turns = score.messages_for(b"\x89PNG", "Transcribe it.")

    assert [part["type"] for part in turns[0]["content"]] == ["image_url", "text"]
    assert turns[0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")

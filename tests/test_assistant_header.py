"""Where the transcription starts, and why the obvious way to find it is wrong.

The collator masks everything before the assistant header out of the loss, so the
header has to be *the tokens that actually precede the answer in a training
sample*. The obvious derivation — diff the conversation with and without
``add_generation_prompt=True`` — gives that for Qwen and does not for Gemma 4:
asked for a generation prompt, Gemma emits an empty thinking channel that
disappears the moment the assistant turn has content. The header taken from there
occurs in no training sample at all, and ``HTRCollator`` refuses the first batch.

Correctly refuses — but after a job has been queued, scheduled and has loaded a
12B base. These tests pin the derivation to the render the loss is computed over.

The two templates below are the shapes measured on 2026-09-25 against
transformers 5.17.0, reduced to what matters here.
"""

from __future__ import annotations

import pytest

from vlm_train_svc.train_qlora import assistant_header_ids


class FakeTokenizer:
    """Whitespace-ish tokenizer: one id per token, stable across calls."""

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def _pieces(self, text: str) -> list[str]:
        return [p for p in text.replace("\n", " \n ").split(" ") if p]

    def __call__(self, text: str, add_special_tokens: bool = True):
        ids = [self._ids.setdefault(p, len(self._ids) + 1) for p in self._pieces(text)]
        return type("Encoded", (), {"input_ids": ids})()


class FakeProcessor:
    """Renders a conversation the way one family's chat template does."""

    def __init__(self, render) -> None:
        self.tokenizer = FakeTokenizer()
        self._render = render

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, **kwargs) -> str:
        return self._render(messages, add_generation_prompt)


def _answer_of(messages) -> str | None:
    for message in messages:
        if message["role"] == "assistant":
            return message["content"][0]["text"]
    return None


def qwen_like(messages, add_generation_prompt: bool) -> str:
    """The generation prompt and the assistant turn agree. The easy case."""
    out = "<|im_start|>user\n<image>Transcribe.<|im_end|>\n"
    answer = _answer_of(messages)
    if answer is not None:
        out += f"<|im_start|>assistant\n{answer}<|im_end|>\n"
    elif add_generation_prompt:
        out += "<|im_start|>assistant\n"
    return out


def gemma_like(messages, add_generation_prompt: bool) -> str:
    """They disagree: the generation prompt opens an empty thinking channel that
    the assistant turn never contains. Measured on google/gemma-4-12B-it."""
    out = "<bos><|turn>user\n<|image|>Transcribe.<turn|>\n"
    answer = _answer_of(messages)
    if answer is not None:
        out += f"<|turn>model\n{answer}<turn|>\n"
    elif add_generation_prompt:
        out += "<|turn>model\n<|channel>thought\n<channel|>"
    return out


def render_and_find(processor, render) -> bool:
    """Does the derived header occur in a real training sample's tokens?"""
    header = assistant_header_ids(processor, "Transcribe.")
    full = render([{"role": "user", "content": [{"type": "text", "text": "Transcribe."}]},
                   {"role": "assistant",
                    "content": [{"type": "text", "text": "Hochzeitlich"}]}], False)
    ids = processor.tokenizer(full, add_special_tokens=False).input_ids
    n = len(header)
    return any(ids[i:i + n] == header for i in range(len(ids) - n + 1))


def test_the_header_is_present_in_a_real_training_sample_for_qwen():
    processor = FakeProcessor(qwen_like)
    assert render_and_find(processor, qwen_like)


def test_the_header_is_present_in_a_real_training_sample_for_gemma():
    """The regression this exists for: taken from the generation prompt, the
    header would carry `<|channel>thought` and match nothing."""
    processor = FakeProcessor(gemma_like)
    assert render_and_find(processor, gemma_like)


def test_the_header_is_the_assistant_turn_opener_and_nothing_before_it():
    processor = FakeProcessor(gemma_like)
    header = assistant_header_ids(processor, "Transcribe.")
    expected = processor.tokenizer("<|turn>model\n", add_special_tokens=False).input_ids
    assert header == expected, "the instruction's tokens leaked into the header"


def test_the_generation_prompt_is_not_consulted():
    """A template that refuses to produce a generation prompt at all still trains.

    Nothing about the loss mask depends on that path, and a family that answers
    differently there — or raises — is not thereby untrainable."""
    def no_generation_prompt(messages, add_generation_prompt):
        if add_generation_prompt:
            raise AssertionError("the derivation must not ask for a generation prompt")
        return qwen_like(messages, False)

    processor = FakeProcessor(no_generation_prompt)
    assert assistant_header_ids(processor, "Transcribe.")


def test_a_template_that_drops_the_answer_is_refused():
    """Without the answer in the render there is no boundary to find, and a
    silent fallback would train the model on its own instruction."""
    def swallows_the_answer(messages, add_generation_prompt):
        return "<|im_start|>user\nTranscribe.<|im_end|>\n"

    with pytest.raises(SystemExit, match="did not render the assistant"):
        assistant_header_ids(FakeProcessor(swallows_the_answer), "Transcribe.")


def test_a_template_with_no_header_at_all_is_refused():
    def no_header(messages, add_generation_prompt):
        answer = _answer_of(messages)
        return "Transcribe." + (answer or "")

    with pytest.raises(SystemExit, match="could not derive"):
        assistant_header_ids(FakeProcessor(no_header), "Transcribe.")

"""What ``generate`` stops on, and the newline that must not be in it.

The hardcoded pair ``("<|im_end|>", "<|endoftext|>")`` is Qwen's. Gemma 4 has
neither and ends a turn with ``<turn|>`` (id 106), so the lookup refused it
outright — and a run that had trained to completion, 19,162 of 19,162 steps with
a final loss of 1.117, died in its test stage with no number to show for a day of
GPU time.

The terminator is now derived from the chat template, the way the assistant header
is: the token a training render puts after the answer is by construction the one
the model was taught to emit. Measured against the real processors, that fix
reproduces Qwen's ids exactly — ``[151645, 151643]`` for Qwen3-VL-4B, ``[248046,
248044]`` for Qwen3.5-4B — and gives Gemma ``[106, 1]``.

**The trap these tests exist for.** The rendered tail is ``"<|im_end|>\\n"`` for
Qwen and ``"<turn|>\\n"`` for Gemma. Taking the whole tail makes a bare newline a
stop token, and a page or block transcription *is* newlines: every multi-line
prediction would have been cut at its first line break and scored as a model that
cannot read past one line. Only the first token counts, and only when it is a
special or added one.
"""

from __future__ import annotations

import pytest

from vlm_train_svc.evaluate_qlora import STOP_TOKENS, stop_token_ids, turn_end_ids

#: A newline is an ordinary token in every tokenizer here, and must never stop.
NEWLINE_ID = 10


class FakeTokenizer:
    """Ids are code points; added tokens get ids above 1000."""

    def __init__(self, added: dict[str, int], eos_token_id=None) -> None:
        self._added = added
        self.eos_token_id = eos_token_id
        self.unk_token_id = 999
        self.all_special_ids = [eos_token_id] if eos_token_id is not None else []

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._added)

    def convert_tokens_to_ids(self, name: str) -> int:
        return self._added.get(name, -1)

    def __call__(self, text: str, add_special_tokens: bool = True):
        ids = []
        rest = text
        while rest:
            for name, tid in self._added.items():
                if rest.startswith(name):
                    ids.append(tid)
                    rest = rest[len(name):]
                    break
            else:
                ids.append(ord(rest[0]) if rest[0] != "\n" else NEWLINE_ID)
                rest = rest[1:]
        return type("E", (), {"input_ids": ids})()

    def decode(self, ids):
        back = {v: k for k, v in self._added.items()}
        return "".join(back.get(i, chr(i)) for i in ids)


class FakeProcessor:
    def __init__(self, tokenizer, tail: str) -> None:
        self.tokenizer = tokenizer
        self._tail = tail

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, **kwargs) -> str:
        answer = next((m["content"][0]["text"] for m in messages
                       if m["role"] == "assistant"), None)
        if answer is None:
            return "<u>T."
        return f"<u>T.<a>{answer}{self._tail}"


def qwen_like():
    tok = FakeTokenizer({"<|im_end|>": 1645, "<|endoftext|>": 1643, "<a>": 1600,
                         "<u>": 1601}, eos_token_id=1645)
    return tok, FakeProcessor(tok, "<|im_end|>\n")


def gemma_like():
    tok = FakeTokenizer({"<turn|>": 1106, "<|turn>": 1105, "<a>": 1600, "<u>": 1601},
                        eos_token_id=1001)
    return tok, FakeProcessor(tok, "<turn|>\n")


def test_a_newline_is_never_a_stop_token():
    """The whole point. A page transcription is newlines."""
    for tok, proc in (qwen_like(), gemma_like()):
        assert NEWLINE_ID not in stop_token_ids(tok, proc)
        assert NEWLINE_ID not in turn_end_ids(proc, tok)


def test_the_terminator_comes_from_the_template():
    tok, proc = gemma_like()
    assert turn_end_ids(proc, tok) == [1106]          # <turn|>, not <|turn>
    assert 1105 not in stop_token_ids(tok, proc), "stopping on the opener would " \
        "end every generation at zero tokens"


def test_qwen_keeps_the_ids_it_always_had():
    tok, proc = qwen_like()
    assert stop_token_ids(tok, proc) == [1645, 1643]


def test_gemma_gets_its_terminator_and_its_eos():
    tok, proc = gemma_like()
    assert stop_token_ids(tok, proc) == [1106, 1001]


def test_a_tokenizer_alone_still_works_for_qwen():
    """The processor is optional; the old call site kept working."""
    tok, _ = qwen_like()
    assert stop_token_ids(tok) == [1645, 1643]


def test_a_family_with_neither_qwen_token_no_longer_refuses():
    """What killed the completed Gemma run: names this file did not know."""
    tok, proc = gemma_like()
    assert not any(tok.convert_tokens_to_ids(n) > 0 for n in STOP_TOKENS[:2])
    assert stop_token_ids(tok, proc)


def test_a_tail_of_plain_text_contributes_nothing():
    """A template whose tail is not a special token proves nothing about stopping."""
    tok = FakeTokenizer({"<a>": 1600, "<u>": 1601}, eos_token_id=1001)
    proc = FakeProcessor(tok, "END")
    assert turn_end_ids(proc, tok) == []
    assert stop_token_ids(tok, proc) == [1001], "falls back to the eos token"


def test_no_stop_condition_at_all_is_refused():
    tok = FakeTokenizer({"<a>": 1600, "<u>": 1601}, eos_token_id=None)
    proc = FakeProcessor(tok, "END")
    with pytest.raises(RuntimeError, match="no stop condition"):
        stop_token_ids(tok, proc)


def test_a_broken_template_is_survived_not_raised():
    """turn_end_ids is the best of three sources, never a requirement."""
    class Broken(FakeProcessor):
        def apply_chat_template(self, *a, **k):
            raise ValueError("no template here")

    tok, _ = qwen_like()
    proc = Broken(tok, "")
    assert turn_end_ids(proc, tok) == []
    assert stop_token_ids(tok, proc) == [1645, 1643]

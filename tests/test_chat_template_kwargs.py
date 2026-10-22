"""#274: a template gets the kwargs it reads, and ``enable_thinking`` is proven.

The olmOCR-2 run of 21.10.2026 logged, at every step, ``[transformers] Kwargs
passed to `processor.__call__` have to be in `processor_kwargs` dict, not in
`**kwargs```. Read in transformers 5.17.0 (``processing_utils.py``,
``apply_chat_template``): the warning fires for every kwarg that is **not** a
variable of the Jinja template (``_get_template_variables``, jinja2's
``find_undeclared_variables``), and the kwarg is then moved into the processor
kwargs. ``CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}`` went to every
template; olmOCR-2's and Qwen3-VL's read no such name, Qwen3.5's and Gemma 4's
do. So the warning marks exactly the templates the flag never reached — and its
absence, the ones it does.

The templates are vendored verbatim (``fixtures/chat_templates``, commits in
its README) and rendered here with jinja2 the way transformers renders them, so
the question "what does ``enable_thinking`` do to the Qwen3.5 render" is
answered by the render, in both transformers lines, without a tokenizer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from atr_training.vlm_dataset import (
    CHAT_TEMPLATE_KWARGS,
    chat_example,
    chat_template_of,
    template_kwargs,
    template_reads,
)

TEMPLATES = Path(__file__).parent / "fixtures" / "chat_templates"
QWEN35 = (TEMPLATES / "qwen3.5-4b.jinja").read_text(encoding="utf-8")
OLMOCR = (TEMPLATES / "olmocr-2-7b-1025.jinja").read_text(encoding="utf-8")
QWEN3VL = (TEMPLATES / "qwen3-vl-4b-instruct.jinja").read_text(encoding="utf-8")
GEMMA4 = (TEMPLATES / "gemma-4-e4b-it.jinja").read_text(encoding="utf-8")

ANSWER = "Item ontfaen van Janne"


def processor_with(template) -> SimpleNamespace:
    """A processor as far as `template_kwargs` looks at one."""
    return SimpleNamespace(chat_template=template, tokenizer=SimpleNamespace(chat_template=None))


def render(template: str, messages: list[dict], *, add_generation_prompt: bool,
           **variables) -> str:
    """Render as transformers does: the sandboxed environment, trim/lstrip,
    `raise_exception`, `tojson`, and `bos_token` for templates that print it."""
    jinja2 = pytest.importorskip("jinja2")
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda value, **kw: __import__("json").dumps(value, **kw)
    return env.from_string(template).render(
        messages=messages, add_generation_prompt=add_generation_prompt,
        bos_token="<bos>", **variables)


# ── what each template reads ────────────────────────────────────────────────

@pytest.mark.parametrize("name, template, reads", [
    ("Qwen3.5-4B", QWEN35, True),
    ("Gemma 4 E4B", GEMMA4, True),
    ("olmOCR-2-7B", OLMOCR, False),
    ("Qwen3-VL-4B", QWEN3VL, False),
])
def test_which_templates_read_enable_thinking(name, template, reads):
    """The warning of 21.10. came from the olmOCR-2 run: its template reads no
    `enable_thinking`, so transformers 5 had something to move and warn about."""
    assert template_reads(template, "enable_thinking") is reads, name


@pytest.mark.parametrize("template, expected", [
    (QWEN35, {"enable_thinking": False}),
    (GEMMA4, {"enable_thinking": False}),
    (OLMOCR, {}),
    (QWEN3VL, {}),
])
def test_a_template_gets_the_kwargs_it_reads_and_nothing_else(template, expected):
    assert template_kwargs(processor_with(template)) == expected


def test_every_kwarg_handed_over_is_a_template_variable():
    """The invariant transformers 5 checks before it warns, stated as such:
    nothing `template_kwargs` returns can be moved into processor kwargs."""
    meta = pytest.importorskip("jinja2.meta")
    from jinja2 import Environment
    for template in (QWEN35, GEMMA4, OLMOCR, QWEN3VL):
        variables = meta.find_undeclared_variables(Environment().parse(template))
        assert set(template_kwargs(processor_with(template))) <= variables


def test_the_template_is_found_on_the_processor_or_its_tokenizer():
    on_tokenizer = SimpleNamespace(chat_template=None,
                                   tokenizer=SimpleNamespace(chat_template=QWEN35))
    assert chat_template_of(on_tokenizer) == QWEN35
    assert template_kwargs(on_tokenizer) == {"enable_thinking": False}
    named = SimpleNamespace(chat_template={"default": OLMOCR, "other": QWEN35})
    assert chat_template_of(named) == OLMOCR
    assert template_kwargs(named) == {}
    assert chat_template_of(SimpleNamespace()) is None
    assert template_kwargs(SimpleNamespace()) == {}


def test_without_jinja2_the_word_match_gives_the_same_answers(monkeypatch):
    """The service and the suite import vlm_dataset without an ML stack; jinja2
    is transformers' and may be absent. The fallback must agree on these four."""
    template_reads.cache_clear()
    monkeypatch.setitem(sys.modules, "jinja2", None)
    monkeypatch.setitem(sys.modules, "jinja2.meta", None)
    try:
        assert template_reads(QWEN35, "enable_thinking") is True
        assert template_reads(GEMMA4, "enable_thinking") is True
        assert template_reads(OLMOCR, "enable_thinking") is False
        assert template_reads(QWEN3VL, "enable_thinking") is False
        assert template_reads("{{ enable_thinking_mode }}", "enable_thinking") is False
    finally:
        template_reads.cache_clear()


# ── what enable_thinking does to the Qwen3.5 render (point 3 of #274) ───────

def test_qwen35_generation_prompt_closes_the_think_block_only_with_the_flag():
    """4B treats an unset variable as *on*: the generation prompt ends in an open
    `<think>` and the model reasons before it transcribes. With the flag the
    block is empty and closed. The render is the same whichever transformers
    renders it — the template, not the library, decides."""
    with_flag = render(QWEN35, chat_example("Transcribe."), add_generation_prompt=True,
                       **CHAT_TEMPLATE_KWARGS)
    without = render(QWEN35, chat_example("Transcribe."), add_generation_prompt=True)

    assert with_flag.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert without.endswith("<|im_start|>assistant\n<think>\n")
    assert "</think>" not in without


def test_qwen35_training_render_carries_the_empty_block_and_matches_the_prompt():
    """What the loss is computed over: the assistant turn with the transcription.
    The template puts the empty block in front of it regardless of the flag —
    so the header the model is trained to continue from equals the generation
    prompt only when the flag is off. On, the model would be asked to continue
    from a header it never saw."""
    training = render(QWEN35, chat_example("Transcribe.", ANSWER), add_generation_prompt=False,
                      **CHAT_TEMPLATE_KWARGS)
    header = training[:training.index(ANSWER)]
    assert header.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")

    prompt_off = render(QWEN35, chat_example("Transcribe."), add_generation_prompt=True,
                        **CHAT_TEMPLATE_KWARGS)
    prompt_on = render(QWEN35, chat_example("Transcribe."), add_generation_prompt=True)
    assert prompt_off == header
    assert prompt_on != header


def test_the_flag_never_reaches_a_template_that_does_not_read_it_and_changes_nothing():
    """olmOCR-2: the same render with and without the flag. Nothing was lost by
    withholding it — and under transformers 5 nothing is warned about."""
    for template in (OLMOCR, QWEN3VL):
        a = render(template, chat_example("Transcribe.", ANSWER), add_generation_prompt=False,
                   **CHAT_TEMPLATE_KWARGS)
        b = render(template, chat_example("Transcribe.", ANSWER), add_generation_prompt=False)
        assert a == b and ANSWER in a


def test_gemma4_reads_the_flag_inverted_and_off_means_no_think_token():
    """`train_qlora.assistant_header_ids` records the inversion: on Gemma 4 the
    flag *adds* `<|think|>` to the system turn. Off — what is passed — leaves it out."""
    off = render(GEMMA4, chat_example("Transcribe."), add_generation_prompt=True,
                 **CHAT_TEMPLATE_KWARGS)
    on = render(GEMMA4, chat_example("Transcribe."), add_generation_prompt=True,
                enable_thinking=True)
    assert "<|think|>" not in off
    assert "<|think|>" in on

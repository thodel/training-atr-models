"""Image tokens counted on a real line and a real page, per base and budget (#278).

Two layers. The rule — placeholders in ``input_ids``, the grid's implied count,
the refusal to count zero of an unnamed token, and that ``measure`` encodes the
way the training collator does — runs everywhere, on fakes. The measurement
itself runs only where the processors are cached (the training container, a
laptop that has pulled them) and is **skipped, not passed**, elsewhere: the
figures below were measured on 23.10.2026 with transformers 5.17.0 and 4.57.6
(the Qwen-family three agree between the versions; Gemma 4 loads under 5 only)
on ``tests/fixtures/image_tokens/`` — see its README for where the pixels come
from — and a run that finds other numbers has found a change in the processor,
not in the pixels.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atr_training.contracts import VLM_PIXEL_BUDGET
from atr_training.image_tokens import (
    ImageTokenCount,
    ImageTokensUnknown,
    count_image_tokens,
    image_token_id,
    measure,
)

FIXTURES = Path(__file__).parent / "fixtures" / "image_tokens"

#: (budget, image) -> image tokens the model is handed. Qwen3-VL and Qwen3.5
#: share one image processor (16 px patches merged 2x2 = a 32 px cell) and so
#: one column; olmOCR-2 is Qwen2.5-VL with 14 px patches (a 28 px cell), so the
#: same pixels cost 30 % more; Gemma 4 charges its step whatever the image.
QWEN3 = {("line", "line"): 120, ("line", "page"): 252,
         ("block", "line"): 120, ("block", "page"): 1008,
         ("page", "line"): 120, ("page", "page"): 2040}
EXPECTED = {
    "Qwen/Qwen3-VL-4B-Instruct": QWEN3,
    "Qwen/Qwen3.5-4B": QWEN3,
    "allenai/olmOCR-2-7B-1025": {("line", "line"): 138, ("line", "page"): 320,
                                 ("block", "line"): 138, ("block", "page"): 1312,
                                 ("page", "line"): 138, ("page", "page"): 2668},
    "google/gemma-4-E4B-it": {("line", "line"): 132, ("line", "page"): 130,
                              ("block", "line"): 528, ("block", "page"): 546,
                              ("page", "line"): 1071, ("page", "page"): 1073},
}


# ── the rule, on fakes ───────────────────────────────────────────────────────

class _Tensor:
    """Enough of a tensor: ``shape`` and ``tolist()``."""

    def __init__(self, rows):
        self.rows = rows
        dims, cur = [], rows
        while isinstance(cur, list):
            dims.append(len(cur))
            cur = cur[0] if cur else None
        self.shape = tuple(dims)

    def tolist(self):
        return self.rows


class _ImageProcessor:
    def __init__(self, **attrs):
        for key, value in attrs.items():
            setattr(self, key, value)


class _Processor:
    def __init__(self, image_token_id=None, image_token=None, tokenizer=None, **ip):
        if image_token_id is not None:
            self.image_token_id = image_token_id
        if image_token is not None:
            self.image_token = image_token
        if tokenizer is not None:
            self.tokenizer = tokenizer
        self.image_processor = _ImageProcessor(**ip)


def test_placeholders_are_the_image_token_s_occurrences_in_input_ids():
    count = count_image_tokens(_Processor(image_token_id=7),
                               {"input_ids": [[1, 7, 7, 7, 2, 3]]})
    assert count.placeholders == 3
    assert count.sequence_len == 6
    assert count.grid is None and count.from_grid is None and count.patches is None
    assert count.agrees, "no grid is nothing to disagree with"


def test_tensors_count_the_same_as_lists():
    count = count_image_tokens(_Processor(image_token_id=7),
                               {"input_ids": _Tensor([[7, 7, 1], [7, 1, 1]])}, index=1)
    assert count.placeholders == 1 and count.sequence_len == 3


def test_the_grid_implies_a_count_over_merge_size_squared():
    processor = _Processor(image_token_id=7, merge_size=2)
    inputs = {"input_ids": [[7] * 120 + [1]], "image_grid_thw": _Tensor([[1, 12, 40]]),
              "pixel_values": _Tensor([[0.0] * 4] * 480)}
    count = count_image_tokens(processor, inputs)
    assert count.grid == (1, 12, 40)
    assert count.from_grid == 120, "1*12*40 patches, merged 2x2"
    assert count.patches == 480
    assert count.agrees


def test_a_grid_that_disagrees_with_the_placeholders_is_said_so():
    processor = _Processor(image_token_id=7, merge_size=2)
    count = count_image_tokens(processor, {"input_ids": [[7] * 119 + [1]],
                                           "image_grid_thw": [[1, 12, 40]]})
    assert count.from_grid == 120 and count.placeholders == 119
    assert not count.agrees


def test_a_grid_without_a_known_merge_size_implies_nothing():
    count = count_image_tokens(_Processor(image_token_id=7),
                               {"input_ids": [[7, 7]], "image_grid_thw": [[1, 2, 2]]})
    assert count.grid == (1, 2, 2) and count.from_grid is None
    assert count.agrees


def test_gemma_s_pixel_values_carry_a_leading_batch_axis():
    count = count_image_tokens(_Processor(image_token_id=7),
                               {"input_ids": [[7]], "pixel_values": _Tensor([[[0.0] * 3] * 1260])})
    assert count.patches == 1260


def test_the_image_token_is_looked_up_when_only_its_text_is_known():
    class Tok:
        unk_token_id = 0

        def convert_tokens_to_ids(self, token):
            return {"<|image_pad|>": 151655}.get(token, 0)

    assert image_token_id(_Processor(image_token="<|image_pad|>", tokenizer=Tok())) == 151655
    with pytest.raises(ImageTokensUnknown):
        image_token_id(_Processor(image_token="<|no such|>", tokenizer=Tok()))


def test_a_processor_that_names_no_image_token_is_refused_not_counted_as_zero():
    with pytest.raises(ImageTokensUnknown, match="cannot be counted"):
        count_image_tokens(_Processor(), {"input_ids": [[1, 2, 3]]})


def test_within_budget_compares_placeholders_with_what_the_budget_buys():
    from atr_training.image_tokens import Measurement
    from atr_training.vlm_dataset import AppliedBudget
    budget = AppliedBudget(knob="k", max_pixels=1, cell_px=1, visual_tokens=256, grid_known=True)
    at = Measurement("line", "line", (1, 1), budget, ImageTokenCount(256, 300))
    over = Measurement("line", "line", (1, 1), budget, ImageTokenCount(257, 300))
    assert at.within_budget and not over.within_budget


def test_measure_encodes_the_way_the_training_collator_does():
    """Fresh processor per budget, fit_pixels first, the chat template with the
    filtered kwargs and no generation prompt, images nested one list per text."""
    from PIL import Image

    loads, calls = [], []

    class Fake:
        image_token_id = 7
        chat_template = "{% for m in messages %}{{ m['role'] }}{% endfor %}"

        def __init__(self):
            self.image_processor = _ImageProcessor(size={"longest_edge": 10 ** 9},
                                                   patch_size=16, merge_size=2)

        def apply_chat_template(self, messages, **kwargs):
            calls.append(("template", kwargs))
            return "rendered"

        def __call__(self, text, images, **kwargs):
            calls.append(("encode", text, images, kwargs))
            return {"input_ids": [[7] * 5 + [1]]}

    def load():
        loads.append(Fake())
        return loads[-1]

    big = Image.new("RGB", (2000, 2000))
    rows = measure(load, {"big": big}, kinds=("line", "block"), prompt="Transcribe.")

    assert len(loads) == 2, "one processor per budget: the knob is written onto it"
    assert [r.kind for r in rows] == ["line", "block"]
    assert rows[0].fitted == (512, 512), "2000x2000 fitted to the line budget of 512x512"
    assert rows[1].fitted == (1024, 1024)
    assert rows[0].budget.max_pixels == VLM_PIXEL_BUDGET["line"]
    assert rows[0].count.placeholders == 5
    template, encode = calls[0], calls[1]
    assert template[1]["tokenize"] is False and template[1]["add_generation_prompt"] is False
    assert "enable_thinking" not in template[1], "the template reads no such name (#274)"
    assert encode[1] == ["rendered"]
    assert len(encode[2]) == 1 and len(encode[2][0]) == 1, "one list of images per text"
    assert encode[2][0][0].size == (512, 512)
    assert encode[3] == {"return_tensors": "pt", "padding": True}


# ── the measurement, where the processors are ───────────────────────────────

def _load(model: str):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from transformers import AutoProcessor
    try:
        return lambda: AutoProcessor.from_pretrained(model, local_files_only=True)
    finally:
        del transformers


@pytest.mark.parametrize("model", sorted(EXPECTED))
def test_a_real_line_and_page_cost_what_the_table_says(model: str):
    load = _load(model)
    try:
        load()
    except (OSError, ValueError) as exc:          # not cached, or not loadable here
        pytest.skip(f"{model} is not in the local cache: {str(exc).splitlines()[0][:120]}")
    from PIL import Image

    images = {"line": Image.open(FIXTURES / "line.jpg"), "page": Image.open(FIXTURES / "page.jpg")}
    rows = measure(load, images)
    assert len(rows) == 6

    expected = EXPECTED[model]
    for row in rows:
        where = f"{model} {row.kind}/{row.image}"
        assert row.count.placeholders == expected[(row.kind, row.image)], where
        assert row.within_budget, f"{where}: {row.count.placeholders} > {row.budget.visual_tokens}"
        assert row.count.agrees, f"{where}: grid {row.count.from_grid} vs {row.count.placeholders}"
        if row.budget.stepped:
            # Gemma: the step is the price, paid in full (less the rounding of
            # two edges), by a line and a page alike.
            assert row.count.placeholders >= 0.9 * row.budget.visual_tokens, where
        elif row.image == "page":
            # Qwen: a ceiling, and a page that overruns it is fitted to it — so
            # the ceiling is used, not merely respected. A budget that bounded
            # every image to 16 tokens would pass "within" and fail this.
            assert row.count.placeholders >= 0.95 * row.budget.visual_tokens, where

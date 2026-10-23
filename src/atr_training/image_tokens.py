"""How many image tokens a processor actually hands the model (#278).

``apply_visual_budget`` writes a budget onto the processor and reads it back, and
its docstring is honest about the limit of that: the read-back proves the knob
holds the value, not that the processor honours it. Honouring is only visible on
a real image, as the number of image placeholders in ``input_ids`` — which is
the number the model sees, the one ``max_seq_len`` is spent on, and the one the
``line = 256 tokens`` assumption in :data:`~atr_training.contracts.VLM_PIXEL_BUDGET`
is about.

This module counts that number, two ways where two are available:

* **placeholders** — occurrences of the processor's image token in ``input_ids``.
  Every family has this, and it is what reaches the model.
* **from the grid** — Qwen-style processors also return ``image_grid_thw``; its
  product over ``merge_size²`` is what the model *should* see. The two agreeing
  is the processor's own consistency check; disagreeing would mean the
  placeholders and the patches no longer index each other, which is the failure
  of #86 in a new shape.

Nothing here is ever 0 by default. A processor whose image token cannot be
named raises :class:`ImageTokensUnknown` rather than counting zero of something,
because zero image tokens reads like a measurement and is the opposite of one.

The measurements this was written for are in ``scripts/count_image_tokens.py``
and pinned in ``tests/test_image_tokens.py`` where the processors are cached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from atr_training.contracts import VLM_PIXEL_BUDGET
from atr_training.vlm_dataset import (
    AppliedBudget,
    apply_visual_budget,
    chat_example,
    fit_pixels,
    template_kwargs,
)

__all__ = [
    "ImageTokenCount",
    "ImageTokensUnknown",
    "Measurement",
    "count_image_tokens",
    "image_token_id",
    "measure",
]


class ImageTokensUnknown(ValueError):
    """The processor does not say which token stands for an image."""


@dataclass(frozen=True)
class ImageTokenCount:
    """What one encoded sample carries for its image."""

    #: Image placeholders in ``input_ids`` — the tokens the model is given.
    placeholders: int
    #: Length of the whole sequence the placeholders sit in.
    sequence_len: int
    #: ``(t, h, w)`` from ``image_grid_thw`` when the processor reports one.
    grid: tuple[int, int, int] | None = None
    #: ``t*h*w / merge_size²`` — the count the grid implies. None without a grid
    #: or without a known ``merge_size``; never a guess.
    from_grid: int | None = None
    #: Rows of ``pixel_values``: patches before merging (Qwen) or pooling (Gemma).
    patches: int | None = None

    @property
    def agrees(self) -> bool:
        """The grid, where there is one, says the same as the placeholders."""
        return self.from_grid is None or self.from_grid == self.placeholders


def image_token_id(processor) -> int:
    """The id the processor writes for one image patch, or raise.

    ``Qwen3VLProcessor``, ``Qwen2_5_VLProcessor`` and ``Gemma4Processor`` all
    carry ``image_token_id``; older processors carry only ``image_token`` and a
    tokenizer to look it up with. A lookup that lands on the unknown token is no
    answer either.
    """
    found = getattr(processor, "image_token_id", None)
    if found is None:
        token = getattr(processor, "image_token", None)
        tokenizer = getattr(processor, "tokenizer", None)
        if token and tokenizer is not None:
            found = tokenizer.convert_tokens_to_ids(token)
            if found == getattr(tokenizer, "unk_token_id", object()):
                found = None
    if found is None:
        raise ImageTokensUnknown(
            f"{type(processor).__name__} names no image token (no image_token_id, "
            "and image_token does not resolve), so image tokens cannot be counted "
            "— and an uncounted image is not an image of zero tokens")
    return int(found)


def _as_list(value):
    """A tensor, an array or a plain sequence, as nested Python lists."""
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else list(value)


def _shape(value) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(n) for n in shape)
    rows = _as_list(value)
    dims: list[int] = []
    while isinstance(rows, list):
        dims.append(len(rows))
        rows = rows[0] if rows else None
    return tuple(dims) or None


def count_image_tokens(processor, inputs, index: int = 0) -> ImageTokenCount:
    """Count the image tokens of sample *index* in a processor's output.

    *inputs* is what ``processor(text=..., images=..., return_tensors="pt")``
    returned — a mapping with ``input_ids`` and, for a Qwen-style processor,
    ``image_grid_thw``. Tensors, arrays and plain lists are all accepted, so the
    rule can be tested without torch.
    """
    token = image_token_id(processor)
    ids = _as_list(inputs["input_ids"])
    row = ids[index] if ids and isinstance(ids[0], list) else ids
    placeholders = sum(1 for each in row if int(each) == token)

    grid = None
    from_grid = None
    raw_grid = inputs.get("image_grid_thw") if hasattr(inputs, "get") else None
    if raw_grid is not None:
        rows = _as_list(raw_grid)
        t, h, w = (int(n) for n in rows[index])
        grid = (t, h, w)
        merge = getattr(getattr(processor, "image_processor", None), "merge_size", None)
        if merge:
            from_grid = (t * h * w) // (int(merge) ** 2)

    patches = None
    pixels = inputs.get("pixel_values") if hasattr(inputs, "get") else None
    if pixels is not None:
        shape = _shape(pixels)
        if shape and len(shape) >= 2:
            patches = shape[-2]

    return ImageTokenCount(placeholders=placeholders, sequence_len=len(row),
                           grid=grid, from_grid=from_grid, patches=patches)


@dataclass(frozen=True)
class Measurement:
    """One image through one processor set to one granularity's budget."""

    kind: str
    image: str
    #: Width and height after :func:`fit_pixels`, as the collator feeds it.
    fitted: tuple[int, int]
    budget: AppliedBudget
    count: ImageTokenCount

    @property
    def within_budget(self) -> bool:
        """At or under what ``apply_visual_budget`` said the budget buys."""
        return self.count.placeholders <= self.budget.visual_tokens


def measure(load_processor: Callable[[], object], images: dict[str, object],
            kinds: Iterable[str] = ("line", "block", "page"),
            prompt: str = "Transcribe.", text: str = "Rathsbücher") -> list[Measurement]:
    """Encode every image at every granularity's budget, exactly as training does.

    *load_processor* returns a **fresh** processor each time: the budget is
    written onto the processor, so one instance cannot be measured at two.
    *images* maps a label to a PIL image. The encoding mirrors the training
    collator — ``fit_pixels`` first, the chat template with the filtered kwargs,
    then ``processor(text=[...], images=[[image]])`` — so what is counted is what
    a run would spend.
    """
    out: list[Measurement] = []
    for kind in kinds:
        max_pixels = VLM_PIXEL_BUDGET[kind]
        processor = load_processor()
        budget = apply_visual_budget(processor, max_pixels)
        kwargs = template_kwargs(processor)
        for label, image in images.items():
            fitted = fit_pixels(image.convert("RGB"), max_pixels)
            rendered = processor.apply_chat_template(
                chat_example(prompt, text), tokenize=False,
                add_generation_prompt=False, **kwargs)
            inputs = processor(text=[rendered], images=[[fitted]],
                               return_tensors="pt", padding=True)
            out.append(Measurement(kind=kind, image=label, fitted=tuple(fitted.size),
                                   budget=budget, count=count_image_tokens(processor, inputs)))
    return out

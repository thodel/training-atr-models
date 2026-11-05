"""Which ``transformers`` auto class loads a VLM base (#316).

Every base this project trained before Apertus loads through
``AutoModelForImageTextToText``. Apertus 1.5 does not: its model card loads it
through ``AutoModelForMultimodalLM`` (image, audio and text in, text out), and the
image-text auto class has no mapping for ``apertus1p5``. One loader for the three
places that load a base — training, evaluation, the adapter merge — so the
second class is tried only where the first one says it does not know the model,
and every Qwen and Gemma run loads exactly as before.

``transformers`` is imported inside the function: this module is importable in
the repo venv, where there is none.
"""

from __future__ import annotations

__all__ = ["load_vlm"]

#: How ``transformers`` says an auto class has no mapping for a config.
UNMAPPED = "Unrecognized configuration class"


def load_vlm(base_model: str, **kwargs):
    """``from_pretrained`` through the first auto class that knows ``base_model``."""
    import transformers

    try:
        return transformers.AutoModelForImageTextToText.from_pretrained(base_model, **kwargs)
    except ValueError as exc:
        multimodal = getattr(transformers, "AutoModelForMultimodalLM", None)
        if multimodal is None or UNMAPPED not in str(exc):
            raise
        print(f"[load] {base_model}: not an image-text model to transformers; "
              "loading it as AutoModelForMultimodalLM", flush=True)
        return multimodal.from_pretrained(base_model, **kwargs)

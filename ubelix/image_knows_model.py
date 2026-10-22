#!/usr/bin/env python3
"""Does this image's transformers know the spec's base model? Run INSIDE the image (#206).

    apptainer exec --env PYTHONPATH=$REPO/src "$SIF" python3 ubelix/image_knows_model.py spec.json

Prints one line and exits 0 (known), 3 (unknown architecture: do not submit),
or 2 (could not tell: no config.json for the base model here, no transformers).
`submit.sh` runs it before `sbatch` and refuses on 3: 17808414 died after 59
seconds on `model type qwen3_5 but Transformers does not recognize this
architecture`, in an image that cost nothing to ask first.

The verdict is a function of two strings and a set, so it is tested without an
image; what the image contributes is the set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def verdict(model_type: str | None, known: set[str] | None) -> tuple[int, str]:
    """``(exit code, line)`` — the decision, with the evidence in the line."""
    if known is None:
        return 2, "IMAGE=unknown transformers is not importable in this image"
    if not model_type:
        return 2, "IMAGE=unknown the base model's config.json names no model_type"
    if model_type in known:
        return 0, "IMAGE=knows %s" % model_type
    return 3, ("IMAGE=refuses %s: this image's transformers does not know the architecture — "
               "submit with SIF= of an image that does" % model_type)


def model_type_of(base_model: str) -> str | None:
    """``model_type`` from the base model's ``config.json``: a local path, the
    HF cache, or the Hub (the login node has the network; a compute node may not)."""
    local = Path(base_model).expanduser() / "config.json"
    if local.is_file():
        return json.loads(local.read_text(encoding="utf-8")).get("model_type")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(base_model, "config.json")
    return json.loads(Path(path).read_text(encoding="utf-8")).get("model_type")


def known_types() -> set[str] | None:
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    except ImportError:
        return None
    return set(CONFIG_MAPPING_NAMES)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: image_knows_model.py <spec.json>", file=sys.stderr)
        return 2
    spec = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    base_model = spec.get("base_model")
    if not base_model:
        print("IMAGE=knows (the spec names no base_model; nothing to check)")
        return 0
    try:
        model_type = model_type_of(base_model)
    except Exception as exc:  # noqa: BLE001 — "could not tell" is exit 2, never a crash
        print("IMAGE=unknown could not read %s's config.json: %s" % (base_model, exc))
        return 2
    code, line = verdict(model_type, known_types())
    print(line)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

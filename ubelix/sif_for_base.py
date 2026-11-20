#!/usr/bin/env python3
"""The UBELIX image a VLM job on BASE_MODEL runs in, by the trainer's own rule (#427).

    python3 ubelix/sif_for_base.py <base_model> [<home>]

Prints the image path and exits 0, or names the reason and exits 3 when no
UBELIX image can load that base.

A batch file with a fixed default image sends a Qwen3.5 adapter into
transformers 4: `score_federal_minutes` 22861171 and 22861173 (20.11.2026) died
after 40 s on "model type `qwen3_5` but Transformers does not recognize this
architecture", while the Qwen3.5 page arm in `vlm-train-tf5.sif` ran. The rule
is `atr_training.backends.needs_transformers_5` — the one the trainer uses to
pick its venv — so the two cannot drift apart. Stdlib only, without site-packages:
it runs on the host python3 of a compute node, before the container starts.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _backends():
    """``atr_training/backends.py`` loaded as a file, not through the package:
    ``atr_training/__init__`` imports ``contracts`` and with it pydantic, which a
    compute node's python3 does not have. ``backends`` itself is stdlib only.
    (CI caught it on 20.11.2026; the dev box's python3 happened to have pydantic.)"""
    path = Path(__file__).resolve().parents[1] / "src" / "atr_training" / "backends.py"
    spec = importlib.util.spec_from_file_location("_atr_backends", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module      # dataclasses resolves annotations through it
    spec.loader.exec_module(module)
    return module


_b = _backends()
needs_apertus_fork, needs_transformers_5 = _b.needs_apertus_fork, _b.needs_transformers_5

TF4 = "ubelix/vlm-train.sif"
TF5 = "ubelix/vlm-train-tf5.sif"


def sif_for(base_model: str, home: str) -> str:
    """The image for ``base_model`` under ``home``; ValueError if there is none."""
    if needs_apertus_fork(base_model):
        raise ValueError(f"{base_model}: Apertus 1.5 needs swiss-ai's transformers fork, "
                         "and no UBELIX image has it (training-atr-models#316)")
    return str(Path(home) / (TF5 if needs_transformers_5(base_model) else TF4))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not 1 <= len(argv) <= 2 or not argv[0]:
        print("usage: sif_for_base.py <base_model> [<home>]", file=sys.stderr)
        return 2
    home = argv[1] if len(argv) == 2 else str(Path.home())
    try:
        print(sif_for(argv[0], home))
    except ValueError as exc:
        print(f"no image: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

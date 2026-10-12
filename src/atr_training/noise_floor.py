"""What every noise-floor measurement shares, whichever engine ran it (#115).

Two measurements exist: ``scripts/measure_noise_floor.py`` drives ``ketos``
itself, and ``scripts/measure_vlm_noise_floor.py`` goes through the trainer.
They differ in how a run is made and agree on how its result is read — a spread
over empty hypotheses is no resolution — and on how the number reaches the
manifest, with its provenance and without erasing the comments beside it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

__all__ = ["COLLAPSE_LENGTH_RATIO", "is_collapsed", "write_floor", "splice_noise_floor"]

#: A hypothesis this short is not a bad reading, it is no reading. CTC blank
#: collapse emits the blank label at nearly every timestep, so the output is
#: empty and every reference character counts as missing. A VLM has no blank
#: label and collapses differently — an empty or one-token answer — but the
#: signal is the same: nothing was read.
COLLAPSE_LENGTH_RATIO = 0.05


def is_collapsed(cer: float | None, length_ratio: float | None) -> bool:
    """Did this run produce nothing, rather than something poor?

    Both signals, because either alone can be misread: a CER at or above 1.0 can
    also come from a model that over-generates, and a length ratio near zero
    could in principle accompany a short but correct reading of a long page. An
    empty hypothesis shows both at once.
    """
    if cer is None:
        return False
    if length_ratio is not None and length_ratio <= COLLAPSE_LENGTH_RATIO:
        return True
    return cer >= 1.0


def write_floor(path: Path, block: Mapping) -> None:
    """Put a measured floor into a sweep manifest, with what it was measured on.

    Retyping the number is how it ends up stale or attached to the wrong corpus,
    so the manifest's own data version is checked against ``block["measured_on"]``
    first. The block is the shape ``sweep_manifest`` validates: a bare number
    would parse, but nobody reading it later could say where it came from.
    """
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    digest = str(((raw.get("data") or {}).get("digest") or "")).strip()
    measured_on = str(block.get("measured_on") or "")
    if digest != measured_on:
        raise ValueError(
            f"{path} runs on data {digest!r} and this floor was measured on "
            f"{measured_on!r}. A floor belongs to the corpus and the budget it "
            "was measured at; writing it here would license a ranking it never "
            "earned.")
    splice_noise_floor(path, {k: v for k, v in block.items() if v is not None})


def splice_noise_floor(path: Path, block: Mapping) -> None:
    """Write the block into the manifest **without re-dumping the file**.

    ``yaml.safe_dump`` of the parsed document would have worked and would have
    deleted every comment in it — and in these manifests the comments are the
    reasoning: which axes were left out and why, which base and why not another.
    A measurement that silently erased that would have cost more than it added.
    """
    import yaml

    rendered = yaml.safe_dump({"noise_floor": dict(block)}, sort_keys=False,
                              allow_unicode=True).rstrip("\n").splitlines()
    lines = path.read_text(encoding="utf-8").splitlines()
    kept, index, replaced = [], 0, False
    while index < len(lines):
        line = lines[index]
        if line.startswith("noise_floor:"):
            # Drop the old block: itself and everything indented under it.
            index += 1
            while index < len(lines) and (not lines[index].strip()
                                          or lines[index][:1].isspace()):
                index += 1
            kept.extend(rendered)
            replaced = True
            continue
        kept.append(line)
        index += 1
    if not replaced:
        # Before `notes:` when there is one, so the prose stays at the bottom
        # where a reader expects it; otherwise at the end.
        at = next((i for i, line in enumerate(kept) if line.startswith("notes:")),
                  len(kept))
        kept[at:at] = rendered + [""]
    path.write_text("\n".join(kept).rstrip("\n") + "\n", encoding="utf-8")

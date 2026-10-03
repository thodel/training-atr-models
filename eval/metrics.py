"""Ground-truth loading for the eval harness; CER/WER come from the trainer's.

The metrics themselves live in :mod:`atr_training.textmetrics` and are
re-exported here. One implementation, so a CER printed by an eval run and a CER
recorded on a training job are the same number computed the same way — the whole
point of `eval/run_eval.py` comparing a freshly trained model against the served
ones. ``score_pairs`` there additionally aggregates corpus-level, which is what
``ketos test`` reports.

That shared implementation is why `eval/` moved here with #11 rather than
staying beside the gateway: `textmetrics` had no other consumer on the serving
side, so the move left it with none at all and the question of a shared package
answered itself.

Ground truth is read from ``.txt`` or PAGE-XML.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# eval/ is run from the repo root, where src/ is not importable unless the venv
# has the package installed. Here that is the NORMAL case, not the exception:
# the harness runs out of .venvs/kraken-train (the only venv on asteraix that
# carries httpx), and nothing installs this repo into it. The fallback is what
# makes `python eval/run_eval.py` work at all.
try:
    from atr_training.textmetrics import cer, levenshtein, score_pairs, wer
except ModuleNotFoundError:  # pragma: no cover - only without an installed package
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from atr_training.textmetrics import cer, levenshtein, score_pairs, wer

#: Kept under its historical private name — eval code imported it directly.
_levenshtein = levenshtein

__all__ = ["cer", "wer", "levenshtein", "score_pairs",
           "parse_page_xml", "load_ground_truth", "find_ground_truth"]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_page_xml(path: Path) -> str:
    """Extract line text from a PAGE-XML file, one TextLine per line, in document order."""
    root = ET.parse(path).getroot()
    lines: list[str] = []
    for el in root.iter():
        if _local(el.tag) != "TextLine":
            continue
        for sub in el.iter():
            if _local(sub.tag) == "Unicode" and sub.text:
                lines.append(sub.text)
                break
    return "\n".join(lines)


def load_ground_truth(path: Path) -> str:
    if path.suffix.lower() == ".xml":
        return parse_page_xml(path)
    return path.read_text(encoding="utf-8").strip()


def find_ground_truth(image_path: Path, gt_dir: Path | None) -> Path | None:
    """Locate ground truth for an image: <stem>.txt / .gt.txt / .xml in gt_dir
    (default: alongside the image)."""
    base = gt_dir if gt_dir is not None else image_path.parent
    for name in (f"{image_path.stem}.txt", f"{image_path.stem}.gt.txt", f"{image_path.stem}.xml"):
        cand = base / name
        if cand.is_file():
            return cand
    return None

#!/usr/bin/env python3
"""A kraken model's reading of every line of an evaluation set, line by line (agentic_historian#875).

``ketos test`` — how the kraken backend evaluates (``ketos_cmd.evaluate_cmd``,
``Metrics`` is filled from its report) — prints accuracies and a confusion table
and nothing per line. The scorer measurement (``measure_scorer_tau.py``,
``calibrate_veto_margin.py``) needs the reading of each line, keyed by the same
``image`` path as the eval set's ``val.jsonl``, beside the VLM's ``raw`` and
TrOCR's ``prediction``. This writes it.

The lines are crops already (``materialize_benchmark.py``, or any run's
``val.jsonl``), so there is no segmentation: each crop is handed to kraken as one
line covering the whole image — what ``kraken -n`` (``--no-segmentation``) does,
a ``BBoxLine`` of ``(0, 0, width, height)``. ``--line-type baselines`` gives a
baseline model a rectangle boundary and a straight baseline instead, for the
comparison if the bbox path reads a baseline model badly.

    scripts/read_lines_kraken.py --model <trained>/<id>/<id>.mlmodel \\
        --val-jsonl $E/data/val.jsonl --data-root $E \\
        --report $E/readings/kraken-german-xix-v1.json --label kraken-german-xix-v1

**Unicode form.** The kraken models here are trained with ``--normalization NFD``
(``ketos_cmd``), so they write ``u`` + combining diaeresis where the reference
has ``ü``. Scored raw against the NFC references, every umlaut would be two
errors and an unseen character to the scorer — a kraken penalty that is not in
the reading. The readings are therefore brought to **NFC** before they are
written (``--normalize``, recorded in the report); ``none`` keeps them raw.

Writes ``<report>.json`` (CER over the lines, ``engine: kraken``, the model) and
``<report>.raw.jsonl``, one ``{image, raw, reference}`` per line — the shape
``measure_scorer_tau.load_predictions`` reads. Runs in the kraken venv or
``kraken-train.sif`` (kraken 7.0.2); kraken is imported only there, so the
tests need none.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from pathlib import Path
from typing import Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atr_training.textmetrics import score_pairs  # noqa: E402
from atr_training.vlm_dataset import Sample, read_jsonl  # noqa: E402

#: image path -> the model's reading of that one line
Recogniser = Callable[[Path], str]


def select(pool: list[Sample], max_samples: int | None, seed: int) -> tuple[list[Sample], str]:
    """All lines, or the seeded draw ``evaluate_qlora`` and ``evaluate_trocr``
    make — the same rule, so a capped kraken reading covers the same lines as a
    capped VLM reading at the same seed."""
    if max_samples is not None and len(pool) > max_samples:
        drawn = random.Random(seed).sample(sorted(pool, key=lambda s: s.image), max_samples)
        return drawn, f"seeded draw of {max_samples} from {len(pool)}, seed {seed}"
    return pool, f"all {len(pool)} samples"


def read_lines(recognise: Recogniser, samples: Iterable[Sample], root: Path,
               progress_every: int = 250, normalize: str | None = "NFC") -> list[dict]:
    """One ``{image, raw, reference}`` row per sample, in the order given, the
    reading in Unicode form ``normalize`` (None: as the model wrote it)."""
    rows = []
    for index, sample in enumerate(samples, 1):
        text = recognise(root / sample.image)
        if normalize:
            text = unicodedata.normalize(normalize, text)
        rows.append({"image": sample.image, "raw": text, "reference": sample.text})
        if progress_every and index % progress_every == 0:
            print(f"{index} lines read", flush=True)
    return rows


def kraken_recogniser(model: Path, *, device: str = "cpu", line_type: str = "bbox",
                      padding: int = 16) -> Recogniser:
    """A function that reads one line crop with a kraken 7 recognition model.

    Imported lazily: kraken lives in its own venv / image, never in this tree's
    test environment.
    """
    from PIL import Image
    from kraken.configs import RecognitionInferenceConfig
    from kraken.containers import BaselineLine, BBoxLine, Segmentation
    from kraken.ketos.util import to_ptl_device
    from kraken.tasks import RecognitionTaskModel

    accelerator, devices = to_ptl_device(device)
    config = RecognitionInferenceConfig(accelerator=accelerator, device=devices,
                                        num_line_workers=0, batch_size=1, padding=padding)
    task = RecognitionTaskModel.load_model(model)
    print(f"kraken model {model}: trained on {task.seg_type or 'unknown'} lines, "
          f"read as {line_type}", flush=True)

    def recognise(path: Path) -> str:
        with Image.open(path) as raw:
            im = raw.convert("RGB")   # the transforms take it to what the model wants
        w, h = im.size
        if line_type == "baselines":
            # inside the image: kraken refuses a polygon that touches its far edge
            right, bottom = w - 1, h - 1
            line = BaselineLine(id="l0", baseline=[(0, round(bottom * 0.75)),
                                                   (right, round(bottom * 0.75))],
                                boundary=[(0, 0), (right, 0), (right, bottom), (0, bottom)])
        else:
            line = BBoxLine(id="l0", bbox=(0, 0, w, h))
        seg = Segmentation(type=line.type, imagename=str(path), text_direction="horizontal-lr",
                           script_detection=False, lines=[line])
        return "".join(record.prediction for record in task.predict(im, seg, config)).strip()

    return recognise


def write_report(rows: list[dict], report: Path, *, label: str, model: str,
                 selection: str, pool: int, extra: dict | None = None) -> dict:
    score = score_pairs((r["raw"], r["reference"]) for r in rows)
    out = score.as_report()
    out.update({"engine": "kraken", "model": label, "model_path": model,
                "granularity": "line", "eval_selection": selection, "val_total": pool,
                **(extra or {})})
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with report.with_suffix(".raw.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def main(argv: list[str] | None = None, recogniser: Recogniser | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="a kraken recognition model "
                    "(.mlmodel or .safetensors)")
    ap.add_argument("--val-jsonl", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True,
                    help="what the val.jsonl image paths resolve against")
    ap.add_argument("--report", type=Path, required=True,
                    help="<name>.json; the readings go to <name>.raw.jsonl")
    ap.add_argument("--label", default=None,
                    help="the model's name in the report (default: the report's stem)")
    ap.add_argument("--device", default="cpu", help="cpu, cuda:0, ...")
    ap.add_argument("--line-type", choices=("bbox", "baselines"), default="bbox")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="a seeded draw of this many lines (default: every line)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--normalize", choices=("NFC", "NFD", "none"), default="NFC",
                    help="Unicode form of the readings (default NFC, the references' form)")
    args = ap.parse_args(argv)

    pool = list(read_jsonl(args.val_jsonl))
    if not pool:
        raise SystemExit(f"{args.val_jsonl} has no lines")
    samples, selection = select(pool, args.max_samples, args.seed)
    print(f"eval selection: {selection}", flush=True)
    if recogniser is None:
        recogniser = kraken_recogniser(Path(args.model), device=args.device,
                                       line_type=args.line_type)
    normalize = None if args.normalize == "none" else args.normalize
    rows = read_lines(recogniser, samples, args.data_root, normalize=normalize)
    label = args.label or args.report.stem
    out = write_report(rows, args.report, label=label, model=str(args.model),
                       selection=selection, pool=len(pool),
                       extra={"line_type": args.line_type, "normalize": normalize})
    print(f"CER {out['cer']:.4f}  over {out['samples']} lines -> {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

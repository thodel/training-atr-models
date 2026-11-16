"""scripts/read_lines_kraken.py — a kraken model's reading per line (#875).

``ketos test`` reports accuracies and nothing per line, so kraken had no place
in the cross-engine pairs of ``measure_scorer_tau.py``. The reader has to key
every reading by the eval set's own ``image`` path, write the shape
``load_predictions`` reads, and stay importable without kraken. A fake
recogniser stands in for the model.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rlk = _load("read_lines_kraken")
mst = _load("measure_scorer_tau")

LINES = {"data/crops/benchmark_P/000000.jpg": "Der Bundesrat beschliesst",
         "data/crops/benchmark_P/000001.jpg": "in Anwesenheit des Kanzlers",
         "data/crops/benchmark_P/000002.jpg": "wird dem Departement"}


def _eval_set(tmp_path: Path) -> Path:
    val = tmp_path / "data" / "val.jsonl"
    val.parent.mkdir(parents=True)
    val.write_text("".join(json.dumps({"image": i, "text": t, "source_type": "line"}) + "\n"
                           for i, t in LINES.items()), encoding="utf-8")
    return val


def _fake(seen: list[Path]):
    """Reads every line right except the one about the Kanzler."""
    def recognise(path: Path) -> str:
        seen.append(path)
        rel = "/".join(path.parts[-4:])
        text = LINES[rel]
        return text.replace("Kanzlers", "Kanzlcrs")
    return recognise


def test_every_line_is_read_once_and_keyed_by_the_eval_sets_image_path(tmp_path):
    val = _eval_set(tmp_path)
    seen: list[Path] = []
    report = tmp_path / "readings" / "kraken-xix.json"
    rc = rlk.main(["--model", "m.mlmodel", "--val-jsonl", str(val), "--data-root", str(tmp_path),
                   "--report", str(report)], recogniser=_fake(seen))
    assert rc == 0
    # resolved against --data-root, in the eval set's order
    assert seen == [tmp_path / i for i in LINES]
    rows = [json.loads(x) for x in report.with_suffix(".raw.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert [r["image"] for r in rows] == list(LINES)
    assert rows[1] == {"image": "data/crops/benchmark_P/000001.jpg",
                       "raw": "in Anwesenheit des Kanzlcrs",
                       "reference": "in Anwesenheit des Kanzlers"}
    summary = json.loads(report.read_text(encoding="utf-8"))
    assert summary["engine"] == "kraken" and summary["model"] == "kraken-xix"
    assert summary["samples"] == 3 and summary["errors"] == 1
    assert summary["cer"] == pytest.approx(1 / sum(len(t) for t in LINES.values()))
    assert summary["eval_selection"] == "all 3 samples"


def test_the_readings_are_what_measure_scorer_tau_reads(tmp_path):
    val = _eval_set(tmp_path)
    report = tmp_path / "kraken-xix.json"
    rlk.main(["--model", "m", "--val-jsonl", str(val), "--data-root", str(tmp_path),
              "--report", str(report), "--label", "kraken-german-xix-v1"], recogniser=_fake([]))
    label, preds = mst.load_predictions(report)
    assert label == "kraken-german-xix-v1"
    assert set(preds) == set(mst.load_references(val))
    assert mst.load_engine(report) == "kraken"


def test_a_capped_reading_draws_the_lines_the_vlm_evaluator_would(tmp_path):
    """The same seeded rule as evaluate_qlora/evaluate_trocr, so capped readings
    of two engines at one seed cover the same lines."""
    from atr_training.vlm_dataset import read_jsonl

    pool = list(read_jsonl(_eval_set(tmp_path)))
    a, sel = rlk.select(pool, 2, seed=42)
    b, _ = rlk.select(list(reversed(pool)), 2, seed=42)
    assert [s.image for s in a] == [s.image for s in b] and len(a) == 2
    assert sel == "seeded draw of 2 from 3, seed 42"
    assert rlk.select(pool, None, 42)[0] == pool



def test_a_decomposed_reading_is_brought_to_the_references_form(tmp_path):
    """ketos trains with NFD here; an NFD umlaut against an NFC reference would
    be two errors and an unseen character to the scorer, not a misreading."""
    import unicodedata

    val = tmp_path / "data" / "val.jsonl"
    val.parent.mkdir(parents=True)
    val.write_text(json.dumps({"image": "a.jpg", "text": "Bürgerrecht für Müller"}) + "\n",
                   encoding="utf-8")

    def nfd(path):
        return unicodedata.normalize("NFD", "Bürgerrecht für Müller")

    report = tmp_path / "k.json"
    rlk.main(["--model", "m", "--val-jsonl", str(val), "--data-root", str(tmp_path),
              "--report", str(report)], recogniser=nfd)
    summary = json.loads(report.read_text(encoding="utf-8"))
    assert summary["cer"] == 0.0 and summary["normalize"] == "NFC"
    rlk.main(["--model", "m", "--val-jsonl", str(val), "--data-root", str(tmp_path),
              "--report", str(report), "--normalize", "none"], recogniser=nfd)
    assert json.loads(report.read_text(encoding="utf-8"))["errors"] > 0

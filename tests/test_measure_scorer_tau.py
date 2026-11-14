"""scripts/measure_scorer_tau.py — L2 of agentic_historian#491: τ between bits/char and CER.

The script decides the epic (τ < 0.3 on plausible candidates → veto only), so it
has to measure what it claims: a reading that is noisier against the reference
must cost more bits under a scorer of the right register, and the verdict has to
follow the number, in both directions.
"""

from __future__ import annotations

import importlib.util
import json
import random

import pytest
from pathlib import Path

from atr_training.scorer import Cell, CharNgramScorer

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "measure_scorer_tau.py"
spec = importlib.util.spec_from_file_location("measure_scorer_tau", SCRIPT)
mst = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mst)

TEXT = ("Der Regierungsrat hat an seiner heutigen Sitzung die Verordnung über die Benützung der "
        "kantonalen Strassen genehmigt und die Direktion des Innern beauftragt den Vollzug zu "
        "regeln die Gemeinden werden mit Kreisschreiben über die neuen Bestimmungen informiert "
        "die Rechnung des vergangenen Jahres schliesst mit einem Überschuss ab welcher dem "
        "Reservefonds zugewiesen wird der Grosse Rat genehmigt das Budget der Staatskanzlei ")


def _lines() -> list[str]:
    words = TEXT.split()
    return [" ".join(words[i:i + 7]) for i in range(0, len(words) - 7, 4)]


def _noisy(text: str, rate: float, rng: random.Random) -> str:
    out = []
    for ch in text:
        r = rng.random()
        if r < rate / 2:
            out.append(rng.choice("qxzkjvw"))
        elif r < rate:
            continue
        else:
            out.append(ch)
    return "".join(out)


def _scorer(tmp_path: Path) -> Path:
    rng = random.Random(0)
    pages = [(f"d{i}", "\n".join(rng.sample(_lines(), 6))) for i in range(30)]
    s = CharNgramScorer(Cell(lang="de", region="Swiss", document_type="protocol",
                             century_band="19-20"), order=5).train(pages, min_chars=1_000)
    s.calibrate([(f"h{i}", "\n".join(rng.sample(_lines(), 5))) for i in range(6)])
    return s.save(tmp_path / "xix.scorer.json")


def _material(tmp_path: Path, rates: dict[str, float]) -> tuple[Path, list[Path]]:
    rng = random.Random(1)
    refs = [{"image": f"l{i:03d}.jpg", "text": t} for i, t in enumerate(_lines())]
    val = tmp_path / "val.jsonl"
    val.write_text("".join(json.dumps(r) + "\n" for r in refs), encoding="utf-8")
    reports = []
    for model, rate in rates.items():
        rep = tmp_path / f"eval_report.{model}.json"
        rep.write_text(json.dumps({"model": model}), encoding="utf-8")
        rows = [{"image": r["image"], "raw": _noisy(r["text"], rate, rng), "at_cap": False}
                for r in refs]
        rep.with_suffix(".raw.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
        reports.append(rep)
    return val, reports


def test_kendall_tau_b_on_known_orderings():
    assert mst.kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert mst.kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert mst.kendall_tau_b([1, 1, 1], [1, 2, 3]) is None          # one side constant
    assert mst.kendall_tau_b([1], [1]) is None
    assert abs(mst.kendall_tau_b([1, 2, 3, 4], [1, 3, 2, 4]) - 2 / 3) < 1e-9


def test_noisier_readings_cost_more_bits_and_the_verdict_says_so(tmp_path):
    scorer = _scorer(tmp_path)
    val, reports = _material(tmp_path, {"clean": 0.02, "worn": 0.15, "bad": 0.45})
    refs = mst.load_references(val)
    runs = dict(mst.load_predictions(p) for p in reports)
    table = mst.measure(refs, runs, {"xix": CharNgramScorer.load(scorer)}, plausible=0.3)
    s = table["scorers"]["xix"]
    assert s["cell"] == "de|Swiss|protocol|19-20"
    # across models: of two readings of one line, the noisier one costs more bits
    assert s["pairs"]["n"] > 30 and s["pairs"]["ordered_like_cer"] > 0.8
    # the bad arm hardly has plausible lines, the clean one is nearly all plausible
    assert s["per_model"]["bad"]["n_plausible"] < s["per_model"]["clean"]["n_plausible"]
    assert s["per_model"]["bad"]["mean_bits"] > s["per_model"]["clean"]["mean_bits"]
    text = mst.render(table)
    assert "| xix |" in text and "pairs across models" in text
    assert ("orders plausible candidates" in text) or ("veto only" in text)


def test_the_verdict_follows_the_threshold_in_both_directions():
    table = {"scorers": {"a": {"cell": "de|*|*|*", "per_model": {
        "m1": {"tau_plausible": 0.12}, "m2": {"tau_plausible": 0.22}}},
        "b": {"cell": "de|Swiss|*|*", "per_model": {"m1": {"tau_plausible": 0.44}},
              "pairs": {"ordered_like_cer_plausible": 0.567}},
        "c": {"cell": "la|*|*|*", "per_model": {"m1": {"tau_plausible": None}}},
        # the 14.11. shape: the best arm clears 0.3, the median does not
        "d": {"cell": "de|*|*|*", "per_model": {"m1": {"tau_plausible": 0.21}, "m2": {"tau_plausible": 0.25},
                                                 "m3": {"tau_plausible": 0.44}}}}}
    v = mst.verdict(table, threshold=0.3)
    assert "a (de|*|*|*): τ on the plausible band 0.12…0.22 (median 0.17) < 0.3 — veto only" in v
    assert "b (de|Swiss|*|*): τ on the plausible band 0.44…0.44 (median 0.44) — orders plausible" in v
    assert "cross-model pairs ordered like CER 0.57" in v
    assert "c (la|*|*|*): no plausible band" in v
    assert "d (de|*|*|*): τ on the plausible band 0.21…0.44 (median 0.25) < 0.3 — veto only" in v


def test_cli_writes_the_json_and_names_the_scorer(tmp_path, capsys):
    scorer = _scorer(tmp_path)
    val, reports = _material(tmp_path, {"clean": 0.02, "bad": 0.4})
    out = tmp_path / "tau.json"
    rc = mst.main([str(val), *map(str, reports), "--scorer", f"xix={scorer}", "--json", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data["scorers"]) == {"xix"} and data["n_lines"] == len(_lines())
    assert data["scorers"]["xix"]["per_model"]["clean"]["n"] == len(_lines())
    assert "xix" in capsys.readouterr().out


def test_reports_are_labelled_by_name_model_or_job_directory_and_read_both_shapes(tmp_path):
    # a bare job report: no JSON beside it, so the job directory is the label
    job = tmp_path / "20261012T135709Z-vlm-xix-noise-s43" / "data"
    job.mkdir(parents=True)
    raw = job / "eval_report.raw.jsonl"
    raw.write_text(json.dumps({"image": "a.jpg", "raw": "lesung"}) + "\n", encoding="utf-8")
    label, preds = mst.load_predictions(raw)
    assert label == "20261012T135709Z-vlm-xix-noise-s43" and preds == {"a.jpg": "lesung"}
    assert mst.load_predictions(job / "eval_report.json") == (label, preds)     # via the .json name
    # the TrOCR trainer writes prediction/reference
    trocr = tmp_path / "t" / "data"
    trocr.mkdir(parents=True)
    (trocr / "eval_report.raw.jsonl").write_text(
        json.dumps({"image": "a.jpg", "reference": "ref", "prediction": "pred"}) + "\n", encoding="utf-8")
    assert mst.load_predictions(trocr / "eval_report.raw.jsonl")[1] == {"a.jpg": "pred"}
    # a given name wins over everything
    assert mst.load_predictions(raw, "arm1")[0] == "arm1"
    # the model field in the JSON wins over the directory
    (job / "eval_report.json").write_text(json.dumps({"model": "qwen-xix"}), encoding="utf-8")
    assert mst.load_predictions(job / "eval_report.json")[0] == "qwen-xix"


def test_cli_takes_name_equals_path_and_refuses_two_runs_under_one_label(tmp_path):
    scorer = _scorer(tmp_path)
    val, reports = _material(tmp_path, {"clean": 0.02, "bad": 0.4})
    out = tmp_path / "tau.json"
    rc = mst.main([str(val), f"one={reports[0]}", f"two={reports[1].with_suffix('.raw.jsonl')}",
                   "--scorer", f"xix={scorer}", "--json", str(out)])
    assert rc == 0
    assert set(json.loads(out.read_text(encoding="utf-8"))["scorers"]["xix"]["per_model"]) == {"one", "two"}
    with pytest.raises(SystemExit):
        mst.main([str(val), f"same={reports[0]}", f"same={reports[1]}", "--scorer", f"xix={scorer}"])

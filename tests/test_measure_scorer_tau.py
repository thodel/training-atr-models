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


# ── across engines: the L4 rule of agentic_historian#875 ─────────────────────
def _engine_material(tmp_path: Path, arms: dict[str, tuple[str, float, dict]]) -> tuple[Path, list[str]]:
    """arms: label -> (engine, noise rate, report JSON). Returns val and NAME=PATH specs."""
    rng = random.Random(7)
    refs = [{"image": f"l{i:03d}.jpg", "text": t} for i, t in enumerate(_lines())]
    val = tmp_path / "val.jsonl"
    val.write_text("".join(json.dumps(r) + "\n" for r in refs), encoding="utf-8")
    specs = []
    for label, (engine, rate, meta) in arms.items():
        rep = tmp_path / f"{label}.json"
        rep.write_text(json.dumps(meta), encoding="utf-8")
        key = "prediction" if engine == "trocr" else "raw"
        rep.with_suffix(".raw.jsonl").write_text("".join(
            json.dumps({"image": r["image"], key: _noisy(r["text"], rate, rng)}) + "\n"
            for r in refs), encoding="utf-8")
        specs.append(f"{label}={rep}")
    return val, specs


def test_the_engine_is_read_off_each_evaluators_report(tmp_path):
    cases = {"vlm.json": {"adapter": "/x/checkpoints/a", "cer": 0.1},
             "trocr.json": {"checkpoint": "/x/trained/t", "beam_size": 4},
             "kraken.json": {"engine": "kraken", "model": "k"},
             "plain.json": {"model": "m"}}
    for name, data in cases.items():
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    assert mst.load_engine(tmp_path / "vlm.json") == "vlm"
    assert mst.load_engine(tmp_path / "trocr.raw.jsonl") == "trocr"      # via the .raw.jsonl name
    assert mst.load_engine(tmp_path / "kraken.json") == "kraken"
    assert mst.load_engine(tmp_path / "plain.json") is None
    assert mst.load_engine(tmp_path / "missing.json") is None


def test_pairs_are_counted_per_engine_pair_and_across_engines(tmp_path):
    scorer = CharNgramScorer.load(_scorer(tmp_path))
    val, specs = _engine_material(tmp_path, {
        "qwen": ("vlm", 0.03, {"adapter": "/a"}), "gemma": ("vlm", 0.08, {"adapter": "/b"}),
        "trocr": ("trocr", 0.15, {"checkpoint": "/t", "beam_size": 4}),
        "kraken": ("kraken", 0.4, {"engine": "kraken"})})
    refs = mst.load_references(val)
    runs, engines = {}, {}
    for spec in specs:
        name, _, path = spec.partition("=")
        runs[name] = mst.load_predictions(Path(path), name)[1]
        engines[name] = mst.load_engine(Path(path))
    table = mst.measure(refs, runs, {"xix": scorer}, engines=engines)
    s = table["scorers"]["xix"]
    assert table["engines"] == {"gemma": "vlm", "kraken": "kraken", "qwen": "vlm", "trocr": "trocr"}
    assert set(s["pairs_by_engine"]) == {"vlm|vlm", "trocr|vlm", "kraken|vlm", "kraken|trocr"}
    # the per-engine counts partition the pairs; the existing keys are untouched
    assert sum(p["n"] for p in s["pairs_by_engine"].values()) == s["pairs"]["n"]
    assert sum(p["n_plausible"] for p in s["pairs_by_engine"].values()) == s["pairs"]["n_plausible"]
    across = s["pairs_across_engines"]
    assert across["n"] == s["pairs"]["n"] - s["pairs_by_engine"]["vlm|vlm"]["n"]
    # noisier engines cost more bits here, so the scorer orders them like CER
    assert across["ordered_like_cer"] > 0.8
    # the verdict names each engine pair against the 0.7 rule
    text = mst.render(table)
    assert "*pairs kraken–vlm*" in text and "*pairs across engines*" in text
    assert "xix (de|Swiss|protocol|19-20), across engines: pairs ordered like CER" in text
    assert "xix (de|Swiss|protocol|19-20), trocr|vlm:" in text
    assert "vlm|vlm:" not in text.split("\n\n")[-1]            # same engine: not an L4 line


def test_the_engine_verdict_follows_0_7_on_the_plausible_band():
    def s(pl, al):
        return {"n": 100, "ordered_like_cer": al, "n_plausible": 40, "ordered_like_cer_plausible": pl}
    table = {"scorers": {"g": {"cell": "de|*|*|*", "per_model": {},
                               "pairs_by_engine": {"kraken|vlm": s(0.62, 0.91), "trocr|vlm": s(0.74, 0.8),
                                                   "vlm|vlm": s(0.57, 0.6)},
                               "pairs_across_engines": s(0.66, 0.88)}}}
    v = mst.engine_verdict(table, threshold=0.7)
    assert "g (de|*|*|*), across engines: pairs ordered like CER 0.66 on the plausible band" in v
    assert "kraken|vlm: pairs ordered like CER 0.62" in v and v.count("L4 stays unbuilt") == 2
    assert "trocr|vlm: pairs ordered like CER 0.74" in v and "L4 passes this rule" in v
    assert "vlm|vlm" not in v


def test_the_cli_takes_an_engine_for_a_bare_raw_file_and_stays_quiet_on_one_engine(tmp_path, capsys):
    scorer = _scorer(tmp_path)
    val, specs = _engine_material(tmp_path, {"a": ("vlm", 0.03, {"adapter": "/a"}),
                                             "b": ("vlm", 0.2, {"adapter": "/b"})})
    out = tmp_path / "tau.json"
    assert mst.main([str(val), *specs, "--scorer", f"xix={scorer}", "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "L4 stays unbuilt" not in printed and "L4 passes" not in printed   # one engine
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data["scorers"]["xix"]["pairs_by_engine"]) == {"vlm|vlm"}
    # a bare .raw.jsonl names no engine; --engine does
    raw = tmp_path / "k.raw.jsonl"
    raw.write_text((tmp_path / "b.raw.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    assert mst.main([str(val), specs[0], f"k={raw}", "--scorer", f"xix={scorer}",
                     "--engine", "k=kraken", "--json", str(out)]) == 0
    assert set(json.loads(out.read_text(encoding="utf-8"))["scorers"]["xix"]["pairs_by_engine"]) \
        == {"kraken|vlm"}
    with pytest.raises(SystemExit, match="NAME=ENGINE"):
        mst.main([str(val), specs[0], "--scorer", f"xix={scorer}", "--engine", "nobody=kraken"])

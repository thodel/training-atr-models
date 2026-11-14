"""scripts/calibrate_veto_margin.py — the fusion veto's margin from L2's material (agentic_historian#491 L3).

Synthetic readings with known noise: a clean arm, a worn arm and a garbage arm
of the same lines. A margin must count what it loses (plausible readings), what
it removes (garbage), and the fatal case — the line's best reading vetoed — and
the recommendation must follow those counts, in both directions.
"""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import pytest

from atr_training.scorer import Cell, CharNgramScorer

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("calibrate_veto_margin", SCRIPTS / "calibrate_veto_margin.py")
cvm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cvm)

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
    s = CharNgramScorer(Cell(lang="de"), order=5).train(pages, min_chars=1_000)
    s.calibrate([(f"h{i}", "\n".join(rng.sample(_lines(), 5))) for i in range(6)])
    return s.save(tmp_path / "de.scorer.json")


def _material(tmp_path: Path, rates: dict[str, float]) -> tuple[Path, list[Path]]:
    rng = random.Random(1)
    refs = [{"image": f"l{i:03d}.jpg", "text": t} for i, t in enumerate(_lines())]
    val = tmp_path / "val.jsonl"
    val.write_text("".join(json.dumps(r) + "\n" for r in refs), encoding="utf-8")
    reports = []
    for model, rate in rates.items():
        d = tmp_path / model / "data"
        d.mkdir(parents=True)
        raw = d / "eval_report.raw.jsonl"
        raw.write_text("".join(json.dumps({"image": r["image"], "raw": _noisy(r["text"], rate, rng)}) + "\n"
                               for r in refs), encoding="utf-8")
        reports.append(raw)
    return val, reports


def test_the_table_counts_cost_gain_and_the_fatal_case(tmp_path):
    scorer = CharNgramScorer.load(_scorer(tmp_path))
    val, reports = _material(tmp_path, {"clean": 0.02, "worn": 0.15, "garbage": 0.6})
    refs = cvm.load_references(val)
    runs = dict(cvm.load_predictions(p) for p in reports)
    lines = cvm.score_lines(refs, runs, scorer)
    assert len(lines) == len(_lines()) and all(len(c) == 3 for c in lines.values())
    wide = cvm.apply_margin(lines, 100.0)                     # nothing is ever that far above the best
    assert wide["vetoed"] == 0 and wide["plausible_lost"] == 0 and wide["fatal"] == 0
    assert wide["single_survivor"] == 0 and wide["candidates"] == 3 * len(lines)
    tight = cvm.apply_margin(lines, 0.0)                      # only the best survives
    assert tight["single_survivor"] == len(lines) and tight["vetoed"] == 2 * len(lines)
    assert tight["single_survivor_mean_cer"] >= tight["single_survivor_best_cer"]
    # measured on this fixture: the worn arm sits 1.8–4.2 bits above the clean one,
    # the garbage arm 7.6–9.7 — so 3.0 cuts all garbage and some of the worn, 5.0 only garbage
    mid, loose = cvm.apply_margin(lines, 3.0), cvm.apply_margin(lines, 5.0)
    # garbage sits further above the best than a plausible reading does: the veto
    # removes a larger share of it than it loses of the plausible band
    assert mid["garbage_removed_share"] == 1.0
    assert 0 < mid["plausible_lost_share"] < 0.5
    assert len(lines) < mid["vetoed"] < tight["vetoed"]
    assert loose["plausible_lost"] == 0 and loose["garbage_removed"] == len(lines)
    # a wider margin vetoes less of both
    assert loose["plausible_lost"] <= mid["plausible_lost"] and loose["garbage_removed"] <= mid["garbage_removed"]
    assert loose["vetoed"] <= mid["vetoed"]
    # fatal: the lowest-CER reading was vetoed — counted per line, never more than the lines
    assert 0 <= mid["fatal"] <= len(lines)
    assert mid["plausible"] == sum(1 for c in lines.values() for x in c if x["cer"] < 0.3)
    assert mid["garbage"] == sum(1 for c in lines.values() for x in c if x["cer"] >= 0.5)


def test_the_recommendation_is_the_smallest_margin_within_both_limits(tmp_path):
    scorer = CharNgramScorer.load(_scorer(tmp_path))
    val, reports = _material(tmp_path, {"clean": 0.02, "worn": 0.15, "garbage": 0.6})
    refs = cvm.load_references(val)
    runs = dict(cvm.load_predictions(p) for p in reports)
    table = cvm.calibrate(refs, runs, scorer, margins=(0.0, 1.0, 2.0, 3.0, 5.0, 100.0),
                          max_plausible_lost=0.05, max_fatal=0.01)
    rows = {r["margin"]: r for r in table["rows"]}
    ok = [m for m, r in rows.items() if r["plausible_lost_share"] <= 0.05 and r["fatal_share"] <= 0.01]
    assert table["recommended_margin"] == min(ok)
    assert table["recommended_margin"] > 0.0                   # margin 0 vetoes plausible readings
    # a limit nothing meets gives no recommendation, and render says so
    strict = cvm.calibrate(refs, runs, scorer, margins=(0.0,), max_plausible_lost=0.0, max_fatal=0.0)
    assert strict["recommended_margin"] is None
    assert "no margin keeps" in cvm.render(strict)
    text = cvm.render(table)
    assert "| margin |" in text and f"recommended margin: {table['recommended_margin']:.2f}" in text


def test_cli_writes_the_json_and_labels_arms_by_job(tmp_path, capsys):
    scorer = _scorer(tmp_path)
    val, reports = _material(tmp_path, {"clean": 0.02, "garbage": 0.6})
    out = tmp_path / "margin.json"
    rc = cvm.main([str(val), *map(str, reports), "--scorer", str(scorer), "--margins", "0.5,1.5",
                   "--json", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["models"] == ["clean", "garbage"] and [r["margin"] for r in data["rows"]] == [0.5, 1.5]
    assert "recommended margin" in capsys.readouterr().out or "no margin keeps" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cvm.main([str(val), f"x={reports[0]}", f"x={reports[1]}", "--scorer", str(scorer)])


def test_the_fatal_case_and_the_survivor_are_counted_on_a_hand_made_line():
    lines = {"a": [{"model": "x", "bits": 1.0, "cer": 0.4}, {"model": "y", "bits": 3.0, "cer": 0.1}],
             "b": [{"model": "x", "bits": 1.0, "cer": 0.0}, {"model": "y", "bits": 9.0, "cer": 0.9}]}
    r = cvm.apply_margin(lines, 1.0)
    assert r["vetoed"] == 2 and r["fatal"] == 1 and r["fatal_share"] == 0.5     # on "a" the better reading went
    assert r["single_survivor"] == 2
    assert r["single_survivor_mean_cer"] == pytest.approx(0.2)                   # (0.4 + 0.0) / 2, the survivors'
    assert r["single_survivor_best_cer"] == pytest.approx(0.05)                  # (0.1 + 0.0) / 2, the lines' best
    assert r["plausible_lost"] == 1 and r["garbage_removed"] == 1
    wide = cvm.apply_margin(lines, 2.5)
    assert wide["fatal"] == 0 and wide["vetoed"] == 1 and wide["single_survivor"] == 1


def test_lines_read_by_one_model_only_are_left_out(tmp_path):
    scorer = CharNgramScorer.load(_scorer(tmp_path))
    refs = {"a.jpg": "der Regierungsrat hat", "b.jpg": "die Gemeinden werden", "c.jpg": "nicht gelesen"}
    runs = {"x": {"a.jpg": "der Regierungsrat hat", "b.jpg": "die Gemeinden werden"},
            "y": {"a.jpg": "der Regierungsrat hat", "z.jpg": "ohne Referenz"}}
    lines = cvm.score_lines(refs, runs, scorer)
    assert set(lines) == {"a.jpg"}                                               # b: one reader; z: no reference

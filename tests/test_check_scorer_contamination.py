"""scripts/check_scorer_contamination.py — did the scorer read the eval lines? (#875)

The Federal Council τ is a lower bound only if the scorer never saw those lines.
Each of the three checks has to fire on the case it exists for and stay quiet
on a clean set.
"""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

from atr_training.scorer import Cell, CharNgramScorer

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_scorer_contamination.py"
spec = importlib.util.spec_from_file_location("check_scorer_contamination", SCRIPT)
csc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(csc)

TRAIN = ("Die Kommission des Nationalrates hat in ihrer Sitzung den Entwurf beraten und "
         "beantragt dem Rate die Annahme des Gesetzes über die Organisation der Bundesrechtspflege "
         "mit den folgenden Abänderungen welche die Mehrheit der Kommission vorschlägt ")
UNSEEN = ["Der Bundesrat hat beschlossen dem Gesuche des Kantons Tessin zu entsprechen",
          "Das Militärdepartement wird ermächtigt die nötigen Anschaffungen zu besorgen",
          "Auf den Antrag des Finanzdepartements wird der Kredit für das Jahr bewilligt",
          "Dem Postdepartement wird die Ausschreibung der Stelle eines Kreispostdirektors"]


def _scorer(tmp_path: Path, extra_pages: list[tuple[str, str]] = ()) -> Path:
    rng = random.Random(0)
    words = TRAIN.split()
    lines = [" ".join(words[i:i + 8]) for i in range(0, len(words) - 8, 3)]
    pages = [(f"000{i:03d}_08362{i % 3}_0201_scan{i}", "\n".join(rng.sample(lines, 5)))
             for i in range(30)] + list(extra_pages)
    s = CharNgramScorer(Cell(lang="de", region="Swiss", document_type="protocol",
                             century_band="19-20"), order=5).train(pages, min_chars=1_000,
                                                                   sources=["nr-sr-xix"])
    s.calibrate([(f"h{i}", "\n".join(rng.sample(lines, 4))) for i in range(6)])
    return s.save(tmp_path / "xix.scorer.json")


def _eval_set(tmp_path: Path, texts: list[str]) -> Path:
    val = tmp_path / "val.jsonl"
    val.write_text("".join(json.dumps({
        "image": f"data/crops/benchmark_P/{i:06d}.jpg", "text": t,
        "page": f"data/benchmarks/P/{i:06d}_{i}_page_9{i}_line_1_docId_5151{i}.xml"}) + "\n"
        for i, t in enumerate(texts)), encoding="utf-8")
    return val


def test_a_clean_set_shows_no_sign(tmp_path, capsys):
    val = _eval_set(tmp_path, UNSEEN)
    draw = tmp_path / "draw.jsonl"
    draw.write_text(json.dumps({"image": "x", "text": TRAIN[:80], "page": "000001_083620_0201_a.xml"})
                    + "\n", encoding="utf-8")
    out = tmp_path / "c.json"
    assert csc.main([str(val), "--scorer", f"xix={_scorer(tmp_path)}", "--draw", str(draw),
                     "--json", str(out)]) == 0
    r = json.loads(out.read_text(encoding="utf-8"))["xix"]
    assert r["documents"] == {"eval_set": 4, "scorer_units": 30, "shared": []}
    assert r["lines"]["verbatim_in_draw"] == 0 and r["lines"]["checked"] == 4
    assert r["perplexity"]["mean_z"] > -1
    assert "no sign of the eval lines" in capsys.readouterr().out


def test_a_shared_document_and_a_verbatim_line_are_named(tmp_path):
    # the scorer trained on a page of document 51512, which the eval set holds too
    scorer = CharNgramScorer.load(_scorer(tmp_path, [("000099_51512_0001_scan", UNSEEN[2])]))
    rows = [json.loads(x) for x in _eval_set(tmp_path, UNSEEN).read_text(encoding="utf-8").splitlines()]
    r = csc.check(rows, scorer, {csc._norm(UNSEEN[1])})
    assert r["documents"]["shared"] == ["51512"]
    assert r["lines"]["verbatim_in_draw"] == 1 and r["lines"]["examples"] == [UNSEEN[1]]
    assert "may have read these lines" in csc.render("xix", r)


def test_memorised_text_reads_far_below_the_held_out_pages(tmp_path):
    """The perplexity check: lines the scorer trained on cost fewer bits than its
    own held-out pages, unseen lines of the register do not."""
    scorer = CharNgramScorer.load(_scorer(tmp_path))
    words = TRAIN.split()
    seen = [" ".join(words[i:i + 8]) for i in range(0, len(words) - 8, 3)]
    rows_seen = [{"image": f"{i}.jpg", "text": t, "page": f"x_docId_{i}.xml"} for i, t in enumerate(seen)]
    rows_new = [{"image": f"{i}.jpg", "text": t, "page": f"x_docId_{i}.xml"} for i, t in enumerate(UNSEEN)]
    z_seen = csc.check(rows_seen, scorer, None)["perplexity"]["mean_z"]
    z_new = csc.check(rows_new, scorer, None)["perplexity"]["mean_z"]
    assert z_seen < z_new


def test_documents_are_read_from_either_naming():
    assert csc.row_document({"page": "data/benchmarks/P/000003_12_page_7_line_1_docId_51514.xml"}) == "51514"
    assert csc.row_document({"page": "data/x/005533_083620_0201_a.xml"}) == "083620"
    assert csc.unit_document("005533_083620_0201_a") == "083620"
    assert csc.unit_document("083620") == "083620"

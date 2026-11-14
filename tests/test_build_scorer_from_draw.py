"""scripts/build_scorer_from_draw.py — L1 of agentic_historian#491 from a job's val.jsonl.

The draw is ground truth, so it may train a scorer; the scored crops' documents
must leave the training text (#23), the split by document is deterministic, and
the artefact that comes out is one the scorer code reads back with the cell,
the sources and the held-out distribution intact.
"""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import pytest

from atr_training.scorer import Cell, CharNgramScorer, ContaminationError

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_scorer_from_draw.py"
spec = importlib.util.spec_from_file_location("build_scorer_from_draw", SCRIPT)
bsd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsd)

WORDS = ("der Regierungsrat hat an seiner heutigen Sitzung die Verordnung über die Benützung der "
         "kantonalen Strassen genehmigt und die Direktion des Innern beauftragt den Vollzug zu regeln "
         "die Gemeinden werden mit Kreisschreiben über die neuen Bestimmungen informiert die Rechnung "
         "schliesst mit einem Überschuss ab welcher dem Reservefonds zugewiesen wird").split()


def _draw(tmp_path: Path, n_docs: int = 30, lines_per_doc: int = 8, name="val.jsonl",
          start: int = 80000) -> Path:
    rng = random.Random(7)
    rows = []
    k = 0
    for d in range(n_docs):
        doc = f"{start + d:06d}"
        for p in range(2):                                   # two pages per document
            page = f"data/pages/005533_{doc}_{p:04d}_{3478000 + k}.xml"
            for _ in range(lines_per_doc // 2):
                rows.append({"image": f"data/crops/val/{k:07d}.jpg",
                             "text": " ".join(rng.sample(WORDS, 6)), "page": page})
                k += 1
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _raw(tmp_path: Path, images: list[str], name="eval_report.raw.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps({"image": i, "raw": "x"}) for i in images) + "\n",
                    encoding="utf-8")
    return path


def test_rows_group_by_the_transkribus_document_in_the_page_name(tmp_path):
    rows = bsd.read_draw(_draw(tmp_path, n_docs=3, lines_per_doc=4))
    by_doc = bsd.pages_by_document(rows)
    assert set(by_doc) == {"080000", "080001", "080002"}
    assert all(len(v) == 4 for v in by_doc.values())          # both pages of a document together
    assert bsd.document_of_row({"image": "c.jpg", "page": "data/pages/odd-name.xml"}) == "odd-name"
    assert bsd.document_of_row({"image": "c.jpg"}) == "c.jpg"


def test_the_split_is_deterministic_and_holds_every_tenth_document_out():
    train, calib = bsd.split_documents([f"{i:03d}" for i in range(25)][::-1])
    assert calib == ["009", "019"] and len(train) == 23
    assert train == sorted(train)


def test_scored_documents_leave_the_training_text_and_the_scorer_refuses_them(tmp_path):
    draw = _draw(tmp_path)
    # crops 0 and 1 lie in document 080000; crop 100 in document 080012
    raw = _raw(tmp_path, ["data/crops/val/0000000.jpg", "data/crops/val/0000001.jpg",
                          "data/crops/val/0000100.jpg"])
    cell = Cell(lang="de", region="Swiss", document_type="protocol", century_band="19-20")
    scorer, report = bsd.build([draw], cell, exclude=[raw], order=4, min_chars=1,
                               sources=["synthetic"], gt_commit="abc")
    assert report["excluded_documents"] == 2 and report["scored_crops"] == 3
    assert "080000" not in scorer.documents and "080012" not in scorer.documents
    assert report["train_documents"] + report["calibration_documents"] == 28
    assert report["heldout"]["n"] == report["calibration_documents"] >= 2
    with pytest.raises(ContaminationError):
        scorer.refuse_if_contaminated(next(iter(scorer.documents)))
    scorer.refuse_if_contaminated("080000")                    # never seen: no refusal


def test_without_exclusions_every_document_is_used_and_the_artefact_reads_back(tmp_path):
    draw = _draw(tmp_path)
    out = tmp_path / "s" / "de.scorer.json"
    rc = bsd.main(["--draw", str(draw), "--lang", "de", "--order", "4", "--min-chars", "1",
                   "--source", "a", "--source", "b", "--gt-commit", "f00", "--out", str(out), "--curve"])
    assert rc == 0 and out.exists()
    s = CharNgramScorer.load(out)
    assert s.cell == Cell(lang="de") and s.sources == ["a", "b"] and s.gt_commit == "f00"
    assert len(s.documents) == 27 and s.heldout["n"] == 3
    assert s.bits_per_char("der Regierungsrat hat") < s.bits_per_char("qxz kjv wq")


def test_two_draws_pool_into_one_text(tmp_path):
    a = _draw(tmp_path, n_docs=10, name="a.jsonl")
    b = _draw(tmp_path, n_docs=10, name="b.jsonl", start=90000)      # other documents
    rows_a = bsd.read_draw(a)
    one, rep1 = bsd.build([a], Cell(lang="de"), order=3, min_chars=1, calib_every=5)
    two, rep2 = bsd.build([a, b], Cell(lang="de"), order=3, min_chars=1, calib_every=5)
    assert rep2["rows"] == 2 * rep1["rows"] == 2 * len(rows_a)
    assert rep2["documents"] == 20 and two.train_chars > one.train_chars

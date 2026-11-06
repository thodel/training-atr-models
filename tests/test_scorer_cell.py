"""A scorer only scores — and says which cell answered (agentic_historian#491 L1, #110).

The pre-experiment in agentic_historian#491 showed the shape: a character n-gram
separates garbage from language cleanly (8.46 vs 1.62 bits/char) and cannot rank
the plausible candidates. So the artefact here is a veto with a z-score, a cell
that names itself, a backoff that admits it is one, and a refusal when it has
seen the document — never a model somebody could ask for text.
"""

from __future__ import annotations

import json
import random

import pytest

from atr_training.scorer import (
    Cell,
    CharNgramScorer,
    ContaminationError,
    ScorerMatrix,
    TooLittleText,
    century_band,
    learning_curve,
    pages_from_lines_manifest,
)

# Two registers, each a short "edition": charter German of the 15th century and
# modern administrative German. Repeated with variation so the n-gram has
# something to count without reading a real corpus.
CHARTER = (
    "Wir Hans von Wiler ritter tuon kunt allen den die disen brief ansehent oder hoerent lesen "
    "das wir mit guter vorbetrachtung und wolbedachtem mut verkouft haben dem erwirdigen "
    "gotzhus ze Kungsvelden unser gut gelegen ze Brugg mit allen rechten nutzen und zugehoerden "
    "um zwenzig guldin guter und genemer ze Zurich und ze Bern geng und geb der wir gentzlich "
    "gewert sind und des ze urkund haben wir unser eigen insigel offenlich gehenkt an disen brief "
    "der geben ist an sant Martins tag des jars do man zalt von Cristus geburt vierzehenhundert "
    "und zwenzig jar "
)
MODERN = (
    "Der Regierungsrat hat an seiner heutigen Sitzung die Verordnung über die Benützung der "
    "kantonalen Strassen genehmigt und die Direktion des Innern beauftragt den Vollzug zu "
    "regeln die Gemeinden werden mit Kreisschreiben über die neuen Bestimmungen informiert "
    "die Rechnung des vergangenen Jahres schliesst mit einem Überschuss von dreitausend Franken "
    "ab welcher dem Reservefonds zugewiesen wird "
)


def _pages(seed_text: str, n: int, prefix: str, *, seed: int = 0) -> list[tuple[str, str]]:
    """``n`` pages of one register, one document each: a random subset of its
    lines in random order, so pages differ in length and content (a held-out
    set of identical pages has no spread, and the scorer refuses that)."""
    rng = random.Random(seed)
    lines = [s.strip() for s in seed_text.replace(" und ", " und\n").split("\n") if s.strip()]
    pages = []
    for i in range(n):
        k = rng.randint(max(2, len(lines) // 2), len(lines))
        chosen = rng.sample(lines, k)
        pages.append((f"{prefix}{i:03d}", "\n".join(chosen * 2)))
    return pages


def _random_letters(n: int, seed: int = 1) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(n))


@pytest.fixture
def charter_cell() -> CharNgramScorer:
    cell = Cell(lang="de", region="Swiss", document_type="charter", century_band="15-16")
    s = CharNgramScorer(cell, order=5).train(_pages(CHARTER, 12, "kf"), min_chars=1_000,
                                            sources=["koenigsfelden-edition"], gt_commit="33f55fc")
    s.calibrate(_pages(CHARTER, 6, "kf-held", seed=7))
    return s


# ── the veto: garbage is far, the own register is near ──────────────────────

def test_the_own_register_scores_far_below_garbage_and_another_register(charter_cell):
    own = charter_cell.bits_per_char("wir haben verkouft dem gotzhus ze Kungsvelden unser gut ze Brugg")
    other = charter_cell.bits_per_char("der Regierungsrat genehmigt die Verordnung über die Strassen")
    garbage = charter_cell.bits_per_char(_random_letters(60))
    assert own < other < garbage
    assert garbage - own > 3.0          # the 8.46-vs-1.62 shape of #491, in small


def test_z_is_read_against_the_cells_own_heldout_text(charter_cell):
    # a held-out page of the register sits inside its own distribution; garbage far outside
    own_page = _pages(CHARTER, 6, "kf-held", seed=7)[2][1]
    z_own, bits = charter_cell.z(own_page)
    z_garbage, _ = charter_cell.z(_random_letters(80))
    assert abs(z_own) < 3.0 and z_garbage > 5.0
    assert charter_cell.heldout["n"] == 6 and charter_cell.heldout["std"] > 0
    assert bits == charter_cell.bits_per_char(own_page)


def test_a_line_is_scored_after_its_predecessor_not_cold(charter_cell):
    # a line and the line above it, as they stand on a trained page
    previous, line = _pages(CHARTER, 12, "kf")[0][1].split("\n")[:2]
    cold = charter_cell.bits_per_char(line)
    warm = charter_cell.bits_per_char(line, previous=previous)
    assert warm < cold


def test_identical_heldout_pages_are_refused_as_a_calibration():
    s = CharNgramScorer(Cell(lang="de"), order=5).train(_pages(CHARTER, 8, "a"), min_chars=1_000)
    with pytest.raises(ValueError, match="indistinguishable"):
        s.calibrate([("h1", CHARTER), ("h2", CHARTER)])


# ── the cell names itself, and a fallback says so ────────────────────────────

def test_century_band_is_two_centuries_wide():
    assert century_band(15) == "15-16" and century_band(16) == "15-16"
    assert century_band(19) == "19-20" and century_band(None) is None


def test_backoff_order_is_the_model_selectors_band_genre_region_generic():
    cell = Cell(lang="de", region="Swiss", document_type="charter", century_band="15-16")
    keys = [c.key() for c in cell.backoff_chain()]
    assert keys == ["de|Swiss|charter|15-16", "de|Swiss|charter|*", "de|Swiss|*|*", "de|*|*|*"]
    assert cell.backoff_chain()[-1].is_generic


def test_matrix_answers_from_the_cell_or_names_the_fallback(charter_cell):
    generic = CharNgramScorer(Cell(lang="de"), order=5).train(
        _pages(CHARTER, 6, "g") + _pages(MODERN, 6, "m"), min_chars=1_000)
    generic.calibrate(_pages(MODERN, 4, "m-held", seed=3))
    matrix = ScorerMatrix()
    matrix.add(charter_cell)
    matrix.add(generic)

    exact = matrix.score("verkouft haben dem gotzhus", {"lang": "de", "region": "Swiss",
                                                        "document_type": "charter", "century": 15})
    assert exact.cell == "de|Swiss|charter|15-16" and exact.fallback is False

    nearby = matrix.score("verkouft haben dem gotzhus", {"lang": "de", "region": "Swiss",
                                                         "document_type": "letter", "century": 19})
    assert nearby.cell == "de|*|*|*" and nearby.fallback is True
    assert nearby.asked == "de|Swiss|letter|19-20"

    with pytest.raises(LookupError, match="no scorer for cell la"):
        matrix.score("in nomine domini", {"lang": "la", "century": 13})


# ── refusals: too little text, a document it has seen ────────────────────────

def test_a_thin_cell_is_not_built_and_says_why():
    s = CharNgramScorer(Cell(lang="de", region="Saxon"), order=5)
    with pytest.raises(TooLittleText, match=r"de\|Saxon\|\*\|\*: .* below the 16,000"):
        s.train(_pages(CHARTER, 1, "x"))
    with pytest.raises(ValueError, match="not trained"):
        s.bits_per_char("anything")


def test_a_document_the_scorer_trained_on_is_refused(charter_cell):
    matrix = ScorerMatrix()
    matrix.add(charter_cell)
    crit = {"lang": "de", "region": "Swiss", "document_type": "charter", "century": 15}
    with pytest.raises(ContaminationError, match="kf003.*training set"):
        matrix.score("verkouft haben", crit, document="kf003")
    assert matrix.score("verkouft haben", crit, document="kf-unseen").cell.endswith("15-16")
    with pytest.raises(ContaminationError, match="both sides"):
        charter_cell.calibrate(_pages(CHARTER, 2, "kf"))     # kf000, kf001 were trained on


# ── the artefact: metadata travels, nothing generates ────────────────────────

def test_the_artefact_carries_cell_amount_sources_commit_and_heldout(tmp_path, charter_cell):
    path = charter_cell.save(tmp_path / "de-swiss-charter-15-16.scorer.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["cell"] == {"lang": "de", "region": "Swiss", "document_type": "charter",
                            "century_band": "15-16"}
    assert data["train_chars"] > 10_000 and data["sources"] == ["koenigsfelden-edition"]
    assert data["gt_commit"] == "33f55fc" and "kf003" in data["documents"]
    assert set(data["heldout"]) == {"mean", "std", "n", "chars"}
    again = ScorerMatrix.load_dir(tmp_path).scorers["de|Swiss|charter|15-16"]
    text = "verkouft haben dem gotzhus ze Kungsvelden"
    assert again.bits_per_char(text) == pytest.approx(charter_cell.bits_per_char(text))
    assert again.z(text)[0] == pytest.approx(charter_cell.z(text)[0])


def test_nothing_on_the_scorer_can_produce_text():
    surface = {n for n in dir(CharNgramScorer) if not n.startswith("_")}
    forbidden = {n for n in surface if any(w in n.lower() for w in
                                           ("generate", "sample", "decode", "complete", "predict", "beam"))}
    assert not forbidden, forbidden
    assert surface >= {"bits_per_char", "z", "train", "calibrate", "save", "load"}


# ── the measurement that sets the floor, and the manifest reader ─────────────

def test_learning_curve_falls_with_more_text():
    curve = learning_curve(_pages(CHARTER, 40, "t"), _pages(CHARTER, 5, "h", seed=9),
                           steps=[500, 5_000, 40_000], order=5)
    chars = [c for c, _ in curve]
    bits = [b for _, b in curve]
    assert chars[0] < chars[1] < chars[2]
    assert bits[0] > bits[1] > bits[2]


def test_pages_from_a_lines_manifest_group_by_document(tmp_path):
    manifest = tmp_path / "lines_train.jsonl"
    rows = [{"image": "line_images_train/000001_DOC7_0001_1.jpg", "text": "erste zeile"},
            {"image": "line_images_train/000002_DOC7_0001_2.jpg", "text": "zweite zeile"},
            {"image": "line_images_train/000003_DOC9_0003_1.jpg", "text": "andere seite"}]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    pages = pages_from_lines_manifest(manifest, document=lambda p: p.rsplit("/", 1)[-1].split("_")[1])
    assert pages == [("DOC7", "erste zeile\nzweite zeile"), ("DOC9", "andere seite")]

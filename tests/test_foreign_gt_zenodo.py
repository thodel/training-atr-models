"""Zenodo-Hinterlegungen einlesen (#198): was anders ist als beim Klonen.

* die versionierte DOI ist Pflicht; der Konzept-DOI wird verweigert;
* die Lizenz steht in der API **und** in einer ``LICENSE*``-Datei; ein Widerspruch
  ist Ablehnung mit beiden Fundstellen, eine unlesbare Datei auch;
* die Vorteilung der Quelle (Train/Validation) wird übernommen und benannt;
* ``__MACOSX``-Schatten und Einträge mit ``..`` werden verworfen und gezählt;
* je Teilsammlung ziehbar: Reichenaus lateinischer Teil ist nicht Vorgabe;
* 18441031 und 18377766 sind als derselbe Seitensatz markiert;
* das Bild zur Seite wird im Archiv gefunden: daneben, eine Ebene höher
  (``page/``), oder in einem eigenen Bild-Archiv — nie geraten.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from atr_training.foreign_gt import SOURCES, ForeignGtError
from atr_training.foreign_gt_zenodo import (
    ZENODO_SOURCES,
    ZenodoPart,
    ZenodoRecord,
    check_licences,
    check_record,
    is_shadow,
    is_unsafe,
    licence_from_text,
    select_pages,
    undeclared_twins,
    zenodo_card,
    zenodo_source_by_id,
)

ROOT = Path(__file__).resolve().parents[1]
NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"


def page(image: str = "", lines: int = 2, text: bool = True) -> bytes:
    body = "".join(f'<TextLine id="l{i}"><TextEquiv><Unicode>z{i}</Unicode></TextEquiv>'
                   f'</TextLine>' if text else f'<TextLine id="l{i}"/>' for i in range(lines))
    attr = f' imageFilename="{image}"' if image else ""
    return (f'<?xml version="1.0"?>\n<PcGts xmlns="{NS}"><Page{attr}>{body}</Page>'
            f'</PcGts>').encode()


ALTO = b'<?xml version="1.0"?>\n<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#"/>'
METS = b'<?xml version="1.0"?>\n<mets:mets xmlns:mets="http://www.loc.gov/METS/"/>'


def zip_of(entries: dict[str, bytes]) -> zipfile.ZipFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    buf.seek(0)
    return zipfile.ZipFile(buf)


def record(**kw) -> ZenodoRecord:
    base = dict(id="x", record="111", doi="10.5281/zenodo.111", licence="CC-BY-4.0",
                licence_at="API", attribution="A", image_source="I", script_kind="S",
                period="P", project="x", target="dh-unibe/image-text_x",
                parts=(ZenodoPart("alle", ("a.zip",)),))
    base.update(kw)
    return ZenodoRecord(**base)


def meta(doi="10.5281/zenodo.111", concept="10.5281/zenodo.110", lic="cc-by-4.0",
         files=("a.zip",), related=()):
    return {"doi": doi, "conceptdoi": concept,
            "metadata": {"license": {"id": lic}, "related_identifiers": list(related)},
            "files": [{"key": k, "checksum": f"md5:{i}"} for i, k in enumerate(files)]}


# ── DOI ──────────────────────────────────────────────────────────────────────

def test_the_versioned_doi_is_required():
    with pytest.raises(ForeignGtError, match="concept DOI"):
        record(doi="10.5281/zenodo.110")


def test_the_api_must_confirm_the_versioned_doi():
    with pytest.raises(ForeignGtError, match="API names doi"):
        check_record(record(), meta(doi="10.5281/zenodo.999"))
    assert check_record(record(), meta()) == ("CC-BY-4.0", {"a.zip": "md5:0"})


def test_every_registered_deposit_is_a_version_not_a_concept():
    for z in ZENODO_SOURCES:
        assert z.doi == f"10.5281/zenodo.{z.record}"


# ── Lizenz an zwei Stellen ───────────────────────────────────────────────────

@pytest.mark.parametrize("text,licence", [
    ("Attribution 4.0 International\n=====", "CC-BY-4.0"),
    ("Attribution-ShareAlike 4.0 International", "CC-BY-SA-4.0"),
    ("Attribution-NonCommercial-ShareAlike 4.0 International", "CC-BY-NC-SA-4.0"),
    ("Lizenz: CC-BY-SA 4.0", "CC-BY-SA-4.0"),
    ("CC0 1.0 Universal", "CC0-1.0"),
    ("All rights reserved.", None),
])
def test_a_licence_file_is_read(text, licence):
    assert licence_from_text(text) == licence


def test_a_licence_that_changed_against_the_register_is_refused():
    with pytest.raises(ForeignGtError, match="CC-BY-SA-4.0.*register CC-BY-4.0"):
        check_record(record(), meta(lic="cc-by-sa-4.0"))


def test_a_contradiction_is_refused_naming_both_places():
    with pytest.raises(ForeignGtError) as err:
        check_licences(record(), "CC-BY-4.0",
                       {"a.zip/LICENSE.txt (im Archiv)": "Attribution-NonCommercial 4.0"})
    assert "metadata.license.id says CC-BY-4.0" in str(err.value)
    assert "a.zip/LICENSE.txt (im Archiv) says CC-BY-NC-4.0" in str(err.value)


def test_an_unreadable_licence_file_is_not_taken_as_agreement():
    with pytest.raises(ForeignGtError, match="names no licence"):
        check_licences(record(), "CC-BY-4.0", {"LICENSE.md": "see website"})


def test_agreeing_places_are_all_named_in_the_card():
    check = check_licences(record(), "CC-BY-4.0", {"LICENSE.md (Datei)": "Attribution 4.0 "
                                                                          "International"})
    assert "metadata.license.id" in check.licence_at and "LICENSE.md (Datei)" in check.licence_at


# ── das Archiv ───────────────────────────────────────────────────────────────

REICHENAU = {
    "R/Train Set/1.jpg": b"j1", "R/Train Set/page/1.xml": page("1.jpg", 3),
    "R/Train Set/2.jpg": b"j2", "R/Train Set/page/2.xml": page("2.jpg", 4),
    "R/Validation Set/9.jpg": b"j9", "R/Validation Set/page/9.xml": page("9.jpg", 5),
    "__MACOSX/R/Train Set/._1.jpg": b"shadow", "R/Train Set/._2.jpg": b"shadow",
    "../evil.xml": page(), "R/../../evil2.xml": page(),
    "R/mets.xml": METS, "R/Train Set/alto/1.xml": ALTO,
    "R/Train Set/page/empty.xml": page(lines=0), "R/Train Set/page/blank.xml": page(text=False),
}


def _reichenau():
    src = record(splits=(("/Train Set/", "train"), ("/Validation Set/", "validation")))
    return src, select_pages(src, {"a.zip": zip_of(REICHENAU)})


def test_the_source_split_is_taken_over_and_counted():
    _, sel = _reichenau()
    assert sel.by_split() == {"train": (2, 7), "validation": (1, 5)}


def test_shadows_and_escaping_paths_are_dropped_and_counted():
    _, sel = _reichenau()
    assert (sel.dropped_shadow, sel.dropped_unsafe) == (2, 2)
    assert sel.dropped_alto == 1 and sel.dropped_no_lines == 1 and sel.dropped_no_text == 1
    assert is_shadow("__MACOSX/x") and is_shadow("a/._b") and not is_shadow("a/b")
    assert is_unsafe("../x") and is_unsafe("/etc/x") and is_unsafe("C:\\x") and not is_unsafe("a/b")


def test_the_image_one_level_above_page_is_found():
    _, sel = _reichenau()
    assert {p.stem: p.image for p in sel.pages} == {
        "1": ("a.zip", "R/Train Set/1.jpg"), "2": ("a.zip", "R/Train Set/2.jpg"),
        "9": ("a.zip", "R/Validation Set/9.jpg")}


def test_the_same_stem_in_train_and_validation_is_two_images():
    """Transkribus nummeriert je Set; ``1.jpg`` kann in beiden liegen. Nur die
    Regel "eine Ebene über ``page/``" trennt sie — ohne sie wäre es mehrdeutig."""
    src = record(splits=(("/Train Set/", "train"), ("/Validation Set/", "validation")))
    sel = select_pages(src, {"a.zip": zip_of({
        "R/Train Set/1.jpg": b"t", "R/Train Set/page/1.xml": page("1.jpg"),
        "R/Validation Set/1.jpg": b"v", "R/Validation Set/page/1.xml": page("1.jpg")})})
    assert {p.split: p.image[1] for p in sel.pages} == {
        "train": "R/Train Set/1.jpg", "validation": "R/Validation Set/1.jpg"}
    assert sel.ambiguous_image == 0


def test_images_in_their_own_archive_are_found():
    src = record(parts=(ZenodoPart("alle", ("gt.zip", "images.zip")),))
    sel = select_pages(src, {"gt.zip": zip_of({"gt/a.xml": page("a.jpg")}),
                             "images.zip": zip_of({"images/a.jpg": b"j"})})
    assert sel.pages[0].image == ("images.zip", "images/a.jpg")


def test_an_ambiguous_image_is_counted_not_guessed():
    src = record()
    sel = select_pages(src, {"a.zip": zip_of({
        "gt/a.xml": page("a.jpg"), "x/a.jpg": b"1", "y/a.jpg": b"2"})})
    assert sel.pages[0].image is None and sel.ambiguous_image == 1 and sel.no_image == 0


def test_a_page_outside_every_declared_split_is_refused():
    src = record(splits=(("/Train Set/", "train"),))
    with pytest.raises(ForeignGtError, match="none of the declared splits"):
        select_pages(src, {"a.zip": zip_of({"R/Other/page/1.xml": page()})})


def test_a_deposit_without_images_has_none_to_miss():
    src = record(has_images=False)
    sel = select_pages(src, {"a.zip": zip_of({"Page/1.xml": page(), "Alto/1.xml": ALTO})})
    assert len(sel.pages) == 1 and sel.no_image == 0 and sel.dropped_alto == 1


# ── Teilsammlungen und Zwillinge ─────────────────────────────────────────────

def test_reichenaus_latin_part_is_not_fetched_by_default():
    r = zenodo_source_by_id("reichenau-inkunabeln")
    assert [p.name for p in r.chosen_parts()] == ["deutsch", "lateinisch-deutsch"]
    assert [p.name for p in r.chosen_parts(everything=True)][-1] == "lateinisch"
    assert [p.name for p in r.chosen_parts(["lateinisch"])] == ["lateinisch"]
    with pytest.raises(ForeignGtError, match="known: deutsch"):
        r.chosen_parts(["griechisch"])


def test_the_two_regesta_deposits_are_marked_as_one_page_set():
    g, d = zenodo_source_by_id("regesta-graphemic"), zenodo_source_by_id("regesta-diplomatic")
    assert (g.record, d.record) == ("18441031", "18377766")
    assert g.same_pages_as == d.id and d.same_pages_as == g.id


def test_an_undeclared_twin_is_noticed():
    g = zenodo_source_by_id("regesta-graphemic")
    unmarked = record(id="g2", record=g.record, doi=g.doi)
    rel = [{"relation": "isVersionOf", "identifier": "10.5281/zenodo.18377766"}]
    assert undeclared_twins(unmarked, meta(related=rel), ZENODO_SOURCES) == ["regesta-diplomatic"]
    assert undeclared_twins(g, meta(related=rel), ZENODO_SOURCES) == []


def test_targets_are_unique_across_both_registers():
    targets = [s.target for s in SOURCES] + [z.target for z in ZENODO_SOURCES]
    assert len(targets) == len(set(targets))
    assert all(t.startswith("dh-unibe/") for t in targets)


# ── die Karte ────────────────────────────────────────────────────────────────

def test_the_card_names_the_version_the_split_the_twin_and_what_was_left_out():
    src, sel = _reichenau()
    src = record(splits=src.splits, same_pages_as="y",
                 parts=(ZenodoPart("de", ("a.zip",)),
                        ZenodoPart("lat", ("b.zip",), default=False, note="Latein")))
    card = zenodo_card(src, sel, with_images=3, parts=src.chosen_parts())
    assert "[10.5281/zenodo.111](https://doi.org/10.5281/zenodo.111)" in card
    assert "| `train` | 2 | 7 |" in card and "| `validation` | 1 | 5 |" in card
    assert "Derselbe Seitensatz wie `y`" in card
    assert "Nicht eingelesen: `lat` — Latein." in card
    assert "2 `__MACOSX`-Schatten, 2\nEinträge mit `..`" in card


# ── das Skript, ohne Netz ────────────────────────────────────────────────────

def _script():
    spec = importlib.util.spec_from_file_location(
        "foreign_gt_to_hf", ROOT / "scripts" / "foreign_gt_to_hf.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["foreign_gt_to_hf"] = module
    spec.loader.exec_module(module)
    return module


def _args(tmp_path, **kw):
    base = dict(part=[], all_parts=False, workdir=tmp_path, no_images=False,
                keep_imageless=False, limit=None, dry_run=True, out=None, public=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _fetch_from(entries_by_key):
    def fetch(url, dest, checksum):
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w") as zf:
            for name, data in entries_by_key[dest.name].items():
                zf.writestr(name, data)
        return dest
    return fetch


def test_the_script_dry_runs_a_deposit_and_refuses_a_contradiction(tmp_path, capsys):
    mod = _script()
    src = record(parts=(ZenodoPart("alle", ("gt.zip", "images.zip")),))
    m = meta(files=("gt.zip", "images.zip", "LICENSE.md"))
    entries = {"gt.zip": {"gt/a.xml": page("a.jpg", 3)}, "images.zip": {"images/a.jpg": b"j"}}
    ok = mod.zenodo_main(src, _args(tmp_path), get_json=lambda url: m,
                         get=lambda url: b"Attribution 4.0 International",
                         fetch=_fetch_from(entries))
    out = capsys.readouterr().out
    assert ok == 0 and "1 Seiten, 3 Zeilen" in out and "LICENSE.md (Datei des Datensatzes)" in out

    refused = mod.zenodo_main(src, _args(tmp_path), get_json=lambda url: m,
                              get=lambda url: b"Attribution-NonCommercial 4.0 International",
                              fetch=_fetch_from(entries))
    assert refused == 7 and "contradicts itself" in capsys.readouterr().err


def test_the_script_wants_consent_for_a_deposit_without_images(tmp_path, capsys):
    mod = _script()
    src = record(has_images=False)
    entries = {"a.zip": {"Page/1.xml": page()}}
    kw = dict(get_json=lambda url: meta(), get=lambda url: b"", fetch=_fetch_from(entries))
    assert mod.zenodo_main(src, _args(tmp_path), **kw) == 5
    assert mod.zenodo_main(src, _args(tmp_path, no_images=True), **kw) == 0

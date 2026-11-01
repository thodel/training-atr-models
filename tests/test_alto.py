"""ALTO lesen (#197): an echten Dateien aus dach-gt und gt-fraktur.

ALTO hält seinen Text in ``String/@CONTENT``, unser Parser sucht ``Unicode``. Eine
ALTO-Datei, die unumgewandelt durchkäme, läse sich als Seite ohne Transkription.
Diese Tests halten fest:

* die Umwandlung trifft an gt-fraktur, wo dieselbe Seite als PAGE vorliegt, die
  PAGE-Fassung Zeile für Zeile — bis auf eine Zeile, in der die beiden Fassungen
  der Quelle selbst verschieden sind;
* Wortfugen kommen aus ``SP``, nicht aus Koordinaten; zwei ``SP`` sind zwei
  Leerzeichen; ``HYP`` bleibt stehen;
* Polygone und Grundlinien kommen mit, leere Zeilen werden gezählt und verworfen;
* der Importweg liest ALTO nur dort, wo die Quelle es verlangt: in dach-gt DE-4,
  nicht DE-12 (``alto/ocr`` ist OCR, ``alto/gt`` hat dieselben Seiten wie
  ``GT-PAGE``); ``GT-PAGE`` gewinnt gegen dieselbe Seite als ALTO;
* gt-fraktur bekommt dadurch keine Seite mehr.
"""

from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from atr_training import pagexml as px
from atr_training.alto import AltoError, alto_to_pagexml, convert, line_text
from atr_training.foreign_gt import (
    deduplicate,
    image_urls,
    page_xml,
    source_by_id,
    tracked_pages,
)

FIX = Path(__file__).parent / "fixtures" / "alto"
ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"


def _read(name: str) -> bytes:
    return (FIX / name).read_bytes()


# ── an echten Dateien ────────────────────────────────────────────────────────

def test_a_dach_gt_page_becomes_pagexml_the_training_path_reads():
    page = convert(_read("dach-de4-alto-image000.xml"))
    boxes = px.line_boxes(page.pagexml)
    assert (page.lines, page.dropped_empty) == (23, 1)
    assert px.image_filename(page.pagexml) == "image000.jpg"
    assert boxes[0].text == "RATISBONA MONASTICA."
    assert boxes[1].text == "Cloͤſterliches Regenſpurg."
    assert (boxes[0].left, boxes[0].top, boxes[0].right, boxes[0].bottom) == (366, 240, 1735, 407)
    assert 'imageWidth="2111" imageHeight="2572"' in page.pagexml


def test_the_polygon_and_the_baseline_come_along():
    root = ET.fromstring(alto_to_pagexml(_read("dach-de4-alto-image000.xml")))
    line = next(el for el in root.iter() if el.tag.endswith("TextLine"))
    coords = next(el for el in line if el.tag.endswith("Coords")).get("points")
    base = next(el for el in line if el.tag.endswith("Baseline")).get("points")
    assert len(coords.split()) > 4, "the polygon, not the rectangle"
    assert base == "371,334 1017,362 1736,375"


def test_on_gt_fraktur_the_alto_reads_as_the_page_fassung_of_the_same_page():
    """21 lines each; 20 identical. Line 10 is a difference of the source's two
    fassungen, not of the conversion: the PAGE fassung has the comma."""
    alto = px.line_texts(alto_to_pagexml(_read("fraktur-alto-agtck_00002.xml")))
    page = [t for t in px.line_texts(_read("fraktur-page-agtck_00002.xml").decode()) if t]
    assert len(alto) == len(page) == 21
    assert [i for i, (a, p) in enumerate(zip(alto, page)) if a != p] == [10]
    assert alto[10] == "Bauer, Eiſenſchmid Fritzſche, Genßler,"
    assert page[10] == "Bauer, Eiſenſchmid, Fritzſche, Genßler,"


def test_empty_lines_are_dropped_and_counted():
    page = convert(_read("dach-de4-alto-image025.xml"))
    assert (page.lines, page.dropped_empty) == (17, 2)
    assert all(px.line_texts(page.pagexml))


def test_a_real_line_with_an_sp_keeps_its_gap():
    page = convert(_read("fraktur-alto-artl_00006.xml"))
    assert page.lines == 46 and page.dropped_empty == 1


def test_the_iiif_identifier_is_kept_for_the_image_path():
    xml = alto_to_pagexml(_read("dach-de12-alto-gt.xml"))
    assert 'externalRef="https://api.digitale-sammlungen.de/iiif/image/v2/' in xml


# ── die Wortfugen ────────────────────────────────────────────────────────────

def _line(inner: str) -> ET.Element:
    return ET.fromstring(f'<TextLine xmlns="{ALTO_NS}">{inner}</TextLine>')


def test_the_gap_is_the_sp_not_the_distance():
    with_sp = _line('<String CONTENT="vnd" HPOS="0"/><SP/><String CONTENT="guͦtten" HPOS="400"/>')
    without = _line('<String CONTENT="vnd" HPOS="0"/><String CONTENT="guͦtten" HPOS="400"/>')
    assert line_text(with_sp) == "vnd guͦtten"
    assert line_text(without) == "vndguͦtten", "a split word, joined as the file has it"


def test_two_sp_are_two_spaces_and_hyp_stays():
    assert line_text(_line('<String CONTENT="."/><SP/><SP/><String CONTENT="79"/>')) == ".  79"
    assert line_text(_line('<String CONTENT="Arbei"/><HYP CONTENT="⸗"/>')) == "Arbei⸗"


def test_an_alto2_baseline_is_a_single_y():
    xml = ('<alto xmlns="http://www.loc.gov/standards/alto/ns-v2#"><Layout><Page WIDTH="100" '
           'HEIGHT="50"><PrintSpace><TextBlock HPOS="0" VPOS="0" WIDTH="100" HEIGHT="50">'
           '<TextLine HPOS="10" VPOS="5" WIDTH="80" HEIGHT="20" BASELINE="22">'
           '<String CONTENT="Zeile"/></TextLine></TextBlock></PrintSpace></Page></Layout></alto>')
    out = alto_to_pagexml(xml)
    assert '<Baseline points="10,22 90,22"/>' in out
    assert '<Coords points="10,5 90,5 90,25 10,25"/>' in out


def test_what_is_not_alto_is_refused():
    with pytest.raises(AltoError, match="not ALTO"):
        convert(_read("dach-de4-page-image000.xml"))


# ── im Importweg ─────────────────────────────────────────────────────────────

@pytest.fixture
def dach(tmp_path: Path) -> Path:
    """dach-gt in klein, als echtes Git-Repo, mit den echten Dateien."""
    root = tmp_path / "dach-gt"
    de4 = root / "data/DE-4/urn_nbn_de_hebis_04-eb2010-01689"
    de12 = root / "data/DE-12/urn_nbn_de_bvb_12-bsb00034304-6"
    for d in (de4 / "GT-PAGE", de4 / "alto", de12 / "alto/gt", de12 / "alto/ocr"):
        d.mkdir(parents=True)
    (de4 / "GT-PAGE/image000.xml").write_bytes(_read("dach-de4-page-image000.xml"))
    (de4 / "alto/image000.xml").write_bytes(_read("dach-de4-alto-image000.xml"))
    (de4 / "alto/image025.xml").write_bytes(_read("dach-de4-alto-image025.xml"))
    (de12 / "alto/gt/10_01d8e_default.xml").write_bytes(_read("dach-de12-alto-gt.xml"))
    (de12 / "alto/ocr/0_889bf_default.xml").write_bytes(_read("dach-de12-alto-ocr.xml"))
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


def test_without_the_request_alto_is_counted_and_dropped(dach):
    sel = tracked_pages(dach, "data")
    assert [p.path.rsplit("/", 2)[-2] for p in sel.pages] == ["GT-PAGE"]
    assert sel.dropped_alto == 4 and sel.alto_read == 0


def test_dach_gt_reads_de4_alto_and_never_de12(dach):
    src = source_by_id("dach-gt")
    sel = tracked_pages(dach, src.xml_root, src.alto)
    deduplicate(sel, src.prefer, src.variants)
    kept = {p.path.split("urn_nbn_de_hebis_04-eb2010-01689/")[-1]: p.fmt for p in sel.pages}
    assert kept == {"GT-PAGE/image000.xml": "page", "alto/image025.xml": "alto"}
    assert sel.dropped_duplicate == 1, "image000 as ALTO is the same page as its GT-PAGE"
    assert sel.dropped_alto == 2, "DE-12: OCR and the duplicate gt fassung"
    page_lines = _read("dach-de4-page-image000.xml").count(b"<TextLine")
    assert sel.dropped_empty_lines == 3 and sel.lines == page_lines + 17


def test_an_alto_page_reaches_the_dataset_as_pagexml_with_its_image(dach):
    src = source_by_id("dach-gt")
    sel = tracked_pages(dach, src.xml_root, src.alto)
    deduplicate(sel, src.prefer, src.variants)
    alto_page = next(p for p in sel.pages if p.fmt == "alto")
    assert px.line_texts(page_xml(dach, alto_page))[0]
    urls = image_urls(dach, src, sel.pages)
    assert urls[alto_page.path] == ("https://archiv.ub.uni-marburg.de/eb/2010/0168/jpg/def/"
                                    "image025.jpg")
    page = next(p for p in sel.pages if p.fmt == "page")
    assert urls[page.path].endswith("/jpg/def/image000.jpg"), "the GT-PAGE page gets one too"


def test_gt_fraktur_gets_no_page_more():
    """Its ALTO is the same 208 pages as its PAGE; the source does not ask for it."""
    assert source_by_id("gt-fraktur").alto == ()

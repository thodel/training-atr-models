"""Das geholte Bild muss die Seite sein, die das PAGE beschreibt (#188).

Der Trainingsweg schneidet jede Zeile mit den Koordinaten des Dokuments aus dem
Bild. Ein Bild anderer Grösse — die falsche Seite, oder die richtige in anderem
Massstab — ergibt falsche Ausschnitte, und nichts schlägt fehl. Gemessen am
02.11.2026 an einer Stichprobe jeder git-Quelle mit erreichbarem Bildserver:
jedes Bild hat die Grösse, die das PAGE nennt, auf das Pixel.

Diese Tests halten fest, dass der Importweg das prüft und ein Bild, das nicht
passt, wie einen gescheiterten Abruf behandelt: gemeldet, nicht hochgeladen.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

from atr_training.foreign_gt import image_size_mismatch

ROOT = Path(__file__).resolve().parents[1]

PAGE = ('<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
        '<Metadata/><Page imageFilename="a.tif" imageWidth="{w}" imageHeight="{h}">'
        '<TextLine id="l"><TextEquiv><Unicode>Zeile</Unicode></TextEquiv></TextLine>'
        '</Page></PcGts>')


def _png(w: int, h: int, mode: str = "L") -> bytes:
    out = io.BytesIO()
    Image.new(mode, (w, h)).save(out, format="PNG")
    return out.getvalue()


def test_the_right_size_passes():
    assert image_size_mismatch(_png(30, 50), PAGE.format(w=30, h=50)) is None


def test_another_size_is_named():
    assert image_size_mismatch(_png(30, 50), PAGE.format(w=31, h=50)) == \
        "Bild 30×50, das PAGE sagt 31×50"


def test_the_right_page_at_another_scale_is_still_wrong():
    """Die Koordinaten gelten in Pixeln dieser Grösse; halbe Auflösung schneidet daneben."""
    assert image_size_mismatch(_png(15, 25), PAGE.format(w=30, h=50))


def test_an_error_page_served_with_200_is_not_an_image():
    reason = image_size_mismatch(b"<html>Service unavailable</html>", PAGE.format(w=30, h=50))
    assert reason and reason.startswith("kein lesbares Bild")


def test_a_bitonal_tiff_is_read_like_any_other():
    """Weisthümer Bd. 6 und 7 liefern bitonale TIFF."""
    out = io.BytesIO()
    Image.new("1", (40, 60)).save(out, format="TIFF")
    assert image_size_mismatch(out.getvalue(), PAGE.format(w=40, h=60)) is None


def test_a_prefixed_page_element_is_read():
    xml = ('<pc:PcGts xmlns:pc="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">'
           '<pc:Page imageHeight="50" imageWidth="30" imageFilename="a.jpg"/></pc:PcGts>')
    assert image_size_mismatch(_png(30, 50), xml) is None
    assert image_size_mismatch(_png(50, 30), xml)


def test_a_document_without_a_size_cannot_be_held_against():
    xml = PAGE.replace(' imageWidth="{w}" imageHeight="{h}"', "")
    assert image_size_mismatch(_png(30, 50), xml) is None


def test_a_missing_pillow_stops_instead_of_dropping_every_page(monkeypatch):
    monkeypatch.setitem(sys.modules, "PIL", None)
    with pytest.raises(ImportError):
        image_size_mismatch(_png(30, 50), PAGE.format(w=30, h=50))


# ── im Importweg ─────────────────────────────────────────────────────────────

def _script():
    spec = importlib.util.spec_from_file_location(
        "foreign_gt_to_hf_size", ROOT / "scripts" / "foreign_gt_to_hf.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["foreign_gt_to_hf_size"] = module
    spec.loader.exec_module(module)
    return module


def test_the_uploader_drops_an_image_of_the_wrong_size(tmp_path, monkeypatch, capsys):
    """Charlottenburger, nachgebaut: die Seite sagt 2215×3643, der Server liefert
    etwas anderes. Die Zeile fällt heraus wie ein gescheiterter Abruf, und es wird
    nichts hochgeladen."""
    from tests.test_foreign_gt_charlottenburger import make_clone

    mod = _script()
    root = make_clone(tmp_path)
    monkeypatch.setattr(mod, "clone", lambda source, workdir: root)
    monkeypatch.setattr(mod, "_get", lambda url, timeout=0: _png(100, 100))
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    code = mod.main(["charlottenburger-amtsschrifttum", "--workdir", str(tmp_path / "w"),
                     "--delay", "0"])
    out = capsys.readouterr()
    assert code == 6
    assert out.out.count("falsches Bild — Bild 100×100, das PAGE sagt 2215×3643") == 3
    assert "3 Zeilen ohne geholtes Bild werden ausgelassen" in out.out

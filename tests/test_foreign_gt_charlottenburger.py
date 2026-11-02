"""charlottenburger-amtsschrifttum (#188): die siebte CC0-Quelle, im Register.

Nachgebaut am Aufbau des echten Repos (Stand ``ae046a7``), mit dessen
Bezugsskript und Bildliste wörtlich:

* ``Marksymbols`` und ``Marksymbols_without_tables`` sind dieselben Seiten mit
  denselben Zeilen; die Fassung ohne Tabellen bleibt, die andere ist eine
  gezählte Dublette — und ``Paragraphsigns`` ist keine Variante davon;
* die Bild-URL kommt aus der Liste und dem base64-Block des Skripts, für eine
  ``.tif``- wie für eine ``.png``-Seite als ``default.jpg`` des ZLB-IIIF-Dienstes,
  mit dem Datensatz der Liste (``34015163_1910``), nicht mit dem Seitenstamm.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from atr_training.foreign_gt import deduplicate, image_urls, source_by_id, tracked_pages

PAGE = ('<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
        '<Page imageFilename="{name}" imageWidth="2215" imageHeight="3643">{body}'
        '</Page></PcGts>')
LINE = '<TextLine id="l{i}"><TextEquiv><Unicode>Charlottenburg {i}</Unicode></TextEquiv></TextLine>'
TABLE = '<TableRegion id="t"><TextRegion id="c">{lines}</TextRegion></TableRegion>'

#: Wörtlich aus ``data/download_images.sh`` des Repos.
SCRIPT = ('#!/usr/bin/env bash\n\ncd $(dirname $0)\n\n'
          'urlbase=`echo -n "aHR0cHM6Ly9kaWdpdGFsLnpsYi5kZS92aWV3ZXIvYXBpL3YxL3JlY29yZHMvCg==" '
          '| base64 -d`\n'
          'cat imageurls.list | while read -r line; do image=${line#* }; '
          'ext=$(echo $image|grep -o "...$");  wget -O "$image" '
          '"${urlbase}${line% *}/files/images/${image/png/tif}/full/max/0/default.$ext"; done\n')
#: Drei Zeilen aus ``data/imageurls.list``.
LISTING = ("16318741_1906 Ber_Stand_Charl_1906_034300.tif\n"
           "34015163_1908 00000372.tif\n"
           "34015163_1910 00000029.png\n")


def _lines(n: int) -> str:
    return "".join(LINE.format(i=i) for i in range(n))


def make_clone(tmp_path: Path) -> Path:
    """The repo in small, as a real git repository; also used by the uploader test."""
    root = tmp_path / "charlottenburger-amtsschrifttum"
    data = root / "data"
    for d in ("Marksymbols", "Marksymbols_without_tables", "Paragraphsigns"):
        (data / d / "GT-PAGE").mkdir(parents=True)
    for stem in ("Ber_Stand_Charl_1906_034300", "00000372"):
        (data / "Marksymbols/GT-PAGE" / f"{stem}.xml").write_text(
            PAGE.format(name=f"{stem}.tif", body=TABLE.format(lines=_lines(4))))
        (data / "Marksymbols_without_tables/GT-PAGE" / f"{stem}.xml").write_text(
            PAGE.format(name=f"{stem}.tif", body=_lines(4)))
    (data / "Paragraphsigns/GT-PAGE/00000029.xml").write_text(
        PAGE.format(name="00000029.png", body=_lines(3)))
    (data / "download_images.sh").write_text(SCRIPT)
    (data / "imageurls.list").write_text(LISTING)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.org", "commit",
                    "-q", "-m", "ae046a7"], cwd=root, check=True)
    return root


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    return make_clone(tmp_path)


def _select(root: Path):
    src = source_by_id("charlottenburger-amtsschrifttum")
    sel = tracked_pages(root, src.xml_root, src.alto)
    deduplicate(sel, src.prefer, src.variants)
    return src, sel


def test_the_fassung_without_tables_is_kept_and_the_other_counted(clone):
    _, sel = _select(clone)
    assert [p.path.split("/")[1] for p in sel.pages] == [
        "Marksymbols_without_tables", "Marksymbols_without_tables", "Paragraphsigns"]
    assert sel.dropped_duplicate == 2
    assert sel.lines == 4 + 4 + 3


def test_every_page_gets_the_zlb_jpeg_from_the_list(clone):
    src, sel = _select(clone)
    urls = image_urls(clone, src, sel.pages)
    base = "https://digital.zlb.de/viewer/api/v1/records/"
    assert urls == {
        "data/Marksymbols_without_tables/GT-PAGE/00000372.xml":
            base + "34015163_1908/files/images/00000372.tif/full/max/0/default.jpg",
        "data/Marksymbols_without_tables/GT-PAGE/Ber_Stand_Charl_1906_034300.xml":
            base + "16318741_1906/files/images/Ber_Stand_Charl_1906_034300.tif/full/max/0/"
                   "default.jpg",
        "data/Paragraphsigns/GT-PAGE/00000029.xml":
            base + "34015163_1910/files/images/00000029.tif/full/max/0/default.jpg",
    }


def test_the_card_names_the_stricter_reading_and_the_authors():
    src = source_by_id("charlottenburger-amtsschrifttum")
    assert src.licence == "CC0-1.0" and "cc-by-4.0" in src.licence_at
    assert "Stefan Weil" in src.attribution and "10686500" in src.attribution
    assert src.image_rights == "ungeprüft"

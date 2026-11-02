"""Weisthuemer Bd. 6 und 7: die zehn Seiten aus den ``for``-Schleifen (#188).

``get_images`` holt sie in zwei Schleifen, die der Skriptleser nicht expandiert:
eine falsch expandierte Schleife holte die falsche Seite, und nichts würde es
sagen. Sie kommen stattdessen über eine Vorlage, die nichts rät: der
Schleifenkörper nennt das Archivmitglied genau so wie die Datei, die er schreibt,
und das PAGE nennt diese Datei als ``imageFilename``. Gemessen am 02.11.2026:
alle zehn Bilder haben die Grösse, die ihr PAGE angibt.

Nachgebaut am echten Repo: die Schleifen wörtlich, ein Band mit expliziten
``curl``-Zeilen daneben, damit sichtbar bleibt, dass der Skriptweg vorgeht.
"""

from __future__ import annotations

from pathlib import Path

from atr_training.foreign_gt import PageFile, image_urls, source_by_id

PAGE = ('<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
        '<Page imageFilename="{img}" imageWidth="2984" imageHeight="4920"/></PcGts>')

#: Wörtlich aus ``get_images`` (Bd. 5 gekürzt auf eine Zeile).
SCRIPT = '''cd Weisthuemer_Bd_5
curl -L -o bub_gb_MMcFAAAAQAAJ_0076.png "https://archive.org/download/bub_gb_MMcFAAAAQAAJ/bub_gb_MMcFAAAAQAAJ_images.tar/gb_MMcFAAAAQAAJ_000077.png"
cd ..

cd Weisthuemer_Bd_6
for page in 13 14 15 16 17; do
  curl -L -o weisthmer02drongoog_00$page.tif "https://archive.org/download/weisthmer02drongoog/weisthmer02drongoog_tif.zip/weisthmer02drongoog_tif%2Fweisthmer02drongoog_00$page.tif"
done
cd ..

cd Weisthuemer_Bd_7
for page in 12 13 14 15 16; do
  curl -L -o weisthmer09maurgoog_00$page.tif "https://archive.org/download/weisthmer09maurgoog/weisthmer09maurgoog_tif.zip/weisthmer09maurgoog_tif%2Fweisthmer09maurgoog_00$page.tif"
done
cd ..
'''


def _repo(tmp_path: Path) -> tuple[Path, list[PageFile]]:
    root = tmp_path / "Weisthuemer"
    (root).mkdir()
    (root / "get_images").write_text(SCRIPT)
    pages = []
    for band, stem, img in (
            (5, "bub_gb_MMcFAAAAQAAJ_0076", "bub_gb_MMcFAAAAQAAJ_0076.png"),
            *((6, f"weisthmer02drongoog_00{n}", f"weisthmer02drongoog_00{n}.tif")
              for n in range(13, 18)),
            *((7, f"weisthmer09maurgoog_00{n}", f"weisthmer09maurgoog_00{n}.tif")
              for n in range(12, 17))):
        path = f"Transcription/Weisthuemer_Bd_{band}/{stem}.xml"
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(PAGE.format(img=img))
        pages.append(PageFile(path, stem, 1, True))
    return root, pages


def test_all_ten_loop_pages_get_the_member_their_page_names(tmp_path):
    root, pages = _repo(tmp_path)
    urls = image_urls(root, source_by_id("Weisthuemer"), pages)
    assert len(urls) == 11
    assert urls["Transcription/Weisthuemer_Bd_6/weisthmer02drongoog_0013.xml"] == (
        "https://archive.org/download/weisthmer02drongoog/weisthmer02drongoog_tif.zip/"
        "weisthmer02drongoog_tif%2Fweisthmer02drongoog_0013.tif")
    assert urls["Transcription/Weisthuemer_Bd_7/weisthmer09maurgoog_0016.xml"] == (
        "https://archive.org/download/weisthmer09maurgoog/weisthmer09maurgoog_tif.zip/"
        "weisthmer09maurgoog_tif%2Fweisthmer09maurgoog_0016.tif")


def test_each_loop_url_is_the_one_the_loop_would_fetch(tmp_path):
    """Die Schleife, für jede ihrer Zahlen von Hand ausgeschrieben, ergibt
    dieselben URLs wie die Vorlage — der Test expandiert, der Importweg nicht."""
    root, pages = _repo(tmp_path)
    urls = image_urls(root, source_by_id("Weisthuemer"), pages)
    for item, numbers in (("weisthmer02drongoog", range(13, 18)),
                          ("weisthmer09maurgoog", range(12, 17))):
        for n in numbers:
            band = 6 if "drongoog" in item else 7
            want = (f"https://archive.org/download/{item}/{item}_tif.zip/"
                    f"{item}_tif%2F{item}_00{n}.tif")
            assert urls[f"Transcription/Weisthuemer_Bd_{band}/{item}_00{n}.xml"] == want


def test_the_explicit_lines_still_come_from_the_script(tmp_path):
    """Bd. 5 benennt lokal ``_0076`` und holt ``_000077``: nur das Skript weiss das."""
    root, pages = _repo(tmp_path)
    urls = image_urls(root, source_by_id("Weisthuemer"), pages)
    assert urls["Transcription/Weisthuemer_Bd_5/bub_gb_MMcFAAAAQAAJ_0076.xml"].endswith(
        "bub_gb_MMcFAAAAQAAJ_images.tar/gb_MMcFAAAAQAAJ_000077.png")

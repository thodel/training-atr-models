"""Was beim Einlesen fremder Ground Truth verworfen wird, und warum (#187).

Diese Tests existieren für zwei stumme Fallen, die beide die Zeilenzahl nach oben
fälschen und beide an echten Quellen gemessen sind (08.10.2026):

**Symlinks.** ``dach-gt`` hält 173 Einträge unter ``data/DE-12/.../alto/``, die auf
ein nie eingechecktes ``gt/`` zeigen. Wer nach ``*.xml`` filtert statt nach
Git-Modus, liest 173 einzeilige Textdateien als Ground Truth ein — der Pfad endet
auf ``.xml``, der Inhalt ist der Linkziel-Pfad.

**Dubletten.** ``reichsanzeiger-gt`` liefert dieselben 101 Seiten zweimal, unter
``…-1939/GT-PAGE`` und ``…-1939_with-TableRegion/GT-PAGE``. Die beiden Fassungen
sind **nicht** byte-identisch — die Tabellenregionen unterscheiden sich — also
fängt kein Hash-Vergleich sie. Ungefiltert gezählt ergibt die Quelle 238.862
Zeilen statt 119.431, und 238.862 ist genau das Doppelte: der Beweis, dass es
Dubletten sind und kein Fund.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from atr_training.foreign_gt import (
    ForeignGtError,
    ImageShellScript,
    ImageUrlList,
    PageFile,
    SOURCES,
    Selection,
    Source,
    dataset_card,
    deduplicate,
    image_urls,
    source_by_id,
    stabilise_archive_org,
    tracked_pages,
)

PAGE = ('<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">'
        '<Page imageFilename="{name}.jpg">'
        '{lines}'
        '</Page></PcGts>')
LINE = '<TextLine id="l{i}"><TextEquiv><Unicode>Zeile {i}</Unicode></TextEquiv></TextLine>'
ALTO = ('<?xml version="1.0"?>\n'
        '<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">'
        '<Layout><TextLine><String CONTENT="x"/></TextLine></Layout></alto>')


def page_xml(name: str, lines: int = 3, text: bool = True) -> str:
    body = "".join(LINE.format(i=i) for i in range(lines)) if text else \
        "".join(f'<TextLine id="l{i}"/>' for i in range(lines))
    return PAGE.format(name=name, lines=body)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Ein echtes Git-Repo mit einem echten Symlink.

    Nachgebaut statt nachgeahmt: der Modus 120000 entsteht nur, wenn git wirklich
    einen Symlink indexiert, und genau diesen Modus prüft der Filter.
    """
    root = tmp_path / "src"
    (root / "data" / "GT-PAGE").mkdir(parents=True)
    (root / "data" / "alto").mkdir(parents=True)
    for i in (1, 2, 3):
        (root / "data" / "GT-PAGE" / f"p{i}.xml").write_text(page_xml(f"p{i}", lines=i + 1))
    (root / "data" / "alto" / "p1.xml").write_text(ALTO)
    (root / "data" / "GT-PAGE" / "empty.xml").write_text(page_xml("empty", lines=0))
    (root / "data" / "GT-PAGE" / "notext.xml").write_text(
        page_xml("notext", lines=2, text=False))
    (root / "data" / "alto" / "broken.xml").symlink_to("../gt/broken.xml")
    (root / "outside.xml").write_text(page_xml("outside"))

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


class TestTheFilter:
    def test_a_symlink_is_never_read_as_a_document(self, repo: Path):
        """Die dach-gt-Falle. Der Pfad endet auf .xml, der Inhalt ist ein Pfad."""
        sel = tracked_pages(repo, "data")
        assert sel.dropped_symlink == 1
        assert not any(p.path.endswith("broken.xml") for p in sel.pages)

    def test_the_mode_is_what_decides_not_the_extension(self, repo: Path):
        """Belegt, dass der Filter wirklich am Modus hängt: das Linkziel existiert
        nicht, ein Lesen nach Endung würde entweder scheitern oder Unsinn liefern."""
        listing = subprocess.run(["git", "ls-files", "-s"], cwd=repo,
                                 capture_output=True, text=True, check=True).stdout
        modes = {line.split("\t")[1]: line.split()[0] for line in listing.splitlines()}
        assert modes["data/alto/broken.xml"] == "120000"
        assert not (repo / "data" / "alto" / "broken.xml").exists(), \
            "das Linkziel darf nicht existieren, sonst prüft der Test nichts"

    def test_alto_is_counted_but_not_read_as_page(self, repo: Path):
        """ALTO ist ein eigener Adapter, kein PAGE mit anderem Namensraum.

        Stillschweigend durchzulassen wäre schlimmer als abzulehnen: ALTO hält
        seinen Text in ``String/@CONTENT``, und unser Parser sucht ``Unicode`` —
        die Seite käme als leer durch."""
        sel = tracked_pages(repo, "data")
        assert sel.dropped_alto == 1
        assert not any("alto" in p.path for p in sel.pages)

    def test_a_page_without_lines_is_dropped(self, repo: Path):
        assert tracked_pages(repo, "data").dropped_no_lines == 1

    def test_a_page_without_transcription_is_dropped(self, repo: Path):
        """Koordinaten ohne Text sind Layout-GT, kein Lesematerial."""
        assert tracked_pages(repo, "data").dropped_no_text == 1

    def test_the_subtree_is_honoured(self, repo: Path):
        sel = tracked_pages(repo, "data")
        assert not any(p.path == "outside.xml" for p in sel.pages)
        assert any(p.path == "outside.xml" for p in tracked_pages(repo, "").pages)

    def test_what_survives_is_counted_correctly(self, repo: Path):
        sel = tracked_pages(repo, "data")
        assert [p.stem for p in sel.pages] == ["p1", "p2", "p3"]
        assert sel.lines == 2 + 3 + 4


class TestDeduplicate:
    """Die reichsanzeiger-Falle, am gemessenen Verhältnis nachgebaut."""

    #: Die gemessene Verteilung ist nicht gleichmässig: 101 Seiten tragen 119.431
    #: Zeilen, also 1.182,5 im Schnitt. Hier als 100 x 1.182 + 1 x 1.231 nachgebaut,
    #: damit die Summe die echte ist und nicht eine, die sich schön teilt.
    REAL_PAGES = 101
    REAL_LINES = 119_431

    def _two_variants(self) -> Selection:
        per_page = [1182] * self.REAL_PAGES
        per_page[-1] += self.REAL_LINES - sum(per_page)
        assert sum(per_page) == self.REAL_LINES
        pages = []
        for i, lines in enumerate(per_page):
            pages.append(PageFile(
                f"data/r-1820-1939/GT-PAGE/{i:03d}.xml", f"{i:03d}", lines, True))
            pages.append(PageFile(
                f"data/r-1820-1939_with-TableRegion/GT-PAGE/{i:03d}.xml",
                f"{i:03d}", lines, True))
        return Selection("reichsanzeiger-gt", pages=pages)

    def test_the_double_count_is_exactly_double(self):
        """Warum wir wissen, dass es Dubletten sind: 238.862 = 2 x 119.431."""
        sel = self._two_variants()
        assert sel.lines == 2 * self.REAL_LINES == 238_862
        deduplicate(sel, ("r-1820-1939/GT-PAGE",))
        assert sel.lines == self.REAL_LINES == 119_431
        assert len(sel.pages) == self.REAL_PAGES
        assert sel.dropped_duplicate == self.REAL_PAGES

    def test_prefer_picks_the_named_variant(self):
        sel = self._two_variants()
        deduplicate(sel, ("_with-TableRegion",))
        assert len(sel.pages) == self.REAL_PAGES
        assert all("_with-TableRegion" in p.path for p in sel.pages)

    def test_without_a_preference_the_choice_is_still_deterministic(self):
        """Nicht die Reihenfolge des Dateisystems: zwei Läufe müssen denselben
        Datensatz ergeben, sonst ist keine Zahl reproduzierbar."""
        first = self._two_variants()
        second = Selection("x", pages=list(reversed(self._two_variants().pages)))
        deduplicate(first, ())
        deduplicate(second, ())
        assert [p.path for p in first.pages] == [p.path for p in second.pages]

    def test_a_preference_that_matches_nothing_falls_back(self):
        sel = self._two_variants()
        deduplicate(sel, ("gibt-es-nicht",))
        assert len(sel.pages) == self.REAL_PAGES


class TestTheLicenceIsRequired:
    def _source(self, **kwargs):
        base = dict(id="x", origin="o", clone_url="c", xml_root="r",
                    licence="CC0-1.0", licence_at="LICENSE", attribution="Wer auch immer",
                    image_source="i", script_kind="s", period="p", project="pr",
                    target="t")
        return Source(**{**base, **kwargs})

    def test_a_licence_without_its_place_is_refused(self):
        """Die Lehre aus Teklia/NewsEye-Austrian-line, das eine CC-BY-4.0-Quelle
        auf HF als ``license: mit`` weitergibt: eine Lizenz ohne Fundstelle ist
        genau die Angabe, die so entsteht."""
        with pytest.raises(ForeignGtError, match="licence_at"):
            self._source(licence_at="")

    def test_no_licence_at_all_is_refused(self):
        with pytest.raises(ForeignGtError, match="licence"):
            self._source(licence="")

    def test_image_rights_default_to_unchecked(self):
        """Nie etwas anderes als 'ungeprüft': die Transkriptionslizenz sagt nichts
        über den Scan (#193)."""
        assert self._source().image_rights == "ungeprüft"
        assert all(s.image_rights == "ungeprüft" for s in SOURCES)


class TestImageUrls:
    def _repo(self, tmp_path: Path, script: str, listing: str) -> Path:
        root = tmp_path / "r"
        (root / "data").mkdir(parents=True)
        (root / "data" / "download_images.sh").write_text(script)
        (root / "data" / "imageurls.list").write_text(listing)
        return root

    #: Wörtlich der Block, den UB Mannheim in reichsanzeiger-gt ausliefert.
    B64 = ("aHR0cHM6Ly9kaWdpLmJpYi51bmktbWFubmhlaW0uZGUvcmVpY2hzYW56ZWlnZXIu"
           "ZmNnaT9GSUY9L3JlaWNoc2FuemVpZ2VyL2ZpbG0vCg==")

    def test_the_base_comes_from_the_repos_own_script(self, tmp_path: Path):
        """Nicht hartkodiert: wir entschlüsseln denselben Block, den die Quelle
        selbst benutzt, damit eine Änderung oben nicht an uns vorbeigeht."""
        root = self._repo(tmp_path, f'urlbase=`echo "{self.B64}" | base64 -d`\n',
                          "097-9978/0126.jp2&CVT=jpeg 1870_14_0126.jpg\n")
        source = Source(id="x", origin="o", clone_url="c", xml_root="data",
                        licence="CC0-1.0", licence_at="LICENSE",
                        attribution="UB Mannheim", image_source="UB MA",
                        script_kind="s", period="p", project="pr", target="t",
                        images=ImageUrlList(list_path="data/imageurls.list",
                                            script_path="data/download_images.sh",
                                            base_b64=self.B64))
        urls = image_urls(root, source)
        assert urls == {"1870_14_0126":
                        "https://digi.bib.uni-mannheim.de/reichsanzeiger.fcgi"
                        "?FIF=/reichsanzeiger/film/097-9978/0126.jp2&CVT=jpeg"}

    def test_a_changed_base_fails_loudly(self, tmp_path: Path):
        """Eine Quelle, die ihre Bilder woanders herholt, ist eine Quelle, deren
        Bildrechte neu zu prüfen sind — das darf nicht stumm durchlaufen."""
        root = self._repo(tmp_path, 'urlbase=`echo "aHR0cHM6Ly9ldmlsLmV4YW1wbGUvCg==" '
                                    '| base64 -d`\n', "a/b.jp2 x.jpg\n")
        source = Source(id="x", origin="o", clone_url="c", xml_root="data",
                        licence="CC0-1.0", licence_at="LICENSE",
                        attribution="UB Mannheim", image_source="UB MA",
                        script_kind="s", period="p", project="pr", target="t",
                        images=ImageUrlList(list_path="data/imageurls.list",
                                            script_path="data/download_images.sh",
                                            base_b64=self.B64))
        with pytest.raises(ForeignGtError, match="no longer contains"):
            image_urls(root, source)

    def test_explicit_curl_lines_are_read(self, tmp_path: Path):
        """Weisthuemers Form."""
        root = tmp_path / "w"
        root.mkdir()
        (root / "get_images").write_text(
            'cd Weisthuemer_Bd_3\n'
            'curl -Lo bub_gb_X_0008.png "https://archive.org/download/a/b.tar/c.png"\n'
            'curl -L -o bub_gb_Y_0012.png "https://archive.org/download/d/e.tar/f.png"\n')
        source = Source(id="w", origin="o", clone_url="c", xml_root="T",
                        licence="CC0-1.0", licence_at="LICENSE",
                        attribution="UB Mannheim", image_source="archive.org",
                        script_kind="s", period="p", project="pr", target="t",
                        images=ImageShellScript(script_path="get_images"))
        urls = image_urls(root, source)
        assert urls["bub_gb_Y_0012"] == "https://archive.org/download/d/e.tar/f.png"
        assert len(urls) == 2

    def test_a_loop_body_is_not_guessed_at(self, tmp_path: Path):
        """Zehn der 35 Weisthuemer-Seiten stehen in ``for page in 13 14 15``.

        Eine falsch expandierte Schleife holt die falsche Seite, und nichts würde
        es sagen. Nicht zu matchen heisst 'kein Bild' im Bericht — ein Zustand, den
        man sieht."""
        root = tmp_path / "w"
        root.mkdir()
        (root / "get_images").write_text(
            'for page in 13 14 15; do\n'
            '  curl -L -o w_00$page.tif "https://archive.org/x/w_00$page.tif"\n'
            'done\n')
        source = Source(id="w", origin="o", clone_url="c", xml_root="T",
                        licence="CC0-1.0", licence_at="LICENSE",
                        attribution="UB Mannheim", image_source="archive.org",
                        script_kind="s", period="p", project="pr", target="t",
                        images=ImageShellScript(script_path="get_images"))
        urls = image_urls(root, source)
        assert all("$page" in stem for stem in urls), \
            "eine Schleifenvariable darf nicht als Seitenstamm durchgehen"

    def test_no_image_plan_yields_nothing(self, tmp_path: Path):
        """DTGT hat kein Bezugsskript. Nichts ist das richtige Ergebnis."""
        source = source_by_id("DTGT")
        assert source.images is None
        assert image_urls(tmp_path, source) == {}


class TestTheCard:
    def test_it_names_the_licence_and_where_it_is_written(self):
        source = source_by_id("reichsanzeiger-gt")
        card = dataset_card(source, Selection("x", pages=[PageFile("a", "a", 1182, True)]),
                            with_images=1)
        assert source.licence in card
        assert "LICENSE (CC0 1.0 Universal)" in card
        assert f"license: {source.licence.lower()}" in card

    def test_it_links_back_to_the_original(self):
        """Was der Auftrag ausdrücklich verlangt: ein Verweis auf das Ursprungs-Repo."""
        for source in SOURCES:
            card = dataset_card(source, Selection("x"), with_images=0)
            assert source.origin in card

    def test_it_says_the_image_rights_are_unchecked(self):
        card = dataset_card(source_by_id("Weisthuemer"), Selection("x"), with_images=0)
        assert "ungeprüft" in card
        assert "privat" in card

    def test_it_reports_what_was_dropped(self):
        sel = Selection("x", dropped_duplicate=101, dropped_symlink=173)
        card = dataset_card(source_by_id("reichsanzeiger-gt"), sel, with_images=0)
        assert "101 Dubletten" in card
        assert "173 Symlinks" in card


class TestTheRegister:
    def test_every_source_is_reachable_by_id(self):
        for source in SOURCES:
            assert source_by_id(source.id) is source

    def test_an_unknown_id_lists_the_known_ones(self):
        with pytest.raises(ForeignGtError, match="reichsanzeiger-gt"):
            source_by_id("gibt-es-nicht")

    def test_every_target_is_under_dh_unibe(self):
        assert all(s.target.startswith("dh-unibe/image-text_") for s in SOURCES)

    def test_no_two_sources_share_a_target(self):
        targets = [s.target for s in SOURCES]
        assert len(targets) == len(set(targets))


# ── archive.org verschiebt Items, die URL im Repo nicht (#187) ────────────────
class TestArchiveOrgPinning:
    """Weisthuemers ``get_images`` nagelt teils einen Knoten fest.

    Gemessen am 08.10.2026: die Form
    ``ia903405.us.archive.org/view_archive.php?archive=/0/items/…`` antwortet
    überhaupt nicht mehr (``http=000`` nach 60 s), weil das Item laut
    ``archive.org/metadata`` inzwischen auf ``ia600607`` liegt. Dieselbe Datei über
    ``archive.org/download/…`` kam in 3,4 s als 324.862-Byte-PNG.
    """

    PINNED = ("https://ia903405.us.archive.org/view_archive.php?archive=/0/items/"
              "bub_gb_2J0ZKYG7on8C/bub_gb_2J0ZKYG7on8C_images.tar"
              "&file=gb_2J0ZKYG7on8C_000009.png")
    STABLE = ("https://archive.org/download/bub_gb_2J0ZKYG7on8C/"
              "bub_gb_2J0ZKYG7on8C_images.tar/gb_2J0ZKYG7on8C_000009.png")

    def test_a_pinned_node_becomes_the_stable_form(self):
        assert stabilise_archive_org(self.PINNED) == self.STABLE

    def test_the_stable_form_is_left_alone(self):
        assert stabilise_archive_org(self.STABLE) == self.STABLE

    def test_an_unrelated_url_is_not_guessed_at(self):
        """Was wir nicht erkennen, lassen wir in Ruhe."""
        other = "https://digi.bib.uni-mannheim.de/x.fcgi?FIF=/y/0126.jp2&CVT=jpeg"
        assert stabilise_archive_org(other) == other

    def test_the_zip_form_works_too(self):
        """Die sechste und siebte Band-Schleife holen aus ``_tif.zip`` statt ``.tar``.

        Der Pfad *innerhalb* des Archivs enthält hier einen Schrägstrich. Das Repo
        schreibt ihn als ``%2F``; wir lassen ihn nackt stehen. Beides ist am
        08.10.2026 gegen archive.org gemessen und liefert dieselbe Datei —
        96.094 Bytes, einmal in 7,7 s und einmal in 1,5 s. Also keine Kodierung
        erzwingen, die nichts bewirkt."""
        pinned = ("https://ia801234.us.archive.org/view_archive.php?archive=/5/items/"
                  "weisthmer02drongoog/weisthmer02drongoog_tif.zip"
                  "&file=weisthmer02drongoog_tif/weisthmer02drongoog_0013.tif")
        out = stabilise_archive_org(pinned)
        assert out == ("https://archive.org/download/weisthmer02drongoog/"
                       "weisthmer02drongoog_tif.zip/"
                       "weisthmer02drongoog_tif/weisthmer02drongoog_0013.tif")

    def test_it_is_applied_when_a_script_is_read(self, tmp_path: Path):
        root = tmp_path / "w"
        root.mkdir()
        (root / "get_images").write_text(
            f'curl -o bub_gb_2J0ZKYG7on8C_0008.png "{self.PINNED}"\n')
        source = Source(id="w", origin="o", clone_url="c", xml_root="T",
                        licence="CC0-1.0", licence_at="LICENSE",
                        attribution="UB Mannheim",
                        image_source="archive.org", script_kind="s", period="p",
                        project="pr", target="t",
                        images=ImageShellScript(script_path="get_images"))
        assert image_urls(root, source) == {"bub_gb_2J0ZKYG7on8C_0008": self.STABLE}


# ── der Rückbezug je Eintrag, nicht nur je Datensatz ─────────────────────────
class TestProvenancePerEntry:
    """Ein Verweis auf das Repo sagt, woher der Datensatz kommt. Er sagt nicht,
    woher *diese Seite* kommt — und ohne Commit zeigt er auf ``main``, wo die
    Datei morgen verschoben sein kann."""

    COMMIT = "0a3a0daf03679dc1d206e17dfdabf62b762b0476"

    def test_an_entry_url_is_pinned_to_the_commit(self):
        source = source_by_id("reichsanzeiger-gt")
        url = source.entry_url("data/reichsanzeiger-1820-1939/GT-PAGE/1820_84_0220.xml",
                               self.COMMIT)
        assert url == ("https://github.com/UB-Mannheim/reichsanzeiger-gt/blob/"
                       f"{self.COMMIT}/data/reichsanzeiger-1820-1939/GT-PAGE/"
                       "1820_84_0220.xml")
        assert "/blob/main/" not in url, "ein Link auf main ist kein Rückbezug"

    def test_the_web_url_drops_the_git_suffix(self):
        assert source_by_id("Weisthuemer").web_url == \
            "https://github.com/UB-Mannheim/Weisthuemer"

    def test_the_card_names_the_commit_and_an_example_entry(self):
        source = source_by_id("reichsanzeiger-gt")
        sel = Selection("x", pages=[PageFile("data/GT-PAGE/a.xml", "a", 1182, True)])
        card = dataset_card(source, sel, with_images=1, commit=self.COMMIT)
        assert self.COMMIT[:12] in card
        assert "source_path" in card and "source_url" in card
        assert f"/blob/{self.COMMIT}/data/GT-PAGE/a.xml" in card

    def test_a_card_without_a_commit_says_so_rather_than_implying_one(self):
        """"nicht festgehalten" ist eine Angabe; ein Link auf main wäre eine Behauptung."""
        card = dataset_card(source_by_id("Weisthuemer"), Selection("x"), with_images=0)
        assert "nicht festgehalten" in card

    def test_every_source_names_its_author(self):
        for source in SOURCES:
            assert source.attribution
            card = dataset_card(source, Selection("x"), with_images=0)
            assert source.attribution in card

    def test_a_source_without_attribution_is_refused(self):
        """Die reichsanzeiger-Karte behauptete "der Urheber genannt" und nannte ihn
        nicht. Eine Ableitung, die ihre Quelle nicht nennt, nennt niemanden."""
        with pytest.raises(ForeignGtError, match="attribution"):
            Source(id="x", origin="o", clone_url="c", xml_root="r", licence="CC0-1.0",
                   licence_at="LICENSE", attribution="", image_source="i",
                   script_kind="s", period="p", project="pr", target="t")


class TestTheBranchIsNotAlwaysMain:
    """Gemessen über ``git ls-remote --symref`` am 08.10.2026: von sieben CC0-Quellen
    liegen drei auf ``master`` — Weisthuemer, Fibeln, gt-fraktur. Mein Register sagte
    für Weisthuemer ``main``, und der erste Lauf bemerkte es nicht, weil der Klon
    schon auf der Platte lag. Ein frischer Klon wäre gescheitert."""

    def test_weisthuemer_is_on_master(self):
        assert source_by_id("Weisthuemer").branch == "master"

    def test_the_others_are_on_main(self):
        assert source_by_id("reichsanzeiger-gt").branch == "main"
        assert source_by_id("DTGT").branch == "main"

    def test_the_card_names_the_branch_it_read(self):
        card = dataset_card(source_by_id("Weisthuemer"), Selection("x"), with_images=0)
        assert "`master`" in card

"""Fremde Ground Truth einlesen: Quellenregister und die Auswahl je Quelle (#187).

Neunzehn fremde Korpora (Epic #186) liegen als GitHub-Repo oder Zenodo-Hinterlegung
vor, keiner als HF-Datensatz. Der Trainingspfad liest aber nur HF-Datensätze. Dieses
Modul ist die Seite davon, die sich ohne Netz testen lässt: welche Dateien einer
geklonten Quelle zählen, welches Bild zu welcher Seite gehört, und was in der
Datensatzkarte stehen muss.

**Warum Dateien verworfen werden, ist hier wichtiger als wie viele.** Zwei der
sieben CC0-Quellen liefern jede Seite zweimal, und eine liefert 173 defekte
Symlinks — beide Fallen schlagen stumm zu und fälschen die Zeilenzahl nach oben:

    reichsanzeiger-gt  101 Seiten in `…-1939/GT-PAGE` **und** in
                       `…-1939_with-TableRegion/GT-PAGE` — nicht byte-identisch
                       (die Tabellenregionen unterscheiden sich), aber dieselben
                       101 Seiten mit denselben 119.431 Zeilen. Ungefiltert
                       gezählt: 238.862, also genau das Doppelte.
    dach-gt            173 Einträge unter `data/DE-12/.../alto/` sind Symlinks auf
                       ein nie eingechecktes `gt/`. Wer nach `*.xml` filtert, liest
                       173 einzeilige Textdateien als Ground Truth.

Darum filtert :func:`tracked_pages` nach **Git-Modus**, nicht nach Endung, und
:func:`deduplicate` nach **Seitenstamm**, nicht nach Dateinamen. Alles gemessen am
08.10.2026 an flachen Klonen der Quellen.
"""

from __future__ import annotations

import base64
import re
import subprocess
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ForeignGtError",
    "ImageInXml",
    "ImagePlan",
    "ImageShellScript",
    "ImageTemplate",
    "ImageUrlList",
    "PageFile",
    "SOURCES",
    "Selection",
    "Source",
    "dataset_card",
    "deduplicate",
    "head_commit",
    "image_urls",
    "source_by_id",
    "stabilise_archive_org",
    "url_in_document",
    "tracked_pages",
]

#: Git's mode for a symlink. The dach-gt trap: 173 of its `*.xml` entries carry it.
SYMLINK_MODE = "120000"

#: A PAGE document announces itself in its root element. ALTO announces `alto/ns-v…`
#: and is a different adapter (#187) — not silently read as if it were PAGE.
PAGE_MARKER = b"pagecontent"
ALTO_MARKER = b"alto/ns-v"

#: How far into a file the marker has to appear. The declaration and root element
#: are in the first few hundred bytes; reading more would only find the words in a
#: comment or a transcription.
HEAD_BYTES = 800

#: Die Stelle, an der OCR-D die Bildherkunft notiert.
_EXTERNAL_REF = re.compile(r'externalRef="([^"]+)"')
#: Eine Bild-URL irgendwo im Dokument, als zweite Wahl nach externalRef.
_IMAGE_URL = re.compile(
    r"(https?://[^\"\s<>]+?(?:\.(?:jpg|jpeg|png|tif|tiff)|/default\.jpg))",
    re.IGNORECASE)

_TEXTLINE = re.compile(rb"<([A-Za-z0-9]+:)?TextLine\b")
_UNICODE = re.compile(rb"<([A-Za-z0-9]+:)?Unicode>")
_B64 = re.compile(r"[A-Za-z0-9+/=\n]{40,}")
#: Weisthuemer schreibt beide Formen, ``curl -Lo NAME URL`` und
#: ``curl -L -o NAME URL``. Ein Regex, der nur die zweite kennt, verliert die
#: Hälfte der Bänder — gemessen: 1 von 2 Zeilen im Test, 10 von 25 im Repo.
#: Ein an einen Knoten genagelter archive.org-Abruf. Weisthuemers ``get_images``
#: schreibt beide Formen: die stabile ``archive.org/download/…`` und diese, die
#: ``ia903405.us.archive.org`` fest verdrahtet. Dort liegt das Item nicht mehr —
#: laut ``archive.org/metadata`` ist es auf ``ia600607`` gewandert — und der alte
#: Knoten antwortet überhaupt nicht (``http=000`` nach 60 s). Gemessen 08.10.2026.
_IA_PINNED = re.compile(
    r"https://ia\d+\.us\.archive\.org/view_archive\.php"
    r"\?archive=/\d+/items/(?P<item>[^/]+)/(?P<archive>[^&]+)&file=(?P<file>.+)$")

_CURL_O = re.compile(
    r"""curl\s+(?:-[A-Za-z]*\s+)*-[A-Za-z]*o\s+(\S+)\s+["\']([^"\']+)["\']"""
)


class ForeignGtError(RuntimeError):
    """A source does not look the way the register says it does."""


@dataclass(frozen=True)
class ImageUrlList:
    """``<remote> <local>`` pairs plus a base URL the repo keeps base64-encoded.

    UB Mannheim ships both the list and a tiny shell script that decodes the base
    and wgets each line. We read the same two files rather than hardcoding a URL,
    and :data:`base_b64` is checked against the script so a change upstream fails
    loudly instead of silently fetching from the wrong host.
    """

    list_path: str
    script_path: str
    base_b64: str
    #: ``{base}`` and ``{remote}`` are filled; ``{stem}``/``{ext}`` come from the
    #: local name, which is what the ZLB form needs.
    template: str = "{base}{remote}"


@dataclass(frozen=True)
class ImageShellScript:
    """A hand-written script of explicit ``curl -o NAME URL`` lines.

    Weisthuemer's form. Loop bodies (``for page in 13 14 15``) are *not* expanded:
    a loop we mis-expand fetches the wrong page and nothing would say so, whereas
    an unmatched line is reported as missing.
    """

    script_path: str


@dataclass(frozen=True)
class ImageInXml:
    """Die Bild-URL steht im PAGE-Dokument selbst.

    Die angenehmste Form, weil sie nichts voraussetzt als das Dokument: OCR-D-
    Werkzeuge schreiben die Herkunft als ``Metadata/@externalRef``. Gemessen am
    08.10.2026 tragen alle 182 DTGT-Dokumente eine, 147 von 162 bei dach-gt und
    41 von 453 bei Fibeln.

    Mein Register hatte DTGT als "kein Bezugsweg" geführt, weil kein *Skript* im
    Repo liegt. Das war richtig und der Schluss daraus falsch — die URL lag die
    ganze Zeit in den Dateien.
    """


@dataclass(frozen=True)
class ImageTemplate:
    """Eine URL-Vorlage je Sammlung, gefüllt aus dem Pfad und dem Dokument.

    Platzhalter: ``{stem}`` der XML-Stamm, ``{base}`` derselbe ohne abschliessendes
    ``_NNN``, ``{top}`` die erste Pfadkomponente unter ``xml_root``, ``{dir}`` das
    unmittelbare Elternverzeichnis, ``{img}``/``{imgbase}``/``{imgext}`` der
    ``imageFilename`` des Dokuments mit und ohne Endung.

    ``only`` begrenzt die Vorlage auf Pfade, die dieses Fragment enthalten — bei
    dach-gt holt nur DE-17 aus Darmstadt, der Rest über :class:`ImageInXml`.
    """

    template: str
    only: str = ""


#: Je Quelle in dieser Reihenfolge versucht; der erste Weg, der eine URL ergibt,
#: gewinnt. Fibeln und dach-gt brauchen das, weil ihr Bezugsweg je
#: Unterverzeichnis verschieden ist.
ImagePlan = ImageUrlList | ImageShellScript | ImageInXml | ImageTemplate


@dataclass(frozen=True)
class Source:
    """One fremde Quelle, with its licence and the place that licence is written.

    ``licence`` and ``licence_at`` are both required and both end up in the dataset
    card. That is the whole point: ``Teklia/NewsEye-Austrian-line`` republishes a
    CC-BY-4.0 source on HF as ``license: mit``, and a licence without its
    provenance is how that happens.
    """

    id: str
    origin: str
    clone_url: str
    xml_root: str
    licence: str
    licence_at: str
    #: Wer die Transkription gemacht hat, und wo die Quelle das sagt. Bei CC0 ist
    #: die Nennung Höflichkeit, bei CC BY Pflicht — und eine Karte, die "der
    #: Urheber genannt" behauptet und ihn dann nicht nennt, ist schlechter als eine,
    #: die schweigt.
    attribution: str
    image_source: str
    script_kind: str
    period: str
    project: str
    target: str
    branch: str = "main"
    images: tuple[ImagePlan, ...] = ()
    #: Path fragments that win when one page stem appears more than once.
    prefer: tuple[str, ...] = ()
    #: Whether the holding institution's terms for the *images* have been checked.
    #: Never defaults to anything but "ungeprüft": the transcription licence says
    #: nothing about the scan (#193).
    image_rights: str = "ungeprüft"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.licence or not self.licence_at:
            raise ForeignGtError(
                f"{self.id}: licence and licence_at are both required — a licence "
                "without the place it is written cannot be put in a dataset card")
        if not self.attribution:
            raise ForeignGtError(
                f"{self.id}: attribution is required — the card says this copy is a "
                "derivation, and a derivation that does not name its source names "
                "nobody")

    @property
    def web_url(self) -> str:
        """Das Repo als Webadresse, aus der Klon-URL."""
        return self.clone_url.removesuffix(".git")

    def entry_url(self, path: str, commit: str) -> str:
        """Der Rückbezug auf **einen** Eintrag, an den gelesenen Stand genagelt.

        Ein Verweis auf das Repo sagt, woher der Datensatz als Ganzes kommt; dieser
        sagt, woher *diese Seite* kommt — und zwar dauerhaft, weil der Commit darin
        steht. Ohne ihn zeigt der Link auf ``main``, und dort kann die Datei morgen
        verschoben sein.
        """
        return f"{self.web_url}/blob/{commit}/{path}"


@dataclass(frozen=True)
class PageFile:
    """A tracked PAGE document with transcribed lines."""

    path: str
    stem: str
    lines: int
    has_text: bool


@dataclass
class Selection:
    """What a source yielded, and everything that was dropped on the way."""

    source_id: str
    pages: list[PageFile] = field(default_factory=list)
    dropped_symlink: int = 0
    dropped_alto: int = 0
    dropped_no_lines: int = 0
    dropped_duplicate: int = 0
    dropped_no_text: int = 0

    @property
    def lines(self) -> int:
        return sum(p.lines for p in self.pages)

    def report(self) -> str:
        return (f"{self.source_id}: {len(self.pages)} Seiten, {self.lines} Zeilen "
                f"(verworfen: {self.dropped_duplicate} Dubletten, "
                f"{self.dropped_symlink} Symlinks, {self.dropped_alto} ALTO, "
                f"{self.dropped_no_lines} ohne Zeilen, {self.dropped_no_text} ohne Text)")


def head_commit(root: Path) -> str:
    """Der Stand, der gelesen wurde. Ein flacher Klon von ``main`` ist sonst undatiert."""
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def _git_ls_files(root: Path) -> list[tuple[str, str]]:
    """``(mode, path)`` for every tracked file. The mode is what the filter needs."""
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-s"],
                         capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        if "\t" not in line:
            continue
        meta, path = line.split("\t", 1)
        rows.append((meta.split()[0], path))
    return rows


def tracked_pages(root: Path, xml_root: str = "") -> Selection:
    """Every tracked PAGE document under ``xml_root`` that carries ``TextLine``s.

    Filters by git mode, so a symlink is never read as a document, and counts the
    reasons rather than only the result.
    """
    sel = Selection(source_id=root.name)
    for mode, rel in _git_ls_files(root):
        if not rel.endswith(".xml"):
            continue
        if xml_root and not rel.startswith(xml_root):
            continue
        if mode == SYMLINK_MODE:
            sel.dropped_symlink += 1
            continue
        try:
            blob = (root / rel).read_bytes()
        except OSError:
            sel.dropped_symlink += 1
            continue
        head = blob[:HEAD_BYTES]
        if ALTO_MARKER in head:
            sel.dropped_alto += 1
            continue
        if PAGE_MARKER not in head:
            continue
        count = len(_TEXTLINE.findall(blob))
        if count == 0:
            sel.dropped_no_lines += 1
            continue
        has_text = bool(_UNICODE.search(blob))
        if not has_text:
            sel.dropped_no_text += 1
            continue
        sel.pages.append(PageFile(path=rel, stem=Path(rel).stem,
                                  lines=count, has_text=has_text))
    sel.pages.sort(key=lambda p: p.path)
    return sel


def deduplicate(sel: Selection, prefer: tuple[str, ...] = ()) -> Selection:
    """Keep one file per page stem.

    ``prefer`` is matched as a substring against the path, in order. Without a
    match the lexicographically first path wins, so the choice is at least
    deterministic rather than filesystem order.
    """
    by_stem: dict[str, list[PageFile]] = {}
    for page in sel.pages:
        by_stem.setdefault(page.stem, []).append(page)

    kept: list[PageFile] = []
    dropped = 0
    for stem in sorted(by_stem):
        candidates = sorted(by_stem[stem], key=lambda p: p.path)
        chosen = candidates[0]
        for fragment in prefer:
            match = next((c for c in candidates if fragment in c.path), None)
            if match is not None:
                chosen = match
                break
        kept.append(chosen)
        dropped += len(candidates) - 1

    sel.pages = sorted(kept, key=lambda p: p.path)
    sel.dropped_duplicate += dropped
    return sel


def stabilise_archive_org(url: str) -> str:
    """Einen an einen Knoten genagelten archive.org-Abruf auf die stabile Form bringen.

    archive.org verschiebt Items zwischen Knoten; eine URL, die ``ia903405`` fest
    verdrahtet, ist deshalb nur so lange gültig, wie das Item dort liegt.
    ``archive.org/download/<item>/<archiv>/<datei>`` leitet dagegen immer auf den
    Knoten um, der das Item gerade hält.

    Alles andere kommt unverändert zurück — eine URL, die wir nicht erkennen, wird
    nicht geraten.
    """
    found = _IA_PINNED.match(url)
    if not found:
        return url
    return ("https://archive.org/download/"
            f"{found['item']}/{found['archive']}/"
            f"{urllib.parse.quote(found['file'])}")


def _decode_base(root: Path, plan: ImageUrlList) -> str:
    """The URL base, read from the repo's own script and checked against the register.

    The script keeps it base64-encoded. We decode the same blob rather than
    hardcoding the host, and refuse if the repo no longer carries the blob we
    recorded — a source that changed where its images come from is a source whose
    image rights have to be looked at again.
    """
    script = (root / plan.script_path).read_text(encoding="utf-8", errors="replace")
    squashed = "".join(plan.base_b64.split())
    if squashed not in "".join(script.split()):
        raise ForeignGtError(
            f"{plan.script_path} no longer contains the recorded image base — "
            "the upstream download script changed; re-check where the images come "
            "from before fetching them")
    found = _B64.search(script)
    if not found:
        raise ForeignGtError(f"{plan.script_path}: no base64 base found")
    return base64.b64decode(found.group(0)).decode("utf-8").strip()


def _from_url_list(root: Path, plan: ImageUrlList) -> dict[str, str]:
    base = _decode_base(root, plan)
    urls: dict[str, str] = {}
    listing = (root / plan.list_path).read_text(encoding="utf-8", errors="replace")
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        remote, local = parts
        stem = Path(local).stem
        urls[stem] = plan.template.format(
            base=base, remote=remote, local=local,
            stem=stem, ext=Path(local).suffix.lstrip("."))
    return urls


def _from_shell_script(root: Path, plan: ImageShellScript) -> dict[str, str]:
    script = (root / plan.script_path).read_text(encoding="utf-8", errors="replace")
    return {Path(name).stem: stabilise_archive_org(url)
            for name, url in _CURL_O.findall(script)}


def url_in_document(xml_text: str) -> str | None:
    """Die Bild-URL, die das PAGE-Dokument selbst nennt.

    Zuerst ``externalRef``, weil das die Stelle ist, die OCR-D dafür vorsieht;
    sonst die erste Bild-URL irgendwo im Dokument. Findet sich keine, ist das
    Ergebnis ``None`` und nicht eine geratene.
    """
    found = _EXTERNAL_REF.search(xml_text)
    if found and _IMAGE_URL.fullmatch(found.group(1)):
        return found.group(1)
    anywhere = _IMAGE_URL.search(xml_text)
    return anywhere.group(1) if anywhere else None


def _needs_document(plan: ImagePlan) -> bool:
    """Ob dieser Weg das Dokument lesen muss.

    :class:`ImageInXml` immer; eine Vorlage nur, wenn sie einen ``{img…}``-
    Platzhalter enthält. Sonst zu lesen kostet bei Fibeln 409 Dateizugriffe für
    nichts — und lässt einen Test über einen Pfad scheitern, den die Vorlage
    gar nicht gebraucht hätte.
    """
    if isinstance(plan, ImageInXml):
        return True
    return isinstance(plan, ImageTemplate) and "{img" in plan.template


def _fill_template(plan: ImageTemplate, page: "PageFile", xml_root: str,
                   xml_text: str) -> str | None:
    if plan.only and plan.only not in page.path:
        return None
    relative = page.path[len(xml_root):].lstrip("/") if xml_root else page.path
    parts = Path(relative).parts
    image_name = ""
    if xml_text:
        try:
            from atr_training.pagexml import image_filename

            image_name = image_filename(xml_text)
        except Exception:  # noqa: BLE001 — ein Dokument ohne imageFilename ist erlaubt
            image_name = ""
    return plan.template.format(
        stem=page.stem,
        base=re.sub(r"_\d+$", "", page.stem),
        top=parts[0] if parts else "",
        dir=Path(page.path).parent.name,
        img=image_name,
        imgbase=Path(image_name).stem,
        imgext=Path(image_name).suffix.lstrip("."),
    )


def image_urls(root: Path, source: Source,
               pages: "Sequence[PageFile]" = ()) -> dict[str, str]:
    """``{Seitenstamm: Bild-URL}`` für eine geklonte Quelle.

    Die Wege aus ``source.images`` werden in ihrer Reihenfolge versucht; der erste,
    der für eine Seite eine URL ergibt, gewinnt. Ein Stamm, der im Ergebnis fehlt,
    hat in der Quelle keine Bild-URL — das wird berichtet, nicht geraten.

    ``pages`` braucht nur, wer :class:`ImageInXml` oder :class:`ImageTemplate`
    benutzt: beide lesen das Dokument selbst.
    """
    urls: dict[str, str] = {}
    needs_pages = any(isinstance(p, (ImageInXml, ImageTemplate))
                      for p in source.images)
    if needs_pages and not pages:
        raise ForeignGtError(
            f"{source.id}: dieser Bezugsweg liest die Dokumente selbst, also "
            "müssen die ausgewählten Seiten übergeben werden")

    for plan in source.images:
        if isinstance(plan, ImageUrlList):
            for stem, url in _from_url_list(root, plan).items():
                urls.setdefault(stem, url)
        elif isinstance(plan, ImageShellScript):
            for stem, url in _from_shell_script(root, plan).items():
                urls.setdefault(stem, url)
        else:
            reads = _needs_document(plan)
            for page in pages:
                if page.stem in urls:
                    continue
                xml_text = ((root / page.path).read_text(
                    encoding="utf-8", errors="replace") if reads else "")
                url = (url_in_document(xml_text) if isinstance(plan, ImageInXml)
                       else _fill_template(plan, page, source.xml_root, xml_text))
                if url:
                    urls[page.stem] = url
    return urls


def dataset_card(source: Source, sel: Selection, *, with_images: int,
                 commit: str = "") -> str:
    """The card. Licence, where it is written, and what was dropped.

    Frontmatter stays minimal on purpose: ``license`` is the SPDX string the source
    actually carries, never a convenient one.
    """
    notes = "\n".join(f"- {n}" for n in source.notes)
    pinned = (f"[`{commit[:12]}`]({source.web_url}/tree/{commit})" if commit
              else "**nicht festgehalten**")
    example = (f"`{source.entry_url(sel.pages[0].path, commit)}`"
               if sel.pages and commit else "—")
    return f"""---
license: {source.licence.lower()}
language:
- de
tags:
- htr
- ocr
- ground-truth
- pagexml
---

# {source.id}

Fremde Ground Truth, eingelesen für das Training deutscher ATR-Modelle.
Diese Kopie ist eine **Ableitung**, nicht das Original — die Quelle ist:

**{source.origin}**

Gelesener Stand: {pinned} (Branch `{source.branch}`).

## Herkunft und Lizenz

| | |
|---|---|
| Urheber der Transkriptionen | {source.attribution} |
| Transkriptionen | **{source.licence}** |
| Fundstelle der Lizenz | {source.licence_at} |
| Bilder | Quelle: {source.image_source} |
| Bildrechte | **{source.image_rights}** |

**Jede Zeile nennt ihren Ursprung selbst.** Die Spalten `source_path` und
`source_url` zeigen auf die Datei, aus der die Seite gelesen wurde — `source_url`
an den Commit oben genagelt, damit der Link gilt, auch wenn die Datei im
Ursprungs-Repo später verschoben wird. Beispiel:

{example}

> **Zu den Bildern.** Die Lizenz der Transkriptionen sagt nichts über die Rechte
> an den Seitenbildern: das sind zwei Fragen mit zwei Rechteinhabern. Die
> Bildrechte sind hier **{source.image_rights}** — für die Weitergabe der Spalte
> `image` liegt also **keine** geprüfte Grundlage vor, und wer sie weiterverwendet,
> muss die Bedingungen bei {source.image_source} selbst klären. Die
> Transkriptionen in `xml_content` sind davon nicht betroffen; für sie gilt die
> Lizenz oben.

## Inhalt

| | |
|---|---|
| Seiten | {len(sel.pages)} |
| Zeilen | {sel.lines} |
| Seiten mit Bild | {with_images} |
| Schrift | {source.script_kind} |
| Zeit | {source.period} |
| Format | PAGE-XML in `xml_content`, Seitenbild in `image` |
| Rückbezug je Zeile | `source_path`, `source_url` |

Verworfen beim Einlesen: {sel.dropped_duplicate} Dubletten (gleicher Seitenstamm),
{sel.dropped_symlink} Symlinks, {sel.dropped_alto} ALTO-Dateien (eigener Adapter),
{sel.dropped_no_lines} ohne `TextLine`, {sel.dropped_no_text} ohne Transkription.

{notes}

## Training

```json
{{"hf_repo": "{source.target}", "train_projects": ["{source.project}"],
 "granularity": "line"}}
```
"""


# ── das Register ───────────────────────────────────────────────────────────────
# Alle Zahlen in den Notizen am 08.10.2026 an einem flachen Klon gemessen, nicht
# aus dem README der Quelle übernommen — bei reichsanzeiger-gt weichen vier
# README-Angaben voneinander ab und keine nennt 101.

SOURCES: tuple[Source, ...] = (
    Source(
        id="reichsanzeiger-gt",
        origin="https://github.com/UB-Mannheim/reichsanzeiger-gt",
        clone_url="https://github.com/UB-Mannheim/reichsanzeiger-gt.git",
        xml_root="data/reichsanzeiger-1820-1939/GT-PAGE",
        prefer=("reichsanzeiger-1820-1939/GT-PAGE",),
        images=(ImageUrlList(
            list_path="data/imageurls.list",
            script_path="data/download_images.sh",
            base_b64="aHR0cHM6Ly9kaWdpLmJpYi51bmktbWFubmhlaW0uZGUvcmVpY2hzYW56ZWlnZXIu"
                     "ZmNnaT9GSUY9L3JlaWNoc2FuemVpZ2VyL2ZpbG0vCg==",
        ),),
        licence="CC0-1.0",
        licence_at="LICENSE (CC0 1.0 Universal); METADATA.yml; .zenodo.json (cc-zero); "
                   "GitHub-API spdx_id",
        attribution="Jan Kamlah, Thomas Schmidt, Renat Shigapov, Stefan Weil "
                    "(UB Mannheim) — genannt in `.zenodo.json` des Repos, je mit ORCID",
        image_source="UB Mannheim, digi.bib.uni-mannheim.de",
        script_kind="Druck, Fraktur + Antiqua",
        period="1820–1939",
        project="reichsanzeiger-1820-1939",
        target="dh-unibe/image-text_reichsanzeiger-gt",
        notes=(
            "**101 Seiten, 119.431 Zeilen** — 1.182 Zeilen je Seite, der höchste "
            "Zeilenertrag je Bild aller 24 geprüften Quellen.",
            "Die Quelle liefert dieselben 101 Seiten zweimal: einmal unter "
            "`reichsanzeiger-1820-1939/GT-PAGE` und einmal unter "
            "`…_with-TableRegion/GT-PAGE`. Hier ist die erste Fassung eingelesen; "
            "die zweite unterscheidet sich nur in den Tabellenregionen.",
            "Das README nennt 197 Seiten, `METADATA.yml` 197, sein Beschreibungstext "
            "117, das Repo hält 202 `*.xml`. Gemessen sind es **101** GT-Seiten, und "
            "die Bildliste passt exakt 1:1 darauf.",
            "Der Zenodo-Spiegel 10.5281/zenodo.10144428 trägt `cc-by-4.0`, während "
            "die `LICENSE` des Repos CC0 sagt. Beide erlauben die Weitergabe; hier "
            "ist die strengere Lesart angewandt und der Urheber genannt.",
        ),
    ),
    Source(
        id="Weisthuemer",
        origin="https://github.com/UB-Mannheim/Weisthuemer",
        clone_url="https://github.com/UB-Mannheim/Weisthuemer.git",
        xml_root="Transcription",
        branch="master",
        images=(ImageShellScript(script_path="get_images"),),
        licence="CC0-1.0",
        licence_at="LICENSE (CC0 1.0 Universal); GitHub-API spdx_id",
        attribution="Universitätsbibliothek Mannheim — das Repo nennt keine "
                    "Einzelpersonen; Textgrundlage ist Jacob Grimms *Weisthümer*",
        image_source="archive.org (sieben Bandscans)",
        script_kind="Druck, Antiqua (Ausgabe 1840–1878)",
        period="Edition 1840–1878, Texte mittelhochdeutsch/lateinisch",
        project="weisthuemer",
        target="dh-unibe/image-text_weisthuemer",
        notes=(
            "**35 Seiten, 1.973 Zeilen** — gemessen, und deckungsgleich mit dem README.",
            "Jacob Grimms *Weisthümer*, fünf Seiten je Band über sieben Bände.",
            "`get_images` holt die Scans aus Archiven heraus (`…_images.tar/…`, "
            "`…_tif.zip/…`). Zehn der 35 Seiten stehen in zwei `for`-Schleifen, die "
            "dieser Importweg **nicht** expandiert: eine falsch expandierte Schleife "
            "holt die falsche Seite, ohne dass es auffiele. Sie fehlen als "
            "'kein Bild' im Bericht.",
            "archive.org leitet jeden Abruf auf einen wechselnden "
            "`ia*.us.archive.org`-Knoten um, und ein toter Knoten antwortet nicht. "
            "Mit kurzem Zeitlimit und Wiederholung ist das unauffällig (3,5–4,6 s "
            "je Bild, gemessen); mit langem Zeitlimit kostet ein einziger solcher "
            "Knoten acht Minuten.",
        ),
    ),
    Source(
        id="DTGT",
        origin="https://github.com/tboenig/DTGT",
        clone_url="https://github.com/tboenig/DTGT.git",
        xml_root="data",
        images=(ImageInXml(),),
        licence="CC0-1.0",
        licence_at="LICENSE (CC0 1.0 Universal); METADATA.yml (`license: - name: CC0 1.0`); "
                   "README-Metadatentabelle; GitHub-API spdx_id",
        attribution="Martin Faßnacht, Stefan Weil (UB Tübingen / Theologie digital) "
                    "— genannt in `METADATA.yml` und im README",
        image_source="UB Tübingen, idb.ub.uni-tuebingen.de/opendigi",
        script_kind="Druck, Fraktur",
        period="1860–1872, plus ein Stück des 17. Jahrhunderts",
        project="dtgt",
        target="dh-unibe/image-text_dtgt",
        notes=(
            "**177 Seiten, 6.599 Zeilen** — die Zeilenzahl deckt sich exakt mit dem "
            "README; dort stehen 182 Dateien, fünf davon ohne `TextLine`.",
            "`METADATA.yml` nennt `count: 6182`, das ist veraltet.",
            "`METADATA.yml` setzt `notBefore: 1860 / notAfter: 1872`. Das ist falsch: "
            "*Gründtlicher Bericht von den zwo roten Neben-Sonnen* ist ein "
            "Fraktur-Einblattdruck des 17. Jahrhunderts (nennt die Schlacht bei "
            "Oldendorp, 1633).",
            "Das Repo enthält kein Bezugs*skript* — die Bild-URL steht aber in "
            "**jedem der 182 Dokumente** als `externalRef`. Dass ich die Quelle "
            "zuerst als 'kein Bezugsweg' geführt habe, war ein Fehlschluss aus dem "
            "fehlenden Skript.",
        ),
    ),
    Source(
        id="gt-fraktur",
        origin="https://github.com/ubtue/gt-fraktur",
        clone_url="https://github.com/ubtue/gt-fraktur.git",
        branch="master",
        xml_root="",
        images=(ImageTemplate(
            template="https://opendigi.ub.uni-tuebingen.de/opendigi/image/"
                     "{base}/{stem}.jp2/full/full/0/default.jpg"),),
        licence="CC0-1.0",
        licence_at="**nur README** §2 ('released by UB, Uni-Tuebingen as Open Data "
                   "under the CC0 public license'); es gibt keine LICENSE-Datei, und "
                   "die GitHub-API meldet `license: null`",
        attribution="Universitätsbibliothek Tübingen (ubtue) — theologische "
                    "Zeitschriften aus dem OpenDigi-Bestand",
        image_source="UB Tübingen, opendigi.ub.uni-tuebingen.de",
        script_kind="Druck, Fraktur",
        period="1830–1875",
        project="gt-fraktur",
        target="dh-unibe/image-text_gt-fraktur",
        notes=(
            "**207 Seiten, 14.617 Zeilen** — gemessen. Das Repo hält 208 PAGE-"
            "Dokumente; eines trägt `TextLine`-Elemente ohne `Unicode`-Text und "
            "fällt heraus.",
            "Dieselben 208 Seiten liegen zusätzlich als ALTO. Hier ist die "
            "PAGE-Fassung eingelesen.",
            "**Die Lizenz steht nur im README.** Inhaltlich genügt uns das, aber "
            "GitHubs Lizenzerkennung sieht nichts. Eine Bitte an die UB Tübingen, "
            "eine `LICENSE`-Datei nachzulegen, kostet nichts und macht die Quelle "
            "maschinell prüfbar.",
            "Das Bezugsskript holt JPEG und speichert sie mit `.tif`-Endung, weil "
            "die PAGE-Dokumente das erwarten. Wir behalten die echte Endung.",
        ),
    ),
    Source(
        id="Fibeln",
        origin="https://github.com/UB-Mannheim/Fibeln",
        clone_url="https://github.com/UB-Mannheim/Fibeln.git",
        branch="master",
        xml_root="",
        images=(
            ImageInXml(),
            ImageTemplate(
                template="https://gei-digital.gei.de/viewer/api/v1/records/{top}/"
                         "files/images/{imgbase}.tif/full/max/0/default.{imgext}"),
        ),
        licence="CC0-1.0",
        licence_at="LICENSE (CC0 1.0 Universal); GitHub-API spdx_id",
        attribution="Universitätsbibliothek Mannheim — Vorlagen aus GEI-Digital "
                    "(Georg-Eckert-Institut) und SUB Göttingen",
        image_source="GEI-Digital Braunschweig und SUB Göttingen",
        script_kind="Druck, Fraktur (Fibeln)",
        period="1782 sowie 19. und frühes 20. Jahrhundert",
        project="fibeln",
        target="dh-unibe/image-text_fibeln",
        notes=(
            "**409 Seiten, 8.895 Zeilen** — gemessen. Das Repo hält 453 PAGE-"
            "Dokumente; 44 tragen keine Transkription.",
            "Zwei Bezugswege, je Unterverzeichnis verschieden: 41 Dokumente "
            "(PPN643815198) nennen ihre Bild-URL selbst und holen aus Göttingen, "
            "die übrigen fünf PPN über eine Vorlage von GEI-Digital.",
            "Der Repo-Titel sagt '19. Jahrhundert'. Das ist schon durch "
            "PPN643815198 widerlegt — *Neue Fibel*, Göttingen **1782** — und die "
            "GEI-Sammlung *Fibeln Kaiserreich* reicht bis 1918.",
            "Zwei der sechs Verzeichnisse (PPN1024784126, PPN1025195825) haben "
            "kein README; 225 der 409 Seiten sind damit nach Werk und Datum "
            "unbestimmt.",
            "Der Klon muss **flach** sein: das Repo trägt 3,83 GB Geschichte, weil "
            "Bilder einmal eingecheckt und später entfernt wurden. Der Arbeitsbaum "
            "hält nur ~24 MB XML.",
        ),
    ),
    Source(
        id="dach-gt",
        origin="https://github.com/UB-Mannheim/dach-gt",
        clone_url="https://github.com/UB-Mannheim/dach-gt.git",
        xml_root="data",
        images=(
            ImageInXml(),
            ImageTemplate(
                only="DE-17",
                template="https://tudigit.ulb.tu-darmstadt.de/image/"
                         "GK-9099-S322-1/3/{stem}.jpg"),
        ),
        licence="CC0-1.0",
        licence_at="LICENSE (CC0 1.0 Universal); GitHub-API spdx_id",
        attribution="Stefan Weil (UB Mannheim) und die sieben haltenden "
                    "Einrichtungen DE-1, DE-4, DE-12, DE-17, DE-23, DE-525, DE-Mh40",
        image_source="sieben Bibliotheken, u.a. Staatsbibliothek zu Berlin und "
                     "ULB Darmstadt, über IIIF und METS",
        script_kind="Druck, Inkunabel- und Fraktur- und Antiqua-Typen",
        period="1486–1913",
        project="dach-gt",
        target="dh-unibe/image-text_dach-gt",
        notes=(
            "**162 Seiten, 4.636 Zeilen** in PAGE-Form — gemessen.",
            "**Nur ein Sechstel der Quelle.** Weitere **864 Seiten liegen als "
            "ALTO** und brauchen einen Adapter, den es noch nicht gibt. Die "
            "Schätzung von ~33.000 Zeilen für die ganze Quelle war im Ergebnis "
            "nicht grob falsch, in der Verteilung aber schon.",
            "**173 Einträge unter `data/DE-12/.../alto/` sind defekte Symlinks** "
            "auf ein nie eingechecktes `gt/`. Dieser Importweg filtert nach "
            "Git-Modus und verwirft sie; wer nach Endung filtert, liest 173 "
            "einzeilige Textdateien als Ground Truth ein.",
            "DE-4 hält 15 Seiten doppelt vor, als PAGE und als ALTO.",
            "Nur 7 der 11 im README genannten Einrichtungen haben Daten; DE-27, "
            "DE-38, DE-46 und DE-61 sind leere Platzhalter.",
            "Die Staatsbibliothek zu Berlin antwortete in der Probe in 9,7 s je "
            "Bild — deutlich langsamer als die anderen Server.",
        ),
    ),
)


def source_by_id(source_id: str) -> Source:
    for source in SOURCES:
        if source.id == source_id:
            return source
    known = ", ".join(s.id for s in SOURCES)
    raise ForeignGtError(f"unknown source {source_id!r}; known: {known}")

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
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ForeignGtError",
    "ImageShellScript",
    "ImageUrlList",
    "PageFile",
    "SOURCES",
    "Selection",
    "Source",
    "dataset_card",
    "deduplicate",
    "image_urls",
    "source_by_id",
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

_TEXTLINE = re.compile(rb"<([A-Za-z0-9]+:)?TextLine\b")
_UNICODE = re.compile(rb"<([A-Za-z0-9]+:)?Unicode>")
_B64 = re.compile(r"[A-Za-z0-9+/=\n]{40,}")
#: Weisthuemer schreibt beide Formen, ``curl -Lo NAME URL`` und
#: ``curl -L -o NAME URL``. Ein Regex, der nur die zweite kennt, verliert die
#: Hälfte der Bänder — gemessen: 1 von 2 Zeilen im Test, 10 von 25 im Repo.
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
    image_source: str
    script_kind: str
    period: str
    project: str
    target: str
    branch: str = "main"
    images: ImageUrlList | ImageShellScript | None = None
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


def image_urls(root: Path, source: Source) -> dict[str, str]:
    """``{page stem: image URL}`` for one cloned source.

    A stem missing from the result has no image URL in the source, which is
    reported rather than guessed at.
    """
    if source.images is None:
        return {}

    if isinstance(source.images, ImageUrlList):
        base = _decode_base(root, source.images)
        urls: dict[str, str] = {}
        listing = (root / source.images.list_path).read_text(
            encoding="utf-8", errors="replace")
        for line in listing.splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            remote, local = parts
            stem = Path(local).stem
            urls[stem] = source.images.template.format(
                base=base, remote=remote, local=local,
                stem=stem, ext=Path(local).suffix.lstrip("."))
        return urls

    script = (root / source.images.script_path).read_text(
        encoding="utf-8", errors="replace")
    return {Path(name).stem: url for name, url in _CURL_O.findall(script)}


def dataset_card(source: Source, sel: Selection, *, with_images: int) -> str:
    """The card. Licence, where it is written, and what was dropped.

    Frontmatter stays minimal on purpose: ``license`` is the SPDX string the source
    actually carries, never a convenient one.
    """
    notes = "\n".join(f"- {n}" for n in source.notes)
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

## Lizenz

| | |
|---|---|
| Transkriptionen | **{source.licence}** |
| Fundstelle | {source.licence_at} |
| Bilder | Quelle: {source.image_source} |
| Bildrechte | **{source.image_rights}** |

Die Lizenz der Transkriptionen sagt nichts über die Rechte an den Seitenbildern:
das sind zwei Fragen mit zwei Rechteinhabern. Solange die Bildrechte
`{source.image_rights}` sind, bleibt dieser Datensatz **privat** (#193).

## Inhalt

| | |
|---|---|
| Seiten | {len(sel.pages)} |
| Zeilen | {sel.lines} |
| Seiten mit Bild | {with_images} |
| Schrift | {source.script_kind} |
| Zeit | {source.period} |
| Format | PAGE-XML in `xml_content`, Seitenbild in `image` |

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
        images=ImageUrlList(
            list_path="data/imageurls.list",
            script_path="data/download_images.sh",
            base_b64="aHR0cHM6Ly9kaWdpLmJpYi51bmktbWFubmhlaW0uZGUvcmVpY2hzYW56ZWlnZXIu"
                     "ZmNnaT9GSUY9L3JlaWNoc2FuemVpZ2VyL2ZpbG0vCg==",
        ),
        licence="CC0-1.0",
        licence_at="LICENSE (CC0 1.0 Universal); METADATA.yml; .zenodo.json (cc-zero); "
                   "GitHub-API spdx_id",
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
        images=ImageShellScript(script_path="get_images"),
        licence="CC0-1.0",
        licence_at="LICENSE (CC0 1.0 Universal); GitHub-API spdx_id",
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
        ),
    ),
    Source(
        id="DTGT",
        origin="https://github.com/tboenig/DTGT",
        clone_url="https://github.com/tboenig/DTGT.git",
        xml_root="data",
        images=None,
        licence="CC0-1.0",
        licence_at="LICENSE (CC0 1.0 Universal); METADATA.yml (`license: - name: CC0 1.0`); "
                   "README-Metadatentabelle; GitHub-API spdx_id",
        image_source="UB Tübingen, idb.ub.uni-tuebingen.de/digitue/theo/ — kein "
                     "Bezugsskript im Repo",
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
            "**Dieses Repo enthält kein Bezugsskript für die Bilder.** Bis der Weg "
            "zu den Scans geklärt ist, trägt der Datensatz nur XML.",
        ),
    ),
)


def source_by_id(source_id: str) -> Source:
    for source in SOURCES:
        if source.id == source_id:
            return source
    known = ", ".join(s.id for s in SOURCES)
    raise ForeignGtError(f"unknown source {source_id!r}; known: {known}")

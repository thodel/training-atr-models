"""Zenodo-Hinterlegungen als Quelle fremder Ground Truth (#198).

Vier der verwendbaren Quellen aus #186 liegen als Zenodo-ZIP, nicht als Git-Repo.
Der Klon-Weg aus #187 (:mod:`atr_training.foreign_gt`) passt auf sie nicht: es
gibt keinen Commit, keinen Git-Modus, und die Lizenz steht in der API statt in
einer Datei. Dieses Modul ist die Seite davon, die sich ohne Netz testen lässt.

**Die versionierte DOI ist Pflicht.** ``10.5281/zenodo.11046062`` ist eine
Version; ``…11046061`` ist der Konzept-DOI und zeigt auf "die neueste" — dasselbe
Problem wie ein Link auf ``main``. Die versionierte gehört in die Karte und in
``source_url``, und ein Register-Eintrag mit dem Konzept-DOI wird verweigert.

**Die Lizenz wird an zwei Stellen gelesen.** ``metadata.license.id`` aus der API
und eine ``LICENSE*``-Datei, im Datensatz oder im Archiv. Widersprechen sie sich,
wird die Quelle abgelehnt, mit beiden Fundstellen in der Meldung — nicht die
bequemere Angabe genommen (#191).

**Die Vorteilung der Quelle ist eine Angabe.** Reichenau und NewsEye liefern
``Train Set`` / ``Validation Set`` getrennt. Wer die Teilung verwirft, kann seine
Zahlen nicht mehr gegen die der Quelle stellen.

**Zwei Fallen, die ein ZIP hat und ein Klon nicht:** Einträge mit ``..`` im Pfad
und die ``__MACOSX``-Schattendateien. Beide werden verworfen und gezählt. Gelesen
wird aus dem Archiv, nie ausgepackt: ein Pfad aus dem ZIP wird nie zu einem Pfad
auf der Platte.

Gemessen am 01.11.2026 an den Inhaltsverzeichnissen der Archive:

    Reichenau  205 + 393 + 1.609 Seiten (Deutsch, Lateinisch-Deutsch, Lateinisch),
               je ``Train Set`` / ``Validation Set``, Bild neben ``page/``
    NewsEye    148 + 13 Seiten in zwei ZIP (Training / Validation), Bild daneben
    Fanny      189 + 61 PAGE, dazu je so viele ALTO, keine Bilder
    Regesta    50 PAGE in ``graphemic-gt.zip``, 50 Bilder in ``images.zip``
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from atr_training.foreign_gt import ALTO_MARKER, HEAD_BYTES, PAGE_MARKER, ForeignGtError

__all__ = [
    "IMAGE_SUFFIXES",
    "LicenceCheck",
    "ZENODO_SOURCES",
    "ZPage",
    "ZenodoPart",
    "ZenodoRecord",
    "ZenodoSelection",
    "check_licences",
    "check_record",
    "is_shadow",
    "is_unsafe",
    "licence_from_text",
    "select_pages",
    "spdx",
    "undeclared_twins",
    "zenodo_card",
    "zenodo_source_by_id",
]

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

_TEXTLINE = re.compile(rb"<([A-Za-z0-9]+:)?TextLine\b")
_UNICODE = re.compile(rb"<([A-Za-z0-9]+:)?Unicode>")
_IMAGE_FILENAME = re.compile(rb'imageFilename="([^"]+)"')

#: Beziehungen in ``related_identifiers``, die "dieselben Seiten" bedeuten können.
TWIN_RELATIONS = ("isVersionOf", "hasVersion", "isVariantFormOf", "hasVariantForm",
                  "isIdenticalTo")


@dataclass(frozen=True)
class ZenodoPart:
    """Eine Teilsammlung, einzeln ziehbar: die Datensatzdateien, die sie ausmachen.

    Reichenaus lateinischer Teil (1.609 von 2.207 Seiten, 1,45 GB) ist für ein
    deutsches Modell Typen-Robustheit, nicht Sprache; er ist darum nicht Vorgabe.
    """

    name: str
    files: tuple[str, ...]
    default: bool = True
    note: str = ""


@dataclass(frozen=True)
class ZenodoRecord:
    """Eine Zenodo-Hinterlegung im Register, mit der versionierten DOI als Pflichtfeld."""

    id: str
    record: str
    doi: str
    licence: str
    licence_at: str
    attribution: str
    image_source: str
    script_kind: str
    period: str
    project: str
    target: str
    parts: tuple[ZenodoPart, ...]
    #: ``(Pfadfragment, Teilung)``, gegen ``<Datei>/<Eintrag>`` geprüft. Leer: die
    #: Quelle teilt nicht vor, und alles ist ``train``.
    splits: tuple[tuple[str, str], ...] = ()
    #: Die Register-Id eines Datensatzes mit **denselben Seiten** (#198):
    #: wer beide einliest, zählt sie doppelt.
    same_pages_as: str = ""
    has_images: bool = True
    image_rights: str = "ungeprüft"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.licence or not self.licence_at:
            raise ForeignGtError(f"{self.id}: licence and licence_at are both required")
        if not self.attribution:
            raise ForeignGtError(f"{self.id}: attribution is required")
        if self.doi != f"10.5281/zenodo.{self.record}":
            raise ForeignGtError(
                f"{self.id}: doi {self.doi!r} is not the versioned DOI of record "
                f"{self.record} — a concept DOI points at 'the newest', like a link to main")
        if not self.parts:
            raise ForeignGtError(f"{self.id}: no parts — nothing to read")

    @property
    def api_url(self) -> str:
        return f"https://zenodo.org/api/records/{self.record}"

    @property
    def record_url(self) -> str:
        return f"https://zenodo.org/records/{self.record}"

    @property
    def doi_url(self) -> str:
        return f"https://doi.org/{self.doi}"

    def file_url(self, key: str) -> str:
        from urllib.parse import quote
        return f"{self.record_url}/files/{quote(key)}"

    def part(self, name: str) -> ZenodoPart:
        for p in self.parts:
            if p.name == name:
                return p
        known = ", ".join(p.name for p in self.parts)
        raise ForeignGtError(f"{self.id}: no part {name!r}; known: {known}")

    def chosen_parts(self, names: Sequence[str] = (), everything: bool = False
                     ) -> tuple[ZenodoPart, ...]:
        if everything:
            return self.parts
        if names:
            return tuple(self.part(n) for n in names)
        return tuple(p for p in self.parts if p.default)


# ── die API und die Lizenz ───────────────────────────────────────────────────

def spdx(zenodo_id: str) -> str:
    """Zenodos Lizenz-Id als SPDX-Schreibweise: ``cc-by-sa-4.0`` → ``CC-BY-SA-4.0``."""
    value = zenodo_id.strip()
    if value.lower() in ("cc-zero", "cc0-1.0", "cc0"):
        return "CC0-1.0"
    return value.upper()


#: Vom Spezifischen zum Allgemeinen: "Attribution 4.0" steckt auch in
#: "Attribution-ShareAlike 4.0" nicht, aber "CC BY" steckt in "CC BY-SA".
_LICENCE_TEXTS: tuple[tuple[str, str], ...] = (
    (r"Attribution-NonCommercial-ShareAlike\s+4\.0", "CC-BY-NC-SA-4.0"),
    (r"Attribution-NonCommercial-NoDerivatives\s+4\.0", "CC-BY-NC-ND-4.0"),
    (r"Attribution-NonCommercial\s+4\.0", "CC-BY-NC-4.0"),
    (r"Attribution-NoDerivatives\s+4\.0", "CC-BY-ND-4.0"),
    (r"Attribution-ShareAlike\s+4\.0", "CC-BY-SA-4.0"),
    (r"Attribution\s+4\.0\s+International", "CC-BY-4.0"),
    (r"\bCC[- ]?BY[- ]SA[- ]4\.0\b", "CC-BY-SA-4.0"),
    (r"\bCC[- ]?BY[- ]4\.0\b", "CC-BY-4.0"),
    (r"\bCC0\b|Public Domain Dedication", "CC0-1.0"),
)


def licence_from_text(text: str) -> str | None:
    """Die Lizenz, die eine ``LICENSE``-Datei nennt, oder ``None`` wenn keine erkannt."""
    for pattern, licence in _LICENCE_TEXTS:
        if re.search(pattern, text, re.IGNORECASE):
            return licence
    return None


@dataclass(frozen=True)
class LicenceCheck:
    licence: str
    found_at: tuple[str, ...]

    @property
    def licence_at(self) -> str:
        return "; ".join(self.found_at)


def check_record(source: ZenodoRecord, meta: Mapping) -> tuple[str, dict[str, str]]:
    """Die API-Antwort gegen das Register: DOI, Konzept-DOI, Lizenz. ``(SPDX, md5 je Datei)``.

    Eine Hinterlegung, deren Lizenz sich gegen das Register geändert hat, ist eine
    Quelle, die neu zu prüfen ist — sie läuft nicht stumm weiter.
    """
    doi = meta.get("doi", "")
    if doi != source.doi:
        raise ForeignGtError(f"{source.id}: the API names doi {doi!r}, the register "
                             f"{source.doi!r}")
    if meta.get("conceptdoi") == source.doi:
        raise ForeignGtError(f"{source.id}: {source.doi} is the concept DOI")
    lic = ((meta.get("metadata") or {}).get("license") or {}).get("id", "")
    if not lic:
        raise ForeignGtError(f"{source.id}: the API names no licence "
                             "(metadata.license.id) — nothing to put in the card")
    found = spdx(lic)
    if found != source.licence:
        raise ForeignGtError(
            f"{source.id}: the API says {found} (metadata.license.id = {lic!r}), the "
            f"register {source.licence} — the deposit changed; read it again before "
            "importing")
    checksums = {f["key"]: f.get("checksum", "") for f in meta.get("files", [])}
    return found, checksums


def check_licences(source: ZenodoRecord, api_licence: str,
                   files: Mapping[str, str]) -> LicenceCheck:
    """Die API-Lizenz gegen jede ``LICENSE*``-Datei. ``files``: ``{Fundstelle: Text}``.

    Widerspruch oder eine Datei, deren Lizenz nicht erkannt wird, ist Ablehnung —
    mit beiden Fundstellen in der Meldung. "Nicht erkannt" heisst nicht "passt".
    """
    found_at = [f"Zenodo-API `metadata.license.id` ({api_licence}) der Version "
                f"{source.doi}"]
    for where, text in sorted(files.items()):
        named = licence_from_text(text)
        if named is None:
            raise ForeignGtError(
                f"{source.id}: {where} names no licence this importer recognises; the "
                f"API says {api_licence}. Read it by hand before importing.")
        if named != api_licence:
            raise ForeignGtError(
                f"{source.id}: the licence contradicts itself — Zenodo-API "
                f"metadata.license.id says {api_licence}, {where} says {named}. "
                "Refused rather than taking the convenient one (#191).")
        found_at.append(f"{where} ({named})")
    return LicenceCheck(api_licence, tuple(found_at))


def undeclared_twins(source: ZenodoRecord, meta: Mapping,
                     register: Sequence[ZenodoRecord]) -> list[str]:
    """Register-Ids, auf die der Datensatz als Version oder Variante zeigt, ohne dass
    das Register sie als denselben Seitensatz führt."""
    related = (meta.get("metadata") or {}).get("related_identifiers") or []
    by_doi = {r.doi: r for r in register}
    out = []
    for rel in related:
        if rel.get("relation") not in TWIN_RELATIONS:
            continue
        other = by_doi.get(str(rel.get("identifier", "")).removeprefix("https://doi.org/"))
        if other and other.id != source.id and source.same_pages_as != other.id:
            out.append(other.id)
    return out


# ── das Archiv ───────────────────────────────────────────────────────────────

def is_shadow(name: str) -> bool:
    """macOS-Schattendateien: ``__MACOSX/…`` und ``._name``."""
    parts = PurePosixPath(name).parts
    return bool(parts) and (parts[0] == "__MACOSX" or parts[-1].startswith("._"))


def is_unsafe(name: str) -> bool:
    """Ein Eintrag, der aus dem Archiv herauszeigt: ``..``, absolut, Laufwerk."""
    if name.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", name):
        return True
    return ".." in re.split(r"[\\/]", name)


@dataclass(frozen=True)
class ZPage:
    """Eine PAGE-Seite in einem Archiv, ihr Bild (``(Datei, Eintrag)``) und ihre Teilung."""

    file: str
    path: str
    stem: str
    lines: int
    split: str
    image: tuple[str, str] | None

    @property
    def source_path(self) -> str:
        return f"{self.file}/{self.path}"


@dataclass
class ZenodoSelection:
    source_id: str
    pages: list[ZPage] = field(default_factory=list)
    dropped_shadow: int = 0
    dropped_unsafe: int = 0
    dropped_alto: int = 0
    dropped_no_lines: int = 0
    dropped_no_text: int = 0
    no_image: int = 0
    ambiguous_image: int = 0

    @property
    def lines(self) -> int:
        return sum(p.lines for p in self.pages)

    def by_split(self) -> dict[str, tuple[int, int]]:
        """``{Teilung: (Seiten, Zeilen)}``."""
        out: dict[str, list[int]] = {}
        for p in self.pages:
            row = out.setdefault(p.split, [0, 0])
            row[0] += 1
            row[1] += p.lines
        return {k: (v[0], v[1]) for k, v in sorted(out.items())}

    def report(self) -> str:
        splits = ", ".join(f"{k} {n} Seiten / {lines} Zeilen"
                           for k, (n, lines) in self.by_split().items())
        return (f"{self.source_id}: {len(self.pages)} Seiten, {self.lines} Zeilen "
                f"[{splits or '—'}] (verworfen: {self.dropped_shadow} __MACOSX-Schatten, "
                f"{self.dropped_unsafe} Pfade mit '..', {self.dropped_alto} ALTO, "
                f"{self.dropped_no_lines} ohne Zeilen, {self.dropped_no_text} ohne Text; "
                f"ohne Bild {self.no_image}, Bild mehrdeutig {self.ambiguous_image})")


def _split_of(source: ZenodoRecord, where: str) -> str:
    if not source.splits:
        return "train"
    for fragment, split in source.splits:
        if fragment in where:
            return split
    raise ForeignGtError(
        f"{source.id}: {where} falls in none of the declared splits "
        f"{[f for f, _ in source.splits]} — the deposit changed shape")


def _pick_image(candidates: list[tuple[str, str]], xml_dir: PurePosixPath
                ) -> tuple[str, str] | None | str:
    """Das Bild zur Seite: im selben Verzeichnis, dann eines höher (``page/``), dann
    das einzige. Sonst ``"ambiguous"`` — geraten wird nicht."""
    if not candidates:
        return None
    for want in (xml_dir, xml_dir.parent):
        here = [c for c in candidates if PurePosixPath(c[1]).parent == want]
        if len(here) == 1:
            return here[0]
    if len(candidates) == 1:
        return candidates[0]
    return "ambiguous"


def select_pages(source: ZenodoRecord, archives: Mapping[str, zipfile.ZipFile]
                 ) -> ZenodoSelection:
    """Jede PAGE-Seite mit ``TextLine``s in ``archives`` (``{Datei: ZipFile}``).

    Bilder werden über alle übergebenen Archive gesucht, weil Regesta seine Bilder
    in einem eigenen ZIP hält; gefunden über ``imageFilename``, sonst den Stamm.
    """
    sel = ZenodoSelection(source_id=source.id)
    images: dict[str, list[tuple[str, str]]] = {}
    documents: list[tuple[str, str]] = []
    for key, zf in sorted(archives.items()):
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            if is_shadow(name):
                sel.dropped_shadow += 1
                continue
            if is_unsafe(name):
                sel.dropped_unsafe += 1
                continue
            lower = name.lower()
            if lower.endswith(IMAGE_SUFFIXES):
                images.setdefault(PurePosixPath(name).stem, []).append((key, name))
            elif lower.endswith(".xml"):
                documents.append((key, name))

    for key, name in documents:
        blob = archives[key].read(name)
        head = blob[:HEAD_BYTES].lower()
        if ALTO_MARKER in head:
            sel.dropped_alto += 1
            continue
        if PAGE_MARKER not in head:
            continue                       # METS und anderes: keine Seite
        count = len(_TEXTLINE.findall(blob))
        if count == 0:
            sel.dropped_no_lines += 1
            continue
        if not _UNICODE.search(blob):
            sel.dropped_no_text += 1
            continue
        path = PurePosixPath(name)
        named = _IMAGE_FILENAME.search(blob)
        wanted = (PurePosixPath(named.group(1).decode("utf-8", "replace")).stem
                  if named else path.stem)
        image = None
        if source.has_images:
            picked = _pick_image(images.get(wanted) or images.get(path.stem) or [],
                                 path.parent)
            if picked == "ambiguous":
                sel.ambiguous_image += 1
            elif picked is None:
                sel.no_image += 1
            else:
                image = picked
        sel.pages.append(ZPage(file=key, path=name, stem=path.stem, lines=count,
                               split=_split_of(source, f"{key}/{name}"), image=image))
    sel.pages.sort(key=lambda p: (p.split, p.file, p.path))
    return sel


# ── die Karte ────────────────────────────────────────────────────────────────

def zenodo_card(source: ZenodoRecord, sel: ZenodoSelection, *, with_images: int,
                parts: Sequence[ZenodoPart], licence: LicenceCheck | None = None) -> str:
    notes = "\n".join(f"- {n}" for n in source.notes)
    twin = (f"\n> **Derselbe Seitensatz wie `{source.same_pages_as}`** — dieselben Bilder "
            "auf anderer Transkriptionsebene. Nicht zusammen trainieren oder zählen: "
            "wer beide einliest, zählt die Seiten doppelt.\n" if source.same_pages_as else "")
    splits = sel.by_split()
    if source.splits:
        split_rows = "\n".join(f"| `{k}` | {n} | {lines} |"
                               for k, (n, lines) in splits.items())
        split_text = ("Die Quelle teilt selbst vor; diese Teilung ist **übernommen**, "
                      "nicht neu gezogen, damit Zahlen gegen die der Quelle stehen können. "
                      "`train` liegt unter `data/train/`, `validation` unter "
                      "`data/validation/`; jede Zeile trägt sie in `source_split`.\n\n"
                      "| Teilung | Seiten | Zeilen |\n|---|---:|---:|\n" + split_rows)
    else:
        split_text = "Die Quelle teilt nicht vor; alle Seiten liegen unter `data/train/`."
    found_at = licence.licence_at if licence else source.licence_at
    part_list = ", ".join(f"`{p.name}` ({', '.join(p.files)})" for p in parts)
    left_out = [p for p in source.parts if p not in parts]
    left = ("Nicht eingelesen: " + ", ".join(f"`{p.name}`" + (f" — {p.note}" if p.note else "")
                                             for p in left_out) + ".") if left_out else ""
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
Diese Kopie ist eine **Ableitung**, nicht das Original — die Quelle ist die
Zenodo-Hinterlegung **[{source.doi}]({source.doi_url})** (diese Version, nicht
"die neueste").
{twin}
## Herkunft und Lizenz

| | |
|---|---|
| Urheber | {source.attribution} |
| Lizenz | **{source.licence}** |
| Fundstelle der Lizenz | {found_at} |
| Bilder | Quelle: {source.image_source} |
| Bildrechte | **{source.image_rights}** |

**Jede Zeile nennt ihren Ursprung selbst.** `source_path` ist `<Datei>/<Eintrag>`
im Archiv, `source_url` die versionierte DOI.

## Inhalt

| | |
|---|---|
| Seiten | {len(sel.pages)} |
| Zeilen | {sel.lines} |
| Seiten mit Bild | {with_images} |
| Teilsammlungen | {part_list} |
| Schrift | {source.script_kind} |
| Zeit | {source.period} |

{left}

{split_text}

Verworfen beim Einlesen: {sel.dropped_shadow} `__MACOSX`-Schatten, {sel.dropped_unsafe}
Einträge mit `..` im Pfad, {sel.dropped_alto} ALTO-Dateien (eigener Adapter),
{sel.dropped_no_lines} ohne `TextLine`, {sel.dropped_no_text} ohne Transkription.

{notes}

## Training

```json
{{"hf_repo": "{source.target}", "train_projects": ["{source.project}"],
 "granularity": "line"}}
```
"""


# ── das Register ─────────────────────────────────────────────────────────────

_SPLITS_SET = (("/Train Set/", "train"), ("/Validation Set/", "validation"))

ZENODO_SOURCES: tuple[ZenodoRecord, ...] = (
    ZenodoRecord(
        id="reichenau-inkunabeln",
        record="11046062",
        doi="10.5281/zenodo.11046062",
        licence="CC-BY-SA-4.0",
        licence_at="Zenodo-API `metadata.license.id` (cc-by-sa-4.0); README_DE.txt "
                   "('Lizenz: CC-BY-SA 4.0')",
        attribution="Badische Landesbibliothek; Katharina Ost, Annika Stello, Gerrit "
                    "Heim — Zenodo `creators`",
        image_source="Badische Landesbibliothek (Bilder im Archiv der Hinterlegung)",
        script_kind="Druck, Inkunabeltypen",
        period="15. Jahrhundert (ehemals Reichenauer Inkunabeln)",
        project="reichenau-inkunabeln",
        target="dh-unibe/image-text_reichenau-inkunabeln",
        parts=(
            ZenodoPart("deutsch", ("Reichenauer_Inkunabeln_Deutsch.zip",)),
            ZenodoPart("lateinisch-deutsch",
                       ("Reichenauer_Inkunabeln_Lateinisch-Deutsch.zip",)),
            ZenodoPart("lateinisch", ("Reichenauer_Inkunabeln_Lateinisch.zip",),
                       default=False,
                       note="1.609 von 2.207 Seiten und 1,45 GB; für ein deutsches "
                            "Modell Typen-Robustheit, nicht Sprache"),
        ),
        splits=_SPLITS_SET,
        notes=(
            "**2.207 Seiten, 125.330 Zeilen**, gemessen am 01.11.2026: Deutsch 205 "
            "Seiten / 9.671 Zeilen, Lateinisch-Deutsch 393 / 18.610, Lateinisch "
            "1.609 / 97.049. Drei ZIP, nicht vier, wie #198 annahm.",
            "Vorteilung der Quelle: Deutsch 185 + 20, Lateinisch-Deutsch 354 + 39, "
            "Lateinisch 1.449 + 160 Seiten (Train + Validation).",
            "Die Bilder liegen im Archiv selbst, hinterlegt von der haltenden "
            "Einrichtung unter derselben Lizenz. Das spricht dafür, die Bildrechte "
            "als geklärt zu führen — entscheiden muss das ein Mensch (#193).",
            "Transkribus-Modelle derselben Daten nennt die README der Hinterlegung.",
        ),
    ),
    ZenodoRecord(
        id="newseye-onb",
        record="3387369",
        doi="10.5281/zenodo.3387369",
        licence="CC-BY-4.0",
        licence_at="Zenodo-API `metadata.license.id` (cc-by-4.0)",
        attribution="Günter Mühlberger, Günter Hackl (NewsEye / READ) — Zenodo "
                    "`creators`; Vorlagen der Österreichischen Nationalbibliothek",
        image_source="Österreichische Nationalbibliothek (Bilder im Archiv der "
                     "Hinterlegung, JPG und TIF)",
        script_kind="Druck, Fraktur und Antiqua (Zeitungen)",
        period="1864–1911 (nach den Dateinamen)",
        project="newseye-onb",
        target="dh-unibe/image-text_newseye-onb",
        parts=(ZenodoPart("alle", ("TrainingSet_ONB_Newseye_GT_M1+.zip",
                                   "ValidationSet_ONB_Newseye_GT_M1+.zip")),),
        splits=(("TrainingSet_", "train"), ("ValidationSet_", "validation")),
        notes=(
            "**161 Seiten, 58.796 Zeilen**, gemessen am 01.11.2026: Training 148 "
            "Seiten / 53.983 Zeilen, Validation 13 / 4.813.",
            "Auf HF liegt dieselbe Quelle als `Teklia/NewsEye-Austrian-line` mit "
            "`license: mit` — die Hinterlegung sagt CC BY 4.0, und das gilt.",
        ),
    ),
    ZenodoRecord(
        id="fanny-mendelssohn",
        record="15223355",
        doi="10.5281/zenodo.15223355",
        licence="CC-BY-4.0",
        licence_at="Zenodo-API `metadata.license.id` (cc-by-4.0)",
        attribution="Roman Kuhn, Roland Schmidt-Hensel (Staatsbibliothek zu Berlin, "
                    "Seminar mit der UdK Berlin 2024) — Zenodo `creators`",
        image_source="Staatsbibliothek zu Berlin — Digitalisate SBB00035B5E00000000 "
                     "und SBB00035B6800000000, **nicht** in der Hinterlegung",
        script_kind="Handschrift (Briefe)",
        period="19. Jahrhundert (Briefe Fanny Mendelssohns an Wilhelm Hensel)",
        project="fanny-mendelssohn",
        target="dh-unibe/image-text_fanny-mendelssohn",
        parts=(ZenodoPart("alle", ("1878144588.zip", "1878121294.zip")),),
        has_images=False,
        notes=(
            "**200 Seiten, 2.238 Zeilen**, gemessen am 01.11.2026. Die Hinterlegung "
            "hält 250 PAGE (189 + 61); 49 davon haben keine `TextLine`, eine keinen "
            "Text. Daneben je dieselben Seiten als ALTO, hier verworfen.",
            "**Keine Bilder.** Ohne Bildspalte trainiert der Datensatz nicht; er "
            "braucht `--no-images`, bis die Digitalisate der SBB geholt werden.",
        ),
    ),
    ZenodoRecord(
        id="regesta-graphemic",
        record="18441031",
        doi="10.5281/zenodo.18441031",
        licence="CC-BY-4.0",
        licence_at="Zenodo-API `metadata.license.id` (cc-by-4.0); LICENSE.md des "
                   "Datensatzes",
        attribution="Adam Juszczak, Frederik Skidzun — Zenodo `creators`",
        image_source="Bayerisches Hauptstaatsarchiv (Bilder im Archiv der "
                     "Hinterlegung, `images.zip`)",
        script_kind="Handschrift, frühneuhochdeutsche Kanzleischrift",
        period="15. Jahrhundert",
        project="regesta-graphemic",
        target="dh-unibe/image-text_regesta-graphemic",
        parts=(ZenodoPart("alle", ("graphemic-gt.zip", "images.zip")),),
        same_pages_as="regesta-diplomatic",
        notes=(
            "**50 Seiten, 2.177 Zeilen**, gemessen am 01.11.2026; graphemische "
            "Transkription.",
            "`images.zip` ist byte-gleich mit dem von 10.5281/zenodo.18377766 "
            "(dieselbe MD5): dieselben 50 Bilder, diplomatisch transkribiert.",
        ),
    ),
    ZenodoRecord(
        id="regesta-diplomatic",
        record="18377766",
        doi="10.5281/zenodo.18377766",
        licence="CC-BY-4.0",
        licence_at="Zenodo-API `metadata.license.id` (cc-by-4.0); LICENSE.md des "
                   "Datensatzes",
        attribution="Adam Juszczak, Frederik Skidzun — Zenodo `creators`",
        image_source="Bayerisches Hauptstaatsarchiv (Bilder im Archiv der "
                     "Hinterlegung, `images.zip`)",
        script_kind="Handschrift, frühneuhochdeutsche Kanzleischrift",
        period="15. Jahrhundert",
        project="regesta-diplomatic",
        target="dh-unibe/image-text_regesta-diplomatic",
        parts=(ZenodoPart("alle", ("diplomatic-gt.zip", "images.zip")),),
        same_pages_as="regesta-graphemic",
        notes=("**50 Seiten, 2.177 Zeilen**, gemessen am 01.11.2026 — dieselben Seiten "
               "und dieselbe Zeilenzahl wie 10.5281/zenodo.18441031, diplomatisch statt "
               "graphemisch transkribiert.",),
    ),
)


def zenodo_source_by_id(source_id: str) -> ZenodoRecord:
    for source in ZENODO_SOURCES:
        if source.id == source_id:
            return source
    known = ", ".join(s.id for s in ZENODO_SOURCES)
    raise ForeignGtError(f"unknown Zenodo source {source_id!r}; known: {known}")

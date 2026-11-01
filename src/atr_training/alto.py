"""ALTO → PageXML, so that ALTO ground truth reaches the training path (#197).

Our datasets are PageXML (``PAGEXML_COLUMNS``) and ``prepare`` cuts lines from
``TextLine`` coordinates. ALTO keeps its text in ``String/@CONTENT``, where our
parser looks for ``Unicode``: an ALTO file let through unconverted reads as a page
**without transcription** — empty, and nothing would say so. That is why
``tracked_pages`` counts and drops ALTO unless the source asks for this adapter.

The mapping, as the files of dach-gt and gt-fraktur carry it:

    ALTO                                      PAGE
    TextBlock (Shape/Polygon or HPOS…)        TextRegion/Coords
    TextLine (Shape/Polygon or HPOS…)         TextLine/Coords
    TextLine/@BASELINE                        TextLine/Baseline
    String/@CONTENT, SP, HYP in order         TextLine/TextEquiv/Unicode
    Page/@WIDTH, @HEIGHT                      Page/@imageWidth, @imageHeight
    sourceImageInformation/fileName           Page/@imageFilename
    sourceImageInformation/fileIdentifier     Metadata/@externalRef

**The word gaps are the work.** ALTO splits a line into ``String`` elements with
``SP`` between words; whether two ``String`` are separated by a space is the
presence of an ``SP``, not the distance between their boxes. Joining blindly with
spaces would produce transcriptions that differ systematically from our
convention, and that goes straight into the CER. eScriptorium's exports (dach-gt)
put a whole line into one ``String`` with the spaces inside ``CONTENT``; that is
read as it is.

A line whose text is empty is dropped and counted: ALTO exports carry them (400
of dach-gt's 672 DE-4 pages have at least one), and an empty line in PAGE is a
line the training path would read as "nothing written here".
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import PurePosixPath
from xml.sax.saxutils import escape, quoteattr

__all__ = ["AltoError", "AltoPage", "alto_to_pagexml", "convert", "line_text"]

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"


class AltoError(ValueError):
    """Not an ALTO document we can convert."""


@dataclass(frozen=True)
class AltoPage:
    """A converted page, and what the conversion left out."""

    pagexml: str
    lines: int
    dropped_empty: int
    image_filename: str


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(el: ET.Element, name: str):
    return [c for c in el if _local(c.tag) == name]


def _find(el: ET.Element, name: str) -> ET.Element | None:
    return next((c for c in el.iter() if _local(c.tag) == name), None)


def _num(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def _rect(el: ET.Element) -> list[tuple[int, int]]:
    x, y = _num(el.get("HPOS")), _num(el.get("VPOS"))
    w, h = _num(el.get("WIDTH")), _num(el.get("HEIGHT"))
    if None in (x, y, w, h):
        return []
    x0, y0, x1, y1 = (round(v) for v in (x, y, x + w, y + h))
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _pairs(text: str) -> list[tuple[int, int]]:
    """``"10 40 200 40"`` or ``"10,40 200,40"`` → points."""
    nums = [n for n in re.split(r"[\s,]+", (text or "").strip()) if n]
    try:
        vals = [round(float(n)) for n in nums]
    except ValueError:
        return []
    return list(zip(vals[0::2], vals[1::2])) if len(vals) % 2 == 0 else []


def _outline(el: ET.Element) -> list[tuple[int, int]]:
    """The element's own polygon when it has one, else its rectangle."""
    for shape in _children(el, "Shape"):
        poly = _find(shape, "Polygon")
        if poly is not None:
            points = _pairs(poly.get("POINTS", ""))
            if len(points) >= 3:
                return points
    return _rect(el)


def _baseline(line: ET.Element) -> list[tuple[int, int]]:
    """ALTO 4 gives points; ALTO 2/3 a single y, drawn across the line's width."""
    raw = (line.get("BASELINE") or "").strip()
    if not raw:
        return []
    points = _pairs(raw)
    if len(points) >= 2:
        return points
    y, x, w = _num(raw), _num(line.get("HPOS")), _num(line.get("WIDTH"))
    if None in (y, x, w):
        return []
    return [(round(x), round(y)), (round(x + w), round(y))]


def line_text(line: ET.Element) -> str:
    """The line's text: ``String/@CONTENT`` in order, a space only where an ``SP`` is.

    ``HYP`` contributes its ``CONTENT`` (the hyphen as printed). Two ``String``
    with no ``SP`` between them are one word split by the segmenter and are joined
    without a space.
    """
    out: list[str] = []
    for child in line:
        name = _local(child.tag)
        if name in ("String", "HYP"):
            out.append(child.get("CONTENT") or "")
        elif name == "SP":
            out.append(" ")
    # Not collapsed: two SP are two spaces in the source, and the PAGE fassung
    # of the same gt-fraktur pages keeps them (". . .  79").
    return "".join(out).strip()


def _points(points: list[tuple[int, int]]) -> str:
    return " ".join(f"{x},{y}" for x, y in points)


def convert(xml: str | bytes) -> AltoPage:
    """One ALTO document as PageXML (2019 schema), with what was left out."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise AltoError(f"unparsable ALTO: {exc}") from exc
    if _local(root.tag) != "alto":
        raise AltoError(f"not ALTO: root element is <{_local(root.tag)}>")
    page = _find(root, "Page")
    if page is None:
        raise AltoError("ALTO without a <Page>")

    info = _find(root, "sourceImageInformation")
    file_name = (_find(info, "fileName").text or "").strip() if info is not None and \
        _find(info, "fileName") is not None else ""
    identifier = (_find(info, "fileIdentifier").text or "").strip() if info is not None and \
        _find(info, "fileIdentifier") is not None else ""
    image = PurePosixPath(file_name).name if file_name else \
        PurePosixPath(identifier.split("?")[0]).name

    width, height = _num(page.get("WIDTH")), _num(page.get("HEIGHT"))
    body: list[str] = []
    lines = dropped = 0
    for b, block in enumerate(el for el in page.iter() if _local(el.tag) == "TextBlock"):
        region_lines = []
        for li, line in enumerate(_children(block, "TextLine")):
            text = line_text(line)
            if not text:
                dropped += 1
                continue
            outline = _outline(line)
            if len(outline) < 3:
                dropped += 1
                continue
            lid = line.get("ID") or f"l{b}_{li}"
            base = _baseline(line)
            region_lines.append(
                f'<TextLine id={quoteattr(lid)}><Coords points="{_points(outline)}"/>'
                + (f'<Baseline points="{_points(base)}"/>' if base else "")
                + f"<TextEquiv><Unicode>{escape(text)}</Unicode></TextEquiv></TextLine>")
            lines += 1
        if not region_lines:
            continue
        region = _outline(block)
        coords = f'<Coords points="{_points(region)}"/>' if len(region) >= 3 else ""
        rid = block.get("ID") or f"r{b}"
        body.append(f'<TextRegion id={quoteattr(rid)} type="paragraph">{coords}'
                    + "".join(region_lines) + "</TextRegion>")

    size = (f' imageWidth="{round(width)}" imageHeight="{round(height)}"'
            if width and height else "")
    ref = f" externalRef={quoteattr(identifier)}" if identifier else ""
    pagexml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<PcGts xmlns="{PAGE_NS}">'
        f"<Metadata{ref}><Creator>atr_training.alto (#197)</Creator>"
        "<Created>1970-01-01T00:00:00</Created><LastChange>1970-01-01T00:00:00</LastChange>"
        "</Metadata>"
        f"<Page imageFilename={quoteattr(image)}{size}>" + "".join(body) + "</Page></PcGts>\n")
    return AltoPage(pagexml=pagexml, lines=lines, dropped_empty=dropped, image_filename=image)


def alto_to_pagexml(xml: str | bytes) -> str:
    """The PageXML of an ALTO document; :func:`convert` also says what was dropped."""
    return convert(xml).pagexml

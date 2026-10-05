"""Measuring a page corpus's line geometry before a run pays for it (#164).

The script answers two questions off one parse, and the test exists for the
mistake the first draft made: it read the texts from `line_texts` and the boxes
from `line_boxes` and paired them by position. `line_boxes` yields only
TRANSCRIBED lines, `line_texts` every line — so one untranscribed line shifts
every pairing after it, and the characters counted belong to another line.
"""

from __future__ import annotations

import pytest

from atr_training.line_ceiling import summarise
from atr_training.vgsl_geometry import aspect_per_char
from scripts.measure_corpus_geometry import samples

HEADER = ('<?xml version="1.0"?>\n<PcGts xmlns="http://schema.primaresearch.org/'
          'PAGE/gts/pagecontent/2013-07-15">\n <Page imageFilename="p.jpg" '
          'imageWidth="2000" imageHeight="3000">\n  <TextRegion>'
          '<Coords points="0,0 2000,0 2000,3000 0,3000"/>')
FOOTER = "  </TextRegion>\n </Page>\n</PcGts>"


def line(top: int, width: int, height: int, text: str) -> str:
    right, bottom = 100 + width, top + height
    return (f'<TextLine><Coords points="100,{top} {right},{top} {right},{bottom} '
            f'100,{bottom}"/><TextEquiv><Unicode>{text}</Unicode></TextEquiv></TextLine>')


def page(*lines: str) -> str:
    return HEADER + "".join(lines) + FOOTER


class FakeSource:
    """`HFPageSource`'s one method, without a hub."""

    def __init__(self, pages: list[str], cache: bool = True) -> None:
        self._pages = pages

    def stream(self, repo, files, revision=None):
        yield from ({"xml_content": xml} for xml in self._pages)


@pytest.fixture
def streamed(monkeypatch):
    def run(pages: list[str], want: int = 100):
        monkeypatch.setattr("scripts.measure_corpus_geometry.HFPageSource",
                            lambda cache=True: FakeSource(pages))
        return samples("dh-unibe/x", ["data/train/p/0.parquet"], want, False, None)
    return run


# ── the pairing ──────────────────────────────────────────────────────────────
def test_characters_are_counted_from_the_line_they_belong_to(streamed):
    """An untranscribed line between two transcribed ones. Paired by position
    against `line_texts`, the long line below would be measured with the short
    line's character count and the corpus would look twice as dense as it is."""
    pages = [page(line(100, 1600, 60, "x" * 20),
                  line(200, 300, 60, ""),
                  line(300, 1600, 60, "x" * 80))]
    aspects, per_char, read, untranscribed = streamed(pages)

    assert read == 1
    assert untranscribed == 1
    assert [round(a, 2) for a in aspects] == [26.67, 26.67], "the empty line has no text"
    assert sorted(chars for _, _, chars in per_char) == [20, 80]


def test_an_untranscribed_line_is_reported_but_not_in_the_tail(streamed):
    """It has geometry and lands in a batch — but `ketos compile` drops it
    (`--skip-empty-lines`), so counting it would overstate the ceiling's reach."""
    wide = page(line(100, 1800, 10, ""))          # 180:1, and untranscribed
    aspects, _, _, untranscribed = streamed([wide, page(line(100, 600, 60, "abcdef"))])

    assert untranscribed == 1
    assert summarise(aspects).over_ceiling == 0


# ── the two numbers ──────────────────────────────────────────────────────────
def test_the_tail_names_the_line_over_the_ceiling(streamed):
    pages = [page(line(100, 600, 60, "abcdef"), line(200, 1800, 10, "x" * 40))]
    aspects, _, _, _ = streamed(pages)

    tail = summarise(aspects)
    assert tail.lines == 2
    assert tail.maximum == pytest.approx(180.0)
    assert tail.over_ceiling == 1 and tail.state == "above_ceiling"


def test_aspect_per_char_ignores_lines_too_short_to_measure(streamed):
    """A two-character line's ratio is its margins, not its hand."""
    pages = [page(line(100, 1000, 50, "xy"), line(200, 1000, 50, "x" * 25))]
    _, per_char, _, _ = streamed(pages)

    assert len(per_char) == 2, "both are measured here"
    assert aspect_per_char(per_char) == pytest.approx(1000 / (50 * 25))


# ── the cap ──────────────────────────────────────────────────────────────────
def test_it_stops_at_the_page_it_was_asked_for(streamed):
    """A sample of a 15.6 TB corpus that reads it all is not a sample."""
    one = page(line(100, 600, 60, "abcdef"))
    aspects, _, read, _ = streamed([one] * 50, want=4)

    assert read == 4 and len(aspects) == 4


def test_a_row_without_pagexml_is_skipped_rather_than_counted(streamed, monkeypatch):
    monkeypatch.setattr("scripts.measure_corpus_geometry.HFPageSource",
                        lambda cache=True: _RowsWithoutXml())
    aspects, _, read, _ = samples("dh-unibe/x", ["f"], 10, False, None)

    assert read == 1 and len(aspects) == 1


class _RowsWithoutXml:
    def stream(self, repo, files, revision=None):
        yield {"image": b"JPEG"}
        yield {"xml_content": page(line(100, 600, 60, "abcdef"))}

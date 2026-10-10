"""Is the ground truth plausible? — auditing materialized pages before trusting a CER.

Written for serving-atr-inference#52 — whose measurement this module rests on,
and whose reading of that measurement was inverted. Both kraken runs there were
scored on the same material (identical ``chars``):

    kraken-thun-missiven-v1      CER 0.9838   11,566 chars   11,191 ins    2 del
    kraken-medieval-scripts-v1   CER 0.7074   11,566 chars    5,381 ins   48 del

``hypothesis_chars == chars - insertions + deletions`` — the kraken convention,
pinned in ``tests/test_edit_convention.py`` (#55) — makes those **377** and
**6,233** characters against a reference of 11,566. Neither model produced more
text than the reference contains; the first produced 3 % of it. **An
insertion-dominated CER means the hypothesis is far too short.**

The issue read the asymmetry the other way ("every model tried here outputs
substantially more characters than the reference says are on the line"). That
holds only for the un-adapted VLM base, whose CER of 1.837 is unreachable without
emitting more text than the reference contains — a different run in the other
direction, folded into one story.

It matters here because the two tails of this audit point at opposite symptoms:

* **px/char above the ceiling** — the crop holds more text than its reference
  admits (a truncated or offset ``TextEquiv``). A model reading the image
  correctly then emits more than the reference and scores **deletions**.
* **px/char below the floor** — the reference claims more text than the crop can
  hold. Even a perfect model reads only what is in the crop, emits less, and
  scores **insertions**. *This* is the tail whose shape matches #52's kraken
  numbers — if the material is at fault there at all.

That last caveat is the audit's limit, and it is not a small one: an
insertion-dominated CER is also exactly what an undertrained CTC network produces
when it collapses to blank, which is a model fault and no business of the
material. 377 characters out of 11,566 is more collapse than any misalignment
explains. This audit says whether the material *could* account for the symptom.
It does not say that it does.

So before scoring a known-good model against this material (the expensive half of
#52), ask the material a question it can answer on its own: **how many pixels of
line is each reference character supposed to account for?**

For handwriting at these resolutions a character occupies very roughly 15–40 px of
line width. A line 800 px wide with a 5-character transcription implies 160 px per
character, which no hand produces — that line is cropped from an image containing
far more text than its reference admits to. One such line is a typo; a
distribution centred there means the ground truth is not aligned with the images,
and no amount of training will fix it.

Pure: stdlib plus :mod:`atr_training.pagexml`. It reads the PageXML the
prepare stage already wrote, so it needs no GPU, no model and no network.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from atr_training.pagexml import PageXMLError, line_boxes

__all__ = [
    "LineAudit",
    "MaterialAudit",
    "PX_PER_CHAR_PLAUSIBLE",
    "audit_pages",
    "audit_xml",
    "percentiles",
]

#: Plausible range for line-width pixels per reference character, for handwriting
#: at the resolutions dh-unibe scans at (~1600 px wide pages). Wide on purpose:
#: the point is to catch a distribution centred at 150, not to police 12 vs 45.
#: Below the floor means the reference claims more text than the crop can hold —
#: the direction that produces **insertions**, because the model can only read
#: what is there. Above the ceiling means the crop holds more text than the
#: reference admits — the direction that produces **deletions**.
PX_PER_CHAR_PLAUSIBLE = (6.0, 60.0)


@dataclass
class LineAudit:
    """One transcribed line: what it says, and how much image it occupies."""

    page: str
    index: int
    chars: int
    width: int
    height: int
    text: str

    @property
    def reading_length(self) -> int:
        """Extent along the reading direction — the longer side of the box.

        Not always the width: this corpus contains vertical lines (marginalia, and
        rotated regions). Real examples from the Thun eval set, 64x310 px and
        51x399 px, are read top-to-bottom, and dividing their *width* by their
        character count gave 3.2 and 2.0 px/char — flagging perfectly good ground
        truth as "reference too long for its crop". The longer side is the one
        the glyphs run along, whichever way the line is turned.
        """
        return max(self.width, self.height)

    @property
    def is_vertical(self) -> bool:
        return self.height > self.width

    @property
    def px_per_char(self) -> float | None:
        return self.reading_length / self.chars if self.chars else None


def percentiles(values: list[float], points: Iterable[int] = (5, 25, 50, 75, 95)) -> dict[str, float]:
    """Percentiles as a plain dict. Empty input gives an empty dict, not a crash."""
    if not values:
        return {}
    ordered = sorted(values)
    out: dict[str, float] = {}
    for p in points:
        # Nearest-rank; exact interpolation is false precision for this purpose.
        index = min(int(round(p / 100 * (len(ordered) - 1))), len(ordered) - 1)
        out[f"p{p}"] = round(ordered[index], 2)
    return out


@dataclass
class MaterialAudit:
    """What a set of materialized pages looks like as training material."""

    pages: int = 0
    pages_unreadable: int = 0
    lines: int = 0
    chars: int = 0
    #: COUNTS over every line, kept separately from the examples below — the
    #: example list is capped for display, and deriving the counts from it once
    #: reported "20 of 96 lines (21%)" for a set where all 96 were implausible.
    implausible_count: int = 0
    too_wide_count: int = 0
    too_narrow_count: int = 0
    #: A capped sample of the worst offenders, for a human to eyeball.
    examples: list[LineAudit] = field(default_factory=list)
    px_per_char: list[float] = field(default_factory=list)
    chars_per_line: list[float] = field(default_factory=list)
    widths: list[float] = field(default_factory=list)

    @property
    def implausible_fraction(self) -> float:
        return self.implausible_count / self.lines if self.lines else 0.0

    def summary(self) -> dict:
        return {
            "pages": self.pages,
            "pages_unreadable": self.pages_unreadable,
            "lines": self.lines,
            "chars": self.chars,
            "chars_per_line": {
                "mean": round(statistics.fmean(self.chars_per_line), 1) if self.chars_per_line else 0,
                **percentiles(self.chars_per_line),
            },
            "px_per_char": {
                "mean": round(statistics.fmean(self.px_per_char), 1) if self.px_per_char else 0,
                **percentiles(self.px_per_char),
                "plausible_range": list(PX_PER_CHAR_PLAUSIBLE),
            },
            "line_width_px": percentiles(self.widths),
            "implausible_lines": self.implausible_count,
            "implausible_fraction": round(self.implausible_fraction, 4),
            # Split out because the two directions mean opposite things: too much
            # image per character explains a deletion-dominated CER, too little
            # explains an insertion-dominated one. Reporting only one of them
            # left the tail that matches #52 uncounted.
            "too_much_image_per_char": self.too_wide_count,
            "too_little_image_per_char": self.too_narrow_count,
        }

    def verdict(self) -> str:
        """One sentence a human can act on."""
        if not self.lines:
            return "NO LINES — nothing here is usable as training material."
        low, high = PX_PER_CHAR_PLAUSIBLE
        wide = self.too_wide_count
        narrow = self.too_narrow_count
        median = percentiles(self.px_per_char).get("p50", 0)
        if wide / self.lines > 0.2 and narrow / self.lines > 0.2:
            # Beide Seiten gerissen. Nur eine zu nennen hiesse, die Hälfte des
            # Befundes zu verschweigen — und die genannte Seite würde die
            # Fehlersuche in die eine Richtung lenken, die vielleicht die
            # kleinere ist.
            return (
                f"SUSPECT — both directions at once: {wide} of {self.lines} lines "
                f"({wide / self.lines:.0%}) have more than {high:.0f} px per "
                f"reference character and {narrow} ({narrow / self.lines:.0%}) have "
                f"less than {low:.0f} (median {median}). Some references hold less "
                "than their crop, others more, so the CER is neither cleanly "
                "insertion- nor deletion-dominated. The pairing is broken, not just "
                "skewed — fix the material before reading any score from it."
            )
        if wide / self.lines > 0.2:
            return (
                f"SUSPECT — {wide} of {self.lines} lines ({wide / self.lines:.0%}) "
                f"have more than {high:.0f} px of line per reference character "
                f"(median {median}). The crops contain more text than the references "
                "admit to, which is what produces a deletion-dominated CER. "
                "Fix the material before reading any score from it."
            )
        if narrow / self.lines > 0.2:
            return (
                f"SUSPECT — {narrow} of {self.lines} lines ({narrow / self.lines:.0%}) "
                f"have less than {low:.0f} px of line per reference character "
                f"(median {median}). The references claim more text than the crops "
                "can hold, which is what produces an insertion-dominated CER — the "
                "shape #52 measured. Fix the material before reading any score from it."
            )
        if self.implausible_fraction > 0.2:
            return (
                f"SUSPECT — {self.implausible_count} of {self.lines} lines are outside "
                f"{low:.0f}–{high:.0f} px per character (median {median}). Inspect the "
                "listed examples before trusting a CER."
            )
        return (
            f"PLAUSIBLE — median {median} px per character over {self.lines} lines, "
            f"{self.implausible_fraction:.1%} outside {low:.0f}–{high:.0f}. The ground "
            "truth is not obviously misaligned, so a CER dominated by either "
            "direction points at training or decoding rather than at the material "
            "— a blank collapse for insertions, a failure to stop for deletions."
        )


def audit_xml(xml_text: str, page: str, into: MaterialAudit) -> None:
    """Add one PageXML document's transcribed lines to ``into``."""
    low, high = PX_PER_CHAR_PLAUSIBLE
    for box in line_boxes(xml_text):
        text = box.text.strip()
        if not text:
            continue
        line = LineAudit(page=page, index=box.index, chars=len(text),
                         width=box.width, height=box.height, text=text)
        into.lines += 1
        into.chars += line.chars
        into.chars_per_line.append(float(line.chars))
        into.widths.append(float(line.width))
        ratio = line.px_per_char
        if ratio is None:
            continue
        into.px_per_char.append(ratio)
        if not (low <= ratio <= high):
            into.implausible_count += 1
            if ratio > high:
                into.too_wide_count += 1
            else:
                into.too_narrow_count += 1
            into.examples.append(line)


def audit_pages(xml_paths: Iterable[str | Path], max_examples: int = 20) -> MaterialAudit:
    """Audit every PageXML in ``xml_paths``.

    Unreadable pages are counted rather than raised: an audit that dies on the
    first malformed file cannot tell you how many malformed files there are.
    """
    audit = MaterialAudit()
    for path in xml_paths:
        path = Path(path)
        audit.pages += 1
        try:
            audit_xml(path.read_text(encoding="utf-8", errors="replace"), path.name, audit)
        except (PageXMLError, OSError):
            audit.pages_unreadable += 1
    # "Worst offenders first" has to mean *how far outside the band*, not the
    # largest px/char. Sorted by px/char alone, a set broken at the floor — the
    # tail that explains an insertion-dominated CER, the symptom #52 actually had
    # — was listed with its mildest case first: px/char 5.0 before 1.0, when 1.0
    # is six times outside the band and 5.0 barely outside it. A human eyeballing
    # the first twenty examples saw the least broken lines of the set. The key
    # below is symmetric and scale-free, so either tail is ranked by severity.
    def _outside(line: LineAudit) -> float:
        low, high = PX_PER_CHAR_PLAUSIBLE
        ratio = line.px_per_char or 0.0
        if ratio > high:
            return ratio / high
        if ratio < low and ratio > 0:
            return low / ratio
        return 0.0

    audit.examples.sort(key=_outside, reverse=True)
    del audit.examples[max_examples:]   # counts already recorded; this is display only
    return audit


def report(audit: MaterialAudit, as_json: bool = False) -> str:
    if as_json:
        return json.dumps(
            {"summary": audit.summary(),
             "verdict": audit.verdict(),
             "examples": [asdict(ln) | {"px_per_char": round(ln.px_per_char or 0, 1)}
                          for ln in audit.examples]},
            indent=2, ensure_ascii=False,
        )
    s = audit.summary()
    lines = [
        f"pages           {s['pages']}  ({s['pages_unreadable']} unreadable)",
        f"lines           {s['lines']}",
        f"characters      {s['chars']}",
        f"chars/line      mean {s['chars_per_line'].get('mean')}  "
        f"p5 {s['chars_per_line'].get('p5')}  p50 {s['chars_per_line'].get('p50')}  "
        f"p95 {s['chars_per_line'].get('p95')}",
        f"line width px   p5 {s['line_width_px'].get('p5')}  "
        f"p50 {s['line_width_px'].get('p50')}  p95 {s['line_width_px'].get('p95')}",
        f"px per char     mean {s['px_per_char'].get('mean')}  "
        f"p5 {s['px_per_char'].get('p5')}  p50 {s['px_per_char'].get('p50')}  "
        f"p95 {s['px_per_char'].get('p95')}   (plausible {PX_PER_CHAR_PLAUSIBLE[0]:.0f}"
        f"–{PX_PER_CHAR_PLAUSIBLE[1]:.0f})",
        f"implausible     {s['implausible_lines']} lines "
        f"({s['implausible_fraction']:.1%}): "
        f"{s['too_much_image_per_char']} with too much image per character "
        f"(-> deletions), "
        f"{s['too_little_image_per_char']} with too little (-> insertions)",
        "",
        audit.verdict(),
    ]
    if audit.examples:
        lines += ["", "worst lines (furthest outside the plausible band):"]
        for ln in audit.examples[:10]:
            lines.append(
                f"  {ln.px_per_char:7.1f} px/char  {ln.width:5d}x{ln.height:<4d} px"
                f"{' (vertical)' if ln.is_vertical else '           '}  "
                f"{ln.chars:3d} chars  {ln.page}#{ln.index}  {ln.text[:56]!r}"
            )
    return "\n".join(lines)

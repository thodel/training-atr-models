"""Was this corpus built by a prepare that knew about the line ceiling? (#115)

serving-atr-inference#90 found that one mis-segmented line sets the VRAM
ceiling for every line in its batch — kraken pads a batch to its widest member,
so a 135:1 "line" asked for a single 21.69 GiB allocation at ``batch_size: 16``.
It was fixed in this repo on 17.09.2026: :data:`pagexml.MAX_LINE_ASPECT` is 60,
and ``prepare`` drops the outliers before a page is judged.

The fix does not travel backwards. ``sweep_train.arrow`` was compiled from a
page pool materialised on **05.09.2026**, from PageXML that predates
``drop_wide_lines``, and it still holds a 177:1 line. Four attempts to fine-tune
from CATMuS on it died of CUDA OOM — three to thirteen minutes in, mid-epoch,
which is the signature: the peak hangs on the widest line in whichever batch it
lands in, not on the mean.

So the number #115 asks for would have been measured on a corpus **today's
pipeline would not produce**. That is the same class of fault as the one that
had the old sweep corpus deleted on 16.09: compiled before #89/#90.

## Why a count of zero is not an answer

``DatasetCounts.wide_lines`` used to default to ``0`` and ``max_aspect`` to
``0.0``. Read back from an artefact built before the ceiling existed, that says
"nought over-wide lines, widest ratio zero" — a finding, in the shape of a
default, about a check that never ran. Exactly the trap ``reserved_pages: 0``
set in #119, and the fix is the same: ``None`` means nobody looked.

Three states, and the middle one is why this module exists:

``applied``        the counts are recorded and the tail is under the ceiling.
``unchecked``      at least one dataset has no record. **Not** "clean" — a
                   measurement on it is not reproducible by today's prepare.
``above_ceiling``  recorded, and the tail is still over. The drop ran against a
                   different threshold than the one in force now, so the
                   difference has to be looked at rather than assumed benign.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from atr_training.pagexml import MAX_LINE_ASPECT

__all__ = [
    "APPLIED",
    "ABOVE_CEILING",
    "CeilingAudit",
    "LineTail",
    "UNCHECKED",
    "aspects_from_arrow",
    "audit",
    "summarise",
]

APPLIED = "applied"
UNCHECKED = "unchecked"
ABOVE_CEILING = "above_ceiling"


@dataclass(frozen=True)
class CeilingAudit:
    """What is known about the over-wide lines of a corpus."""

    state: str
    #: Datasets with no record of the check, by ``hf_repo``.
    unchecked: tuple[str, ...] = ()
    #: Lines the drop removed, summed over the datasets that recorded it.
    wide_lines: int = 0
    #: The widest ratio any dataset recorded, or ``None`` if none did.
    max_aspect: float | None = None
    ceiling: float = MAX_LINE_ASPECT

    @property
    def verifiable(self) -> bool:
        """Whether a number measured on this corpus can be reproduced today."""
        return self.state == APPLIED

    def describe(self) -> str:
        if self.state == APPLIED:
            tail = ("" if self.max_aspect is None
                    else f", widest line {self.max_aspect:.1f}:1")
            return (f"the line ceiling of {self.ceiling:.0f}:1 was applied: "
                    f"{self.wide_lines:,} line(s) dropped{tail}")
        if self.state == ABOVE_CEILING:
            return (f"the widest line is {self.max_aspect:.1f}:1, over the "
                    f"{self.ceiling:.0f}:1 ceiling in force now — this corpus was "
                    "built against a different threshold, and one such line sets "
                    "the VRAM peak for its whole batch (serving#90)")
        named = ", ".join(self.unchecked) or "the corpus"
        return (f"{named}: no record of the over-wide-line check, so this corpus "
                "was materialised by a prepare that did not look. It is not "
                "'clean' — a number measured on it cannot be reproduced by "
                "today's pipeline (#115)")

    def refusal(self, what: str) -> str:
        """The message for a caller that will not run on this corpus."""
        return (f"{what} refused: {self.describe()}. Rebuild the corpus with "
                "scripts/apply_line_ceiling.py and compile it again, or pass "
                "--allow-unverified-corpus and accept that the number says so.")


def audit(counts: Iterable, ceiling: float = MAX_LINE_ASPECT) -> CeilingAudit:
    """Hold a corpus's per-dataset counts against the ceiling in force now.

    ``counts`` is whatever carries ``hf_repo``, ``wide_lines`` and
    ``max_aspect`` — ``DatasetCounts``, or the mappings an artefact payload
    holds, which is where this is read from in practice.
    """
    rows: Sequence = list(counts)
    if not rows:
        # No per-dataset record at all. An artefact whose payload predates
        # `dataset_counts` is the oldest form of the same gap (#108).
        return CeilingAudit(state=UNCHECKED, ceiling=ceiling)

    unchecked: list[str] = []
    dropped = 0
    widest: float | None = None
    for row in rows:
        wide = _field(row, "wide_lines")
        aspect = _field(row, "max_aspect")
        if wide is None or aspect is None:
            unchecked.append(str(_field(row, "hf_repo") or "<unnamed dataset>"))
            continue
        dropped += int(wide)
        widest = float(aspect) if widest is None else max(widest, float(aspect))

    if unchecked:
        return CeilingAudit(state=UNCHECKED, unchecked=tuple(unchecked),
                            wide_lines=dropped, max_aspect=widest, ceiling=ceiling)
    if widest is not None and widest > ceiling:
        return CeilingAudit(state=ABOVE_CEILING, wide_lines=dropped,
                            max_aspect=widest, ceiling=ceiling)
    return CeilingAudit(state=APPLIED, wide_lines=dropped, max_aspect=widest,
                        ceiling=ceiling)


def _field(row, name: str):
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)


# ─── measuring the corpus itself ──────────────────────────────────────────────
#
# The audit above reads a record of how a corpus was made. This reads the corpus.
# It is the stronger answer and the one #115 needs: an arrow compiled outside the
# pipeline has no record at all, and a record can be right about a page pool that
# was compiled from a different one.


@dataclass(frozen=True)
class LineTail:
    """The aspect-ratio tail of a compiled corpus, and what it rules out."""

    lines: int
    median: float
    p99: float
    maximum: float
    over_ceiling: int
    ceiling: float = MAX_LINE_ASPECT

    @property
    def share_over(self) -> float:
        return 0.0 if not self.lines else self.over_ceiling / self.lines

    @property
    def state(self) -> str:
        return APPLIED if self.over_ceiling == 0 else ABOVE_CEILING

    @property
    def verifiable(self) -> bool:
        return self.state == APPLIED

    def describe(self) -> str:
        head = (f"{self.lines:,} lines: median {self.median:.1f}:1, "
                f"p99 {self.p99:.1f}:1, max {self.maximum:.1f}:1")
        if self.over_ceiling == 0:
            return f"{head} — all under the {self.ceiling:.0f}:1 ceiling"
        return (f"{head} — {self.over_ceiling:,} line(s) "
                f"({100 * self.share_over:.2f} %) over the {self.ceiling:.0f}:1 "
                "ceiling")

    def refusal(self, what: str) -> str:
        """Why this corpus must not carry ``what``.

        The cost is specific rather than general: kraken pads a batch to its
        widest member, so the peak is set by the one outlier that lands in it.
        That is why halving the batch raised memory (64 → 32.3 GiB, 32 → 36.8
        GiB) and why four fine-tune attempts died three to thirteen minutes in,
        mid-epoch, rather than at the first step (serving#90, #115).
        """
        return (
            f"{what} refused: {self.describe()}. At the height a base model "
            f"normalises to, the widest line is {self.maximum:.0f} x that height "
            "wide, and kraken pads every batch to its widest member — so one "
            "such line is paid for by every other line in its batch. Rebuild "
            "with scripts/apply_line_ceiling.py, or pass "
            "--allow-unverified-corpus and the floor will say it was measured "
            "on a corpus today's prepare would not produce.")


def _nearest_rank(values: Sequence[float], q: float) -> float:
    """The nearest-rank percentile. Chosen over interpolation because the point
    of this is the tail: an interpolated p99 over few lines invents a value
    between two real ones, and the question is which real lines are in there."""
    if not values:
        raise ValueError("no lines to take a percentile of")
    index = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return values[index]


def summarise(aspects: Iterable[float], ceiling: float = MAX_LINE_ASPECT) -> LineTail:
    """The tail of a corpus's width-to-height ratios.

    >>> tail = summarise([8.0, 9.0, 10.0, 11.0, 177.0])
    >>> tail.over_ceiling, tail.state
    (1, 'above_ceiling')
    >>> summarise([8.0, 9.0, 10.0]).state
    'applied'
    """
    values = sorted(float(a) for a in aspects if a and a > 0)
    if not values:
        raise ValueError(
            "no usable line geometry: every ratio was zero, negative or absent, "
            "so nothing can be said about the tail — which is not the same as "
            "there being no tail")
    middle = len(values) // 2
    median = (values[middle] if len(values) % 2
              else (values[middle - 1] + values[middle]) / 2)
    return LineTail(lines=len(values), median=median,
                    p99=_nearest_rank(values, 0.99), maximum=values[-1],
                    over_ceiling=sum(1 for v in values if v > ceiling),
                    ceiling=ceiling)


def aspects_from_arrow(path: str | Path, limit: int | None = None) -> list[float]:
    """Width-to-height of every line image in a compiled ketos dataset.

    The image header alone, not the pixels: ``Image.open`` is lazy and ``.size``
    reads the header, so 236,908 lines cost a header read each rather than a
    decode each.

    ``pyarrow`` and ``Pillow`` are imported here rather than at the top because
    this is the only function that needs them and the module is read by code
    that runs where neither is installed — the same reason
    ``cropping.py`` imports Pillow locally.
    """
    try:
        import pyarrow as pa
    except ImportError as exc:                       # pragma: no cover — box only
        raise RuntimeError(
            "reading a compiled corpus needs pyarrow, which is not installed "
            "here. Run this in the training venv: "
            ".venvs/kraken-train/bin/python") from exc
    from io import BytesIO

    from PIL import Image

    path = Path(path)
    with pa.memory_map(str(path), "rb") as handle:
        table = pa.ipc.open_file(handle).read_all()
    column = _image_column(table)
    out: list[float] = []
    for blob in table.column(column):
        raw = blob.as_py()
        if not raw:
            continue
        with Image.open(BytesIO(raw)) as image:
            width, height = image.size
        if height > 0:
            out.append(width / height)
        if limit is not None and len(out) >= limit:
            break
    return out


def _image_column(table) -> str:
    """The column holding the line crops.

    Found rather than assumed: ketos has named it differently across versions,
    and a wrong guess here would report "no usable line geometry" for a corpus
    that is fine — which is exactly the kind of false clean this module exists
    to avoid.
    """
    import pyarrow as pa

    for field in table.schema:
        if pa.types.is_binary(field.type) or pa.types.is_large_binary(field.type):
            return field.name
    raise RuntimeError(
        f"no binary column in {table.schema.names} — this does not look like a "
        "compiled ketos dataset, so its line geometry cannot be read")

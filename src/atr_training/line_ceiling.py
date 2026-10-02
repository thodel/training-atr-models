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
    "ArrowCut",
    "aspects_from_arrow",
    "audit",
    "cut_arrow",
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
    column, field = _image_column(table)
    out: list[float] = []
    for raw in _blobs(table, column, field):
        if not raw:
            continue
        with Image.open(BytesIO(raw)) as image:
            width, height = image.size
        if height > 0:
            out.append(width / height)
        if limit is not None and len(out) >= limit:
            break
    return out


@dataclass(frozen=True)
class ArrowCut:
    """What applying the ceiling to a compiled corpus did, or would do."""

    lines_before: int
    removed: int
    widest_before: float
    widest_after: float
    #: Characters the old alphabet had and the new one does not. Dropping 5 % of
    #: lines can take a rare character with it, and the codec is built from this
    #: histogram — so a model trained on the cut corpus can never emit them. Not
    #: a reason to refuse, a reason to say so.
    characters_lost: tuple[str, ...] = ()
    written: Path | None = None

    @property
    def lines_after(self) -> int:
        return self.lines_before - self.removed

    @property
    def share(self) -> float:
        return 100.0 * self.removed / max(1, self.lines_before)

    def __str__(self) -> str:
        where = f" -> {self.written}" if self.written else " (dry run)"
        lost = (f", {len(self.characters_lost)} character(s) lost: "
                f"{''.join(self.characters_lost)}" if self.characters_lost else "")
        return (f"{self.removed} of {self.lines_before} lines over the ceiling "
                f"({self.share:.2f} %), widest {self.widest_before:.1f}:1 -> "
                f"{self.widest_after:.1f}:1{lost}{where}")


def cut_arrow(src: str | Path, dest: str | Path | None = None, *,
              ceiling: float = MAX_LINE_ASPECT, dry_run: bool = False) -> ArrowCut:
    """Write ``src`` without the lines whose **image** exceeds ``ceiling``.

    The ceiling on the PageXML (``pagexml.MAX_LINE_ASPECT``, applied by
    ``prepare``) bounds the ``Coords`` box. kraken pads a batch to its widest
    member *as extracted*, and the two are not the same quantity: measured on the
    German sweep corpus, the boxes top out at exactly 60.0:1 with none above,
    while the extracted images reach 177:1 with 4.97 % above (#145). So the box
    ceiling is necessary and not sufficient, and this is where the binding
    quantity can be measured — after the compile, exactly, rather than predicted
    before it. Neither the box width nor the baseline's arc length predicts it;
    the discrepancy sits in the height ketos normalises to, and that is not yet
    explained.

    **The metadata is rebuilt, not copied.** ``alphabet`` is a character
    histogram that the codec is built from, and ``counts`` says how many lines
    the file holds; carrying either over unchanged would leave a file describing
    the corpus it used to be. The alphabet is recounted from the kept texts, so a
    character that only occurred on a removed line disappears from it — reported
    in ``characters_lost`` rather than passed over.

    Two passes over the file: the keep mask and the new alphabet cannot be known
    until every row has been read, and the metadata has to be written before the
    rows.
    """
    try:
        import pyarrow as pa
    except ImportError as exc:                       # pragma: no cover — box only
        raise RuntimeError(
            "cutting a compiled corpus needs pyarrow, which is not installed "
            "here. Run this in the training venv: "
            ".venvs/kraken-train/bin/python") from exc
    from collections import Counter
    from io import BytesIO

    from PIL import Image

    src = Path(src)
    if dest is not None:
        dest = Path(dest)
        if dest.resolve() == src.resolve():
            raise RuntimeError(
                f"refusing to write {dest} over its own input. A cut corpus is a "
                "different data version and needs its own digest (#113); "
                "overwriting would leave every number measured on the old one "
                "pointing at a file that no longer holds that corpus.")
    if dest is None and not dry_run:
        raise RuntimeError("cut_arrow needs a destination unless dry_run is set")

    with pa.memory_map(str(src), "rb") as handle:
        table = pa.ipc.open_file(handle).read_all()
        column, field = _image_column(table)
        keep: list[bool] = []
        widest_before = widest_after = 0.0
        alphabet: Counter = Counter()
        texts = _texts(table, column, field)
        for raw, text in zip(_blobs(table, column, field), texts):
            if not raw:
                keep.append(False)
                continue
            with Image.open(BytesIO(raw)) as image:
                width, height = image.size
            aspect = width / height if height else 0.0
            widest_before = max(widest_before, aspect)
            if aspect > ceiling:
                keep.append(False)
                continue
            keep.append(True)
            widest_after = max(widest_after, aspect)
            alphabet.update(text or "")

        removed = sum(1 for k in keep if not k)
        lost = _characters_lost(table.schema, alphabet)
        result = ArrowCut(lines_before=len(keep), removed=removed,
                          widest_before=widest_before, widest_after=widest_after,
                          characters_lost=lost)
        if dry_run:
            return result

        filtered = table.filter(pa.array(keep))
        schema = filtered.schema.with_metadata(
            _rebuilt_metadata(table.schema, alphabet, filtered.num_rows))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with pa.OSFile(str(dest), "wb") as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                writer.write_table(filtered.replace_schema_metadata(schema.metadata))
    return ArrowCut(lines_before=result.lines_before, removed=result.removed,
                    widest_before=result.widest_before,
                    widest_after=result.widest_after,
                    characters_lost=result.characters_lost, written=dest)


def _texts(table, column: str, field: str | None) -> list[str]:
    """Every line's transcription, in file order; empty where there is none."""
    if field is None:
        for name in ("text", "lines", "transcription"):
            if name in table.schema.names and name != column:
                return [v.as_py() or "" for v in table.column(name)]
        return [""] * table.num_rows
    return [(v.as_py() or {}).get("text") or "" for v in table.column(column)]


def _old_alphabet(schema) -> dict:
    import json

    raw = (schema.metadata or {}).get(b"lines")
    if not raw:
        return {}
    try:
        return dict(json.loads(raw.decode()).get("alphabet") or {})
    except (ValueError, AttributeError):
        return {}


def _characters_lost(schema, alphabet) -> tuple[str, ...]:
    before = set(_old_alphabet(schema))
    if not before:
        return ()
    return tuple(sorted(before - set(alphabet)))


def _rebuilt_metadata(schema, alphabet, rows: int) -> dict:
    """The source metadata with ``alphabet`` and ``counts`` made true again."""
    import json

    metadata = dict(schema.metadata or {})
    raw = metadata.get(b"lines")
    if not raw:
        return metadata
    try:
        record = json.loads(raw.decode())
    except ValueError:
        return metadata
    record["alphabet"] = dict(sorted(alphabet.items()))
    counts = dict(record.get("counts") or {})
    if "all" in counts:
        counts["all"] = rows
    record["counts"] = counts
    metadata[b"lines"] = json.dumps(record).encode("utf-8")
    return metadata


def _image_column(table) -> tuple[str, str | None]:
    """``(column, field)`` holding the line crops; ``field`` for a struct column.

    Found rather than assumed: ketos has named it differently across versions,
    and a wrong guess here would report "no usable line geometry" for a corpus
    that is fine — which is exactly the kind of false clean this module exists
    to avoid.

    **kraken 7.0.2 writes a struct**, and only looking for a top-level binary
    column missed it: a real compiled corpus has
    ``lines: struct<text: string, im: binary>`` beside three boolean split masks,
    so this raised "does not look like a compiled ketos dataset" about every
    dataset the pipeline produces. Measured on `german_test.arrow`, 01.10.2026 —
    the audit had never been run against a real one, which is why the message
    read as a verdict on the data rather than on the reader (#145).
    """
    import pyarrow as pa

    def binary(kind) -> bool:
        return pa.types.is_binary(kind) or pa.types.is_large_binary(kind)

    for field in table.schema:
        if binary(field.type):
            return field.name, None
    for field in table.schema:
        if pa.types.is_struct(field.type):
            for member in field.type:
                if binary(member.type):
                    return field.name, member.name
    raise RuntimeError(
        f"no binary column or struct field in {table.schema.names} — this does "
        "not look like a compiled ketos dataset, so its line geometry cannot be "
        "read")


def _blobs(table, column: str, field: str | None):
    """Every line's image bytes, in file order."""
    values = table.column(column)
    if field is None:
        for value in values:
            yield value.as_py()
        return
    for value in values:
        row = value.as_py()
        yield None if row is None else row.get(field)

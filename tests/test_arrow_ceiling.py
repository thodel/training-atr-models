"""Cutting a compiled corpus to the lines kraken can batch (#145).

The ceiling on the PageXML bounds the ``Coords`` box, and that is not the
quantity that sets memory. Measured on the German sweep corpus, 01.–02.10.2026:

| | median | p99 | max | > 60:1 |
|---|---|---|---|---|
| box | 10.3 | 53.2 | 60.0 | 0.00 % |
| extracted image | 11.4 | 81.1 | 177.0 | 4.97 % |

``scripts/apply_line_ceiling.py`` removes 0 of 345,359 lines from that pool. The
boxes are clean; the images are not.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from atr_training.line_ceiling import ArrowCut, aspects_from_arrow, cut_arrow

pa = pytest.importorskip("pyarrow", reason="reading a compiled corpus needs pyarrow")
PIL = pytest.importorskip("PIL", reason="reading line images needs Pillow")
from PIL import Image  # noqa: E402


def png(width: int, height: int = 10) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def write_arrow(path: Path, lines: list[tuple[int, str]], *, struct: bool = True) -> Path:
    """A corpus in the shape kraken 7.0.2 writes: a struct beside split masks."""
    images = [png(w) for w, _ in lines]
    texts = [t for _, t in lines]
    alphabet: dict[str, int] = {}
    for text in texts:
        for char in text:
            alphabet[char] = alphabet.get(char, 0) + 1
    metadata = {b"lines": json.dumps({
        "type": "kraken_recognition_baseline",
        "alphabet": alphabet,
        "text_type": "raw", "image_type": "raw",
        "splits": ["train", "eval", "test"], "im_mode": "RGB",
        "legacy_polygons": False,
        "counts": {"all": len(lines), "train": 0, "validation": 0, "test": 0},
    }).encode()}

    if struct:
        column = pa.array([{"text": t, "im": im} for t, im in zip(texts, images)],
                          type=pa.struct([("text", pa.string()), ("im", pa.binary())]))
        table = pa.table({"lines": column,
                          "train": pa.array([False] * len(lines)),
                          "validation": pa.array([False] * len(lines)),
                          "test": pa.array([False] * len(lines))})
    else:   # the older flat shape the reader also has to accept
        table = pa.table({"im": pa.array(images, type=pa.binary()),
                          "text": pa.array(texts)})
    table = table.replace_schema_metadata(metadata)
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    return path


def read_metadata(path: Path) -> dict:
    with pa.memory_map(str(path), "rb") as handle:
        schema = pa.ipc.open_file(handle).schema
    return json.loads((schema.metadata or {})[b"lines"].decode())


# ── the reader has to accept what the pipeline actually writes ──────────────

def test_a_struct_column_is_read(tmp_path):
    """kraken 7.0.2 writes `lines: struct<text, im>` beside three boolean masks.
    Only looking for a top-level binary column raised "this does not look like a
    compiled ketos dataset" about every dataset the pipeline produces."""
    path = write_arrow(tmp_path / "s.arrow", [(100, "a"), (600, "bb")])
    assert aspects_from_arrow(path) == [10.0, 60.0]


def test_the_older_flat_shape_still_works(tmp_path):
    path = write_arrow(tmp_path / "f.arrow", [(100, "a")], struct=False)
    assert aspects_from_arrow(path) == [10.0]


def test_a_file_that_really_is_not_a_corpus_still_says_so(tmp_path):
    path = tmp_path / "x.arrow"
    table = pa.table({"a": pa.array([1, 2]), "b": pa.array(["x", "y"])})
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    with pytest.raises(RuntimeError, match="does not look like"):
        aspects_from_arrow(path)


# ── the cut ─────────────────────────────────────────────────────────────────

def test_lines_over_the_ceiling_go_and_the_rest_stays(tmp_path):
    path = write_arrow(tmp_path / "in.arrow",
                       [(100, "ok"), (1770, "wide"), (200, "fine")])
    cut = cut_arrow(path, tmp_path / "out.arrow", ceiling=60.0)

    assert (cut.lines_before, cut.removed, cut.lines_after) == (3, 1, 2)
    assert cut.widest_before == pytest.approx(177.0)
    assert cut.widest_after == pytest.approx(20.0)
    assert aspects_from_arrow(tmp_path / "out.arrow") == [10.0, 20.0]


def test_a_line_exactly_on_the_ceiling_stays(tmp_path):
    """60.0:1 is the limit the PageXML ceiling uses, and it keeps its own value."""
    path = write_arrow(tmp_path / "in.arrow", [(600, "edge")])
    assert cut_arrow(path, tmp_path / "out.arrow", ceiling=60.0).removed == 0


def test_a_dry_run_writes_nothing_and_still_measures(tmp_path):
    path = write_arrow(tmp_path / "in.arrow", [(100, "a"), (1770, "b")])
    cut = cut_arrow(path, tmp_path / "out.arrow", ceiling=60.0, dry_run=True)
    assert cut.removed == 1 and cut.written is None
    assert not (tmp_path / "out.arrow").exists()


# ── the metadata is the half that goes wrong silently ───────────────────────

def test_the_alphabet_is_recounted_not_carried(tmp_path):
    """The codec is built from this histogram. Carried over unchanged it would
    describe the corpus the file used to be."""
    path = write_arrow(tmp_path / "in.arrow", [(100, "aa"), (1770, "zz")])
    before = read_metadata(path)["alphabet"]
    assert before == {"a": 2, "z": 2}

    cut_arrow(path, tmp_path / "out.arrow", ceiling=60.0)
    after = read_metadata(tmp_path / "out.arrow")["alphabet"]
    assert after == {"a": 2}


def test_a_character_only_on_a_removed_line_is_reported_as_lost(tmp_path):
    """Dropping 5 % of lines can take a rare character with it, and then the
    model can never emit it. Not a reason to refuse — a reason to say so."""
    path = write_arrow(tmp_path / "in.arrow", [(100, "aa"), (1770, "ſz")])
    cut = cut_arrow(path, tmp_path / "out.arrow", ceiling=60.0)
    assert set(cut.characters_lost) == {"ſ", "z"}
    assert "ſ" in str(cut)


def test_the_line_count_in_the_metadata_follows(tmp_path):
    path = write_arrow(tmp_path / "in.arrow", [(100, "a"), (1770, "b"), (150, "c")])
    assert read_metadata(path)["counts"]["all"] == 3
    cut_arrow(path, tmp_path / "out.arrow", ceiling=60.0)
    assert read_metadata(tmp_path / "out.arrow")["counts"]["all"] == 2


def test_everything_else_in_the_metadata_survives(tmp_path):
    path = write_arrow(tmp_path / "in.arrow", [(100, "a"), (1770, "b")])
    cut_arrow(path, tmp_path / "out.arrow", ceiling=60.0)
    after = read_metadata(tmp_path / "out.arrow")
    assert after["type"] == "kraken_recognition_baseline"
    assert after["im_mode"] == "RGB" and after["legacy_polygons"] is False
    assert after["splits"] == ["train", "eval", "test"]


def test_the_split_masks_are_kept_and_filtered_with_the_rows(tmp_path):
    path = write_arrow(tmp_path / "in.arrow", [(100, "a"), (1770, "b")])
    cut_arrow(path, tmp_path / "out.arrow", ceiling=60.0)
    with pa.memory_map(str(tmp_path / "out.arrow"), "rb") as handle:
        table = pa.ipc.open_file(handle).read_all()
    assert table.schema.names == ["lines", "train", "validation", "test"]
    assert table.num_rows == 1


# ── refusals ────────────────────────────────────────────────────────────────

def test_writing_over_the_input_is_refused(tmp_path):
    """A cut corpus is a different data version. Overwriting would leave every
    number measured on the old one pointing at a file that no longer holds it."""
    path = write_arrow(tmp_path / "in.arrow", [(100, "a")])
    with pytest.raises(RuntimeError, match="over its own input"):
        cut_arrow(path, path, ceiling=60.0)


def test_a_cut_without_a_destination_is_refused(tmp_path):
    path = write_arrow(tmp_path / "in.arrow", [(100, "a")])
    with pytest.raises(RuntimeError, match="needs a destination"):
        cut_arrow(path, None, ceiling=60.0)


def test_nothing_over_the_ceiling_is_a_finding_not_a_failure(tmp_path):
    path = write_arrow(tmp_path / "in.arrow", [(100, "a"), (200, "b")])
    cut = cut_arrow(path, tmp_path / "out.arrow", ceiling=60.0)
    assert cut.removed == 0 and cut.lines_after == 2
    assert isinstance(cut, ArrowCut) and cut.written is not None

"""The corpus #115 was about to be measured on is not one we can rebuild.

serving-atr-inference#90 found that one mis-segmented line sets the VRAM ceiling
for every line in its batch, and it was fixed here on 17.09.2026:
``pagexml.MAX_LINE_ASPECT`` is 60 and ``prepare`` drops the outliers. The fix
does not travel backwards. ``sweep_train.arrow`` was compiled from a page pool
materialised on **05.09.2026** and still holds a 177:1 line; four attempts to
fine-tune on it died of CUDA OOM three to thirteen minutes in, mid-epoch — the
signature of a peak that hangs on one outlier rather than on the mean.

So the noise floor would have been measured on a corpus today's pipeline would
not produce. Same class as the fault that had the old sweep corpus deleted on
16.09: compiled before #89/#90.

These cover the three things that follow, and the first is the one that keeps
coming back in this project:

* **A count of zero is not an answer.** ``wide_lines`` defaulted to ``0`` and
  ``max_aspect`` to ``0.0``; read back from an artefact built before the ceiling
  existed, that states a finding about a check that never ran. Three-valued now,
  for the reason ``reserved_pages`` is (#119).
* **Measure the corpus, do not trust a record about it.** An arrow compiled
  outside the pipeline has no record at all.
* **The source pool is never edited.** ``german_test`` was built from it, and
  0.2131 means 188,022 errors over 882,255 of its characters.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atr_training import line_ceiling  # noqa: E402
from atr_training.contracts import DatasetCounts  # noqa: E402
from atr_training.line_ceiling import (  # noqa: E402
    ABOVE_CEILING,
    APPLIED,
    UNCHECKED,
    audit,
    summarise,
)
from atr_training.pagexml import MAX_LINE_ASPECT  # noqa: E402

CEILING = MAX_LINE_ASPECT


def counted(repo="dh/one", **over) -> DatasetCounts:
    return DatasetCounts(hf_repo=repo, **{"wide_lines": 12, "max_aspect": 41.0,
                                          **over})


# ── a count of zero is not an answer ───────────────────────────────────────
def test_the_fields_default_to_nobody_looked():
    """The whole finding, as one assertion. `0` would read as "checked, none
    found" for every artefact built before 17.09.2026."""
    bare = DatasetCounts(hf_repo="dh/one")

    assert bare.wide_lines is None
    assert bare.max_aspect is None


def test_a_corpus_with_no_record_is_unchecked_not_clean():
    verdict = audit([DatasetCounts(hf_repo="dh/one")])

    assert verdict.state == UNCHECKED
    assert not verdict.verifiable
    assert "dh/one" in verdict.describe()
    assert "not 'clean'" in verdict.describe()


def test_checked_and_none_found_is_a_finding():
    """Zero is a real answer when somebody looked; that is the distinction."""
    verdict = audit([counted(wide_lines=0, max_aspect=38.2)])

    assert verdict.state == APPLIED
    assert verdict.verifiable
    assert verdict.wide_lines == 0


def test_one_unchecked_dataset_taints_the_corpus():
    """A sweep trains on all of them, so the weakest record decides."""
    verdict = audit([counted("dh/one"), DatasetCounts(hf_repo="dh/two")])

    assert verdict.state == UNCHECKED
    assert verdict.unchecked == ("dh/two",)


def test_no_counts_at_all_is_unchecked():
    """An artefact payload older than `dataset_counts` is the oldest form of
    the same gap (#108)."""
    assert audit([]).state == UNCHECKED


def test_a_tail_over_the_current_ceiling_is_its_own_verdict():
    """The drop ran, against a different threshold than the one in force now.
    That is neither 'applied' nor 'nobody looked'."""
    verdict = audit([counted(max_aspect=177.0)])

    assert verdict.state == ABOVE_CEILING
    assert "177.0:1" in verdict.describe()
    assert "serving#90" in verdict.describe()


def test_the_counts_sum_and_the_worst_aspect_wins():
    verdict = audit([counted("dh/one", wide_lines=12, max_aspect=41.0),
                     counted("dh/two", wide_lines=30, max_aspect=58.0)])

    assert (verdict.wide_lines, verdict.max_aspect) == (42, 58.0)


def test_the_payload_form_reads_the_same():
    """This is read from an artefact payload in practice, where the rows are
    mappings rather than models."""
    rows = [{"hf_repo": "dh/one", "wide_lines": 3, "max_aspect": 44.0}]

    assert audit(rows).state == APPLIED
    assert audit([{"hf_repo": "dh/one"}]).state == UNCHECKED


def test_the_refusal_names_the_way_out():
    assert "apply_line_ceiling.py" in audit([]).refusal("measuring")
    assert "--allow-unverified-corpus" in audit([]).refusal("measuring")


# ── measuring the corpus itself ────────────────────────────────────────────
def test_the_tail_is_the_users_own_numbers():
    """From the sample measured on 30.09: median 11.4, p99 81.1, max 177.0."""
    aspects = [11.4] * 98 + [81.1, 177.0]

    tail = summarise(aspects)

    assert tail.lines == 100
    assert tail.median == pytest.approx(11.4)
    assert tail.p99 == pytest.approx(81.1)
    assert tail.maximum == pytest.approx(177.0)


def test_lines_over_the_ceiling_are_counted_and_shared():
    tail = summarise([10.0] * 95 + [70.0] * 5)

    assert tail.over_ceiling == 5
    assert tail.share_over == pytest.approx(0.05)
    assert tail.state == ABOVE_CEILING
    assert not tail.verifiable


def test_a_corpus_under_the_ceiling_is_verifiable():
    tail = summarise([9.9, 11.4, 58.1])

    assert tail.state == APPLIED
    assert tail.verifiable
    assert "all under the 60:1 ceiling" in tail.describe()


def test_the_percentile_is_nearest_rank_not_interpolated():
    """An interpolated p99 invents a value between two real lines, and the
    question here is which real lines are in the tail."""
    assert summarise([1.0, 2.0, 3.0, 4.0, 100.0]).p99 == 100.0


def test_the_median_of_an_even_count_is_the_midpoint():
    assert summarise([10.0, 12.0]).median == pytest.approx(11.0)


def test_no_usable_geometry_is_an_error_not_an_empty_tail():
    """"Nothing can be said about the tail" and "there is no tail" are the two
    answers this module exists to keep apart."""
    with pytest.raises(ValueError, match="not the same as"):
        summarise([0.0, 0.0])

    with pytest.raises(ValueError, match="no usable line geometry"):
        summarise([])


def test_the_refusal_explains_the_cost_rather_than_asserting_it():
    tail = summarise([10.0] * 99 + [177.0])

    refusal = tail.refusal("measuring the noise floor")
    assert "177" in refusal
    assert "pads every batch to its widest member" in refusal
    assert "--allow-unverified-corpus" in refusal


def test_reading_an_arrow_without_pyarrow_says_which_python():
    """It is not installed in every environment this module is read in, and
    "cannot check" must not come out as "clean"."""
    with pytest.raises(RuntimeError, match=r"\.venvs/kraken-train/bin/python"):
        line_ceiling.aspects_from_arrow(ROOT / "nonexistent.arrow")


# ── the source pool is never edited ────────────────────────────────────────
PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="{img}" imageWidth="2000" imageHeight="1500">
    <TextRegion id="r1">
{lines}
    </TextRegion>
  </Page>
</PcGts>
"""


def a_line(index: int, width: int, height: int = 100) -> str:
    return (f'      <TextLine id="l{index}">'
            f'<Coords points="0,0 {width},0 {width},{height} 0,{height}"/>'
            f"<TextEquiv><Unicode>abcdefghij</Unicode></TextEquiv></TextLine>")


@pytest.fixture
def pool(tmp_path: Path) -> Path:
    """Three pages: one with an outlier, one that is nothing but outliers, one
    clean — the three cases `prepare` distinguishes."""
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "0001_a.xml").write_text(PAGE.format(img="0001_a.jpg", lines="\n".join(
        [a_line(1, 1000), a_line(2, 900), a_line(3, 1100), a_line(4, 17700)])))
    (pages / "0002_b.xml").write_text(PAGE.format(img="0002_b.jpg", lines="\n".join(
        [a_line(1, 9000), a_line(2, 8000)])))
    (pages / "0003_c.xml").write_text(
        PAGE.format(img="0003_c.jpg", lines=a_line(1, 800)))
    for stem in ("0001_a", "0002_b", "0003_c"):
        (pages / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    return pages


def ceiling_script(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "apply_line_ceiling.py"), *args],
        capture_output=True, text=True, timeout=120)


def test_the_source_pool_is_untouched(pool, tmp_path):
    before = {p.name: p.read_bytes() for p in pool.iterdir()}

    ceiling_script("--pool", str(pool), "--out", str(tmp_path / "cut"))

    assert {p.name: p.read_bytes() for p in pool.iterdir()} == before


def test_a_page_with_nothing_trainable_left_is_dropped(pool, tmp_path):
    """Exactly as `prepare` drops it: every line was an outlier."""
    out = tmp_path / "cut"
    ceiling_script("--pool", str(pool), "--out", str(out))

    assert not (out / "0002_b.xml").exists()
    assert sorted(p.name for p in out.iterdir()) == [
        "0001_a.jpg", "0001_a.xml", "0003_c.jpg", "0003_c.xml"]


def test_the_outlier_is_gone_and_the_rest_of_the_page_stays(pool, tmp_path):
    out = tmp_path / "cut"
    ceiling_script("--pool", str(pool), "--out", str(out))

    kept = (out / "0001_a.xml").read_text(encoding="utf-8")
    assert kept.count("<TextLine") == 3
    assert "17700" not in kept


def test_the_report_says_how_much_would_go(pool, tmp_path):
    done = ceiling_script("--pool", str(pool), "--out", str(tmp_path / "cut"),
                          "--dry-run")

    assert "3 of 7 lines removed (42.86 %)" in done.stdout
    assert "widest before the cut 177.0:1, after 11.0:1" in done.stdout


def test_a_dry_run_writes_nothing(pool, tmp_path):
    out = tmp_path / "cut"

    done = ceiling_script("--pool", str(pool), "--out", str(out), "--dry-run")

    assert done.returncode == 0
    assert not out.exists()


def test_writing_inside_the_pool_is_refused_and_has_no_flag(pool, tmp_path):
    """The measurement set's identity hangs on this directory, so there is no
    --force for it."""
    done = ceiling_script("--pool", str(pool), "--out", str(pool / "inner"))

    assert done.returncode == 2
    assert "would land inside --pool" in done.stderr
    assert not (pool / "inner").exists()


def test_the_pool_itself_as_out_is_refused(pool):
    done = ceiling_script("--pool", str(pool), "--out", str(pool))

    assert done.returncode == 2


def test_the_images_come_along(pool, tmp_path):
    out = tmp_path / "cut"
    done = ceiling_script("--pool", str(pool), "--out", str(out))

    assert "2 images hardlinked" in done.stdout
    assert (out / "0001_a.jpg").read_bytes() == b"\xff\xd8\xff\xd9"


def test_a_page_without_an_image_is_reported_not_skipped(pool, tmp_path):
    (pool / "0003_c.jpg").unlink()
    out = tmp_path / "cut"

    done = ceiling_script("--pool", str(pool), "--out", str(out))

    assert (out / "0003_c.xml").exists()
    assert "1 missing" in done.stdout


def test_a_directory_with_no_pagexml_says_so(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()

    done = ceiling_script("--pool", str(empty), "--out", str(tmp_path / "cut"))

    assert done.returncode != 0
    assert "is that the pages directory" in done.stderr


# ── the floor carries what it was measured on ──────────────────────────────
sys.path.insert(0, str(ROOT / "scripts"))


def manifest_with(tmp_path: Path, digest="sha256:abc") -> Path:
    path = tmp_path / "sweep.yaml"
    path.write_text(
        "# A comment that carries the reasoning, and must survive a write.\n"
        "name: t\n"
        "data:\n"
        f"  digest: {digest}\n"
        "  train: x\n"
        "  eval: y\n"
        "  datasets:\n"
        "    - hf_repo: a/b\n"
        "budget:\n"
        "  steps: 1000\n"
        "notes: |\n"
        "  prose at the bottom\n", encoding="utf-8")
    return path


SUMMARY = {
    "data_digest": "sha256:abc", "seeds": [42, 43, 44, 45], "commit": "abc1234",
    "steps": 10186, "measured_at": "2026-10-01T12:00:00Z",
    "spec": "[256,64,0,1 …]", "input_height": 64,
    "line_tail": {"lines": 236908, "median": 11.4, "p99": 81.1, "max": 177.0,
                  "over_ceiling": 11781, "ceiling": 60.0,
                  "state": ABOVE_CEILING, "sampled": None},
    "noise_floor": {"spread": 0.0085, "n": 4},
}


def test_writing_the_floor_keeps_the_comments(tmp_path):
    """`yaml.safe_dump` of the parsed document would have deleted every comment
    — and in these manifests the comments are the reasoning."""
    from measure_noise_floor import write_into_manifest

    path = manifest_with(tmp_path)
    write_into_manifest(path, SUMMARY)

    text = path.read_text(encoding="utf-8")
    assert "# A comment that carries the reasoning" in text
    assert "prose at the bottom" in text


def test_the_written_block_says_which_shape_it_was_measured_at(tmp_path):
    """What makes it a lower bound rather than an error bar: the spread grows
    with the height (#115)."""
    import yaml
    from measure_noise_floor import write_into_manifest

    path = manifest_with(tmp_path)
    write_into_manifest(path, SUMMARY)

    block = yaml.safe_load(path.read_text(encoding="utf-8"))["noise_floor"]
    assert block["value"] == 0.0085
    assert block["input_height"] == 64
    assert block["line_tail"]["state"] == ABOVE_CEILING


def test_writing_twice_replaces_rather_than_duplicates(tmp_path):
    from measure_noise_floor import write_into_manifest

    path = manifest_with(tmp_path)
    write_into_manifest(path, SUMMARY)
    write_into_manifest(path, SUMMARY)

    assert path.read_text(encoding="utf-8").count("\nnoise_floor:") == 1


def test_the_block_lands_above_the_notes(tmp_path):
    """So the prose stays at the bottom where a reader expects it."""
    from measure_noise_floor import write_into_manifest

    path = manifest_with(tmp_path)
    write_into_manifest(path, SUMMARY)

    text = path.read_text(encoding="utf-8")
    assert text.index("noise_floor:") < text.index("notes:")


def test_a_floor_from_other_material_is_refused(tmp_path):
    from measure_noise_floor import write_into_manifest

    path = manifest_with(tmp_path, digest="sha256:somethingelse")

    with pytest.raises(ValueError, match="license a ranking it never earned"):
        write_into_manifest(path, SUMMARY)


def test_the_manifest_still_loads_after_the_write(tmp_path):
    from atr_training.sweep_manifest import load_manifest
    from measure_noise_floor import write_into_manifest

    path = manifest_with(tmp_path)
    write_into_manifest(path, SUMMARY)

    assert load_manifest(path).noise_floor == 0.0085


# ── the leaderboard says what the floor is and is not ──────────────────────
def test_the_floor_line_calls_it_a_lower_bound():
    from atr_training.leaderboard import _floor_line
    from atr_training.sweep_manifest import parse_manifest

    manifest = parse_manifest({
        "name": "t", "data": {"train": "x", "eval": "y", "digest": "sha256:abc",
                              "datasets": [{"hf_repo": "a/b"}]},
        "budget": {"steps": 1000}, "base": {"lrate": 1.0e-4},
        "noise_floor": {"value": 0.0085, "measured_on": "sha256:abc",
                        "seeds": [42, 43], "commit": "abc1234", "steps": 2000,
                        "input_height": 64},
    })

    line = _floor_line(manifest)
    assert "lower bound" in line
    assert "height 64" in line
    assert "not thereby distinguished" in line


def test_the_floor_line_flags_an_unverified_corpus():
    from atr_training.leaderboard import _floor_line
    from atr_training.sweep_manifest import parse_manifest

    manifest = parse_manifest({
        "name": "t", "data": {"train": "x", "eval": "y", "digest": "sha256:abc",
                              "datasets": [{"hf_repo": "a/b"}]},
        "budget": {"steps": 1000}, "base": {"lrate": 1.0e-4},
        "noise_floor": {"value": 0.0085, "measured_on": "sha256:abc",
                        "seeds": [42, 43], "commit": "abc1234", "steps": 2000,
                        "line_tail": {"state": ABOVE_CEILING, "over_ceiling": 11781,
                                      "ceiling": 60.0}},
    })

    line = _floor_line(manifest)
    assert "unverified corpus" in line
    assert "11781" in line


def test_the_measurement_script_has_the_guard_and_the_override():
    source = (ROOT / "scripts" / "measure_noise_floor.py").read_text(encoding="utf-8")

    assert "--allow-unverified-corpus" in source
    assert "aspects_from_arrow" in source
    assert "return 2" in source


def test_the_guard_runs_before_any_training(tmp_path):
    """Four runs of several hours each: the corpus is judged first, not after."""
    source = (ROOT / "scripts" / "measure_noise_floor.py").read_text(encoding="utf-8")

    assert source.index("aspects_from_arrow") < source.index("for seed in args.seeds")


def test_json_round_trips_the_tail():
    """It travels with the number in noise_floor.json, like the commit."""
    assert json.loads(json.dumps(SUMMARY))["line_tail"]["max"] == 177.0

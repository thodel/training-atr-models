"""A sweep is a file, and a configuration is a hash of what it resolves to (#113).

K1's acceptance is three sentences, and the middle one is the hard part: two
manifests that are written differently but mean the same thing must produce the
same `config_id`s, and a changed data version must produce different ones.

"Written differently" is doing more work there than it looks. PyYAML resolves
`1e-4` to the string `'1e-4'` — its float pattern requires a decimal point —
while `1.0e-4` and `0.0001` both resolve to the float. The notation in #113's own
example manifest would therefore have given one configuration two ids depending
on how somebody typed it. `test_the_notation_of_a_number_does_not_change_its_id`
is that case.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from atr_training.sweep_manifest import (
    MANIFEST_VERSION, ManifestError, canonical, config_id, load_manifest,
    parse_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts" / "check_sweep.py"

MANIFEST = """
name: kraken-medieval-augment-01
data:
  train: shard_00
  eval: german_test
  digest: "sha256:0123456789abcdef"
  datasets:
    - {hf_repo: "dh-unibe/image-text_aaeb-xiv-xvii", granularity: line}
budget:
  steps: 4000
  rungs: [12, 4, 1]
base:
  spec: "[1,48,0,1 Cr3,3,32]"
  lrate: 1.0e-4
  schedule: cosine
  batch_size: 256
  augment: false
  normalization: NFD
axes:
  augment: [false, true]
  normalization: [NFD, NFC]
  lrate: [3.0e-5, 1.0e-4, 3.0e-4]
"""


def parse(text: str = MANIFEST, **edits):
    raw = yaml.safe_load(text)
    for dotted, value in edits.items():
        node = raw
        *path, leaf = dotted.split(".")
        for step in path:
            node = node[step]
        if value is None:
            node.pop(leaf, None)
        else:
            node[leaf] = value
    return parse_manifest(raw, source="test")


def ids(manifest) -> list[str]:
    return [c.config_id for c in manifest.configs()]


# ── the cross product ───────────────────────────────────────────────────────
def test_the_axes_resolve_to_the_full_cross_product():
    assert len(parse().configs()) == 2 * 2 * 3


def test_every_configuration_has_its_own_id():
    assert len(set(ids(parse()))) == 12


def test_a_configuration_carries_the_base_as_well_as_its_axis_values():
    """The trainer needs all of it; the leaderboard needs only the axes."""
    config = parse().configs()[0]

    assert config.params["spec"] == "[1,48,0,1 Cr3,3,32]"
    assert set(config.axes) == {"augment", "normalization", "lrate"}


def test_a_manifest_without_axes_is_one_configuration():
    """Useful on its own: a single run whose data version is on the record."""
    manifest = parse(**{"axes": None, "budget.rungs": None})

    assert len(manifest.configs()) == 1


# ── the same experiment, written differently ────────────────────────────────
def test_the_order_of_the_axes_does_not_change_the_ids():
    reordered = yaml.safe_load(MANIFEST)
    reordered["axes"] = dict(reversed(list(reordered["axes"].items())))

    assert ids(parse_manifest(reordered, source="test")) == ids(parse())


def test_the_order_of_the_base_keys_does_not_change_the_ids():
    shuffled = yaml.safe_load(MANIFEST)
    shuffled["base"] = dict(reversed(list(shuffled["base"].items())))

    assert ids(parse_manifest(shuffled, source="test")) == ids(parse())


def test_the_notation_of_a_number_does_not_change_its_id():
    """The one that fails without `canonical`, and the one #113's own example
    would have tripped over.

    `1e-4` is a *string* to PyYAML; `1.0e-4` and `0.0001` are the float. Three
    spellings of one learning rate, and without folding they are three
    configurations.
    """
    spellings = ["1.0e-4", "0.0001", "1e-4", ".0001", "1.0E-4"]
    written = [parse(**{"base.lrate": yaml.safe_load(s),
                        "axes.lrate": [yaml.safe_load(s)],
                        "budget.rungs": None})
               for s in spellings]

    assert len({tuple(ids(m)) for m in written}) == 1, \
        {s: ids(m)[0] for s, m in zip(spellings, written)}


def test_a_whole_number_is_one_value_however_it_is_written():
    trio = [parse(**{"base.batch_size": v, "axes.batch_size": [v]})
            for v in (256, 256.0, "256")]

    assert len({tuple(ids(m)) for m in trio}) == 1


def test_a_flag_is_not_folded_into_a_number():
    """`float(True)` is 1.0 in Python, so an unguarded numeric branch would make
    `augment: true` and `augment: 1` one value and lose the distinction."""
    assert canonical(True) is True
    assert canonical(1) == 1
    assert config_id({"a": True}, data_digest="d") != config_id({"a": 1}, data_digest="d")


# ── the data version ────────────────────────────────────────────────────────
def test_a_different_data_version_gives_different_ids():
    """K1's other explicit criterion. A number measured on other material is not
    the same number."""
    other = parse(**{"data.digest": "sha256:ffffffffffffffff"})

    assert set(ids(other)).isdisjoint(ids(parse()))


def test_a_manifest_without_a_data_digest_is_refused():
    """Refused, not warned: that is the difference #113 asks for, and the first
    sweep's numbers became unreadable for exactly this reason."""
    with pytest.raises(ManifestError, match="data.digest is required"):
        parse(**{"data.digest": None})


def test_an_empty_data_digest_is_refused_like_a_missing_one():
    with pytest.raises(ManifestError, match="data.digest is required"):
        parse(**{"data.digest": "   "})


def test_the_refusal_says_where_to_get_the_digest():
    with pytest.raises(ManifestError, match="restore_eval_split"):
        parse(**{"data.digest": None})


# ── stability under editing ─────────────────────────────────────────────────
def test_adding_a_value_to_an_axis_leaves_every_other_id_alone():
    """The failure #113 names: with index-based ids, inserting `1e-5` at the
    front renames every configuration after it and the leaderboard compares
    unlike things under one name."""
    before = set(ids(parse(**{"budget.rungs": None})))
    after = set(ids(parse(**{"axes.lrate": [1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4],
                             "budget.rungs": None})))

    assert before < after
    assert len(after - before) == 4        # the new rate, once per other axis


def test_promoting_a_base_value_to_an_axis_keeps_that_configuration():
    """The useful half of "stable when the file gains an axis": the run that
    used the default is recognised as the run that used the default."""
    without = parse(**{"axes.normalization": None, "budget.rungs": None})
    with_axis = parse(**{"budget.rungs": None})

    assert set(ids(without)) < set(ids(with_axis))


def test_a_parameter_that_was_never_declared_cannot_become_an_axis():
    """The half no scheme can make stable, refused at the door instead.

    A run that did not pin `momentum` is not the same run as one that pinned it
    to 0.9. Requiring a declared default in `base` makes that edit visible
    instead of letting it silently renumber the sweep.
    """
    with pytest.raises(ManifestError, match="no entry in `base`"):
        parse(**{"axes.momentum": [0.9, 0.95]})


def test_renaming_the_sweep_does_not_change_the_ids():
    """Deliberate: the same parameters on the same data are the same experiment,
    and two overlapping sweeps should be able to see that they overlap."""
    assert ids(parse(**{"name": "something-else"})) == ids(parse())


def test_the_version_is_part_of_the_hash():
    """So a sweep resumed after the meaning of a field changes gets new ids
    rather than comparing across two definitions. Same role as KEY_VERSION."""
    payload = json.dumps({"version": MANIFEST_VERSION, "data": "d",
                          "params": {"a": 1}}, sort_keys=True,
                         ensure_ascii=False, separators=(",", ":"))
    import hashlib

    assert config_id({"a": 1}, data_digest="d") == \
        hashlib.sha256(payload.encode()).hexdigest()[:12]


# ── refusals that save a re-run ─────────────────────────────────────────────
def test_an_epoch_budget_is_refused_rather_than_converted():
    """#111, lesson 3. An epoch budget hands a smaller batch size more optimizer
    steps, so a sweep with batch_size on an axis ranks by training received.
    Converting silently would need the data version and hide the assumption."""
    with pytest.raises(ManifestError, match="optimizer steps, not epochs"):
        parse(**{"budget.epochs": 30})


def test_an_empty_axis_is_refused():
    """A cross product with an empty factor is empty; a sweep that runs nothing
    should say so at parse time, not after submission."""
    with pytest.raises(ManifestError, match="empty axis"):
        parse(**{"axes.lrate": []})


def test_two_spellings_of_one_value_on_one_axis_are_refused():
    """They would be two configurations with one id, and the second result would
    overwrite the first — a sweep quietly measuring eleven of twelve."""
    with pytest.raises(ManifestError, match="same value once canonicalised"):
        parse(**{"axes.lrate": [1.0e-4, 0.0001, 3.0e-4]})


def test_a_ladder_that_does_not_narrow_is_refused():
    with pytest.raises(ManifestError, match="does not narrow"):
        parse(**{"budget.rungs": [12, 4, 4, 1]})


def test_a_ladder_that_never_reaches_one_is_refused():
    with pytest.raises(ManifestError, match="never names a winner"):
        parse(**{"budget.rungs": [12, 4]})


def test_a_ladder_wider_than_the_axes_is_refused():
    """#113's own example has `rungs: [45, 15, 5, 1]` over axes of 2 x 2 x 3 = 12.
    Rung 0 would start short of its width and eliminate nothing."""
    with pytest.raises(ManifestError, match="rung 0 wants 45"):
        parse(**{"budget.rungs": [45, 15, 5, 1]})


def test_a_ladder_is_optional():
    """K2 decides the ladder from the config count via plan_rungs; a manifest is
    allowed to leave it to the driver."""
    assert parse(**{"budget.rungs": None}).rungs == ()


def test_a_misspelt_top_level_key_is_refused_rather_than_dropped():
    """`budgets:` would otherwise take the budget with it in silence."""
    raw = yaml.safe_load(MANIFEST)
    raw["budgets"] = raw.pop("budget")

    with pytest.raises(ManifestError, match="unknown top-level key budgets"):
        parse_manifest(raw, source="test")


@pytest.mark.parametrize("missing", ["name", "data", "budget"])
def test_a_manifest_missing_a_required_section_is_refused(missing):
    raw = yaml.safe_load(MANIFEST)
    raw.pop(missing)

    with pytest.raises(ManifestError, match=missing):
        parse_manifest(raw, source="test")


@pytest.mark.parametrize("steps", [0, -1, "many", 1.5, True])
def test_a_step_budget_must_be_a_positive_whole_number(steps):
    with pytest.raises(ManifestError, match="budget.steps"):
        parse(**{"budget.steps": steps})


# ── loading from disk ───────────────────────────────────────────────────────
def test_a_manifest_loads_from_a_file(tmp_path: Path):
    path = tmp_path / "sweep.yaml"
    path.write_text(MANIFEST, encoding="utf-8")

    assert ids(load_manifest(path)) == ids(parse())


def test_a_missing_file_says_so(tmp_path: Path):
    with pytest.raises(ManifestError, match="no sweep manifest at"):
        load_manifest(tmp_path / "nope.yaml")


def test_a_file_that_is_not_yaml_says_so(tmp_path: Path):
    path = tmp_path / "sweep.yaml"
    path.write_text("name: [unclosed\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="not readable as YAML"):
        load_manifest(path)


# ── the check script ────────────────────────────────────────────────────────
def _run(*args):
    return subprocess.run([sys.executable, str(CHECK), *args],
                          capture_output=True, text=True, timeout=120)


@pytest.fixture
def manifest_file(tmp_path: Path) -> Path:
    path = tmp_path / "sweep.yaml"
    path.write_text(MANIFEST, encoding="utf-8")
    return path


def test_the_check_prints_the_ladder_and_the_data_version(manifest_file):
    result = _run(str(manifest_file))

    assert result.returncode == 0, result.stderr
    assert "sha256:0123456789abcdef" in result.stdout
    assert "12 → 4 → 1" in result.stdout
    assert "configs 12" in result.stdout


def test_the_check_can_print_ids_alone_for_a_diff(manifest_file):
    result = _run(str(manifest_file), "--ids")

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ids(parse())


def test_two_manifests_that_mean_the_same_thing_diff_clean(tmp_path, manifest_file):
    """The question a resumed sweep asks, answered by `diff`."""
    other = tmp_path / "other.yaml"
    raw = yaml.safe_load(MANIFEST)
    raw["name"] = "renamed"
    raw["axes"] = dict(reversed(list(raw["axes"].items())))
    raw["base"]["lrate"] = 0.0001
    other.write_text(yaml.safe_dump(raw), encoding="utf-8")

    assert _run(str(manifest_file), "--ids").stdout == _run(str(other), "--ids").stdout


def test_the_check_refuses_a_bad_manifest_with_the_reason_on_stderr(tmp_path):
    path = tmp_path / "sweep.yaml"
    raw = yaml.safe_load(MANIFEST)
    del raw["data"]["digest"]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = _run(str(path))

    assert result.returncode == 1
    assert "data.digest is required" in result.stderr

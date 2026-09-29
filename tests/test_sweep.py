"""The sweep file: what to try, on which data, for how long (#113).

The property that matters is not that a file parses. It is that two files which
mean the same sweep produce the same configuration ids, and that a file which
does not pin its data does not load at all — the first architecture search lost
the meaning of its whole ranking when the corpus behind it was deleted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from atr_training.sweep import (
    NOT_MEASURED,
    SweepError,
    config_id_for,
    effective_params,
    load_sweep,
)

EVAL_SHA = "b96679b559cdc861f1ec30cab1e1c2bc2f295c21cb222ff07ba1ad98ddd5f93f"

MINIMAL = {
    "name": "t",
    "engine": "kraken",
    "data": {"eval": {"name": "german-medieval-v1", "sha256": EVAL_SHA}},
    "budget": {"steps": 500},
    "base": {"batch_size": 256},
    "axes": {"lrate": [1e-4, 3e-4]},
}


def write(tmp_path: Path, doc: dict, name: str = "s.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def ids(sweep) -> list[str]:
    return [c.config_id for c in sweep.configs]


# ── the cross product ───────────────────────────────────────────────────────

def test_the_axes_are_expanded_to_their_cross_product(tmp_path):
    doc = {**MINIMAL, "axes": {"lrate": [1e-4, 3e-4], "augment": [False, True]}}
    sweep = load_sweep(write(tmp_path, doc))
    assert len(sweep.configs) == 4
    assert {(c.axes["lrate"], c.axes["augment"]) for c in sweep.configs} == {
        (1e-4, False), (1e-4, True), (3e-4, False), (3e-4, True)}


def test_a_single_valued_axis_is_a_constant_not_a_column(tmp_path):
    """Every row identical is a header, not a column."""
    doc = {**MINIMAL, "axes": {"lrate": [1e-4, 3e-4], "schedule": ["cosine"]}}
    sweep = load_sweep(write(tmp_path, doc))
    assert all(set(c.axes) == {"lrate"} for c in sweep.configs)
    assert all(c.params["schedule"] == "cosine" for c in sweep.configs)


def test_an_empty_axis_is_refused(tmp_path):
    doc = {**MINIMAL, "axes": {"lrate": []}}
    with pytest.raises(SweepError, match="no configurations"):
        load_sweep(write(tmp_path, doc))


# ── the identity ────────────────────────────────────────────────────────────

def test_two_files_that_mean_the_same_sweep_give_the_same_ids(tmp_path):
    """Whether a value is written in `base` or as a one-element axis, and in
    which order the axes appear, is a matter of style."""
    a = {**MINIMAL, "base": {"batch_size": 256, "schedule": "cosine"},
         "axes": {"lrate": [1e-4, 3e-4]}}
    b = {**MINIMAL, "base": {"schedule": "cosine"},
         "axes": {"batch_size": [256], "lrate": [1e-4, 3e-4]}}
    assert sorted(ids(load_sweep(write(tmp_path, a, "a.yaml")))) == \
           sorted(ids(load_sweep(write(tmp_path, b, "b.yaml"))))


def test_an_axis_written_at_its_default_does_not_rename_a_configuration(tmp_path):
    """The identity is over the *effective* configuration. Otherwise adding
    `normalization: [NFD, NFC]` — where NFD is already the default — would give
    the unchanged half new ids and split one row of the leaderboard into two."""
    before = load_sweep(write(tmp_path, MINIMAL, "before.yaml"))
    doc = {**MINIMAL, "axes": {**MINIMAL["axes"], "normalization": ["NFD", "NFC"]}}
    after = load_sweep(write(tmp_path, doc, "after.yaml"))
    assert set(ids(before)) < set(ids(after))          # the old two survive unchanged
    assert len(after.configs) == 4


def test_extending_the_search_space_does_not_rename_what_was_there(tmp_path):
    """A sweep runs for days and gets extended. An id tied to a position would
    make every configuration after the insertion a different one."""
    before = ids(load_sweep(write(tmp_path, MINIMAL, "before.yaml")))
    doc = {**MINIMAL, "axes": {"lrate": [3e-5, 1e-4, 3e-4]}}   # inserted in front
    after = ids(load_sweep(write(tmp_path, doc, "after.yaml")))
    assert set(before) < set(after)


def test_a_different_data_version_gives_different_ids(tmp_path):
    """Same hyperparameters on a different corpus is a different measurement and
    must not share a leaderboard row."""
    other = {**MINIMAL, "data": {"eval": {"name": "x", "sha256": "f" * 64}}}
    assert set(ids(load_sweep(write(tmp_path, MINIMAL, "a.yaml")))).isdisjoint(
        ids(load_sweep(write(tmp_path, other, "b.yaml"))))


def test_execution_detail_is_not_part_of_the_identity():
    """Moving a sweep to a machine with more workers must not rename everything."""
    digest = "d" * 64
    a = config_id_for({"lrate": 1e-4, "workers": 8}, digest, "kraken")
    b = config_id_for({"lrate": 1e-4, "workers": 2, "device": "cuda:1"}, digest, "kraken")
    assert a == b
    assert NOT_MEASURED == {"workers", "device"}


def test_the_seed_is_part_of_the_identity():
    """#115 varies exactly this to measure the noise floor; two seeds of one
    configuration are two measurements, not one."""
    digest = "d" * 64
    assert config_id_for({"seed": 42}, digest, "kraken") != \
           config_id_for({"seed": 43}, digest, "kraken")


# ── the refusals ────────────────────────────────────────────────────────────

def test_a_sweep_without_a_data_digest_is_refused(tmp_path):
    doc = {**MINIMAL, "data": {"eval": {"name": "german-medieval-v1"}}}
    with pytest.raises(SweepError, match="sha256"):
        load_sweep(write(tmp_path, doc))


def test_a_sweep_without_a_data_block_is_refused(tmp_path):
    doc = {k: v for k, v in MINIMAL.items() if k != "data"}
    with pytest.raises(SweepError, match="'data' is required"):
        load_sweep(write(tmp_path, doc))


def test_a_misspelled_axis_is_refused_rather_than_ignored(tmp_path):
    """pydantic drops unknown fields, so `lr` instead of `lrate` would train
    every configuration identically and the flat ranking would be reported as a
    finding."""
    doc = {**MINIMAL, "axes": {"lr": [1e-4, 3e-4]}}
    with pytest.raises(SweepError, match="not a parameter"):
        load_sweep(write(tmp_path, doc))


def test_an_invalid_value_is_refused_at_load_time_not_hours_later(tmp_path):
    doc = {**MINIMAL, "axes": {"batch_size": [256, 0]}}
    with pytest.raises(SweepError):
        load_sweep(write(tmp_path, doc))


def test_the_budget_is_required_and_counted_in_steps(tmp_path):
    doc = {k: v for k, v in MINIMAL.items() if k != "budget"}
    with pytest.raises(SweepError, match="optimizer"):
        load_sweep(write(tmp_path, doc))


def test_an_unknown_engine_is_refused(tmp_path):
    doc = {**MINIMAL, "engine": "tesseract"}
    with pytest.raises(SweepError, match="unknown engine"):
        load_sweep(write(tmp_path, doc))


# ── the ladder ──────────────────────────────────────────────────────────────

def test_the_ladder_is_counted_in_steps_not_epochs(tmp_path):
    """h256 costs ~12x h48 per step: a budget in epochs ranks by cost."""
    doc = {**MINIMAL, "axes": {"lrate": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]},
           "budget": {"steps": 500, "eta": 3}}
    sweep = load_sweep(write(tmp_path, doc))
    assert [(r.configs, r.steps) for r in sweep.rungs] == [(6, 500), (2, 1500), (1, 4500)]


def test_the_noise_floor_is_absent_until_it_is_measured(tmp_path):
    assert load_sweep(write(tmp_path, MINIMAL)).noise_floor is None
    doc = {**MINIMAL, "noise_floor": 0.03}
    assert load_sweep(write(tmp_path, doc, "f.yaml")).noise_floor == 0.03


# ── the shipped example must stay loadable ──────────────────────────────────

def test_the_example_in_the_repo_loads(tmp_path):
    """It documents the format; a format documented by a file that does not load
    is documentation of something else."""
    sweep = load_sweep(Path(__file__).resolve().parents[1] / "sweeps" / "example-kraken.yaml")
    assert len(sweep.configs) == 6
    assert sweep.data_digest and all(d.sha256 for d in sweep.data)
    assert effective_params(sweep.configs[0].params, "kraken")["schedule"] == "cosine"

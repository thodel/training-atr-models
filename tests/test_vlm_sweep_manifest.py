"""The first VLM sweep's manifest (#138, the last open part of #12).

It is a **draft on purpose**: `data.digest` and the four `revision`s need the
boxes and may not be guessed, so `parse_manifest` refuses the file as it stands.
This checks the design that does not depend on them, by parsing a copy with the
two filled in — so the file on disk cannot be run by accident and the decisions
in it are still pinned.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from atr_training.contracts import VlmTrainParams
from atr_training.sweep_manifest import ManifestError, parse_manifest

MANIFEST = (Path(__file__).resolve().parents[1] / "config" / "sweeps"
            / "vlm-xix-effective-batch-01.yaml")
RAW = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
#: A revision shape the parser accepts, used only to make the draft parseable
#: here. Deliberately not a real sha: a test that invented one would be the
#: first place someone copied it from.
FAKE_REV = "0" * 40


def runnable_copy() -> dict:
    raw = copy.deepcopy(RAW)
    raw["data"]["digest"] = "sha256:" + "0" * 64
    for spec in raw["data"]["datasets"]:
        spec["revision"] = FAKE_REV
    return raw


# ── it must not be runnable as it stands ─────────────────────────────────────
def test_the_file_on_disk_refuses_to_parse():
    """Absent on purpose. The digest changes when the revisions go in, so it has
    to be computed after them and before the first run — and a manifest carrying
    a plausible-looking placeholder is one somebody runs."""
    with pytest.raises(ManifestError):
        parse_manifest(RAW, source=str(MANIFEST))


def test_no_dataset_claims_a_revision_yet():
    assert all(spec.get("revision") is None for spec in RAW["data"]["datasets"])


def test_the_header_says_how_to_fill_both():
    text = MANIFEST.read_text(encoding="utf-8")
    assert "key_for_specs" in text, "the digest command"
    assert "dataset_info" in text, "the revision command"
    assert "UNPINNED_MAX_AGE_DAYS" in text, "why pinning is not housekeeping"


# ── the design ───────────────────────────────────────────────────────────────
def test_it_is_one_coupled_axis_and_three_cells():
    manifest = parse_manifest(runnable_copy())
    configs = manifest.configs()

    assert list(manifest.axes) == ["effective_batch"]
    assert len(configs) == 3


def test_the_effective_batch_doubles_and_the_rate_follows():
    cells = [c.params for c in parse_manifest(runnable_copy()).configs()]
    pairs = [(p["batch_size"] * p["accumulate_grad_batches"], p["lrate"]) for p in cells]

    assert pairs == [(16, 2.0e-4), (32, 4.0e-4), (64, 8.0e-4)]


def test_batch_size_is_held_constant_so_memory_is_not_the_variable():
    """Raising `batch_size` multiplies activation memory; raising the
    accumulation does not. A sweep about learning dynamics must not confound
    them, which is why only the accumulation moves."""
    cells = [c.params for c in parse_manifest(runnable_copy()).configs()]

    assert {p["batch_size"] for p in cells} == {16}
    assert sorted(p["accumulate_grad_batches"] for p in cells) == [1, 2, 4]


def test_cell_one_is_the_baselines_own_setting():
    """Without the baseline measured here, at this budget, on this corpus,
    "better" is a comparison across sweeps."""
    first = parse_manifest(runnable_copy()).configs()[0].params

    assert (first["batch_size"], first["accumulate_grad_batches"]) == (16, 1)
    assert first["lrate"] == 2.0e-4
    assert RAW["baseline"]["model"] == "qwen3vl-german-xix-v2"


def test_every_swept_and_fixed_parameter_is_one_the_engine_has():
    """The check that makes a sweep readable at all: a misspelled key is dropped
    in silence, every cell then trains identically, and the leaderboard ranks
    seed noise (#138)."""
    known = set(VlmTrainParams.model_fields)
    for key in RAW["base"]:
        assert key in known, key
    for value in RAW["axes"]["effective_batch"]:
        for key in value:
            assert key in known, key


def test_four_bit_is_fixed_and_off_rather_than_swept():
    """#137 measured bf16 as fitting one card and 22 % faster, so this is a
    setting and not a question."""
    assert RAW["base"]["load_in_4bit"] is False
    assert "load_in_4bit" not in RAW["axes"]


def test_the_baseline_names_its_metric_and_its_set():
    """Without both, the leaderboard compares this sweep's `cer` against a number
    measured elsewhere — the mistake #111 opens by naming."""
    baseline = RAW["baseline"]

    assert baseline["metric"] == "cer"
    assert baseline["measured_on"]
    assert baseline["draw_md5"] == "e04ca483", "the draw v2's 0.0533 came from"


def test_there_is_no_noise_floor_and_the_notes_say_so():
    """It is measured on a corpus and a budget and cannot exist before them."""
    assert "noise_floor" not in RAW
    assert "NO NOISE FLOOR YET" in RAW["notes"]
    assert "measure_noise_floor.py" in RAW["notes"]


def test_the_budget_has_a_ceiling():
    """A continuously running sweep without an upper bound is not a process, it
    is a leak (#117)."""
    assert RAW["budget"]["max_total_steps"] >= sum(
        n * s for n, s in zip(RAW["budget"]["rungs"], (3000, 9000)))

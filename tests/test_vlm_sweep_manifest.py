"""The first VLM sweep's manifest (#138, the last open part of #12).

It was a draft that refused to parse until the four `revision`s and the digest
could be read off the boxes — filled 12.10.2026, `pinned: True`. The tests that
held the draft shut now hold the pinning instead, because that is the property
worth keeping: an unpinned artefact expires after seven days, a three-rung
ladder runs longer, and rung 1 would then be compared against a validation
partition rebuilt between the rungs. The artefact these corpora were last
measured on expired between 03.10. and 12.10.2026 and took the sample set with
it, which is why this is checked and not trusted.
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


def runnable_copy() -> dict:
    """The manifest as it is. Kept as a function so the design tests below read
    the same whether the file is a draft or pinned."""
    return copy.deepcopy(RAW)


# ── it is pinned, and that is not housekeeping ───────────────────────────────
def test_the_file_on_disk_parses():
    assert parse_manifest(RAW, source=str(MANIFEST)).name == "vlm-xix-effective-batch-01"


@pytest.mark.parametrize("index", range(4))
def test_every_dataset_names_a_revision(index):
    """Without one the artefact is unpinned, expires after seven days, and a
    ladder that outlives it compares its rungs against different data."""
    revision = RAW["data"]["datasets"][index].get("revision")

    assert isinstance(revision, str) and len(revision) == 40, revision
    assert revision != "0" * 40, "a placeholder is worse than an absence"
    assert all(ch in "0123456789abcdef" for ch in revision), revision


def test_the_digest_is_a_sha256_and_not_a_placeholder():
    digest = RAW["data"]["digest"]

    assert digest.startswith("sha256:")
    assert len(digest.split(":", 1)[1]) == 64
    assert set(digest.split(":", 1)[1]) != {"0"}


def test_the_header_says_how_both_were_obtained():
    text = MANIFEST.read_text(encoding="utf-8")
    assert "key_for_specs" in text, "the digest command"
    assert "dataset_info" in text, "the revision command"
    assert "UNPINNED_MAX_AGE_DAYS" in text, "why pinning is not housekeeping"
    assert "pinned: True" in text, "what the computation reported"


def test_a_manifest_error_is_still_what_a_missing_digest_gets():
    """The draft refused to parse, and that guard must keep working — it is what
    stops a half-filled manifest from running."""
    without = copy.deepcopy(RAW)
    del without["data"]["digest"]
    with pytest.raises(ManifestError):
        parse_manifest(without)


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


def test_the_noise_floor_is_measured_and_says_on_what():
    """0.117 over four seeds of cell one at 3,000 steps, on ONE draw — the one
    every cell of this sweep is scored on (#138, 14.10.2026). The first
    measurement scored four seeds on four draws; the block has to name the draw
    or the number could be read as a floor it is not."""
    block = RAW["noise_floor"]
    assert block["measured_on"] == RAW["data"]["digest"]
    assert block["value"] == pytest.approx(max(block["values"]) - min(block["values"]))
    assert block["n"] == len(block["seeds"]) == 4
    # 3,000 steps at batch 16 = the 48,000 samples rung 0 gives every cell.
    assert block["steps"] * 16 == RAW["budget"]["samples"]
    assert block["eval_seed"] == RAW["base"]["seed"] == 42
    assert "md5" in block["draw"]
    m = parse_manifest(RAW, source=str(MANIFEST))
    assert m.noise_floor == pytest.approx(0.117)
    assert block["cell"] == m.configs()[0].config_id
    assert "NOISE FLOOR: 0.1170" in RAW["notes"]


def test_the_budget_has_a_ceiling():
    """A continuously running sweep without an upper bound is not a process, it
    is a leak (#117)."""
    samples = RAW["budget"]["samples"]
    assert RAW["budget"]["max_total_samples"] >= sum(
        n * samples * 3 ** rung for rung, n in enumerate(RAW["budget"]["rungs"]))


def test_the_budget_is_in_samples_because_the_batch_moves():
    """At equal steps the batch of 64 saw four times the lines of the batch of
    16, and the first rung 0 ranked that (#138, 17.10.2026)."""
    assert "steps" not in RAW["budget"]
    m = parse_manifest(RAW, source=str(MANIFEST))
    assert m.budget_unit == "samples"

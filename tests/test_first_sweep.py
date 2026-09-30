"""The first Kraken search space, as a manifest that actually loads (#118).

`config/sweeps/kraken-medieval-height-augment-lrate-01.yaml` is K6's deliverable:
three axes, twelve cells, and a reason for every axis left out. These tests hold
the parts of it that are claims rather than taste — the ones that would quietly
stop being true if the code moved underneath the file.

The finding the file rests on: `KRAKEN_PLUS_SPEC`, the default every kraken run
has used and the shape behind the only trustworthy German CER this project has
(0.2131), is height 64 — and `vgsl_geometry` puts it at 1.97 frames per
character, just under its own warn threshold. The geometry preflight has been
warning about the production configuration all along, which is why #118's
ordering (height last, "only if an axis is free") is inverted here.
"""

from pathlib import Path

import pytest
import yaml

from atr_training.contracts import KRAKEN_PLUS_SPEC, DatasetSpec, KrakenTrainParams
from atr_training.artefact_cache import key_for_specs
from atr_training.sweep_manifest import (
    LABEL_MAX, distinguishing, load_manifest,
)
from atr_training.vgsl_geometry import (
    FLOOR_REFUSE, FLOOR_WARN, check_line_geometry, frames_per_char, input_height,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "sweeps" / "kraken-medieval-height-augment-lrate-01.yaml"


@pytest.fixture(scope="module")
def sweep():
    return load_manifest(MANIFEST)


# ── the finding the search space rests on ───────────────────────────────────
def test_the_default_spec_is_height_64():
    """`kraken+` in ARCHITECTURE_SEARCH.md, and what every kraken run has used."""
    assert input_height(KRAKEN_PLUS_SPEC) == 64
    assert KrakenTrainParams().spec == KRAKEN_PLUS_SPEC


def test_the_default_spec_sits_under_the_geometry_guard_s_warn_threshold():
    """1.97 against a threshold of 2.0. Not refused — the guard has been warning
    about the production configuration, which is the argument for putting the
    height in the first sweep rather than the last."""
    frames = frames_per_char(KRAKEN_PLUS_SPEC)

    assert FLOOR_REFUSE < frames < FLOOR_WARN
    assert check_line_geometry(KRAKEN_PLUS_SPEC).severity == "warn"


# ── the manifest loads, and is what it says ─────────────────────────────────
def test_it_loads(sweep):
    assert sweep.name == "kraken-medieval-height-augment-lrate-01"
    assert sweep.engine == "kraken"


def test_rung_zero_is_twelve_cells(sweep):
    """#118: at most ~12 in rung 0. Five axes of three would be 243."""
    assert len(sweep.configs()) == 12
    assert sweep.rungs[0] == 12


def test_the_three_axes_are_height_augmentation_and_learning_rate(sweep):
    assert set(sweep.axes) == {"spec", "augment", "lrate"}


def test_the_heights_are_the_baseline_and_two_above_it(sweep):
    """h64 is the control: #111 asks for a configuration that beats
    kraken-medieval-german-v2 *on the same measurement set*, and without the
    baseline measured here, at this budget, "beats" is a comparison across
    sweeps."""
    heights = sorted(input_height(s) for s in sweep.axes["spec"])

    assert heights == [64, 128, 192]


def test_every_height_in_the_sweep_is_at_least_trainable(sweep):
    """h48 would be 1.48 frames/char. Nothing here is below the refusal floor,
    so no cell can be thrown out by the geometry guard for a reason the file
    already knew."""
    for spec in sweep.axes["spec"]:
        assert frames_per_char(spec) > FLOOR_REFUSE, spec


def test_the_two_taller_heights_clear_the_warn_threshold(sweep):
    for spec in sweep.axes["spec"]:
        if input_height(spec) > 64:
            assert frames_per_char(spec) >= FLOOR_WARN, spec


def test_augmentation_is_on_the_axis_with_the_historical_baseline_first(sweep):
    """The first architecture search ran `--no-augment` throughout
    (ARCHITECTURE_SEARCH.md), so `false` is the value everything so far was
    measured at — even though KrakenTrainParams has defaulted to True."""
    assert sweep.axes["augment"] == (False, True)
    assert sweep.base["augment"] is False
    assert KrakenTrainParams().augment is True


def test_the_lstm_width_is_not_an_axis(sweep):
    """Over both seeds: +0.0009 and +0.0006. Measured, and measured to do
    nothing — the one axis the first search settled."""
    widths = {s.count("Lbx") for s in sweep.axes["spec"]}
    assert widths == {3}, "the specs differ in more than the height"
    assert all("Lbx256" in s for s in sweep.axes["spec"])


def test_the_seed_is_fixed(sweep):
    """A sweep that varies the seed measures seed noise. Varying it is what #115
    does, on one configuration, on purpose."""
    assert "seed" not in sweep.axes
    assert sweep.base["seed"] == 42


def test_the_batch_size_is_fixed_because_it_rides_with_the_learning_rate(sweep):
    """Linear scaling ties them together, so they are one axis, not two."""
    assert "batch_size" not in sweep.axes
    assert sweep.base["batch_size"] == 256


# ── the budget ──────────────────────────────────────────────────────────────
def test_rung_zero_clears_the_convergence_floor(sweep):
    """No cell may be refused for being too short to converge — that would spend
    the sweep proving something the guard already knew."""
    from atr_training.convergence import floor_for

    assert sweep.steps >= floor_for("kraken", from_scratch=True)


def test_the_sweep_has_a_ceiling(sweep):
    """A continuously running sweep without an upper bound is a leak (#117)."""
    assert sweep.max_total_steps is not None


def test_the_ceiling_covers_the_ladder_with_room_to_spare(sweep):
    """12x4,000 + 4x12,000 + 1x36,000 = 132,000. Room above it is for preempted
    cells, which count against the ceiling because they cost the card."""
    from atr_training.sweep_driver import steps_at_rung

    planned = sum(width * steps_at_rung(sweep.steps, rung)
                  for rung, width in enumerate(sweep.rungs))
    assert planned == 132_000
    assert sweep.max_total_steps > planned


# ── the data version ────────────────────────────────────────────────────────
def test_the_datasets_are_the_four_medieval_corpora(sweep):
    assert {d["hf_repo"] for d in sweep.datasets} == {
        "dh-unibe/image-text_rats-und-richtebuecher_xv-xvi",
        "dh-unibe/image-text_bullinger-autoren",
        "dh-unibe/image-text_koenigsfelden-charters-post-1500",
        "dh-unibe/image-text_aaeb-xiv-xvii",
    }


def test_the_project_lists_are_the_ones_already_in_the_repo(sweep):
    """Copied from ubelix/specs/medieval-german-page-v1.json, not retyped. This
    fails if either side is edited alone, which is the point."""
    import json

    known = json.loads((ROOT / "ubelix" / "specs" / "medieval-german-page-v1.json")
                       .read_text(encoding="utf-8"))["datasets"]
    by_repo = {d["hf_repo"]: d for d in sweep.datasets}

    for spec in known:
        mine = by_repo[spec["hf_repo"]]
        assert mine.get("train_projects", []) == spec.get("train_projects", [])
        assert mine.get("all_projects", False) == spec.get("all_projects", False)


def test_the_digest_is_the_content_key_of_those_datasets(sweep):
    """Not a number typed in: the manifest's data version is what
    `artefact_cache` would compute for this corpus under the current held-out
    registry. Editing the projects, or the reserved documents, changes it."""
    expected = key_for_specs(
        [DatasetSpec.model_validate(d) for d in sweep.datasets], "kraken").digest

    assert sweep.data_digest == f"sha256:{expected}"


def test_the_corpus_is_not_pinned_and_the_file_says_so():
    """None of the four names a `revision`, so the digest identifies the specs
    and not the bytes behind them. That is a real weakness and it is written
    down rather than left to be discovered."""
    sweep = load_manifest(MANIFEST)
    key = key_for_specs([DatasetSpec.model_validate(d) for d in sweep.datasets], "kraken")

    assert key.pinned is False
    assert "NOT PINNED" in yaml.safe_load(MANIFEST.read_text())["notes"]


def test_no_noise_floor_is_claimed_yet(sweep):
    """It is measured on a corpus and a budget and cannot exist before them."""
    assert sweep.noise_floor is None


def test_the_notes_say_which_metric_to_run_it_with(sweep):
    """`benchmark_cer` cannot be filled here — german-medieval-v1 is drawn
    across all four corpora, so it is not the hf_repo + project a BenchmarkSpec
    needs. `cer` is comparable within this sweep because every cell shares one
    compiled artefact."""
    notes = yaml.safe_load(MANIFEST.read_text())["notes"]

    assert "--metric cer" in notes
    assert "BenchmarkSpec" in notes


def test_every_cell_would_be_sent_the_same_corpus(sweep):
    """Which is what makes `cer` comparable across them, and what makes the
    sweep affordable: one artefact-cache entry for all twelve."""
    keys = {key_for_specs([DatasetSpec.model_validate(d) for d in sweep.datasets],
                          "kraken").digest
            for _ in sweep.configs()}

    assert len(keys) == 1


# ── long axis values stay readable ──────────────────────────────────────────
def test_values_that_differ_in_one_place_are_shown_by_that_place():
    values = [f"[256,{h},0,1 Cr4,2,8,4,2 Lbx256 Do0.5 Cr255,1,85,1,1]"
              for h in (64, 128, 192)]

    labels = distinguishing(values)

    assert sorted(labels.values()) == ["…128…", "…192…", "…64…"]


def test_short_values_are_left_alone():
    assert distinguishing(["NFD", "NFC"]) == {"NFD": "NFD", "NFC": "NFC"}


def test_a_single_value_is_left_alone():
    long = "x" * (LABEL_MAX + 10)

    assert distinguishing([long]) == {long: long}


def test_values_that_differ_in_several_places_are_not_abbreviated():
    """An abbreviation that hides the difference is worse than a long cell."""
    a = "[256,64,0,1 Cr4,2,8,4,2 Lbx256 Do0.5]"
    b = "[256,128,0,1 Cr3,3,32,1,1 Lbx200 Do0.2]"

    assert distinguishing([a, b]) == {a: a, b: b}


def test_the_manifest_s_specs_print_as_their_heights(sweep):
    shown = {str(c) for c in sweep.configs()}

    assert any("spec=…64…" in s for s in shown), shown
    assert not any("Cr4,2,8,4,2" in s for s in shown), "the whole spec is still printed"


def test_the_leaderboard_uses_the_same_short_labels(sweep):
    from atr_training.leaderboard import rows_for
    from atr_training.sweep_driver import SweepState

    state = SweepState.for_manifest(sweep, "cer")
    for config in sweep.configs()[:3]:
        state.put(0, config.config_id, {"job_id": "j", "status": "completed",
                                        "score": 0.8, "raw": 0.2, "metric": "cer"})

    rows = rows_for(sweep, state)

    assert all(len(r.labels["spec"]) == 3 for r in rows)
    assert {r.labels["spec"][str(r.axes["spec"])] for r in rows} <= {"…64…", "…128…", "…192…"}

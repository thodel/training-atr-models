"""A noise floor belongs to the corpus it was measured on (#115).

`promote()` can be told what the material resolves, `measure_noise_floor.py`
measures it, and the manifest carries it. What was missing between those three is
everything that makes the number *attributable*, and #115 asks for it in as many
words: "min/Mittel/max über N Seeds derselben Konfiguration, mit Seeds, Commit
und Datendigest notiert" — and "die Zahl steht in der Sweep-Datei und auf dem
Leaderboard, nicht nur in diesem Issue".

Three gaps, and they share one failure. #115's own table has one configuration
spread 0.0085 across two seeds while another moved 0.1924: a floor is a property
of a corpus *and* a budget. A number without its provenance can be carried from
other material, and then a ranking that the material cannot support is published
with a mark saying it can.

  * a floor could be written as a bare number with no way to say where it came
    from, and nothing checked it against the sweep's own data version;
  * `boundary_margin` and `decided_within_noise` never reached the state file, so
    the mark lived in a log line and #116 had nothing to read;
  * the measurement recorded neither the commit nor the data digest.
"""

import json
from pathlib import Path

import pytest
import yaml

from atr_training.convergence import epochs_for as convergence_epochs_for
from atr_training.rungs import promote
from atr_training.sweep_driver import SweepDriver, SweepState, epochs_for
from atr_training.sweep_manifest import ManifestError, parse_manifest

from tests.test_sweep_driver import FakeTrainer, scores_for

DIGEST = "sha256:0123456789abcdef"

MANIFEST = f"""
name: kraken-medieval-augment-01
data:
  train: shard_00
  eval: german_test
  digest: "{DIGEST}"
  datasets:
    - {{hf_repo: "dh-unibe/x", granularity: line}}
budget:
  steps: 4000
  rungs: [12, 4, 1]
base:
  spec: "[1,48,0,1 Cr3,3,32]"
  lrate: 1.0e-4
  batch_size: 256
  accumulate_grad_batches: 1
  augment: false
  normalization: NFD
axes:
  augment: [false, true]
  normalization: [NFD, NFC]
  lrate: [3.0e-5, 1.0e-4, 3.0e-4]
"""


def manifest(**edits):
    raw = yaml.safe_load(MANIFEST)
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


MEASURED = {
    "value": 0.0085,
    "measured_on": DIGEST,
    "seeds": [42, 43, 44, 45],
    "commit": "1a429b3",
    "steps": 2000,
}


# ── the floor says what it was measured on ──────────────────────────────────
def test_a_measured_floor_is_read_with_its_provenance():
    m = manifest(**{"noise_floor": MEASURED})

    assert m.noise_floor == pytest.approx(0.0085)
    assert m.noise_floor_provenance["seeds"] == [42, 43, 44, 45]
    assert m.noise_floor_provenance["commit"] == "1a429b3"
    assert m.noise_floor_provenance["provenance"] == "measured"


def test_a_floor_measured_on_other_material_is_refused():
    """The failure the block exists to prevent. #115's own table: one
    configuration spread 0.0085 over two seeds, another moved 0.1924 — a floor
    is a property of the corpus and the budget, not a constant."""
    other = {**MEASURED, "measured_on": "sha256:ffffffffffffffff"}

    with pytest.raises(ManifestError, match="never earned"):
        manifest(**{"noise_floor": other})


def test_a_bare_number_is_still_accepted_but_marked_unstated():
    """The field was defined as a number and that stays true. What changes is
    that a reader can tell it from one somebody measured."""
    m = manifest(**{"noise_floor": 0.0085})

    assert m.noise_floor == pytest.approx(0.0085)
    assert m.noise_floor_provenance == {"provenance": "unstated"}


def test_a_block_without_a_value_is_refused():
    with pytest.raises(ManifestError, match="without a `value`"):
        manifest(**{"noise_floor": {"measured_on": DIGEST, "seeds": [42]}})


def test_a_zero_floor_is_still_refused_in_block_form():
    """It would mark every cut as decided outside the noise."""
    with pytest.raises(ManifestError, match="greater than zero"):
        manifest(**{"noise_floor": {**MEASURED, "value": 0}})


def test_a_manifest_without_a_floor_has_no_provenance_either():
    """Absent stays absent: a floor cannot exist before the corpus it is
    measured on."""
    m = manifest()

    assert m.noise_floor is None
    assert m.noise_floor_provenance == {}


# ── the decision reaches the record ─────────────────────────────────────────
def _within_noise_state(tmp_path: Path, floor):
    """A sweep whose rung-0 cut falls inside the floor."""
    m = manifest(**{"noise_floor": floor})
    # Twelve configurations within 0.0001 of each other: any cut is noise.
    cers = [0.20 + i / 100000 for i in range(12)]
    trainer = FakeTrainer(scores_for(m, cers))
    state = SweepState.for_manifest(m, "benchmark_cer", tmp_path / "s.json")
    state.train_lines = 120_000
    SweepDriver(m, state, trainer, sleep=lambda _s: None).run()
    return json.loads((tmp_path / "s.json").read_text())


def test_a_cut_inside_the_noise_is_recorded_as_such(tmp_path: Path):
    """#115's second criterion. Without this the mark lives in a log line that
    scrolls past and #116 has nothing to print."""
    recorded = _within_noise_state(tmp_path, MEASURED)

    rung0 = recorded["promotions"][0]
    assert rung0["decided_within_noise"] is True
    assert rung0["boundary_margin"] < 0.0085
    assert rung0["noise_floor"] == pytest.approx(0.0085)


def test_the_margin_is_recorded_even_when_no_floor_is_known(tmp_path: Path):
    """It is the number the whole ladder turns on, and a later measurement can
    be held against a sweep that ran before it."""
    recorded = _within_noise_state(tmp_path, None)

    rung0 = recorded["promotions"][0]
    assert rung0["boundary_margin"] is not None
    assert rung0["decided_within_noise"] is False
    assert rung0["noise_floor"] is None


def test_a_clear_cut_is_not_marked(tmp_path: Path):
    m = manifest(**{"noise_floor": MEASURED})
    cers = [0.05, 0.06, 0.07, 0.08] + [0.60 + i / 100 for i in range(8)]
    trainer = FakeTrainer(scores_for(m, cers))
    state = SweepState.for_manifest(m, "benchmark_cer", tmp_path / "s.json")
    state.train_lines = 120_000
    SweepDriver(m, state, trainer, sleep=lambda _s: None).run()

    recorded = json.loads((tmp_path / "s.json").read_text())
    assert recorded["promotions"][0]["decided_within_noise"] is False


def test_the_provenance_travels_into_the_state_file(tmp_path: Path):
    """So a reader of the record never has to trust a bare number."""
    recorded = _within_noise_state(tmp_path, MEASURED)

    assert recorded["noise_floor_provenance"]["commit"] == "1a429b3"
    assert recorded["noise_floor_provenance"]["measured_on"] == DIGEST


def test_an_unstated_floor_says_so_in_the_state_file(tmp_path: Path):
    recorded = _within_noise_state(tmp_path, 0.0085)

    assert recorded["noise_floor_provenance"] == {"provenance": "unstated"}


def test_a_promotion_still_happens_inside_the_noise():
    """A rung has to narrow. Refusing to cut on a tight field would stall the
    sweep on exactly the material where every configuration is similar — the
    floor changes what the record claims, not who advances."""
    scores = {f"c{i}": 0.80 + i / 100000 for i in range(12)}

    decision = promote(scores, keep=4, rung=0, noise_floor=0.0085)

    assert len(decision.promoted) == 4
    assert decision.decided_within_noise is True


# ── one formula ─────────────────────────────────────────────────────────────
def test_the_driver_and_the_measurement_share_one_formula():
    """Both docstrings warn that two formulas for one quantity is how the first
    sweep handed its large configurations a quarter of their budget — and there
    were two functions named `epochs_for`, one of them unguarded."""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "mnf", root / "scripts" / "measure_noise_floor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.epochs_for is convergence_epochs_for
    assert epochs_for(4000, 120_000, 256) == convergence_epochs_for(4000, 120_000, 256)


def test_the_shared_formula_refuses_nonsense():
    with pytest.raises(ValueError):
        convergence_epochs_for(0, 120_000, 256)
    with pytest.raises(ValueError):
        convergence_epochs_for(4000, 0, 256)


# ── writing the measurement into a manifest ─────────────────────────────────
def _script():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "mnf", root / "scripts" / "measure_noise_floor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUMMARY = {
    "data_digest": DIGEST, "commit": "1a429b3", "seeds": [42, 43, 44, 45],
    "steps": 2000, "measured_at": "2026-09-30T04:00:00+00:00",
    "noise_floor": {"min": 0.20, "max": 0.2085, "mean": 0.204,
                    "spread": 0.0085, "stdev": 0.004, "n": 4},
}


def test_the_measurement_writes_itself_into_the_manifest(tmp_path: Path):
    """#115: the number belongs in the sweep file. Retyping it is how it ends up
    stale or attached to the wrong corpus."""
    path = tmp_path / "sweep.yaml"
    path.write_text(MANIFEST, encoding="utf-8")

    _script().write_into_manifest(path, SUMMARY)

    written = parse_manifest(yaml.safe_load(path.read_text()), source=str(path))
    assert written.noise_floor == pytest.approx(0.0085)
    assert written.noise_floor_provenance["seeds"] == [42, 43, 44, 45]
    assert written.noise_floor_provenance["provenance"] == "measured"


def test_it_refuses_a_manifest_that_runs_on_other_data(tmp_path: Path):
    path = tmp_path / "sweep.yaml"
    raw = yaml.safe_load(MANIFEST)
    raw["data"]["digest"] = "sha256:ffffffffffffffff"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="never earned"):
        _script().write_into_manifest(path, SUMMARY)


def test_the_written_manifest_still_parses_as_one(tmp_path: Path):
    """A round trip through yaml.safe_dump must not lose a field — the manifest
    refuses unknown keys and would refuse a mangled one outright."""
    path = tmp_path / "sweep.yaml"
    path.write_text(MANIFEST, encoding="utf-8")
    before = parse_manifest(yaml.safe_load(MANIFEST), source="before")

    _script().write_into_manifest(path, SUMMARY)
    after = parse_manifest(yaml.safe_load(path.read_text()), source=str(path))

    assert [c.config_id for c in after.configs()] == [c.config_id for c in before.configs()]
    assert after.datasets == before.datasets
    assert after.rungs == before.rungs


def test_the_spread_is_what_is_written_not_the_stdev(tmp_path: Path):
    """The floor is the full spread over the seeds. A stdev over four runs is a
    smaller number describing the same data, and the ladder would then treat
    differences the material cannot resolve as real."""
    path = tmp_path / "sweep.yaml"
    path.write_text(MANIFEST, encoding="utf-8")

    _script().write_into_manifest(path, SUMMARY)

    raw = yaml.safe_load(path.read_text())
    assert raw["noise_floor"]["value"] == pytest.approx(SUMMARY["noise_floor"]["spread"])

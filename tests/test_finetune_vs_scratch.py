"""Fine-tune or from scratch — the axis #118 names and nobody had tried.

#118 lists "Feintuning statt von Null" among the untried axes and says why it
comes before the rest of the ladder:

    Ein Feintuning einer passenden Basis kann jede Architekturvariante von Null
    schlagen, und dann ist der ganze Suchraum die falsche Frage.

Two cells answer that, and they could not be written before: `base_model` was a
sweep-wide key, so one manifest meant one base, and the two arms would have been
two files with no shared leaderboard. It is an axis now.

Three things about making it one are claims rather than taste, and each is a
failure that would have been silent:

* **It belongs in `config_id`.** A fine-tune of CATMuS and a run from scratch
  with the same hyperparameters would otherwise have shared an id, and the
  second result would have overwritten the first.
* **It is not a `ketos` parameter.** It is a request field, so it is swept and
  shown and never passed as one.
* **`spec` and `base_model` cannot both be axes.** `ketos train` ignores
  `--spec` when `--load` is given, so a fine-tune cell labelled h192 would have
  trained at whatever height the base has — and the leaderboard would carry a
  column of heights half its rows never used.

And one thing about the sweep itself: a fine-tune inherits its base's training
data, so a base that saw the held-out pages leaks the way #100 did, one step
further back where nobody looks. The registry is the only place that could say,
and for this base it says nothing — which is reported as nothing.
"""

from pathlib import Path

import pytest
import yaml

from atr_training.artefact_cache import key_for_specs
from atr_training.base_models import Provenance, provenance
from atr_training.contracts import DatasetSpec
from atr_training.shared_registry import BaseEntry, SharedRegistry
from atr_training.sweep_manifest import (
    BASE_AXIS,
    ManifestError,
    config_id,
    load_manifest,
    parse_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
SWEEPS = ROOT / "config" / "sweeps"
SCRATCH_VS_TUNE = SWEEPS / "kraken-medieval-finetune-vs-scratch-02.yaml"
FIRST = SWEEPS / "kraken-medieval-height-augment-lrate-01.yaml"
BASE = "kraken-catmus_medieval"


@pytest.fixture(scope="module")
def sweep():
    return load_manifest(SCRATCH_VS_TUNE)


@pytest.fixture(scope="module")
def first():
    return load_manifest(FIRST)


def minimal(**over) -> dict:
    raw = {
        "name": "t",
        "data": {"train": "x", "eval": "y", "digest": "sha256:abc",
                 "datasets": [{"hf_repo": "a/b"}]},
        "budget": {"steps": 1000, "rungs": [2, 1]},
        "base": {"lrate": 1.0e-4, "spec": "[1,64,0,1]"},
        "axes": {BASE_AXIS: [None, BASE]},
    }
    raw.update(over)
    return raw


# ── the manifest ────────────────────────────────────────────────────────────
def test_it_loads_as_two_cells(sweep):
    assert len(sweep.configs()) == 2


def test_the_only_axis_is_what_it_starts_from(sweep):
    assert list(sweep.axes) == [BASE_AXIS]
    assert sweep.axes[BASE_AXIS] == (None, BASE)


def test_one_cell_trains_from_scratch_and_one_fine_tunes(sweep):
    assert sorted(c.base_model or "" for c in sweep.configs()) == ["", BASE]


def test_the_two_cells_are_two_configurations(sweep):
    """The collision this change exists to avoid: same hyperparameters, two
    experiments. Before `base_model` entered the id they shared one, and the
    second result would have overwritten the first."""
    ids = [c.config_id for c in sweep.configs()]

    assert len(set(ids)) == 2


def test_the_base_is_not_passed_as_a_ketos_parameter(sweep):
    """It is a request field. A parameter named base_model would reach
    `KrakenTrainParams`, which has no such field."""
    for config in sweep.configs():
        assert BASE_AXIS not in config.params


def test_the_base_is_still_a_leaderboard_column(sweep):
    """Swept and shown: `axes` is what the table puts in its columns."""
    for config in sweep.configs():
        assert BASE_AXIS in config.axes


def test_from_scratch_reads_as_from_scratch_not_as_none(sweep):
    """A column reading `None` invites the reader to wonder what went missing."""
    scratch = next(c for c in sweep.configs() if c.base_model is None)

    assert "from scratch" in str(scratch)
    assert "None" not in str(scratch)


def test_the_ladder_narrows_to_one(sweep):
    assert sweep.rungs == (2, 1)
    assert sweep.max_total_steps == 30000


def test_rung_zero_clears_the_from_scratch_convergence_floor(sweep):
    """The scratch cell is the strict one: 4,000 against a floor of 2,000. The
    fine-tune's floor is 500, so one budget serves both."""
    from atr_training.convergence import FLOOR_FINETUNE, FLOOR_FROM_SCRATCH

    assert sweep.steps > FLOOR_FROM_SCRATCH > FLOOR_FINETUNE


# ── the comparison is only legitimate on the same data ──────────────────────
def test_the_data_block_is_the_one_from_sweep_one(sweep, first):
    """Copied, not retyped. Both sweeps then hit one entry of the artefact cache
    and share one validation partition, which is the whole of why `cer` is
    comparable between them."""
    assert sweep.data_digest == first.data_digest
    assert sweep.datasets == first.datasets


def test_changing_either_file_alone_fails_this(sweep, first):
    """The point of the test above, stated as what it catches."""
    mine = yaml.safe_load(SCRATCH_VS_TUNE.read_text(encoding="utf-8"))
    theirs = yaml.safe_load(FIRST.read_text(encoding="utf-8"))

    assert mine["data"]["datasets"] == theirs["data"]["datasets"]
    assert mine["data"]["digest"] == theirs["data"]["digest"]


def test_the_digest_is_the_content_key_of_those_datasets(sweep):
    specs = [DatasetSpec.model_validate(dict(s)) for s in sweep.datasets]

    assert sweep.data_digest == f"sha256:{key_for_specs(specs, 'kraken').digest}"


def test_the_corpus_is_still_not_pinned_and_the_file_says_so(sweep):
    specs = [DatasetSpec.model_validate(dict(s)) for s in sweep.datasets]

    assert key_for_specs(specs, "kraken").pinned is False
    assert "NOT PINNED" in (sweep.source and SCRATCH_VS_TUNE.read_text(encoding="utf-8"))


def test_no_noise_floor_is_claimed(sweep):
    """For a two-cell question that is less fatal than for a ranking, and the
    file says why rather than assuming it."""
    assert sweep.noise_floor is None
    assert "NO NOISE FLOOR YET" in SCRATCH_VS_TUNE.read_text(encoding="utf-8")


def test_resize_is_union_because_a_base_codec_will_not_cover_four_corpora(sweep):
    """`fail` would stop the run at exactly that discovery. It is inert for the
    scratch cell: ketos_cmd emits --resize only alongside --load."""
    for config in sweep.configs():
        assert config.params["resize"] == "union"


def test_the_scratch_cell_is_its_own_control_not_sweep_one_s(sweep, first):
    """Deliberately a different configuration: `resize: union` is set here and
    not there. Sweep 1 wrote the rule for its own h64 control — a comparison
    across sweeps is what the noise-floor work forbids — so both arms run here.
    """
    assert {c.config_id for c in sweep.configs()}.isdisjoint(
        {c.config_id for c in first.configs()})


# ── the two ids that must not have moved ────────────────────────────────────
def test_sweep_one_s_ids_are_unchanged_by_all_this(first):
    """`base_model` is omitted from the payload when there is none, so every id
    written before it could be swept still resolves to itself. A sweep resumed
    after this change must not re-run its whole field."""
    for config in first.configs():
        assert config.config_id == config_id(config.params,
                                             data_digest=first.data_digest)


def test_an_absent_base_and_an_explicit_none_are_one_configuration():
    """They mean the same thing to the trainer — train from scratch — so
    collapsing them is the right answer and not a silent one."""
    params = {"lrate": 1.0e-4}

    assert (config_id(params, data_digest="d")
            == config_id(params, data_digest="d", base_model=None))


def test_a_base_changes_the_id():
    params = {"lrate": 1.0e-4}

    assert (config_id(params, data_digest="d", base_model=BASE)
            != config_id(params, data_digest="d"))


# ── the combination that would print a lie ──────────────────────────────────
def test_spec_and_base_model_cannot_both_be_axes():
    raw = minimal(axes={BASE_AXIS: [None, BASE], "spec": ["[1,64,0,1]", "[1,192,0,1]"]},
                  budget={"steps": 1000, "rungs": [4, 1]})

    with pytest.raises(ManifestError, match="ignores --spec when --load"):
        parse_manifest(raw)


def test_spec_alone_is_still_an_axis():
    raw = minimal(axes={"spec": ["[1,64,0,1]", "[1,192,0,1]"]})

    assert len(parse_manifest(raw).configs()) == 2


def test_the_base_axis_needs_no_entry_in_base():
    """Every other axis must have a declared default in `base`; this one's
    default is the manifest's top-level `base_model`, which may be absent."""
    assert BASE_AXIS not in parse_manifest(minimal()).base


def test_a_top_level_base_is_the_default_when_it_is_not_an_axis():
    raw = minimal(axes={"lrate": [1.0e-4, 3.0e-4]}, base_model=BASE)

    assert {c.base_model for c in parse_manifest(raw).configs()} == {BASE}


def test_two_identical_base_values_are_refused():
    with pytest.raises(ManifestError, match="same value"):
        parse_manifest(minimal(axes={BASE_AXIS: [BASE, BASE]}))


# ── the request each cell is submitted with ─────────────────────────────────
def built(manifest, config):
    from tests.test_sweep_driver import FakeTrainer, driver

    return driver(FakeTrainer(), m=manifest).request_for(config, rung=0)


def test_each_cell_is_submitted_with_its_own_base(sweep):
    bodies = {c.base_model: built(sweep, c)["base_model"] for c in sweep.configs()}

    assert bodies == {None: None, BASE: BASE}


def test_the_scratch_cell_does_not_fall_back_to_the_sweep_s_base():
    """The failure mode of taking `base_model` from the manifest: a cell that
    chose `null` would fine-tune anyway, and the leaderboard would show two
    fine-tunes under two names."""
    manifest = parse_manifest(minimal(base_model=BASE))
    scratch = next(c for c in manifest.configs() if c.base_model is None)

    assert built(manifest, scratch)["base_model"] is None


def test_both_cells_are_sent_the_same_corpus(sweep):
    bodies = [built(sweep, c) for c in sweep.configs()]

    assert bodies[0]["datasets"] == bodies[1]["datasets"]


# ── what the base has already seen (#100) ───────────────────────────────────
def registry(**fields) -> SharedRegistry:
    return SharedRegistry([BaseEntry(id=BASE, engine="kraken",
                                     zenodo_id="10.5281/zenodo.7516057", **fields)])


def test_a_run_from_scratch_has_no_inherited_leak():
    assert provenance(None) is None
    assert provenance("") is None


def test_a_recorded_base_is_checkable():
    got = provenance(BASE, registry(training_datasets=["ds/one", "ds/two"]))

    assert got.state == "recorded"
    assert got.leak_checkable
    assert got.datasets == ("ds/one", "ds/two")


def test_an_unrecorded_base_is_not_called_clean():
    """The honest reading of every entry written before #100 added the field —
    and the state #100 itself came out of."""
    got = provenance(BASE, registry())

    assert got.state == "unrecorded"
    assert not got.leak_checkable
    assert "cannot be said not to have seen" in got.describe()


def test_no_registry_is_not_the_same_as_no_record():
    """Nothing was looked up. Reporting that as "unrecorded" would turn an
    unreadable file into a statement about the model."""
    got = provenance(BASE, None, "the share is not mounted")

    assert got.state == "unknown"
    assert "the share is not mounted" in got.describe()


def test_an_id_the_registry_does_not_have_is_unknown():
    got = provenance("kraken-nothing-like-it", registry())

    assert got.state == "unknown"
    assert "DOI or a path" in got.describe()


def test_the_registry_carries_training_datasets_at_all():
    """`BaseEntry` ignores unknown fields, so the field has to be declared or
    `provenance` would report every entry as unrecorded."""
    entry = BaseEntry(id="x", engine="kraken", training_datasets=["ds/one"])

    assert entry.training_datasets == ["ds/one"]


def test_this_sweep_s_base_is_the_unrecorded_case_today(sweep):
    """Stated as a test so it stops being true loudly, rather than quietly,
    once somebody records it."""
    assert Provenance(base=BASE, state="unrecorded").state == "unrecorded"
    assert any(c.base_model == BASE for c in sweep.configs())


# ── the report in front of the GPU time ─────────────────────────────────────
def test_check_sweep_reports_the_base_s_provenance(capsys):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_sweep import main

    assert main([str(SCRATCH_VS_TUNE)]) == 0

    printed = capsys.readouterr().out
    assert "bases this sweep fine-tunes from" in printed
    assert BASE in printed
    assert "#100" in printed


def test_check_sweep_says_nothing_about_bases_for_a_from_scratch_sweep(capsys):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_sweep import main

    assert main([str(FIRST)]) == 0

    assert "bases this sweep fine-tunes from" not in capsys.readouterr().out


def test_the_listing_names_what_each_cell_starts_from(capsys):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_sweep import main

    main([str(SCRATCH_VS_TUNE)])

    printed = capsys.readouterr().out
    assert "base    an axis: from scratch, kraken-catmus_medieval" in printed


# ── the written file and the planned order must not drift ───────────────────
PLAN = ROOT / "docs" / "ARCHITECTURE_SEARCH.md"


def test_the_plan_names_this_manifest():
    """#118's second done-when is that the order of the further sweeps is
    written down. A plan that names a file which does not exist, or a file the
    plan does not know about, is the drift this catches."""
    plan = PLAN.read_text(encoding="utf-8")

    assert SCRATCH_VS_TUNE.name in plan
    assert FIRST.name in plan


def test_the_plan_puts_the_fine_tune_question_second():
    """It was sweep 3 with a sentence recommending it be pulled forward. The
    recommendation is acted on, so the order has to say so — otherwise the next
    reader runs pre-processing next, which is what the sentence warned about."""
    plan = PLAN.read_text(encoding="utf-8")
    second = plan.index("### Sweep 2 —")
    third = plan.index("### Sweep 3 —")

    assert "fine-tuning against from-scratch" in plan[second:third]
    assert "pre-processing" in plan[third:]


def test_the_plan_records_why_this_base_and_not_a_matching_one():
    """The decision a later reader will question first: why a Caroline base for
    German chancery hands."""
    plan = PLAN.read_text(encoding="utf-8")

    assert BASE in plan
    assert "#101" in plan

"""Batch size and learning rate as ONE axis, and a typo that cannot be ignored (#138).

#138 asks for them as one axis and says why: linear scaling ties them, so varying
them separately "erzeugt eine Rangfolge der Rechenzeit" — a larger batch at an
unchanged rate simply takes fewer steps over the same data, and the ranking is of
compute, not of learning. The manifest's cross product mapped one axis name to
one parameter, so that could not be written down. A coupled axis carries a
mapping of several parameters as one value.

The second half is the hazard that made this worth doing carefully. The params
classes ignore unknown fields, so a misspelled key is dropped in silence. For one
run that is one wrong run; for a sweep it is worse, because the misspelling still
changes the `config_id` — every cell trains identically, each gets its own row,
and the leaderboard ranks seed noise as an effect. A confident ordering of
nothing is the one output a sweep must never produce.
"""

from __future__ import annotations

import pytest

from atr_training.sweep_manifest import ManifestError, params_fields, parse_manifest

DATA = {"train": "t", "eval": "e", "digest": "sha256:" + "0" * 64,
        "datasets": [{"hf_repo": "dh-unibe/image-text_x", "granularity": "line"}]}
COUPLED = [{"batch_size": 1, "accumulate_grad_batches": 16, "lrate": 2.0e-4},
           {"batch_size": 2, "accumulate_grad_batches": 16, "lrate": 4.0e-4},
           {"batch_size": 4, "accumulate_grad_batches": 16, "lrate": 8.0e-4}]


def manifest(**over):
    raw = {"name": "probe", "engine": "vllm", "data": DATA,
           "budget": {"steps": 1000, "rungs": [3, 1]},
           "base": {"batch_size": 1, "accumulate_grad_batches": 16, "lrate": 2.0e-4},
           "axes": {"effective_batch": COUPLED}}
    raw.update(over)
    return parse_manifest(raw)


# ── one axis, several parameters ─────────────────────────────────────────────
def test_a_coupled_axis_is_one_dimension_not_three():
    """Three parameters, three cells — not twenty-seven."""
    assert len(manifest().configs()) == 3


def test_the_parameters_move_together():
    cells = [c.params for c in manifest().configs()]
    effective = [(p["batch_size"] * p["accumulate_grad_batches"], p["lrate"])
                 for p in cells]

    assert effective == [(16, 2.0e-4), (32, 4.0e-4), (64, 8.0e-4)]
    # Linear scaling: the rate follows the effective batch, which is the whole
    # reason these are one axis.
    for (batch, rate), (batch2, rate2) in zip(effective, effective[1:]):
        assert pytest.approx(rate2 / rate) == batch2 / batch


def test_the_axis_name_is_a_label_and_never_a_parameter():
    """`effective_batch` is not a field of anything — it must not reach the
    request, or the trainer would be handed a parameter named after the axis."""
    for config in manifest().configs():
        assert "effective_batch" not in config.params
        assert config.axes["effective_batch"]["batch_size"] in (1, 2, 4)


def test_each_cell_still_gets_its_own_id():
    ids = {c.config_id for c in manifest().configs()}
    assert len(ids) == 3


def test_a_coupled_axis_needs_its_keys_declared_in_base_not_its_name():
    """The old rule asked for `base.<axis name>`, which for a label is a default
    for something no engine has."""
    with pytest.raises(ManifestError, match="accumulate_grad_batches"):
        manifest(base={"batch_size": 1, "lrate": 2.0e-4})      # accumulate missing

    assert manifest().configs(), "with all three declared it parses"


# ── a key the engine does not have ───────────────────────────────────────────
def test_a_misspelled_axis_is_refused():
    with pytest.raises(ManifestError, match="axes.btach_size"):
        manifest(base={"btach_size": 1, "lrate": 2.0e-4},
                 axes={"btach_size": [1, 4]})


def test_a_misspelling_consistent_between_base_and_axes_is_still_refused():
    """The dangerous one: spelled the same in both places, the base-entry rule is
    satisfied and only a field check can see it. Without this, every cell trains
    with the default batch size and the leaderboard ranks the seed."""
    with pytest.raises(ManifestError, match="not a parameter of engine 'vllm'"):
        manifest(base={"btach_size": 1, "lrate": 2.0e-4},
                 axes={"btach_size": [1, 4]})


def test_a_misspelled_key_inside_a_coupled_axis_is_refused():
    with pytest.raises(ManifestError, match="axes.eb.lrate_"):
        manifest(base={"batch_size": 1, "lrate_": 1.0e-4},
                 axes={"eb": [{"batch_size": 1, "lrate_": 1.0e-4},
                              {"batch_size": 2, "lrate_": 2.0e-4}]})


def test_a_misspelled_base_key_is_refused_even_without_an_axis():
    with pytest.raises(ManifestError, match="base.wokers"):
        manifest(base={"batch_size": 1, "accumulate_grad_batches": 16,
                       "lrate": 2.0e-4, "wokers": 4},
                 axes={})


def test_base_model_is_a_request_field_and_not_judged_as_a_parameter():
    """It is swept and shown but never passed as a parameter, so it must not be
    held against the params class."""
    got = manifest(axes={"base_model": ["Qwen/Qwen3-VL-4B-Instruct", None]},
                   budget={"steps": 1000, "rungs": [2, 1]})
    assert len(got.configs()) == 2


@pytest.mark.parametrize("engine,field", [("vllm", "lora_r"), ("kraken", "spec")])
def test_the_whitelist_is_the_engines_own_parameter_class(engine, field):
    known = params_fields(engine)
    assert known is not None and field in known


def test_an_engine_without_a_parameter_class_is_unchecked_rather_than_refused():
    """A new engine must not be blocked by this check; it is simply not covered,
    and `params_fields` says so by returning None."""
    assert params_fields("something-new") is None

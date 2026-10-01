"""#111's acceptance sentence, as something the table can answer.

    … für mindestens einen Suchraum steht eine Kraken-Konfiguration, die
    `kraken-medieval-german-v2` auf demselben Messsatz um mehr als dieses
    Rauschmass schlägt — **oder es ist belegt, dass keine der geprüften das
    tut.**

The second half is the one nobody builds. A sweep that finds nothing is a
result, and only if it is written down as one; otherwise it reads as a sweep
somebody abandoned. So the leaderboard now carries a verdict, and it refuses
before it compares — twice, because both refusals are the epic's own opening
argument:

* it holds 0.2131 against 0.111 and 0.0680 and says the comparison is **not**
  established, because the measurement sets differ;
* and a sweep ranking on `cer` — its own validation partition, which overlaps
  the training projects — against a baseline measured as `benchmark_cer` over
  200 unseen documents is that same mistake one level down. Which is the state
  both written sweeps are actually in, so the verdict says so rather than
  printing a difference nobody can read.

The rest is the chain the epic asks for, end to end and in one test: a manifest
in, the driver over every rung, a rendered leaderboard out. The modules each
have their own suite; what this adds is that they meet.
"""

import json
from pathlib import Path

import pytest
import yaml

from atr_training.leaderboard import (
    BEATS,
    DOES_NOT_BEAT,
    INSIDE_NOISE,
    METRIC_MISMATCH,
    NO_BASELINE,
    NO_FLOOR,
    NOTHING_SCORED,
    judge,
    render,
    rows_for,
)
from atr_training.sweep_driver import SweepDriver, SweepState
from atr_training.sweep_manifest import ManifestError, load_manifest, parse_manifest

from tests.test_sweep_driver import FakeTrainer

ROOT = Path(__file__).resolve().parents[1]
SWEEPS = ROOT / "config" / "sweeps"
WRITTEN = sorted(SWEEPS.glob("kraken-medieval-*.yaml"))

#: The only honest Kraken CER this project has, and what #111 asks to be beaten.
V2 = 0.2131
BASELINE = {"model": "kraken-medieval-german-v2", "value": V2,
            "metric": "benchmark_cer", "measured_on": "german-medieval-v1",
            "chars": 882255, "errors": 188022}


def a_manifest(**over) -> dict:
    raw = {
        "name": "t", "engine": "kraken",
        "data": {"train": "x", "eval": "y", "digest": "sha256:abc",
                 "datasets": [{"hf_repo": "a/b", "granularity": "line"}]},
        "budget": {"steps": 4000, "rungs": [2, 1]},
        "base": {"spec": "[1,48,0,1 Cr3,3,32]", "lrate": 1.0e-4,
                 "batch_size": 256, "accumulate_grad_batches": 1},
        "axes": {"lrate": [1.0e-4, 3.0e-4]},
        "baseline": dict(BASELINE),
    }
    raw.update(over)
    return raw


def judged(*, scores, metric="benchmark_cer", floor=None, baseline=BASELINE,
           tmp_path=None):
    """Run a whole sweep against a fake trainer and judge the result."""
    raw = a_manifest()
    if baseline is None:
        raw.pop("baseline")
    else:
        raw["baseline"] = baseline if isinstance(baseline, dict) else baseline
    if floor is not None:
        raw["noise_floor"] = {"value": floor, "measured_on": "sha256:abc",
                              "seeds": [42, 43], "commit": "abc1234",
                              "steps": 4000, "input_height": 48}
    manifest = parse_manifest(raw, source="test")
    ids = [c.config_id for c in manifest.configs()]
    trainer = FakeTrainer(dict(zip(ids, scores)))
    state = SweepState.for_manifest(manifest, metric,
                                    (tmp_path / "s.json") if tmp_path else None)
    state.train_lines = 120_000
    SweepDriver(manifest, state, trainer, sleep=lambda _s: None).run()
    rows = rows_for(manifest, state)
    return manifest, state, rows, judge(manifest, state, rows)


# ── the chain, end to end ──────────────────────────────────────────────────
def test_a_manifest_goes_in_and_a_leaderboard_comes_out(tmp_path):
    """The epic's first clause: one command works a manifest through the rungs
    and the table shows the ranking. The modules each have a suite; this is the
    one test that they meet."""
    manifest, state, rows, _ = judged(scores=[0.30, 0.25], tmp_path=tmp_path)

    table = render(manifest, state, rows)

    assert "# t" in table
    assert "**rung 0**" in table and "**rung 1**" in table
    assert all(c.config_id in table for c in manifest.configs())
    assert "sha256:abc" in table


def test_the_ladder_narrows_and_the_table_says_which_cell_survived(tmp_path):
    manifest, state, rows, verdict = judged(scores=[0.30, 0.25], tmp_path=tmp_path)

    rung1 = [r for r in rows if r.rung == 1]
    assert len(rung1) == 1
    assert verdict.best.config_id == rung1[0].config_id


def test_the_state_survives_a_restart_mid_sweep(tmp_path):
    """#114's criterion, held here because the epic's clause is "one command",
    and a multi-day sweep is restarted more than once."""
    state_path = tmp_path / "s.json"
    manifest, state, _, _ = judged(scores=[0.30, 0.25], tmp_path=tmp_path)

    reloaded = SweepState.load(state_path, manifest, "benchmark_cer")

    assert reloaded.results == state.results


# ── the verdict: the half nobody builds ────────────────────────────────────
def test_a_configuration_that_beats_the_reference_is_named_as_doing_so(tmp_path):
    _, _, _, verdict = judged(scores=[0.30, 0.18], floor=0.01, tmp_path=tmp_path)

    assert verdict.state == BEATS
    assert verdict.conclusive
    assert "beats kraken-medieval-german-v2" in verdict.sentence
    assert "0.0331 better" in verdict.sentence


def test_no_configuration_beating_it_is_a_finding_not_a_silence(tmp_path):
    """#111's second half, verbatim: "oder es ist belegt, dass keine der
    geprüften das tut"."""
    _, _, _, verdict = judged(scores=[0.30, 0.28], floor=0.01, tmp_path=tmp_path)

    assert verdict.state == DOES_NOT_BEAT
    assert verdict.conclusive
    assert "none of the" in verdict.sentence
    assert "established, not merely unobserved" in verdict.sentence


def test_a_win_smaller_than_the_floor_is_neither(tmp_path):
    """The failure that killed the first search's ranking: read finer than the
    material resolves."""
    _, _, _, verdict = judged(scores=[0.30, 0.2100], floor=0.01, tmp_path=tmp_path)

    assert verdict.state == INSIDE_NOISE
    assert not verdict.conclusive
    assert "Not a win and not a loss" in verdict.sentence
    assert "about the sweep and not about the model" in verdict.sentence


def test_a_loss_smaller_than_the_floor_is_also_neither(tmp_path):
    _, _, _, verdict = judged(scores=[0.30, 0.2160], floor=0.01, tmp_path=tmp_path)

    assert verdict.state == INSIDE_NOISE


def test_without_a_floor_the_sign_is_all_there_is(tmp_path):
    """#115 comes first for this reason, and the sentence says which reason."""
    _, _, _, verdict = judged(scores=[0.30, 0.18], tmp_path=tmp_path)

    assert verdict.state == NO_FLOOR
    assert not verdict.conclusive
    assert "flipped sign on a seed change" in verdict.sentence


# ── the two refusals ───────────────────────────────────────────────────────
def test_a_sweep_ranking_on_cer_is_not_compared_to_a_benchmark_baseline(tmp_path):
    """The state both written sweeps are in. `cer` is the validation partition,
    which overlaps the training projects; 0.2131 is 200 unseen documents."""
    _, _, _, verdict = judged(scores=[0.30, 0.18], metric="cer", floor=0.01,
                              tmp_path=tmp_path)

    assert verdict.state == METRIC_MISMATCH
    assert "not comparable" in verdict.sentence
    assert "`cer`" in verdict.sentence and "`benchmark_cer`" in verdict.sentence
    assert "german-medieval-v1" in verdict.sentence


def test_the_refusal_says_what_to_do_instead(tmp_path):
    _, _, _, verdict = judged(scores=[0.30, 0.18], metric="cer", tmp_path=tmp_path)

    assert "Measure the winner on german-medieval-v1" in verdict.sentence


def test_a_bare_baseline_number_is_shown_and_not_compared(tmp_path):
    """It cannot be: nothing says which metric it is."""
    _, _, _, verdict = judged(scores=[0.30, 0.18], baseline=V2, floor=0.01,
                              tmp_path=tmp_path)

    assert verdict.state == METRIC_MISMATCH
    assert "metric unstated" in verdict.sentence
    assert "shown, not compared" in verdict.sentence


def test_a_sweep_with_no_baseline_says_its_ranking_is_internal(tmp_path):
    _, _, _, verdict = judged(scores=[0.30, 0.18], baseline=None, floor=0.01,
                              tmp_path=tmp_path)

    assert verdict.state == NO_BASELINE
    assert "its ranking is internal" in verdict.sentence


def test_nothing_scored_is_not_a_loss():
    """An empty table must not read as "the baseline won"."""
    manifest = parse_manifest(a_manifest())
    state = SweepState.for_manifest(manifest, "benchmark_cer", None)

    verdict = judge(manifest, state, [])

    assert verdict.state == NOTHING_SCORED
    assert not verdict.conclusive
    assert "no configuration here has produced a number" in verdict.sentence


# ── the baseline block has to say what it is ───────────────────────────────
def test_a_baseline_block_without_a_metric_is_refused():
    """0.2131 on held-out documents and 0.2131 on a validation partition are
    different claims, and without the field the table compares them as one."""
    raw = a_manifest()
    raw["baseline"].pop("metric")

    with pytest.raises(ManifestError, match="baseline needs a `metric`"):
        parse_manifest(raw)


def test_a_baseline_block_without_a_measurement_set_is_refused():
    raw = a_manifest()
    raw["baseline"].pop("measured_on")

    with pytest.raises(ManifestError, match="needs a `measured_on`"):
        parse_manifest(raw)


def test_a_baseline_block_without_a_value_is_refused():
    with pytest.raises(ManifestError, match="without a `value`"):
        parse_manifest(a_manifest(baseline={"metric": "cer",
                                            "measured_on": "x"}))


@pytest.mark.parametrize("bad", [0, -0.1, "nought"])
def test_a_baseline_that_is_not_a_positive_number_is_refused(bad):
    with pytest.raises(ManifestError):
        parse_manifest(a_manifest(baseline=bad))


def test_a_sweep_without_a_baseline_still_loads():
    raw = a_manifest()
    raw.pop("baseline")

    assert parse_manifest(raw).baseline is None


# ── the written sweeps ─────────────────────────────────────────────────────
def test_there_are_written_sweeps_to_check():
    assert len(WRITTEN) >= 2


@pytest.mark.parametrize("path", WRITTEN, ids=lambda p: p.stem)
def test_every_written_sweep_names_what_it_has_to_beat(path):
    manifest = load_manifest(path)

    assert manifest.baseline == V2
    assert manifest.baseline_provenance["model"] == "kraken-medieval-german-v2"
    assert manifest.baseline_provenance["metric"] == "benchmark_cer"
    assert manifest.baseline_provenance["measured_on"] == "german-medieval-v1"


@pytest.mark.parametrize("path", WRITTEN, ids=lambda p: p.stem)
def test_the_acceptance_number_is_the_one_from_the_epic(path):
    """882,255 characters and 188,022 errors is 0.2131, and the file carries
    both so the number can be checked rather than trusted."""
    block = yaml.safe_load(path.read_text(encoding="utf-8"))["baseline"]

    assert block["errors"] / block["chars"] == pytest.approx(V2, abs=5e-5)


@pytest.mark.parametrize("path", WRITTEN, ids=lambda p: p.stem)
def test_the_written_sweeps_say_why_their_verdict_will_refuse(path):
    """They run `--metric cer`, so the comparison is refused by construction.
    Written in the file, because a reader meeting "not comparable" on the table
    should not have to work out why."""
    text = path.read_text(encoding="utf-8")

    assert "not comparable" in text
    assert "german-medieval-v1" in text


def test_the_verdict_lands_on_the_rendered_table(tmp_path):
    manifest, state, rows, verdict = judged(scores=[0.30, 0.18], floor=0.01,
                                            tmp_path=tmp_path)

    assert verdict.sentence in render(manifest, state, rows)


def test_the_verdict_sits_under_the_floor_line(tmp_path):
    """A reader meets the resolution before the conclusion drawn with it."""
    manifest, state, rows, verdict = judged(scores=[0.30, 0.18], floor=0.01,
                                            tmp_path=tmp_path)
    table = render(manifest, state, rows)

    assert table.index("noise floor") < table.index(verdict.sentence)


def test_the_state_file_is_json_a_person_can_read(tmp_path):
    """Not part of the acceptance, and the thing somebody reaches for at 2am."""
    judged(scores=[0.30, 0.25], tmp_path=tmp_path)

    assert json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))["results"]

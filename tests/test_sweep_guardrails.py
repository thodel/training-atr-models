"""The guards are the only reader a sweep's runs get (#119).

A sweep computes dozens of runs nobody looks at individually. That makes the
guardrails the only thing still checking whether a measurement means anything —
and a guard switched off for one sweep is switched off for every run in the
system.

The hole this file closes is the one #119 names: `reserved_pages: 0` meant two
different things. Measured on the finished run
`20260916T205123Z-qwen3vl-german-pages-v5-asteraix`, which adopted artefact
`82db328c96d7` and skipped `prepare`: it recorded `reserved_pages: 0` while the
run that *built* that artefact recorded 3,187. The corpus really was clean — the
cache key carries the held-out fingerprint, so an artefact built under a
different reservation is never adopted — but nothing in the adopting job's record
said so, and a zero that can mean "checked, nothing found" or "nobody looked" is
not recoverable across dozens of runs.
"""

import json
from pathlib import Path

import pytest
import yaml

from atr_training.contracts import Progress
from atr_training.leaderboard import REFUSED, render, rows_for
from atr_training.sweep_driver import SweepDriver, SweepState
from atr_training.sweep_manifest import parse_manifest

from tests.test_sweep_driver import FakeTrainer, manifest, scores_for


# ── the three-valued count ──────────────────────────────────────────────────
def test_a_fresh_progress_says_nothing_was_checked():
    """The default used to be 0, which claimed a finding nobody had produced."""
    assert Progress().reserved_pages is None
    assert Progress().reserved_pages_source is None


def test_zero_and_unchecked_are_different_values():
    checked = Progress(reserved_pages=0, reserved_pages_source="prepare")
    unchecked = Progress()

    assert checked.reserved_pages != unchecked.reserved_pages
    assert checked.reserved_pages == 0 and unchecked.reserved_pages is None


def test_a_count_from_an_adopted_artefact_names_the_artefact():
    progress = Progress(reserved_pages=3187,
                        reserved_pages_source="artefact 82db328c96d7 built by "
                                              "20260910T110352Z-qwen3vl-german-pages-v3")

    assert "82db328c96d7" in progress.reserved_pages_source


def test_the_heldout_fingerprint_has_a_place_on_the_record():
    """It is already in the cache key, so a corpus built under a different
    reservation is never adopted. The key is not in the job record; this is."""
    assert "heldout_fingerprint" in Progress.model_fields


# ── what travels with a reused corpus ───────────────────────────────────────
def _payload_fields() -> set[str]:
    """The keys `runner_base` writes into an artefact's payload."""
    import re

    source = (Path(__file__).resolve().parents[1]
              / "src" / "atr_training" / "runner_base.py").read_text(encoding="utf-8")
    block = source[source.index("inner=self.ARTEFACT_INNER, payload={"):]
    block = block[:block.index("})")]
    return set(re.findall(r'"([a-z_]+)":', block))


def test_the_reservation_count_travels_with_the_artefact():
    """#119's proposal in one assertion: what is true about how a corpus was
    made has to move with it."""
    assert "reserved_pages" in _payload_fields()


def test_the_heldout_fingerprint_travels_with_the_artefact():
    assert "heldout_fingerprint" in _payload_fields()


def test_the_dataset_counts_travel_with_the_artefact():
    """The same class as #108: an adopted artefact lost `dataset_counts`, and the
    test stage then drew at random instead of stratified."""
    assert "dataset_counts" in _payload_fields()


def test_the_line_counts_still_travel():
    """The guards read them and they cannot be recomputed once the pages are
    gone — adding fields must not displace the ones already there."""
    fields = _payload_fields()

    assert {"train_lines", "lines_written", "pages_written", "aspect_per_char"} <= fields


def test_adopting_sets_the_source_rather_than_leaving_a_bare_number():
    """The adopt path reads the payload onto `progress`. Without the extra line
    the number arrives looking exactly like one this job measured."""
    source = (Path(__file__).resolve().parents[1]
              / "src" / "atr_training" / "runner_base.py").read_text(encoding="utf-8")

    assert "reserved_pages_source = (" in source
    assert 'f"artefact {entry.key[:12]} built by "' in source


# ── no sweep path goes round a guard ────────────────────────────────────────
def test_the_driver_sends_nothing_that_disables_a_guard():
    """`force` is the only field in a TrainRequest that overrides one — it
    overrides both the convergence guard and the line-geometry guard in
    `runner_base`, recording an override on the job."""
    trainer = FakeTrainer(scores_for(manifest(), [0.2] * 12))
    m = manifest()
    state = SweepState.for_manifest(m, "benchmark_cer")
    state.train_lines = 120_000
    SweepDriver(m, state, trainer, sleep=lambda _s: None).run()

    assert trainer.submitted
    for request in trainer.submitted:
        assert request.get("force") in (None, False), request


def test_force_is_the_only_override_a_request_can_carry():
    """A guard that a future field could switch off would pass the test above
    while reopening the hole. This fails when such a field appears, which is the
    moment to decide whether a sweep may set it."""
    from atr_training.contracts import TrainRequest

    overriding = {name for name in TrainRequest.model_fields
                  if "force" in name or "override" in name or "skip" in name}

    assert overriding == {"force"}, overriding


def test_the_driver_never_names_force_at_all():
    """Not even as False: a field the driver does not know about cannot be set
    by accident in a later edit."""
    source = (Path(__file__).resolve().parents[1]
              / "src" / "atr_training" / "sweep_driver.py").read_text(encoding="utf-8")
    request_block = source[source.index("def request_for"):]
    request_block = request_block[:request_block.index("\n    # ──")]

    assert '"force"' not in request_block


# ── a refused configuration is a finding, not a gap ─────────────────────────
DIGEST = "sha256:0123456789abcdef"

REFUSAL = ("18,420 training lines at effective batch 256 is 72 step(s) per epoch; "
           "over 1 epochs that is 72 optimizer steps, from scratch, against a "
           "floor of 2,000.")


def _state_with(entry_updates: dict, m=None) -> tuple:
    m = m or manifest()
    state = SweepState.for_manifest(m, "benchmark_cer")
    state.train_lines = 120_000
    for index, config in enumerate(m.configs()):
        base = {"job_id": f"job-{index}", "status": "completed",
                "score": 0.8 - index / 100, "raw": 0.2 + index / 100,
                "metric": "benchmark_cer", "minutes": 40.0, "commit": "ff9effd8d99d",
                "reserved_pages": 3187, "reserved_pages_source": "prepare"}
        state.put(0, config.config_id, {**base, **(entry_updates if index == 0 else {})})
    return m, state


def test_a_guard_refusal_is_its_own_status():
    """#119: "abgewiesen mit Grund, nicht als schlechte CER". It never produced
    a number, so an empty CER column would read as a crash."""
    m, state = _state_with({"status": "failed", "error": REFUSAL,
                            "score": None, "raw": None})

    refused = [r for r in rows_for(m, state) if r.status == REFUSED]

    assert len(refused) == 1
    assert "optimizer steps" in refused[0].refusal


def test_the_refusal_and_its_reason_reach_the_table():
    m, state = _state_with({"status": "failed", "error": REFUSAL,
                            "score": None, "raw": None})

    text = render(m, state)

    assert "REFUSED" in text
    assert "Refused by a guard" in text
    assert "against a floor of 2,000" in text


def test_a_geometry_refusal_is_recognised_too():
    m, state = _state_with({"status": "failed", "score": None, "raw": None,
                            "error": "the spec leaves 0.94 frames per character"})

    assert [r.status for r in rows_for(m, state)].count(REFUSED) == 1


def test_an_ordinary_crash_is_not_dressed_up_as_a_refusal():
    """A machine that fell over is not a configuration the material cannot
    support, and putting both in one column loses the distinction the guards
    exist to draw."""
    m, state = _state_with({"status": "failed", "score": None, "raw": None,
                            "error": "CUDA out of memory"})

    assert REFUSED not in [r.status for r in rows_for(m, state)]


# ── the table says what was and was not checked ─────────────────────────────
def test_a_run_that_checked_nothing_is_called_out():
    m, state = _state_with({"reserved_pages": None, "reserved_pages_source": None})

    text = render(m, state)

    assert "record no held-out check" in text
    assert "not the same as looking and finding nothing" in text


def test_a_reused_corpus_says_where_its_number_came_from():
    m, state = _state_with({
        "reserved_pages": 3187,
        "reserved_pages_source": "artefact 82db328c96d7 built by 20260910T110352Z",
    })

    text = render(m, state)

    assert "Corpus reused" in text
    assert "82db328c96d7" in text


def test_a_sweep_where_every_run_did_its_own_prepare_says_nothing_extra():
    """The note has to mean something when it appears."""
    m, state = _state_with({})

    text = render(m, state)

    assert "Corpus reused" not in text
    assert "record no held-out check" not in text


def test_a_running_row_is_not_accused_of_skipping_the_check():
    """It has not reached prepare yet."""
    m, state = _state_with({"status": "running", "score": None, "raw": None,
                            "reserved_pages": None, "reserved_pages_source": None})

    assert "record no held-out check" not in render(m, state)


# ── the same measurement set for every cell ─────────────────────────────────
def test_every_configuration_is_scored_on_the_manifest_s_one_benchmark():
    """#119: all cells on the same set, or the leaderboard compares measurement
    sets. The manifest names one `data.eval` and every request carries the same
    dataset list, so there is one place this could diverge and it does not."""
    m = manifest()
    trainer = FakeTrainer(scores_for(m, [0.2] * 12))
    state = SweepState.for_manifest(m, "benchmark_cer")
    state.train_lines = 120_000
    SweepDriver(m, state, trainer, sleep=lambda _s: None).run()

    datasets = {json.dumps(r["datasets"], sort_keys=True) for r in trainer.submitted}
    assert len(datasets) == 1, "configurations were sent different corpora"


def test_the_benchmark_is_named_on_the_table():
    m, state = _state_with({})

    assert "german_test" in render(m, state)


def test_two_sweeps_on_different_data_cannot_share_a_state_file(tmp_path: Path):
    """The coarsest version of the same rule, already enforced — repeated here
    because #119 asks for the measurement set to be the same and this is what
    stops it silently changing mid-sweep."""
    from atr_training.sweep_driver import SweepError

    m = manifest()
    path = tmp_path / "s.json"
    SweepState.for_manifest(m, "benchmark_cer", path).save()
    other = parse_manifest({**yaml.safe_load(_MANIFEST_TEXT),
                            "data": {**yaml.safe_load(_MANIFEST_TEXT)["data"],
                                     "digest": "sha256:ffffffffffffffff"}},
                           source="other")

    with pytest.raises(SweepError, match="data.digest"):
        SweepState.load(path, other, "benchmark_cer")


_MANIFEST_TEXT = f"""
name: kraken-medieval-augment-01
data:
  train: shard_00
  eval: german_test
  digest: "{DIGEST}"
  datasets:
    - {{hf_repo: "dh-unibe/image-text_aaeb-xiv-xvii", granularity: line}}
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

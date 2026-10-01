"""What `max_seq_len` actually does, and what the record says about it (#138).

#12 and the issue written from it both state that `max_seq_len: 4096` truncates
samples, citing v5's *"a sample tokenized to 4515 tokens"*. The code says the
opposite, in as many words:

    # No ``truncation``/``max_length``. On a text-only sequence truncation
    # loses the tail; here it severs image tokens from the placeholders that
    # index them, and the result is not a shorter sample but an invalid one —
    # which is what killed 20260814T192904Z at step 2 of 774 (#86).

And the warning v5 emitted ends *"Not truncated"*. `max_seq_len` reaches exactly
one place — the collator's threshold — and sizes nothing, refuses nothing and
truncates nothing.

So the real gap is a different one, and it is the class of #119: a fact about a
run that never reached the run's record. `over_budget` was a counter in the
training subprocess, printed at most every hundredth time, so "how far over
budget was this corpus" was answerable only by grepping a log — and a run that
OOMed could not be read against it.
"""

import json
from pathlib import Path

import pytest

from atr_training.contracts import Progress, VlmTrainParams

ROOT = Path(__file__).resolve().parents[1]
COLLATOR = ROOT / "engines" / "vlm_train_svc" / "train_qlora.py"
RUNNER = ROOT / "engines" / "vlm_train_svc" / "runner.py"


# ── what max_seq_len is, and is not ─────────────────────────────────────────
def test_the_collator_never_truncates():
    """The claim both issues make, refuted from the source. A truncated
    multimodal sequence is invalid, not short."""
    text = COLLATOR.read_text(encoding="utf-8")
    call = text[text.index("def __call__(self, batch"):]
    call = call[:call.index("\n\n\n")]

    assert "truncation=True" not in call
    assert "max_length=" not in call


def test_the_warning_says_it_did_not_truncate():
    """v5's line is quoted in #12 as evidence of truncation. It says the
    opposite, and whoever reads the log next should not have to find that out
    twice."""
    assert "Not truncated" in COLLATOR.read_text(encoding="utf-8")


def test_max_seq_len_only_warns_and_is_reported():
    """It sizes nothing and refuses nothing — the two places it reaches are the
    collator's threshold and the record. This fails the day a third appears,
    which is the day the comments above stop being true.

    Written as a whitelist rather than a count because adding the report was
    itself a second use: a count would have called that a regression, and a
    number nobody can read is not better than a number that acts."""
    uses = [line.strip() for line in COLLATOR.read_text(encoding="utf-8").splitlines()
            if "args.max_seq_len" in line]

    assert len(uses) == 2, uses
    assert any("HTRCollator" in u for u in uses), uses
    assert any('"sequence_budget"' in u for u in uses), uses


def test_the_budget_still_has_a_per_granularity_default():
    """A page sample and a line crop do not have the same honest ceiling."""
    page = VlmTrainParams(granularity="page").sequence_budget()
    line = VlmTrainParams(granularity="line").sequence_budget()

    assert page != line
    assert VlmTrainParams(max_seq_len=8192).sequence_budget() == 8192


# ── the gap that is real ────────────────────────────────────────────────────
def test_the_record_has_somewhere_to_put_the_count():
    for field in ("over_budget_samples", "max_sequence_tokens", "sequence_budget"):
        assert field in Progress.model_fields, field


def test_an_unmeasured_run_says_nothing_rather_than_zero():
    """Zero over-budget samples is a finding; `None` is the absence of one. The
    same three-valued honesty as `reserved_pages` (#119)."""
    assert Progress().over_budget_samples is None
    assert Progress().max_sequence_tokens is None


def test_the_subprocess_reports_all_three():
    """`training_summary.json` is the channel that already existed and that the
    VLM runner did not read."""
    text = COLLATOR.read_text(encoding="utf-8")
    summary = text[text.index('(out_dir / "training_summary.json")'):]
    summary = summary[:summary.index("encoding=")]

    for key in ("sequence_budget", "over_budget_samples", "max_sequence_tokens"):
        assert f'"{key}"' in summary, key


def test_the_collator_tracks_the_longest_not_only_the_count():
    """"20 % of samples are 1 % over" is a different run from "2 % are 60 %
    over", and a count alone cannot tell them apart."""
    text = COLLATOR.read_text(encoding="utf-8")

    assert "self.longest = max(self.longest, length)" in text


# ── the runner carries it across ────────────────────────────────────────────
class _Store:
    def __init__(self) -> None:
        self.saves = 0

    def save(self, job) -> None:
        self.saves += 1


class _Runner:
    """Just the method under test, bound to a stand-in store."""

    def __init__(self) -> None:
        self.store = _Store()

    _record_sequence_budget = None  # bound below


def _runner():
    import engines.vlm_train_svc.runner as module

    runner = _Runner()
    runner._record_sequence_budget = module.Pipeline._record_sequence_budget.__get__(runner)
    return runner


class _Job:
    def __init__(self) -> None:
        self.progress = Progress()


@pytest.fixture
def summary(tmp_path: Path) -> Path:
    (tmp_path / "training_summary.json").write_text(json.dumps({
        "base_model": "Qwen/Qwen3-VL-8B-Instruct",
        "sequence_budget": 4096,
        "over_budget_samples": 37,
        "max_sequence_tokens": 4515,
    }), encoding="utf-8")
    return tmp_path


def test_the_numbers_reach_the_record(summary):
    job = _Job()

    _runner()._record_sequence_budget(job, summary)

    assert job.progress.sequence_budget == 4096
    assert job.progress.over_budget_samples == 37
    assert job.progress.max_sequence_tokens == 4515


def test_v5s_own_number_is_what_this_would_have_recorded(summary):
    """4515 against 4096 is 10 % over — the figure #12 cites, and the figure
    nobody could get at from the record."""
    job = _Job()

    _runner()._record_sequence_budget(job, summary)

    over = job.progress.max_sequence_tokens / job.progress.sequence_budget - 1
    assert 0.09 < over < 0.11


def test_a_run_whose_summary_is_missing_is_not_failed(tmp_path):
    """The summary is written after the adapter, so a run that trained and could
    not write it has still trained."""
    job = _Job()

    _runner()._record_sequence_budget(job, tmp_path)

    assert job.progress.over_budget_samples is None


def test_a_summary_that_does_not_parse_is_not_failed_either(tmp_path):
    (tmp_path / "training_summary.json").write_text("{ not json", encoding="utf-8")
    job = _Job()

    _runner()._record_sequence_budget(job, tmp_path)

    assert job.progress.over_budget_samples is None


def test_an_older_summary_without_the_keys_leaves_them_unknown(tmp_path):
    """Runs from before this change wrote the file without them. `None` is the
    right answer, and a 0 would be a measurement nobody made."""
    (tmp_path / "training_summary.json").write_text(
        json.dumps({"base_model": "x", "epochs": 3}), encoding="utf-8")
    job = _Job()

    _runner()._record_sequence_budget(job, tmp_path)

    assert job.progress.over_budget_samples is None
    assert job.progress.sequence_budget is None


def test_the_record_is_written_once_the_numbers_are_in(summary):
    runner = _runner()

    runner._record_sequence_budget(_Job(), summary)

    assert runner.store.saves == 1


def test_a_run_that_fitted_everything_records_zero_not_none(tmp_path):
    """Zero is a finding: the corpus fitted. It must be told from "nobody
    looked"."""
    (tmp_path / "training_summary.json").write_text(json.dumps({
        "sequence_budget": 4096, "over_budget_samples": 0,
        "max_sequence_tokens": 3011}), encoding="utf-8")
    job = _Job()

    _runner()._record_sequence_budget(job, tmp_path)

    assert job.progress.over_budget_samples == 0
    assert job.progress.max_sequence_tokens == 3011


def test_the_runner_records_it_before_looking_for_the_adapter():
    """A run that trained and wrote no adapter is exactly the run whose
    over-budget numbers somebody wants."""
    text = RUNNER.read_text(encoding="utf-8")

    assert text.index("_record_sequence_budget(job, out_dir)") < \
        text.index("adapter = find_adapter(out_dir)")

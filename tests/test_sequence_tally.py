"""The sequence numbers survive the DataLoader's worker processes (#138).

`over_budget_samples` and `max_sequence_tokens` were **0 in all seven** VLM
records that carried them, including page runs at budget 4096 where a real
sequence is thousands of tokens. The collator was right and the handover was
right; what was wrong is that `Trainer` runs `collate_fn` inside the DataLoader's
worker subprocesses when `dataloader_num_workers > 0` — the default here is 4 —
so each worker mutated its own copy and the parent wrote back its untouched
zeros. Measured on asteraix with a counting collate_fn:

    num_workers=0: parent sees longest= 239 over= 47   (truth: 239 / 47)
    num_workers=2: parent sees longest=   0 over=  0
    num_workers=4: parent sees longest=   0 over=  0

`0` is not a harmless wrong number here: the contract reserves it for "measured,
nothing was over budget" (`None` means not measured), so the records claimed a
measurement nobody took. These tests use real processes rather than torch, which
the suite deliberately does not have.
"""

from __future__ import annotations

import multiprocessing as mp

import pytest

from atr_training.vlm_dataset import SequenceTally


def _collate(tally: SequenceTally, lengths: list[int], budget: int) -> None:
    """What a worker does: mutate the tally it was handed."""
    for length in lengths:
        tally.saw(length, budget)


# ── in one process ───────────────────────────────────────────────────────────
def test_it_keeps_the_longest_and_counts_the_ones_over():
    tally = SequenceTally()
    for length in (900, 1500, 1200, 1024):
        tally.saw(length, 1024)

    assert tally.longest == 1500
    assert tally.over_budget == 2, "1500 and 1200; 1024 is not over its own budget"


def test_the_longest_counts_samples_under_budget_too():
    """The question is how close the corpus runs to the budget, not only how often
    it passes it — a run with nothing over budget still has a longest sample."""
    tally = SequenceTally()
    for length in (300, 700, 500):
        tally.saw(length, 1024)

    assert tally.longest == 700
    assert tally.over_budget == 0


# ── across processes, which is the whole point ───────────────────────────────
@pytest.mark.parametrize("workers", [2, 4])
def test_what_the_workers_see_reaches_the_parent(workers):
    """The failure this class exists for: with plain attributes the parent read
    back 0 here, whatever the workers had counted."""
    tally = SequenceTally()
    budget = 1024
    batches = [[50 + i * 3 for i in range(w, 64, workers)] for w in range(workers)]
    truth_longest = max(n for b in batches for n in b)
    truth_over = sum(1 for b in batches for n in b if n > budget)

    procs = [mp.Process(target=_collate, args=(tally, b, budget)) for b in batches]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    assert tally.longest == truth_longest
    assert tally.over_budget == truth_over


def test_two_workers_cannot_lose_each_others_increments():
    """A plain `+=` across processes loses writes even when it is visible at all;
    the lock is what makes the count the sum."""
    tally = SequenceTally()
    procs = [mp.Process(target=_collate, args=(tally, [2000] * 200, 1024))
             for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    assert tally.over_budget == 800
    assert tally.longest == 2000


# ── and it refuses to claim a measurement it did not take ────────────────────
def test_an_unobserved_run_reports_none_not_zero():
    """`0` is the contract's word for "measured, all of it fitted". A tally that
    never saw a sample must not say that — which is exactly what seven records
    said."""
    record = SequenceTally().as_record(4096)

    assert record == {"sequence_budget": 4096, "over_budget_samples": None,
                      "max_sequence_tokens": None}


def test_a_run_where_everything_fitted_reports_zero_over_and_the_longest():
    tally = SequenceTally()
    tally.saw(3800, 4096)
    record = tally.as_record(4096)

    assert record["over_budget_samples"] == 0, "measured, and nothing was over"
    assert record["max_sequence_tokens"] == 3800
    assert record["sequence_budget"] == 4096

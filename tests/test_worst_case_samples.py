"""`worst_case_samples` — the tail a peak-memory measurement must run on (#163).

#137 measured one peak over "96 of 9,441 samples, from the head of the file" and
said what that was worth: a lower bound. #163 asks for the opposite — the peak
from the distribution — because on this project an OOM was twice the edge of the
distribution and not the batch size.
"""
from __future__ import annotations

from atr_training.vlm_dataset import Sample, worst_case_samples


def line(text: str, width: int = 100, height: int = 40) -> Sample:
    return Sample(image="p.jpg", text=text, source_type="line",
                  bbox=[0, 0, width, height], page="p.xml")


def page(text: str) -> Sample:
    return Sample(image="p.jpg", text=text, source_type="page", page="p.xml")


# ── the text tail ───────────────────────────────────────────────────────────
def test_the_longest_transcription_is_taken_first():
    """The fp32 logit tensor is the largest single allocation in the step."""
    pool = [line("a" * n) for n in (10, 5000, 20, 300)]
    got = worst_case_samples(pool, 1)
    assert [len(s.text) for s in got.samples] == [5000]
    assert got.longest_chars == 5000
    assert got.considered == 4


def test_both_tails_are_represented():
    """A sample can be cheap in text and expensive in pixels.

    The short-but-huge crop must not be excluded by the long-but-narrow one.
    """
    wordy = line("a" * 9000, width=50, height=20)
    wide = line("ab", width=6000, height=400)
    middling = [line("a" * 200, width=300, height=40) for _ in range(8)]
    got = worst_case_samples([*middling, wordy, wide], 2)
    assert wordy in got.samples
    assert wide in got.samples


def test_the_budget_is_filled_even_when_the_tails_coincide():
    """Interleaving, not two slices.

    On a page corpus the longest text usually is the largest image, so two
    half-slices of the same ordering would return half the samples asked for.
    """
    pool = [line("a" * (100 * n), width=100 * n, height=50) for n in range(1, 21)]
    got = worst_case_samples(pool, 6)
    assert len(got.samples) == 6
    assert len({id(s) for s in got.samples}) == 6


# ── whole pages ─────────────────────────────────────────────────────────────
def test_the_text_tail_has_priority_when_only_one_sample_fits():
    """A budget of one can only measure one tail, and text is the one to measure.

    Both OOMs this project has recorded were in `logits.float()`, which is the
    text tail; `drop_long_samples` exists for the same reason. So at count=1 the
    longest transcription wins over the largest image, deliberately.
    """
    pool = [line("a" * 500, width=10, height=10), page("b" * 10)]
    got = worst_case_samples(pool, 1)
    assert [s.source_type for s in got.samples] == ["line"]


def test_a_whole_page_outranks_every_crop_in_the_pixel_tail():
    """It is larger than any crop of itself, so it enters as soon as two fit."""
    pool = [line("a" * 500, width=4000, height=3000), page("b" * 10)]
    got = worst_case_samples(pool, 2)
    assert [s.source_type for s in got.samples] == ["line", "page"]


def test_pages_are_counted_rather_than_given_a_false_area():
    """`widest_pixels` reports crops only; a page has no box to measure."""
    got = worst_case_samples([page("a"), page("b"), line("c", width=70, height=30)], 3)
    assert got.page_samples == 2
    assert got.widest_pixels == 70 * 30
    assert "2 whole page(s)" in str(got)


def test_no_crops_says_so_instead_of_zero_pixels():
    got = worst_case_samples([page("a")], 1)
    assert got.widest_pixels == 0
    assert "no crops" in str(got)


# ── the honest edges ────────────────────────────────────────────────────────
def test_asking_for_more_than_exists_returns_everything():
    """And `considered` then says the peak covers the whole set."""
    pool = [line("a" * 10), line("b" * 20)]
    got = worst_case_samples(pool, 50)
    assert len(got.samples) == 2
    assert got.considered == 2


def test_an_empty_pool_is_not_an_error():
    got = worst_case_samples([], 10)
    assert got.samples == []
    assert got.considered == 0
    assert got.longest_chars == 0


def test_a_zero_budget_selects_nothing():
    got = worst_case_samples([line("a" * 99)], 0)
    assert got.samples == []
    assert got.considered == 1


def test_a_degenerate_box_does_not_crash_the_ordering():
    """Right < left happens; `line_samples` clamps, but this must not raise."""
    got = worst_case_samples([line("a", width=-5, height=-5), line("b" * 3)], 2)
    assert len(got.samples) == 2
    assert got.widest_pixels >= 0


def test_the_summary_names_what_drove_the_selection():
    got = worst_case_samples([line("a" * 1234, width=800, height=60)], 1)
    text = str(got)
    assert "1 worst-case sample(s) of 1" in text
    assert "longest 1234 chars" in text
    assert "48000 px" in text

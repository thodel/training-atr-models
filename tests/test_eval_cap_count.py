"""Which readings ran to ``max_new_tokens`` — counted on the generated ids (#301).

The count used to re-tokenise the decoded text, which comes out shorter than
what was generated: on the held-out page draw at 1536 it reported 38, 29, 37
and 47 pages for the four olmOCR arms where 47, 46, 60 and 60 went on writing
once the cap was doubled. ``truncated_at_cap`` is the column that is supposed to
say a CER was cut off, so a low count hides exactly what it exists to show.

    pytest tests/test_eval_cap_count.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engines"))

from vlm_train_svc.evaluate_qlora import _ran_to_cap  # noqa: E402

STOPS = [151645, 151643]
CAP = 8


def test_every_allowed_token_spent_without_a_stop_is_at_the_cap():
    assert _ran_to_cap([11] * CAP, CAP, STOPS)


def test_a_stop_on_the_last_allowed_token_is_a_finished_reading():
    """The model ended the turn exactly at the cap: done, not cut off."""
    assert not _ran_to_cap([11] * (CAP - 1) + [151645], CAP, STOPS)


def test_a_reading_that_stopped_early_is_not_at_the_cap():
    """Two tokens short was close enough for the old count (``n >= cap - 2``)."""
    assert not _ran_to_cap([11] * (CAP - 2), CAP, STOPS)
    assert not _ran_to_cap([11] * (CAP - 3) + [151643], CAP, STOPS)


def test_an_empty_reading_is_not_at_the_cap():
    assert not _ran_to_cap([], CAP, STOPS)

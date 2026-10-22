"""Should this finished model be published, and where (#88).

The decision is separated from the upload so it can be tested without a token, a
network, or a model on disk — and so the reasons it says *no* are as legible as
the one time it says yes.

Publishing is outward-facing and effectively irreversible: the hub keeps history,
so an unpublish is a deletion that does not undo the copy anyone already pulled.
Four things therefore have to hold, and each of them is a separate refusal:

* the threshold is set at all (``auto_publish_min_accuracy`` defaults to 0 = off);
* the run actually measured an accuracy (``char_accuracy`` is None for a job whose
  test stage did not report one);
* that accuracy reaches the threshold;
* a token exists, because failing inside the upload is a worse place to find out.
  In the environment, or where ``hf auth login`` stored it — which is where it
  lives on asteraix, and where ``HfApi()`` reads it from by itself (#33).

Repos are private and no licence is invented — the same two rules
``scripts/publish_to_hub.py`` enforces for a human at a shell. Automation gets
*less* latitude than a person, not more.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

__all__ = ["PublishDecision", "TOKEN_ENV", "TOKEN_PREFIX", "decide"]

#: Checked in order; the first one set wins. `HF_TOKEN` is what the trainer venvs
#: use, `HUGGINGFACE_HUB_TOKEN` is what `huggingface_hub` reads on its own.
TOKEN_ENV = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")


@dataclass(frozen=True)
class PublishDecision:
    publish: bool
    reason: str
    org: str = ""
    #: Always True when publishing. Kept explicit so a future caller cannot make
    #: it configurable without reading this.
    private: bool = True

    def __str__(self) -> str:
        return f"{'publish' if self.publish else 'skip'}: {self.reason}"


#: HuggingFace user tokens are `hf_` followed by ~34 characters. Requiring the
#: prefix is what distinguishes a token from a placeholder someone pasted without
#: substituting — `HF_TOKEN=...` really was set on the box, three characters long,
#: and "non-empty" accepted it happily (#88).
TOKEN_PREFIX = "hf_"
TOKEN_MIN_LEN = 12


def _hub_stored_token() -> str | None:
    """The token ``hf auth login`` stored, as ``huggingface_hub`` finds it.

    The value stays in this process: it is only shape-checked, never logged and
    never returned to a caller that could print it. ``None`` where the library
    is not installed (the gateway venv) or nothing is stored.
    """
    try:
        from huggingface_hub import get_token  # noqa: PLC0415 — engine venvs only
    except ImportError:
        return None
    try:
        return get_token()
    except Exception:  # noqa: BLE001 — an unreadable store is "no token", not a crash
        return None


def _token_problem(environ, stored_token: Callable[[], str | None]) -> str | None:
    """Why there is no usable token, or None when there is one.

    Never returns the value or any part of it: this string is logged and lands on
    the job record.
    """
    found = [(name, (environ.get(name) or "").strip()) for name in TOKEN_ENV]
    stored = (stored_token() or "").strip()
    if stored:
        found.append(("the token stored by `hf auth login`", stored))
    present = [(name, value) for name, value in found if value]
    if not present:
        return (f"no token is set ({' or '.join(TOKEN_ENV)}) and none is stored "
                "by `hf auth login`")
    for name, value in present:
        if value.startswith(TOKEN_PREFIX) and len(value) >= TOKEN_MIN_LEN:
            return None
    name, value = present[0]
    return (f"{name} is set but does not look like a HuggingFace token "
            f"(expected {TOKEN_PREFIX}… of at least {TOKEN_MIN_LEN} characters, "
            f"got {len(value)} character(s)) — a placeholder left unsubstituted "
            f"would otherwise fail inside the upload instead of here")


def decide(char_accuracy: float | None, threshold: float, org: str,
           environ=None, stored_token: Callable[[], str | None] | None = None
           ) -> PublishDecision:
    """Whether to auto-publish a model that scored ``char_accuracy`` percent.

    Without *environ* this is the trainer's own call: ``os.environ``, then the
    token ``hf auth login`` stored. A caller that passes its own environment gets
    exactly that and no store unless it passes *stored_token* too — a decision
    under test must not depend on what the machine running the test holds.
    """
    if environ is None:
        environ = os.environ
        stored_token = stored_token or _hub_stored_token
    stored_token = stored_token or (lambda: None)

    if threshold <= 0:
        return PublishDecision(False, "auto-publish is off "
                                      "(ATR_TRAIN_AUTO_PUBLISH_MIN_ACCURACY=0)")
    if char_accuracy is None:
        return PublishDecision(
            False, "the run reported no char_accuracy, so the threshold cannot be "
                   "applied — a model whose score is unknown is not published")
    if char_accuracy < threshold:
        return PublishDecision(
            False, f"char_accuracy {char_accuracy:.2f}% is below the "
                   f"{threshold:.2f}% threshold")
    problem = _token_problem(environ, stored_token)
    if problem:
        return PublishDecision(
            False, f"char_accuracy {char_accuracy:.2f}% reaches the threshold, but "
                   f"{problem}. Nothing was uploaded — authenticate with "
                   "`hf auth login` or run scripts/publish_to_hub.py by hand.")
    return PublishDecision(
        True, f"char_accuracy {char_accuracy:.2f}% reaches the "
              f"{threshold:.2f}% threshold", org=org, private=True)

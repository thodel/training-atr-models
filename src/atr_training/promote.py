"""The promotion gate (#36): advertise a trained model only once it has served.

Registering a model writes a row in a YAML file. That is not evidence of
anything — and confidently advertising models the host could not run is this
repo's most repeated failure (#21, #30, #31, #32). So a trained model is
registered ``enabled: false``, and only a **real transcription through the real
engine** flips it: the trainer posts one held-out page to the gateway's ``/ocr``
with the new model id, and non-empty text is the gate.

Four properties worth stating, because each is a decision:

* **The page comes from the run's own validation split.** Any page would prove
  the engine can load the weights, but a held-out page also exercises the
  material the model was scored on, and it is already on disk.
* **A failed gate does not fail the job.** The model trained, it scored, it is
  registered — it simply is not servable yet, which is a fact about the serving
  side. `docs/TRAINING_PLAN.md` §5 originally said a job completes only with a
  passing smoke recognition; that reads well until the VLM backend, whose
  adapters *cannot* be served until ``scripts/merge_loras.py`` bakes them in.
  Failing those jobs would call a good training run a failure. The job completes,
  ``promoted`` is false, the reason is on the record, and ``/models`` stays quiet.
* **Empty text is a failure, not a pass.** A 200 with ``""`` is exactly what #21
  was about: the gateway used to answer that way for a model it could not run.
* **"Unknown model" is asked again, for a while.** The registration was written
  milliseconds before the gate runs, and the gateway learns of it only from a
  look at the share that a request starts, in a thread, while that request is
  answered from what the gateway knew before (serving ``RegistryWatch.poll``) —
  at most one look per ``registry_reload_interval_s`` (5 s), and the CIFS
  attribute cache on idhefix lags on top. A single request therefore got
  ``404 unknown model`` on every run: reproduced against the gateway's own app
  (#14 review), cold, warm and with the interval at 0. Asked every 6 s against
  that app, the third request passed, 12 s in. Nothing else is retried: any
  other failure is the answer.

The HTTP call is injectable, so the whole gate is testable without a gateway.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from atr_training.manifests import read_manifest

__all__ = ["PROMOTION_GATE_HEADER", "NotYetVisible", "PromotionResult", "Recognizer",
           "held_out_page", "promote", "http_recognizer"]

#: Sent with the gate's request, value ``1``. The model under test is registered
#: ``enabled: false``, and the gateway refuses a disabled id to every caller — so
#: without a way to ask for exactly this one, the gate cannot pass at all. It never
#: has: on the serving side merge() dropped disabled overlay entries, and every
#: kraken gate got ``404 unknown model`` (found in the review of
#: serving-atr-inference#138). The gateway honours the header only with the
#: shared registry on, and only for a trained registration without a
#: ``disabled_reason``. Must match serving-atr-inference's value.
PROMOTION_GATE_HEADER = "X-ATR-Promotion-Gate"


#: How the gateway's 404 for an id it has no registration for begins
#: (``_resolve_spec_strict`` in serving's ``api/routes.py``). Its other 404 — a
#: registered model that is disabled for a stated reason — is final.
UNKNOWN_MODEL_DETAIL = "unknown model"


class NotYetVisible(RuntimeError):
    """The gateway does not know the model (yet): ``404 unknown model``.

    Raised by a :class:`Recognizer` so :func:`promote` can tell "the gateway has
    not read the registration" — which time fixes — from every other failure,
    which it does not.
    """


@dataclass(frozen=True)
class PromotionResult:
    promoted: bool
    reason: str
    #: What came back, trimmed — enough to see *what* was transcribed, in the log
    #: and on the job record, without pasting a page of text into json.
    sample: str | None = None


class Recognizer(Protocol):
    """Posts one image to the gateway and returns the transcription."""

    def __call__(self, model_id: str, image: Path) -> str: ...


def held_out_page(data_dir: Path, manifest_name: str = "pages_val.lst") -> Path | None:
    """The first validation page's image, or None when there is nothing to send.

    The manifests list PageXML paths; the materialized image sits beside each one
    with the same stem (``prepare`` writes ``<stem>.xml`` and ``<stem>.jpg``).
    """
    manifest = Path(data_dir) / manifest_name
    if not manifest.exists():
        return None
    for entry in read_manifest(manifest):
        image = Path(entry).with_suffix(".jpg")
        if image.exists():
            return image
    return None


def promote(model_id: str, page: Path | None, recognize: Recognizer, *,
            wait_s: float = 0.0, retry_every_s: float = 10.0,
            sleep: Callable[[float], None] = time.sleep) -> PromotionResult:
    """Run the gate. Never raises: every outcome is a reportable result.

    While the recognizer raises :class:`NotYetVisible`, it is asked again every
    ``retry_every_s`` for up to ``wait_s`` — a bounded number of requests, each
    of which also starts the gateway's next look at the share. The spacing is
    what matters: wider than the gateway's reload interval, so every retry finds
    the look its predecessor started already finished.
    """
    if page is None:
        return PromotionResult(False, "no held-out page was available to test with")
    tries = 1 + (int(wait_s // retry_every_s) if retry_every_s > 0 else 0)
    for attempt in range(1, tries + 1):
        try:
            text = recognize(model_id, page)
            break
        except NotYetVisible as exc:
            last = exc
            if attempt < tries:
                sleep(retry_every_s)
        except Exception as exc:  # noqa: BLE001 — a failed gate is a result, not a crash
            return PromotionResult(False, f"{type(exc).__name__}: {exc}")
    else:
        waited = (tries - 1) * retry_every_s
        return PromotionResult(
            False, f"the gateway never saw trained/{model_id}.yaml: {tries} request(s) over "
                   f"{waited:.0f} s, each answered {last}. The model stays registered but "
                   "disabled. Check that the gateway's ATR_REGISTRY_ROOT is this host's "
                   "ATR_TRAIN_REGISTRY_ROOT, and whether its log says it skipped the file.")

    if not (text or "").strip():
        # #21: a 200 with empty text is precisely how the gateway used to answer
        # for a model it could not actually run.
        return PromotionResult(False, f"the engine returned no text for {page.name}")
    return PromotionResult(True, f"transcribed {page.name} through the gateway",
                           sample=text.strip()[:200])


def http_recognizer(gateway_url: str, api_key: str, timeout: float = 120.0) -> Recognizer:
    """A :class:`Recognizer` that posts to the gateway's ``/ocr``.

    Deliberately goes through the gateway rather than straight to the engine: the
    question the gate asks is "can this box *serve* it", and the gateway is what
    clients talk to. A model that only works when addressed directly is not
    promoted, because that is not how anyone will call it.
    """

    def recognize(model_id: str, image: Path) -> str:
        import httpx  # trainer venv only

        with image.open("rb") as fh:
            response = httpx.post(
                f"{gateway_url.rstrip('/')}/ocr",
                headers={"X-API-Key": api_key, PROMOTION_GATE_HEADER: "1"},
                files={"image": (image.name, fh, "image/jpeg")},
                data={"model": model_id},
                timeout=timeout,
            )
        if response.status_code == 404:
            detail = _detail(response)
            if detail.startswith(UNKNOWN_MODEL_DETAIL):
                # Only its first sentence: the rest lists every known id.
                raise NotYetVisible(f"404 {detail.split('. ')[0]}")
        response.raise_for_status()
        return str(response.json().get("text") or "")

    return recognize


def _detail(response) -> str:
    """FastAPI's ``{"detail": "..."}``, or "" for any other body."""
    try:
        body = response.json()
    except ValueError:
        return ""
    detail = body.get("detail") if isinstance(body, dict) else None
    return detail if isinstance(detail, str) else ""

"""The promotion gate (#36) — advertise only what has actually served.

No gateway here: `promote` takes the recognizer as an argument, so every outcome
is exercised without a network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atr_training.promote import NotYetVisible, held_out_page, promote


# ── the gate ────────────────────────────────────────────────────────────────
def test_a_real_transcription_promotes(tmp_path: Path):
    page = tmp_path / "p.jpg"
    page.write_bytes(b"JPEG")
    result = promote("m", page, lambda model_id, image: "die brief von thun")
    assert result.promoted
    assert "die brief von thun" in result.sample


def test_empty_text_does_not_promote(tmp_path: Path):
    """#21 exactly: a 200 with "" is how the gateway used to answer for a model it
    could not run. Registering that as a pass would rebuild the original bug."""
    page = tmp_path / "p.jpg"
    page.write_bytes(b"JPEG")
    for answer in ("", "   ", "\n"):
        assert promote("m", page, lambda *_: answer).promoted is False


def test_an_engine_error_is_a_verdict_not_a_crash(tmp_path: Path):
    page = tmp_path / "p.jpg"
    page.write_bytes(b"JPEG")

    def boom(model_id, image):
        raise RuntimeError("502 Bad Gateway")

    result = promote("m", page, boom)
    assert result.promoted is False
    assert "502" in result.reason


# ── the gateway has not read the registration yet (#14 review) ─────────────
class Gateway:
    """Answers "unknown model" ``misses`` times, then transcribes — the gateway's
    RegistryWatch, which learns of a registration only from a look that a
    request starts and that request does not wait for."""

    def __init__(self, misses: int | None) -> None:
        self.misses = misses
        self.asked = 0

    def __call__(self, model_id, image):
        self.asked += 1
        if self.misses is None or self.asked <= self.misses:
            raise NotYetVisible(f"404 unknown model '{model_id}'")
        return "die brief von thun"


@pytest.fixture
def page(tmp_path: Path) -> Path:
    p = tmp_path / "p.jpg"
    p.write_bytes(b"JPEG")
    return p


def test_the_gate_asks_again_until_the_gateway_has_read_the_registration(page):
    gateway, slept = Gateway(misses=2), []
    result = promote("kraken-new", page, gateway, wait_s=60, retry_every_s=10,
                     sleep=slept.append)
    assert result.promoted, result.reason
    assert gateway.asked == 3
    assert slept == [10, 10]


def test_a_gateway_that_never_sees_it_is_asked_a_bounded_number_of_times(page):
    gateway, slept = Gateway(misses=None), []
    result = promote("kraken-new", page, gateway, wait_s=60, retry_every_s=10,
                     sleep=slept.append)
    assert result.promoted is False
    assert gateway.asked == 7                  # the first request and six retries
    assert slept == [10] * 6                   # no sleep after the last one
    assert "never saw trained/kraken-new.yaml" in result.reason
    assert "over 60 s" in result.reason
    assert "unknown model" in result.reason
    assert "ATR_REGISTRY_ROOT" in result.reason


def test_without_a_wait_the_gate_asks_once(page):
    gateway = Gateway(misses=None)
    result = promote("kraken-new", page, gateway, sleep=lambda s: pytest.fail("slept"))
    assert result.promoted is False and gateway.asked == 1


def test_only_unknown_model_is_asked_again(page):
    """Any other failure is the answer: retrying a 502 for a minute only hides it."""
    asked = []

    def broken(model_id, image):
        asked.append(model_id)
        raise RuntimeError("502 Bad Gateway")

    result = promote("kraken-new", page, broken, wait_s=60, retry_every_s=10,
                     sleep=lambda s: pytest.fail("slept"))
    assert result.promoted is False and "502" in result.reason
    assert asked == ["kraken-new"]


def _gateway_answering(monkeypatch, status: int, body: dict):
    import httpx

    def post(url, headers=None, **kwargs):
        return httpx.Response(status, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)


def test_the_gateways_unknown_model_404_means_not_yet(monkeypatch, page):
    from atr_training.promote import http_recognizer

    _gateway_answering(monkeypatch, 404, {
        "detail": "unknown model 'kraken-new'. Pass a registered id (see GET /models) "
                  "or a raw Zenodo ref (10.xxxx/zenodo.NNNN). Known ids: ['a', 'b']"})
    with pytest.raises(NotYetVisible) as caught:
        http_recognizer("http://gw", "k")("kraken-new", page)
    assert str(caught.value) == "404 unknown model 'kraken-new'"   # not the id list


@pytest.mark.parametrize("status, body", [
    # A registered model disabled for a reason: final, not a matter of time.
    (404, {"detail": "model 'kraken-new' is registered but not servable on this host"}),
    # No such route: a wrong ATR_TRAIN_GATEWAY_URL, not a slow gateway.
    (404, {"detail": "Not Found"}),
    (502, {"detail": "unknown model 'kraken-new'"}),
])
def test_every_other_error_is_not_a_reason_to_wait(monkeypatch, page, status, body):
    import httpx

    from atr_training.promote import http_recognizer

    _gateway_answering(monkeypatch, status, body)
    with pytest.raises(httpx.HTTPStatusError):
        http_recognizer("http://gw", "k")("kraken-new", page)


def test_no_page_means_no_promotion(tmp_path: Path):
    result = promote("m", None, lambda *_: "text")
    assert result.promoted is False and "no held-out page" in result.reason


# ── picking the page ────────────────────────────────────────────────────────
def test_the_page_comes_from_the_validation_manifest(tmp_path: Path):
    pages = tmp_path / "pages"
    pages.mkdir()
    for stem in ("000001_a", "000002_b"):
        (pages / f"{stem}.xml").write_text("<PcGts/>")
        (pages / f"{stem}.jpg").write_bytes(b"JPEG")
    (tmp_path / "pages_val.lst").write_text(
        f"{pages / '000002_b.xml'}\n", encoding="utf-8")

    assert held_out_page(tmp_path).name == "000002_b.jpg"


def test_a_manifest_entry_whose_image_is_gone_is_skipped(tmp_path: Path):
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "gone.xml").write_text("<PcGts/>")
    (pages / "here.xml").write_text("<PcGts/>")
    (pages / "here.jpg").write_bytes(b"JPEG")
    (tmp_path / "pages_val.lst").write_text(
        f"{pages / 'gone.xml'}\n{pages / 'here.xml'}\n", encoding="utf-8")

    assert held_out_page(tmp_path).name == "here.jpg"


def test_no_manifest_is_not_an_error(tmp_path: Path):
    assert held_out_page(tmp_path) is None


# What the gate is allowed to write — one registration's `enabled` — is tested
# with the writer, in tests/test_registration.py.


def test_the_gate_asks_for_a_model_that_is_not_enabled_yet(monkeypatch, tmp_path):
    """Without the header the gateway answers 404 for every fresh registration.

    That is how the gate behaved until the review of serving-atr-inference#138:
    the model under test is registered enabled: false, and the gateway refuses a
    disabled id to every caller. The value must match the gateway's.
    """
    import httpx

    from atr_training.promote import PROMOTION_GATE_HEADER, http_recognizer

    sent = {}

    def post(url, headers=None, **kwargs):
        sent.update(headers or {})
        return httpx.Response(200, json={"text": "ok"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    page = tmp_path / "page.jpg"
    page.write_bytes(b"\xff\xd8\xff")
    http_recognizer("http://127.0.0.1:8200", "k")("kraken-new", page)
    assert PROMOTION_GATE_HEADER == "X-ATR-Promotion-Gate"
    assert sent.get(PROMOTION_GATE_HEADER) == "1"

"""The promotion gate (#36) — advertise only what has actually served.

No gateway here: `promote` takes the recognizer as an argument, so every outcome
is exercised without a network.
"""

from __future__ import annotations

from pathlib import Path

from atr_training.promote import held_out_page, promote


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

    class Response:
        def raise_for_status(self): pass
        def json(self): return {"text": "ok"}

    def post(url, headers=None, **kwargs):
        sent.update(headers or {})
        return Response()

    monkeypatch.setattr(httpx, "post", post)
    page = tmp_path / "page.jpg"
    page.write_bytes(b"\xff\xd8\xff")
    http_recognizer("http://127.0.0.1:8200", "k")("kraken-new", page)
    assert PROMOTION_GATE_HEADER == "X-ATR-Promotion-Gate"
    assert sent.get(PROMOTION_GATE_HEADER) == "1"

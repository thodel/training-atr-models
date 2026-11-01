"""Bildwege verrotten; dieser Lauf soll es merken, bevor der Import es merkt (#199).

Gemessen am 08.10.2026: Weisthuemer nagelte einen archive.org-Knoten fest, der
nicht mehr antwortete (http=000), und dach-gt trug 27 Bild-URLs mit HTTP 404.
Diese Tests halten fest:

* fünf URLs je Quelle, gleichmässig gezogen, erste und letzte dabei, deterministisch;
* vier Lagen, nicht eine: unreachable, 404, 403, timeout — und was zu tun ist;
* berichtet wird nur eine Veränderung; der erste Lauf meldet, was schon tot ist;
* eine Quelle, die sich nicht auflisten lässt, ist nicht "in Ordnung";
* Umschreibungen sind Register-Einträge und gelten für jeden Bildweg.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import urllib.error
from pathlib import Path

import pytest

from atr_training import image_probe as ip
from atr_training.foreign_gt import (
    REWRITES,
    ImageInXml,
    PageFile,
    Source,
    UrlRewrite,
    image_urls,
    rewrite_url,
)

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location(
        "probe_image_paths", ROOT / "scripts" / "probe_image_paths.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["probe_image_paths"] = module
    spec.loader.exec_module(module)
    return module


# ── die Stichprobe ───────────────────────────────────────────────────────────

def test_five_urls_evenly_with_first_and_last():
    urls = {f"p{i:03d}.xml": f"https://x/{i}.jpg" for i in range(98)}
    got = ip.sample(urls)
    assert [p for p, _ in got] == ["p000.xml", "p024.xml", "p048.xml", "p073.xml", "p097.xml"]
    assert ip.sample(dict(reversed(list(urls.items())))) == got, "order-independent"
    assert len(ip.sample({"a": "u"})) == 1


# ── vier Lagen ───────────────────────────────────────────────────────────────

def _raise(exc):
    def fetch(url, timeout):
        raise exc
    return fetch


@pytest.mark.parametrize("exc,state", [
    (urllib.error.HTTPError("u", 404, "Not Found", {}, None), "404"),
    (urllib.error.HTTPError("u", 410, "Gone", {}, None), "404"),
    (urllib.error.HTTPError("u", 403, "Forbidden", {}, None), "403"),
    (urllib.error.HTTPError("u", 500, "Error", {}, None), "http_500"),
    (TimeoutError("timed out"), "timeout"),
    (socket.timeout("timed out"), "timeout"),
    (urllib.error.URLError(socket.timeout("timed out")), "timeout"),
    (urllib.error.URLError(ConnectionRefusedError(111, "refused")), "unreachable"),
    (urllib.error.URLError(socket.gaierror(-2, "Name or service not known")), "unreachable"),
    (ConnectionResetError(104, "reset"), "unreachable"),
    (ValueError("unknown url type"), "invalid_url"),
])
def test_each_failure_is_its_own_state(exc, state):
    assert ip.probe("https://x/1.jpg", fetch=_raise(exc)) == state


def test_a_2xx_is_ok_and_partial_content_is_too():
    assert ip.probe("u", fetch=lambda url, timeout: 206) == ip.OK
    assert ip.probe("u", fetch=lambda url, timeout: 200) == ip.OK


def test_the_four_states_each_say_what_to_do():
    for state in ("unreachable", "404", "403", "timeout"):
        assert ip.ADVICE[state]
    assert len({ip.ADVICE[s] for s in ("unreachable", "404", "403", "timeout")}) == 4


# ── nur Veränderung ──────────────────────────────────────────────────────────

def _snap(at, **sources):
    return ip.Snapshot(at, {k: dict(v) for k, v in sources.items()})


def test_an_unchanged_run_reports_nothing():
    a = _snap("t1", dtgt={"u1": "ok", "u2": "ok"}, dach={"u3": "404"})
    b = _snap("t2", dtgt={"u1": "ok", "u2": "ok"}, dach={"u3": "404"})
    assert ip.diff(a, b) == [] and ip.render([], a, b) == ""


def test_a_change_names_the_source_the_url_and_what_to_do():
    a = _snap("t1", weisthuemer={"u1": "ok", "u2": "ok"}, dtgt={"u3": "ok"})
    b = _snap("t2", weisthuemer={"u1": "ok", "u2": "unreachable"}, dtgt={"u3": "ok"})
    changes = ip.diff(a, b)
    assert [c.source_id for c in changes] == ["weisthuemer"]
    report = ip.render(changes, a, b)
    assert "| `weisthuemer` | 2/2 ok | 1/2 ok (1× unreachable) |" in report
    assert "**unreachable** — `u2`" in report
    assert ip.ADVICE["unreachable"] in report and "dtgt" not in report


def test_the_first_run_reports_what_is_already_dead_and_nothing_else():
    first = _snap("t0", dtgt={"u1": "ok"}, **{"dach-gt": {"u2": "ok", "u3": "404"}})
    changes = ip.diff(None, first)
    assert [c.source_id for c in changes] == ["dach-gt"]
    assert "ersten Lauf" in ip.render(changes, None, first)


# ── das Skript ───────────────────────────────────────────────────────────────

def test_the_script_keeps_state_and_writes_a_report_only_on_change(tmp_path, monkeypatch):
    mod = _script()
    states = {"https://h/a.jpg": "ok", "https://h/b.jpg": "ok"}
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    sources = [s for s in mod.SOURCES if s.id in ("DTGT", "dach-gt")]
    monkeypatch.setattr(mod, "SOURCES", sources)
    monkeypatch.setattr(mod, "ZENODO_SOURCES", ())

    def list_urls(source, workdir):
        if source.id == "dach-gt":
            raise RuntimeError("clone failed")
        return {f"p{i}.xml": url for i, url in enumerate(states)}

    def probe(url, timeout):
        return states[url]

    args = ["--state-dir", str(tmp_path), "--workdir", str(tmp_path / "src")]
    assert mod.main(args, list_urls=list_urls, probe=probe) == 0
    reports = tmp_path / "reports"
    first = list(reports.iterdir())
    assert len(first) == 1, "the first run reports the source that cannot be listed"
    assert "listing_failed" in first[0].read_text()

    first[0].unlink()
    assert mod.main(args, list_urls=list_urls, probe=probe) == 0
    assert not list(reports.iterdir()), "unchanged: no report"

    states["https://h/b.jpg"] = "403"
    assert mod.main(args, list_urls=list_urls, probe=probe) == 0
    report = next(reports.iterdir()).read_text()
    assert "`DTGT` | 2/2 ok | 1/2 ok (1× 403)" in report
    runs = [json.loads(line) for line in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert len(runs) == 3 and runs[-1]["changes"] == ["DTGT"]


# ── Umschreibungen als Register-Einträge ─────────────────────────────────────

def test_a_second_pinned_host_is_a_register_entry_not_a_function():
    rule = UrlRewrite(name="beispiel", pattern=r"https://node\d+\.example\.org/(?P<p>.+)",
                      template="https://example.org/{p}", reason="test")
    assert rewrite_url("https://node7.example.org/a/b.jpg", [rule]) == "https://example.org/a/b.jpg"
    assert rewrite_url("https://other.org/x.jpg", [rule]) == "https://other.org/x.jpg"
    assert [r.name for r in REWRITES] == ["archive.org-knoten"]


def test_the_rewrite_applies_to_a_url_from_the_document_too(tmp_path):
    pinned = ("https://ia903405.us.archive.org/view_archive.php?archive=/0/items/"
              "it/it_images.tar&file=p_0001.png")
    root = tmp_path / "r"
    root.mkdir()
    (root / "a.xml").write_text(f'<PcGts><Metadata externalRef="{pinned}"/></PcGts>')
    source = Source(id="x", origin="o", clone_url="c", xml_root="", licence="CC0-1.0",
                    licence_at="LICENSE", attribution="a", image_source="i",
                    script_kind="s", period="p", project="pr", target="t",
                    images=(ImageInXml(),))
    page = PageFile("a.xml", "a", 1, True)
    assert image_urls(root, source, [page]) == {
        "a.xml": "https://archive.org/download/it/it_images.tar/p_0001.png"}

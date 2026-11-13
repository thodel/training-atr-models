"""The daily report's view of both discoveries (serving-atr-inference#112, D3).

Neither half reached a reader: the dataset reports stayed on asteraix's disk and
the model report pointed at a closed issue. The timer now publishes one small
record to the share, which the results MCP reads with the standard library.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.error
from pathlib import Path

import pytest

from atr_training import discovery_digest as dd
from tests.test_discovery import BASE, _catalogue, _script, row

TODAY = dt.date(2026, 11, 16)

REPORT = """# Training material: 2 change(s) in `dh-unibe`

Compared 2026-11-15 06:41 CET → 2026-11-16 06:42 CET. Scores are estimates.

## `luzern`
- **new** — new dataset: 1200 pages, 3 project(s), period (1400, 1550)
  - against 1300–1600: score — → 0.80; **would enter the plan at rank 2** (1,200 pages)

## `rats`
- **projects** — +1 project: r3
  - against 1300–1600: score 0.90 → 0.92; moves within the plan
"""

CATALOGUE = """version: 1
candidates:
- {id: a/vlm, source: hf, first_seen: '2026-11-16', last_seen: '2026-11-16', verdict: new,
   backend: vllm, judgement: '1.0B, chat template', downloads: 900}
- {id: a/odd, source: hf, first_seen: '2026-11-16', last_seen: '2026-11-16', verdict: new,
   judgement: nothing identifies a backend that could load it}
- {id: '7', source: zenodo, first_seen: '2026-11-16', last_seen: '2026-11-16',
   verdict: rejected, by: discover_models.py, at: '2026-11-16', reason: no model file}
- {id: b/old, source: hf, first_seen: '2026-11-02', last_seen: '2026-11-16', verdict: new,
   backend: trocr}
- {id: c/kept, source: hf, first_seen: '2026-10-01', last_seen: '2026-11-16',
   verdict: watching, by: tobias, at: '2026-11-14', backend: vllm}
"""


def _state(tmp_path: Path, reports=None, runs=None) -> Path:
    state = tmp_path / "state"
    (state / "reports").mkdir(parents=True)
    for day, text in (reports or {}).items():
        (state / "reports" / f"{day}.md").write_text(text, encoding="utf-8")
    (state / "runs.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in (runs or [{"at": "2026-11-16 06:42 CET",
                                                         "changes": 0}])), encoding="utf-8")
    return state


# ── the dataset half ────────────────────────────────────────────────────────
def test_a_report_is_read_back_as_datasets_changes_and_plan_effects():
    parsed = dd.parse_report(REPORT)

    assert [d["dataset"] for d in parsed] == ["luzern", "rats"]
    assert parsed[0]["changes"][0]["kind"] == "new"
    assert parsed[0]["plan"] == ["would enter the plan at rank 2** (1,200 pages)"]
    assert parsed[1]["changes"][0]["what"] == "+1 project: r3"


def test_only_reports_inside_the_window_count(tmp_path):
    state = _state(tmp_path, reports={"2026-11-16": REPORT, "2026-11-02": REPORT})

    half = dd.dataset_half(state, TODAY, days=7)

    assert [r["date"] for r in half["reports"]] == ["2026-11-16"]
    assert half["changed"] == ["luzern", "rats"]
    assert half["last_run"]["at"] == "2026-11-16 06:42 CET"


# ── the model half ──────────────────────────────────────────────────────────
def test_usable_arrivals_are_named_and_the_rest_counted():
    half = dd.model_half(CATALOGUE, "main", TODAY)

    assert [c["key"] for c in half["new_usable"]] == ["hf:a/vlm"]
    assert half["new_without_backend"] == {"nichts weist auf ein Backend hin": 1}
    assert half["rejected_automatically"] == 1
    assert half["waiting_usable"] == 1, "b/old arrived before the window"
    assert half["counts"] == {"new": 3, "watching": 1, "rejected": 1, "adopted": 0}


def test_a_persons_verdict_is_news_and_the_machines_is_not():
    half = dd.model_half(CATALOGUE, "main", TODAY)

    assert [d["key"] for d in half["decided"]] == ["hf:c/kept"]


def test_the_branch_is_read_first_and_main_when_it_is_gone():
    asked = []

    def get(url):
        asked.append(url)
        if "discovery/catalogue" in url:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return CATALOGUE

    text, source, error = dd.fetch_catalogue(get)

    assert source == "main" and text == CATALOGUE
    assert error is None, "a missing branch is the state between a merge and a run"
    assert "discovery/catalogue" in asked[0]


def test_an_unreachable_github_is_said_not_hidden(tmp_path):
    def get(url):
        raise urllib.error.URLError("no route")

    record = dd.build(_state(tmp_path), today=TODAY, get=get)

    assert record["models"]["source"] is None
    assert "Katalog nicht lesbar" in record["summary"]


# ── the one line ────────────────────────────────────────────────────────────
def test_a_quiet_week_is_one_line(tmp_path):
    quiet = "version: 1\ncandidates: []\n"

    record = dd.build(_state(tmp_path), today=TODAY, get=lambda url: quiet)

    assert record["summary"].startswith("Nichts Neues in 7 Tagen")


def test_a_busy_week_names_the_datasets_and_counts_the_models(tmp_path):
    state = _state(tmp_path, reports={"2026-11-16": REPORT})

    record = dd.build(state, today=TODAY, get=lambda url: CATALOGUE)

    assert "luzern" in record["summary"] and "rats" in record["summary"]
    assert "1 neue als Basis verwendbar, 1 Urteil(e)" in record["summary"]


# ── publishing ──────────────────────────────────────────────────────────────
def test_publish_writes_the_record_and_mirrors_runs_and_reports(tmp_path):
    state = _state(tmp_path, reports={"2026-11-16": REPORT})
    share = tmp_path / "share" / "discovery"

    dd.publish(state, share, {"summary": "x"})

    assert json.loads((share / dd.FILENAME).read_text())["summary"] == "x"
    assert (share / "runs.jsonl").read_text() == (state / "runs.jsonl").read_text()
    assert (share / "reports" / "2026-11-16.md").read_text() == REPORT
    assert not list(share.rglob("*.part")), "a temporary file was left behind"


def test_the_script_publishes_after_its_run(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "_http_get", lambda url: "version: 1\ncandidates: []\n")
    state, share = tmp_path / "state", tmp_path / "share"

    assert _script("--catalogue", _catalogue(tmp_path / "a.json", BASE),
                   "--state-dir", state, "--publish", share) == 0

    record = json.loads((share / dd.FILENAME).read_text())
    assert record["datasets"]["last_run"]["note"] == "baseline"


def test_a_share_that_cannot_be_written_does_not_fail_the_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dd, "_http_get", lambda url: "version: 1\ncandidates: []\n")
    blocked = tmp_path / "file-not-dir"
    blocked.write_text("")

    code = _script("--catalogue", _catalogue(tmp_path / "a.json", BASE + [row("x")]),
                   "--state-dir", tmp_path / "state", "--publish", blocked / "discovery")

    assert code == 0
    assert "not published" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--dry-run"])
def test_a_dry_run_publishes_nothing(tmp_path, monkeypatch, flag):
    monkeypatch.setattr(dd, "_http_get", lambda url: pytest.fail("a dry run asked GitHub"))
    share = tmp_path / "share"

    _script("--catalogue", _catalogue(tmp_path / "a.json", BASE), "--state-dir",
            tmp_path / "state", "--publish", share, flag)

    assert not share.exists()

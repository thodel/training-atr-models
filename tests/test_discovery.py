"""#32: a scheduled pass reports changes in the training material, not an inventory.

The "Done when" of the issue, as tests:

* a run reports dataset changes, not a dataset inventory — the first run is a
  baseline and reports nothing;
* a project added to a watched dataset's card surfaces in the next run, with its
  effect on the corpus plan;
* an unchanged Hub produces an empty report.

And the two ways such a memory goes wrong, each pinned: a dataset whose card
could not be read this morning must not be "gone" (or, next time, "new"), and an
outage must not overwrite the memory with an empty org.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atr_training.discovery import (
    SIZE_MIN_PAGES,
    Snapshot,
    diff,
    implausible,
    render,
)

ROOT = Path(__file__).resolve().parents[1]

CARD = ("Geographical scope: Switzerland<br>Period: {period}<br>"
        "Languages: Middle High German, Early Modern German<br>"
        "Type of document: Protocols<br>Provenance: State Archive of Zurich")


def row(name: str, pages: int = 2000, projects=("a", "b"), period: str = "1400-1550",
        card: str | None = None) -> dict:
    return {"repo": f"dh-unibe/image-text_{name}", "pages": pages, "gb": 10.0,
            "card": card if card is not None else CARD.format(period=period),
            "projects": list(projects)}


def snap(rows, unreadable=None, listed=None, at="2026-10-22 06:40 CEST") -> Snapshot:
    listed = listed if listed is not None else [r["repo"] for r in rows] + list(unreadable or {})
    return Snapshot.from_catalogue("dh-unibe", listed, rows, unreadable, taken_at=at)


BASE = [row("rats", projects=("r1", "r2")), row("bullinger", projects=("b1", "b2")),
        row("charters", projects=("c1",), pages=800)]


# ── the "Done when" of #32 ──────────────────────────────────────────────────

def test_an_unchanged_hub_produces_an_empty_report():
    before, after = snap(BASE), snap(BASE, at="2026-10-23 06:40 CEST")

    changes = diff(before, after)

    assert changes == []
    assert render(changes, before, after) == ""


def test_an_added_project_surfaces_with_its_effect_on_the_plan():
    grown = [r if r["repo"] != "dh-unibe/image-text_charters"
             else row("charters", projects=("c1", "c2", "c3"), pages=2400) for r in BASE]

    changes = diff(snap(BASE), snap(grown))

    kinds = {c.kind for c in changes if c.repo.endswith("charters")}
    assert kinds == {"projects", "size"}
    projects = next(c for c in changes if c.kind == "projects")
    assert "2 project(s) added" in projects.what and "`c2`" in projects.what
    assert projects.consequence and "1300–1600" in projects.consequence[0]
    assert "plan" in projects.consequence[0]


def test_a_new_dataset_says_where_it_would_enter_the_plan():
    changes = diff(snap(BASE), snap(BASE + [row("luzern", projects=("l1",))]))

    (new,) = changes
    assert new.kind == "new" and new.repo.endswith("luzern")
    assert "would enter the plan at rank" in new.consequence[0]


# ── the kinds of change ─────────────────────────────────────────────────────

def test_a_dataset_the_hub_no_longer_lists_is_gone():
    changes = diff(snap(BASE), snap(BASE[:2]))

    assert [(c.kind, c.repo.split("_")[-1]) for c in changes] == [("gone", "charters")]


@pytest.mark.parametrize("old,new,reported", [
    (2000, 2000 + SIZE_MIN_PAGES, True),          # 100 pages and 5 %
    (2000, 2000 + SIZE_MIN_PAGES - 1, False),     # under the page floor
    (40_000, 40_000 + 150, False),                # 150 pages, but 0.4 %
    (300, 200, True),                             # shrinking counts as well
])
def test_a_size_change_needs_both_a_page_floor_and_a_share(old, new, reported):
    changes = diff(snap([row("rats", pages=old)]), snap([row("rats", pages=new)]))

    assert any(c.kind == "size" for c in changes) is reported


def test_a_card_whose_period_moves_changes_the_score():
    changes = diff(snap([row("zh", period="1803-1883")]),
                   snap([row("zh", period="1500-1600")]))

    (meta,) = changes
    assert meta.kind == "metadata" and "period" in meta.what
    assert "0.00 → 1.00" in meta.consequence[0]


def test_a_card_that_starts_declaring_machine_output_is_flagged():
    machine = CARD.format(period="1400-1550") + (
        "\n\nThis dataset DOES NOT CONTAIN GROUND TRUTH: it was automatically "
        "transcribed.")
    changes = diff(snap([row("rats")]), snap([row("rats", card=machine)]))

    assert any(c.kind == "metadata" and "MACHINE OUTPUT" in c.what for c in changes)


# ── the memory must not lie ─────────────────────────────────────────────────

def test_an_unreadable_card_is_not_a_deletion():
    bullinger = "dh-unibe/image-text_bullinger"
    today = snap([r for r in BASE if r["repo"] != bullinger],
                 unreadable={bullinger: "HTTPError: 503"})

    changes = diff(snap(BASE), today)

    assert [(c.kind, c.repo) for c in changes] == [("unreadable", bullinger)]
    kept = today.carry_forward(snap(BASE))
    assert kept.datasets[bullinger].projects == ("b1", "b2"), "last known state kept"


def test_an_unreadable_card_is_reported_once_and_its_return_is_not_new():
    bullinger = "dh-unibe/image-text_bullinger"
    first = snap(BASE)
    down = snap([r for r in BASE if r["repo"] != bullinger],
                unreadable={bullinger: "HTTPError: 503"}, at="day 2")
    memory = down.carry_forward(first)

    still_down = snap([r for r in BASE if r["repo"] != bullinger],
                      unreadable={bullinger: "HTTPError: 503"}, at="day 3")
    assert diff(memory, still_down) == [], "reported on day 2 already"

    back = snap(BASE, at="day 4")
    assert diff(still_down.carry_forward(memory), back) == [], \
        "its projects were known all along — nothing was added"


def test_a_listed_dataset_that_yields_no_row_and_no_error_is_unreadable_not_absent():
    """Silence must not become absence."""
    today = snap(BASE[:2], listed=[r["repo"] for r in BASE])

    assert "dh-unibe/image-text_charters" in today.unreadable
    assert all(c.kind != "gone" for c in diff(snap(BASE), today))


def test_a_new_dataset_without_a_readable_card_is_still_announced():
    today = snap(BASE, unreadable={"dh-unibe/image-text_new": "404: no README"})

    (new,) = diff(snap(BASE), today)
    assert new.kind == "new" and "not readable" in new.what


@pytest.mark.parametrize("listed,readable,trusted", [
    (3, 3, True),
    (0, 0, False),      # the Hub listed nothing — an outage, not a deletion
    (1, 1, False),      # a third of yesterday's org
    (3, 1, False),      # listed, but almost no card readable (auth gone)
])
def test_an_implausible_run_does_not_replace_the_memory(listed, readable, trusted):
    repos = [r["repo"] for r in BASE]
    today = snap(BASE[:readable], listed=repos[:listed])

    assert (implausible(snap(BASE), today) is None) is trusted


def test_the_snapshot_survives_a_round_trip():
    original = snap(BASE, unreadable={"dh-unibe/image-text_x": "503"})

    again = Snapshot.from_json(json.loads(original.dumps()))

    assert again == original


def test_a_snapshot_of_another_version_is_refused_not_misread():
    data = json.loads(snap(BASE).dumps())
    data["version"] = 99

    with pytest.raises(ValueError, match="version"):
        Snapshot.from_json(data)


# ── the script ──────────────────────────────────────────────────────────────

def _script(*args) -> int:
    import importlib.util
    spec = importlib.util.spec_from_file_location("discover_datasets",
                                                  ROOT / "scripts" / "discover_datasets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main([str(a) for a in args])


def _catalogue(path: Path, rows, unreadable=None) -> Path:
    path.write_text(json.dumps({"listed": [r["repo"] for r in rows] + list(unreadable or {}),
                                "rows": rows, "unreadable": unreadable or {}}))
    return path


def test_the_first_run_is_a_baseline_and_writes_no_report(tmp_path, capsys):
    state = tmp_path / "state"

    assert _script("--catalogue", _catalogue(tmp_path / "c.json", BASE),
                   "--state-dir", state) == 0

    assert (state / "snapshot.json").is_file()
    assert not (state / "reports").exists()
    assert "baseline" in capsys.readouterr().out


def test_a_change_writes_a_report_and_no_change_writes_none(tmp_path):
    state = tmp_path / "state"
    _script("--catalogue", _catalogue(tmp_path / "a.json", BASE), "--state-dir", state)
    _script("--catalogue", _catalogue(tmp_path / "b.json", BASE), "--state-dir", state)
    assert not (state / "reports").exists(), "an unchanged Hub wrote a report"

    _script("--catalogue", _catalogue(tmp_path / "c.json", BASE + [row("luzern")]),
            "--state-dir", state)

    (report,) = (state / "reports").iterdir()
    assert "luzern" in report.read_text()
    runs = [json.loads(line) for line in (state / "runs.jsonl").read_text().splitlines()]
    assert [r["changes"] for r in runs] == [0, 0, 1]


def test_an_outage_exits_1_and_keeps_the_memory(tmp_path):
    state = tmp_path / "state"
    _script("--catalogue", _catalogue(tmp_path / "a.json", BASE), "--state-dir", state)
    before = (state / "snapshot.json").read_text()

    assert _script("--catalogue", _catalogue(tmp_path / "b.json", []),
                   "--state-dir", state) == 1

    assert (state / "snapshot.json").read_text() == before


def test_a_dry_run_writes_nothing(tmp_path):
    state = tmp_path / "state"

    _script("--catalogue", _catalogue(tmp_path / "a.json", BASE), "--state-dir", state,
            "--dry-run")

    assert not state.exists()


# ── the units ───────────────────────────────────────────────────────────────

def test_the_timer_runs_the_service_and_the_service_runs_this_script():
    units = ROOT / "deploy" / "systemd"
    timer = (units / "atr-discovery.timer").read_text()
    service = (units / "atr-discovery.service").read_text()

    assert "Unit=atr-discovery.service" in timer
    assert "Persistent=true" in timer
    exec_line = next(line for line in service.splitlines() if line.startswith("ExecStart="))
    assert "scripts/discover_datasets.py" in exec_line
    assert (ROOT / "scripts" / "discover_datasets.py").is_file()
    # The shared hub cache must stay the effective one (see atr-train.service).
    assert "HF_HOME" not in "\n".join(
        line for line in service.splitlines() if not line.startswith("#"))

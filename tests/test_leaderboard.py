"""One table per sweep, with what the numbers mean written on it (#116).

The four things #116 asks to have *on* the table rather than beside it are four
things that were lost beside it before: the data version (the old corpus was
deleted on 16.09.2026 and nothing in its numbers says so), the noise floor (a
ranking without its resolution reads as a ranking), anomalies (h256's 0.5591 is
a collapse, not a weak run), and the cost (h256 is ~12x h48 per step, so a gain
of 0.0021 for 70 % more compute is a different statement).

The band tests are the subtle ones. Ties have to be non-transitive: chaining
a≈b and b≈c into a≈c would merge a whole field through small steps even when its
ends are far apart, and the table would then claim nothing is distinguishable
from anything.
"""

import json
from pathlib import Path

import pytest
import yaml

from atr_training.leaderboard import (
    ANOMALY, PROMOTED, RUNNING, Row, band_ranks, render, rows_for,
)
from atr_training.sweep_driver import SweepDriver, SweepState
from atr_training.sweep_manifest import parse_manifest

from tests.test_sweep_driver import FakeTrainer, scores_for

DIGEST = "sha256:0123456789abcdef"

MANIFEST = f"""
name: kraken-medieval-augment-01
data:
  train: shard_00
  eval: german_test
  digest: "{DIGEST}"
  datasets:
    - {{hf_repo: "dh-unibe/x", granularity: line}}
budget:
  steps: 4000
  rungs: [12, 4, 1]
noise_floor:
  value: 0.0085
  measured_on: "{DIGEST}"
  seeds: [42, 43, 44, 45]
  commit: "1a429b3abcdef01"
  steps: 2000
base:
  spec: "[1,48,0,1 Cr3,3,32]"
  lrate: 1.0e-4
  batch_size: 256
  accumulate_grad_batches: 1
  augment: false
  normalization: NFD
axes:
  augment: [false, true]
  normalization: [NFD, NFC]
  lrate: [3.0e-5, 1.0e-4, 3.0e-4]
"""


def manifest(**edits):
    raw = yaml.safe_load(MANIFEST)
    for dotted, value in edits.items():
        node = raw
        *path, leaf = dotted.split(".")
        for step in path:
            node = node[step]
        if value is None:
            node.pop(leaf, None)
        else:
            node[leaf] = value
    return parse_manifest(raw, source="test")


def swept(m=None, cers=None, *, decorate=True, path: Path | None = None):
    """Run a whole sweep against the fake trainer and return (manifest, state)."""
    m = m or manifest()
    cers = cers or [0.200 + i * 0.003 for i in range(12)]
    state = SweepState.for_manifest(m, "benchmark_cer", path)
    state.train_lines = 120_000
    SweepDriver(m, state, FakeTrainer(scores_for(m, cers)), sleep=lambda _s: None).run()
    if decorate:                       # the fake trainer records no timing
        for rung in state.results.values():
            for index, entry in enumerate(rung.values()):
                entry["minutes"] = 40.0 + index * 7
                entry["commit"] = "ff9effd8d99d0dae"
        state.save()
    return m, state


# ── bands ───────────────────────────────────────────────────────────────────
def _rows(*scores: float) -> list[Row]:
    return [Row(config_id=f"c{i}", rung=0, axes={}, steps=4000, raw=None,
                score=s, minutes=None, status=PROMOTED)
            for i, s in enumerate(scores)]


def test_rows_further_apart_than_the_floor_get_their_own_ranks():
    rows = _rows(0.90, 0.80, 0.70)

    band_ranks(rows, 0.0085)

    assert [r.rank for r in rows] == [1, 2, 3]
    assert not any(r.tied for r in rows)


def test_rows_the_material_cannot_separate_share_a_rank():
    """#116: "nicht Platz 3 gegen Platz 4"."""
    rows = _rows(0.9000, 0.8990, 0.8985)

    band_ranks(rows, 0.0085)

    assert [r.rank for r in rows] == [1, 1, 1]
    assert all(r.tied for r in rows)


def test_a_band_does_not_chain():
    """The subtle one. 0.900, 0.894, 0.888: each gap is 0.006 < 0.0085, so
    chaining would tie all three — but the ends are 0.012 apart, which the
    material *can* resolve. A band is measured against its own leader, so the
    third row starts a new one."""
    rows = _rows(0.900, 0.894, 0.888)

    band_ranks(rows, 0.0085)

    assert [r.rank for r in rows] == [1, 1, 3]


def test_the_next_band_is_numbered_by_position_not_by_band_count():
    """Two tied at the top means the next row is third, not second — otherwise
    the rank stops counting configurations."""
    rows = _rows(0.9000, 0.8990, 0.5)

    band_ranks(rows, 0.0085)

    assert [r.rank for r in rows] == [1, 1, 3]


def test_without_a_floor_every_row_stands_alone():
    """No measurement, no tie: claiming two results are indistinguishable needs
    a number saying so."""
    rows = _rows(0.9000, 0.8999, 0.8998)

    band_ranks(rows, None)

    assert [r.rank for r in rows] == [1, 2, 3]
    assert not any(r.tied for r in rows)


def test_an_unscored_row_gets_no_rank():
    rows = _rows(0.9, 0.8)
    rows.append(Row(config_id="cx", rung=0, axes={}, steps=4000, raw=None,
                    score=None, minutes=None, status=RUNNING))

    band_ranks(rows, 0.0085)

    assert rows[-1].rank is None


# ── the rows ────────────────────────────────────────────────────────────────
def test_a_row_carries_the_axis_values_that_differ():
    """Not the base: it is identical on every row and belongs in the header."""
    m, state = swept()

    row = rows_for(m, state)[0]

    assert set(row.axes) == {"augment", "normalization", "lrate"}
    assert "spec" not in row.axes


def test_a_row_carries_the_budget_of_its_rung():
    m, state = swept()
    rows = rows_for(m, state)

    assert {r.steps for r in rows if r.rung == 0} == {4000}
    assert {r.steps for r in rows if r.rung == 1} == {12000}


def test_the_cost_of_a_run_is_on_the_row():
    """h256 is about twelve times h48 per step; a gain bought with 70 % more
    compute is a different claim."""
    m, state = swept()

    assert all(r.minutes is not None for r in rows_for(m, state))


def test_a_collapse_is_marked_as_an_anomaly_not_as_a_last_place():
    """Otherwise a collapse looks like a result that happened to be poor."""
    cers = [0.200 + i * 0.003 for i in range(11)] + [0.95]
    m, state = swept(cers=cers)

    rows = rows_for(m, state)
    flagged = [r for r in rows if r.status == ANOMALY]
    assert flagged, [r.status for r in rows]
    assert flagged[0].raw == pytest.approx(0.95)


def test_rows_inside_a_band_are_ordered_by_score():
    """They share a rank, so nothing else orders them — and listing 0.2030
    above 0.2000 under one `=1` reads as a mistake even when the rank is
    honest."""
    m, state = swept()

    band = [r for r in rows_for(m, state) if r.rung == 0 and r.rank == 1]
    assert len(band) > 1
    assert band == sorted(band, key=lambda r: r.raw)


# ── the header ──────────────────────────────────────────────────────────────
def test_the_data_version_is_on_the_table():
    """Two leaderboards over different corpora are not comparable, and that has
    to be visible without asking."""
    m, state = swept()

    assert DIGEST in render(m, state)


def test_a_measured_floor_is_shown_with_its_seeds_and_commit():
    m, state = swept()

    text = render(m, state)

    assert "0.0085" in text
    assert "[42, 43, 44, 45]" in text
    assert "1a429b3abcde" in text


def test_an_unmeasured_floor_says_that_no_gap_is_known_to_be_one():
    m, state = swept(manifest(**{"noise_floor": None}))

    assert "not measured" in render(m, state)


def test_a_floor_without_provenance_is_marked_unstated():
    """A bare number in the manifest. The ties rest on it, so the table says
    nobody can check it."""
    m, state = swept(manifest(**{"noise_floor": 0.0085}))

    assert "UNSTATED" in render(m, state)


def test_the_commit_is_on_the_table():
    m, state = swept()

    assert "ff9effd8d99d" in render(m, state)


def test_a_sweep_spanning_two_commits_says_so():
    """A confound the ladder cannot tell from a configuration effect."""
    m, state = swept()
    first = next(iter(state.results["0"].values()))
    first["commit"] = "0000000000000000"

    text = render(m, state)

    assert "2 different commits" in text
    assert "not comparable" in text


def test_jobs_without_a_commit_are_named_as_such():
    m, state = swept(decorate=False)

    assert "no commit recorded" in render(m, state)


# ── the state of the sweep ──────────────────────────────────────────────────
def test_a_finished_sweep_is_not_called_unfinished():
    m, state = swept()

    assert "unfinished" not in render(m, state).lower()


def test_a_sweep_stopped_mid_rung_still_produces_a_table():
    """#116's third criterion: no error, no empty file. A multi-day sweep spends
    most of its life in this state, and that is when somebody looks."""
    m = manifest()
    state = SweepState.for_manifest(m, "benchmark_cer")
    state.train_lines = 120_000
    for index, config in enumerate(m.configs()[:5]):
        state.put(0, config.config_id, {
            "job_id": f"job-{index}", "status": "completed" if index < 4 else "running",
            "score": 0.8 - index / 100, "raw": 0.2 + index / 100,
            "metric": "benchmark_cer", "minutes": 40.0, "commit": "ff9effd8d99d",
        })

    text = render(m, state)

    assert "unfinished" in text.lower()
    assert "rung 0" in text
    assert text.count("|") > 20, "the table itself is missing"


def test_a_running_row_is_shown_as_running_rather_than_omitted():
    m = manifest()
    state = SweepState.for_manifest(m, "benchmark_cer")
    config = m.configs()[0]
    state.put(0, config.config_id, {"job_id": "j", "status": "running", "score": None,
                                    "raw": None, "metric": "benchmark_cer"})

    assert "running" in render(m, state)


def test_a_rung_decided_inside_the_noise_is_called_out_once():
    """Repeating the paragraph per rung buries the point under its own
    restatement."""
    cers = [0.200 + i * 0.0001 for i in range(12)]
    m, state = swept(cers=cers)

    text = render(m, state)

    assert text.count("Decided by noise") == 1
    assert "rung 0 on" in text and "rung 1 on" in text


def test_an_overridden_guard_is_visible_on_the_row():
    """This driver never sets `force`, but a CER from a run known not to
    converge must never be read as an ordinary one, and the table is where it
    would be read."""
    m, state = swept()
    first = next(iter(state.results["0"].values()))
    first["overrides"] = ["convergence_override"]

    assert "overridden" in render(m, state)


# ── the script ──────────────────────────────────────────────────────────────
def _run(*args):
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    return subprocess.run([sys.executable, str(root / "scripts" / "sweep_leaderboard.py"),
                           *args], capture_output=True, text=True, timeout=120)


@pytest.fixture
def on_disk(tmp_path: Path):
    path = tmp_path / "sweep.yaml"
    path.write_text(MANIFEST, encoding="utf-8")
    state_path = tmp_path / "sweep.yaml.state.json"
    swept(path=state_path)
    return path, state_path


def test_the_script_writes_markdown_beside_the_manifest(on_disk):
    path, _ = on_disk

    result = _run(str(path))

    assert result.returncode == 0, result.stderr
    written = path.with_suffix(path.suffix + ".leaderboard.md").read_text()
    assert written.startswith("# kraken-medieval-augment-01")
    assert DIGEST in written


def test_the_script_can_print_instead(on_disk):
    path, _ = on_disk

    result = _run(str(path), "--stdout")

    assert result.returncode == 0, result.stderr
    assert DIGEST in result.stdout
    assert not path.with_suffix(path.suffix + ".leaderboard.md").exists()


def test_the_script_uses_the_metric_the_sweep_actually_ran_with(on_disk):
    """Not one given on the command line: `SweepState.load` refuses a metric
    that differs, and a leaderboard reports a sweep rather than changing it."""
    path, state_path = on_disk
    recorded = json.loads(state_path.read_text())
    recorded["metric"] = "cer"
    state_path.write_text(json.dumps(recorded))

    result = _run(str(path), "--stdout")

    assert result.returncode == 0, result.stderr
    assert "**metric** cer" in result.stdout


def test_a_sweep_that_never_ran_says_so_rather_than_writing_an_empty_table(tmp_path):
    path = tmp_path / "sweep.yaml"
    path.write_text(MANIFEST, encoding="utf-8")

    result = _run(str(path))

    assert result.returncode == 1
    assert "has not run yet" in result.stderr
    assert not path.with_suffix(path.suffix + ".leaderboard.md").exists()

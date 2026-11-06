"""Hand-run measurements leave a record that live() and results() read (#385).

The records are written with the helper the measure scripts use and read with
the probe the MCP runs, so the two sides are checked against each other, not
against a fixture that only one of them knows.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from atr_results_mcp import probe
from atr_results_mcp.probe import Paths
from atr_training import measurement_record as mr
from atr_training.measurement_record import measurement

REPO = Path(__file__).resolve().parents[1]
ASTERAIX = {"ATR_TRAIN_HOST_ID": "asteraix", "CUDA_VISIBLE_DEVICES": "1"}


@pytest.fixture
def cluster(tmp_path: Path) -> Paths:
    share = tmp_path / "share" / "Textrecognition_Training"
    (share / "training_folder" / "jobs").mkdir(parents=True)
    (tmp_path / "scratch" / "runs" / "jobs").mkdir(parents=True)
    return Paths(user="u", home=tmp_path / "home", scratch=tmp_path / "scratch", share=share)


def share_root(paths: Paths) -> Path:
    return paths.shared_jobs_root.parent / mr.MEASUREMENTS_DIRNAME


def at(hour: int, minute: int = 0):
    return lambda: datetime(2026, 11, 6, hour, minute, tzinfo=timezone.utc)


def records(root: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(root.glob("*/measurement.json"))]


# ── where the records go ─────────────────────────────────────────────────────

def test_the_root_sits_beside_the_job_store(tmp_path):
    jobs = tmp_path / "training_folder" / "jobs"
    jobs.mkdir(parents=True)
    assert mr.measurements_root({"ATR_TRAIN_JOBS_ROOT": str(jobs)}) == \
        tmp_path / "training_folder" / "measurements"
    assert mr.measurements_root({"ATR_MEASUREMENTS_ROOT": str(tmp_path / "x"),
                                 "ATR_TRAIN_JOBS_ROOT": str(jobs)}) == tmp_path / "x"


def test_no_store_means_no_record_not_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "DEFAULT_JOBS_ROOT", tmp_path / "not-mounted" / "jobs")
    assert mr.measurements_root({}) is None
    with measurement("measure_x", argv=["m"], env={}) as record:
        pass
    assert record["status"] == "completed"


def test_an_unwritable_store_does_not_stop_the_measurement(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("")
    with measurement("measure_x", argv=["m"], root=blocker / "sub", env={}) as record:
        ran = True
    assert ran and record["status"] == "completed"
    assert "measurement record not written" in capsys.readouterr().err


def test_the_output_flag_is_read_from_argv(tmp_path):
    out = tmp_path / "peak.json"
    assert mr.output_from_argv(["--seconds", "240", "--out", str(out)]) == str(out)
    assert mr.output_from_argv([f"--report={out}"]) == str(out)
    assert mr.output_from_argv(["--seconds", "240"]) is None


# ── the record across a run ──────────────────────────────────────────────────

def test_a_run_is_running_at_the_start_and_completed_at_the_end(tmp_path):
    root = tmp_path / "measurements"
    argv = ["measure_kraken_arms.py", "--out", str(tmp_path / "peak.json")]
    times = iter([at(9, 18)(), at(9, 21)()])
    with measurement("measure_kraken_arms", argv=argv, root=root, env=ASTERAIX,
                     now=lambda: next(times)):
        [during] = records(root)
        assert during["status"] == "running" and during["ended_at"] is None
    [after] = records(root)
    assert after["id"] == during["id"] and after["kind"] == "measurement"
    assert after["status"] == "completed" and after["exit_code"] == 0
    assert after["host"] == "asteraix" and after["gpus"] == "1"
    assert after["started_at"] == "2026-11-06T09:18:00+00:00"
    assert after["ended_at"] == "2026-11-06T09:21:00+00:00"
    assert after["output"] == str(tmp_path / "peak.json")


@pytest.mark.parametrize("exc, status, code", [
    (SystemExit(2), "failed", 2),
    (SystemExit(0), "completed", 0),
    (RuntimeError("CUDA out of memory"), "failed", 1),
    (KeyboardInterrupt(), "aborted", 130),
])
def test_how_a_run_ended_is_in_the_record(tmp_path, exc, status, code):
    root = tmp_path / "m"
    with pytest.raises(type(exc)):
        with measurement("measure_x", argv=["m"], root=root, env=ASTERAIX):
            raise exc
    [record] = records(root)
    assert (record["status"], record["exit_code"]) == (status, code)
    assert ("error" in record) == isinstance(exc, RuntimeError)


def test_sigterm_ends_the_record_as_aborted(tmp_path):
    """A plain ``kill`` is how a measurement is stopped by hand; it must not
    leave a record that reads as running forever."""
    root = tmp_path / "m"
    code = (
        "import os, signal, sys, time\n"
        f"sys.path.insert(0, {str(REPO / 'src')!r})\n"
        "from pathlib import Path\n"
        "from atr_training.measurement_record import measurement\n"
        f"with measurement('measure_x', argv=['m'], root=Path({str(root)!r}), env={{}}):\n"
        "    os.kill(os.getpid(), signal.SIGTERM)\n"
        "    time.sleep(5)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], timeout=30)
    assert proc.returncode == 128 + signal.SIGTERM
    [record] = records(root)
    assert record["status"] == "aborted" and record["ended_at"]


def test_every_measure_script_is_wrapped():
    scripts = sorted((REPO / "scripts").glob("measure_*.py"))
    assert len(scripts) >= 9
    for script in scripts:
        text = script.read_text()
        assert "with measurement(Path(__file__).stem):" in text, script.name


# ── what the MCP reads ───────────────────────────────────────────────────────

def test_a_running_measurement_on_asteraix_is_live(cluster):
    root = share_root(cluster)
    with measurement("measure_vlm_arms", argv=["m", "--out", "/tmp/r.json"], root=root,
                     env=ASTERAIX, now=at(8)):
        answer = probe.live(cluster, host="asteraix")
        assert answer["rows"] == []
        [row] = answer["measurements"]
        assert row["name"] == "measure_vlm_arms" and row["status"] == "running"
        assert row["host"] == "asteraix" and row["gpus"] == "1" and row["pid"] == os.getpid()
        assert row["started_cest"] == "2026-11-06 09:00 CET"
        assert probe.live(cluster, host="ubelix")["measurements"] == []
        assert probe.results(cluster)["measurements"] == []
    assert probe.live(cluster)["measurements"] == []


def test_a_finished_measurement_is_in_results_with_its_output(cluster):
    root = share_root(cluster)
    times = iter([at(8)(), at(10, 30)()])
    with measurement("measure_kraken_arms", argv=["m", "--out", "/tmp/peak.json"], root=root,
                     env=ASTERAIX, now=lambda: next(times)):
        pass
    answer = probe.results(cluster)
    assert answer["rows"] == [], "a measurement is not a training row"
    [row] = answer["measurements"]
    assert row["status"] == "completed"
    assert row["output"] == str(Path("/tmp/peak.json").resolve()), "absolute, as written"
    assert row["ended_cest"] == "2026-11-06 11:30 CET"
    assert probe.results(cluster, status="failed")["measurements"] == []


def test_a_record_without_an_end_stamp_is_never_finished(cluster):
    """kill -9 or a lost box: the start was written, the end never was. Whatever
    the record says otherwise, it reads as running, not as a result."""
    run_dir = share_root(cluster) / "20261106T080000Z-measure_x-abc123"
    run_dir.mkdir(parents=True)
    (run_dir / "measurement.json").write_text(json.dumps({
        "kind": "measurement", "id": run_dir.name, "name": "measure_x",
        "host": "asteraix", "status": "completed", "started_at": "2026-11-06T08:00:00+00:00",
        "ended_at": None}))
    assert [m["status"] for m in probe.live(cluster)["measurements"]] == ["running"]
    assert probe.results(cluster)["measurements"] == []


def test_a_job_record_beside_it_is_not_read_as_a_measurement(cluster):
    other = share_root(cluster) / "stray"
    other.mkdir(parents=True)
    (other / "measurement.json").write_text(json.dumps({"request": {}, "status": "training"}))
    assert probe.live(cluster)["measurements"] == []

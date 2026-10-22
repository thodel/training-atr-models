"""#206 after the fact: the image a record's number came from, or unknown.

A Slurm log prints ``== SIF: <image> (transformers <version>)`` since
10.10.2026. A record from before carries no runtime; where its log has the
line, the runtime is recovered into its GPU stages; where no log has it, the
record stays as it is and the table says *unknown* — never an image guessed
from the model type.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from atr_training import runtime_provenance
from atr_training.runtime_provenance import SIDECAR, apply_to_job, recover_job, sif_line

JOB = "20260925T045500Z-ladder-med-qwen35-27b"
TF5 = "/storage/homefs/th19c587/ubelix/vlm-train-tf5.sif"


def test_the_module_parses_as_python_3_9():
    ast.parse(Path(runtime_provenance.__file__).read_text(encoding="utf-8"), feature_version=(3, 9))


def _job(store: Path, job_id: str = JOB, *, status: str = "completed", runtime=None,
         stages=("prepare", "train", "test")) -> Path:
    job = store / job_id
    job.mkdir(parents=True)
    record_stages = []
    for name in stages:
        stage = {"name": name, "status": "completed"}
        if runtime is not None:
            stage["runtime"] = runtime
        record_stages.append(stage)
    (job / "job.json").write_text(json.dumps({"id": job_id, "status": status,
                                              "stages": record_stages}), encoding="utf-8")
    return job


def _log(logs: Path, name: str, job_id: str, *, sif: str | None = TF5,
         version: str | None = "5.17.0", attempt: int = 1) -> Path:
    lines = ["== code: pinned to fa02f2d 2026-10-10 x"]
    if sif:
        lines.append("== SIF: %s%s" % (sif, " (transformers %s)" % version if version else ""))
    lines.append("== attempt %d — job %s (training) on gnode26" % (attempt, job_id))
    path = logs / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_the_sif_line_is_read_with_its_version():
    assert sif_line(["== SIF: %s (transformers 5.17.0)" % TF5]) == (TF5, "5.17.0")
    assert sif_line(["== SIF: %s (transformers nicht abfragbar)" % TF5]) == (TF5, None)
    assert sif_line(["== SIF: %s" % TF5]) == (TF5, None)
    assert sif_line(["== code: x", "nothing"]) is None
    # A requeue appends: the last line is the attempt that stands.
    assert sif_line(["== SIF: /a.sif (transformers 4.57.6)", "== SIF: %s (transformers 5.17.0)" % TF5]) \
        == (TF5, "5.17.0")


def test_a_record_with_a_sif_line_in_its_log_is_recovered(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    job = _job(tmp_path / "jobs")
    _log(logs, "train-17527777.out", JOB)

    recovery = recover_job(job, logs)

    assert recovery.verdict == "recover"
    assert (recovery.image, recovery.transformers) == (TF5, "5.17.0")
    assert recovery.stages == ("train", "test"), "the GPU stages, not prepare"
    assert recovery.log.endswith("train-17527777.out")


def test_the_newest_log_with_a_sif_line_wins(tmp_path):
    """A record resumed by hand later may have run in another image; the last
    attempt wrote the stages that stand."""
    import os
    import time
    logs = tmp_path / "logs"
    logs.mkdir()
    job = _job(tmp_path / "jobs")
    old = _log(logs, "train-1.out", JOB, sif="/old/vlm-train.sif", version="4.57.6")
    new = _log(logs, "train-2.out", JOB)
    os.utime(old, (time.time() - 100, time.time() - 100))
    os.utime(new, None)

    recovery = recover_job(job, logs)
    assert recovery.image == TF5 and recovery.log == str(new)


def test_without_a_sif_line_the_record_is_unknown_not_guessed(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    job = _job(tmp_path / "jobs")
    _log(logs, "train-16830477.out", JOB, sif=None)

    recovery = recover_job(job, logs)

    assert recovery.verdict == "unknown"
    assert recovery.image is None and "before 10.10.2026" in recovery.why
    with pytest.raises(RuntimeError, match="nothing to apply"):
        apply_to_job(job, recovery)
    assert "runtime" not in json.loads((job / "job.json").read_text())["stages"][1]


def test_a_log_of_another_record_is_not_believed(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    job = _job(tmp_path / "jobs")
    _log(logs, "train-9.out", "20261001T000000Z-someone-else")
    assert recover_job(job, logs).verdict == "unknown"


def test_a_record_that_carries_a_runtime_is_left_alone(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    job = _job(tmp_path / "jobs", runtime={"image": None, "transformers": "4.57.6"})
    _log(logs, "train-1.out", JOB)
    assert recover_job(job, logs).verdict == "already"


def test_a_live_job_is_left_alone(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    job = _job(tmp_path / "jobs", status="training")
    _log(logs, "train-1.out", JOB)
    assert recover_job(job, logs).verdict == "skip"


def test_applying_writes_the_gpu_stages_and_the_sidecar_once(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    job = _job(tmp_path / "jobs")
    _log(logs, "train-1.out", JOB)
    before = json.loads((job / "job.json").read_text())

    apply_to_job(job, recover_job(job, logs))

    after = json.loads((job / "job.json").read_text())
    by_name = {s["name"]: s for s in after["stages"]}
    assert by_name["train"]["runtime"] == {"image": TF5, "transformers": "5.17.0"}
    assert by_name["test"]["runtime"] == {"image": TF5, "transformers": "5.17.0"}
    assert "runtime" not in by_name["prepare"]
    for stage in after["stages"]:
        stage.pop("runtime", None)
    assert after == before, "nothing but the runtime changed"
    sidecar = json.loads((job / SIDECAR).read_text())
    assert sidecar["from"].endswith("train-1.out") and sidecar["issue"] == "#206"
    assert recover_job(job, logs).verdict == "already"


def test_the_dry_run_writes_nothing_and_counts_the_unknown(tmp_path, capsys):
    import importlib.util
    script = Path(__file__).resolve().parents[1] / "scripts" / "runtime_provenance.py"
    spec = importlib.util.spec_from_file_location("runtime_provenance_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logs = tmp_path / "logs"
    logs.mkdir()
    known = _job(tmp_path / "jobs", JOB)
    _log(logs, "train-1.out", JOB)
    _job(tmp_path / "jobs", "20260903T104406Z-qwen3vl-sg-missiven-v1")
    record = (known / "job.json").read_text()

    assert module.main(["--store", str(tmp_path / "jobs"), "--logs", str(logs)]) == 0

    assert (known / "job.json").read_text() == record
    out = capsys.readouterr().out
    assert "1 recovered" in out and "1 unknown" in out and "dry run, nothing written" in out
    assert "vlm-train-tf5.sif" in out

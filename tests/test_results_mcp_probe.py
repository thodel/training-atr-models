"""#174: the probe half of the results MCP, driven on a temporary cluster.

Slurm answers are canned from submit02 on 07.10.2026; the job store, logs and
artefacts are built under tmp_path. Nothing here needs the cluster, and the
first test keeps the probe runnable on the cluster's Python 3.9.
"""
from __future__ import annotations

import ast
import json
import time
from pathlib import Path

import pytest

from atr_results_mcp import probe
from atr_results_mcp.probe import Paths, ProbeError

PROBE_FILE = Path(probe.__file__)

SQUEUE = """\
17545210|line-dataset|epyc2|job_gratis|PENDING|0:00|1-00:00:00|N/A|MaxCpuRunMinsPerUser||8|cpu=8,mem=48G,node=1,billing=17|
17545199|line-dataset|epyc2|job_gratis|RUNNING|27:58|12:00:00|2026-10-07T20:52:24|None|bnode046|8|cpu=8,mem=48G,node=1,billing=17|
17488077|fedmin1|gpu|job_gratis|RUNNING|50:35|3:00:00|2026-10-07T15:08:57|None|gnode34|16|cpu=16,gres/gpu:h100=1|
17520966|train|gpu-invest|job_gpu_preemptable|PENDING|0:00|1-00:00:00|N/A|ReqNodeNotAvail, UnavailableNodes:gnode20||16|cpu=16,mem=90G,gres/gpu:h100=1|
"""

SINFO = """\
gnode20|drained*|Hardware failure|gpu:rtx3090:8
gnode20|drained*|Hardware failure|gpu:rtx3090:8
"""

SACCT = """\
JobID|JobName|State|Elapsed|Timelimit|Start|End|NodeList|ExitCode|NCPUS
17328125|line-dataset|TIMEOUT|12:00:16|12:00:00|2026-10-05T15:02:51|2026-10-06T03:03:07|bnode065|0:0|8
16830477|train|COMPLETED|00:00:03|1-00:00:00|2026-10-06T00:54:04|2026-10-06T00:54:07|gnode26|0:0|16
17488076|fedmin1|COMPLETED|00:53:43|03:00:00|2026-10-07T08:55:07|2026-10-07T09:48:50|gnode28|0:0|16
17328126|line-dataset|OUT_OF_MEMORY|00:47:41|12:00:00|2026-10-05T17:02:47|2026-10-05T17:50:28|bnode052|0:125|8
17545199|line-dataset|RUNNING|00:27:58|12:00:00|2026-10-07T20:52:24|Unknown|bnode046|0:0|8
17520797|line-dataset|CANCELLED by 33813|00:54:21|1-00:00:00|2026-10-07T16:21:19|2026-10-07T17:15:40|cnode01|0:0|8
"""

SCONTROL = """\
JobId=17520966 JobName=train
   UserId=th19c587(33813) GroupId=wbkolleg(1234) MCS_label=N/A
   JobState=PENDING Reason=ReqNodeNotAvail,_UnavailableNodes:gnode20 Dependency=(null)
   RunTime=00:00:00 TimeLimit=1-00:00:00 TimeMin=N/A
   SubmitTime=2026-10-07T16:22:52 EligibleTime=2026-10-07T16:22:52
   StartTime=Unknown EndTime=Unknown Deadline=N/A
   Partition=gpu-invest AllocNode:Sid=submit02:1
   ReqNodeList=(null) ExcNodeList=(null)
   NodeList=
   NumNodes=1-1 NumCPUs=16 NumTasks=1 CPUs/Task=16 ReqB:S:C:T=0:0:*:*
   ReqTRES=cpu=16,mem=90G,node=1,billing=34,gres/gpu=1,gres/gpu:h100=1
   AllocTRES=(null)
   Command=/storage/homefs/th19c587/training-atr-models/ubelix/train.sbatch
   WorkDir=/storage/homefs/th19c587/training-atr-models
   StdOut=/storage/homefs/th19c587/ubelix/logs/train-17520966.out
"""

TRAIN_LOG_HEAD = """\
== code: pinned to ff9effd 2026-09-30 K3: der Rauschboden wird zuschreibbar (#115) (#127)
== attempt 1 — job 20260930T170315Z-ladder-xix-gemma4-12b (training) on gnode25
Updating files: 100% (239/239), done.
[2026-10-05T15:40:31.083] error: *** JOB 16830477 ON gnode25 CANCELLED AT 2026-10-05T15:40:31 DUE TO PREEMPTION ***
== attempt 4 — job 20260930T170315Z-ladder-xix-gemma4-12b (training) on gnode26
2026-10-05 16:34:58.314 | WARNING  | atr_training.runner_base:_stage:334 - stage train: running ff9effd8d99d, but the job was created with 0870e50c79e0 — this stage does not run the code the job was submitted with
2026-10-05 23:42:19.898 | INFO     | __main__:_test:413 - CER 0.2122 / WER 0.2820976491862568 over 200 samples
== code: pinned to ff9effd 2026-09-30 K3: der Rauschboden wird zuschreibbar (#115) (#127)
== job 20260930T170315Z-ladder-xix-gemma4-12b is already completed — nothing to do, not re-running
"""


def canned(outputs: dict[str, str]):
    """A runner that answers by the command's name and records what was asked."""
    calls: list[list[str]] = []

    def run(argv: list[str]) -> str:
        calls.append(argv)
        key = argv[0]
        if key not in outputs:
            raise ProbeError("%s: not canned" % key)
        return outputs[key]

    run.calls = calls  # type: ignore[attr-defined]
    return run


def write_job(root: Path, job_id: str, *, status: str = "completed", model_id: str | None = None,
              base_model: str = "google/gemma-4-12B-it", granularity: str = "line",
              cer: float | None = 0.2122, draw: str | None = "a\nb\nc\n", created: str = "0870e50c79e0",
              stage_commit: str | None = "ff9effd8d99d", finished: str | None = "2026-10-05T21:42:21Z",
              artefact: str | None = None, store: str = "scratch", host: str | None = "ubelix",
              engine: str = "vllm", stage: str | None = None,
              extra: dict | None = None) -> Path:
    """A record in the UBELIX scratch store (``root`` is the scratch directory) or,
    with ``store="share"``, in the shared store (``root`` is the share's
    ``Textrecognition_Training``). ``host=None`` writes no stamp at all."""
    if store == "share":
        job_dir = root / "training_folder" / "jobs" / job_id
    else:
        job_dir = root / "runs" / "jobs" / job_id
    (job_dir / "data").mkdir(parents=True)
    record = {
        "id": job_id, "status": status, "stage": stage,
        "request": {"model_id": model_id or job_id.split("-", 1)[1], "engine": engine,
                    "base_model": base_model,
                    "params": {"granularity": granularity, "load_in_4bit": True, "epochs": 1,
                               "lora_r": 64}},
        "code": {"commit": created + "0" * 28, "dirty": False},
        "created_at": "2026-09-30T17:03:18.891717Z", "finished_at": finished,
        "metrics": None if cer is None else {"cer": cer, "wer": 0.28, "length_ratio": 0.82,
                                              "truncated_cer": 0.2153, "samples": 200},
        "stages": [] if stage_commit is None else [
            {"name": "train", "status": "completed", "started_at": "2026-10-05T14:34:58Z",
             "finished_at": "2026-10-05T21:38:12Z", "exit_code": 0,
             "code": {"commit": stage_commit + "0" * 28, "dirty": False}}],
        "progress": {"artefact": artefact},
        "registration": "not registered: this ran as Slurm job 16830477, and a Slurm job never "
                        "writes the registry (#17).\nPYTHONPATH=...",
    }
    if host is not None:
        record["host"] = host
    record.update(extra or {})
    (job_dir / "job.json").write_text(json.dumps(record), encoding="utf-8")
    if draw is not None:
        (job_dir / "data" / "val_eval.jsonl").write_text(draw, encoding="utf-8")
    return job_dir


@pytest.fixture
def cluster(tmp_path: Path) -> Paths:
    home = tmp_path / "home"
    (home / "ubelix" / "logs").mkdir(parents=True)
    scratch = tmp_path / "scratch"
    (scratch / "runs" / "jobs").mkdir(parents=True)
    (scratch / "expA" / "artefacts").mkdir(parents=True)
    share = tmp_path / "share" / "Textrecognition_Training"
    (share / "training_folder" / "jobs").mkdir(parents=True)
    return Paths(user="th19c587", home=home, scratch=scratch, share=share)


# ── the cluster's Python ────────────────────────────────────────────────────
def test_the_probe_parses_as_python_3_9():
    """submit02 runs 3.9.25; a match statement or a walrus-free 3.10 grammar fails there,
    not here, and only on the morning the report needs it."""
    ast.parse(PROBE_FILE.read_text(encoding="utf-8"), feature_version=(3, 9))


def test_the_probe_imports_only_the_standard_library():
    tree = ast.parse(PROBE_FILE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"hashlib", "json", "os", "re", "subprocess", "sys", "datetime",
                        "pathlib", "typing", "zoneinfo", "__future__"}


# ── time and duration parsing ───────────────────────────────────────────────
@pytest.mark.parametrize("text, seconds", [
    ("27:58", 27 * 60 + 58), ("12:00:16", 12 * 3600 + 16), ("1-00:00:00", 86400),
    ("3:00:00", 3 * 3600), ("0:00", 0), ("UNLIMITED", None), ("Unknown", None), ("", None),
])
def test_slurm_durations(text, seconds):
    assert probe.slurm_seconds(text) == seconds


def test_utc_stamps_become_zurich_time_with_a_label():
    assert probe.utc_to_cest("2026-10-05T21:42:21.235264Z") == "2026-10-05 23:42 CEST"
    assert probe.utc_to_cest("2026-12-01T10:00:00Z") == "2026-12-01 11:00 CET"
    assert probe.utc_to_cest(None) is None


# ── logs ────────────────────────────────────────────────────────────────────
def test_log_edges_skip_the_middle_of_a_large_log(tmp_path):
    path = tmp_path / "big.out"
    path.write_text("first\n" + ("x" * 100 + "\n") * 5000 + "last\n")
    edges = probe.read_log_edges(path, head_bytes=1024, tail_bytes=1024)
    assert edges["truncated"] is True
    assert edges["head"][0] == "first"
    assert edges["tail"][-1] == "last"
    assert all(len(line) <= 100 for line in edges["tail"])  # no byte-window fragment


def test_the_cause_comes_from_the_last_attempt_not_the_first():
    """A requeued job appends attempts to one log: the head says PREEMPTION, the tail
    says why the attempt sacct reports ended."""
    lines = TRAIN_LOG_HEAD.splitlines()
    notes = probe.log_notes(lines[:4], lines[4:])
    assert "already completed — nothing to do" in notes["cause"]
    assert notes["code"].startswith("== code: pinned to ff9effd")
    assert notes["code_drift"] == [
        "stage train: running ff9effd8d99d, but the job was created with 0870e50c79e0"]


def test_progress_ignores_the_worktree_checkout_counter():
    assert probe.last_progress(["Updating files: 100% (239/239), done."]) is None
    found = probe.last_progress(["Updating files: 100% (239/239), done.", "2350/2751"])
    assert found == {"done": 2350, "total": 2751, "line": "2350/2751"}


def test_progress_reads_a_tqdm_bar_and_skips_dates():
    line = "Generating train split:  60%|█████▉| 94567/158525 [53:03<31:04, 34.30 examples/s]"
    assert probe.last_progress([line])["done"] == 94567
    assert probe.last_progress(["2026-10-07T17:15:40 done 10/10/2026"]) is None


# ── the wall ────────────────────────────────────────────────────────────────
def test_a_projection_within_fifteen_percent_of_the_wall_is_at_risk():
    # 05.10.2026: 150/164 at 43 min of 60 was called "makes it, barely" and timed out.
    at_risk = probe.judge_wall(43 * 60, 60 * 60, {"done": 150, "total": 164})
    assert at_risk["verdict"] == "fits"  # projects to 47 min: this one truly fits
    assert probe.judge_wall(50 * 60, 60 * 60, {"done": 150, "total": 164})["verdict"] == "at risk"
    assert probe.judge_wall(50 * 60, 60 * 60, {"done": 100, "total": 164})["verdict"] \
        == "will hit the wall"
    assert probe.judge_wall(100, 3600, None)["verdict"] == "unknown"


# ── queue ───────────────────────────────────────────────────────────────────
def test_queue_explains_reasons_and_judges_running_jobs(cluster):
    (cluster.logs / "fedmin1-17488077.out").write_text("25/2751\n" + "2350/2751\n")
    run = canned({"squeue": SQUEUE, "sinfo": SINFO})
    answer = probe.queue(run, cluster)
    by_id = {j["slurm_job_id"]: j for j in answer["jobs"]}
    assert answer["running"] == 2 and answer["pending"] == 2
    assert "11,520 CPU-minutes" in by_id["17545210"]["reason_meaning"]
    fedmin = by_id["17488077"]
    assert fedmin["progress"]["done"] == 2350
    assert fedmin["wall"]["verdict"] == "fits"
    assert by_id["17545199"]["wall"]["verdict"] == "unknown"  # no log on disk
    node = by_id["17520966"]["unavailable_nodes"][0]
    assert node == {"node": "gnode20", "state": "drained*", "reason": "Hardware failure",
                    "gres": "gpu:rtx3090:8"}
    assert run.calls[0][:3] == ["squeue", "-u", "th19c587"]


def test_queue_asks_sinfo_only_when_a_node_is_named(cluster):
    run = canned({"squeue": SQUEUE.splitlines()[0] + "\n"})
    probe.queue(run, cluster)
    assert [c[0] for c in run.calls] == ["squeue"]


# ── finished ────────────────────────────────────────────────────────────────
def test_finished_pulls_causes_and_marks_requeue_stubs(cluster):
    (cluster.logs / "train-16830477.out").write_text(TRAIN_LOG_HEAD)
    (cluster.logs / "line-dataset-17328125.out").write_text(
        "cropped a -> b\n[2026-10-06T03:03:07.021] error: *** JOB 17328125 ON bnode065 "
        "CANCELLED AT 2026-10-06T03:03:07 DUE TO TIME LIMIT ***\n")
    run = canned({"sacct": SACCT})
    answer = probe.finished(run, cluster, days=2)
    by_id = {j["slurm_job_id"]: j for j in answer["jobs"]}
    assert "17545199" not in by_id  # still running
    assert "DUE TO TIME LIMIT" in by_id["17328125"]["cause"]
    stub = by_id["16830477"]
    assert stub["short_run"] is True
    assert "not a run" in stub["note"]
    assert "already completed" in stub["cause"]
    assert stub["code_drift"]
    assert by_id["17488076"]["short_run"] is False and "cause" not in by_id["17488076"]
    assert by_id["17328126"]["state"] == "OUT_OF_MEMORY"
    assert run.calls[0][:5] == ["sacct", "-u", "th19c587", "-S", "now-2days"]


# ── job records ─────────────────────────────────────────────────────────────
def test_job_summary_converts_times_and_reports_drift(cluster):
    write_job(cluster.scratch, "20260930T170315Z-ladder-xix-gemma4-12b")
    answer = probe.job(cluster, "20260930T170315Z-ladder-xix-gemma4-12b")
    assert answer["finished_cest"] == "2026-10-05 23:42 CEST"
    assert answer["base_model"] == "google/gemma-4-12B-it"
    assert answer["load_in_4bit"] is True
    assert answer["metrics"]["cer"] == 0.2122
    assert answer["code_drift"] == [
        "stage train: running ff9effd8d99d, but the job was created with 0870e50c79e0"]
    assert answer["slurm_jobs"] == ["16830477"]
    assert answer["registration"].startswith("not registered")
    assert answer["draw"]["lines"] == 3


def test_a_missing_job_is_an_error_not_an_exception(cluster):
    assert "error" in probe.job(cluster, "nope")


def test_results_filters_and_fingerprints(cluster):
    write_job(cluster.scratch, "20260930T170315Z-ladder-xix-gemma4-12b")
    write_job(cluster.scratch, "20260926T080902Z-ladder-xix-gemma4-e4b",
              base_model="google/gemma-4-E4B-it", cer=0.1110)
    write_job(cluster.scratch, "20261001T093909Z-olmocr2-7b-medieval-german-page-v1",
              base_model="allenai/olmOCR-2-7B-1025", granularity="page", cer=0.3492,
              draw="p1\np2\n")
    write_job(cluster.scratch, "20260925T045500Z-ladder-med-qwen35-27b", status="training",
              base_model="Qwen/Qwen3.5-27B", cer=None, draw=None, finished=None)
    everything = probe.results(cluster)["rows"]
    assert [r["id"][:8] for r in everything] == ["20260925", "20260926", "20260930", "20261001"]
    assert probe.results(cluster, granularity="page")["rows"][0]["draw_lines"] == 2
    gemma = probe.results(cluster, base_model="gemma")["rows"]
    assert {r["cer"] for r in gemma} == {0.1110, 0.2122}
    assert gemma[0]["draw_md5"] == gemma[1]["draw_md5"]  # same three lines
    assert probe.results(cluster, status="training")["rows"][0]["cer"] is None


def test_draw_names_the_jobs_that_share_it(cluster):
    write_job(cluster.scratch, "20260930T170315Z-ladder-xix-gemma4-12b")
    write_job(cluster.scratch, "20260926T080902Z-ladder-xix-gemma4-e4b", cer=0.1110)
    write_job(cluster.scratch, "20260916T090417Z-qwen3vl-german-xix-v2", draw="x\n" * 200)
    answer = probe.draw(cluster, "20260930T170315Z-ladder-xix-gemma4-12b")
    assert answer["draw"]["lines"] == 3
    assert [s["id"] for s in answer["shared_with"]] == ["20260926T080902Z-ladder-xix-gemma4-e4b"]
    assert "error" in probe.draw(cluster, "20260925T045500Z-nothing")


def test_prepared_lists_waiting_corpora_and_their_successors(cluster):
    write_job(cluster.scratch, "20260925T045500Z-ladder-med-qwen35-9b", status="training",
              cer=None, draw=None, finished=None)
    write_job(cluster.scratch, "20260930T211455Z-ladder-med-qwen35-9b", cer=0.1213)
    write_job(cluster.scratch, "20260925T045500Z-ladder-med-qwen35-27b", status="training",
              cer=None, draw=None, finished=None)
    rows = {r["id"]: r for r in probe.prepared(cluster)["rows"]}
    assert set(rows) == {"20260925T045500Z-ladder-med-qwen35-9b",
                         "20260925T045500Z-ladder-med-qwen35-27b"}
    assert rows["20260925T045500Z-ladder-med-qwen35-9b"]["superseded_by"] == [
        "20260930T211455Z-ladder-med-qwen35-9b"]
    assert rows["20260925T045500Z-ladder-med-qwen35-27b"]["superseded_by"] == []
    assert rows["20260925T045500Z-ladder-med-qwen35-27b"]["scratch_purge_cest"]


# ── the second store: what asteraix's service writes on the share (#156) ─────
def test_results_read_both_stores_and_name_the_host(cluster):
    """#156 measured `find ~/atr-cache -name job.json` → 0 and concluded asteraix
    has no job store. It has one: the service writes training_folder/jobs on the
    research share, which the login node mounts. Only hand-run measurements
    have no record."""
    write_job(cluster.scratch, "20260930T170315Z-ladder-xix-gemma4-12b")
    write_job(cluster.share, "20261007T131819Z-kraken-german-xix-v1", store="share",
              host="asteraix", engine="kraken", base_model="bifrost", cer=0.0912,
              draw="k1\nk2\n")
    write_job(cluster.share, "20260807T100000Z-thun-kurrent-v1", store="share", host=None,
              engine="kraken", base_model="kurrent", cer=0.2350, draw=None,
              finished="2026-08-13T10:00:00Z")
    rows = {r["id"]: r for r in probe.results(cluster)["rows"]}
    assert rows["20260930T170315Z-ladder-xix-gemma4-12b"]["host"] == "ubelix"
    assert rows["20261007T131819Z-kraken-german-xix-v1"]["host"] == "asteraix"
    assert rows["20261007T131819Z-kraken-german-xix-v1"]["engine"] == "kraken"
    assert rows["20261007T131819Z-kraken-german-xix-v1"]["cer"] == 0.0912
    # A share record without a stamp is the retired idhefix trainer's (#15).
    assert rows["20260807T100000Z-thun-kurrent-v1"]["host"] == "idhefix"
    stores = probe.results(cluster)["stores"]
    assert stores["scratch"] == {"root": str(cluster.jobs_root), "mounted": True, "jobs": 1}
    assert stores["share"]["mounted"] is True and stores["share"]["jobs"] == 2


def test_an_absent_share_is_said_not_shown_as_zero_asteraix_rows(tmp_path):
    """The share is a mount. Away, its store must read as absent (#165), and the
    scratch rows still come."""
    paths = Paths(user="u", home=tmp_path / "home", scratch=tmp_path / "scratch",
                  share=tmp_path / "not-mounted")
    write_job(paths.scratch, "20260930T170315Z-ladder-xix-gemma4-12b")
    answer = probe.results(paths)
    assert [r["id"] for r in answer["rows"]] == ["20260930T170315Z-ladder-xix-gemma4-12b"]
    assert answer["stores"]["share"]["mounted"] is False
    assert answer["stores"]["share"]["jobs"] is None
    assert "not found" in answer["stores"]["share"]["note"]
    assert probe.live(paths)["stores"]["share"]["mounted"] is False


def test_job_and_draw_find_a_record_in_either_store(cluster):
    write_job(cluster.scratch, "20260930T170315Z-ladder-xix-gemma4-12b")
    write_job(cluster.share, "20261007T131819Z-kraken-german-xix-v1", store="share",
              host="asteraix", engine="kraken", draw="a\nb\nc\n", cer=0.0912)
    answer = probe.job(cluster, "20261007T131819Z-kraken-german-xix-v1")
    assert answer["host"] == "asteraix" and answer["store"] == "share"
    assert answer["metrics"]["cer"] == 0.0912
    assert probe.job(cluster, "20260930T170315Z-ladder-xix-gemma4-12b")["store"] == "scratch"
    missing = probe.job(cluster, "nope")
    assert "error" in missing and set(missing["stores"]) == {"scratch", "share"}
    # The same three lines on both sides: the draw query crosses the stores.
    shared = probe.draw(cluster, "20260930T170315Z-ladder-xix-gemma4-12b")["shared_with"]
    assert shared == [{"id": "20261007T131819Z-kraken-german-xix-v1", "host": "asteraix",
                       "status": "completed", "cer": 0.0912}]
    assert "any store" in probe.draw(cluster, "nope")["error"]


def test_prepared_gives_no_purge_date_to_a_job_on_the_share(cluster):
    write_job(cluster.scratch, "20260925T045500Z-ladder-med-qwen35-27b", status="training",
              cer=None, draw=None, finished=None)
    write_job(cluster.share, "20261007T131819Z-kraken-german-xix-v1", store="share",
              host="asteraix", engine="kraken", status="training", cer=None, draw=None,
              finished=None)
    rows = {r["id"]: r for r in probe.prepared(cluster)["rows"]}
    assert rows["20260925T045500Z-ladder-med-qwen35-27b"]["scratch_purge_cest"]
    assert rows["20261007T131819Z-kraken-german-xix-v1"]["scratch_purge_cest"] is None
    assert rows["20261007T131819Z-kraken-german-xix-v1"]["host"] == "asteraix"


def test_live_reads_asteraix_from_the_record_and_the_stage_log(cluster):
    """asteraix has no Slurm: its queue is the record's status and stage, and its
    progress is the counter at the end of logs/<stage>.log — the state a person
    read by hand on 10.10.2026 (kraken-german-xix-v1 in training on card 1)."""
    job_dir = write_job(cluster.share, "20261007T131819Z-kraken-german-xix-v1", store="share",
                        host="asteraix", engine="kraken", base_model="bifrost",
                        status="training", stage="train", cer=None, draw=None, finished=None,
                        extra={"pid": 41213, "gpus": [1], "started_at": "2026-10-07T13:20:00Z",
                               "updated_at": "2026-10-10T07:58:00Z",
                               "progress": {"epoch": 3, "epochs": 50, "total_steps": 12000,
                                            "val_accuracy": 0.9412,
                                            "peak_gpu_mib": {"gpu1": {"own_mib": 26773}}}})
    (job_dir / "logs").mkdir()
    (job_dir / "logs" / "train.log").write_text("stage 3/50\n 1200/4000\n")
    write_job(cluster.share, "20260807T100000Z-thun-kurrent-v1", store="share", host=None,
              engine="kraken", status="training", cer=None, draw=None, finished=None)
    write_job(cluster.scratch, "20260925T045500Z-ladder-med-qwen35-27b", status="training",
              cer=None, draw=None, finished=None)
    write_job(cluster.scratch, "20260930T170315Z-ladder-xix-gemma4-12b")  # completed: not live

    answer = probe.live(cluster, host="asteraix")
    assert [r["id"] for r in answer["rows"]] == ["20261007T131819Z-kraken-german-xix-v1"]
    row = answer["rows"][0]
    assert row["stage"] == "train" and row["pid"] == 41213 and row["gpus"] == [1]
    assert row["epoch"] == 3 and row["epochs"] == 50 and row["total_steps"] == 12000
    assert row["peak_gpu_mib"] == {"gpu1": {"own_mib": 26773}}
    assert row["progress"] == {"done": 1200, "total": 4000, "line": "1200/4000"}
    assert row["log"].endswith("logs/train.log") and row["log_modified_cest"]
    assert row["updated_cest"] == "2026-10-10 09:58 CEST"

    everyone = probe.live(cluster)
    assert [(r["host"], r["id"]) for r in everyone["rows"]] == [
        ("asteraix", "20261007T131819Z-kraken-german-xix-v1"),
        ("idhefix", "20260807T100000Z-thun-kurrent-v1"),
        ("ubelix", "20260925T045500Z-ladder-med-qwen35-27b")]
    # No stage log at all: the counter is absent, not zero.
    assert everyone["rows"][2]["progress"] is None and everyone["rows"][2]["log"] is None
    assert everyone["rows"][2]["slurm_jobs"] == ["16830477"]


def test_live_falls_back_to_the_runner_log_for_an_in_process_stage(cluster):
    """prepare and the VLM compile run in-process and write runner.log only, the
    same case the runner's _failure_log covers."""
    job_dir = write_job(cluster.share, "20261010T090000Z-x", store="share", host="asteraix",
                        status="preparing", stage="prepare", cer=None, draw=None, finished=None)
    (job_dir / "logs").mkdir()
    (job_dir / "logs" / "prepare.log").write_text("")  # created, never written
    (job_dir / "logs" / "runner.log").write_text("streaming pages 37/1640\n")
    row = probe.live(cluster, host="asteraix")["rows"][0]
    assert row["log"].endswith("logs/runner.log")
    assert row["progress"]["done"] == 37


# ── deadlines ───────────────────────────────────────────────────────────────
def write_artefact(root: Path, key: str, *, built_at: float, pinned: bool, job_id: str,
                   claims: dict[str, float]) -> None:
    folder = root / "expA" / "artefacts" / (key + "0" * (64 - len(key)))
    folder.mkdir(parents=True)
    (folder / "artefact.json").write_text(json.dumps({
        "key": key + "0" * (64 - len(key)), "pinned": pinned, "built_at": built_at,
        "bytes": 39919527785, "last_used": built_at, "job_id": job_id, "claims": claims}))


def test_deadlines_flag_an_expiring_artefact_with_a_live_claimant(cluster):
    tomorrow_expiry = time.time() - 6 * 86400  # built six days ago, one day left
    write_artefact(cluster.scratch, "aaaa", built_at=tomorrow_expiry, pinned=False,
                   job_id="20261001T090507Z-x-page-v1", claims={"20261001T090507Z-x-page-v1": 1.0})
    write_artefact(cluster.scratch, "bbbb", built_at=tomorrow_expiry, pinned=True,
                   job_id="20261001T093909Z-x-page-v1", claims={})
    write_artefact(cluster.scratch, "cccc", built_at=tomorrow_expiry, pinned=False,
                   job_id="20261001T092000Z-y-page-v1", claims={"20261001T092000Z-y-page-v1": 1.0})
    write_job(cluster.scratch, "20261001T090507Z-x-page-v1", status="training", cer=None,
              draw=None, finished=None, granularity="page")
    write_job(cluster.scratch, "20261001T093909Z-x-page-v1", cer=0.3492, granularity="page")
    write_job(cluster.scratch, "20261001T092000Z-y-page-v1", status="training", cer=None,
              draw=None, finished=None, granularity="page")
    run = canned({"df": "Filesystem 1024-blocks Used Available Capacity Mounted on\n"
                        "rs_gpfs 375809638400 315634860032 60174778368 84% /storage\n"})
    answer = probe.deadlines(run, cluster)
    by_key = {a["key"]: a for a in answer["artefacts"]}
    # x-page-v1's claimant is superseded by the completed 093909Z job: a leftover.
    assert by_key["aaaa00000000"]["at_risk"] is False
    assert by_key["aaaa00000000"]["claimant_superseded_by"] == {
        "20261001T090507Z-x-page-v1": ["20261001T093909Z-x-page-v1"]}
    assert by_key["bbbb00000000"]["expires_cest"] is None  # pinned
    assert by_key["cccc00000000"]["at_risk"] is True  # nobody finished y-page-v1
    assert 0 < by_key["cccc00000000"]["expires_in_days"] <= 1.1
    assert answer["home"]["use_percent"] == "84%"
    assert {p["id"] for p in answer["scratch_purge"]} == {"20261001T090507Z-x-page-v1",
                                                          "20261001T092000Z-y-page-v1"}


# ── log, report, slurm_job, checkout ────────────────────────────────────────
def test_log_filters_noise_and_keeps_the_last_raw_lines(cluster):
    (cluster.logs / "line-dataset-17545199.out").write_text(
        "== code: pinned to 5a21fb5\n"
        "2026-10-07 21:00:00.000 | DEBUG    | atr_training.cropping:write_crops:120 - cropped\n"
        "Generating train split: 100%|█| 152732/152786 [32:37<00:00, 89.70 examples/s]\n")
    answer = probe.log(cluster, "17545199", lines=5)
    assert answer["head"] == ["== code: pinned to 5a21fb5"]
    assert answer["tail"] == ["== code: pinned to 5a21fb5"]
    assert answer["last_raw"][-1].startswith("Generating train split")
    assert answer["notes"]["progress"]["done"] == 152732
    assert "error" in probe.log(cluster, "0")


def test_report_lists_an_evalset_by_cer_and_shows_quantisation(cluster):
    root = cluster.jobs_root / "evalset-federal-minutes"
    root.mkdir()
    (root / "report_qwen35-4b-v2.json").write_text(json.dumps(
        {"cer": 0.0680, "samples": 2751, "truncated_at_cap": 0, "base_model": "Qwen/Qwen3.5-4B",
         "adapter": "/x/checkpoints/20260918T070300Z-qwen3.5-4b-german-xix-v2", "by_source": None}))
    (root / "report_gemma4-e4b-xix.json").write_text(json.dumps(
        {"cer": 0.1024, "samples": 2751, "truncated_at_cap": 3, "load_in_4bit": True,
         "base_model": "google/gemma-4-E4B-it", "adapter": "/x/checkpoints/e4b"}))
    rows = probe.report(cluster)["rows"]
    assert [r["tag"] for r in rows] == ["qwen35-4b-v2", "gemma4-e4b-xix"]
    assert rows[0]["load_in_4bit"] is None and rows[1]["load_in_4bit"] is True
    assert rows[0]["adapter"] == "20260918T070300Z-qwen3.5-4b-german-xix-v2"
    one = probe.report(cluster, tag="gemma4-e4b-xix")
    assert one["report"]["cer"] == 0.1024 and "by_source" not in one["report"]
    assert "error" in probe.report(cluster, evalset="nope")


def test_slurm_job_parses_scontrol_and_names_the_drained_node(cluster):
    run = canned({"scontrol": SCONTROL, "sinfo": SINFO})
    answer = probe.slurm_job(run, cluster, "17520966")
    assert answer["Command"].endswith("ubelix/train.sbatch")
    assert answer["ReqTRES"].endswith("gres/gpu:h100=1")
    assert answer["unavailable_nodes"][0]["gres"] == "gpu:rtx3090:8"
    assert "missing capacity" in answer["reason_meaning"]


def test_slurm_job_falls_back_to_accounting_once_slurm_forgot_it(cluster):
    def run(argv):
        if argv[0] == "scontrol":
            raise ProbeError("scontrol show job: exit 1: slurm_load_jobs error: Invalid job id specified")
        if argv[0] == "sacct":
            return SACCT
        raise AssertionError(argv)
    answer = probe.slurm_job(run, cluster, "17328125")
    assert answer["error"] == "not in scontrol any more"
    assert answer["accounting"]["state"] == "TIMEOUT"


def test_checkout_reports_head_against_origin(cluster):
    answers = {
        ("rev-parse", "HEAD"): "c98048cdcec7deadbeef\n",
        ("log", "-1", "--format=%h %s"): "c98048c Das 12B (#175)\n",
        ("status", "--porcelain"): " M ubelix/x.py\n",
        ("fetch", "-q", "origin"): "",
        ("rev-list", "--left-right", "--count", "HEAD...origin/main"): "0\t2\n",
        ("log", "-1", "--format=%h %s", "origin/main"): "abcdef0 newer\n",
    }

    def run(argv):
        assert argv[:3] == ["git", "-C", str(cluster.checkout)]
        return answers[tuple(argv[3:])]

    answer = probe.checkout(run, cluster)
    assert answer == {"checkout": str(cluster.checkout), "head": "c98048cdcec7",
                      "head_subject": "c98048c Das 12B (#175)", "origin_main": "abcdef0 newer",
                      "fetched": True, "ahead": 0, "behind": 2, "dirty_files": 1,
                      "note": "submit.sh refuses a checkout that is behind origin/main or dirty"}


# ── dispatch ────────────────────────────────────────────────────────────────
def test_dispatch_knows_every_tool_and_refuses_the_rest(cluster):
    run = canned({"squeue": "", "sacct": SACCT, "df": "x\nfs 1 1 1 1% /\n"})
    for cmd in ("queue", "finished", "results", "prepared", "deadlines", "live"):
        assert isinstance(probe.dispatch(cmd, {}, run, cluster), dict)
    assert probe.dispatch("live", {"host": "asteraix"}, run, cluster)["host"] == "asteraix"
    with pytest.raises(ProbeError):
        probe.dispatch("sbatch", {}, run, cluster)


def test_main_prints_json_and_reports_errors_as_json(cluster, capsys, monkeypatch):
    monkeypatch.setattr(probe, "shell_runner", lambda argv, timeout=60.0: "")
    monkeypatch.setattr(probe, "Paths", lambda: cluster)
    assert probe.main(["-", "queue", "{}"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["jobs"] == [] and "_host" in out
    assert probe.main(["-", "sbatch", "{}"]) == 1
    assert json.loads(capsys.readouterr().out)["error"].startswith("unknown command")


# ── research-storage (die tägliche Routine, statt ssh von Hand) ──────────────
def test_storage_delegates_to_the_deployed_script(tmp_path, monkeypatch):
    """Der Probe ist Zusteller, nicht Urheber der Regeln.

    Welche Verzeichnisse wiederherstellbar sind, was nur die einzige lokale
    Kopie ist und was abgeleitet — das steht in `research_storage.py` und darf
    nicht ein zweites Mal hier stehen. Zwei Fassungen derselben Regeln wären
    eine Gabelung, und genau diese Regeln haben sich am 09.10.2026 zweimal
    geändert, weil Messungen sie widerlegt haben.
    """
    script = tmp_path / "ubelix" / "research_storage.py"
    script.parent.mkdir(parents=True)
    script.write_text("# Platzhalter\n")
    calls = []

    def run(argv, timeout=60.0):
        calls.append(argv)
        return json.dumps({"usage": {"free": 1, "total": 10, "free_fraction": 0.1},
                           "below_threshold": False, "suggestions": []})

    paths = probe.Paths(user="u", home=tmp_path)
    out = probe.dispatch("storage", {}, run, paths)
    assert calls == [["python3", str(script), "--json"]]
    assert out["usage"]["free_fraction"] == 0.1
    assert out["script"] == str(script)


def test_storage_says_when_the_script_is_not_deployed(tmp_path):
    """Nicht ausgerollt und "alles in Ordnung" sind zwei Zustände (#165).

    Das Skript liegt unter ~/ubelix/, nicht im Checkout: der Checkout ist für
    Trainingsjobs an einen Commit geheftet, und eine tägliche Prüfung darf nicht
    davon abhängen, auf welchem Stand er steht."""
    def run(argv, timeout=60.0):  # pragma: no cover — darf nicht gerufen werden
        raise AssertionError("ohne ausgerolltes Skript darf nichts laufen")

    out = probe.dispatch("storage", {}, run, probe.Paths(user="u", home=tmp_path))
    assert out["deployed"] is False
    assert "research_storage.py" in out["error"]


def test_storage_does_not_pretend_a_broken_answer_is_a_measurement(tmp_path):
    """Kein JSON heisst kein Ergebnis — nicht null Vorschläge."""
    script = tmp_path / "ubelix" / "research_storage.py"
    script.parent.mkdir(parents=True)
    script.write_text("# Platzhalter\n")

    out = probe.dispatch("storage", {}, lambda argv, timeout=60.0: "Traceback …",
                         probe.Paths(user="u", home=tmp_path))
    assert "kein JSON" in out["error"]
    assert "suggestions" not in out


# ── das Inventar schreibt fortlaufend (10.10.2026) ──────────────────────────
def test_the_inventory_survives_being_cut_short(tmp_path, monkeypatch):
    """Der erste Versuch lief sechs Stunden in den Timeout und hinterliess nichts.

    Das Skript sammelte alle du-Messungen und schrieb erst am Ende. Ein
    Teilergebnis mit Datum ist brauchbar; sechs Stunden ohne Ergebnis sind es
    nicht. Jetzt wird nach jeder Messung geschrieben, und `ours_complete` sagt,
    ob die Posten, die die Vorschläge tragen, vollständig sind.
    """
    import importlib.util
    import sys

    root = tmp_path / "research"
    ours = root / "Textrecognition_Training"
    (ours / "hf_hub").mkdir(parents=True)
    (ours / "training_folder").mkdir()
    (root / "Projekt_Fremd").mkdir()
    (ours / "hf_hub" / "blob").write_bytes(b"x" * 1000)

    spec = importlib.util.spec_from_file_location(
        "rs", Path(__file__).resolve().parents[1] / "scripts" / "research_storage.py")
    rs = importlib.util.module_from_spec(spec)
    sys.modules["rs"] = rs
    spec.loader.exec_module(rs)
    monkeypatch.setattr(rs, "ROOT", root)

    out = tmp_path / "inv.json"
    # Erst nur unser Unterbaum: das ist der Teil, der in Minuten messbar ist.
    rs.inventory(root, write=out, with_top=False)
    partial = json.loads(out.read_text())
    assert partial["ours_complete"] is True
    assert partial["complete"] is False, "ohne die oberste Ebene ist es unvollständig"
    assert "Projekt_Fremd" not in partial["top"]
    assert "hf_hub" in partial["ours"]

    # Und mit: dann ist beides da.
    rs.inventory(root, write=out, with_top=True)
    full = json.loads(out.read_text())
    assert full["complete"] is True
    assert "Projekt_Fremd" in full["top"]


def test_a_partial_inventory_says_so_instead_of_suggesting_from_it(tmp_path, monkeypatch):
    """Abgebrochen, bevor unser Unterbaum fertig war, heisst: die Vorschläge sind
    unvollständig. Das gehört in die Antwort, nicht weggelassen (#165)."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "rs2", Path(__file__).resolve().parents[1] / "scripts" / "research_storage.py")
    rs = importlib.util.module_from_spec(spec)
    sys.modules["rs2"] = rs
    spec.loader.exec_module(rs)

    inv = tmp_path / "inv.json"
    inv.write_text(json.dumps({"measured": time.time(), "ours": {}, "top": {},
                               "ours_complete": False, "complete": False}))
    monkeypatch.setattr(rs, "INVENTORY", inv)
    _, note = rs.load_inventory()
    assert "abgebrochen" in note

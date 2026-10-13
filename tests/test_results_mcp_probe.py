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
    only_checkout = probe.last_progress(["Updating files: 100% (239/239), done."])
    assert only_checkout["state"] == "unattributable" and "done" not in only_checkout
    found = probe.last_progress(["Updating files: 100% (239/239), done.", "2350/2751"])
    assert found == {"state": "found", "done": 2350, "total": 2751, "line": "2350/2751",
                     "source": None}
    assert probe.last_progress(["nothing here"]) is None


def test_a_retry_counter_is_not_progress_and_does_not_project():
    """09.10.2026: `Retrying in 2s [Retry 2/5].` in the prepare log of 17795292 was
    read as 2/5 and projected to "fits, 30.1 % of the wall" — a number that looks
    measured and is not. The same for a model load that dies in OOM at
    `Loading weights: 644/677` (17611939). Found-but-not-attributable is a third
    state, and only `found` may feed the projection (#203)."""
    retry = probe.last_progress(["Streaming x", "Retrying in 2s [Retry 2/5]."])
    assert retry["state"] == "unattributable"
    assert "done" not in retry and "Retry" in retry["why"]
    wall = probe.judge_wall(8682, 72000, retry)
    assert wall["verdict"] == "unknown" and "projected_total_s" not in wall
    assert wall["why"] == ("the last counter is not progress: 2/5 is a known non-progress "
                           "counter (Retrying in)")
    shards = probe.last_progress(["Loading weights: 644/677 [00:40<00:02, 15.9it/s]"])
    assert shards["state"] == "unattributable" and "Loading weights" in shards["why"]
    # A real counter further up still wins over a later retry line.
    real = probe.last_progress(["streaming pages 37/1640", "Retrying in 1s [Retry 1/5]."])
    assert real["state"] == "found" and real["done"] == 37


VLM_TRAIN_LOG = (
    "{'loss': 0.7012, 'grad_norm': 1.2, 'learning_rate': 1.1e-04, 'epoch': 0.45}\r"
    " 45%|████▌     | 8622/19162 [5:12:33<6:22:10, 2.18s/it]\r"
    " 45%|████▌     | 8623/19162 [5:12:35<6:22:08, 2.18s/it]"
)


def test_queue_reads_a_vlm_run_from_its_train_log(cluster):
    """A VLM run writes its steps to <job>/logs/train.log and nothing to the Slurm
    stdout after the attempt line. On 09.10.2026 a run at 100 % GPU and step
    19162/19162 read as "no progress counter in the log" (#203). The attempt line
    names the record, the record names the stage, the stage names the file."""
    job_dir = write_job(cluster.scratch, "20260925T045500Z-ladder-med-qwen35-27b",
                        status="training", stage="train", cer=None, draw=None, finished=None)
    (job_dir / "logs").mkdir()
    (job_dir / "logs" / "train.log").write_text(VLM_TRAIN_LOG)
    (cluster.logs / "fedmin1-17488077.out").write_text(
        "== code: pinned to c19e1ca 2026-10-09 x\n"
        "Updating files: 100% (299/299), done.\n"
        "== attempt 2 — job 20260925T045500Z-ladder-med-qwen35-27b (training) on gnode34 "
        "2026-10-09T14:00:15+02:00\n"
        "Retrying in 2s [Retry 2/5].\n")
    run = canned({"squeue": SQUEUE, "sinfo": SINFO})
    row = {j["slurm_job_id"]: j for j in probe.queue(run, cluster)["jobs"]}["17488077"]
    assert row["job_id"] == "20260925T045500Z-ladder-med-qwen35-27b"
    assert row["stage_log"].endswith("logs/train.log") and row["stage_log_modified_cest"]
    assert row["progress"]["source"] == "train.log"
    assert row["progress"]["state"] == "found"
    assert (row["progress"]["done"], row["progress"]["total"]) == (8623, 19162)
    # The line says 6:22:08 are left (tqdm, this attempt's rate): 50:35 elapsed
    # plus that is far over a 3 h wall, whatever the ratio 8623/19162 says (#216).
    assert row["progress"]["remaining_s"] == 6 * 3600 + 22 * 60 + 8
    assert row["progress"]["stage"] == "train"
    assert row["wall"]["method"] == "tqdm"
    assert row["wall"]["projected_total_s"] == 3035 + 6 * 3600 + 22 * 60 + 8
    assert row["wall"]["verdict"] == "will hit the wall"
    # No completed line job with a measured test stage in this store: said, not zero.
    assert row["following"]["stages"] == ["test", "register"]
    assert row["following"]["seconds"] is None
    assert row["wall"]["uncounted_stages"] == ["test", "register"]
    assert "test, register not counted" in row["wall"]["why"]


# ── the wall counts the stages still ahead, and trusts tqdm over the ratio (#216)
def test_tqdm_remaining_is_read_and_beats_the_ratio_after_a_resume():
    """After a resume the HF trainer's bar starts at the checkpoint's global_step:
    `done` holds earlier attempts' steps, `elapsed` only this one, and
    `elapsed × total / done` comes out too small — optimistic, the direction
    #203 already failed in. tqdm's `<remaining` is computed from this attempt's
    rate and is the number to trust."""
    found = probe.last_progress(VLM_TRAIN_LOG.split("\r"))
    assert found["remaining_s"] == 6 * 3600 + 22 * 60 + 8 and found["rate"] == "2.18s/it"
    # A log being written ends mid-suffix (17838224 at 11:05 CEST, 11.10.2026):
    # the previous complete line of the same bar supplies the estimate.
    cut = probe.last_progress(VLM_TRAIN_LOG.split("\r") + [" 45%|████▌| 8624/19162 [5:12:37<6:2"])
    assert cut["done"] == 8624 and cut["remaining_s"] == 6 * 3600 + 22 * 60 + 8
    # But not from a different bar: a cut line after an unrelated counter has none.
    other = probe.last_progress(["Generating train split: 50%| 500/1000 [00:10<00:10, 50 it/s]",
                                 " 45%|████▌| 8624/19162 [5:12:37<6:2"])
    assert other["done"] == 8624 and "remaining_s" not in other
    resumed = {"state": "found", "done": 10321, "total": 13704, "stage": "train",
               "remaining_s": 90000}
    with_tqdm = probe.judge_wall(1800, 86400, resumed)
    assert with_tqdm["verdict"] == "will hit the wall" and with_tqdm["method"] == "tqdm"
    assert with_tqdm["why"].startswith("train: 1800 s + rest 90000 s (tqdm 10321/13704)")
    ratio_only = probe.judge_wall(1800, 86400, {k: v for k, v in resumed.items()
                                                if k != "remaining_s"})
    assert ratio_only["verdict"] == "fits" and ratio_only["method"] == "ratio"  # the lie


def test_the_test_stage_of_a_page_job_is_added_from_a_measured_sibling(cluster):
    """cap3072's test stage took 68 min, v1's 59; the TIMEOUT of 05.10.2026 fell in
    the test stage. A `train` counter at a third of the wall is not `fits` when
    the test stage measured on a sibling pushes the sum past 85 %."""
    stages = [
        {"name": "train", "status": "completed", "started_at": "2026-10-10T02:17:00Z",
         "finished_at": "2026-10-10T15:58:00Z", "code": {"commit": "0b58494a18b0" + "0" * 28}},
        {"name": "test", "status": "completed", "started_at": "2026-10-10T15:58:00Z",
         "finished_at": "2026-10-10T17:54:40Z", "code": {"commit": "0b58494a18b0" + "0" * 28}},
        {"name": "register", "status": "completed", "started_at": "2026-10-10T17:54:40Z",
         "finished_at": "2026-10-10T17:54:41Z", "code": {"commit": "0b58494a18b0" + "0" * 28}},
    ]
    write_job(cluster.scratch, "20261008T103636Z-olmocr2-7b-medieval-page-cap3072",
              base_model="allenai/olmOCR-2-7B-1025", granularity="page", cer=0.4674,
              finished="2026-10-10T17:06:00Z", extra={"stages": stages})
    # A line job of the same base model is not comparable: granularity rules.
    write_job(cluster.scratch, "20261001T085321Z-ladder-med-olmocr2-7b",
              base_model="allenai/olmOCR-2-7B-1025", granularity="line", cer=0.7465,
              finished="2026-10-11T00:00:00Z", extra={"stages": stages})
    job_dir = write_job(cluster.scratch, "20261009T174425Z-olmocr2-7b-medieval-page-px4m",
                        base_model="allenai/olmOCR-2-7B-1025", granularity="page",
                        status="training", stage="train", cer=None, draw=None, finished=None)
    (job_dir / "logs").mkdir()
    (job_dir / "logs" / "train.log").write_text("1300/1518\n")
    (cluster.logs / "fedmin1-17488077.out").write_text(
        "== attempt 1 — job 20261009T174425Z-olmocr2-7b-medieval-page-px4m (training) on gnode34\n")
    row = {j["slurm_job_id"]: j for j in
           probe.queue(canned({"squeue": SQUEUE, "sinfo": SINFO}), cluster)["jobs"]}["17488077"]
    assert row["following"] == {
        "from": "20261008T103636Z-olmocr2-7b-medieval-page-cap3072",
        "match": "base_model+granularity", "seconds": {"test": 7000, "register": 1},
        "stages": ["test", "register"]}
    # 3035 s + 3035 × 218 / 1300 = 3544 s on its own (a third of 10 800); with the test stage 10 544.
    wall = row["wall"]
    assert wall["method"] == "ratio" and wall["projected_total_s"] == 10544  # 3543.9 + 7001, truncated
    assert wall["verdict"] == "at risk"
    assert wall["why"] == ("train: 3035 s + rest 508 s (ratio 1300/1518) + test ≈ 7000 s + "
                           "register ≈ 1 s (from 20261008T103636Z-olmocr2-7b-medieval-page-cap3072)"
                           " = 10544 s of 10800")


def test_without_a_reference_the_missing_stage_is_said_and_a_close_call_is_not_fits():
    following = {"stages": ["test"], "seconds": None, "from": None}
    early = probe.judge_wall(3000, 10800, {"state": "found", "done": 150, "total": 164,
                                            "stage": "train"}, following)
    assert early["verdict"] == "fits" and early["uncounted_stages"] == ["test"]
    assert early["why"].endswith("test not counted: no completed job of this granularity to "
                                 "measure it")
    late = probe.judge_wall(7700, 10800, {"state": "found", "done": 150, "total": 164,
                                           "stage": "train"}, following)
    assert late["fraction_of_wall"] < 0.85  # would be "fits" if nothing were ahead
    assert late["verdict"] == "at risk"
    # A counter from the last stage of its group has nothing ahead: no caveat.
    assert probe.following_stages("test") == ["register"]
    assert probe.following_stages("register") == [] and probe.following_stages(None) == []


def test_queue_says_when_the_only_counter_is_a_retry(cluster):
    """No stage log to fall back on, and the stdout's last counter is a retry:
    the verdict is unknown and `why` says what was found, so the "unknown"
    invites a look instead of a made-up "fits"."""
    (cluster.logs / "fedmin1-17488077.out").write_text(
        "== attempt 1 — job 20261009T101216Z-nope (preparing) on cnode01\n"
        "Retrying in 2s [Retry 2/5].\n")
    row = {j["slurm_job_id"]: j for j in probe.queue(canned({"squeue": SQUEUE, "sinfo": SINFO}), cluster)["jobs"]}
    fedmin = row["17488077"]
    assert fedmin["progress"]["state"] == "unattributable"
    assert fedmin["progress"]["source"] == "slurm"
    assert fedmin["wall"]["verdict"] == "unknown"
    assert "not progress" in fedmin["wall"]["why"] and "Retry" in fedmin["wall"]["why"]
    assert "stage_log" not in fedmin  # no record of that id: nothing invented


def test_the_stdout_counter_wins_when_it_is_real(cluster):
    """A cropping job writes its own counter to stdout; the stage log is not
    consulted for a counter that is already attributable."""
    job_dir = write_job(cluster.scratch, "20260925T045500Z-ladder-med-qwen35-27b",
                        status="training", stage="train", cer=None, draw=None, finished=None)
    (job_dir / "logs").mkdir()
    (job_dir / "logs" / "train.log").write_text(VLM_TRAIN_LOG)
    (cluster.logs / "fedmin1-17488077.out").write_text(
        "== attempt 1 — job 20260925T045500Z-ladder-med-qwen35-27b (training) on gnode34\n"
        "2350/2751\n")
    row = {j["slurm_job_id"]: j for j in probe.queue(canned({"squeue": SQUEUE, "sinfo": SINFO}), cluster)["jobs"]}
    assert row["17488077"]["progress"]["source"] == "slurm"
    assert row["17488077"]["progress"]["done"] == 2350
    assert "stage_log" not in row["17488077"]


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


# ── the wall's consequence and the fallback resubmission (#226) ─────────────
TRAIN_SBATCH = """\
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu-invest
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
# the trap below runs `scontrol requeue` — this comment must not count
trap '
  echo "== SIGTERM at $(date -Is) — requeueing $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || echo "   (requeue refused)"
' TERM
"""

RESCORE_SBATCH = """\
#!/bin/bash
#SBATCH --job-name=rescore-shared-draw
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
"""


def write_checkout(cluster: Paths) -> None:
    ubelix = cluster.checkout / "ubelix"
    ubelix.mkdir(parents=True)
    (ubelix / "train.sbatch").write_text(TRAIN_SBATCH)
    (ubelix / "rescore_on_shared_draw.sbatch").write_text(RESCORE_SBATCH)


def test_on_timeout_is_read_from_the_batch_file_not_assumed(cluster):
    """train.sbatch asks for SIGTERM before the wall and its trap requeues; the
    rescore file has neither, and for it TIMEOUT is the end. --requeue alone
    would cover a preemption and not a wall."""
    write_checkout(cluster)
    train = probe.on_timeout(cluster, "train")
    assert train["file"] == "ubelix/train.sbatch"
    assert train["signal"] == "B:TERM@120" and train["signal_before_wall_s"] == 120
    assert train["requeue"] is True and train["trap_requeues"] is True
    assert train["meaning"].startswith("TERM 120 s before the wall; the trap requeues")
    rescore = probe.on_timeout(cluster, "rescore-shared-draw")
    assert rescore["file"] == "ubelix/rescore_on_shared_draw.sbatch"
    assert rescore["requeue"] is False and rescore["trap_requeues"] is False
    assert rescore["meaning"] == "none: TIMEOUT ends the job"
    assert probe.on_timeout(cluster, "fedmin1")["file"] is None
    (cluster.checkout / "ubelix" / "preempt_only.sbatch").write_text(
        "#SBATCH --job-name=preempt-only\n#SBATCH --requeue\n")
    assert probe.on_timeout(cluster, "preempt-only")["meaning"].startswith(
        "requeue on preemption only")


def test_queue_names_the_attempt_and_what_the_wall_does(cluster):
    """Measured 12.10.2026 13:16 CEST on 17890142: SIGTERM 120 s before the wall,
    the trap requeued, attempt 2 started on the same Slurm id four minutes later.
    The report had to be able to read "attempt 2" instead of guessing it."""
    write_checkout(cluster)
    (cluster.logs / "fedmin1-17488077.out").write_text(
        "== attempt 1 — job x (training) on gnode34\n"
        "[2026-10-12T13:16:10.004] error: *** JOB 17488077 ON gnode26 CANCELLED AT "
        "2026-10-12T13:16:10 DUE to SIGNAL Terminated ***\n"
        "== SIGTERM at 2026-10-12T13:16:10+02:00 — requeueing 17488077\n"
        "== attempt 2 — job x (training) on gnode34\n25/2751\n2350/2751\n")
    squeue = SQUEUE + ("17961545|rescore-shared-draw|gpu|job_gratis|PENDING|0:00|4:00:00|N/A|"
                       "QOSGrpGRES||4|cpu=4,gres/gpu:rtx4090=1|\n")
    run = canned({"squeue": squeue, "sinfo": SINFO})
    by_id = {j["slurm_job_id"]: j for j in probe.queue(run, cluster)["jobs"]}
    fedmin = by_id["17488077"]
    assert fedmin["attempt"] == 2
    assert fedmin["wall"]["on_timeout"]["file"] is None  # no fedmin1 file in this checkout
    # A running job without a log still says what its wall does.
    assert "on_timeout" in by_id["17545199"]["wall"]
    assert "ALL its users" in by_id["17961545"]["reason_meaning"]


def resubmit_cluster(cluster: Paths, *, status: str = "training", with_log: bool = True,
                     state: str = "TIMEOUT", checkpoint: bool = True) -> dict:
    write_checkout(cluster)
    job_id = "20261009T174425Z-olmocr2-7b-medieval-page-px4m"
    ckpt = cluster.scratch / "runs" / "checkpoints" / job_id
    if checkpoint:
        (ckpt / "checkpoint-1200").mkdir(parents=True)
        (ckpt / "checkpoint-975").mkdir(parents=True)
    write_job(cluster.scratch, job_id, status=status, stage="train", cer=None, draw=None,
              finished=None, created="4de946820d03",
              stage_commit="fa02f2d4db00" if with_log else None,
              extra={"checkpoint_dir": str(ckpt), "registration": None,
                     "created_at": "2026-10-09T17:44:25Z"})
    if with_log:
        (cluster.logs / "train-17890142.out").write_text(
            "== code: pinned to fa02f2d 2026-10-10 eval_material (#55)\n"
            "== SIF: /storage/homefs/th19c587/ubelix/vlm-train-tf5.sif (transformers 5.17.0)\n"
            "== attempt 1 — job %s (training) on gnode26 2026-10-11T13:18:29+02:00\n"
            "stage train: running fa02f2d4db00, but the job was created with 4de946820d03\n"
            "[2026-10-12T13:18:00.000] error: *** JOB 17890142 ON gnode26 CANCELLED AT "
            "2026-10-12T13:18:00 DUE TO TIME LIMIT ***\n" % job_id)
        # An older log of another record must not be attributed to this one.
        (cluster.logs / "train-17800000.out").write_text(
            "== attempt 1 — job 20260925T045500Z-other (training) on gnode25\n")
    sacct = "17890142|train|%s|1-00:00:12|2026-10-12T13:18:12\n" % state
    squeue = "17545210|line-dataset|PENDING|\n"
    git = "0\t15\n"
    return {"job_id": job_id, "sacct": sacct, "squeue": squeue, "git": git}


def test_resubmit_writes_the_pinned_command_after_the_wall(cluster):
    """After a TIMEOUT with nothing of the record left in Slurm, the command is
    written out: pinned to the commit the train stage ran (not HEAD, which is
    where every drift line so far came from), with the container the attempt
    printed, resuming from the newest checkpoint."""
    fx = resubmit_cluster(cluster)
    run = canned({"sacct": fx["sacct"], "squeue": fx["squeue"], "git": fx["git"]})
    answer = probe.job(cluster, fx["job_id"], run=run)["resubmit"]
    assert answer["command"] == (
        "cd %s && ATR_CODE_COMMIT=fa02f2d4db00%s JOB_ID=%s ubelix/submit.sh ubelix/train.sbatch "
        "-- --export=ALL,SIF=/storage/homefs/th19c587/ubelix/vlm-train-tf5.sif"
        % (cluster.checkout, "0" * 28, fx["job_id"]))
    assert answer["kind"] == "resume" and answer["resume_from"] == "checkpoint-1200"
    assert answer["pin_source"] == "stage train of this record"
    assert answer["sif_source"] == "log of Slurm job 17890142"
    assert answer["attempts"] == [{"slurm_job_id": "17890142", "attempt": 1, "state": "TIMEOUT",
                                   "end": "2026-10-12T13:18:12",
                                   "cause": answer["attempts"][0]["cause"]}]
    assert "DUE TO TIME LIMIT" in answer["attempts"][0]["cause"]
    assert answer["checkout_behind"] == 15
    assert "15 behind origin/main" in answer["why"]
    assert "pinned to fa02f2d4db00" in answer["why"]
    # One sacct over the attributed ids, one squeue for unattributed pending jobs.
    assert [c[0] for c in run.calls] == ["sacct", "squeue", "git"]
    assert run.calls[0][:3] == ["sacct", "-j", "17890142"]


def test_resubmit_is_null_while_an_attempt_lives_or_after_a_failure(cluster):
    fx = resubmit_cluster(cluster, state="RUNNING")
    run = canned({"sacct": fx["sacct"], "squeue": fx["squeue"], "git": fx["git"]})
    answer = probe.job(cluster, fx["job_id"], run=run)["resubmit"]
    assert answer["command"] is None and "17890142 is RUNNING" in answer["why"]
    assert "command" in answer and "kind" not in answer
    pending = canned({"sacct": fx["sacct"].replace("RUNNING", "PENDING")})
    assert probe.job(cluster, fx["job_id"], run=pending)["resubmit"]["command"] is None
    failed = canned({"sacct": fx["sacct"].replace("RUNNING", "FAILED")})
    answer = probe.job(cluster, fx["job_id"], run=failed)["resubmit"]
    assert answer["command"] is None
    assert answer["why"].startswith("the last attempt 17890142 ended FAILED (")
    assert "needs a person" in answer["why"]
    # A record in another status, or on another host, has nothing to resubmit.
    done = canned({})
    write_job(cluster.scratch, "20260930T170315Z-ladder-xix-gemma4-12b")
    assert probe.job(cluster, "20260930T170315Z-ladder-xix-gemma4-12b", run=done)["resubmit"] == {
        "command": None, "why": "status completed: nothing to resubmit"}
    write_job(cluster.share, "20261007T131819Z-kraken-german-xix-v1", store="share",
              host="asteraix", status="training", cer=None, draw=None, finished=None)
    assert probe.job(cluster, "20261007T131819Z-kraken-german-xix-v1", run=done)["resubmit"][
        "why"].startswith("host asteraix")
    assert done.calls == []


def test_resubmit_of_a_never_run_record_is_called_a_first_submission(cluster):
    """A corpus built by prepare.sbatch and never trained has no attempt, no
    checkpoint and no container line: the command still stands, pinned to the
    commit the job was created with, and `kind` says it is a decision."""
    fx = resubmit_cluster(cluster, with_log=False, checkpoint=False)
    run = canned({"squeue": "17807894|train|PENDING|\n", "git": "0\t0\n"})
    answer = probe.job(cluster, fx["job_id"], run=run)["resubmit"]
    assert answer["kind"] == "first train submission" and answer["resume_from"] is None
    assert answer["attempts"] == []
    assert answer["command"] == (
        "cd %s && ATR_CODE_COMMIT=4de946820d03%s JOB_ID=%s ubelix/submit.sh ubelix/train.sbatch"
        % (cluster.checkout, "0" * 28, fx["job_id"]))
    assert answer["pin_source"] == "the commit the job was created with"
    assert answer["sif"] is None and "default applies" in answer["sif_source"]
    assert answer["unattributed_pending"] == ["17807894 (train)"]
    assert "not attributable: 17807894 (train)" in answer["why"]
    assert answer["checkout_behind"] == 0 and "behind" not in answer["why"]
    assert "sacct" not in [c[0] for c in run.calls]  # no attempts, nothing to ask


def test_a_checkpoint_without_a_slurm_log_is_a_resume_not_a_first_submission(cluster):
    """Seen 12.10.2026 on throughput-qwen35-4b over the HTTPS route: checkpoint-400
    on disk, no Slurm log naming the record — the answer said `kind: resume` and
    `why: never ran on Slurm` in one breath. The checkpoint is the fact."""
    fx = resubmit_cluster(cluster, with_log=False, checkpoint=True)
    run = canned({"squeue": "", "git": "0\t0\n"})
    answer = probe.job(cluster, fx["job_id"], run=run)["resubmit"]
    assert answer["kind"] == "resume" and answer["resume_from"] == "checkpoint-1200"
    assert answer["attempts"] == []
    assert "never ran" not in answer["why"]
    assert answer["why"].startswith("no Slurm log since 2026-10-09 19:44 CEST names this record, "
                                    "but checkpoint-1200 exists under ")
    assert "resumes from it" in answer["why"]


def test_a_dependency_that_can_never_be_met_is_explained(cluster):
    squeue = ("18005151|prep|epyc2|job_gratis|PENDING|0:00|1:00:00|N/A|DependencyNeverSatisfied||8|"
              "cpu=8,mem=16G|\n")
    row = probe.queue(canned({"squeue": squeue}), cluster)["jobs"][0]
    assert "will never start" in row["reason_meaning"] and "scancel" in row["reason_meaning"]


def test_a_restart_without_a_checkpoint_is_said(cluster):
    fx = resubmit_cluster(cluster, checkpoint=False, state="CANCELLED by 33813")
    run = canned({"sacct": fx["sacct"], "squeue": "", "git": fx["git"]})
    answer = probe.job(cluster, fx["job_id"], run=run)["resubmit"]
    assert answer["kind"] == "restart without checkpoint"
    assert "starts the stage over" in answer["why"]
    assert answer["attempts"][0]["state"] == "CANCELLED by 33813"


def test_job_without_a_runner_has_no_resubmit_field(cluster):
    fx = resubmit_cluster(cluster)
    assert "resubmit" not in probe.job(cluster, fx["job_id"])


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
    assert row["progress"] == {"state": "found", "done": 1200, "total": 4000,
                               "line": "1200/4000", "source": "train.log"}
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


def test_a_scoped_sweep_is_not_an_aborted_one(tmp_path, monkeypatch):
    """Ein leeres ``top`` hat zwei Ursachen, die verschieden zu lesen sind.

    Mit ``--ours-only`` war die oberste Ebene nie verlangt; nach einem Timeout
    fehlt sie. Beide Male steht dort nichts, und wer das verwechselt, hält ein
    Teilergebnis für eine vollständige Erhebung — genau der Fehler, der am
    09.10.2026 beinahe 992 GB echter Daten gekostet hätte. Darum steht der
    Umfang im Datensatz (#204).
    """
    import importlib.util
    import sys

    root = tmp_path / "research"
    ours = root / "Textrecognition_Training"
    (ours / "hf_hub").mkdir(parents=True)
    (root / "Projekt_Fremd").mkdir()

    spec = importlib.util.spec_from_file_location(
        "rs3", Path(__file__).resolve().parents[1] / "scripts" / "research_storage.py")
    rs = importlib.util.module_from_spec(spec)
    sys.modules["rs3"] = rs
    spec.loader.exec_module(rs)
    monkeypatch.setattr(rs, "ROOT", root)

    inv = tmp_path / "inv.json"
    monkeypatch.setattr(rs, "INVENTORY", inv)

    rs.inventory(root, write=inv, with_top=False)
    scoped = json.loads(inv.read_text())
    assert scoped["scope"] == "ours"
    assert scoped["complete"] is False
    assert scoped["top"] == {}
    _, note = rs.load_inventory()
    assert "verlangt" in note, "ein gewollter Teilumfang ist kein Abbruch"
    assert "abgebrochen" not in note

    rs.inventory(root, write=inv, with_top=True)
    full = json.loads(inv.read_text())
    assert full["scope"] == "full"
    assert full["complete"] is True
    _, note = rs.load_inventory()
    assert "verlangt" not in note and "abgebrochen" not in note

    # Ein voller Durchgang, den der Timeout erwischt hat: scope sagt es.
    inv.write_text(json.dumps({"measured": time.time(), "ours": {"hf_hub": 1},
                               "top": {}, "scope": "full",
                               "ours_complete": True, "complete": False}))
    _, note = rs.load_inventory()
    assert "abgebrochen" in note

    # Und ein Datensatz von vor der scope-Angabe behauptet nichts, was er nicht weiss.
    inv.write_text(json.dumps({"measured": time.time(), "ours": {"hf_hub": 1},
                               "top": {}, "ours_complete": True, "complete": False}))
    _, note = rs.load_inventory()
    assert "nicht vermerkt" in note


def test_a_running_sweep_is_not_an_aborted_one(tmp_path, monkeypatch):
    """„Abgebrochen" über einen Lauf, der gerade läuft, ist eine falsche Aussage.

    Am 10.10.2026 meldete der Teildatensatz eines laufenden Inventarjobs
    „abgebrochen, bevor unser Unterbaum fertig war" — während der Job auf
    bnode004 rechnete. Das Skript konnte laufend nicht von abgebrochen
    unterscheiden, weil nichts im Datensatz davon wusste.

    Jetzt steht `running` darin und `touched` sagt, wann zuletzt gemessen wurde.
    Ob ein stiller Lauf beendet, abgebrochen oder nur langsam ist, weiss der
    Datensatz **nicht** — darum nennt der Hinweis das Alter und keine Ursache
    (#204).
    """
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "rs4", Path(__file__).resolve().parents[1] / "scripts" / "research_storage.py")
    rs = importlib.util.module_from_spec(spec)
    sys.modules["rs4"] = rs
    spec.loader.exec_module(rs)

    inv = tmp_path / "inv.json"
    monkeypatch.setattr(rs, "INVENTORY", inv)
    now = time.time()

    def write(**kw):
        base = {"measured": now, "ours": {"hf_hub": 1}, "top": {},
                "scope": "ours", "ours_complete": False, "complete": False}
        inv.write_text(json.dumps({**base, **kw}))

    # Läuft gerade: eben noch gemessen.
    write(running=True, touched=now - 20)
    _, note = rs.load_inventory()
    assert "noch nicht fertig" in note
    assert "abgebrochen" not in note, "ein laufender Lauf ist kein abgebrochener"

    # Still, aber als laufend vermerkt: das Alter wird genannt, keine Ursache.
    write(running=True, touched=now - 4 * 3600)
    _, note = rs.load_inventory()
    assert "keine neue Messung" in note
    assert "4.0 h" in note
    assert "beendet, hängt oder wurde abgebrochen" in note, \
        "drei mögliche Ursachen, nicht eine behauptete"

    # Ordentlich beendet, Umfang ours: der alte, richtige Hinweis.
    write(running=False, ours_complete=True)
    _, note = rs.load_inventory()
    assert "verlangt" in note and "fertig" not in note

    # Ein voller Durchgang zählt als fertig, auch wenn die Marke stehen blieb.
    write(running=True, touched=now - 9999, ours_complete=True, complete=True,
          scope="full")
    _, note = rs.load_inventory()
    assert "Messung" not in note and "abgebrochen" not in note


def test_the_marker_is_set_while_measuring_and_cleared_at_the_end(tmp_path, monkeypatch):
    """Die Marke muss vom ersten Schreiben an stehen — sonst liest ein
    Teilergebnis sich wie ein beendeter Lauf — und am Ende fallen, in beiden
    Umfängen (#204)."""
    import importlib.util
    import sys

    root = tmp_path / "research"
    (root / "Textrecognition_Training" / "hf_hub").mkdir(parents=True)
    (root / "Projekt_Fremd").mkdir()

    spec = importlib.util.spec_from_file_location(
        "rs5", Path(__file__).resolve().parents[1] / "scripts" / "research_storage.py")
    rs = importlib.util.module_from_spec(spec)
    sys.modules["rs5"] = rs
    spec.loader.exec_module(rs)
    monkeypatch.setattr(rs, "ROOT", root)

    inv = tmp_path / "inv.json"
    seen: list[bool] = []
    real_du = rs.du
    monkeypatch.setattr(rs, "du", lambda p: (
        seen.append(json.loads(inv.read_text())["running"]) if inv.exists() else None,
        real_du(p))[1])

    for with_top in (False, True):
        seen.clear()
        record = rs.inventory(root, write=inv, with_top=with_top)
        assert record["running"] is False, "am Ende steht die Marke nicht mehr"
        assert json.loads(inv.read_text())["running"] is False
        assert seen and all(seen), "während der Messung muss sie stehen"
        assert record["touched"] >= record["measured"]

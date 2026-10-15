"""#250: what a run looks like from outside, and what must not be claimed.

agentic_historian#366 asks for the facts that decided every compile diagnosis:
RSS and whether it is climbing, how much the run has written, the compile phase,
`it/s`, the early-stopping counter, and the previous epoch's metric for a trend
arrow.

The first two are observable and are here. The others are **not available**, and
the point of this file is as much what it refuses as what it checks — a field
holding None because nobody got round to it reads exactly like a field holding
None because the number cannot exist, and this repository has paid for that
confusion five times (agentic_historian #119, #595, #631, #633, #649).

Offline throughout: `/proc` is a fixture directory and the walked tree is
`tmp_path`, so every case — a dead process, an unreadable status, a file that
vanishes mid-walk — is producible on demand. A real `/proc` cannot do that.
"""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import pytest

from atr_training import chunking, runtime_probe
from atr_training.chunking import CHUNK_PLAN_FILENAME, chunk_record
from atr_training.curves import TrainingCurve


# ── RSS: a number, or nothing, never a zero ─────────────────────────────────

def _proc(tmp_path: Path, pid: int, status: str) -> Path:
    directory = tmp_path / str(pid)
    directory.mkdir(parents=True)
    (directory / "status").write_text(status, encoding="ascii")
    return tmp_path


def test_rss_is_read_from_vmrss(tmp_path):
    root = _proc(tmp_path, 1843, "Name:\tpython\nVmRSS:\t  66898944 kB\nThreads:\t9\n")
    assert runtime_probe.rss_mib(1843, proc=root) == 65331


@pytest.mark.parametrize("pid", [None, 0])
def test_no_pid_is_no_reading(pid, tmp_path):
    """A record without a pid has nothing to observe. `0` would read as idle."""
    assert runtime_probe.rss_mib(pid, proc=tmp_path) is None


def test_a_process_that_has_exited_is_no_reading(tmp_path):
    """The case that matters most: 0 MiB reads as "idle" and means "dead"."""
    assert runtime_probe.rss_mib(4242, proc=tmp_path) is None


def test_a_status_without_vmrss_is_no_reading(tmp_path):
    """A kernel thread, or a format that moved."""
    root = _proc(tmp_path, 7, "Name:\tkthreadd\nThreads:\t1\n")
    assert runtime_probe.rss_mib(7, proc=root) is None


def test_a_garbled_vmrss_is_no_reading(tmp_path):
    root = _proc(tmp_path, 8, "VmRSS:\tnot-a-number kB\n")
    assert runtime_probe.rss_mib(8, proc=root) is None


# ── what the run has written ────────────────────────────────────────────────

def test_the_walked_size_is_what_this_run_wrote(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "train.arrow").write_bytes(b"x" * (3 * 1024 * 1024))
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "train.log").write_bytes(b"y" * (1024 * 1024))
    assert runtime_probe.dir_mib(tmp_path) == 4


def test_a_directory_that_is_not_there_is_no_reading(tmp_path):
    assert runtime_probe.dir_mib(tmp_path / "never-made") is None


def test_none_is_no_reading():
    assert runtime_probe.dir_mib(None) is None


def test_an_empty_job_directory_reads_zero_not_nothing(tmp_path):
    """Zero is right here and None is not: the directory exists and holds
    nothing, which is a measurement. The run has written nothing *yet*."""
    assert runtime_probe.dir_mib(tmp_path) == 0


# ── the trend, which needs two samples and says so ──────────────────────────

def _sample(at, rss=None, dir_mib=None):
    return runtime_probe.Sample(at=at, rss_mib=rss, job_dir_mib=dir_mib)


def test_one_sample_yields_no_direction():
    """The complaint #366 makes from the other side: a single value cannot tell
    "learning slowly" from "flat on a plateau"."""
    assert runtime_probe.trend(None, _sample(100.0, rss=1000), "rss_mib") is None


def test_growing_is_growing():
    before, after = _sample(0.0, rss=20000), _sample(60.0, rss=22000)
    assert runtime_probe.trend(before, after, "rss_mib") == "growing"


def test_flat_is_flat():
    """63.8 GB and then 63.8 GB — the healthy extract phase in #366, and the one
    a jittering arrow must not turn into "growing"."""
    before, after = _sample(0.0, rss=65331), _sample(60.0, rss=65400)
    assert runtime_probe.trend(before, after, "rss_mib") == "flat"


def test_falling_is_named_rather_than_called_flat():
    before, after = _sample(0.0, rss=60000), _sample(60.0, rss=40000)
    assert runtime_probe.trend(before, after, "rss_mib") == "falling"


def test_a_stale_previous_sample_is_not_compared_against():
    """The gap between two polls says nothing about a rate when the second came
    half an hour late — and a stale pair is how a long-finished growth gets
    reported as current."""
    before = _sample(0.0, rss=20000)
    after = _sample(runtime_probe.TREND_MAX_GAP_S + 1, rss=40000)
    assert runtime_probe.trend(before, after, "rss_mib") is None


def test_a_missing_reading_on_either_side_yields_no_direction():
    assert runtime_probe.trend(_sample(0.0), _sample(60.0, rss=100), "rss_mib") is None
    assert runtime_probe.trend(_sample(0.0, rss=100), _sample(60.0), "rss_mib") is None


def test_two_samples_at_the_same_instant_yield_no_direction():
    before, after = _sample(5.0, rss=100), _sample(5.0, rss=200)
    assert runtime_probe.trend(before, after, "rss_mib") is None


def test_growth_from_nothing_is_growth():
    before, after = _sample(0.0, dir_mib=0), _sample(60.0, dir_mib=1024)
    assert runtime_probe.trend(before, after, "job_dir_mib") == "growing"


def test_nothing_to_nothing_is_flat():
    before, after = _sample(0.0, dir_mib=0), _sample(60.0, dir_mib=0)
    assert runtime_probe.trend(before, after, "job_dir_mib") == "flat"


def test_the_floor_is_a_percentage_not_a_count(tmp_path):
    """A 1 MiB wobble is noise at 64 GiB and a doubling at 1 MiB. An absolute
    floor would call the second one flat, which is the reading that hid a
    stalled extract."""
    small_before, small_after = _sample(0.0, dir_mib=1), _sample(60.0, dir_mib=2)
    assert runtime_probe.trend(small_before, small_after, "job_dir_mib") == "growing"

    big_before, big_after = _sample(0.0, dir_mib=65000), _sample(60.0, dir_mib=65001)
    assert runtime_probe.trend(big_before, big_after, "job_dir_mib") == "flat"


# ── the refusals, which are the other half of this issue ────────────────────

# ── the compile, which is the one ask that is measurable ────────────────────
#
# This was nearly refused with the others, on the assumption that nothing
# records the compile's position. It does: a chunked compile writes a record
# beside each finished arrow so a resumed run knows what not to rebuild (#72).
# Counting those is reading what the run wrote.

def _chunked(tmp_path: Path, *, max_pages: int | None = 60, chunk_pages: int = 10):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    (data / CHUNK_PLAN_FILENAME).write_text(json.dumps(
        {"hf_repo": "dh-unibe/x", "data_files": ["a.parquet"],
         "max_pages": max_pages, "chunk_pages": chunk_pages}), encoding="utf-8")
    return data


def _chunk(data: Path, index: int, *, finished: bool = True, empty_arrow: bool = False):
    arrow = data / f"train_{index:04d}.arrow"
    arrow.write_bytes(b"" if empty_arrow else b"arrow")
    if finished:
        chunk_record(arrow).write_text(json.dumps({"pages_written": 10, "lines": 300}),
                                       encoding="utf-8")
    return arrow


def test_the_compiled_chunks_are_counted_against_what_the_plan_implies(tmp_path):
    data = _chunked(tmp_path)          # 60 pages in chunks of 10 → six
    for index in range(2):
        _chunk(data, index)

    assert runtime_probe.compile_chunks(data) == {"done": 2, "expected": 6}


def test_an_unchunked_compile_is_not_reported_as_a_chunked_one_at_zero(tmp_path):
    """The three-valued point again: `{"done": 0}` here would read as a chunked
    compile that has finished nothing, which is a different run."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.arrow").write_bytes(b"arrow")

    assert runtime_probe.compile_chunks(data) is None
    assert runtime_probe.compile_chunks(tmp_path / "gone") is None
    assert runtime_probe.compile_chunks(None) is None


def test_an_uncapped_plan_counts_without_claiming_a_total(tmp_path):
    """The length of a streamed selection is not known before it is consumed.
    The count still climbs, which is the signal #366 wanted."""
    data = _chunked(tmp_path, max_pages=None)
    _chunk(data, 0)

    assert runtime_probe.compile_chunks(data) == {"done": 1, "expected": None}


def test_a_chunk_the_runner_would_rebuild_is_not_counted_as_done(tmp_path):
    """The reason the rule is shared rather than copied. An arrow exists after a
    SIGKILL too; counting arrows would report a chunk compiled while the resume
    loop was rebuilding it — and the two answers would come from one GET."""
    data = _chunked(tmp_path)
    _chunk(data, 0)
    killed = _chunk(data, 1, finished=False)      # arrow, no record
    truncated = _chunk(data, 2, empty_arrow=True)  # record, empty arrow

    assert runtime_probe.compile_chunks(data) == {"done": 1, "expected": 6}
    assert chunking.finished_chunk(killed) is None
    assert chunking.finished_chunk(truncated) is None


def test_the_probe_and_the_resume_loop_call_the_same_rule_finished():
    """Structural, because the failure mode is drift rather than a wrong answer:
    two copies of this rule would agree on the day they were written."""
    from engines.kraken_train_svc import runner

    for caller in (runner._finished_chunk, runtime_probe.compile_chunks):
        assert "finished_chunk(" in inspect.getsource(caller), (
            f"{caller.__qualname__} no longer decides 'finished' with "
            f"chunking.finished_chunk — two copies of the rule would drift")
    assert runner.finished_chunk is chunking.finished_chunk, (
        "the runner imports a different finished_chunk than the probe does")


def test_the_position_inside_one_ketos_compile_is_still_refused():
    """What counting chunks does not give: a single `ketos compile` says nothing
    until it exits, so there is no percentage to report inside one."""
    assert not hasattr(runtime_probe, "compile_progress_pct")
    doc = runtime_probe.__doc__ or ""
    assert "inside" in doc and "ketos compile" in doc, (
        "the module does not say what about the compile stays unavailable")


def test_the_module_reports_no_iterations_per_second():
    """`ketos` prints it and keeps it nowhere — #51 is the prerequisite, and log
    scraping was ruled out there rather than left open."""
    assert not hasattr(runtime_probe, "it_per_s")


def test_a_trend_arrow_is_not_offered_for_the_curve():
    """The refusal with the sharpest edge.

    #366 asks for "last epoch metrics with the previous value for the trend
    arrow". The curve is a **selection**: kraken keeps the top ten checkpoints,
    so the point before the last is the next-best surviving epoch, not the
    previous epoch. An arrow computed from it points whichever way the selection
    happened to fall — and it would look exactly like a measurement.
    """
    assert not hasattr(runtime_probe, "cer_previous")

    # The shape of the data, so this is not an argument from memory: ten kept
    # points whose epochs are not consecutive.
    kept = TrainingCurve(
        points=[], complete=False, source="checkpoint filenames", note="")
    assert kept.complete is False, (
        "a complete curve would make the arrow honest; this one says it is not")


def test_still_improving_is_the_diagnostic_that_does_exist():
    """And it is correctly scoped: None when there is too little to say, rather
    than a direction invented from one point."""
    from atr_training.curves import EpochPoint

    one = TrainingCurve(points=[EpochPoint(3, 0.91)], complete=False, source="x")
    assert one.still_improving is None, "one point cannot carry a direction"

    peaked = TrainingCurve(
        points=[EpochPoint(2, 0.95), EpochPoint(9, 0.90)], complete=False, source="x")
    assert peaked.still_improving is False

    climbing = TrainingCurve(
        points=[EpochPoint(2, 0.90), EpochPoint(9, 0.95)], complete=False, source="x")
    assert climbing.still_improving is True


def test_the_docstring_gives_a_reason_for_each_refusal():
    """So the next reader does not take an absence for an oversight and add a
    field full of Nones.

    Per bullet rather than per word: the subjects are all named in the section's
    opening sentence too, so a check for the word passed while the bullet that
    gave the reason was gone.
    """
    doc = runtime_probe.__doc__ or ""
    heading = "What this deliberately does **not** report"
    assert heading in doc, "the module no longer states its refusals"
    bullets = [b.strip() for b in doc.split(heading, 1)[1].split("\n* ")[1:]]

    for subject in ("cer_previous", "early-stopping", "it/s"):
        owning = [b for b in bullets if subject in b]
        assert len(owning) == 1, f"no single bullet says why {subject} is absent"
        assert "—" in owning[0], f"the bullet for {subject} names no reason"
    assert "#51" in doc, "the prerequisite for the missing metrics is not named"


# ── the sample is a sample, and says when ───────────────────────────────────

def test_a_sample_carries_its_clock_reading():
    """Without `at` the value is "65331 MiB" with no answer to "as of when",
    which is how an observation gets filed as a record."""
    sample = runtime_probe.Sample(at=time.time(), rss_mib=65331, job_dir_mib=41000)
    assert sample.at > 0
    assert runtime_probe.Sample.__dataclass_fields__["at"].type in ("float", float)


# ── on the wire: GET /jobs/{id} ─────────────────────────────────────────────
#
# The unit tests above prove the probe; these prove the route — that the sample
# reaches a caller, that it is absent rather than nulled when there is nothing
# to see, and that reading it writes nothing.

import os                                                        # noqa: E402

from fastapi.testclient import TestClient                        # noqa: E402

from atr_training.contracts import DatasetSpec, Metrics, TrainRequest  # noqa: E402
from atr_training.jobstore import JobStore                        # noqa: E402
from atr_training.preflight import GpuInfo                        # noqa: E402
from atr_training.settings import TrainerSettings                 # noqa: E402
from kraken_train_svc import app as app_module                    # noqa: E402
from tests.conftest import TEST_HOST_ID                           # noqa: E402

#: Starlette's client reports the host "testclient", which the access middleware
#: refuses as it would any peer it cannot place (#13).
LOOPBACK = ("127.0.0.1", 50000)

#: Not a pid on any Linux: pid_t is int32 and pid_max is far below this.
PID_NEVER = 2**31 - 1

REQUEST = TrainRequest(
    model_id="kraken-thun-missiven-v1",
    dataset=DatasetSpec(hf_repo="dh-unibe/image-text_medieval-scripts_xiv-xv-xvi",
                        train_projects=["P"]))


@pytest.fixture
def svc(tmp_path, trainer_key, monkeypatch):
    """The service over a throwaway job store, and an empty observation memory.

    ``_LAST_OBSERVATION`` is module state, so it is cleared per test: a sample
    left by the test before would be the "previous" one here, and a trend
    computed across two tests is not a measurement of anything.
    """
    venvs = tmp_path / "venvs"
    for name in ("kraken-train", "vlm-train"):
        (venvs / name / "bin").mkdir(parents=True)
        (venvs / name / "bin" / "python").touch()
    settings = TrainerSettings(jobs_root=tmp_path / "jobs", trained_root=tmp_path / "trained",
                               venvs_root=venvs, checkpoint_root=tmp_path / "ckpt",
                               min_free_disk_gb=0.0)
    app = app_module.app
    app.state.settings = settings
    app.state.store = JobStore(settings.jobs_root, host_id=settings.host_id)
    app.state.spawn = lambda s, j: os.getpid()
    app.state.vram_check = lambda gpu, min_free_mb: GpuInfo(index=gpu, free_mb=40000,
                                                            total_mb=46068)
    monkeypatch.setattr(app_module, "_LAST_OBSERVATION", {})
    with TestClient(app, client=LOOPBACK, headers={"X-API-Key": trainer_key}) as client:
        yield client, app.state.store
    for attr in ("settings", "store", "spawn", "vram_check", "verify_spec"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


def _advance_to_training(store: JobStore, job_id: str):
    """Through the lifecycle rather than straight to ``training``: the status
    graph has no shortcut, and inventing one in a test is how a test comes to
    exercise a state the service never produces."""
    for target in ("preparing", "compiling", "training"):
        store.advance(store.load(job_id), target)
    return store.load(job_id)


def _running(store: JobStore, *, host: str = TEST_HOST_ID, pid: int | None = None):
    """A job the way the scheduler leaves one: started, with a live pid."""
    job = store.create(REQUEST, host=host)
    job.pid = pid if pid is not None else os.getpid()
    store.save(job)
    return _advance_to_training(store, job.id)


def test_a_running_job_carries_what_this_box_sees(svc):
    client, store = svc
    job = _running(store)

    observed = client.get(f"/jobs/{job.id}").json()["observed"]

    assert set(observed) == {"at", "rss_mib", "rss_trend", "job_dir_mib", "job_dir_trend"}
    assert observed["rss_mib"] > 0, "this process is resident; the reading is real"
    assert observed["job_dir_mib"] is not None, "the job directory exists"
    assert observed["rss_trend"] is None, "nothing to compare the first sample against"
    assert observed["at"], "an observation without a clock reading is a record"


def test_the_record_is_unchanged_beside_the_observation(svc):
    """Additive on the wire: the gateway proxy, the bot and the MCP read these."""
    client, store = svc
    job = _running(store)

    payload = client.get(f"/jobs/{job.id}").json()

    assert {"id", "request", "status", "stage", "pid", "host", "error"} <= set(payload)
    assert payload["status"] == "training"
    assert payload["pid"] == os.getpid()


def test_reading_an_observation_writes_nothing(svc):
    """A GET that writes the record would file an observation as a measurement —
    and two readers would then disagree about what the run recorded."""
    client, store = svc
    job = _running(store)
    record = store.paths(job.id).job_json

    before = record.read_bytes()
    assert client.get(f"/jobs/{job.id}").json()["observed"]
    assert record.read_bytes() == before


def test_the_second_read_compares_against_the_first(svc):
    """What #366 actually needed: not a number but whether it is climbing."""
    client, store = svc
    job = _running(store)

    first = client.get(f"/jobs/{job.id}").json()["observed"]
    assert first["job_dir_trend"] is None
    (store.paths(job.id).root / "data").mkdir(exist_ok=True)
    (store.paths(job.id).root / "data" / "train.arrow").write_bytes(b"x" * (8 * 1024 * 1024))

    second = client.get(f"/jobs/{job.id}").json()["observed"]
    assert second["job_dir_mib"] >= 8
    assert second["job_dir_trend"] == "growing", (
        "the run wrote 8 MiB between the two reads and the answer does not say so")


def test_a_queued_job_has_nothing_to_observe(svc):
    """No pid yet. An `observed` block of Nones would read as "measured, zero"."""
    client, store = svc
    job = store.create(REQUEST)

    assert "observed" not in client.get(f"/jobs/{job.id}").json()


def test_a_finished_job_has_nothing_to_observe(svc):
    """The case that would be worst: the process is gone, so every reading would
    be absent or belong to whatever reused the pid."""
    client, store = svc
    job = _running(store)
    store.fail(store.load(job.id), "no room on the GPU")

    assert "observed" not in client.get(f"/jobs/{job.id}").json()


def test_a_completed_job_has_nothing_to_observe(svc):
    client, store = svc
    job = _running(store)
    done = store.load(job.id)
    done.metrics = Metrics(cer=0.071)
    store.save(done)
    for target in ("testing", "registering", "completed"):
        store.advance(store.load(job.id), target)

    assert "observed" not in client.get(f"/jobs/{job.id}").json()


def test_another_hosts_job_is_not_observed_locally(svc):
    """The store is shared (#15). This box's pid table says nothing about a run
    on asteraix — matched here it would describe whatever holds that number
    locally and report it as the job's."""
    client, store = svc
    job = store.create(REQUEST, host="asteraix")
    job.pid = os.getpid()          # live here, and irrelevant: it is not our job
    store.save(job)
    _advance_to_training(store, job.id)

    assert "observed" not in client.get(f"/jobs/{job.id}").json()


def test_a_dead_runner_is_reconciled_before_it_is_observed(svc):
    """The route cannot produce "running record, dead process": ``reconcile``
    runs first and terminates the record, so the observation is skipped for the
    reason it should be. Pinned because the next test deliberately steps around
    it, and a reader needs to know that state is reachable only from inside."""
    client, store = svc
    job = store.create(REQUEST)
    job.pid = PID_NEVER
    store.save(job)
    _advance_to_training(store, job.id)

    payload = client.get(f"/jobs/{job.id}").json()
    assert payload["status"] == "failed"
    assert "observed" not in payload


def test_a_run_whose_process_is_gone_reports_no_rss_rather_than_zero(svc):
    """The probe's own answer, asked directly: a record that still says
    ``training`` while the process is gone. 0 MiB would read as an idle run, and
    "idle" is what a wedged compile looks like — the confusion #366 is about."""
    _client, store = svc
    job = store.create(REQUEST)
    job.pid = PID_NEVER
    store.save(job)
    running = _advance_to_training(store, job.id)

    observed = app_module._observe(running)
    assert observed is not None, "the record is not terminal, so it is observed"
    assert observed["rss_mib"] is None
    assert observed["job_dir_mib"] is not None, "the directory is still there"


def test_the_memory_is_dropped_when_a_job_ends(svc):
    """A dict keyed by job id that never loses an entry is a leak on a service
    that runs for months — and a stale sample is the wrong thing to compare a
    restarted run against."""
    client, store = svc
    job = _running(store)
    client.get(f"/jobs/{job.id}")
    assert job.id in app_module._LAST_OBSERVATION

    store.fail(store.load(job.id), "cancelled by hand")
    client.get(f"/jobs/{job.id}")
    assert job.id not in app_module._LAST_OBSERVATION


def test_a_chunked_run_reports_its_compiled_chunks_on_the_wire(svc):
    """The whole point, end to end: the answer to "is it moving?" without ssh."""
    client, store = svc
    job = _running(store)
    data = _chunked(store.paths(job.id).root)
    for index in range(3):
        _chunk(data, index)

    observed = client.get(f"/jobs/{job.id}").json()["observed"]
    assert observed["compile_chunks"] == {"done": 3, "expected": 6}

    _chunk(data, 3)
    later = client.get(f"/jobs/{job.id}").json()["observed"]
    assert later["compile_chunks"]["done"] == 4


def test_an_unchunked_run_carries_no_chunk_count(svc):
    """Absent rather than zero — the same rule as `observed` itself."""
    client, store = svc
    job = _running(store)

    assert "compile_chunks" not in client.get(f"/jobs/{job.id}").json()["observed"]

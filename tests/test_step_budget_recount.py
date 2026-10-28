"""#264 after the fact: the step budget of page runs, rewritten in samples.

The guard divided transcribed lines by the batch; the trainer steps over pages.
The fixtures below are four pages of two lines each, as in the pipeline tests:
eight ``train_lines``, four rows of ``train.jsonl``, and a record that says
``ceil(8 / 2) = 4`` steps an epoch where the trainer took two.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atr_training.step_budget_recount import (
    SIDECAR,
    apply_to_job,
    recount_job,
)

BUILDER = "20261001T093909Z-olmocr2-7b-medieval-german-page-v1"
ADOPTER = "20261008T103636Z-olmocr2-7b-medieval-page-px1288"
KEY = "2ad500ee8414"


def _job(store: Path, job_id: str, *, artefact: str, rows: int | None = 4,
         granularity: str = "page", status: str = "completed", steps: int | None = 4,
         total: int | None = 4, batch: int = 2, epochs: int = 1,
         max_steps: int | None = None, checkpoint: Path | None = None) -> Path:
    job = store / job_id
    (job / "data").mkdir(parents=True)
    if rows is not None:
        (job / "data" / "train.jsonl").write_text(
            "".join(json.dumps({"image": f"data/pages/{i}.jpg", "text": "x"}) + "\n"
                    for i in range(rows)), encoding="utf-8")
    (job / "job.json").write_text(json.dumps({
        "id": job_id, "status": status,
        "checkpoint_dir": str(checkpoint) if checkpoint else None,
        "request": {"engine": "vllm", "params": {
            "granularity": granularity, "batch_size": batch,
            "accumulate_grad_batches": 1, "epochs": epochs, "max_steps": max_steps}},
        "progress": {"train_lines": 8, "steps_per_epoch": steps, "total_steps": total,
                     "artefact": artefact, "pages_written": 6},
    }), encoding="utf-8")
    return job


def _recount(job: Path, *, caches=None):
    store = job.parent
    jobs = {p.name: p for p in store.iterdir() if (p / "job.json").is_file()}
    return recount_job(job, jobs, caches or {})


def _builder(tmp_path, **kw) -> Path:
    return _job(tmp_path, BUILDER, artefact=f"{KEY} built by this job", **kw)


def test_a_page_run_is_rewritten_from_its_own_corpus(tmp_path):
    recount = _recount(_builder(tmp_path))

    assert recount.verdict == "correct"
    assert recount.samples == 4 and recount.source == "own train.jsonl"
    assert (recount.steps_before, recount.steps_after) == (4, 2)
    assert (recount.total_before, recount.total_after) == (4, 2)


def test_an_adopter_counts_the_corpus_it_adopted(tmp_path):
    _builder(tmp_path, rows=4)
    adopter = _job(tmp_path, ADOPTER, artefact=f"{KEY} reused, built by {BUILDER}", rows=None)

    recount = _recount(adopter)

    assert recount.verdict == "correct"
    assert recount.samples == 4 and recount.source == f"builder {BUILDER}"
    assert recount.steps_after == 2


def test_a_builder_of_another_corpus_is_not_believed(tmp_path):
    """The builder's train.jsonl counts only if the builder built *this* key."""
    _job(tmp_path, BUILDER, artefact="ffffffffffff built by this job", rows=9)
    adopter = _job(tmp_path, ADOPTER, artefact=f"{KEY} reused, built by {BUILDER}", rows=None)

    assert _recount(adopter).verdict == "skip"


def test_the_cache_entry_stands_in_for_a_builder_that_is_gone(tmp_path):
    entry = tmp_path / "cache" / (KEY + "0" * 52)
    (entry / "data").mkdir(parents=True)
    (entry / "data" / "train.jsonl").write_text("{}\n{}\n{}\n", encoding="utf-8")
    store = tmp_path / "jobs"
    store.mkdir()
    adopter = _job(store, ADOPTER, artefact=f"{KEY} reused, built by {BUILDER}", rows=None)

    recount = _recount(adopter, caches={KEY: entry})

    assert recount.verdict == "correct"
    assert recount.samples == 3 and recount.source == f"cache {KEY}"


def test_no_corpus_anywhere_means_no_rewrite(tmp_path):
    recount = _recount(_builder(tmp_path, rows=None))
    assert recount.verdict == "skip"
    assert recount.steps_after is None


@pytest.mark.parametrize("granularity", ["line", None])
def test_a_line_run_was_already_in_samples(tmp_path, granularity):
    recount = _recount(_builder(tmp_path, granularity=granularity))
    assert recount.verdict == "line"


def test_a_record_without_a_budget_is_not_listed(tmp_path):
    assert _recount(_builder(tmp_path, steps=None, total=None)).verdict == "line"


def test_a_live_job_is_left_alone(tmp_path):
    recount = _recount(_builder(tmp_path, status="training"))
    assert recount.verdict == "skip" and "not touched" in recount.why


def test_a_budget_already_in_samples_is_left_alone(tmp_path):
    assert _recount(_builder(tmp_path, steps=2, total=2)).verdict == "already"


def test_the_rewrite_honours_epochs_and_the_step_cap(tmp_path):
    recount = _recount(_builder(tmp_path, epochs=3, total=12))
    assert recount.total_after == 6                       # 2 steps × 3 epochs

    capped = _recount(_job(tmp_path / "b", BUILDER, artefact=f"{KEY} built by this job",
                           epochs=3, total=5, max_steps=5))
    assert capped.total_after == 5                        # the cap ends it first


def test_the_trainer_counts_are_reported_and_not_used(tmp_path):
    checkpoint = tmp_path / "ckpt"
    (checkpoint / "checkpoint-6").mkdir(parents=True)
    (checkpoint / "checkpoint-4").mkdir()
    (checkpoint / "training_summary.json").write_text('{"train_samples": 4}')
    (checkpoint / "checkpoint-6" / "trainer_state.json").write_text(
        '{"global_step": 6, "epoch": 3.0}')
    (checkpoint / "checkpoint-4" / "trainer_state.json").write_text(
        '{"global_step": 4, "epoch": 2.0}')
    recount = _recount(_builder(tmp_path / "s", checkpoint=checkpoint))

    assert (recount.trainer_samples, recount.trainer_steps_per_epoch) == (4, 2)
    assert recount.agrees is True

    (checkpoint / "training_summary.json").write_text('{"train_samples": 5}')
    other = _recount(_builder(tmp_path / "t", checkpoint=checkpoint))
    assert other.samples == 4                             # train.jsonl decides
    assert other.agrees is False


def test_a_checkpoint_mid_epoch_says_nothing_about_an_epoch(tmp_path):
    checkpoint = tmp_path / "ckpt"
    (checkpoint / "checkpoint-3").mkdir(parents=True)
    (checkpoint / "checkpoint-3" / "trainer_state.json").write_text(
        '{"global_step": 3, "epoch": 1.5}')
    recount = _recount(_builder(tmp_path / "s", checkpoint=checkpoint))
    assert recount.trainer_steps_per_epoch is None


def test_applying_changes_two_fields_and_writes_the_sidecar(tmp_path):
    job = _builder(tmp_path)
    before = json.loads((job / "job.json").read_text())

    apply_to_job(job, _recount(job))

    after = json.loads((job / "job.json").read_text())
    assert after["progress"].pop("steps_per_epoch") == 2
    assert after["progress"].pop("total_steps") == 2
    before["progress"].pop("steps_per_epoch")
    before["progress"].pop("total_steps")
    assert after == before
    sidecar = json.loads((job / SIDECAR).read_text())
    assert sidecar["steps_per_epoch"] == {"before": 4, "after": 2}
    assert sidecar["train_samples"] == 4 and sidecar["issue"] == "#264"


def test_applying_twice_rewrites_once(tmp_path):
    job = _builder(tmp_path)
    apply_to_job(job, _recount(job))
    assert _recount(job).verdict == "already"


def test_a_job_that_started_running_meanwhile_is_refused(tmp_path):
    job = _builder(tmp_path)
    recount = _recount(job)
    record = json.loads((job / "job.json").read_text())
    record["status"] = "training"
    (job / "job.json").write_text(json.dumps(record))

    with pytest.raises(RuntimeError, match="not touched"):
        apply_to_job(job, recount)
    assert not (job / SIDECAR).exists()


def test_the_dry_run_writes_nothing(tmp_path, capsys):
    import importlib.util
    script = Path(__file__).resolve().parents[1] / "scripts" / "recount_step_budget.py"
    spec = importlib.util.spec_from_file_location("recount_step_budget", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    job = _builder(tmp_path / "jobs")
    record = (job / "job.json").read_text()

    assert module.main(["--store", str(tmp_path / "jobs"), "--cache", str(tmp_path / "none")]) == 0

    assert (job / "job.json").read_text() == record
    assert not (job / SIDECAR).exists()
    out = capsys.readouterr().out
    assert "4 → 2" in out and "dry run, nothing written" in out


def test_without_pydantic_the_script_says_where_to_run_it(tmp_path):
    """The login node's python3 has no pydantic. The first --apply died there
    with a traceback from contracts.py; it should name the way that works."""
    import subprocess
    import sys
    script = Path(__file__).resolve().parents[1] / "scripts" / "recount_step_budget.py"
    blocked = ("import sys, runpy; sys.modules['pydantic'] = None; "
               f"sys.argv = [{str(script)!r}, '--store', {str(tmp_path)!r}, '--apply']; "
               f"runpy.run_path({str(script)!r}, run_name='__main__')")
    done = subprocess.run([sys.executable, "-c", blocked], capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin"})

    assert done.returncode != 0
    assert "Traceback" not in done.stderr
    assert "inside the container" in done.stderr
    assert "Nothing was read or written" in done.stderr

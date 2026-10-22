"""#206: a stage records the image and transformers it ran in — or says unknown.

Two UBELIX images differ by a transformers major version, and until 10.10.2026
neither the record nor the Slurm log said which one produced a CER; the batch
script then started printing it, and from here on the record carries it per
stage, beside ``code``. Three answers, never two: an image, *no container*
(asteraix's bare venvs, measured), or *unknown* (a record from before the
field) — and the MCP must keep the last two apart.
"""

from __future__ import annotations

from pathlib import Path

from atr_results_mcp.probe import results, runtime_of, summarise_job
from atr_training.codeversion import current_runtime, read_runtime
from atr_training.contracts import Runtime, StageRecord
from atr_training.jobstore import JobStore
from atr_training.settings import TrainerSettings
from kraken_train_svc.runner import Pipeline

from test_results_mcp_probe import cluster, write_job  # noqa: F401 — fixture
from test_train_svc_pipeline import FakeRunner, FakeSource, request_with

TF5 = "/storage/homefs/th19c587/ubelix/vlm-train-tf5.sif"


# ── reading the runtime ─────────────────────────────────────────────────────

def test_the_image_comes_from_the_container_runtime_first():
    """APPTAINER_CONTAINER is what the runtime sets inside an image; SIF is the
    batch script's own export and only a fallback."""
    inside = read_runtime({"APPTAINER_CONTAINER": TF5, "SIF": "/elsewhere.sif"},
                          version_of=lambda name: "5.17.0")
    assert inside == Runtime(image=TF5, transformers="5.17.0")
    exported_only = read_runtime({"SIF": TF5}, version_of=lambda name: "5.17.0")
    assert exported_only.image == TF5
    assert read_runtime({"SINGULARITY_CONTAINER": "/old.sif"}, version_of=lambda n: None).image == "/old.sif"


def test_no_container_is_a_measured_none_and_no_transformers_is_none():
    """asteraix runs its venvs bare: image None is a fact, and kraken's venv has
    no transformers at all."""
    bare = read_runtime({}, version_of=lambda name: None)
    assert bare == Runtime(image=None, transformers=None)
    assert bare.short() == "no container / transformers none"
    assert Runtime(image=TF5, transformers="5.17.0").short() == "vlm-train-tf5.sif / transformers 5.17.0"


def test_a_metadata_failure_never_raises():
    def broken(name):
        raise RuntimeError("metadata is odd here")
    assert read_runtime({}, version_of=broken).transformers is None


def test_the_process_reads_its_runtime_once():
    current_runtime.cache_clear()
    try:
        assert current_runtime() is current_runtime()
        assert isinstance(current_runtime(), Runtime)
    finally:
        current_runtime.cache_clear()


# ── the record carries it per stage ─────────────────────────────────────────

def _settings(tmp_path: Path) -> TrainerSettings:
    (tmp_path / "registry").mkdir()
    return TrainerSettings(
        jobs_root=tmp_path / "training", trained_root=tmp_path / "trained",
        registry_root=tmp_path / "registry",
        checkpoint_root=tmp_path / "local-scratch" / "checkpoints",
        ketos=tmp_path / "ketos", min_free_disk_gb=0.0, gpu=1,
        artefact_cache=False, artefact_cache_root=tmp_path / "artefacts")


def test_every_stage_the_runner_opens_records_its_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("APPTAINER_CONTAINER", TF5)
    current_runtime.cache_clear()
    settings = _settings(tmp_path)
    store = JobStore(settings.jobs_root, host_id=settings.host_id)
    try:
        job = store.create(request_with())
        job = Pipeline(store, settings, runner=FakeRunner(),
                       source=FakeSource({"train": 4, "eval": 2})).execute(job.id)
    finally:
        current_runtime.cache_clear()

    assert job.status == "completed", job.error
    opened = [stage for stage in job.stages if stage.runtime is not None]
    assert {stage.name for stage in opened} >= {"prepare", "compile", "train", "test"}
    for stage in opened:
        assert stage.runtime.image == TF5
        assert stage.code is not None, "runtime sits beside code, not instead of it"
    assert all(stage.runtime == opened[0].runtime for stage in opened)


def test_a_record_from_before_the_field_still_loads_without_a_runtime():
    old = StageRecord.model_validate({"name": "train", "status": "completed"})
    assert old.runtime is None
    assert "runtime" not in old.model_dump(exclude_none=True)


# ── the MCP keeps the three answers apart ───────────────────────────────────

def _stages(runtime=None, *, names=("train", "test")) -> list:
    stages = []
    for name in names:
        stage = {"name": name, "status": "completed", "started_at": "2026-10-22T06:00:00Z",
                 "finished_at": "2026-10-22T08:00:00Z", "code": {"commit": "0b58494a18b0" + "0" * 28}}
        if runtime is not None:
            stage["runtime"] = runtime
        stages.append(stage)
    return stages


def test_runtime_of_names_the_image_from_the_test_stage_first():
    record = {"stages": _stages({"image": "/other/vlm-train.sif", "transformers": "4.57.6"},
                                names=("train",))
              + _stages({"image": TF5, "transformers": "5.17.0"}, names=("test",))}
    answer = runtime_of(record)
    assert answer == {"known": True, "stage": "test", "image": "vlm-train-tf5.sif",
                      "transformers": "5.17.0", "text": "vlm-train-tf5.sif / transformers 5.17.0"}


def test_no_container_and_unknown_are_not_the_same_answer():
    bare = runtime_of({"stages": _stages({"image": None, "transformers": "4.57.6"})})
    assert bare["known"] is True and bare["image"] is None
    assert bare["text"] == "no container / transformers 4.57.6"

    before = runtime_of({"stages": _stages()})
    assert before["known"] is False and before["image"] is None
    assert before["text"] == "unknown" and "before #206" in before["why"]
    assert runtime_of({"stages": []})["known"] is False


def test_job_and_results_carry_the_runtime(cluster):  # noqa: F811 — the fixture
    with_field = write_job(cluster.scratch, "20261022T060000Z-ladder-med-qwen35-4b-tf5",
                           extra={"stages": _stages({"image": TF5, "transformers": "5.17.0"})})
    without = write_job(cluster.scratch, "20260930T211455Z-ladder-med-qwen35-9b")

    import json
    new = summarise_job(with_field, json.loads((with_field / "job.json").read_text()))
    old = summarise_job(without, json.loads((without / "job.json").read_text()))
    assert new["runtime"]["text"] == "vlm-train-tf5.sif / transformers 5.17.0"
    assert new["stages"][0]["image"] == "vlm-train-tf5.sif"
    assert old["runtime"]["known"] is False
    assert "image" not in old["stages"][0], "an absent field is absent, not None"

    rows = {row["id"]: row for row in results(cluster)["rows"]}
    assert rows["20261022T060000Z-ladder-med-qwen35-4b-tf5"]["runtime"] == \
        "vlm-train-tf5.sif / transformers 5.17.0"
    assert rows["20260930T211455Z-ladder-med-qwen35-9b"]["runtime"] == "unknown"

"""#108: a run that adopts a compiled corpus draws the same stratified test set.

The defect was a property, not a field: the second arm of a ladder adopted the
first one's corpus from the artefact cache, lost ``dataset_counts`` with it, and
its test stage drew at random — "0 source(s) could be attributed, so no strata"
— so the two arms were scored on different pages and their difference could not
be told apart from the difference between two samples.

``dataset_counts`` has travelled in the payload since ``1b6167d`` (#129), and
``test_sweep_guardrails`` pins that the payload *names* the field. This pins
what the field is for: through the real pipeline, a compiling run and an
adopting run plan the same rows, stratified, each tagged with its source.
"""

from __future__ import annotations

import json

import pytest

from atr_training.contracts import DatasetSpec, TrainRequest, VlmTrainParams
from atr_training.jobstore import JobStore
from atr_training.settings import TrainerSettings

from vlm_train_svc.runner import Pipeline

# The fakes are shared with the VLM pipeline suite (see test_training_preemption).
from test_vlm_train_pipeline import THUN_TEST, THUN_TRAIN, FakeRunner, FakeSource


class TwoSourceFake(FakeSource):
    """Pages whose document id names their dataset.

    The shared fake names every page ``<role>_<i>``, so both datasets would hand
    in a document called ``eval`` — and :func:`~atr_training.eval_subset.attribute`
    rightly refuses a document that votes for two sources. Real Transkribus
    docIds belong to one collection; here the dataset name stands in for one.
    """

    def stream(self, hf_repo, data_files, revision=None):
        name = hf_repo.split("image-text_")[-1]
        for row in super().stream(hf_repo, data_files, revision):
            row["filename"] = f"{name}-{row['filename']}"
            yield row


@pytest.fixture
def settings(tmp_path):
    venvs = tmp_path / "venvs"
    (venvs / "vlm-train" / "bin").mkdir(parents=True)
    (venvs / "vlm-train" / "bin" / "python").touch()
    (tmp_path / "registry").mkdir()
    return TrainerSettings(
        jobs_root=tmp_path / "training",
        trained_root=tmp_path / "trained",
        registry_root=tmp_path / "registry",
        checkpoint_root=tmp_path / "local-scratch" / "checkpoints",
        venvs_root=venvs,
        min_free_disk_gb=0.0,
        gpu=1,
        # Never the developer's real cache: what is in it would decide the test.
        artefact_cache_root=tmp_path / "artefacts",
    )


@pytest.fixture
def store(settings):
    return JobStore(settings.jobs_root, host_id=settings.host_id)


def _two_source_request(model_id: str) -> TrainRequest:
    """Two datasets, so a stratified draw has strata; a cap below the validation
    set, so there is something to draw."""
    datasets = [DatasetSpec(hf_repo=f"dh-unibe/image-text_{name}",
                            train_projects=[THUN_TRAIN], eval_projects=[THUN_TEST])
                for name in ("alpha", "beta")]
    return TrainRequest(engine="vllm", model_id=model_id, datasets=datasets, force=True,
                        params=VlmTrainParams(eval_samples=4))


def _run(store, settings, model_id: str):
    job = store.create(_two_source_request(model_id))
    done = Pipeline(store, settings, runner=FakeRunner(),
                    source=TwoSourceFake({"train": 3, "eval": 4})).execute(job.id)
    assert done.status == "completed", done.error
    rows = [json.loads(line) for line in
            (store.paths(job.id).data / "val_eval.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return done, rows


def test_an_adopting_run_draws_what_the_compiling_run_drew(store, settings):
    builder, built_rows = _run(store, settings, "qwen3vl-ladder-arm-a")
    adopter, adopted_rows = _run(store, settings, "qwen3vl-ladder-arm-b")

    assert "built by this job" in (builder.progress.artefact or "")
    assert "reused" in (adopter.progress.artefact or ""), "the second arm compiled after all"

    # The property first: stratified, on both sources, and the same pages.
    for rows in (built_rows, adopted_rows):
        assert rows and all("source" in row for row in rows), \
            "a row without its source is a random draw"
        assert {row["source"] for row in rows} == {"alpha", "beta"}
    assert [r["image"] for r in adopted_rows] == [r["image"] for r in built_rows]
    # …and the field that carries it, second.
    assert adopter.progress.dataset_counts == builder.progress.dataset_counts

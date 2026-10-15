"""Documents reserved for evaluation never reach the training side (#98).

Filed because it had already happened: 198 of the 200 test documents of the
German hold-out were in the training set of
`20260910T110352Z-qwen3vl-german-pages-v3`, and 146 of its 150 validation
documents. The set existed, the split record said `leak_documents_into_train: 0`,
and nothing carried that from the split into the next run's selection.
"""

from __future__ import annotations

import json

import pytest

from atr_training.heldout import (
    DEFAULT_REGISTRY,
    HeldOutDocuments,
    document_of,
    load_heldout,
)


def page(index: int, doc: str) -> str:
    return f"/j/data/pages/{index:06d}_{doc}_0001_999.xml"


# ── reading a page's document ───────────────────────────────────────────────

def test_the_document_is_the_second_field():
    assert document_of("data/pages/002300_1627569_0049_60954083.xml") == "1627569"


def test_a_crop_name_yields_no_document_rather_than_a_wrong_one():
    """A line-granularity manifest holds crops. Guessing a document from those
    would drop training data for a reason nobody could reconstruct."""
    assert document_of("data/crops/train/000123.jpg") is None
    assert document_of("anything.xml") is None


# ── the split ───────────────────────────────────────────────────────────────

def test_reserved_pages_are_removed_and_the_rest_keeps_its_order():
    held = HeldOutDocuments(frozenset({"777"}), {"s": 1})
    pages = [page(1, "111"), page(2, "777"), page(3, "222"), page(4, "777")]
    keep, reserved = held.split(pages)
    assert keep == [page(1, "111"), page(3, "222")]
    assert reserved == [page(2, "777"), page(4, "777")]


def test_a_document_is_held_out_whole():
    """Pages of one manuscript share a hand: holding out pages leaks the hand."""
    held = HeldOutDocuments(frozenset({"777"}), {"s": 1})
    keep, reserved = held.split([page(i, "777") for i in range(5)])
    assert keep == [] and len(reserved) == 5


# ── the registry as committed ───────────────────────────────────────────────

def test_the_german_set_is_in_the_registry():
    held = load_heldout()
    assert held.sets["german-medieval-v1"] == 350        # 200 test + 150 val
    assert len(held.documents) == 350
    entry = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))["sets"][0]
    assert held.documents == frozenset(entry["test_documents"]) | frozenset(
        entry["val_documents"])


def test_the_committed_registry_is_well_formed():
    raw = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
    for entry in raw["sets"]:
        assert entry["datasets"] and entry["source_job"]
        both = set(entry["test_documents"]) & set(entry["val_documents"])
        assert not both, f"{entry['name']}: a document is in test and val: {both}"


def test_a_missing_registry_reserves_nothing_rather_than_failing(tmp_path):
    """This runs in the prepare stage of every job; an unreadable JSON file must
    not be what stops a run. It is logged at warning level instead."""
    assert not load_heldout(tmp_path / "absent.json")


def test_a_malformed_registry_reserves_nothing_rather_than_failing(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert not load_heldout(bad)


def test_an_entry_without_documents_is_not_counted(tmp_path):
    registry = tmp_path / "r.json"
    registry.write_text(json.dumps({"sets": [
        {"name": "empty", "test_documents": [], "val_documents": []},
        {"name": "real", "test_documents": ["1"], "val_documents": ["2"]},
    ]}))
    held = load_heldout(registry)
    assert held.sets == {"real": 2} and held.documents == frozenset({"1", "2"})


# ── and the pipeline honours it ─────────────────────────────────────────────

def test_the_prepare_stage_drops_reserved_pages(tmp_path, monkeypatch):
    from atr_training.contracts import DatasetSpec, TrainJob, TrainRequest
    from atr_training.jobstore import JobStore
    from atr_training.settings import TrainerSettings
    from kraken_train_svc.runner import Pipeline

    registry = tmp_path / "r.json"
    registry.write_text(json.dumps({"sets": [
        {"name": "german-medieval-v1", "test_documents": ["777"], "val_documents": []}]}))
    monkeypatch.setattr("atr_training.runner_base.load_heldout",
                        lambda: load_heldout(registry))

    store = JobStore(tmp_path / "jobs", host_id="asteraix")
    job = TrainJob(id="20260915T000000Z-kraken-heldout",
                   request=TrainRequest(engine="kraken", model_id="kraken-heldout",
                                        datasets=[DatasetSpec(hf_repo="dh-unibe/x")]))
    data = store.paths(job.id).data
    data.mkdir(parents=True)
    manifest = data / "pages_train.lst"
    manifest.write_text("\n".join([page(1, "111"), page(2, "777"), page(3, "222")]) + "\n")

    pipeline = Pipeline(store=store, settings=TrainerSettings(root=tmp_path / "jobs"))
    out = pipeline._reserve_eval_documents(job, manifest)

    assert out.read_text().splitlines() == [page(1, "111"), page(3, "222")]
    assert job.progress.reserved_pages == 1
    # What was removed is written down, not only counted.
    assert (data / "pages_reserved.lst").read_text().splitlines() == [page(2, "777")]


def test_a_selection_that_is_the_eval_set_is_refused(tmp_path, monkeypatch):
    from atr_training.contracts import DatasetSpec, TrainJob, TrainRequest
    from atr_training.jobstore import JobStore
    from atr_training.runner_base import DatasetSelectionError
    from atr_training.settings import TrainerSettings
    from kraken_train_svc.runner import Pipeline

    registry = tmp_path / "r.json"
    registry.write_text(json.dumps({"sets": [
        {"name": "s", "test_documents": ["777"], "val_documents": []}]}))
    monkeypatch.setattr("atr_training.runner_base.load_heldout",
                        lambda: load_heldout(registry))

    store = JobStore(tmp_path / "jobs", host_id="asteraix")
    job = TrainJob(id="20260915T000000Z-kraken-alleval",
                   request=TrainRequest(engine="kraken", model_id="kraken-alleval",
                                        datasets=[DatasetSpec(hf_repo="dh-unibe/x")]))
    data = store.paths(job.id).data
    data.mkdir(parents=True)
    manifest = data / "pages_train.lst"
    manifest.write_text("\n".join([page(1, "777"), page(2, "777")]) + "\n")

    pipeline = Pipeline(store=store, settings=TrainerSettings(root=tmp_path / "jobs"))
    with pytest.raises(DatasetSelectionError) as exc:
        pipeline._reserve_eval_documents(job, manifest)
    assert "is the eval set" in str(exc.value)


def _line_pipeline(tmp_path, job_id: str):
    from atr_training.contracts import DatasetSpec, TrainJob, TrainRequest
    from atr_training.jobstore import JobStore
    from atr_training.settings import TrainerSettings
    from kraken_train_svc.runner import Pipeline

    store = JobStore(tmp_path / "jobs", host_id="asteraix")
    job = TrainJob(id=job_id,
                   request=TrainRequest(engine="kraken", model_id="kraken-lines",
                                        datasets=[DatasetSpec(hf_repo="dh-unibe/x")]))
    data = store.paths(job.id).data
    data.mkdir(parents=True)
    return Pipeline(store=store, settings=TrainerSettings(root=tmp_path / "jobs")), job, data


def test_a_line_manifest_without_a_page_is_not_checked_rather_than_clean(tmp_path):
    """Crop names carry no document, so nothing may be dropped from them — and
    the record must not then claim the registry was consulted.

    This used to assert `reserved_pages == 0`, which is the same value a run gets
    after a real check that found nothing. #247 has to read 54 of these
    afterwards and cannot tell the two apart (#119).
    """
    pipeline, job, data = _line_pipeline(tmp_path, "20260915T000000Z-kraken-lines")
    manifest = data / "lines_train.jsonl"
    lines = ['{"image": "crops/a.jpg"}', '{"image": "crops/b.jpg"}']
    manifest.write_text("\n".join(lines) + "\n")

    pipeline._reserve_eval_documents(job, manifest)

    assert manifest.read_text().splitlines() == lines, "nothing may be dropped"
    assert job.progress.reserved_pages is None
    assert "no document id" in (job.progress.reserved_pages_source or "")


def test_a_line_manifest_is_checked_through_its_page_field(tmp_path, monkeypatch):
    """The fix, not only the refusal: a line corpus cropped from materialised
    pages *is* checkable — the page is right there in the sample."""
    import json as _json

    from atr_training import heldout, runner_base

    held = heldout.HeldOutDocuments(frozenset({"018155"}), {"german-medieval-v1": 1})
    monkeypatch.setattr(runner_base, "load_heldout", lambda *a, **kw: held)

    pipeline, job, data = _line_pipeline(tmp_path, "20260915T000001Z-kraken-lines")
    manifest = data / "lines_train.jsonl"
    reserved_line = _json.dumps({"image": "images/000001_x.jpg", "text": "wer dis liset",
                                 "page": "0_018155_f001r_IMG001.xml"})
    kept_line = _json.dumps({"image": "images/000002_y.jpg", "text": "der sol wissen",
                             "page": "1_099999_f002v_IMG002.xml"})
    manifest.write_text(reserved_line + "\n" + kept_line + "\n")

    pipeline._reserve_eval_documents(job, manifest)

    assert job.progress.reserved_pages == 1
    assert job.progress.reserved_pages_source == "prepare"
    assert manifest.read_text().splitlines() == [kept_line], "the reserved line is gone"


def test_a_slash_in_the_transcription_no_longer_decides_the_reservation(tmp_path, monkeypatch):
    """The measured defect.

    `document_of` takes `Path(entry).name` — everything after the **last slash in
    the whole JSON blob**. With a transcription containing "/" (`anno 1431 /
    vigilia`, routine in this material) the last slash moved into the text, the
    document came back None, and a held-out page stayed in training. Field order
    and transcription content decided whether the boundary held.
    """
    import json as _json

    from atr_training import heldout, runner_base

    held = heldout.HeldOutDocuments(frozenset({"018155"}), {"german-medieval-v1": 1})
    monkeypatch.setattr(runner_base, "load_heldout", lambda *a, **kw: held)

    pipeline, job, data = _line_pipeline(tmp_path, "20260915T000002Z-kraken-lines")
    manifest = data / "lines_train.jsonl"
    # Two lines, because a manifest that is *only* reserved material is the
    # other refusal — `DatasetSelectionError`, tested above.
    manifest.write_text("\n".join([
        _json.dumps({"image": "images/000001_x.jpg", "text": "anno 1431 / vigilia sancti",
                     "source_type": "line", "bbox": None,
                     "page": "0_018155_f001r_IMG001.xml"}),
        _json.dumps({"image": "images/000002_y.jpg", "text": "item dedi 4 lb / 3 s",
                     "source_type": "line", "bbox": None,
                     "page": "1_099999_f002v_IMG002.xml"}),
    ]) + "\n")

    pipeline._reserve_eval_documents(job, manifest)

    assert job.progress.reserved_pages == 1, (
        "the slash in the transcription decides the reservation again")
    assert "099999" in manifest.read_text(), "the unreserved line is still there"


def test_a_chunk_plan_is_not_read_as_a_page_list(tmp_path, monkeypatch):
    """What `prepare` returns in chunked mode is `train_plan.json`.

    Read as a manifest it is nine lines of JSON that name no document, so nothing
    was reserved and the record said `0`. The pages of a chunked run exist only
    inside the compile loop, so the registry is not consulted at all — and that
    is what the record now says.
    """
    from atr_training import heldout, runner_base
    from atr_training.chunking import CHUNK_PLAN_FILENAME

    held = heldout.HeldOutDocuments(frozenset({"018155"}), {"german-medieval-v1": 1})
    monkeypatch.setattr(runner_base, "load_heldout", lambda *a, **kw: held)

    pipeline, job, data = _line_pipeline(tmp_path, "20260915T000003Z-kraken-chunked")
    plan = data / CHUNK_PLAN_FILENAME
    body = json.dumps({"hf_repo": "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi",
                       "data_files": ["data/train-00000-of-00042.parquet"],
                       "revision": None, "max_pages": None, "chunk_pages": 5000},
                      indent=2)
    plan.write_text(body, encoding="utf-8")

    pipeline._reserve_eval_documents(job, plan)

    assert job.progress.reserved_pages is None
    assert "chunked" in (job.progress.reserved_pages_source or "")
    assert plan.read_text(encoding="utf-8") == body, "the plan must not be rewritten"


# ── a cache hit must not smuggle the reserved pages back in ────────────────

def test_the_artefact_key_changes_when_the_reservation_does(tmp_path, monkeypatch):
    """A cache hit skips `prepare`, which is where the reservation is applied —
    so an artefact compiled under a different reservation is a different corpus
    and must not be served under this one."""
    from atr_training import artefact_cache
    from atr_training.contracts import DatasetSpec

    spec = DatasetSpec(hf_repo="dh-unibe/x")
    monkeypatch.setattr(artefact_cache, "heldout_fingerprint", lambda: "aaaa")
    before = artefact_cache.key_for(spec, "vllm").digest
    monkeypatch.setattr(artefact_cache, "heldout_fingerprint", lambda: "bbbb")
    assert artefact_cache.key_for(spec, "vllm").digest != before


def test_a_box_with_nothing_reserved_keeps_one_namespace(tmp_path, monkeypatch):
    """No registry must not mean a new key namespace — that would invalidate every
    cached artefact on a box that reserves nothing, for no change in the corpus."""
    from atr_training import artefact_cache, heldout

    monkeypatch.setattr(heldout, "DEFAULT_REGISTRY", tmp_path / "absent.json")
    assert artefact_cache.heldout_fingerprint() == "none"


def test_the_fingerprint_follows_the_documents_not_the_file(tmp_path, monkeypatch):
    """Reordering or re-formatting the registry is not a corpus change."""
    from atr_training import artefact_cache, heldout

    first = tmp_path / "a.json"
    first.write_text(json.dumps({"sets": [
        {"name": "s", "test_documents": ["2", "1"], "val_documents": []}]}))
    second = tmp_path / "b.json"
    second.write_text(json.dumps({"sets": [
        {"name": "renamed", "test_documents": ["1"], "val_documents": ["2"]}], "x": 1},
        indent=4))

    monkeypatch.setattr(heldout, "DEFAULT_REGISTRY", first)
    a = artefact_cache.heldout_fingerprint()
    monkeypatch.setattr(heldout, "DEFAULT_REGISTRY", second)
    assert artefact_cache.heldout_fingerprint() == a != "none"

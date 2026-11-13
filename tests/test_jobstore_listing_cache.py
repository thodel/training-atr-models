"""#408: listing the store reads only the records that changed.

`GET /jobs` and every scheduler tick list the whole store, and the store is on
the share. Cold, the gateway's poll took 30–46 s, past its 20 s. One `stat` per
job now tells "unchanged since the last listing"; only a changed `job.json` is
read again. What the listing returns must not change by this.
"""

from pathlib import Path

import pytest

from atr_training.contracts import DatasetSpec, TrainRequest
from atr_training.jobstore import JobStore


def request() -> TrainRequest:
    return TrainRequest(model_id="kraken-thun-missiven-v1", dataset=DatasetSpec(
        hf_repo="dh-unibe/image-text_medieval-scripts_xiv-xv-xvi",
        train_projects=["a"], eval_projects=["b"]))


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "training", host_id="asteraix")


@pytest.fixture
def reads(monkeypatch):
    """Count job.json reads, wherever they come from."""
    counted: list[str] = []
    original = Path.read_text

    def read_text(self, *args, **kwargs):
        if self.name == "job.json":
            counted.append(self.parent.name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    return counted


def make(store, n):
    return [store.create(request(), job_id=store.new_job_id("m", now=f"2026110{i}T120000Z"))
            for i in range(1, n + 1)]


def test_an_unchanged_store_is_listed_without_reading_it_again(store, reads):
    make(store, 3)
    first = store.list()
    reads.clear()
    second = store.list()
    assert reads == []
    assert [j.model_dump() for j in second] == [j.model_dump() for j in first]


def test_a_saved_record_is_read_again_and_shows_its_new_state(store, reads):
    a, b, c = make(store, 3)
    store.list()
    reads.clear()
    loaded = store.load(b.id)
    store.advance(loaded, "preparing")
    reads.clear()
    listed = {j.id: j.status for j in store.list()}
    assert reads == [b.id]
    assert listed[b.id] == "preparing" and listed[a.id] == "queued"


def test_newest_first_and_the_limit_as_before(store):
    jobs = make(store, 4)
    assert [j.id for j in store.list()] == [j.id for j in reversed(jobs)]
    assert [j.id for j in store.list(limit=2)] == [jobs[3].id, jobs[2].id]
    assert store.list_ids() == [j.id for j in store.list()]


def test_a_directory_without_job_json_is_not_a_job_and_does_not_eat_the_limit(store):
    jobs = make(store, 2)
    (store.root / "20261109T120000Z-half-made").mkdir()
    assert [j.id for j in store.list(limit=1)] == [jobs[1].id]


def test_a_deleted_job_leaves_the_listing_and_the_cache(store):
    a, b = make(store, 2)
    store.list()
    store.delete(a.id)
    assert [j.id for j in store.list()] == [b.id]
    assert set(store._listing) == {b.id}


def test_a_corrupt_record_is_skipped_not_fatal(store):
    a, b = make(store, 2)
    store.paths(a.id).job_json.write_text("{not json", encoding="utf-8")
    assert [j.id for j in store.list()] == [b.id]


def test_a_caller_changing_a_listed_record_does_not_change_the_next_listing(store):
    make(store, 1)
    first = store.list()[0]
    first.error = "changed by a caller"
    assert store.list()[0].error is None

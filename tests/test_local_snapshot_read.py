"""Reading a pinned, fully cached selection without asking the hub (#204).

A request with one datasets entry per project — 576 Leuven registers from
``image-text_medieval-scripts_xiv-xv-xvi``, capped at 25 pages each — spent the
hub quota of 1,000 requests per five minutes within the first minute of prepare
(07.11.2026, Slurm 21397276): every entry listed the whole repo once in
``collapse_complete_selection``, and ``load_dataset(repo, …)`` resolved every
pattern with its own tree request, although every byte was in the local cache.
"""

from __future__ import annotations

import sys
import types

import pytest

from atr_training import hf_source
from atr_training.hf_source import local_snapshot_files
from atr_training.prepare import HFPageSource

REPO = "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi"
REV = "729e9b2721ba2a31901f776368d8aacf0a5e9406"
LISTED = ("README.md",
          "data/train/SAL7301_full/a-0000.parquet",
          "data/train/SAL7303_full/b-0000.parquet",
          "data/train/SAL7303_full/b-0001.parquet")


@pytest.fixture
def hf_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    return tmp_path


def snapshot(hf_home, files=LISTED):
    root = hf_home / "hub" / f"datasets--{REPO.replace('/', '--')}" / "snapshots" / REV
    for name in files:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"PAR1")
    return root


def listing(repo, revision):
    return LISTED


# ── the resolution ──────────────────────────────────────────────────────────
def test_a_complete_pinned_selection_is_read_from_the_snapshot(hf_home):
    root = snapshot(hf_home)

    got = local_snapshot_files(REPO, ["data/train/SAL7303_full/*.parquet",
                                      "data/train/SAL7301_full/*.parquet"], REV,
                               files_fn=listing)

    assert got == [str(root / "data/train/SAL7303_full/b-0000.parquet"),
                   str(root / "data/train/SAL7303_full/b-0001.parquet"),
                   str(root / "data/train/SAL7301_full/a-0000.parquet")]


def test_one_shard_missing_sends_the_whole_selection_to_the_hub(hf_home):
    """A partly cached project must not become a smaller corpus without a word."""
    snapshot(hf_home, files=LISTED[:3])           # b-0001 never fetched

    assert local_snapshot_files(REPO, ["data/train/SAL7303_full/*.parquet"], REV,
                                files_fn=listing) is None


@pytest.mark.parametrize("revision", [None, "main", "0" * 40])
def test_no_snapshot_for_the_revision_means_the_hub(hf_home, revision):
    """Unpinned, or pinned to a revision this cache has never seen."""
    snapshot(hf_home)

    assert local_snapshot_files(REPO, ["data/train/SAL7303_full/*.parquet"], revision,
                                files_fn=listing) is None


def test_a_pattern_the_hub_does_not_list_means_the_hub(hf_home):
    snapshot(hf_home)

    assert local_snapshot_files(REPO, ["data/train/SAL9999/*.parquet"], REV,
                                files_fn=listing) is None


def test_a_failed_listing_means_the_hub(hf_home):
    snapshot(hf_home)

    def down(repo, revision):
        raise OSError("429 Too Many Requests")

    assert local_snapshot_files(REPO, ["data/train/SAL7303_full/*.parquet"], REV,
                                files_fn=down) is None


# ── the stream ──────────────────────────────────────────────────────────────
@pytest.fixture
def fake_datasets(monkeypatch):
    """`datasets.load_dataset`, recording how it was called."""
    calls: list[tuple] = []

    def load_dataset(path, **kwargs):
        calls.append((path, kwargs))
        return []

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    return calls


def test_the_stream_reads_parquet_where_it_lies(hf_home, fake_datasets, monkeypatch):
    root = snapshot(hf_home)
    monkeypatch.setattr(hf_source, "repo_files", listing)

    list(HFPageSource(cache=False).stream(REPO, ["data/train/SAL7303_full/*.parquet"], REV))

    (path, kwargs), = fake_datasets
    assert path == "parquet"
    assert kwargs["data_files"] == {"train": [str(root / "data/train/SAL7303_full/b-0000.parquet"),
                                              str(root / "data/train/SAL7303_full/b-0001.parquet")]}
    assert kwargs["streaming"] is True and "revision" not in kwargs


def test_the_stream_falls_back_to_the_hub_as_before(hf_home, fake_datasets, monkeypatch):
    monkeypatch.setattr(hf_source, "repo_files", listing)          # no snapshot on disk

    list(HFPageSource(cache=False).stream(REPO, ["data/train/SAL7303_full/*.parquet"], REV))

    (path, kwargs), = fake_datasets
    assert path == REPO and kwargs["revision"] == REV


# ── one listing per process ─────────────────────────────────────────────────
def test_576_entries_list_the_repo_once(monkeypatch):
    seen: list[tuple] = []

    class FakeApi:
        def list_repo_files(self, repo, revision=None, repo_type="dataset"):
            seen.append((repo, revision))
            return list(LISTED)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))

    for _ in range(576):
        hf_source.collapse_complete_selection("train", ["SAL7303_full"], REPO, REV)

    assert seen == [(REPO, REV)]
    assert hf_source.list_projects(REPO, "train", REV) == ["SAL7301_full", "SAL7303_full"]
    assert seen == [(REPO, REV)]


def test_a_failed_listing_is_not_remembered(monkeypatch):
    attempts: list[int] = []

    class FlakyApi:
        def list_repo_files(self, repo, revision=None, repo_type="dataset"):
            attempts.append(1)
            if len(attempts) == 1:
                raise OSError("429 Too Many Requests")
            return list(LISTED)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FlakyApi))

    with pytest.raises(hf_source.ProjectListingError):
        hf_source.list_projects(REPO, "train", REV)
    assert hf_source.list_projects(REPO, "train", REV) == ["SAL7301_full", "SAL7303_full"]

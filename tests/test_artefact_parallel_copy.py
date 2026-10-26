"""A directory goes into the cache with many files in flight, not one (#238).

The first XIX VLM artefact — 87.5 GB, 1,019,120 files, most of them 34 KB line
crops — took 14.5 h to copy from the CIFS share to the NVMe cache, one `copy2`
after another. Measured on asteraix on 17.10.2026: one stream moves 109 crops/s,
32 move 548–756/s; page images go from 22 to ~100 MB/s. These pin that the
parallel copy produces exactly what `copytree` produced, counts every byte, and
does not swallow a failure.
"""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path

import pytest

import atr_training.artefact_cache as ac


def tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    for sub in ("crops/train", "crops/val", "pages"):
        (root / sub).mkdir(parents=True)
    for i in range(120):
        (root / "crops" / "train" / f"{i:07d}.jpg").write_bytes(os.urandom(100 + i))
    for i in range(30):
        (root / "crops" / "val" / f"{i:07d}.jpg").write_bytes(os.urandom(50 + i))
    (root / "pages" / "a.xml").write_text("<x/>")
    (root / "train.jsonl").write_text('{"image": "data/crops/train/0000000.jpg"}\n')
    (root / "empty").mkdir()
    return root


def test_the_copy_is_the_tree_copytree_would_have_made(corpus, tmp_path):
    reference = tmp_path / "reference"
    shutil.copytree(corpus, reference)
    ours = tmp_path / "ours"
    ac._copytree_parallel(corpus, ours, ac._Counter(), workers=8)
    assert tree(ours) == tree(reference)
    assert (ours / "empty").is_dir()


def test_every_byte_is_counted_across_threads(corpus, tmp_path):
    counted = ac._Counter()
    ac._copytree_parallel(corpus, tmp_path / "ours", counted, workers=16)
    assert counted.total == sum(len(b) for b in tree(corpus).values())


def test_the_files_really_are_copied_concurrently(corpus, tmp_path):
    seen: set[int] = set()

    def copy(src, dst):
        seen.add(threading.get_ident())
        shutil.copy2(src, dst)

    ac._copytree_parallel(corpus, tmp_path / "ours", copy, workers=8)
    assert len(seen) > 1


def test_a_failed_file_is_raised_not_swallowed(corpus, tmp_path):
    """`_store_artefact` turns an OSError into "not stored" and trains on the
    job's own copy; a copy that hid it would store half a corpus."""
    def copy(src, dst):
        if src.endswith("0000042.jpg"):
            raise OSError("CIFS went away")
        shutil.copy2(src, dst)

    with pytest.raises(OSError, match="CIFS went away"):
        ac._copytree_parallel(corpus, tmp_path / "ours", copy, workers=8)


def test_an_existing_target_is_refused_as_copytree_refuses_it(corpus, tmp_path):
    (tmp_path / "ours").mkdir()
    with pytest.raises(FileExistsError):
        ac._copytree_parallel(corpus, tmp_path / "ours", ac._Counter())


def test_one_worker_is_the_old_sequential_copy(corpus, tmp_path):
    ac._copytree_parallel(corpus, tmp_path / "ours", ac._Counter(), workers=1)
    assert tree(tmp_path / "ours") == tree(corpus)


def test_put_stores_the_directory_through_the_parallel_copy(corpus, tmp_path, monkeypatch):
    calls = []
    real = ac._copytree_parallel
    monkeypatch.setattr(ac, "_copytree_parallel",
                        lambda *a, **k: (calls.append(a[0]), real(*a, **k))[1])
    cache = ac.ArtefactCache(tmp_path / "cache")
    key = ac.ArtefactKey(digest="f" * 64, pinned=True)
    entry = cache.put(key, corpus, job_id="j", inner="data")
    assert calls == [corpus]
    assert entry.bytes_ == sum(len(b) for b in tree(corpus).values())
    assert tree(entry.path / "data") == tree(corpus)


# ── same filesystem: hard links, not copies (#238) ──────────────────────────
#
# On UBELIX the job directory and the cache are both on /scratch (st_dev 51 for
# both, 26.10.2026), and storing xix-v2's 966,748 files took 14 of the job's
# 16.5 hours as copies (#22). A hard link writes no data and leaves the job's
# own paths where resume and the promotion gate read them.

def test_on_one_filesystem_the_entry_shares_the_jobs_inodes(corpus, tmp_path):
    cache = ac.ArtefactCache(tmp_path / "cache")
    entry = cache.put(ac.ArtefactKey(digest="e" * 64, pinned=True), corpus,
                      job_id="j", inner="data")
    stored = entry.path / "data" / "crops" / "train" / "0000007.jpg"
    original = corpus / "crops" / "train" / "0000007.jpg"
    assert stored.stat().st_ino == original.stat().st_ino
    assert original.stat().st_nlink == 2
    assert original.exists(), "the job keeps its own path"
    manifest = __import__("json").loads((entry.path / cache.MANIFEST).read_text())
    assert manifest["linked_files"] == len(tree(corpus))
    assert entry.bytes_ == sum(len(b) for b in tree(corpus).values())


def test_a_refused_link_falls_back_to_copying_everything(corpus, tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(ac.os, "link", refuse)
    counted = ac._Counter(link=True)
    ac._copytree_parallel(corpus, tmp_path / "ours", counted, workers=4)
    assert tree(tmp_path / "ours") == tree(corpus)
    assert counted.link is False and counted.linked == 0
    assert (corpus / "pages" / "a.xml").stat().st_nlink == 1


def test_across_filesystems_nothing_is_linked(corpus, tmp_path, monkeypatch):
    """asteraix: the share (CIFS) and the cache (NVMe) are two devices."""
    monkeypatch.setattr(ac, "_same_filesystem", lambda a, b: False)
    cache = ac.ArtefactCache(tmp_path / "cache")
    entry = cache.put(ac.ArtefactKey(digest="d" * 64, pinned=True), corpus,
                      job_id="j", inner="data")
    stored = entry.path / "data" / "pages" / "a.xml"
    assert stored.stat().st_ino != (corpus / "pages" / "a.xml").stat().st_ino
    manifest = __import__("json").loads((entry.path / cache.MANIFEST).read_text())
    assert manifest["linked_files"] == 0


def test_evicting_a_linked_entry_leaves_the_job_its_files(corpus, tmp_path):
    cache = ac.ArtefactCache(tmp_path / "cache")
    entry = cache.put(ac.ArtefactKey(digest="c" * 64, pinned=True), corpus,
                      job_id="j", inner="data")
    shutil.rmtree(entry.path)
    assert tree(corpus)["pages/a.xml"] == b"<x/>"

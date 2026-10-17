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

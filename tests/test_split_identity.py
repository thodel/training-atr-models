"""Restoring the eval split, and what "the same split" is allowed to mean (#112).

The arrows of `german_test`/`german_val` were removed on 21.09.2026; the 350
document ids survived in `config/heldout_eval_documents.json`. K0 says the
restore is done when the drawn document ids match that file, "checked, not
assumed".

The test in the middle of this file is why that criterion is not sufficient on
its own, and it is the reason `split_identity` exists rather than a five-line
comparison: the per-document caps change which PAGES are held out without
changing which DOCUMENTS are, so a rebuild can satisfy K0's check to the letter
and still hold out a different set of pages. The caps are not in the registry's
`built_by`.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from atr_training.heldout import load_heldout
from atr_training.split_identity import (
    PARAM_FIELDS, documents_of, load_identity, page_digest, record_digests,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_split = _load("make_split")


#: A pool where the cap bites on some documents and not others — which is the
#: only situation in which `rng.sample` is reached at all.
def _pool(documents: int = 60, big_every: int = 3) -> list[str]:
    lines = []
    for doc in range(documents):
        pages = 300 if doc % big_every == 0 else 5
        lines += [f"{doc * 10000 + p}_{doc:06d}_{p:04d}_9.xml" for p in range(pages)]
    return sorted(set(lines))


def _draw(pool, **over):
    params = dict(seed=20260810, val_docs=5, val_per_doc=160,
                  test_docs=5, test_per_doc=220, shard_pages=50000)
    params.update(over)
    held, _, record = make_split.build_split(pool, **params)
    return held, record


# ── the digest ──────────────────────────────────────────────────────────────
def test_the_digest_ignores_the_path_a_box_happened_to_use():
    """The same split materialised on asteraix and on UBELIX carries different
    prefixes. Digesting whole lines would call those two different sets."""
    bare = ["495426_171954_0406_6478463.xml", "495427_171954_0407_6478464.xml"]
    prefixed = [f"/mnt/share/pool/{name}" for name in bare]

    assert page_digest(bare) == page_digest(prefixed)


def test_the_digest_ignores_the_order_of_the_manifest():
    pages = ["1_000001_0001_9.xml", "2_000002_0002_9.xml", "3_000003_0003_9.xml"]

    assert page_digest(pages) == page_digest(list(reversed(pages)))


def test_the_digest_ignores_a_repeated_line():
    pages = ["1_000001_0001_9.xml", "2_000002_0002_9.xml"]

    assert page_digest(pages) == page_digest(pages + pages[:1])


def test_one_different_page_is_a_different_digest():
    a = ["1_000001_0001_9.xml", "2_000002_0002_9.xml"]
    b = ["1_000001_0001_9.xml", "2_000002_0003_9.xml"]

    assert page_digest(a) != page_digest(b)


def test_an_empty_list_is_named_rather_than_digested():
    """`empty` beats the sha256 of nothing, which looks like a real value."""
    assert page_digest([]) == "empty"


# ── the finding: documents are not pages ────────────────────────────────────
def test_the_caps_do_not_change_which_documents_are_drawn():
    """Because the draw pops from a list shuffled once and consumes no
    randomness of its own. This is the half K0's criterion relies on, and it
    holds."""
    pool = _pool()
    a, _ = _draw(pool)
    b, _ = _draw(pool, test_per_doc=221)

    for side in ("test", "val"):
        assert documents_of(a[side]) == documents_of(b[side])


def test_the_caps_do_change_which_pages_are_drawn():
    """The half K0's criterion misses, and the reason this module exists.

    `rng.sample` for an over-cap document consumes the generator, so the test
    draw shifts the stream the val draw runs on. One more page of head-room in
    `test_per_doc` exchanged 63 of 335 val pages here while every val *document*
    stayed identical — a rebuild that passes "the document ids match" and holds
    out different pages than the set it claims to be.
    """
    pool = _pool()
    a, _ = _draw(pool)
    b, _ = _draw(pool, test_per_doc=221)

    assert documents_of(a["val"]) == documents_of(b["val"])
    assert page_digest(a["val"]) != page_digest(b["val"])


def test_the_registry_now_records_every_parameter_the_draw_needs():
    """It recorded seed, val_docs and test_docs; the three caps were defaults
    nobody wrote down. Missing any of them makes the split unreproducible."""
    assert load_identity().missing_params == []


def test_the_recovered_parameters_are_marked_as_recovered():
    """They come from make_split.py's history, not from the record. A later
    reader must be able to tell those apart."""
    recovered = load_identity().params.get("_recovered")

    assert "val_per_doc" in recovered and "test_per_doc" in recovered


def test_reading_the_registry_for_held_out_documents_is_unaffected():
    """`load_heldout` runs in every job's prepare stage; adding fields to the
    file must not change what it reserves."""
    held = load_heldout()

    assert len(held.documents) == 350
    assert held.sets == {"german-medieval-v1": 350}


# ── verifying a rebuild ─────────────────────────────────────────────────────
@pytest.fixture
def registry(tmp_path: Path):
    """A registry describing the split that `_pool()` produces, with no digests
    pinned — the state the real one is in."""
    pool = _pool()
    held, _ = _draw(pool)
    path = tmp_path / "heldout.json"
    path.write_text(json.dumps({"sets": [{
        "name": "synthetic-v1",
        "source_job": "20260905T190759Z-kraken-german-eval-pool-v1",
        "built_by": "scripts/make_split.py --val-docs 5 --test-docs 5",
        "split_params": {"seed": 20260810, "val_docs": 5, "val_per_doc": 160,
                         "test_docs": 5, "test_per_doc": 220,
                         "shard_pages": 50000,
                         "manifests": ["pages_train.lst"]},
        "test_documents": sorted(documents_of(held["test"])),
        "val_documents": sorted(documents_of(held["val"])),
    }]}), encoding="utf-8")
    return path, held


def test_an_identical_rebuild_is_accepted(registry):
    path, held = registry
    identity = load_identity("synthetic-v1", path)

    verdict = identity.verify(test_pages=held["test"], val_pages=held["val"])

    assert verdict.ok, verdict.complaints


def test_an_identical_rebuild_is_still_not_proof_about_the_pages(registry):
    """The distinction the whole module turns on: accepted, and unproven."""
    path, held = registry
    identity = load_identity("synthetic-v1", path)

    verdict = identity.verify(test_pages=held["test"], val_pages=held["val"])

    assert verdict.ok
    assert verdict.pages_proven is False
    assert "UNPROVEN" in verdict.summary()


def test_a_missing_document_is_named(registry):
    path, held = registry
    identity = load_identity("synthetic-v1", path)
    dropped = sorted(documents_of(held["test"]))[0]
    thinner = [p for p in held["test"] if f"_{dropped}_" not in p]

    verdict = identity.verify(test_pages=thinner, val_pages=held["val"])

    assert not verdict.ok
    assert any(dropped in c for c in verdict.complaints), verdict.complaints


def test_a_document_that_was_never_in_the_set_is_named(registry):
    path, held = registry
    identity = load_identity("synthetic-v1", path)
    intruder = held["test"] + ["9999_999999_0001_9.xml"]

    verdict = identity.verify(test_pages=intruder, val_pages=held["val"])

    assert not verdict.ok
    assert any("999999" in c for c in verdict.complaints), verdict.complaints


def test_a_document_in_both_halves_is_refused(registry):
    """Two folios of one manuscript are not two independent measurements, and a
    per-side comparison alone would not see it."""
    path, held = registry
    identity = load_identity("synthetic-v1", path)

    verdict = identity.verify(test_pages=held["test"],
                              val_pages=held["val"] + held["test"][:1])

    assert not verdict.ok
    assert any("BOTH" in c for c in verdict.complaints), verdict.complaints


def test_a_pinned_digest_catches_what_the_document_check_cannot(registry):
    """The point of pinning, in one test: same documents, different pages,
    refused — which is what K0's own criterion would have waved through."""
    path, held = registry
    other, _ = _draw(_pool(), test_per_doc=221)
    record_digests("synthetic-v1", test_pages=held["test"],
                   val_pages=held["val"], registry=path)
    identity = load_identity("synthetic-v1", path)

    verdict = identity.verify(test_pages=other["test"], val_pages=other["val"])

    assert documents_of(other["val"]) == identity.val_documents  # documents agree
    assert not verdict.ok
    assert any("page digest" in c for c in verdict.complaints), verdict.complaints


def test_a_pinned_digest_accepts_the_set_it_was_pinned_from(registry):
    path, held = registry
    record_digests("synthetic-v1", test_pages=held["test"],
                   val_pages=held["val"], registry=path)
    identity = load_identity("synthetic-v1", path)

    verdict = identity.verify(test_pages=held["test"], val_pages=held["val"])

    assert verdict.ok and verdict.pages_proven


def test_a_recorded_digest_is_not_quietly_moved(registry):
    """A pin that a later restore can overwrite is not a pin."""
    path, held = registry
    other, _ = _draw(_pool(), test_per_doc=221)
    record_digests("synthetic-v1", test_pages=held["test"],
                   val_pages=held["val"], registry=path)

    with pytest.raises(ValueError, match="two different sets under one name"):
        record_digests("synthetic-v1", test_pages=other["test"],
                       val_pages=other["val"], registry=path)


def test_recording_the_same_digest_twice_is_fine(registry):
    path, held = registry
    record_digests("synthetic-v1", test_pages=held["test"],
                   val_pages=held["val"], registry=path)

    record_digests("synthetic-v1", test_pages=held["test"],
                   val_pages=held["val"], registry=path)


def test_a_set_that_is_not_there_is_an_error_rather_than_the_first_one(registry):
    path, _ = registry

    with pytest.raises(KeyError, match="no set named"):
        load_identity("does-not-exist", path)


# ── the script ──────────────────────────────────────────────────────────────
def _run(*args, cwd=ROOT):
    return subprocess.run([sys.executable, str(SCRIPTS / "restore_eval_split.py"), *args],
                          capture_output=True, text=True, timeout=120, cwd=cwd)


@pytest.fixture
def pool_dir(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    (data / "pages_train.lst").write_text("\n".join(_pool()) + "\n")
    return data


def test_locate_names_the_job_and_the_places_to_look():
    """The operator is on a laptop behind a VPN; the value of this is that the
    paths are right without anybody retyping them."""
    result = _run("--locate")

    assert result.returncode == 0, result.stderr
    assert "20260905T190759Z-kraken-german-eval-pool-v1" in result.stdout
    assert "/mnt/wbkolleg_dh_1" in result.stdout


def test_a_rebuild_stops_short_of_success_while_the_pages_are_unproven(
        registry, pool_dir, tmp_path):
    """Exit 2: not a failure, not a pass. The documents check out and the pages
    cannot, and the operator decides which of those is enough."""
    path, _ = registry

    result = _run("--set", "synthetic-v1", "--registry", str(path),
                  "--from-pool", str(pool_dir), "--out", str(tmp_path / "split"))

    assert result.returncode == 2, result.stdout + result.stderr
    assert "UNPROVEN" in result.stdout


def test_the_operator_can_take_the_unproven_pages_on_the_record(
        registry, pool_dir, tmp_path):
    path, _ = registry

    result = _run("--set", "synthetic-v1", "--registry", str(path),
                  "--from-pool", str(pool_dir), "--out", str(tmp_path / "split"),
                  "--accept-unproven-pages")

    assert result.returncode == 0, result.stdout + result.stderr


def test_recording_from_a_rebuild_says_what_it_did_and_did_not_show(
        registry, pool_dir, tmp_path):
    path, _ = registry

    result = _run("--set", "synthetic-v1", "--registry", str(path),
                  "--from-pool", str(pool_dir), "--out", str(tmp_path / "split"),
                  "--record")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "it does not show it is what it was" in result.stdout
    assert load_identity("synthetic-v1", path).proves_pages


def test_a_rebuild_writes_the_lists_and_a_record_that_carries_the_parameters(
        registry, pool_dir, tmp_path):
    """K1 cites this file; a split.json without the parameters it was drawn with
    puts the next reader back where #112 started."""
    path, _ = registry
    out = tmp_path / "split"

    _run("--set", "synthetic-v1", "--registry", str(path),
         "--from-pool", str(pool_dir), "--out", str(out), "--accept-unproven-pages")

    record = json.loads((out / "split.json").read_text())
    assert (out / "pages_test.lst").is_file() and (out / "pages_val.lst").is_file()
    assert record["leak_documents_into_train"] == 0
    assert set(PARAM_FIELDS) <= set(record["split_params"])
    assert record["page_digests"]["test"]


def test_surviving_lists_are_read_rather_than_redrawn(registry, tmp_path):
    """The good branch: nothing is drawn, so nothing can drift."""
    path, held = registry
    split = tmp_path / "split"
    split.mkdir()
    (split / "pages_test.lst").write_text("\n".join(held["test"]) + "\n")
    (split / "pages_val.lst").write_text("\n".join(held["val"]) + "\n")

    result = _run("--set", "synthetic-v1", "--registry", str(path),
                  "--from-split-dir", str(split), "--record")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "this set is now confirmable" in result.stdout


def test_a_split_dir_that_does_not_match_the_registry_fails(registry, tmp_path):
    path, held = registry
    split = tmp_path / "split"
    split.mkdir()
    (split / "pages_test.lst").write_text("9999_999999_0001_9.xml\n")
    (split / "pages_val.lst").write_text("\n".join(held["val"]) + "\n")

    result = _run("--set", "synthetic-v1", "--registry", str(path),
                  "--from-split-dir", str(split))

    assert result.returncode == 1, result.stdout + result.stderr


def test_an_empty_pool_says_the_material_is_missing_not_that_nothing_matched(
        registry, tmp_path):
    """The likeliest real failure: the pool was never materialised. A split of
    zero pages would otherwise be reported as a document mismatch."""
    path, _ = registry
    empty = tmp_path / "data"
    empty.mkdir()

    result = _run("--set", "synthetic-v1", "--registry", str(path),
                  "--from-pool", str(empty), "--out", str(tmp_path / "split"))

    assert result.returncode != 0
    assert "materialised" in result.stdout + result.stderr


def test_a_registry_without_the_caps_refuses_to_guess_them(pool_dir, tmp_path):
    """The state the real registry was in before this change: it would have
    drawn a split with whatever the defaults are today and called it the set."""
    path = tmp_path / "heldout.json"
    path.write_text(json.dumps({"sets": [{
        "name": "old-shape", "seed": 20260810,
        "test_documents": ["000001"], "val_documents": ["000002"],
    }]}), encoding="utf-8")

    result = _run("--set", "old-shape", "--registry", str(path),
                  "--from-pool", str(pool_dir), "--out", str(tmp_path / "split"))

    assert result.returncode != 0
    assert "val_per_doc" in result.stdout + result.stderr

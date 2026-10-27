"""A new held-out set is registered from what make_split.py wrote (#301, way c).

`german-medieval-v1` was typed into the registry by hand, and three of its six
split parameters had to be recovered from git history afterwards (#112). These
pin the route for the next one, the Kurrent page set of #301:

* make_split.py writes all six parameters and the manifests into split.json;
* the entry carries the documents of both sides and the page digests, so
  build_page_benchmark.py can prove a list against it;
* refused: a taken name, incomplete parameters, a leak into train, a document
  on both sides, a document another set already holds, a page without one;
* nothing is written without --write, because a registered document leaves the
  training side of every later run.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from atr_training.heldout import load_heldout
from atr_training.split_identity import load_identity, page_digest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


make_split = _load("make_split")
register = _load("register_eval_set")


def _pool(tmp_path: Path, documents: int = 30, pages: int = 6) -> Path:
    """A job data/ dir: pages named <page>_<document>_<folio>_<image>.xml."""
    data = tmp_path / "data"
    data.mkdir()
    lines = [f"/scratch/jobs/x/data/pages/{d * 100 + p}_{900000 + d}_{p:04d}_{d}{p}.xml"
             for d in range(documents) for p in range(pages)]
    (data / "pages_train.lst").write_text("\n".join(lines[:150]) + "\n")
    (data / "pages_val.lst").write_text("\n".join(lines[150:]) + "\n")
    return data


def _registry(tmp_path: Path, sets=(), name: str = "heldout.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"_why": "test", "sets": list(sets)}, indent=2) + "\n")
    return path


#: The registry the draw is cut by: one older set holding two documents of the pool.
OLDER = {"name": "older-set", "test_documents": ["900003"], "val_documents": ["900011"]}


def _split(tmp_path: Path, *extra) -> Path:
    data = _pool(tmp_path)
    drawn_against = _registry(tmp_path, [OLDER], name="registry_at_draw.json")
    assert make_split.main([str(data), "--test-docs", "5", "--val-docs", "3",
                            "--test-per-doc", "4", "--seed", "7",
                            "--registry", str(drawn_against), *extra]) == 0
    return data / "split"


def _args(split: Path, registry: Path, *extra):
    return [str(split), "--name", "german-kurrent-xvi-xix-v1", "--source-job", "job-x",
            "--datasets", "dh-unibe/b", "dh-unibe/a", "--registry", str(registry), *extra]


def test_make_split_records_every_parameter(tmp_path):
    record = json.loads((_split(tmp_path) / "split.json").read_text())
    assert record["params"] == {"seed": 7, "val_docs": 3, "val_per_doc": 160,
                                "test_docs": 5, "test_per_doc": 4, "shard_pages": 50000,
                                "manifests": ["pages_train.lst", "pages_val.lst"],
                                "exclude_sets": ["older-set"]}


def test_a_document_another_set_holds_never_enters_the_draw(tmp_path):
    """27.10.2026: `034854` of german-medieval-v1 sat on the XVI–XIX corpus's
    validation side, the draw took it, and the registration was refused."""
    split = _split(tmp_path)
    record = json.loads((split / "split.json").read_text())
    drawn = [line for name in ("pages_test.lst", "pages_val.lst", "shard_00.lst")
             for line in (split / name).read_text().split()]
    assert {register.document_of(line) for line in drawn}.isdisjoint({"900003", "900011"})
    assert record["excluded"] == {"sets": ["older-set"], "documents": 2, "pages": 12,
                                  "which": ["900003", "900011"]}
    assert record["total_documents"] == 28 and record["total_pages"] == 168


def test_the_cut_split_registers_beside_the_older_set(tmp_path):
    split = _split(tmp_path)
    registry = _registry(tmp_path, [OLDER])
    assert register.main(_args(split, registry, "--write")) == 0
    entry = json.loads(registry.read_text())["sets"][1]
    assert entry["built_by"].endswith("--exclude-sets older-set"), \
        "the draw is repeatable after the new set joins the registry"


def test_the_sets_are_named_so_the_draw_repeats_after_registration(tmp_path):
    first = json.loads((_split(tmp_path) / "split.json").read_text())
    later = _registry(tmp_path, [OLDER, {"name": "the-new-one", "test_documents": ["900000"]}],
                      name="later.json")
    data = tmp_path / "data"
    assert make_split.main([str(data), "--test-docs", "5", "--val-docs", "3",
                            "--test-per-doc", "4", "--seed", "7", "--registry", str(later),
                            "--exclude-sets", "older-set", "--out", str(tmp_path / "again")]) == 0
    again = json.loads((tmp_path / "again" / "split.json").read_text())
    assert again["test"] == first["test"] and again["val"] == first["val"]
    assert (tmp_path / "again" / "pages_test.lst").read_text() == \
        (data / "split" / "pages_test.lst").read_text()


@pytest.mark.parametrize("args,why", [
    (["--exclude-sets", "no-such-set"], "no-such-set"),
    (["--registry", "/nonexistent/heldout.json"], "cannot read the held-out registry"),
])
def test_an_unknown_set_or_a_missing_registry_stops_the_draw(tmp_path, capsys, args, why):
    data = _pool(tmp_path)
    registry = _registry(tmp_path, [OLDER], name="r.json")
    with pytest.raises(SystemExit) as stop:
        make_split.main([str(data), "--registry", str(registry), *args])
    assert stop.value.code == 2 and why in capsys.readouterr().err
    assert not (data / "split").exists(), "nothing drawn"


def test_the_entry_holds_documents_digests_and_how_to_draw_it_again(tmp_path):
    split = _split(tmp_path)
    registry = _registry(tmp_path)
    assert register.main(_args(split, registry, "--write")) == 0
    (entry,) = json.loads(registry.read_text())["sets"]
    assert len(entry["test_documents"]) == 5 and len(entry["val_documents"]) == 3
    test_pages = (split / "pages_test.lst").read_text().split()
    assert entry["page_digests"]["test"] == page_digest(test_pages)
    assert entry["datasets"] == ["dh-unibe/a", "dh-unibe/b"]
    assert "--test-per-doc 4" in entry["built_by"] and "--seed 7" in entry["built_by"]
    identity = load_identity("german-kurrent-xvi-xix-v1", registry)
    assert identity.test_page_digest == entry["page_digests"]["test"]
    assert set(identity.params) >= {"seed", "val_docs", "val_per_doc", "test_docs",
                                    "test_per_doc", "shard_pages"}
    held = load_heldout(registry)
    assert held.sets == {"german-kurrent-xvi-xix-v1": 8}, "the trainer will drop all eight"


def test_nothing_is_written_without_write(tmp_path, capsys):
    registry = _registry(tmp_path)
    before = registry.read_text()
    assert register.main(_args(_split(tmp_path), registry)) == 0
    assert registry.read_text() == before
    assert "Not written" in capsys.readouterr().out


def test_writing_keeps_the_rest_of_the_registry_byte_for_byte(tmp_path):
    other = {"name": "german-medieval-v1", "test_documents": ["018155"], "val_documents": []}
    registry = _registry(tmp_path, [other])
    head = registry.read_text()[:30]
    assert register.main(_args(_split(tmp_path), registry, "--write")) == 0
    raw = json.loads(registry.read_text())
    assert raw["sets"][0] == other and raw["_why"] == "test"
    assert registry.read_text().startswith(head)


@pytest.mark.parametrize("breakage,why", [
    ("name", "already registered"),
    ("params", "does not record test_per_doc"),
    ("leak", "held-out document(s) in train"),
    ("both", "on both sides"),
    ("taken", "already belong to 'german-medieval-v1'"),
    ("nodoc", "names no document"),
    ("datasets", "--datasets"),
])
def test_what_is_refused(tmp_path, capsys, breakage, why):
    split = _split(tmp_path)
    sets = []
    record = json.loads((split / "split.json").read_text())
    test_pages = (split / "pages_test.lst").read_text().split()
    if breakage == "name":
        sets = [{"name": "german-kurrent-xvi-xix-v1", "test_documents": ["1"]}]
    elif breakage == "params":
        del record["params"]["test_per_doc"]
    elif breakage == "leak":
        record["leak_documents_into_train"] = 2
    elif breakage == "both":
        (split / "pages_val.lst").write_text(test_pages[0] + "\n")
    elif breakage == "taken":
        doc = register.document_of(test_pages[0])
        sets = [{"name": "german-medieval-v1", "test_documents": [doc]}]
    elif breakage == "nodoc":
        (split / "pages_test.lst").write_text("line_crop.png\n")
    (split / "split.json").write_text(json.dumps(record))
    registry = _registry(tmp_path, sets)
    before = registry.read_text()
    args = _args(split, registry, "--write")
    if breakage == "datasets":
        args = [a for a in args if not a.startswith("dh-unibe/") and a != "--datasets"]
    assert register.main(args) == 3
    assert why in capsys.readouterr().err
    assert registry.read_text() == before, "a refusal writes nothing"

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


def _split(tmp_path: Path) -> Path:
    data = _pool(tmp_path)
    assert make_split.main([str(data), "--test-docs", "5", "--val-docs", "3",
                            "--test-per-doc", "4", "--seed", "7"]) == 0
    return data / "split"


def _registry(tmp_path: Path, sets=()) -> Path:
    path = tmp_path / "heldout.json"
    path.write_text(json.dumps({"_why": "test", "sets": list(sets)}, indent=2) + "\n")
    return path


def _args(split: Path, registry: Path, *extra):
    return [str(split), "--name", "german-kurrent-xvi-xix-v1", "--source-job", "job-x",
            "--datasets", "dh-unibe/b", "dh-unibe/a", "--registry", str(registry), *extra]


def test_make_split_records_every_parameter(tmp_path):
    record = json.loads((_split(tmp_path) / "split.json").read_text())
    assert record["params"] == {"seed": 7, "val_docs": 3, "val_per_doc": 160,
                                "test_docs": 5, "test_per_doc": 4, "shard_pages": 50000,
                                "manifests": ["pages_train.lst", "pages_val.lst"]}


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

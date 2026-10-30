"""#341: one prepared run, checked before its GPU stage, answered in the exit status.

A corpus built after `german-kurrent-xvi-xix-v1` joined the registry must hold
none of its pages. `scripts/heldout_boundary.py --job` says so between prepare
and train, where finding out costs minutes rather than GPU days. Only `clean`
exits 0: `crossed`, `uncheckable` and `gone` are all reasons not to train on.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("heldout_boundary_cli",
                                               ROOT / "scripts" / "heldout_boundary.py")
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)

HELD = "018155"
OTHER = "099999"


def page(document: str, folio: str = "f001r") -> str:
    return f"/scratch/jobs/x/data/pages/0_{document}_{folio}_IMG001.xml"


@pytest.fixture
def registry(tmp_path) -> Path:
    path = tmp_path / "heldout.json"
    path.write_text(json.dumps({"sets": [
        {"name": "german-medieval-v1", "test_documents": ["000001"], "val_documents": []},
        {"name": "german-kurrent-xvi-xix-v1", "test_documents": [HELD], "val_documents": []},
    ]}))
    return path


def job(root: Path, job_id: str, *, pages: list[str] | None, artefact: str | None = None,
        reserved: int | None = 1318) -> Path:
    directory = root / job_id
    (directory / "data").mkdir(parents=True)
    (directory / "job.json").write_text(json.dumps({
        "id": job_id, "status": "training", "stage": "train",
        "progress": {"reserved_pages": reserved, "artefact": artefact}}))
    if pages is not None:
        (directory / "data" / "pages_train.lst").write_text("".join(p + "\n" for p in pages))
    return directory


def test_a_corpus_without_held_out_documents_exits_0(tmp_path, registry, capsys):
    run = job(tmp_path / "jobs", "20261031T000000Z-v2", pages=[page(OTHER), page("077777")])
    assert cli.main(["--job", str(run), "--registry", str(registry)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("clean:")
    assert "german-kurrent-xvi-xix-v1 (1)" in out and "german-medieval-v1 (1)" in out


def test_one_page_of_a_held_out_document_exits_1_and_names_it(tmp_path, registry, capsys):
    run = job(tmp_path / "jobs", "20261031T000000Z-v2", pages=[page(OTHER), page(HELD)])
    assert cli.main(["--job", str(run), "--registry", str(registry)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("crossed:") and HELD in out


def test_a_run_whose_manifest_is_gone_is_not_cleared(tmp_path, registry, capsys):
    run = job(tmp_path / "jobs", "20261031T000000Z-v2", pages=None)
    assert cli.main(["--job", str(run), "--registry", str(registry)]) == 1
    assert capsys.readouterr().out.startswith("gone:")


def test_an_adopting_run_is_judged_by_its_builder(tmp_path, registry, capsys):
    """The second v2 arm adopts the corpus the first one built: no manifest of
    its own, the builder's verdict, found in the same store."""
    store = tmp_path / "jobs"
    builder = job(store, "20261031T000000Z-builder", pages=[page(OTHER)],
                  artefact="abcdef123456 built by this job")
    adopter = job(store, "20261031T010000Z-adopter", pages=None,
                  artefact="abcdef123456 reused, built by 20261031T000000Z-builder")
    assert cli.main(["--job", str(adopter), "--registry", str(registry)]) == 0
    assert "adopted abcdef123456 from 20261031T000000Z-builder" in capsys.readouterr().out
    assert builder.is_dir()


def test_not_a_job_directory_exits_2(tmp_path, registry):
    (tmp_path / "empty").mkdir()
    assert cli.main(["--job", str(tmp_path / "empty"), "--registry", str(registry)]) == 2


def test_the_v2_specs_differ_from_v1_only_where_meant():
    """Same datasets, revisions and recipe as the October arms; a new model id so
    nothing is overwritten, and a token cap that does not cut long pages off."""
    specs = ROOT / "ubelix" / "specs"
    for v1, v2 in [("olmocr2-7b-german-kurrent-xvi-xix-page-px4m-native.json",
                    "olmocr2-7b-german-kurrent-xvi-xix-page-px4m-native-v2.json"),
                   ("german-kurrent-xvi-xix-page.json",
                    "qwen3.5-4b-german-kurrent-xvi-xix-page-v2.json")]:
        old = json.loads((specs / v1).read_text())
        new = json.loads((specs / v2).read_text())
        assert new["model_id"] == v2[:-len(".json")]
        assert new["params"]["max_new_tokens"] == 3072
        for d in (old, new):
            d.pop("model_id")
            d.pop("notes", None)
            d["params"].pop("max_new_tokens", None)
        assert old == new

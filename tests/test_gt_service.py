"""#200: the import path as a service — survey anything, import only the register.

Offline. GitHub and Zenodo answer from a dict; the importer is a stand-in script.
"""

from __future__ import annotations

import json
import sys
import time

import pytest
from fastapi.testclient import TestClient

from atr_training import gt_import
from atr_training.gt_import import GtImportError, GtImports, argv_for, resolve
from atr_training.gt_survey import SurveyError, survey
from atr_training.jobstore import JobStore
from atr_training.settings import TrainerSettings
from kraken_train_svc import app as app_module

CC0 = "Creative Commons Legal Code\n\nCC0 1.0 Universal\n\n... Public Domain Dedication"
CC_BY = "Attribution 4.0 International\n\nCreative Commons Corporation"


def fetcher(responses: dict[str, object]):
    seen: list[str] = []

    def fetch(url: str):
        seen.append(url)
        if url not in responses:
            return 404, b""
        body = responses[url]
        return 200, (json.dumps(body) if not isinstance(body, str) else body).encode()

    fetch.seen = seen
    return fetch


API = "https://api.github.com/repos/UB-Mannheim/toy-gt"
RAW = "https://raw.githubusercontent.com/UB-Mannheim/toy-gt/abc123"


def github(*, spdx="CC0-1.0", files: dict[str, str] | None = None, extra_tree=()):
    files = {"LICENSE": CC0} if files is None else files
    tree = [{"path": p, "type": "blob", "mode": "100644"} for p in files]
    tree += [{"path": "data/p1/page/0001.xml", "type": "blob", "mode": "100644"},
             {"path": "data/p1/page/0002.xml", "type": "blob", "mode": "100644"},
             {"path": "data/p1/0001.jpg", "type": "blob", "mode": "100644"},
             {"path": "data/p1/alto/gt", "type": "blob", "mode": "120000"},
             {"path": "get_images.sh", "type": "blob", "mode": "100755"}, *extra_tree]
    responses = {
        API: {"default_branch": "main", "pushed_at": "2024-01-01T00:00:00Z",
              "license": {"spdx_id": spdx} if spdx else None},
        f"{API}/commits/main": {"sha": "abc123"},
        f"{API}/git/trees/abc123?recursive=1": {"tree": tree, "truncated": False},
    }
    responses.update({f"{RAW}/{p}": text for p, text in files.items()})
    return fetcher(responses)


# ── survey: GitHub ───────────────────────────────────────────────────────────

def test_a_repository_whose_places_agree_gets_a_licence_in_the_draft():
    fetch = github(files={"LICENSE": CC0, ".zenodo.json": json.dumps({"license": "cc-zero"})})
    out = survey("https://github.com/UB-Mannheim/toy-gt", fetch)
    assert out["licence"] == {"state": "agree", "licences": ["CC0-1.0"], "unrecognised_at": []}
    assert {f["at"] for f in out["found"]} == {
        "GitHub-API `license.spdx_id`", "`LICENSE`", "`.zenodo.json`"}
    assert out["commit"] == "abc123" and out["branch"] == "main"
    assert 'licence="CC0-1.0",' in out["draft"]
    assert 'attribution="",' in out["draft"]          # nothing it can prove, left empty


def test_a_contradiction_names_both_places_and_sets_no_licence():
    """#191: the deposit's field against its own file. Reported, not decided."""
    fetch = github(files={"LICENSE": CC0, ".zenodo.json": json.dumps({"license": "cc-by-4.0"})})
    out = survey("https://github.com/UB-Mannheim/toy-gt", fetch)
    assert out["licence"]["state"] == "contradiction"
    assert out["licence"]["licences"] == ["CC-BY-4.0", "CC0-1.0"]
    assert 'licence="",  # nicht gesetzt: contradiction' in out["draft"]
    assert 'licence_at="",' in out["draft"]


def test_an_unreadable_licence_file_is_said_not_passed_over():
    fetch = github(spdx="NOASSERTION", files={"LICENSE": "All rights reserved, ask us."})
    out = survey("https://github.com/UB-Mannheim/toy-gt", fetch)
    assert out["licence"]["state"] == "unrecognised"
    assert "`LICENSE`" in out["licence"]["unrecognised_at"]


def test_a_repository_without_any_licence_says_none():
    out = survey("https://github.com/UB-Mannheim/toy-gt", github(spdx=None, files={}))
    assert out["licence"]["state"] == "none" and out["found"] == []


def test_metadata_files_are_read_by_their_licence_lines():
    meta = "title: x\nlicense:\n  - name: CC-BY 4.0\n    url: https://...\n"
    fetch = github(spdx=None, files={"htr-united.yml": meta,
                                     "CITATION.cff": "cff-version: 1.2.0\nlicense: CC-BY-4.0\n"})
    out = survey("https://github.com/UB-Mannheim/toy-gt", fetch)
    assert out["licence"] == {"state": "agree", "licences": ["CC-BY-4.0"], "unrecognised_at": []}


def test_size_symlinks_and_image_hints_are_counted():
    out = survey("https://github.com/UB-Mannheim/toy-gt/", github())
    assert out["size"]["xml"] == 2 and out["size"]["images"] == 1
    assert out["size"]["symlinks"] == 1               # the dach-gt trap, counted
    assert out["images"]["hints"] == ["get_images.sh"]


def test_a_branch_in_the_url_is_the_one_read():
    out = survey("https://github.com/UB-Mannheim/toy-gt/tree/main", github())
    assert out["branch"] == "main"


# ── survey: Zenodo ───────────────────────────────────────────────────────────

def zenodo(record="11046062", asked=None, licence="cc-by-sa-4.0", files=None, texts=None):
    asked = asked or record
    files = files if files is not None else [{"key": "Deutsch.zip", "size": 10},
                                             {"key": "LICENSE.txt", "size": 1}]
    responses = {f"https://zenodo.org/api/records/{asked}": {
        "id": int(record), "doi": f"10.5281/zenodo.{record}",
        "conceptdoi": "10.5281/zenodo.11046061",
        "metadata": {"title": "Inkunabeln Reichenau", "license": {"id": licence},
                     "creators": [{"name": "Muster, Anna"}]},
        "files": files}}
    for key, text in (texts or {"LICENSE.txt": "CC BY-SA 4.0"}).items():
        responses[f"https://zenodo.org/records/{record}/files/{key}?download=1"] = text
    return fetcher(responses)


def test_a_zenodo_record_reads_the_api_and_the_licence_file():
    out = survey("https://zenodo.org/records/11046062", zenodo())
    assert out["licence"]["state"] == "agree" and out["licence"]["licences"] == ["CC-BY-SA-4.0"]
    assert out["doi"] == "10.5281/zenodo.11046062"
    assert 'doi="10.5281/zenodo.11046062",' in out["draft"]
    assert '("Deutsch.zip",)' in out["draft"]


def test_a_zenodo_file_against_the_api_is_a_contradiction():
    out = survey("10.5281/zenodo.11046062", zenodo(texts={"LICENSE.txt": CC_BY}))
    assert out["licence"]["state"] == "contradiction"


def test_a_concept_record_is_named_and_the_version_taken():
    out = survey("https://doi.org/10.5281/zenodo.11046061",
                 zenodo(record="11046062", asked="11046061"))
    assert out["record"] == "11046062"
    assert any("nicht die Version" in n for n in out["notes"])


@pytest.mark.parametrize("url", [
    "http://github.com/a/b", "https://gitlab.com/a/b", "https://github.com/a",
    "https://127.0.0.1/a/b", "file:///etc/passwd", "https://zenodo.org/records/abc",
    "https://github.com/a/b/../../x", "https://github.com/../..", "",
])
def test_anything_but_github_or_zenodo_is_refused_before_a_request(url):
    fetch = fetcher({})
    with pytest.raises(SurveyError):
        survey(url, fetch)
    assert fetch.seen == []


# ── import: the register only, never public ─────────────────────────────────

def test_an_unknown_source_is_refused_with_the_known_ones():
    with pytest.raises(GtImportError) as exc:
        resolve("https://github.com/x/y")
    assert "reichsanzeiger-gt" in str(exc.value) and "reichenau-inkunabeln" in str(exc.value)


def test_the_command_line_never_carries_public():
    for source in gt_import.all_sources():
        argv = argv_for(source, dry_run=False, parts=[], all_parts=False)
        assert "--public" not in argv and argv[2] == source.id


def test_parts_only_for_zenodo_and_only_known_ones():
    with pytest.raises(GtImportError, match="only for Zenodo"):
        argv_for(resolve("reichsanzeiger-gt"), dry_run=True, parts=["x"], all_parts=False)
    with pytest.raises(GtImportError, match="no part"):
        argv_for(resolve("reichenau-inkunabeln"), dry_run=True, parts=["klingonisch"],
                 all_parts=False)
    argv = argv_for(resolve("reichenau-inkunabeln"), dry_run=True,
                    parts=["lateinisch"], all_parts=False)
    assert argv[-2:] == ["--part", "lateinisch"] and "--dry-run" in argv


STAND_IN = """import sys, time
time.sleep(float(__import__('os').environ.get('GT_FAKE_SLEEP', '0')))
print('── ' + sys.argv[1] + ' ──')
print('  Lizenz:     CC0-1.0  (LICENSE)')
print('  ' + sys.argv[1] + ': 27 Seiten, 3576 Zeilen (verworfen: 15 Dubletten, 0 Symlinks)')
print('  args: ' + ' '.join(sys.argv[2:]))
sys.exit(int(__import__('os').environ.get('GT_FAKE_EXIT', '0')))
"""


@pytest.fixture
def imports(tmp_path) -> GtImports:
    script = tmp_path / "fake_import.py"
    script.write_text(STAND_IN)
    return GtImports(root=tmp_path / "gt-import", python=sys.executable, script=script)


def wait_done(imports: GtImports, job_id: str, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = imports.status(job_id)
        if status["status"] != "running":
            return status
        time.sleep(0.1)
    raise AssertionError(f"{job_id} still running")


def test_an_import_runs_as_a_job_and_reports_its_counts(imports):
    started = imports.start("charlottenburger-amtsschrifttum", dry_run=True)
    assert started["status"] in ("running", "completed")
    done = wait_done(imports, started["id"])
    assert done["status"] == "completed" and done["exit"] == 0
    assert any("verworfen: 15 Dubletten" in line for line in done["report"])
    assert any("args: --dry-run" in line for line in done["log_tail"])


def test_a_refusal_of_the_script_is_reported_as_one(imports, monkeypatch):
    monkeypatch.setenv("GT_FAKE_EXIT", "3")
    done = wait_done(imports, imports.start("Weisthuemer", dry_run=False)["id"])
    assert done["status"] == "refused" and done["refusal"] == "öffentlich verweigert"


def test_one_import_per_source_at_a_time(imports, monkeypatch):
    monkeypatch.setenv("GT_FAKE_SLEEP", "3")
    first = imports.start("Weisthuemer")
    with pytest.raises(GtImportError, match="already being imported"):
        imports.start("Weisthuemer")
    imports.start("DTGT")                              # another source is fine
    wait_done(imports, first["id"])


@pytest.mark.parametrize("job_id", ["../x", "x", "20261113T000000Z-../../etc"])
def test_a_job_id_that_is_not_one_is_refused(imports, job_id):
    with pytest.raises(GtImportError, match="not an import job id"):
        imports.status(job_id)


# ── the routes ───────────────────────────────────────────────────────────────

LOOPBACK = ("127.0.0.1", 50000)


@pytest.fixture
def client(tmp_path, trainer_key, monkeypatch):
    settings = TrainerSettings(jobs_root=tmp_path / "training", trained_root=tmp_path / "t",
                               checkpoint_root=tmp_path / "c", min_free_disk_gb=0.0,
                               gt_import_root=tmp_path / "gt-import")
    app = app_module.app
    app.state.settings = settings
    app.state.store = JobStore(settings.jobs_root, host_id=settings.host_id)
    with TestClient(app, client=LOOPBACK, headers={"X-API-Key": trainer_key}) as c:
        yield c
    for attr in ("settings", "store"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


def test_sources_lists_the_register(client):
    rows = client.get("/gt/sources", params={"check_hf": False}).json()["sources"]
    ids = {r["id"] for r in rows}
    assert {"reichsanzeiger-gt", "gt-fraktur", "reichenau-inkunabeln"} <= ids
    reich = next(r for r in rows if r["id"] == "reichenau-inkunabeln")
    assert reich["kind"] == "zenodo" and reich["doi"] == "10.5281/zenodo.11046062"
    assert all(r["licence"] and r["licence_at"] for r in rows)


def test_sources_says_unknown_when_hf_does_not_answer(client, monkeypatch):
    monkeypatch.setattr(app_module, "_uploaded", lambda targets: {t: None for t in targets})
    rows = client.get("/gt/sources").json()["sources"]
    assert all(r["uploaded"] is None for r in rows)


def test_import_refuses_an_unknown_id_with_404(client):
    resp = client.post("/gt/import", json={"id": "github.com/x/y"})
    assert resp.status_code == 404 and "reichsanzeiger-gt" in resp.json()["detail"]


def test_import_has_no_public_field(client):
    resp = client.post("/gt/import", json={"id": "Weisthuemer", "public": True})
    assert resp.status_code == 422


def test_import_starts_a_dry_run_by_default(client, monkeypatch, tmp_path):
    script = tmp_path / "fake_import.py"
    script.write_text(STAND_IN)
    monkeypatch.setattr(gt_import, "SCRIPT", script)
    resp = client.post("/gt/import", json={"id": "Weisthuemer"})
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["dry_run"] is True
    deadline = time.monotonic() + 20
    while client.get(f"/gt/import/{job['id']}").json()["status"] == "running":
        assert time.monotonic() < deadline
        time.sleep(0.1)
    done = client.get(f"/gt/import/{job['id']}").json()
    assert done["status"] == "completed"
    assert any("args: --dry-run" in line for line in done["log_tail"])   # the stand-in ran


def test_survey_refuses_another_host_with_422(client):
    resp = client.post("/gt/survey", json={"url": "https://intranet.example/x"})
    assert resp.status_code == 422


def test_survey_route_hands_back_the_survey(client, monkeypatch):
    monkeypatch.setattr(app_module, "gt_survey",
                        lambda url: survey(url, github()))
    out = client.post("/gt/survey", json={"url": "https://github.com/UB-Mannheim/toy-gt"}).json()
    assert out["licence"]["state"] == "agree" and out["kind"] == "github"


def test_an_unknown_import_job_is_404(client):
    assert client.get("/gt/import/20261113T000000Z-nothing").status_code == 404

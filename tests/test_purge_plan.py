"""#387: aufräumen als Plan, ausführen als Slurm-Job, nach erneuter Prüfung.

``ubelix/plan_purge.py`` schreibt die Löschliste und lässt weg, was nie weg
darf: den ``source_job`` eines Held-out-Sets, den Erbauer eines übernommenen
Artefakts, einen Job hinter einem registrierten ``local_path``. ``completed``
kommt nur auf ausdrücklichen Wunsch hinein. ``ubelix/apply_purge.sh`` reicht
sich selbst ein und löscht im Job nur, was sich seit dem Plan nicht geändert
hat. Alles offline: ein Jobs-Root in tmp, ``sbatch`` gefälscht.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ubelix" / "plan_purge.py"
_spec = importlib.util.spec_from_file_location("plan_purge", SCRIPT)
purge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(purge)

POOL = "20260905T190759Z-kraken-german-eval-pool-v1"
BUILDER = "20261001T093909Z-qwen3vl-builder"
ADOPTER = "20261019T180118Z-olmocr-arm"
REGISTERED = "20260920T100000Z-kraken-registered"


def make_job(root: Path, job_id: str, status: str, *, finished: str = "2026-09-01T10:00:00+00:00",
             artefact: str | None = None, data: bool = True) -> Path:
    directory = root / job_id
    (directory / "logs").mkdir(parents=True)
    (directory / "logs" / "runner.log").write_text("log\n")
    (directory / "energy.json").write_text("{}")
    if data:
        (directory / "data" / "pages").mkdir(parents=True)
        (directory / "data" / "pages" / "p.xml").write_text("x" * 100)
    (directory / "job.json").write_text(json.dumps({
        "id": job_id, "status": status, "finished_at": finished,
        "progress": {"artefact": artefact}}))
    return directory


@pytest.fixture
def world(tmp_path):
    storage = tmp_path / "storage"
    ours = storage / "Textrecognition_Training"
    jobs = ours / "training_folder" / "jobs"
    jobs.mkdir(parents=True)
    trained = ours / "registry" / "trained"
    trained.mkdir(parents=True)
    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps({"sets": [
        {"name": "german-medieval-v1", "source_job": POOL, "test_documents": []}]}))
    (trained / "kraken.yaml").write_text(
        "id: kraken-registered\n"
        f"local_path: '{jobs}/{REGISTERED}/data/model_best.mlmodel'  # the trained weights\n")

    make_job(jobs, "20260807T153335Z-kraken-thun-missiven-v1", "failed")
    make_job(jobs, "20260807T200230Z-kraken-medieval-scripts-v1", "cancelled",
             finished="2026-08-08T00:00:00+00:00")
    make_job(jobs, POOL, "failed")
    make_job(jobs, BUILDER, "failed", artefact="2ad500ee8414 built by this job")
    make_job(jobs, ADOPTER, "completed",
             artefact=f"2ad500ee8414 reused, built by {BUILDER}")
    make_job(jobs, REGISTERED, "failed")
    make_job(jobs, "20261020T000000Z-done", "completed")
    make_job(jobs, "20261105T000000Z-running", "training")
    make_job(jobs, "20260806T000000Z-no-data", "failed", data=False)

    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({
        "root": str(storage), "measured": 1793000000,
        "jobs_data": {"20260807T153335Z-kraken-thun-missiven-v1": 40_000_000_000}}))
    return {"jobs": jobs, "heldout": heldout, "registry": ours / "registry",
            "inventory": inventory, "plans": tmp_path / "plans", "tmp": tmp_path}


def make_plan(world, **over):
    args = dict(heldout_registry=world["heldout"], registry_root=world["registry"],
                inventory=world["inventory"])
    args.update(over)
    return purge.plan(world["jobs"], **args)


def ids(record):
    return [e["id"] for e in record["entries"]]


def why(record, job_id):
    return " ".join(next(p["why"] for p in record["protected"] if p["id"] == job_id))


# ── der Plan ─────────────────────────────────────────────────────────────────

def test_protected_jobs_never_enter_the_plan_and_say_why(world):
    record = make_plan(world)
    assert ids(record) == ["20260807T153335Z-kraken-thun-missiven-v1",
                           "20260807T200230Z-kraken-medieval-scripts-v1"]
    assert {p["id"] for p in record["protected"]} == {POOL, BUILDER, REGISTERED}
    assert "source_job von german-medieval-v1" in why(record, POOL)
    assert f"Erbauer des Artefakts, das {ADOPTER} übernahm" in why(record, BUILDER)
    assert "local_path in registry/trained/kraken.yaml" in why(record, REGISTERED)


def test_completed_only_when_asked_and_running_never(world):
    assert "20261020T000000Z-done" not in ids(make_plan(world))
    asked = make_plan(world, status=("completed",))
    assert ids(asked) == [ADOPTER, "20261020T000000Z-done"]
    with pytest.raises(purge.PlanError, match="training"):
        make_plan(world, status=("failed", "training"))


def test_a_completed_adopter_in_the_plan_frees_its_builder(world):
    """Fällt der Übernehmer selbst, braucht niemand mehr das Manifest des Erbauers."""
    record = make_plan(world, status=("failed", "completed"))
    assert BUILDER in ids(record) and ADOPTER in ids(record)


def test_a_protected_adopter_keeps_its_builder(world):
    """Fixpunkt: der Übernehmer ist selbst geschützt, also bleibt auch sein Erbauer."""
    raw = json.loads(world["heldout"].read_text())
    raw["sets"].append({"name": "other-v1", "source_job": ADOPTER})
    world["heldout"].write_text(json.dumps(raw))
    record = make_plan(world, status=("failed", "completed"))
    assert ADOPTER not in ids(record) and BUILDER not in ids(record)


def test_ended_before_filters_by_date(world):
    record = make_plan(world, ended_before="2026-08-05")
    assert ids(record) == []
    record = make_plan(world, ended_before="2026-08-31")
    assert ids(record) == ["20260807T200230Z-kraken-medieval-scripts-v1"]


def test_sizes_come_from_the_inventory_not_from_du(world):
    record = make_plan(world)
    sizes = {e["id"]: e["bytes"] for e in record["entries"]}
    assert sizes == {"20260807T153335Z-kraken-thun-missiven-v1": 40_000_000_000,
                     "20260807T200230Z-kraken-medieval-scripts-v1": None}
    assert record["total_bytes"] == 40_000_000_000 and record["unsized"] == 1
    assert record["sources"]["inventory_measured"].startswith("2026-10-26")


def test_data_pointing_into_eval_sets_is_protected(world):
    target = world["tmp"] / "eval_sets" / "toy"
    target.mkdir(parents=True)
    job = world["jobs"] / "20260807T153335Z-kraken-thun-missiven-v1"
    shutil.rmtree(job / "data")
    (job / "data").symlink_to(target)
    record = make_plan(world)
    assert "eval_sets/" in why(record, job.name)


def test_without_the_registry_there_is_no_plan(world):
    with pytest.raises(purge.PlanError, match="trained"):
        make_plan(world, registry_root=world["tmp"] / "nowhere")


def test_write_and_verify(world, capsys):
    common = ["--jobs-root", str(world["jobs"]), "--heldout-registry", str(world["heldout"]),
              "--registry-root", str(world["registry"]), "--inventory", str(world["inventory"]),
              "--plans", str(world["plans"])]
    assert purge.main(common + ["--write"]) == 0
    out = capsys.readouterr().out
    written = next(world["plans"].glob("purge-*Z.json"))
    assert f"ubelix/apply_purge.sh {written}" in out and "3 geschützt" in out
    assert purge.main(["--verify", str(written)]) == 0
    assert capsys.readouterr().out.startswith(hashlib.sha256(written.read_bytes()).hexdigest())

    raw = json.loads(written.read_text())
    raw["entries"].append({"id": POOL, "status": "failed", "ended": None,
                           "path": str(world["jobs"] / POOL / "data"), "bytes": None})
    written.write_text(json.dumps(raw))
    assert purge.main(["--verify", str(written)]) == 1
    assert "von Hand geändert" in capsys.readouterr().err


def test_the_script_runs_on_the_login_nodes_python():
    ast.parse(SCRIPT.read_text(encoding="utf-8"), feature_version=(3, 9))


# ── die Ausführung ───────────────────────────────────────────────────────────

@pytest.fixture
def written(world):
    path = purge.write_plan(make_plan(world), world["plans"])
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def run_apply(path, sha):
    out = io.StringIO()
    counts = purge.apply(path, sha, out=out)
    return counts, out.getvalue()


def test_apply_removes_data_and_keeps_the_record(world, written):
    counts, log = run_apply(*written)
    assert counts["removed"] == 2 and counts["skip"] == 0
    for job_id in ("20260807T153335Z-kraken-thun-missiven-v1",
                   "20260807T200230Z-kraken-medieval-scripts-v1"):
        job = world["jobs"] / job_id
        assert not (job / "data").exists()
        assert (job / "job.json").is_file() and (job / "logs" / "runner.log").is_file()
        assert (job / "energy.json").is_file()
        assert f"removed  {job_id}" in log
    assert (world["jobs"] / POOL / "data").is_dir()


def test_a_changed_plan_file_skips_everything(world, written):
    path, sha = written
    path.write_text(path.read_text() + " ")
    counts, log = run_apply(path, sha)
    assert counts == {"removed": 0, "skip": 2, "bytes": 0}
    assert "nichts gelöscht" in log
    assert (world["jobs"] / "20260807T153335Z-kraken-thun-missiven-v1" / "data").is_dir()


def test_a_changed_status_skips_that_entry(world, written):
    job = world["jobs"] / "20260807T153335Z-kraken-thun-missiven-v1"
    raw = json.loads((job / "job.json").read_text())
    raw["status"] = "queued"     # requeued since the plan was written
    (job / "job.json").write_text(json.dumps(raw))
    counts, log = run_apply(*written)
    assert counts["removed"] == 1 and counts["skip"] == 1
    assert "Status jetzt queued, im Plan failed" in log
    assert (job / "data").is_dir()


def test_a_job_protected_since_the_plan_is_skipped(world, written):
    raw = json.loads(world["heldout"].read_text())
    raw["sets"].append({"name": "new-v1",
                        "source_job": "20260807T200230Z-kraken-medieval-scripts-v1"})
    world["heldout"].write_text(json.dumps(raw))
    counts, log = run_apply(*written)
    assert counts["removed"] == 1 and "geschützt: source_job von new-v1" in log


# ── apply_purge.sh ───────────────────────────────────────────────────────────

@pytest.fixture
def fake_sbatch(tmp_path):
    """Ein ``sbatch``, das seine Argumente notiert und das Skript als Job laufen lässt."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "sbatch.args"
    (bin_dir / "sbatch").write_text(
        "#!/bin/bash\n"
        f'printf "%s\\n" "$@" > {record}\n'
        'script="${@: -1}"\n'
        'for a in "$@"; do case "$a" in --export=*) e="${a#--export=ALL,}";; esac; done\n'
        'IFS=, read -ra kv <<< "$e"; for x in "${kv[@]}"; do export "$x"; done\n'
        'SLURM_JOB_ID=4242 bash "$script"\n')
    (bin_dir / "sbatch").chmod(0o755)
    return bin_dir, record


def test_apply_purge_sh_submits_itself_and_the_job_deletes(world, written, fake_sbatch, tmp_path):
    path, _ = written
    bin_dir, record = fake_sbatch
    # A repo without git: pin_code runs the checkout as it is, quietly.
    repo = tmp_path / "repo"
    (repo / "ubelix").mkdir(parents=True)
    for name in ("apply_purge.sh", "plan_purge.py", "pin_code.sh"):
        shutil.copy(ROOT / "ubelix" / name, repo / "ubelix" / name)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(tmp_path),
           "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    env.pop("SLURM_JOB_ID", None)
    env.pop("ATR_CODE_COMMIT", None)
    done = subprocess.run(["bash", str(repo / "ubelix" / "apply_purge.sh"), str(path)],
                          env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    args = record.read_text().splitlines()
    assert any(a.startswith("--export=ALL,PURGE_PLAN=") for a in args)
    assert "== df vorher" in done.stdout and "== df nachher" in done.stdout
    assert "2 gelöscht" in done.stdout
    assert not (world["jobs"] / "20260807T153335Z-kraken-thun-missiven-v1" / "data").exists()


def test_apply_purge_sh_refuses_a_tampered_plan_before_sbatch(world, written, fake_sbatch):
    path, _ = written
    bin_dir, record = fake_sbatch
    raw = json.loads(path.read_text())
    raw["entries"][0]["id"] = POOL
    path.write_text(json.dumps(raw))
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    env.pop("SLURM_JOB_ID", None)
    done = subprocess.run(["bash", str(ROOT / "ubelix" / "apply_purge.sh"), str(path)],
                          env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 1 and not record.exists()


def test_the_headers_ask_for_nodes_with_mem():
    """`You must request --nodes with --mem!` (06.11.2026)."""
    headers = [line for line in (ROOT / "ubelix" / "apply_purge.sh").read_text().splitlines()
               if line.startswith("#SBATCH")]
    assert "#SBATCH --nodes=1" in headers and "#SBATCH --mem=4G" in headers
    assert "#SBATCH --partition=epyc2" in headers and "#SBATCH --qos=job_gratis" in headers


# ── #204 als Plan ────────────────────────────────────────────────────────────

PURGE_204 = """
20260808T183111Z-kraken-medieval-full-v1 20260825T092439Z-qwen3vl-medieval-german-v2
20260822T143617Z-qwen3vl-medieval-german-v2 20260915T191232Z-qwen3vl-german-pages-v4
20260915T063213Z-qwen3vl-german-pages-v4 20260916T115939Z-qwen3vl-german-pages-v5-asteraix
20260916T041001Z-qwen3vl-german-pages-v5 20260908T101611Z-qwen3vl-german-pages-v1
20260909T190659Z-qwen3vl-german-pages-v2 20260824T093013Z-kraken-medieval-german-v2
20260824T142549Z-kraken-medieval-german-v2 20260825T043258Z-kraken-medieval-german-v2
20260825T092433Z-kraken-medieval-german-v2 20260825T145122Z-kraken-medieval-german-v2
20260824T193523Z-kraken-medieval-german-v2 20260824T193529Z-qwen3vl-medieval-german-v2
20260822T143612Z-kraken-medieval-german-v2 20260824T093019Z-qwen3vl-medieval-german-v2
20260824T142555Z-qwen3vl-medieval-german-v2 20260825T043304Z-qwen3vl-medieval-german-v2
20260915T053651Z-qwen3vl-german-pages-v4 20260822T141526Z-kraken-medieval-german-v1
20260807T201321Z-kraken-medieval-scripts-v1 20260824T092437Z-kraken-medieval-german-v2
20260808T182250Z-kraken-medieval-full-v1 20260814T192904Z-qwen3vl-german-medieval-v1
20260807T154201Z-kraken-thun-missiven-v1 20260807T200230Z-kraken-medieval-scripts-v1
20260807T161137Z-kraken-thun-missiven-v1 20260810T105206Z-kraken-thun-finetune-v1
20260807T153553Z-kraken-thun-missiven-v1 20260910T121127Z-trocr-thun-smoke-v2
20260908T104421Z-trocr-thun-smoke-v1 20260916T115221Z-trocr-thun-smoke-asteraix-v1
20260808T085107Z-kraken-medieval-full-v1 20260807T153335Z-kraken-thun-missiven-v1
20260808T183124Z-kraken-medieval-full-v1 20260822T142222Z-qwen3vl-medieval-german-v1
20260824T092444Z-qwen3vl-medieval-german-v2
""".split()


def test_the_204_purge_is_reproducible_as_a_plan(tmp_path):
    """39 failed/cancelled Jobs vor dem 20.09., dazu der Pool von german-medieval-v1
    (failed wie sie) — der Plan nimmt die 39 und schützt den Pool, mit der echten
    Registry des Repos."""
    assert len(PURGE_204) == 39 and len(set(PURGE_204)) == 39
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    for n, job_id in enumerate(PURGE_204):
        make_job(jobs, job_id, "failed" if n % 3 else "cancelled",
                 finished=f"2026-{job_id[4:6]}-{job_id[6:8]}T12:00:00+00:00")
    make_job(jobs, POOL, "failed", finished="2026-09-05T20:00:00+00:00")
    make_job(jobs, "20261029T133027Z-qwen3.5-kurrent-v2", "completed")
    (tmp_path / "registry" / "trained").mkdir(parents=True)
    record = purge.plan(jobs, ended_before="2026-09-20", registry_root=tmp_path / "registry",
                        inventory=tmp_path / "none.json")
    assert sorted(ids(record)) == sorted(PURGE_204)
    assert [p["id"] for p in record["protected"]] == [POOL]
    assert "german-medieval-v1" in why(record, POOL)


def test_the_inventory_measures_data_per_job(world):
    """``jobs_data`` je Job, und die Gesamtgrösse von ``jobs`` zählt nichts doppelt."""
    spec = importlib.util.spec_from_file_location("research_storage",
                                                  ROOT / "scripts" / "research_storage.py")
    storage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(storage)
    record = storage.inventory(world["jobs"].parents[2], with_top=False)
    whole = int(subprocess.run(["du", "-sb", str(world["jobs"])], capture_output=True,
                               text=True, check=True).stdout.split("\t")[0])
    jobs_dir_itself = int(subprocess.run(["du", "-sb", "-S", str(world["jobs"])],
                                         capture_output=True, text=True,
                                         check=True).stdout.split("\t")[0])
    assert "20260806T000000Z-no-data" not in record["jobs_data"]
    assert record["jobs_data"][POOL] >= 100
    # du of the job directories' own inodes is not in the per-child sum; nothing else differs.
    assert 0 <= whole - record["ours"]["training_folder/jobs"] <= jobs_dir_itself + 9 * 4096

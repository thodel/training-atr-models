"""A prepare must not book the wall of a build it is not going to do (#212).

`ubelix/prepare.sbatch` asked for 20 h on 8 CPUs. That is 9,600 of the 11,520
CPU-minutes `job_gratis` allows — counted as cpus x *remaining* walltime over
every RUNNING job of the user, GPU jobs included — so the prepare went PENDING on
`MaxCpuRunMinsPerUser` behind any other run of ours, and was freed by hand three
times: the four olmOCR page preps on 09.10.2026, which then took 7 s each, and
job 17838197 on 10.10.2026.

Two things are pinned here. The **default** has to leave room beside a running
job while still covering a from-scratch build (the longest was 11:15:04, job
17795292). And `submit.sh` has to **tell the two cases apart**: a prepare whose
corpus is already compiled only adopts it, and every uncertainty has to fall to
the long wall, because a short wall on a real build is a TIMEOUT.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from atr_training.artefact_cache import UNPINNED_MAX_AGE_DAYS, ArtefactCache, key_for
from atr_training.contracts import DatasetSpec
from atr_training.settings import TrainerSettings

ROOT = Path(__file__).resolve().parents[1]
UBELIX = ROOT / "ubelix"
PREPARE = UBELIX / "prepare.sbatch"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"ubelix_{name}", UBELIX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pf = _load("preflight")
probe = _load("artefact_probe")

#: What the batch file declares, as submit.sh reads it.
DECLARED = {
    key: value
    for key, value in re.findall(r"^#(ATR_TIME_\w+)=(\S+)$", PREPARE.read_text(encoding="utf-8"),
                                 re.M)
}
CPUS = int(pf.resources(PREPARE.read_text(encoding="utf-8"), [])["cpus"])
CAP = pf.CPU_MINUTE_CAPS["job_gratis"]
#: The longest from-scratch prepare observed here: job 17795292, 11:15:04.
LONGEST_BUILD_MINUTES = 11 * 60 + 16


# ── the default ─────────────────────────────────────────────────────────────
def test_the_default_leaves_room_for_a_job_running_beside_it():
    """Inside the cap is not enough — the cap is shared with whatever else runs.

    20 h x 8 left 1,920 CPU-minutes, less than one GPU stage needs, so the
    prepare queued behind every other run. Half the cap is the line: it is the
    most that still admits a second job of the same size.
    """
    minutes = pf.parse_minutes(pf.resources(PREPARE.read_text(encoding="utf-8"), [])["time"])
    assert CPUS * minutes <= CAP // 2, (
        f"{CPUS} cpus x {minutes} min = {CPUS * minutes:,} CPU-minutes of a "
        f"{CAP:,} cap leaves only {CAP - CPUS * minutes:,} for anything else"
    )


def test_the_default_still_covers_a_from_scratch_build():
    minutes = pf.parse_minutes(pf.resources(PREPARE.read_text(encoding="utf-8"), [])["time"])
    assert minutes >= LONGEST_BUILD_MINUTES, (
        f"--time is {minutes} min; job 17795292 needed {LONGEST_BUILD_MINUTES}"
    )


def test_the_file_declares_both_walltimes_for_submit_sh():
    assert set(DECLARED) == {"ATR_TIME_CACHED", "ATR_TIME_BUILD"}


def test_the_declared_build_walltime_is_the_sbatch_default():
    """A plain `sbatch`, which reads no declaration, must get the safe one."""
    directive = pf.resources(PREPARE.read_text(encoding="utf-8"), [])["time"]
    assert DECLARED["ATR_TIME_BUILD"] == directive


def test_the_cached_walltime_is_short_but_not_zero():
    cached = pf.parse_minutes(DECLARED["ATR_TIME_CACHED"])
    assert 0 < cached < pf.parse_minutes(DECLARED["ATR_TIME_BUILD"])
    assert CPUS * cached <= CAP // 2


# ── the probe reads the cache the job will read ─────────────────────────────
def test_the_probes_cache_root_is_the_one_prepare_exports():
    """Declared twice — the probe runs before the batch file's `export` — so the
    two are pinned to each other rather than left to drift."""
    line = next(ln for ln in PREPARE.read_text(encoding="utf-8").splitlines()
                if ln.startswith("export ATR_TRAIN_ARTEFACT_CACHE_ROOT="))
    exported = line.split("=", 1)[1].replace("$S", "/scratch/network/users/$USER")
    assert exported == str(probe.DEFAULT_CACHE_ROOT).replace(os.environ.get("USER") or "", "$USER")


def test_the_env_wins_over_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv("ATR_TRAIN_ARTEFACT_CACHE_ROOT", str(tmp_path))
    assert probe.cache_root() == tmp_path


# ── the verdict ─────────────────────────────────────────────────────────────
def _spec_file(tmp_path: Path, **kw) -> Path:
    body = {
        "engine": "kraken",
        "model_id": "probe-test",
        "datasets": [{"hf_repo": "dh-unibe/medieval", "train_projects": ["a", "b"], **kw}],
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _dataset(**kw) -> DatasetSpec:
    return DatasetSpec(hf_repo="dh-unibe/medieval", train_projects=["a", "b"], **kw)


@pytest.fixture
def cache(tmp_path):
    return ArtefactCache(tmp_path / "artefacts")


def _store(cache: ArtefactCache, tmp_path: Path, **kw):
    """Put an entry under the key the probe must compute for `_spec_file(**kw)`.

    Built here from `key_for` and the settings directly, not from the probe: a
    hit then proves that the probe reached the backend's own key, which is the
    one claim the probe makes.
    """
    source = tmp_path / "built"
    source.mkdir(exist_ok=True)
    (source / "train.arrow").write_bytes(b"x" * 1024)
    key = key_for(_dataset(**kw), "kraken",
                  extra={"chunk_pages": TrainerSettings().chunk_pages})
    cache.put(key, source)
    return key


def test_the_vlm_backends_key_is_reached_too(tmp_path, cache):
    """The engine that actually runs the preps, resolved by name.

    Weaker than the kraken case above — the expected key is built from the same
    `Pipeline` the probe reaches — but it is what shows the resolution works at
    all for `vllm`, whose runner module and whose three compile options in the
    key are not kraken's.
    """
    from vlm_train_svc.runner import Pipeline

    from atr_training.contracts import TrainJob, TrainRequest
    from atr_training.jobstore import JobStore

    body = {"engine": "vllm", "model_id": "probe-vlm",
            "datasets": [{"hf_repo": "dh-unibe/medieval", "train_projects": ["a"]}]}
    path = tmp_path / "vlm.json"
    path.write_text(json.dumps(body), encoding="utf-8")

    settings = TrainerSettings()
    request = TrainRequest.model_validate(body)
    expected = Pipeline(JobStore(tmp_path / "jobs"), settings)._cache_key(
        TrainJob(id="x", request=request))
    source = tmp_path / "vlm-built"
    source.mkdir()
    (source / "train.jsonl").write_text("{}\n", encoding="utf-8")
    cache.put(expected, source)

    answer, why = probe.verdict(str(path), root=cache.root, settings=settings)
    assert answer == "cached", why


def _age(cache: ArtefactCache, key, days: float) -> None:
    manifest = cache.root / key.digest / cache.MANIFEST
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["built_at"] = time.time() - days * 86400
    data["last_used"] = data["built_at"]
    manifest.write_text(json.dumps(data), encoding="utf-8")


def test_an_empty_cache_is_a_build(tmp_path, cache):
    answer, why = probe.verdict(str(_spec_file(tmp_path)), root=cache.root)
    assert answer == "build"
    assert "not in" in why


def test_a_fresh_entry_is_adopted(tmp_path, cache):
    _store(cache, tmp_path)
    answer, why = probe.verdict(str(_spec_file(tmp_path)), root=cache.root)
    assert answer == "cached", why
    assert "adopts" in why


def test_a_pinned_entry_is_adopted_however_old_it_is(tmp_path, cache):
    key = _store(cache, tmp_path, revision="a" * 40)
    _age(cache, key, 400)
    answer, why = probe.verdict(str(_spec_file(tmp_path, revision="a" * 40)), root=cache.root)
    assert answer == "cached", why


def test_an_expired_entry_is_a_build(tmp_path, cache):
    key = _store(cache, tmp_path)
    _age(cache, key, UNPINNED_MAX_AGE_DAYS + 1)
    answer, why = probe.verdict(str(_spec_file(tmp_path)), root=cache.root)
    assert answer == "build"
    assert "rebuilds" in why


def test_an_entry_that_could_expire_during_the_queue_wait_is_a_build(tmp_path, cache):
    """It is usable now and the job adopts it later. A prepare can wait hours, and
    an entry that crosses the deadline in between rebuilds — under a wall chosen
    for a job that would not have to."""
    key = _store(cache, tmp_path)
    _age(cache, key, UNPINNED_MAX_AGE_DAYS - probe.MARGIN_DAYS / 2)
    assert cache.entry(key).usable(UNPINNED_MAX_AGE_DAYS)[0]
    answer, why = probe.verdict(str(_spec_file(tmp_path)), root=cache.root)
    assert answer == "build"
    assert "expire" in why


def test_a_key_that_differs_in_the_selection_does_not_hit(tmp_path, cache):
    _store(cache, tmp_path)
    other = tmp_path / "other.json"
    other.write_text(json.dumps({
        "engine": "kraken", "model_id": "probe-test",
        "datasets": [{"hf_repo": "dh-unibe/medieval", "train_projects": ["a"]}],
    }), encoding="utf-8")
    assert probe.verdict(str(other), root=cache.root)[0] == "build"


def test_the_probe_does_not_stamp_last_used(tmp_path, cache):
    """`last_used` is eviction's only evidence that a run still needs an entry.
    A submission that never starts must not forge it."""
    key = _store(cache, tmp_path)
    before = json.loads((cache.root / key.digest / cache.MANIFEST)
                        .read_text(encoding="utf-8"))["last_used"]
    probe.verdict(str(_spec_file(tmp_path)), root=cache.root)
    after = json.loads((cache.root / key.digest / cache.MANIFEST)
                       .read_text(encoding="utf-8"))["last_used"]
    assert after == before


def test_the_cli_prints_one_parsable_line(tmp_path, cache, monkeypatch, capsys):
    monkeypatch.setenv("ATR_TRAIN_ARTEFACT_CACHE_ROOT", str(cache.root))
    _store(cache, tmp_path)
    assert probe.main([str(_spec_file(tmp_path))]) == 0
    assert capsys.readouterr().out.strip() == "ARTEFACT=cached"


# ── submit.sh acts on the answer ────────────────────────────────────────────
JOB = """#!/bin/bash
#SBATCH --job-name=prep
#SBATCH --qos=job_gratis
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#ATR_TIME_CACHED=00:30:00
#ATR_TIME_BUILD=12:00:00
echo hi
"""


@pytest.fixture
def submission(tmp_path):
    """A clean pushed checkout, plus stand-ins for sbatch and apptainer.

    `apptainer` answers both calls submit.sh makes with a spec: the validator,
    which only has to succeed, and the probe, whose verdict is whatever
    `$PROBE_ANSWER` says. Nothing here runs a container.
    """
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    repo = tmp_path / "training-atr-models"
    subprocess.run(["git", "clone", "-q", str(origin), str(repo)], check=True,
                   capture_output=True)
    for key, value in (("user.email", "t@example.org"), ("user.name", "t")):
        subprocess.run(["git", "-C", str(repo), "config", key, value], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "-b", "main"], check=True)
    shutil.copytree(UBELIX, repo / "ubelix", ignore=shutil.ignore_patterns("__pycache__"))
    (repo / "ubelix" / "prep.sbatch").write_text(JOB)
    subprocess.run(["git", "-C", str(repo), "add", "ubelix"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "ubelix"], check=True)
    subprocess.run(["git", "-C", str(repo), "push", "-q", "origin", "main"], check=True)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sbatch").write_text('#!/bin/bash\necho "ARGS=$*"\n')
    (bin_dir / "apptainer").write_text(
        '#!/bin/bash\n'
        'case "$*" in\n'
        '  *artefact_probe.py*) echo "ARTEFACT=${PROBE_ANSWER:-build}" ;;\n'
        '  *) exit 0 ;;\n'
        'esac\n')
    for name in ("sbatch", "apptainer"):
        (bin_dir / name).chmod(0o755)

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "engine": "kraken", "model_id": "x",
        "datasets": [{"hf_repo": "dh-unibe/medieval", "train_projects": ["a"]}],
    }), encoding="utf-8")

    env = {k: v for k, v in os.environ.items()
           if k not in ("ATR_CODE_COMMIT", "ATR_UNPINNED", "PROBE_ANSWER")}
    env.update(PATH=f"{bin_dir}:{env['PATH']}", ATR_TRAIN_REPO=str(repo), HOME=str(tmp_path))
    return repo, spec, env


def _submit(submission, *extra, answer=None):
    repo, spec, env = submission
    if answer:
        env = {**env, "PROBE_ANSWER": answer}
    return subprocess.run(["bash", str(repo / "ubelix" / "submit.sh"),
                           str(repo / "ubelix" / "prep.sbatch"), str(spec), *extra],
                          env=env, capture_output=True, text=True)


def _args(done) -> str:
    return next(ln for ln in done.stdout.splitlines() if ln.startswith("ARGS="))[5:]


def test_a_cached_corpus_gets_the_short_walltime(submission):
    done = _submit(submission, answer="cached")
    assert done.returncode == 0, done.stderr
    assert "--time=00:30:00" in _args(done)


def test_a_corpus_to_build_keeps_the_files_walltime(submission):
    done = _submit(submission, answer="build")
    assert done.returncode == 0, done.stderr
    assert "--time" not in _args(done)


def test_a_probe_that_fails_keeps_the_files_walltime(submission):
    done = _submit(submission, answer="who knows")
    assert done.returncode == 0, done.stderr
    assert "--time" not in _args(done)
    assert "no answer" in done.stdout


def test_a_walltime_given_by_hand_is_never_overridden(submission):
    done = _submit(submission, "--", "--time=02:00:00", answer="cached")
    assert done.returncode == 0, done.stderr
    assert _args(done).count("--time") == 1
    assert "--time=02:00:00" in _args(done)


@pytest.mark.parametrize("opt", ["-t", "--time"])
def test_a_walltime_in_the_two_word_form_is_recognised(submission, opt):
    done = _submit(submission, "--", opt, "02:00:00", answer="cached")
    assert done.returncode == 0, done.stderr
    assert "00:30:00" not in _args(done)


def test_an_older_code_pin_keeps_the_files_walltime(submission):
    """The key folds in the held-out registry, read from this checkout. Pinned
    elsewhere, the probe would be describing a different commit's corpus."""
    repo, _, env = submission
    first = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                           capture_output=True, text=True, check=True).stdout.strip()
    (repo / "later.txt").write_text("later\n")
    subprocess.run(["git", "-C", str(repo), "add", "later.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "later"], check=True)
    subprocess.run(["git", "-C", str(repo), "push", "-q", "origin", "main"], check=True)
    done = _submit((repo, _, {**env, "ATR_CODE_COMMIT": first}), answer="cached")
    assert done.returncode == 0, done.stderr
    assert "--time" not in _args(done)

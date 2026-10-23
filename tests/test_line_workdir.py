"""#183: a line-dataset job's working directory is named after the job, removed
on the way out, and a leftover is matched to sacct's verdict — never to a mtime.

Nine leftovers of killed jobs held 1.1 TB on 08.10.2026, because
``tempfile.TemporaryDirectory`` cleans up on none of TIMEOUT, scancel, OOM or
preemption. Three parts, each run rather than read: the python side (naming,
removal on exit and on a real SIGTERM, delivered to a subprocess), the sbatch's
sourced trap (with its helper run under bash), and the cleaner (with a fake
``sacct`` and ``squeue`` on PATH).
"""

from __future__ import annotations

import importlib.util
import os
import signal
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UBELIX = ROOT / "ubelix"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mld = _load("ubelix_make_line_dataset_183", UBELIX / "make_line_dataset.py")
cleaner = _load("ubelix_clean_line_workdirs", UBELIX / "clean_line_workdirs.py")


# ── the python side ──────────────────────────────────────────────────────────
def test_the_directory_is_named_after_the_job_or_else_the_pid(tmp_path):
    assert mld.workdir_for(tmp_path, "17545199", 4242) == tmp_path / "job-17545199"
    assert mld.workdir_for(tmp_path, " 17545199\n", 4242) == tmp_path / "job-17545199"
    assert mld.workdir_for(tmp_path, None, 4242) == tmp_path / "pid-4242"
    assert mld.workdir_for(tmp_path, "", 4242) == tmp_path / "pid-4242"


def test_the_directory_is_created_fresh_and_removed_on_a_normal_exit(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "17545199")
    with mld.working_directory(tmp_path) as workdir:
        assert workdir == tmp_path / "job-17545199" and workdir.is_dir()
        (workdir / "crops").mkdir()
        (workdir / "crops" / "0000000.jpg").write_bytes(b"x")
    assert not workdir.exists()


def test_the_directory_is_removed_when_the_build_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    with pytest.raises(RuntimeError):
        with mld.working_directory(tmp_path) as workdir:
            (workdir / "partial").write_bytes(b"x")
            raise RuntimeError("materialize died")
    assert not workdir.exists()


def test_a_previous_attempt_of_the_same_job_is_replaced_and_said(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    stale = tmp_path / "job-7"
    stale.mkdir()
    (stale / "old-crop.jpg").write_bytes(b"old")
    from loguru import logger
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(m.record["message"]), level="WARNING")
    try:
        with mld.working_directory(tmp_path) as workdir:
            assert workdir == stale and not (workdir / "old-crop.jpg").exists()
    finally:
        logger.remove(sink)
    assert any("previous attempt" in line and "job-7" in line for line in lines), lines


def test_the_sigterm_handler_is_installed_only_inside_and_restored_after(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "2")
    signal.signal(signal.SIGTERM, signal.SIG_DFL)   # not whatever an earlier test left
    before = signal.getsignal(signal.SIGTERM)
    assert before is signal.SIG_DFL
    with mld.working_directory(tmp_path):
        assert signal.getsignal(signal.SIGTERM) is mld._terminated
        with pytest.raises(SystemExit) as leaving:
            mld._terminated(signal.SIGTERM, None)
        assert leaving.value.code == 128 + signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) is before


def test_a_real_sigterm_removes_the_directory_and_exits_143(tmp_path):
    """Slurm's TERM, delivered to a process that is inside the context: the
    directory goes, the exit status is 143, and nothing is left for a cleaner."""
    script = textwrap.dedent(f"""
        import importlib.util, sys, time
        from pathlib import Path
        spec = importlib.util.spec_from_file_location("mld", {str(UBELIX / "make_line_dataset.py")!r})
        mld = importlib.util.module_from_spec(spec); sys.modules["mld"] = mld
        spec.loader.exec_module(mld)
        with mld.working_directory(Path({str(tmp_path)!r})) as workdir:
            (workdir / "crop.jpg").write_bytes(b"x")
            print("ready", flush=True)
            time.sleep(30)
    """)
    env = {**os.environ, "SLURM_JOB_ID": "99",
           "PYTHONPATH": os.pathsep.join(p for p in [str(ROOT / "src"), os.environ.get("PYTHONPATH", "")] if p)}
    proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env)
    try:
        assert proc.stdout.readline().strip() == "ready", proc.stderr.read()
        assert (tmp_path / "job-99" / "crop.jpg").exists()
        proc.send_signal(signal.SIGTERM)
        code = proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert code == 143, proc.stderr.read()
    assert not (tmp_path / "job-99").exists()


def test_main_builds_in_the_job_directory_under_workdir(tmp_path, monkeypatch, capsys):
    """The seam: ``main`` hands ``build`` a path named after the job, and it is
    gone when ``main`` returns."""
    monkeypatch.setenv("SLURM_JOB_ID", "555")
    seen: list[Path] = []

    class Built:
        lines = 0
        summary = "nothing"
    monkeypatch.setattr(mld, "build", lambda repo, rev, max_pages, workdir, chosen: seen.append(workdir) or Built())
    rc = mld.main(["--repo", "dh-unibe/x", "--revision", "a" * 40, "--out", "dh-unibe/x-lines",
                   "--max-pages", "1", "--dry-run", "--workdir", str(tmp_path)])
    assert rc == 1, "no lines, nothing to publish"
    assert seen == [tmp_path / "job-555"]
    assert not (tmp_path / "job-555").exists()
    assert list(tmp_path.iterdir()) == [], "nothing random-named is left behind either"


# ── the sbatch's trap ────────────────────────────────────────────────────────
HELPER = UBELIX / "line_workdir.sh"


def _bash(script: str, **env) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", f'. "{HELPER}"\n{script}'], capture_output=True,
                          text=True, timeout=30, env={**os.environ, **env})


def test_the_shell_derives_the_same_path_as_python(tmp_path):
    done = _bash(f'line_dataset_workdir "{tmp_path}"', SLURM_JOB_ID="17545199")
    assert done.stdout.strip() == str(mld.workdir_for(tmp_path, "17545199", 1))
    assert _bash(f'line_dataset_workdir "{tmp_path}"', SLURM_JOB_ID="").returncode != 0, \
        "without a job id the shell refuses rather than inventing a name"


def test_the_trap_removes_a_leftover_and_says_so(tmp_path):
    left = tmp_path / "job-17545199" / "crops"
    left.mkdir(parents=True)
    (left / "x.jpg").write_bytes(b"x")
    done = _bash(f'cleanup_line_workdir "{tmp_path / "job-17545199"}"; echo "rc=$?"')
    assert "removed" in done.stdout and "job-17545199" in done.stdout
    assert done.stdout.strip().endswith("rc=0")
    assert not (tmp_path / "job-17545199").exists()


def test_the_trap_is_quiet_about_a_directory_python_already_removed(tmp_path):
    done = _bash(f'cleanup_line_workdir "{tmp_path / "job-1"}"; echo "rc=$?"')
    assert "already gone" in done.stdout and done.stdout.strip().endswith("rc=0")


def test_the_trap_refuses_a_path_that_is_not_a_job_directory(tmp_path):
    other = tmp_path / "hf_datasets"
    other.mkdir()
    (other / "keep").write_bytes(b"x")
    done = _bash(f'cleanup_line_workdir "{other}"; echo "rc=$?"')
    assert "not removing" in done.stdout and done.stdout.strip().endswith("rc=0")
    assert (other / "keep").exists()


def test_the_sbatch_sources_the_helper_signals_term_and_traps_exit():
    text = (UBELIX / "make_line_dataset.sbatch").read_text(encoding="utf-8")
    assert 'source "$REPO/ubelix/line_workdir.sh"' in text
    assert "#SBATCH --signal=TERM@120" in text and "--signal=B:TERM" not in text, \
        "python inside the container must get the TERM too"
    assert "trap 'cleanup_line_workdir \"$WORKDIR\"' EXIT" in text
    assert '--env SLURM_JOB_ID="$SLURM_JOB_ID"' in text, "python names its directory from it"
    assert '--workdir "$WORKDIR_BASE"' in text


def test_the_sbatch_trap_runs_under_bash_and_removes_the_directory(tmp_path):
    """The sbatch's own lines, with apptainer replaced by a stub that leaves a
    directory behind (an OOM kill does exactly that): the EXIT trap removes it."""
    text = (UBELIX / "make_line_dataset.sbatch").read_text(encoding="utf-8")
    start = text.index("WORKDIR_BASE=")
    end = text.index("trap 'cleanup_line_workdir")
    end = text.index("\n", end) + 1
    lines = text[start:end].replace("$REPO/ubelix/line_workdir.sh", str(HELPER))
    script = ("set -euo pipefail\n" + lines +
              'mkdir -p "$WORKDIR/crops"; echo x > "$WORKDIR/crops/0.jpg"\n'
              'false  # the python died, the shell did not\n')
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                          env={**os.environ, "SLURM_JOB_ID": "4321", "ATR_TRAIN_SCRATCH": str(tmp_path)})
    assert done.returncode != 0, "the job still fails"
    assert "removed" in done.stdout and "job-4321" in done.stdout
    assert not (tmp_path / "job-4321").exists()


# ── the cleaner ──────────────────────────────────────────────────────────────
def _fake_slurm(tmp_path: Path, sacct_lines: str, squeue_lines: str = "") -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    calls = tmp_path / "calls.log"
    (bin_dir / "sacct").write_text('#!/usr/bin/env bash\necho "sacct $*" >> "%s"\nprintf "%%b" "%s"\n'
                                   % (calls, sacct_lines.replace("\n", "\\n")), encoding="utf-8")
    (bin_dir / "squeue").write_text('#!/usr/bin/env bash\necho "squeue $*" >> "%s"\nprintf "%%b" "%s"\n'
                                    % (calls, squeue_lines.replace("\n", "\\n")), encoding="utf-8")
    for name in ("sacct", "squeue"):
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IEXEC)
    return bin_dir


def _dirs(base: Path, *names: str) -> None:
    for name in names:
        (base / name / "crops").mkdir(parents=True)
        (base / name / "crops" / "0.jpg").write_bytes(b"x" * 10)


def _entry(name: str, state: str | None, job: str | None = "1") -> cleaner.Entry:
    return cleaner.Entry(path=Path("/scratch") / name, job_id=job, state=state, end=None,
                         size_bytes=1)


@pytest.mark.parametrize("state, verdict", [
    ("COMPLETED", "dead"), ("TIMEOUT", "dead"), ("CANCELLED by 123", "dead"),
    ("OUT_OF_MEMORY", "dead"), ("PREEMPTED", "dead"), ("FAILED", "dead"), ("NODE_FAIL", "dead"),
    ("RUNNING", "alive"), ("PENDING", "alive"), ("REQUEUED", "alive"), ("COMPLETING", "alive"),
    (None, "unknown"), ("", "unknown"), ("SOMETHING_NEW", "unknown"),
])
def test_three_verdicts_never_two(state, verdict):
    assert _entry("job-1", state).verdict == verdict


def test_plan_removes_only_the_dead_by_default():
    found = [_entry("job-1", "TIMEOUT"), _entry("job-2", "RUNNING", "2"),
             _entry("job-3", None, "3"), _entry("line-ds-abc", None, None)]
    actions = {e.path.name: a for e, a, _ in cleaner.plan(found, unattributed=False, queued=[])}
    assert actions == {"job-1": "remove", "job-2": "keep", "job-3": "keep", "line-ds-abc": "keep"}
    why = {e.path.name: w for e, _, w in cleaner.plan(found, unattributed=False, queued=[])}
    assert "unknown is not dead" in why["job-3"] and "unknown is not dead" in why["line-ds-abc"]
    assert "from before #183" in why["line-ds-abc"]


def test_unattributed_directories_go_only_when_no_line_dataset_job_is_queued():
    found = [_entry("job-2", "RUNNING", "2"), _entry("job-3", None, "3"),
             _entry("line-ds-abc", None, None)]
    busy = {e.path.name: a for e, a, _ in cleaner.plan(found, unattributed=True, queued=["77"])}
    assert busy == {"job-2": "keep", "job-3": "keep", "line-ds-abc": "keep"}
    idle = {e.path.name: a for e, a, _ in cleaner.plan(found, unattributed=True, queued=[])}
    assert idle == {"job-2": "keep", "job-3": "remove", "line-ds-abc": "remove"}


def test_sacct_is_asked_once_for_every_job_directory_and_parsed(tmp_path, monkeypatch):
    base = tmp_path / "line-datasets"
    _dirs(base, "job-10", "job-11", "line-ds-zz")
    bin_dir = _fake_slurm(tmp_path, "10|TIMEOUT|2026-10-08T08:52:01\n11|RUNNING|Unknown\n")
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    found = cleaner.entries(base, sizes=False)
    by_name = {e.path.name: e for e in found}
    assert by_name["job-10"].state == "TIMEOUT" and by_name["job-10"].end == "2026-10-08T08:52:01"
    assert by_name["job-11"].verdict == "alive"
    assert by_name["line-ds-zz"].job_id is None and by_name["line-ds-zz"].verdict == "unknown"
    calls = (tmp_path / "calls.log").read_text(encoding="utf-8")
    assert calls.count("sacct") == 1 and "-j 10,11" in calls and "-X -P -n" in calls


def test_the_dry_run_deletes_nothing_and_apply_deletes_only_the_dead(tmp_path, monkeypatch, capsys):
    base = tmp_path / "line-datasets"
    _dirs(base, "job-10", "job-11", "job-12", "line-ds-zz")
    bin_dir = _fake_slurm(tmp_path, "10|TIMEOUT|2026-10-08T08:52:01\n11|RUNNING|Unknown\n"
                                    "12|CANCELLED by 1001|2026-10-08T09:00:00\n")
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    assert cleaner.main(["--base", str(base)]) == 0
    out = capsys.readouterr().out
    assert "would remove 2 directories" in out and "re-run with --apply" in out
    assert all((base / n).exists() for n in ("job-10", "job-11", "job-12", "line-ds-zz"))
    assert "remove job-10" in out and "keep   job-11" in out and "remove job-12" in out
    assert "keep   line-ds-zz" in out

    assert cleaner.main(["--base", str(base), "--apply"]) == 0
    out = capsys.readouterr().out
    assert "removed 2 directories" in out
    assert not (base / "job-10").exists() and not (base / "job-12").exists()
    assert (base / "job-11").exists() and (base / "line-ds-zz").exists()


def test_apply_with_unattributed_asks_squeue_and_keeps_while_a_job_runs(tmp_path, monkeypatch, capsys):
    base = tmp_path / "line-datasets"
    _dirs(base, "line-ds-zz", "pid-4242")
    bin_dir = _fake_slurm(tmp_path, "", "77 line-dataset\n78 train\n")
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("USER", "th19c587")
    assert cleaner.main(["--base", str(base), "--apply", "--unattributed", "--no-size"]) == 0
    out = capsys.readouterr().out
    assert "removed 0" in out and "77" in out and "may be writing" in out
    assert (base / "line-ds-zz").exists() and (base / "pid-4242").exists()
    assert "squeue -u th19c587" in (tmp_path / "calls.log").read_text(encoding="utf-8")

    bin_dir = _fake_slurm(tmp_path / "idle", "", "78 train\n")
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    assert cleaner.main(["--base", str(base), "--apply", "--unattributed", "--no-size"]) == 0
    assert "removed 2" in capsys.readouterr().out
    assert not (base / "line-ds-zz").exists() and not (base / "pid-4242").exists()


def test_sizes_are_measured_with_du_and_summed_for_what_goes(tmp_path, monkeypatch, capsys):
    base = tmp_path / "line-datasets"
    _dirs(base, "job-10")
    bin_dir = _fake_slurm(tmp_path, "10|OUT_OF_MEMORY|2026-10-05T10:00:00\n")
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setattr(cleaner, "du_bytes", lambda path: 380_000_000_000)
    assert cleaner.main(["--base", str(base)]) == 0
    out = capsys.readouterr().out
    assert "380.0 GB" in out and "would remove 1 directory (380.0 GB)" in out


def test_du_is_read_as_bytes_and_a_bad_answer_is_none_not_zero(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    du = bin_dir / "du"
    du.write_text('#!/usr/bin/env bash\nprintf "380000000000\\t%s\\n" "$2"\n', encoding="utf-8")
    du.chmod(du.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    assert cleaner.du_bytes(tmp_path) == 380_000_000_000
    du.write_text('#!/usr/bin/env bash\necho "du: cannot read" >&2\n', encoding="utf-8")
    assert cleaner.du_bytes(tmp_path) is None, "unmeasured is not 0"
    # On GPFS a 300 GB tree takes minutes: the field run of 23.10.2026 read "?"
    # for two of three leftovers at 60 s. du gets its own, longer limit.
    seen: list[float] = []
    monkeypatch.setattr(cleaner, "run", lambda argv, timeout=60: seen.append(timeout) or "7\tx\n")
    assert cleaner.du_bytes(tmp_path) == 7 and seen == [cleaner.DU_TIMEOUT_S] and cleaner.DU_TIMEOUT_S >= 600


def test_a_missing_base_is_nothing_to_clean_not_an_error(tmp_path, capsys):
    assert cleaner.main(["--base", str(tmp_path / "absent")]) == 0
    assert "nothing to clean" in capsys.readouterr().out

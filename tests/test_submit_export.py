"""#41: submit.sh keeps the pin and the spec whatever --export says.

Run, not read: the harness of test_submit_image_check (stub apptainer, sbatch
and python3) with extra sbatch options and an environment.
"""

from __future__ import annotations

import os
import subprocess

from test_submit_image_check import harness  # noqa: F401 — the fixture


def _submit(harness, *extra: str, **env_extra: str) -> subprocess.CompletedProcess:  # noqa: F811
    env = {**os.environ, "HOME": str(harness["home"]), "ATR_TRAIN_REPO": str(harness["repo"]),
           "PATH": f"{harness['bin']}:{os.environ['PATH']}", "ATR_UNPINNED": "1",
           "STUB_KNOWS_CODE": "0", **env_extra}
    env.pop("SIF", None)
    env.pop("SBATCH_EXPORT", None) if "SBATCH_EXPORT" not in env_extra else None
    args = ["bash", str(harness["repo"] / "ubelix" / "submit.sh"),
            str(harness["repo"] / "ubelix" / "train.sbatch"), str(harness["spec"])]
    if extra:
        args += ["--", *extra]
    return subprocess.run(args, capture_output=True, text=True, timeout=60, env=env, cwd=harness["repo"])


def _calls(harness) -> str:  # noqa: F811
    return harness["calls"].read_text(encoding="utf-8") if harness["calls"].exists() else ""


def test_export_all_is_accepted_in_either_spelling(harness):  # noqa: F811
    for spelling in ("--export=ALL,SIF=/x.sif", "--export=all,SIF=/x.sif", "--export=All"):
        done = _submit(harness, spelling)
        assert done.returncode == 0, (spelling, done.stderr)
    assert sum(line.startswith("sbatch ") for line in _calls(harness).splitlines()) == 3


def test_an_export_that_sets_the_pin_or_the_spec_is_refused(harness):  # noqa: F811
    for option in ("--export=ALL,ATR_CODE_COMMIT=main", "--export=ALL,SPEC=/other.json",
                   "--export=ALL,SIF=/x.sif,ATR_CODE_COMMIT=abc"):
        done = _submit(harness, option)
        assert done.returncode == 2, option
        assert "override the pin or the spec" in done.stderr
    assert "sbatch" not in _calls(harness)


def test_an_export_without_all_is_still_refused(harness):  # noqa: F811
    done = _submit(harness, "--export=SIF=/x.sif")
    assert done.returncode == 2 and "use --export=ALL" in done.stderr


def test_sbatch_export_in_the_environment_is_checked(harness):  # noqa: F811
    for value in ("NONE", "SIF,HOME"):
        done = _submit(harness, SBATCH_EXPORT=value)
        assert done.returncode == 2, value
        assert f"SBATCH_EXPORT={value}" in done.stderr
    assert "sbatch" not in _calls(harness)
    assert _submit(harness, SBATCH_EXPORT="ALL").returncode == 0

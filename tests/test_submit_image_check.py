"""#282: submit.sh's image check speaks before it refuses — run, not read.

`KNOWS=$(…); KNOWS_CODE=$?` under `set -euo pipefail` ended submit.sh with
exit 3 and no word when the image did not know the base model: a failing
command substitution in an assignment is a failing command. The structural
test on the source was green the whole time, because the lines were all
there. So this runs submit.sh with a stub `apptainer` (exit 0/2/3), a stub
`sbatch` that records, and a stub `python3` that lets preflight.py through.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

UBELIX = Path(__file__).resolve().parents[1] / "ubelix"


def _executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def harness(tmp_path: Path):
    """A fake $HOME with ~/ubelix/*.sif, a repo copy, and the three stubs."""
    home = tmp_path / "home"
    (home / "ubelix").mkdir(parents=True)
    for sif in ("vlm-train.sif", "vlm-train-tf5.sif"):
        (home / "ubelix" / sif).write_bytes(b"not an image")
    repo = tmp_path / "repo"
    shutil.copytree(UBELIX, repo / "ubelix")
    (repo / "src").mkdir()
    spec = tmp_path / "spec.json"
    spec.write_text('{"model_id": "qwen35-x", "engine": "vllm", "base_model": "Qwen/Qwen3.5-4B"}')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.log"
    # apptainer: the image check answers with $STUB_KNOWS_CODE; everything else
    # (the spec validator, the artefact probe) is a silent success.
    _executable(bin_dir / "apptainer", f'''#!/usr/bin/env bash
echo "apptainer $*" >> "{calls}"
case "$*" in
  *image_knows_model.py*)
    case "${{STUB_KNOWS_CODE:-0}}" in
      0) echo "IMAGE=knows qwen3_5" ;;
      2) echo "IMAGE=unknown could not read the config" ;;
      *) echo "IMAGE=refuses qwen3_5: this image's transformers does not know the architecture" ;;
    esac
    exit "${{STUB_KNOWS_CODE:-0}}" ;;
esac
cat > /dev/null
exit 0
''')
    _executable(bin_dir / "sbatch", f'#!/usr/bin/env bash\necho "sbatch $*" >> "{calls}"\necho "Submitted batch job 1"\n')
    # python3: preflight.py is the real one and wants git state; let it pass.
    _executable(bin_dir / "python3", f'''#!/usr/bin/env bash
case "$1" in *preflight.py) exit 0 ;; esac
exec "{sys.executable}" "$@"
''')
    return {"home": home, "repo": repo, "spec": spec, "bin": bin_dir, "calls": calls}


def _submit(harness, sbatch_name: str, *, knows_code: int) -> subprocess.CompletedProcess:
    env = {**os.environ, "HOME": str(harness["home"]), "ATR_TRAIN_REPO": str(harness["repo"]),
           "PATH": f"{harness['bin']}:{os.environ['PATH']}", "ATR_UNPINNED": "1",
           "STUB_KNOWS_CODE": str(knows_code)}
    env.pop("SIF", None)
    return subprocess.run(["bash", str(harness["repo"] / "ubelix" / "submit.sh"),
                           str(harness["repo"] / "ubelix" / sbatch_name), str(harness["spec"])],
                          capture_output=True, text=True, timeout=60, env=env, cwd=harness["repo"])


def _calls(harness) -> str:
    return harness["calls"].read_text(encoding="utf-8") if harness["calls"].exists() else ""


def test_a_refusal_is_said_and_nothing_is_submitted(harness):
    done = _submit(harness, "train.sbatch", knows_code=3)
    assert done.returncode == 3
    assert "refusing to submit: the image cannot load the base model" in done.stderr
    assert "IMAGE=refuses qwen3_5" in done.stdout, "the image's own line is shown"
    assert "sbatch" not in _calls(harness)


def test_could_not_tell_is_said_and_the_submission_goes_on(harness):
    done = _submit(harness, "train.sbatch", knows_code=2)
    assert done.returncode == 0, done.stderr
    assert "IMAGE=unknown" in done.stdout
    assert "sbatch" in _calls(harness)


def test_a_known_architecture_submits(harness):
    done = _submit(harness, "train.sbatch", knows_code=0)
    assert done.returncode == 0, done.stderr
    assert "IMAGE=knows qwen3_5" in done.stdout and "sbatch" in _calls(harness)


def test_the_check_runs_in_the_image_the_job_would_use(harness):
    _submit(harness, "train.sbatch", knows_code=0)
    line = next(entry for entry in _calls(harness).splitlines() if "image_knows_model.py" in entry)
    assert "vlm-train-tf5.sif" in line, "train.sbatch defaults to tf5 since 10.10."


def test_a_file_that_stops_after_compile_is_not_asked(harness):
    """prepare.sbatch defaults to the transformers-4 image and loads no model:
    refusing it kept a Qwen3.5 corpus from being prepared (#282)."""
    done = _submit(harness, "prepare.sbatch", knows_code=3)
    assert done.returncode == 0, done.stderr
    assert "image check skipped" in done.stdout
    assert "image_knows_model.py" not in _calls(harness)
    assert "sbatch" in _calls(harness)

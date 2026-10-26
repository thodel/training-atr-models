"""kraken on UBELIX: the image, the two batch files, and the batch that fits (#171).

On 07.10.2026 kraken could not train on UBELIX at all — no ketos on PATH, none
in either image, no recipe for one, no compiled arrows. The free cards there are
RTX 4090 with 23'028 MiB, and the only kraken peak measurement says the project
default batch of 256 needs 43'536 MiB. These pin:

* the recipe rebuilds asteraix' venv where the trainer looks for ketos;
* stage 1 compiles on a CPU node in that image and stops after compile;
* stage 2 refuses a batch the card cannot hold before anything trains, and is
  requeue- and preemption-safe the way train_resumable.sbatch is;
* the batch check itself: fits, refused with the largest batch that fits, and
  "not checked" where the table says nothing — never "fits" by default.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from atr_training.contracts import KRAKEN_PLUS_SPEC
from atr_training.settings import TrainerSettings

ROOT = Path(__file__).resolve().parents[1]
UBELIX = ROOT / "ubelix"

_SPEC = importlib.util.spec_from_file_location("kraken_batch_fits", UBELIX / "kraken_batch_fits.py")
fits = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = fits
_SPEC.loader.exec_module(fits)

RTX4090 = 23_028
A40 = 46_068


# ── the batch check ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("batch,card,verdict", [
    (128, RTX4090, "fits"),       # 20'134 + 1'024 <= 23'028
    (64, RTX4090, "fits"),
    (256, RTX4090, "refused"),    # 43'536: the project default does not fit
    (200, RTX4090, "refused"),    # judged by 256, the next measured batch up
    (256, A40, "fits"),           # asteraix runs it, with ~2.5 GB to spare
    (100, RTX4090, "fits"),       # judged by 128
])
def test_the_measured_table_decides(batch, card, verdict):
    assert fits.judge(batch, card, measured_case=True)[0] == verdict


def test_a_refusal_names_the_largest_batch_that_fits():
    verdict, message = fits.judge(256, RTX4090, measured_case=True)
    assert verdict == "refused"
    assert "Batch_size 128 fits" in message and "23028 MiB" in message


@pytest.mark.parametrize("batch,card,measured,why", [
    (128, RTX4090, False, "another network"),
    (128, None, True, "could not be read"),
    (512, A40, True, "above every measurement"),
])
def test_where_the_table_says_nothing_it_says_so(batch, card, measured, why):
    verdict, message = fits.judge(batch, card, measured_case=measured)
    assert verdict == "not checked" and why in message


def test_only_kraken_plus_from_scratch_is_the_measured_case():
    assert fits.measured_case({"params": {"spec": KRAKEN_PLUS_SPEC}})
    assert fits.measured_case({"params": {}}), "the default spec is KRAKEN_PLUS_SPEC"
    assert not fits.measured_case({"base_model": "kraken-catmus_medieval", "params": {}})
    assert not fits.measured_case({"params": {"spec": "[1,48,0,1 Cr3,3,32]"}})


def _job(tmp_path, **request):
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"id": "x", "request": {"engine": "kraken", **request}}))
    return path


def test_the_command_line_exits_3_on_a_refusal_and_0_otherwise(tmp_path, capsys):
    assert fits.main([str(_job(tmp_path, params={"batch_size": 256})),
                      "--card-mib", str(RTX4090)]) == 3
    assert "refused" in capsys.readouterr().err
    assert fits.main([str(_job(tmp_path, params={"batch_size": 128})),
                      "--card-mib", str(RTX4090)]) == 0
    assert fits.main([str(_job(tmp_path, base_model="kraken-catmus_medieval",
                                params={"batch_size": 256})),
                      "--card-mib", str(RTX4090)]) == 0
    assert "not checked" in capsys.readouterr().out


def test_an_unreadable_card_is_not_checked(tmp_path, capsys):
    assert fits.card_mib("/nonexistent/nvidia-smi") is None
    path = _job(tmp_path, params={"batch_size": 256})
    original = fits.card_mib
    fits.card_mib = lambda *a, **k: None
    try:
        assert fits.main([str(path)]) == 0
    finally:
        fits.card_mib = original
    assert "could not be read" in capsys.readouterr().out


# ── the image ───────────────────────────────────────────────────────────────
def test_the_recipe_puts_ketos_where_the_trainer_looks():
    recipe = (UBELIX / "kraken-train.def").read_text(encoding="utf-8")
    assert "engines/kraken_train_svc/requirements.txt /opt/requirements.txt" in recipe
    assert "python3.12 -m venv /opt/kraken-train" in recipe
    assert TrainerSettings(venvs_root=Path("/opt")).ketos == Path("/opt/kraken-train/bin/ketos")
    assert "torch==2.8.0 torchvision==0.23.0" in recipe and "whl/cu128" in recipe
    assert "assert version('kraken') == '7.0.2'" in recipe, "the build fails on another kraken"
    pins = (ROOT / "engines/kraken_train_svc/requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"^kraken==7\.0\.2$", pins, re.M)


# ── the two batch files ─────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["kraken_prepare.sbatch", "kraken_train.sbatch"])
def test_both_run_the_kraken_runner_in_the_kraken_image(name):
    text = (UBELIX / name).read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(UBELIX / name)], check=True)
    assert "SIF=${SIF:-$HOME/ubelix/kraken-train.sif}" in text
    assert "python -m kraken_train_svc.runner" in text and "vlm_train_svc" not in text
    assert re.search(r"^export ATR_TRAIN_VENVS_ROOT=/opt$", text, re.M)
    assert re.search(r"^export ATR_TRAIN_IMAGE_PYTHON=/opt/kraken-train/bin/python$", text, re.M)
    assert '--env ATR_TRAIN_IMAGE_PYTHON="$ATR_TRAIN_IMAGE_PYTHON"' in text


def test_stage_one_compiles_on_a_cpu_node_and_stops_there():
    text = (UBELIX / "kraken_prepare.sbatch").read_text(encoding="utf-8")
    assert "--stop-after compile" in text
    assert "#SBATCH --partition=epyc2" in text and "--gres" not in text
    assert "apptainer exec --nv" not in text


def test_stage_two_checks_the_batch_before_the_runner_and_ends_without_requeue():
    text = (UBELIX / "kraken_train.sbatch").read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if not line.lstrip().startswith("#")]
    check = next(i for i, line in enumerate(lines)
                 if line.startswith('if ! run python "$REPO/ubelix/kraken_batch_fits.py"'))
    runner = next(i for i, line in enumerate(lines)
                  if line.startswith("run python -m kraken_train_svc.runner"))
    assert check < runner
    assert lines[check + 2] == "exit 3", "a refusal ends the job; it is not requeued"
    assert "#SBATCH --gres=gpu:rtx4090:1" in text


def test_stage_two_is_requeue_and_preemption_safe():
    text = (UBELIX / "kraken_train.sbatch").read_text(encoding="utf-8")
    for directive in ("--requeue", "--signal=B:TERM@120", "--open-mode=append"):
        assert f"#SBATCH {directive}" in text
    assert re.search(r"^export ATR_TRAIN_PREEMPTABLE=1$", text, re.M)
    assert '. "$REPO/ubelix/requeue_trap.sh"' in text
    trap = re.search(r"trap '(.*?)' TERM", text, re.S).group(1)
    assert "requeue_once" in trap and "kill -TERM \"$PID\"" in trap
    assert '[ -n "$REQUEUED" ] || requeue_once' in text
    assert 'STATE=$S/runs/slurm-$SLURM_JOB_ID.jobid' in text


def test_stage_two_without_an_id_says_which_one_it_needs(tmp_path):
    """It does not create a job from a default spec, as train_resumable.sbatch
    would: a kraken corpus is compiled by stage 1."""
    text = (UBELIX / "kraken_train.sbatch").read_text(encoding="utf-8")
    assert "submit_job.py" not in text
    assert "set JOB_ID to the id kraken_prepare.sbatch printed" in text


def test_the_walltimes_fit_the_cpu_cap():
    spec = importlib.util.spec_from_file_location("preflight", UBELIX / "preflight.py")
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)
    for name in ("kraken_prepare.sbatch", "kraken_train.sbatch"):
        res = preflight.resources((UBELIX / name).read_text(encoding="utf-8"), [])
        assert preflight.cpu_minute_problem(res) is None, name


def test_the_peak_alone_is_not_enough_the_headroom_counts():
    """20'500 MiB holds batch 128's measured 20'134, but not the CUDA context and
    allocator slack on top — the reason a 95 % card was already a near miss."""
    assert fits.judge(128, 20_500, measured_case=True)[0] == "refused"
    assert fits.judge(128, 20_134 + fits.HEADROOM_MIB, measured_case=True)[0] == "fits"

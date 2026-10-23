"""In a UBELIX container the image has one interpreter, and the image was chosen
for the base model — so the venv switch of #162 must not apply there.

Since 744bede (#162, 17.10.2026) a Qwen3.5 or Qwen3.8 job picks the venv
``vlm-train-tf5``, which is right on asteraix, where two venvs exist side by
side. Every UBELIX batch file sets ``ATR_TRAIN_VENVS_ROOT=/opt``, and both images
put their single venv at ``/opt/vlm-train`` — so the runner looked for
``/opt/vlm-train-tf5/bin/python``, which no image has, and the train stage of
20261022T151812Z-ladder-med-qwen35-4b-bf16-v2 died 25 s in (Slurm 19562346).
Which transformers a container job gets is decided by ``SIF``, and submit.sh
checks before submitting that the image knows the base (#282).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from atr_training.settings import TrainerSettings

ROOT = Path(__file__).resolve().parents[1]
IMAGE = Path("/opt/vlm-train/bin/python")


@pytest.mark.parametrize("base", ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.8-27B",
                                  "Qwen/Qwen3-VL-4B-Instruct", None])
def test_a_container_runs_every_vlm_job_in_the_image_interpreter(base):
    settings = TrainerSettings(venvs_root=Path("/opt"), image_python=IMAGE)

    assert settings.runner_python("vllm", base) == IMAGE


def test_without_an_image_the_venv_switch_of_162_still_holds(tmp_path):
    """asteraix: two venvs, chosen by the base — unchanged."""
    settings = TrainerSettings(venvs_root=tmp_path)

    assert settings.runner_python("vllm", "Qwen/Qwen3.5-4B") == \
        tmp_path / "vlm-train-tf5" / "bin" / "python"
    assert settings.runner_python("vllm", "Qwen/Qwen3-VL-4B-Instruct") == \
        tmp_path / "vlm-train" / "bin" / "python"


def test_the_image_interpreter_can_be_set_from_the_environment(monkeypatch):
    monkeypatch.setenv("ATR_TRAIN_IMAGE_PYTHON", str(IMAGE))

    assert TrainerSettings().runner_python("vllm", "Qwen/Qwen3.5-4B") == IMAGE


@pytest.mark.parametrize("name", ["train.sbatch", "train_resumable.sbatch",
                                  "prepare.sbatch", "smoke.sbatch"])
def test_every_container_batch_file_names_the_image_interpreter(name):
    """Exported for the shell and passed into the container, like VENVS_ROOT:
    `apptainer exec` does not inherit what the batch script exports unless told."""
    text = (ROOT / "ubelix" / name).read_text(encoding="utf-8")

    assert re.search(r"^export ATR_TRAIN_IMAGE_PYTHON=/opt/vlm-train/bin/python$", text, re.M), \
        f"{name} does not export ATR_TRAIN_IMAGE_PYTHON"
    assert "--env ATR_TRAIN_IMAGE_PYTHON=" in text, \
        f"{name} does not pass ATR_TRAIN_IMAGE_PYTHON into the container"

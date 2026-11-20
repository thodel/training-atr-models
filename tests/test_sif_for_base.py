"""ubelix/sif_for_base.py — the image a VLM job runs in follows its base (#427).

`score_federal_minutes.sbatch` had `vlm-train.sif` (transformers 4) as a fixed
default. The two Qwen3.5 line arms of #427 (22861171, 22861173, 20.11.2026) died
after 40 s on "model type `qwen3_5` ... does not recognize this architecture";
the Qwen3.5 page arm, run in `vlm-train-tf5.sif`, read all 2,751 lines.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "ubelix" / "sif_for_base.py"
_spec = importlib.util.spec_from_file_location("sif_for_base", HELPER)
sfb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sfb)


@pytest.mark.parametrize("base,image", [
    # the two arms that died, and the page arm that ran
    ("Qwen/Qwen3.5-4B", "vlm-train-tf5.sif"),
    ("Qwen/Qwen3.8-27B", "vlm-train-tf5.sif"),
    ("google/gemma-4-12B-it", "vlm-train-tf5.sif"),
    # the arms measured in transformers 4 stay there
    ("Qwen/Qwen3-VL-8B-Instruct", "vlm-train.sif"),
    ("Qwen/Qwen2.5-VL-7B-Instruct", "vlm-train.sif"),
    ("allenai/olmOCR-2-7B-1025", "vlm-train.sif"),
])
def test_the_image_follows_the_trainers_rule(base, image):
    assert sfb.sif_for(base, "/home/u") == f"/home/u/ubelix/{image}"


def test_apertus_has_no_ubelix_image_and_says_so():
    with pytest.raises(ValueError, match="Apertus"):
        sfb.sif_for("swiss-ai/Apertus-v1.5-8B", "/home/u")


def test_it_runs_on_a_bare_host_python():
    """It runs before the container, on a compute node's python3: stdlib only,
    no PYTHONPATH, isolated mode."""
    ok = subprocess.run([sys.executable, "-I", str(HELPER), "Qwen/Qwen3.5-4B", "/h"],
                        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert ok.returncode == 0 and ok.stdout.strip() == "/h/ubelix/vlm-train-tf5.sif"
    no = subprocess.run([sys.executable, "-I", str(HELPER), "swiss-ai/Apertus-v1.5-8B", "/h"],
                        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert no.returncode == 3 and "Apertus" in no.stderr and no.stdout == ""
    bad = subprocess.run([sys.executable, "-I", str(HELPER)], capture_output=True, text=True)
    assert bad.returncode == 2


def test_score_federal_minutes_asks_it_and_has_no_fixed_image():
    text = (ROOT / "ubelix" / "score_federal_minutes.sbatch").read_text(encoding="utf-8")
    assert "SIF=${SIF:-$HOME/ubelix/vlm-train.sif}" not in text
    assert 'sif_for_base.py" "$BASE"' in text
    # chosen after BASE is known, used after it is chosen
    code = 'sif_for_base.py" "$BASE"'
    assert text.index("BASE=${BASE_MODEL") < text.index(code) < text.index('"$SIF" /opt')
    # the interpreter both images have, not whatever `python` is on PATH
    assert '"$SIF" /opt/vlm-train/bin/python -m vlm_train_svc.evaluate_qlora' in text


def test_the_choice_line_picks_tf5_for_a_qwen35_adapter(tmp_path):
    """The two lines of the batch file, run for real: BASE from ADAPTER/BASE_MODEL
    as for the HF arms of #427, SIF unset."""
    text = (ROOT / "ubelix" / "score_federal_minutes.sbatch").read_text(encoding="utf-8")
    choose = [ln for ln in text.splitlines() if "sif_for_base.py" in ln or ln.startswith('[ -f "$SIF" ]')]
    (tmp_path / "ubelix").mkdir()
    for name in ("vlm-train.sif", "vlm-train-tf5.sif"):
        (tmp_path / "ubelix" / name).write_text("")
    script = "set -euo pipefail\n" + "\n".join(choose) + '\necho "$SIF"\n'
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "REPO": str(ROOT)}
    for base, image in (("Qwen/Qwen3.5-4B", "vlm-train-tf5.sif"),
                        ("Qwen/Qwen3-VL-8B-Instruct", "vlm-train.sif")):
        r = subprocess.run(["bash", "-c", f'BASE="{base}"; REPO="$REPO"\n' + script],
                           capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == f"{tmp_path}/ubelix/{image}"
    r = subprocess.run(["bash", "-c", 'BASE="Qwen/Qwen3.5-4B"; SIF=/x/own.sif\n' + script],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 3 and "no image at /x/own.sif" in r.stderr   # SIF= wins, and is checked

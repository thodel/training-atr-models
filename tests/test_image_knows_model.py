"""#206: submit.sh asks the image whether it knows the base model before sbatch.

The verdict is a function of the config's ``model_type`` and the image's
``CONFIG_MAPPING_NAMES``; the image contributes only the set, so the decision
is pinned here without one. 17808414 was the run that died in 59 seconds.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "ubelix" / "image_knows_model.py"
SUBMIT = (Path(__file__).resolve().parents[1] / "ubelix" / "submit.sh").read_text(encoding="utf-8")


def _load():
    spec = importlib.util.spec_from_file_location("image_knows_model", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


knows = _load()
TF4 = {"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "gemma3"}
TF5 = TF4 | {"qwen3_5", "gemma4"}


def test_an_architecture_the_image_knows_passes():
    assert knows.verdict("qwen3_vl", TF4) == (0, "IMAGE=knows qwen3_vl")
    assert knows.verdict("qwen3_5", TF5)[0] == 0


def test_the_59_second_death_is_refused_before_sbatch():
    code, line = knows.verdict("qwen3_5", TF4)
    assert code == 3
    assert "qwen3_5" in line and "does not know" in line and "SIF=" in line


def test_could_not_tell_is_two_not_zero():
    """No transformers, or a config without a model_type: not a pass."""
    assert knows.verdict("qwen3_5", None)[0] == 2
    assert knows.verdict(None, TF5)[0] == 2
    assert knows.verdict("", TF5)[0] == 2


def test_a_local_checkpoint_is_read_without_the_hub(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    assert knows.model_type_of(str(tmp_path)) == "qwen3_5"


def test_main_refuses_through_the_exit_code(tmp_path, monkeypatch, capsys):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"model_id": "x", "base_model": str(base)}))

    monkeypatch.setattr(knows, "known_types", lambda: TF4)
    assert knows.main([str(spec)]) == 3
    assert "IMAGE=refuses qwen3_5" in capsys.readouterr().out

    monkeypatch.setattr(knows, "known_types", lambda: TF5)
    assert knows.main([str(spec)]) == 0

    spec.write_text(json.dumps({"model_id": "x"}))
    assert knows.main([str(spec)]) == 0, "a spec without a base model has nothing to check"


def test_an_unreadable_config_is_could_not_tell(tmp_path, monkeypatch, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"model_id": "x", "base_model": str(tmp_path / "nowhere")}))
    monkeypatch.setattr(knows, "model_type_of", lambda base: (_ for _ in ()).throw(OSError("no network")))
    assert knows.main([str(spec)]) == 2
    assert "IMAGE=unknown" in capsys.readouterr().out


def test_submit_runs_the_check_in_the_jobs_image_and_refuses_on_three():
    """Asked of active lines, not comments: submit.sh must *run* the script, in
    the image the job will use, and exit 3 must stop the submission."""
    lines = [line.strip() for line in SUBMIT.splitlines()
             if line.strip() and not line.strip().startswith("#")]
    assert any("image_knows_model.py" in line for line in lines), "submit.sh does not run the check"
    assert any('"$JOB_SIF"' in line and "python3" in line or '"$JOB_SIF"' in line
               for line in lines), "the check runs in a fixed image, not the job's"
    assert any("KNOWS_CODE" in line and "= 3" in line for line in lines), "exit 3 does not refuse"
    assert any(line.startswith("exit 3") for line in lines)

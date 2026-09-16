"""Deploy scripts and settings agree with each other (#8)."""

from __future__ import annotations

import re
from pathlib import Path

from atr_training.backends import BACKENDS
from atr_training.settings import TrainerSettings

REPO = Path(__file__).resolve().parents[1]


def _venvs_in(script: str, pattern: str) -> set[str]:
    return set(re.findall(pattern, (REPO / "scripts" / script).read_text(encoding="utf-8")))


def test_check_venvs_covers_every_backend_it_builds():
    """The serving repo's check_venvs.sh had no trocr-train entry at all.

    make_venvs.sh built it and backends.py declared it, so the gate after
    provisioning did not check one of the venvs it had just provisioned.
    Three lists, one truth: the backends.
    """
    declared = {b.venv for b in BACKENDS.values()}
    built = _venvs_in("make_venvs.sh", r"\[([a-z-]+-train)\]=")
    checked = _venvs_in("check_venvs.sh", r'"([a-z-]+-train)\|')
    assert declared == built, f"backends {declared} vs make_venvs {built}"
    assert declared == checked, f"backends {declared} vs check_venvs {checked}"


def test_the_scripts_read_the_key_the_service_reads():
    """Not a bare VENVS_ROOT, which the service never consulted."""
    for script in ("make_venvs.sh", "check_venvs.sh"):
        text = (REPO / "scripts" / script).read_text(encoding="utf-8")
        assert "grep '^ATR_TRAIN_VENVS_ROOT='" in text, script
        # The bug was LOOKING UP a bare key in .env; a shell-local variable
        # derived from the right key is fine.
        assert "'^VENVS_ROOT='" not in text, script
    assert "venvs_root" in TrainerSettings.model_fields


def test_ketos_follows_venvs_root(tmp_path):
    """Moving the venvs must move ketos with them (found by the check of #3)."""
    s = TrainerSettings(venvs_root=tmp_path / "elsewhere")
    assert s.ketos == tmp_path / "elsewhere" / "kraken-train" / "bin" / "ketos"
    assert s.runner_python("kraken").parent == s.ketos.parent


def test_an_explicit_ketos_still_wins(tmp_path):
    s = TrainerSettings(venvs_root=tmp_path / "v", ketos=tmp_path / "custom-ketos")
    assert s.ketos == tmp_path / "custom-ketos"


def test_the_checkpoint_root_is_never_on_the_share():
    """Checkpoints are saved via temp file + rename, cross-device on CIFS."""
    root = str(TrainerSettings.model_fields["checkpoint_root"].default)
    assert not root.startswith("/mnt/"), root


def test_the_bind_is_written_in_the_unit_itself():
    """Widening the bind has to be a visible edit in git, not a .env line (#13).

    It was loopback until #13 opened it; what makes 0.0.0.0 safe is pinned in
    tests/test_trainer_access.py.
    """
    unit = (REPO / "deploy/systemd/atr-train.service").read_text(encoding="utf-8")
    exec_line = next(line for line in unit.splitlines() if line.startswith("ExecStart="))
    assert "--host 0.0.0.0" in exec_line and "--port 8204" in exec_line
    assert "${" not in exec_line and "$ATR" not in exec_line, \
        "the bind must not come from the environment"


def test_the_unit_keeps_detached_runs_alive():
    """Without it a restart SIGTERMs the training run (idhefix, 2026-08-07)."""
    unit = (REPO / "deploy/systemd/atr-train.service").read_text(encoding="utf-8")
    assert "\nKillMode=process" in unit
    assert "PYTHONPATH=%h/Repo/training-atr-models/src" in unit
    assert "HF_HOME" not in [line.split("=")[0].replace("Environment=", "")
                             for line in unit.splitlines() if line.startswith("Environment=")]

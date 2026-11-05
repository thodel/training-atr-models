"""#357 — a trained Qwen3.5 is registered so that the gateway can start it.

On 02.11.2026 ``qwen3.5-4b-german-kurrent-xvi-xix-blb-page-v1`` was registered by
hand with ``vllm_venv: vllm-next`` and no ``max_num_seqs``. The gateway launched
it at vLLM's default of 256 sequences and the engine died: "max_num_seqs (256)
exceeds available Mamba cache blocks (130)" — the failure the curated Qwen3.5
entries had already been fixed for on 21.09. The base says what the launch
needs; the registration now says it too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atr_training.contracts import DatasetSpec, Metrics, TrainRequest
from atr_training.jobstore import JobStore
from atr_training.registration import (
    HYBRID_MAX_NUM_SEQS,
    HYBRID_VLLM_VENV,
    RegistrationError,
    manual_registration,
    read_registration,
    registration_path,
    vllm_launch_defaults,
    write_registration,
)
from atr_training.settings import TrainerSettings

QWEN35 = "Qwen/Qwen3.5-4B"
GEMMA4 = "google/gemma-4-12B-it"


def _spec(tmp_path: Path, model_id: str, base: str | None, **extra) -> dict:
    weights = tmp_path / "trained" / model_id
    weights.mkdir(parents=True, exist_ok=True)
    return {"id": model_id, "engine": "vllm", "local_path": str(weights),
            "base_model": base, "enabled": False, "level": "page", **extra}


# ── what a base needs ───────────────────────────────────────────────────────

def test_a_qwen35_base_needs_the_next_venv_and_a_lower_ceiling():
    assert vllm_launch_defaults("vllm", QWEN35) == {
        "vllm_venv": HYBRID_VLLM_VENV, "max_num_seqs": HYBRID_MAX_NUM_SEQS}
    assert HYBRID_MAX_NUM_SEQS == 64, "the value of the curated Qwen3.5 entries"


@pytest.mark.parametrize("base", [GEMMA4, "google/gemma-4-E4B-it"])
def test_a_gemma4_base_needs_the_next_venv_and_no_ceiling(base):
    """#83: vLLM 0.11 knows neither Gemma 4 architecture, 0.29 both. Gemma is not
    hybrid, so there is no Mamba cache to keep the sequence count under."""
    assert vllm_launch_defaults("vllm", base) == {"vllm_venv": HYBRID_VLLM_VENV}


def test_a_gemma4_without_the_venv_is_refused(tmp_path):
    spec = _spec(tmp_path, "gemma4-novenv", GEMMA4)
    with pytest.raises(RegistrationError, match="vllm_venv: vllm-next"):
        write_registration(tmp_path / "registry", spec)


def test_a_gemma4_with_the_venv_is_written(tmp_path):
    root = tmp_path / "registry"
    root.mkdir()
    write_registration(root, _spec(tmp_path, "gemma4-ok", GEMMA4, vllm_venv=HYBRID_VLLM_VENV))
    written = read_registration(root, "gemma4-ok")
    assert (written.vllm_venv, written.max_num_seqs) == (HYBRID_VLLM_VENV, None)


@pytest.mark.parametrize("engine, base", [
    ("vllm", "Qwen/Qwen3-VL-8B-Instruct"),
    ("vllm", "allenai/olmOCR-2-7B-1025"),
    ("vllm", None),
    ("kraken", QWEN35),
])
def test_every_other_base_needs_nothing(engine, base):
    assert vllm_launch_defaults(engine, base) == {}


# ── writing refuses what could not start ───────────────────────────────────

def test_the_2_november_registration_is_refused_before_it_exists(tmp_path):
    root = tmp_path / "registry"
    spec = _spec(tmp_path, "qwen35-page", QWEN35, vllm_venv=HYBRID_VLLM_VENV)
    with pytest.raises(RegistrationError, match="max_num_seqs: 64"):
        write_registration(root, spec)
    assert not registration_path(root, "qwen35-page").exists()


def test_a_qwen35_without_the_venv_is_refused_too(tmp_path):
    spec = _spec(tmp_path, "qwen35-novenv", QWEN35, max_num_seqs=64)
    with pytest.raises(RegistrationError, match="vllm_venv: vllm-next"):
        write_registration(tmp_path / "registry", spec)


def test_with_both_it_is_written(tmp_path):
    root = tmp_path / "registry"
    root.mkdir()
    write_registration(root, _spec(tmp_path, "qwen35-ok", QWEN35,
                                   vllm_venv=HYBRID_VLLM_VENV, max_num_seqs=32))
    written = read_registration(root, "qwen35-ok")
    assert written.max_num_seqs == 32, "a value set on purpose is not replaced"
    assert written.vllm_venv == HYBRID_VLLM_VENV


def test_an_olmocr_base_is_written_without_either(tmp_path):
    root = tmp_path / "registry"
    root.mkdir()
    write_registration(root, _spec(tmp_path, "olmocr-page", "allenai/olmOCR-2-7B-1025"))
    written = read_registration(root, "olmocr-page")
    assert written.max_num_seqs is None and written.vllm_venv is None


def test_a_file_already_there_stays_readable(tmp_path):
    """Refused when written, not when read: an old registration can still be
    disabled, and the gateway reads it as it always did."""
    root = tmp_path / "registry"
    path = registration_path(root, "qwen35-old")
    path.parent.mkdir(parents=True)
    weights = tmp_path / "trained" / "qwen35-old"
    weights.mkdir(parents=True)
    path.write_text(f"id: qwen35-old\nengine: vllm\nlocal_path: {weights}\n"
                    f"base_model: {QWEN35}\nenabled: false\n", encoding="utf-8")
    assert read_registration(root, "qwen35-old").max_num_seqs is None


# ── the runner fills them in ───────────────────────────────────────────────

def _pipeline(tmp_path: Path):
    from vlm_train_svc.runner import Pipeline
    registry_root = tmp_path / "registry"
    registry_root.mkdir()
    settings = TrainerSettings(
        _env_file=None, jobs_root=tmp_path / "jobs", trained_root=tmp_path / "trained",
        registry_root=registry_root, checkpoint_root=tmp_path / "ckpt")
    store = JobStore(settings.jobs_root, host_id=settings.host_id)
    return Pipeline(store, settings), store, registry_root


def test_the_runner_registers_a_qwen35_the_gateway_can_start(tmp_path):
    pipeline, store, root = _pipeline(tmp_path)
    job = store.create(TrainRequest(
        engine="vllm", model_id="qwen35-trained", force=True, base_model=QWEN35,
        dataset=DatasetSpec(hf_repo="dh-unibe/x", train_projects=["p"])))
    spec = _spec(tmp_path, "qwen35-trained", QWEN35)
    pipeline._write_registration(job, spec, Path(spec["local_path"]), metrics=Metrics(cer=0.2))
    written = read_registration(root, "qwen35-trained")
    assert (written.vllm_venv, written.max_num_seqs) == (HYBRID_VLLM_VENV, HYBRID_MAX_NUM_SEQS)


def test_the_hand_registration_of_a_slurm_job_carries_them(tmp_path):
    """What an operator pastes after a UBELIX job is built from the same spec."""
    pipeline, store, root = _pipeline(tmp_path)
    pipeline._slurm_job_id = "20187808"
    job = store.create(TrainRequest(
        engine="vllm", model_id="qwen35-ubelix", force=True, base_model=QWEN35,
        dataset=DatasetSpec(hf_repo="dh-unibe/x", train_projects=["p"])))
    spec = _spec(tmp_path, "qwen35-ubelix", QWEN35)
    assert pipeline._write_registration(job, spec, Path(spec["local_path"])) is None
    assert "max_num_seqs: 64" in job.registration
    assert "vllm_venv: vllm-next" in job.registration
    assert not registration_path(root, "qwen35-ubelix").exists(), "a Slurm job never writes (#17)"
    # and pasted, it registers
    assert "max_num_seqs: 64" in manual_registration(root, spec)

"""Which runner, in which venv, for which engine.

There is **one** training service (``atr-train`` on :8204) supervising every
backend, not one per engine. That is not tidiness — it is the GPU. Training and
inference do not share a card politely (serving-atr-inference/docs/TRAINING_PLAN.md §5), so exactly one
training job may run at a time; two services would each enforce
``max_concurrent=1`` against their own job list and cheerfully start a kraken run
and a VLM run into the same 45 GB. One supervisor, one queue, one guard.

The isolation the engines *do* need is the venv: kraken pins ``kraken==7.0.2``
with ``datasets<4``, while the VLM trainer needs ``transformers`` new enough for
Qwen3-VL plus peft/trl/bitsandbytes. Those cannot share a dependency tree — the
same reason the serving engines each have their own (IMPLEMENTATION_PLAN §3).

So the supervisor stays dependency-free and each job is spawned as a **detached
child of the right interpreter**::

    .venvs/kraken-train/bin/python -m kraken_train_svc.runner --root … --job-id …
    .venvs/vlm-train/bin/python    -m vlm_train_svc.runner    --root … --job-id …

Nothing in either engine package is ever imported by the service, so a broken or
missing VLM venv cannot stop kraken jobs from running (and vice versa).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["UnknownBackend", "Backend", "BACKENDS", "backend_for", "runner_python",
           "venv_for", "TF5_VENV", "needs_transformers_5", "is_hybrid_base",
           "APERTUS_VENV", "needs_apertus_fork"]


class UnknownBackend(ValueError):
    """Raised for an engine with no training backend."""


@dataclass(frozen=True)
class Backend:
    engine: str
    #: ``python -m <module>`` — the runner's entry point.
    runner_module: str
    #: Directory under ``.venvs/`` whose interpreter runs it.
    venv: str
    #: Requirements file, for the error message when the venv is missing.
    requirements: str
    #: Whether this backend implements chunked materialize -> compile -> discard.
    #: **Mirrors the runner class's ``supports_chunked_prepare``**, which cannot be
    #: read from here: each runner lives in its own venv and this service imports
    #: neither. Declared rather than introspected, and pinned by
    #: ``tests/test_training_backends.py`` so the two cannot drift.
    #:
    #: The size guard reads this. Without it, ``ATR_TRAIN_CHUNK_PAGES`` alone
    #: cleared a 293 GB vllm corpus on the strength of a setting that backend
    #: ignores, and every page would have been materialized before compile (#85).
    supports_chunked_prepare: bool = False


BACKENDS: dict[str, Backend] = {
    "kraken": Backend(
        engine="kraken",
        runner_module="kraken_train_svc.runner",
        venv="kraken-train",
        requirements="engines/kraken_train_svc/requirements.txt",
        supports_chunked_prepare=True,
    ),
    "vllm": Backend(
        engine="vllm",
        runner_module="vlm_train_svc.runner",
        venv="vlm-train",
        requirements="engines/vlm_train_svc/requirements.txt",
    ),
    "trocr": Backend(
        engine="trocr",
        runner_module="trocr_train_svc.runner",
        venv="trocr-train",
        requirements="engines/trocr_train_svc/requirements.txt",
    ),
}


def backend_for(engine: str) -> Backend:
    try:
        return BACKENDS[engine]
    except KeyError:
        raise UnknownBackend(
            f"no training backend for engine {engine!r}. Trainable: {sorted(BACKENDS)}"
        ) from None


#: The VLM venv for bases that transformers 4.x cannot load. Qwen3.5 folded vision
#: into the base models and reports ``model_type: qwen3_5``; the 4.x line ended at
#: 4.57.6 before it shipped (ubelix/vlm-train-tf5.def). UBELIX has had a second
#: image for that since 10.09.2026; asteraix had only ``vlm-train``, so a Qwen3.5
#: job here died in its train stage after a 1 h 40 prepare
#: (20261014T160430Z-throughput-qwen35-4b-a40, #162). A second venv rather than an
#: upgrade, for the reason the image gives: the Qwen3-VL runs keep the stack they
#: were measured on, and the two differ in exactly one pin.
TF5_VENV = "vlm-train-tf5"
#: Hybrid bases (``qwen3_5``): transformers 5, and a Mamba-style state per sequence.
_HYBRID_FAMILIES = ("qwen3.5-", "qwen3.8-")
#: Gemma 4 (``gemma4``, ``gemma4_unified``) is not in 4.57.6 either, but it is a
#: plain transformer. Measured 05.11.2026 (#83): idhefix's ``vllm`` (0.11.0,
#: transformers 4.57.6) knows neither architecture, ``vllm-next`` (0.29.0, 5.17.0)
#: knows both; the UBELIX Gemma runs trained in ``vlm-train-tf5.sif``.
_GEMMA4_FAMILIES = ("gemma-4-",)


def _repo_name(base_model: str | None) -> str:
    return str(base_model).rsplit("/", 1)[-1].lower() if base_model else ""


def is_hybrid_base(base_model: str | None) -> bool:
    """Is ``base_model`` a hub id of the hybrid ``qwen3_5`` family?

    Matched on the repo name, case-insensitively: ``Qwen/Qwen3.5-4B``,
    ``Qwen/Qwen3.8-27B``. A Qwen3-VL base, a local path and None are not.
    """
    return _repo_name(base_model).startswith(_HYBRID_FAMILIES)


def needs_transformers_5(base_model: str | None) -> bool:
    """Does ``base_model`` need transformers 5 — a ``qwen3_5`` or a Gemma 4 base?"""
    return is_hybrid_base(base_model) or _repo_name(base_model).startswith(_GEMMA4_FAMILIES)


#: The VLM venv for Apertus 1.5 (training-atr-models#316). Its model type,
#: ``apertus1p5``, exists only in swiss-ai's transformers fork
#: (``feature/apertus_1p5_pipeline``), the same commit the serving box's
#: ``vllm-apertus`` venv runs (serving-atr-inference#234). A venv of its own
#: because that fork is a transformers 5 dev build that the Qwen3.5 runs must not
#: be moved onto.
APERTUS_VENV = "vlm-train-apertus"
_APERTUS_FAMILIES = ("apertus-v1.5-",)


def needs_apertus_fork(base_model: str | None) -> bool:
    """Is ``base_model`` an Apertus 1.5 hub id (``swiss-ai/Apertus-v1.5-8B``)?"""
    if not base_model:
        return False
    name = str(base_model).rsplit("/", 1)[-1].lower()
    return name.startswith(_APERTUS_FAMILIES)


def venv_for(engine: str, base_model: str | None = None) -> str:
    """The venv directory a job of ``engine`` on ``base_model`` runs in."""
    backend = backend_for(engine)
    if engine == "vllm" and needs_apertus_fork(base_model):
        return APERTUS_VENV
    if engine == "vllm" and needs_transformers_5(base_model):
        return TF5_VENV
    return backend.venv


def runner_python(engine: str, venvs_root: str | Path,
                  base_model: str | None = None) -> Path:
    """Interpreter that runs ``engine``'s jobs — on ``base_model``, where the
    base decides the venv (a ``qwen3_5`` base needs transformers 5).

    Existence is checked by the caller at spawn time, not here: a box that only
    ever trains kraken has every right not to have built the VLM venv, and that
    should surface as a clear failure on the job that needs it — not as an import
    error in the service. There is no fallback to ``vlm-train`` for a Qwen3.5
    base: that is the venv the job would die in.
    """
    return Path(venvs_root) / venv_for(engine, base_model) / "bin" / "python"

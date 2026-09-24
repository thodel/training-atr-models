"""Resolving ``TrainRequest.base_model`` — and refusing a bad one at submit (#76).

`serving-atr-inference/docs/TRAINING_PLAN.md` §4 promised that ``base_model`` accepts "a registry id or a
Zenodo DOI". It accepted only a DOI: the kraken runner handed the string straight to
``htrmopo.get_model``, so a perfectly reasonable request died in the **train** stage,
after prepare and compile had already run —

    ValueError in train: kraken-medieval_generic_b is not a valid DOI

`kraken-medieval_generic_b` is in ``config/models.yaml``. The user had every reason to
expect it to work, and lost a run finding out otherwise.

Two things are fixed here, and the second matters more than the first:

1. **A registry id resolves** to the `zenodo_id` (or `local_path`) of its entry, which
   is what §4 described all along.
2. **The check happens at submit.** Everything needed to reject a bad reference is
   available the moment the request arrives; nothing about it improves by waiting for a
   ten-hour prepare to finish first. Same argument as the dataset verification (#46) and
   the step-count guard (#72) — the difference between a guard and a post-mortem is
   where it runs.

The base is engine-specific and the namespaces do not overlap: kraken bases are kraken
weights (Zenodo, or a file on disk), while `vllm` and `trocr` bases are HuggingFace repo
ids. Validating one against the other's rules would reject correct requests, so each
engine gets its own.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol


class _Entry(Protocol):
    id: str
    engine: str
    zenodo_id: str | None
    local_path: str | None


class RegistryLike(Protocol):
    """What resolving a base needs from a registry — and all it needs.

    Structural on purpose. The trainer no longer imports the gateway's
    ``Registry``; it reads the published file through
    :class:`atr_training.shared_registry.SharedRegistry`, and the tests build
    their own. Anything with these two methods will do.
    """

    def get(self, model_id: str) -> _Entry | None: ...
    def by_engine(self, engine: str) -> list[_Entry]: ...

__all__ = [
    "BaseModelError",
    "ResolvedBase",
    "DOI_RE",
    "HF_REPO_RE",
    "RegistryLike",
    "resolve_base_model",
]

#: A Zenodo DOI as htrmopo accepts it — ``10.5281/zenodo.15366732``.
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
#: A bare Zenodo record id, which htrmopo also takes.
RECORD_RE = re.compile(r"^\d{4,}$")
#: A HuggingFace repo id — ``owner/name``, the base form for vllm and trocr.
HF_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")

#: Engines whose base is a kraken weights file rather than a HuggingFace repo.
_KRAKEN_LIKE = frozenset({"kraken"})


class BaseModelError(ValueError):
    """Raised when ``base_model`` names nothing this engine can start from."""


@dataclass(frozen=True)
class ResolvedBase:
    """What the runner should actually load."""

    #: Handed to htrmopo (a DOI or record id), or used as a path / HF repo id.
    ref: str
    #: ``registry`` when ``ref`` was looked up, so the error and the metadata can
    #: say which id produced it.
    kind: str
    source_id: str | None = None

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.ref if not self.source_id else f"{self.source_id} → {self.ref}"


def _kraken_base_ids(registry: RegistryLike | None) -> list[str]:
    """Registry ids that can actually serve as a kraken fine-tuning base."""
    if registry is None:
        return []
    return sorted(
        spec.id for spec in registry.by_engine("kraken")
        if spec.zenodo_id or spec.local_path
    )


#: How many ids a refusal lists. The published registry has 43 kraken bases,
#: and a message carrying all of them buried the one that was meant (#39).
SUGGESTION_LIMIT = 10


def _closest_kraken_base_ids(registry: RegistryLike | None, ref: str) -> str:
    """The kraken bases closest to ``ref``, formatted for a refusal message.

    Ranked by :class:`difflib.SequenceMatcher` ratio against what was typed,
    ties alphabetically so the message is the same on every run, and capped at
    :data:`SUGGESTION_LIMIT` with a count of the rest. A near-miss is how this
    error usually happens, so the id that was meant should come first.
    """
    known = _kraken_base_ids(registry)
    ranked = sorted(known, key=lambda cid: (
        -difflib.SequenceMatcher(None, ref, cid).ratio(), cid))
    shown = ranked[:SUGGESTION_LIMIT]
    rest = len(known) - len(shown)
    return f"{shown}" + (f" and {rest} more" if rest else "")


def resolve_base_model(
    base_model: str,
    engine: str = "kraken",
    registry: RegistryLike | None = None,
    path_exists: Callable[[str], bool] | None = None,
    registry_error: str | None = None,
) -> ResolvedBase:
    """Turn ``base_model`` into something the engine can load, or explain why not.

    ``path_exists`` is injectable so this stays testable without touching the
    filesystem; it defaults to a real check.

    ``registry_error`` is why ``registry`` is None, when the caller tried to load
    one and could not. It changes only the message, and the message is the whole
    point: without it, a valid id against an unreadable registry was reported as
    "not a registry id", and the user went looking for a typo that was not
    theirs.
    """
    exists = path_exists or (lambda p: Path(p).expanduser().exists())
    ref = (base_model or "").strip()
    if not ref:
        raise BaseModelError("base_model is empty")

    # A path on disk wins for every engine — it is unambiguous, and it is how a
    # locally trained model is fine-tuned further.
    if exists(ref):
        return ResolvedBase(ref=str(Path(ref).expanduser()), kind="path")

    if engine in _KRAKEN_LIKE:
        return _resolve_kraken(ref, registry, registry_error)
    return _resolve_hf(ref, engine)


def _resolve_kraken(ref: str, registry: RegistryLike | None,
                    registry_error: str | None = None) -> ResolvedBase:
    if registry is not None and (spec := registry.get(ref)) is not None:
        if spec.engine != "kraken":
            raise BaseModelError(
                f"{ref!r} is a {spec.engine} model; a kraken run needs kraken weights. "
                f"Closest kraken bases: {_closest_kraken_base_ids(registry, ref)}"
            )
        target = spec.zenodo_id or spec.local_path
        if not target:
            raise BaseModelError(
                f"registry entry {ref!r} has neither zenodo_id nor local_path, so there "
                "is nothing to fine-tune from"
            )
        return ResolvedBase(ref=target, kind="registry", source_id=ref)

    if DOI_RE.match(ref) or RECORD_RE.match(ref):
        return ResolvedBase(ref=ref, kind="zenodo")

    if registry is None:
        # Not "not a registry id": with no registry there was nothing to look it
        # up in, and saying otherwise sends the user after a typo.
        why = f" ({registry_error})" if registry_error else " (none is configured)"
        raise BaseModelError(
            f"base_model {ref!r} is not a file or a Zenodo reference, and no "
            f"registry was available to look it up as an id{why}. A Zenodo DOI "
            "(10.xxxx/zenodo.NNNN) or a path would not need one."
        )

    hint = (f" Closest registry ids: {_closest_kraken_base_ids(registry, ref)}."
            if _kraken_base_ids(registry) else "")
    raise BaseModelError(
        f"base_model {ref!r} is not a file, a registry id, or a Zenodo reference "
        f"(10.xxxx/zenodo.NNNN, or a bare record id).{hint}"
    )


def _resolve_hf(ref: str, engine: str) -> ResolvedBase:
    # A DOI satisfies owner/name — "10.5281/zenodo.15366732" is a leading
    # alphanumeric, a slash, and word characters. Checking the repo pattern first
    # therefore accepts a kraken base for a VLM run and fails much later, inside
    # transformers. Rule the DOI out explicitly rather than hoping the pattern
    # discriminates.
    if DOI_RE.match(ref) or RECORD_RE.match(ref):
        raise BaseModelError(
            f"base_model {ref!r} is a Zenodo reference — a kraken base. A {engine} run "
            "fine-tunes from a HuggingFace repo id (owner/name) or a local path."
        )
    if HF_REPO_RE.match(ref):
        return ResolvedBase(ref=ref, kind="hf_repo")
    raise BaseModelError(
        f"base_model {ref!r} is not a HuggingFace repo id (owner/name) or a local "
        f"path — which is what a {engine} run fine-tunes from. Zenodo DOIs are kraken "
        "bases and cannot be loaded here."
    )

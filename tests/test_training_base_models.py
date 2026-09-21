"""Resolving base_model, and refusing a bad one at submit (#76).

The run this exists for:

    20260810T105206Z-kraken-thun-finetune-v1  failed
    ValueError in train: kraken-medieval_generic_b is not a valid DOI

`kraken-medieval_generic_b` is in config/models.yaml, and TRAINING_PLAN §4 said a
registry id works. It did not, and the failure landed in the *train* stage — after
prepare and compile had already run.
"""

import re

import pytest

from atr_training.base_models import (
    BaseModelError,
    resolve_base_model,
)
from atr_training.shared_registry import BaseEntry, SharedRegistry


@pytest.fixture
def registry() -> SharedRegistry:
    # BaseEntry ignores the fields a lookup does not need (task, hf_repo, ...),
    # exactly as it does when it reads the published file.
    return SharedRegistry([BaseEntry.model_validate(e) for e in [
        dict(id="kraken-late_medieval_german", engine="kraken",
             zenodo_id="10.5281/zenodo.15366732", task="htr"),
        dict(id="kraken-medieval_generic_b", engine="kraken",
             zenodo_id="10.5281/zenodo.18220238", task="htr"),
        dict(id="kraken-locally-trained", engine="kraken",
             local_path="/atr-cache/trained/x/x.mlmodel", enabled=False),
        dict(id="qwen3vl-8b-hebrew", engine="vllm",
             hf_repo="wjbmattingly/Qwen3-VL-8B-hebrew-3-epochs",
             base_model="Qwen/Qwen3-VL-8B-Instruct"),
    ]])


def never_exists(_path: str) -> bool:
    return False


# ── the case that cost a run ────────────────────────────────────────────────
def test_a_registry_id_resolves_to_its_doi(registry):
    resolved = resolve_base_model("kraken-medieval_generic_b", "kraken", registry,
                                  path_exists=never_exists)
    assert resolved.ref == "10.5281/zenodo.18220238"
    assert resolved.kind == "registry"
    assert resolved.source_id == "kraken-medieval_generic_b"


def test_a_registry_entry_with_only_local_weights_resolves_to_the_path(registry):
    """A model trained here, fine-tuned further — there is no DOI to fetch."""
    resolved = resolve_base_model("kraken-locally-trained", "kraken", registry,
                                  path_exists=never_exists)
    assert resolved.ref == "/atr-cache/trained/x/x.mlmodel"


# ── the forms that already worked, which must keep working ──────────────────
@pytest.mark.parametrize("ref", ["10.5281/zenodo.15366732", "15366732"])
def test_a_zenodo_reference_passes_through(ref, registry):
    resolved = resolve_base_model(ref, "kraken", registry, path_exists=never_exists)
    assert resolved.ref == ref and resolved.kind == "zenodo"


def test_a_path_on_disk_wins_for_every_engine(registry):
    for engine in ("kraken", "vllm", "trocr"):
        resolved = resolve_base_model("/models/base.mlmodel", engine, registry,
                                      path_exists=lambda p: True)
        assert resolved.kind == "path"


# ── refusals, and whether they help ─────────────────────────────────────────
def test_an_unknown_reference_lists_the_ids_that_would_work(registry):
    with pytest.raises(BaseModelError) as exc:
        resolve_base_model("kraken-medieval_generic_z", "kraken", registry,
                           path_exists=never_exists)
    message = str(exc.value)
    assert "kraken-medieval_generic_b" in message      # what they probably meant
    assert "10.xxxx/zenodo.NNNN" in message            # and the other accepted form



def _listed(message: str, label: str) -> list[str]:
    match = re.search(label + r": \[(.*?)\]", message)
    assert match, message
    return re.findall(r"'([^']+)'", match.group(1))


@pytest.fixture
def crowded_registry() -> SharedRegistry:
    """Thirteen kraken bases, five of them one letter from the typo below —
    the shape of the published registry (43 ids) at a size a test can read."""
    near = [f"kraken-medieval_generic_{c}" for c in "abcde"]
    far = ["kraken-early_modern_german", "kraken-late_medieval_german",
           "kraken-hebrew_square", "kraken-greek_minuscule", "kraken-arabic_naskh",
           "kraken-cyrillic_poluustav", "kraken-latin_caroline", "kraken-norse_runic"]
    return SharedRegistry([
        BaseEntry(id=cid, engine="kraken", zenodo_id=f"10.5281/zenodo.{n}")
        for n, cid in enumerate(sorted(far + near), start=1000)
    ] + [BaseEntry(id="qwen3vl-8b-hebrew", engine="vllm")])


def test_the_refusal_lists_the_closest_ids_first_and_at_most_ten(crowded_registry):
    """#39: the typo's neighbours lead, the list stops at ten, the rest is counted."""
    with pytest.raises(BaseModelError) as exc:
        resolve_base_model("kraken-medieval_generic_z", "kraken", crowded_registry,
                           path_exists=never_exists)
    listed = _listed(str(exc.value), "Closest registry ids")
    assert listed[:5] == [f"kraken-medieval_generic_{c}" for c in "abcde"]
    assert len(listed) == 10
    assert "and 3 more" in str(exc.value)


def test_a_short_registry_is_listed_whole(registry):
    with pytest.raises(BaseModelError) as exc:
        resolve_base_model("kraken-medieval_generic_z", "kraken", registry,
                           path_exists=never_exists)
    listed = _listed(str(exc.value), "Closest registry ids")
    assert listed[0] == "kraken-medieval_generic_b"
    assert sorted(listed) == ["kraken-late_medieval_german", "kraken-locally-trained",
                              "kraken-medieval_generic_b"]
    assert "more" not in str(exc.value)


def test_the_wrong_engine_refusal_is_capped_too(crowded_registry):
    with pytest.raises(BaseModelError, match="is a vllm model") as exc:
        resolve_base_model("qwen3vl-8b-hebrew", "kraken", crowded_registry,
                           path_exists=never_exists)
    assert len(_listed(str(exc.value), "Closest kraken bases")) == 10

def test_a_vllm_model_is_refused_as_a_kraken_base(registry):
    """Both are registry ids; only one is kraken weights."""
    with pytest.raises(BaseModelError, match="is a vllm model"):
        resolve_base_model("qwen3vl-8b-hebrew", "kraken", registry,
                           path_exists=never_exists)


def test_an_empty_base_model_is_refused(registry):
    with pytest.raises(BaseModelError, match="empty"):
        resolve_base_model("   ", "kraken", registry, path_exists=never_exists)


# ── the namespaces do not overlap ───────────────────────────────────────────
def test_a_vlm_base_is_a_huggingface_repo(registry):
    resolved = resolve_base_model("Qwen/Qwen3-VL-8B-Instruct", "vllm", registry,
                                  path_exists=never_exists)
    assert resolved.ref == "Qwen/Qwen3-VL-8B-Instruct" and resolved.kind == "hf_repo"


def test_a_zenodo_doi_is_refused_for_a_vlm_run(registry):
    """Validating one engine's base against the other's rules would either accept
    nonsense or reject correct requests; the error says which is which."""
    with pytest.raises(BaseModelError, match="is a Zenodo reference"):
        resolve_base_model("10.5281/zenodo.15366732", "vllm", registry,
                           path_exists=never_exists)


def test_a_doi_is_not_mistaken_for_a_huggingface_repo(registry):
    """`10.5281/zenodo.15366732` satisfies owner/name — leading alphanumeric,
    slash, word characters. Pattern-matching alone accepts a kraken base for a
    VLM run and fails much later, inside transformers."""
    from atr_training.base_models import HF_REPO_RE

    assert HF_REPO_RE.match("10.5281/zenodo.15366732")   # the trap
    with pytest.raises(BaseModelError):
        resolve_base_model("10.5281/zenodo.15366732", "trocr", registry,
                           path_exists=never_exists)


def test_a_bare_word_is_not_a_huggingface_repo(registry):
    with pytest.raises(BaseModelError, match="owner/name"):
        resolve_base_model("qwen3vl", "vllm", registry, path_exists=never_exists)


# ── a missing registry must not break DOI runs ──────────────────────────────
def test_a_doi_resolves_without_any_registry():
    """config/models.yaml being unreadable is about us, not about the request."""
    resolved = resolve_base_model("10.5281/zenodo.15366732", "kraken", None,
                                  path_exists=never_exists)
    assert resolved.ref == "10.5281/zenodo.15366732"


def test_a_registry_id_without_a_registry_says_so_plainly():
    """The name always promised this; the assertion used to check the opposite.

    It matched "not a file, a registry id" — i.e. it pinned the claim that the
    id was invalid, when the actual problem was that there was nothing to look
    it up in. A user reading that goes after a typo that is not theirs.
    """
    with pytest.raises(BaseModelError, match="no registry was available") as err:
        resolve_base_model("kraken-medieval_generic_b", "kraken", None,
                           path_exists=never_exists)
    assert "not a registry id" not in str(err.value)
    assert "none is configured" in str(err.value)


def test_an_unreadable_registry_names_the_reason():
    """The path and the cause, so the fix is obvious from the message alone."""
    why = "/mnt/…/registry/models.yaml: no such file — has the gateway published it?"
    with pytest.raises(BaseModelError) as err:
        resolve_base_model("kraken-early_modern_german", "kraken", None,
                           path_exists=never_exists, registry_error=why)
    assert why in str(err.value)
    assert "no registry was available" in str(err.value)


def test_a_typo_against_a_readable_registry_still_says_typo(registry):
    """The old message is right when there WAS a registry to look in."""
    with pytest.raises(BaseModelError, match="not a file, a registry id"):
        resolve_base_model("kraken-medieval_generic_z", "kraken", registry,
                           path_exists=never_exists)


# ── the published registry, as the box reads it ─────────────────────────────
def test_the_published_registry_resolves_the_id_that_failed():
    """Against a snapshot of the file the gateway publishes (#5).

    This used to read config/models.yaml from this repo. That copy is gone: it
    would have drifted from the gateway's without anyone noticing. The snapshot
    under tests/fixtures is a CONTRACT fixture — it pins the shape, not the
    current list of models.
    """
    from pathlib import Path

    from atr_training.shared_registry import load_shared_registry

    snapshot = Path(__file__).resolve().parent / "fixtures" / "registry" / "models.yaml"
    resolved = resolve_base_model("kraken-medieval_generic_b", "kraken",
                                  load_shared_registry(snapshot),
                                  path_exists=never_exists)
    assert resolved.ref.startswith("10.5281/zenodo.")

"""Reading the registry the gateway publishes to the share (#5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from atr_training.base_models import resolve_base_model
from atr_training.settings import TrainerSettings
from atr_training.shared_registry import (
    RegistryUnavailable,
    SharedRegistry,
    load_shared_registry,
)

SNAPSHOT = Path(__file__).resolve().parent / "fixtures" / "registry" / "models.yaml"


def never_exists(_: str) -> bool:
    return False


# ── the contract with the gateway's published file ──────────────────────────
def test_the_snapshot_parses_without_the_gateways_classes():
    """The trainer reads four fields of some twenty, and ignores the rest.

    Ignoring rather than rejecting is what lets the gateway grow its schema
    without breaking this side. If this fails after a gateway change, the
    change removed or renamed one of the four fields that matter.
    """
    reg = load_shared_registry(SNAPSHOT)
    assert len(reg) > 0
    assert reg.by_engine("kraken"), "the snapshot has no kraken models to fine-tune from"


@pytest.mark.parametrize("model_id", [
    # The two ids real jobs actually named, as of 16.09.2026:
    # 11 jobs with the first, 1 with the second.
    "kraken-early_modern_german",
    "kraken-medieval_generic_b",
])
def test_the_ids_real_jobs_used_still_resolve(model_id):
    resolved = resolve_base_model(model_id, "kraken", load_shared_registry(SNAPSHOT),
                                  path_exists=never_exists)
    assert resolved.kind == "registry"
    assert resolved.ref.startswith("10.5281/zenodo.")


def test_a_disabled_model_is_still_a_valid_base(tmp_path):
    """The reason this is not GET /models, which filters on enabled.

    Disabled for serving on idhefix and a perfectly good starting point for a
    fine-tune are not in conflict.
    """
    f = tmp_path / "models.yaml"
    f.write_text(
        "models:\n"
        "  - id: kraken-retired\n"
        "    engine: kraken\n"
        "    zenodo_id: 10.5281/zenodo.1234567\n"
        "    enabled: false\n"
        "    disabled_reason: superseded for serving\n", encoding="utf-8")
    resolved = resolve_base_model("kraken-retired", "kraken", load_shared_registry(f),
                                  path_exists=never_exists)
    assert resolved.ref == "10.5281/zenodo.1234567"


def test_unknown_fields_are_ignored_not_rejected(tmp_path):
    f = tmp_path / "models.yaml"
    f.write_text(
        "models:\n"
        "  - id: kraken-x\n"
        "    engine: kraken\n"
        "    zenodo_id: 10.5281/zenodo.7654321\n"
        "    a_field_the_gateway_added_next_month: whatever\n", encoding="utf-8")
    assert load_shared_registry(f).get("kraken-x").zenodo_id == "10.5281/zenodo.7654321"


# ── failing loudly, and saying why ──────────────────────────────────────────
def test_a_missing_file_raises_rather_than_returning_an_empty_registry(tmp_path):
    """An empty registry would turn "the file is gone" into "that id is wrong".

    That is the trap load_heldout sets by returning an empty set, and the one
    that nearly reintroduced #98 during the move (#3).
    """
    with pytest.raises(RegistryUnavailable) as err:
        load_shared_registry(tmp_path / "absent.yaml")
    assert "has the gateway published it" in err.value.reason


def test_a_malformed_file_names_itself(tmp_path):
    f = tmp_path / "models.yaml"
    f.write_text("models: [this is: not: valid", encoding="utf-8")
    with pytest.raises(RegistryUnavailable) as err:
        load_shared_registry(f)
    assert err.value.path == f


def test_a_malformed_entry_is_not_silently_dropped(tmp_path):
    f = tmp_path / "models.yaml"
    f.write_text("models:\n  - id: no-engine-here\n", encoding="utf-8")
    with pytest.raises(RegistryUnavailable, match="malformed entry"):
        load_shared_registry(f)


# ── where it is read from ───────────────────────────────────────────────────
def test_the_registry_is_read_from_the_share_not_from_this_repo():
    """A copy in this repo would drift from the gateway's without a word.

    Nothing may reintroduce one: the default is the share, and config/models.yaml
    no longer exists here.
    """
    default = TrainerSettings.model_fields["models_config"].default
    assert str(default).startswith("/mnt/wbkolleg_dh_1/"), default
    repo = Path(__file__).resolve().parents[1]
    assert not (repo / "config" / "models.yaml").exists()


def test_the_resolver_accepts_anything_shaped_like_a_registry():
    """No import of the gateway's Registry: structure is the contract."""
    assert isinstance(load_shared_registry(SNAPSHOT), SharedRegistry)

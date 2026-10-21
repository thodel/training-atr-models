"""#259: the weights digest, measured here, defined by the gateway.

The gateway (serving-atr-inference#200) validates what this repo writes:
``WeightsDigest`` must follow from its file list and carry ``host`` and
``measured_at``, or it does not load. This repo cannot import that code
(``test_isolation.py``), so the two implementations are pinned to one answer
through ``fixtures/weights_digest/contract.json`` — byte-identical in both
repos. The tests below are the rule's load-bearing points, not a copy of the
gateway's suite: the pinned vector, the loader rule, and the three refusals.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from atr_training.weights_digest import (
    WeightFile,
    WeightsDigest,
    WeightsUnreadable,
    combined_digest,
    digest_weights,
)

CONTRACT = Path(__file__).parent / "fixtures" / "weights_digest" / "contract.json"


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# ── the contract both repos hold ────────────────────────────────────────────

def test_the_pinned_vector_still_produces_the_pinned_digest(tmp_path):
    """If this fails, the definition moved — and every recorded digest in both
    repos now means something else. Change the fixture only with that in mind,
    and change it in serving-atr-inference in the same breath."""
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    directory = tmp_path / "pinned"
    for name, content in contract["files"].items():
        _write(directory / name, content.encode("utf-8"))

    digest = digest_weights(directory, host="contract")

    assert digest.sha256 == contract["sha256"]
    assert [f.name for f in digest.files] == contract["hashed"]
    for entry in digest.files:
        assert entry.sha256 == contract["per_file"][entry.name]["sha256"]
        assert entry.size_bytes == contract["per_file"][entry.name]["size_bytes"]


def test_the_fixture_is_the_gateways_byte_for_byte():
    """Not a comparison across repos — that cannot run here — but the fixture
    says so in its own note, and a copy that lost the note is a copy nobody
    will keep in step."""
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert "Byte-identical in training-atr-models" in contract["_note"]


# ── the rule, at the points the gate relies on ──────────────────────────────

def test_a_kraken_file_is_hashed_by_its_basename(tmp_path):
    weights = _write(tmp_path / "deep" / "kraken-a-v1.mlmodel", b"kraken weights")
    digest = digest_weights(weights, host="h")
    assert [f.name for f in digest.files] == ["kraken-a-v1.mlmodel"]
    assert digest.source == str(weights)


def test_the_same_bytes_under_another_mountpoint_give_the_same_digest(tmp_path):
    """asteraix and idhefix mount the share at different paths."""
    a = _write(tmp_path / "mnt_a" / "trained" / "m" / "m.mlmodel", b"same")
    b = _write(tmp_path / "srv_b" / "trained" / "m" / "m.mlmodel", b"same")
    assert digest_weights(a, host="h").sha256 == digest_weights(b, host="h").sha256


def test_one_changed_byte_changes_the_digest(tmp_path):
    before = digest_weights(_write(tmp_path / "m.mlmodel", b"v1"), host="h").sha256
    after = digest_weights(_write(tmp_path / "m.mlmodel", b"v2"), host="h").sha256
    assert before != after


def test_a_directory_hashes_the_coreml_alone_when_it_has_one(tmp_path):
    """What ``kraken_loader.resolve_weights`` serves; a .safetensors beside it
    is a file kraken 7.0.2 cannot load."""
    _write(tmp_path / "model.mlmodel", b"coreml")
    _write(tmp_path / "model.safetensors", b"not served")
    _write(tmp_path / "metadata.json", b"{}")
    assert [f.name for f in digest_weights(tmp_path, host="h").files] == ["model.mlmodel"]


def test_a_sharded_checkpoint_hashes_every_shard_and_no_sidecar(tmp_path):
    _write(tmp_path / "model-00001-of-00002.safetensors", b"a")
    _write(tmp_path / "model-00002-of-00002.safetensors", b"b")
    _write(tmp_path / "config.json", b"{}")
    assert [f.name for f in digest_weights(tmp_path, host="h").files] == [
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]


def test_a_reshard_changes_the_digest_on_purpose(tmp_path):
    one = tmp_path / "one"
    _write(one / "model.safetensors", b"abcd")
    two = tmp_path / "two"
    _write(two / "model-00001-of-00002.safetensors", b"ab")
    _write(two / "model-00002-of-00002.safetensors", b"cd")
    assert digest_weights(one, host="h").sha256 != digest_weights(two, host="h").sha256


# ── the three refusals ──────────────────────────────────────────────────────

def test_a_missing_path_and_a_directory_without_weights_are_refused(tmp_path):
    with pytest.raises(WeightsUnreadable):
        digest_weights(tmp_path / "gone", host="h")
    _write(tmp_path / "only" / "metadata.json", b"{}")
    with pytest.raises(WeightsUnreadable, match="nothing here is weights"):
        digest_weights(tmp_path / "only", host="h")


def test_a_digest_that_does_not_follow_from_its_files_does_not_exist():
    files = [WeightFile(name="m.mlmodel", sha256="a" * 64, size_bytes=1)]
    with pytest.raises(ValidationError, match="does not follow"):
        WeightsDigest(sha256="0" * 64, files=files, host="h", measured_at="t")
    WeightsDigest(sha256=combined_digest(files), files=files, host="h", measured_at="t")


def test_a_digest_over_no_files_or_without_provenance_does_not_exist():
    files = [WeightFile(name="m.mlmodel", sha256="a" * 64, size_bytes=1)]
    with pytest.raises(ValidationError, match="no files"):
        WeightsDigest(sha256=combined_digest([]), files=[], host="h", measured_at="t")
    with pytest.raises(ValidationError):
        WeightsDigest(sha256=combined_digest(files), files=files)  # type: ignore[call-arg]

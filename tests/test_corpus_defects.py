"""A corpus fix does not end a corpus defect — the weights outlive it (#35).

`pagexml.line_texts()` kept the first word of every line and dropped the rest:
28 % of the medieval corpus by characters, and on two 19th-century sources
practically everything after word one. 33f55fc corrected the reading. It did not
correct the three models already trained on the damage, which are still on the
share and still score like ordinary models — the Federal Council benchmark shows
them writing one word and stopping on up to 29.3 % of lines.

So #35 closes on a rule, not on its fix: every model in its table retrained or
explicitly marked superseded. That rule lived in the issue's prose and in
hand-written `notes` on four HuggingFace cards. Here it is data, and
`outstanding()` is the rule itself — which is the difference between a condition
that is checked and one that is remembered.
"""

import json
from pathlib import Path

import pytest

from atr_training.corpus_defects import (
    DEFAULT_REGISTRY,
    STATUSES,
    defects_for,
    load_defects,
    outstanding,
)

ISSUE_35 = "word-segmentation-truncation"


# ── the shipped registry ────────────────────────────────────────────────────
def test_the_shipped_registry_loads():
    assert [d.id for d in load_defects()] == [ISSUE_35]


def test_every_status_is_one_the_code_knows():
    """`pending` is `status == "retrain_pending"`, so a typo anywhere else reads
    as "not pending" and quietly closes the rule it was meant to hold open. This
    is the test that stops that."""
    for defect in load_defects():
        for model in defect.models.values():
            assert model.status in STATUSES, f"{model.model_id}: {model.status}"


def test_a_superseded_model_names_what_replaced_it():
    """"Superseded" without a successor tells a reader to stop using something
    and not what to use instead, which is how a marked model stays in service."""
    for defect in load_defects():
        for model in defect.models.values():
            if model.status == "superseded":
                assert model.successor, model.model_id


def test_the_defect_names_the_commit_that_fixed_it():
    defect = load_defects()[0]

    assert defect.fixed_by["commit"] == "33f55fc"
    assert "33f55fc" in defect.fixed_by_sentence()


# ── the closing rule of #35, as something a test can ask ────────────────────
def test_two_models_are_still_waiting_for_a_retrain():
    """The state of #35 today. When this list empties the issue may close, and
    this test is what will say so."""
    assert outstanding() == {
        ISSUE_35: ["qwen3.5-2b-german-xix-v1", "qwen3.5-4b-german-xix-v1"]}


def test_the_retrained_and_withdrawn_models_are_not_outstanding():
    pending = outstanding()[ISSUE_35]

    assert "qwen3vl-german-xix-v1" not in pending        # v2 exists
    assert "qwen3vl-medieval-german-v1" not in pending   # v3 exists
    assert "qwen3vl-medieval-german-v2" not in pending   # never published


def test_an_all_terminal_registry_is_empty_not_absent(tmp_path: Path):
    """What closing looks like: no pending models, so no entry at all — the
    caller checks `if outstanding()`, not `if outstanding()[id]`."""
    registry = tmp_path / "defects.json"
    registry.write_text(json.dumps({"defects": [{
        "id": "d", "models": {"m": {"status": "superseded", "successor": "m2"}}}]}),
        encoding="utf-8")

    assert outstanding(registry) == {}


# ── reading a model's own record ────────────────────────────────────────────
def test_an_unaffected_model_carries_nothing():
    assert defects_for("qwen3vl-medieval-german-v3") == []


def test_a_pending_model_says_no_retrain_exists():
    (_, affected), = defects_for("qwen3.5-4b-german-xix-v1")

    assert affected.pending is True
    assert "No retrain exists yet" in affected.sentence()


def test_a_superseded_model_points_at_its_successor():
    (_, affected), = defects_for("qwen3vl-german-xix-v1")

    assert affected.pending is False
    assert "qwen3vl-german-xix-v2" in affected.sentence()


def test_a_withdrawn_model_says_the_weights_are_not_published():
    (_, affected), = defects_for("qwen3vl-medieval-german-v2")

    assert "not published" in affected.sentence()


# ── a registry that is not there, or not readable ───────────────────────────
def test_a_missing_registry_is_not_an_error(tmp_path: Path):
    """A publish that dies on an unreadable side file helps nobody."""
    assert load_defects(tmp_path / "nope.json") == []


def test_a_malformed_registry_warns_rather_than_raises(tmp_path: Path, caplog):
    registry = tmp_path / "defects.json"
    registry.write_text("{ not json", encoding="utf-8")

    assert load_defects(registry) == []


def test_an_entry_without_an_id_is_skipped(tmp_path: Path):
    registry = tmp_path / "defects.json"
    registry.write_text(json.dumps({"defects": [{"title": "nameless"},
                                                {"id": "real"}]}), encoding="utf-8")

    assert [d.id for d in load_defects(registry)] == ["real"]


def test_a_model_record_defaults_to_pending_not_to_resolved(tmp_path: Path):
    """An entry somebody added in a hurry, with no status, is unfinished work —
    reading it as resolved would drop it out of the rule silently."""
    registry = tmp_path / "defects.json"
    registry.write_text(json.dumps({"defects": [{"id": "d", "models": {"m": {}}}]}),
                        encoding="utf-8")

    assert outstanding(registry) == {"d": ["m"]}


def test_the_registry_path_resolves_from_the_package(tmp_path: Path):
    assert DEFAULT_REGISTRY.name == "corpus_defects.json"
    assert DEFAULT_REGISTRY.is_file()


# ── what lands on the model card ────────────────────────────────────────────
@pytest.fixture
def card(tmp_path: Path):
    """A card for a model id of the caller's choosing, otherwise the VLM fixture."""
    from tests.test_training_publish import VLM_META, trained_dir

    from atr_training.publish import model_card, repo_id_for, scan_trained

    def _build(model_id: str) -> str:
        meta = {**VLM_META, "model_id": model_id}
        trained_dir(tmp_path, meta, weights="adapter_model.safetensors")
        model = scan_trained(tmp_path, only=[model_id]).models[0]
        return model_card(model, repo_id_for(model_id))

    return _build


def test_an_unaffected_models_card_says_nothing_about_defects(card):
    assert "Known defect in the training corpus" not in card("qwen3vl-8b-thun")


def test_an_affected_models_card_carries_the_warning(card):
    text = card("qwen3.5-4b-german-xix-v1")

    assert "Known defect in the training corpus" in text
    assert "No retrain exists yet" in text
    assert "33f55fc" in text


def test_the_warning_stands_above_the_metrics_it_is_about(card):
    """Below them it would be a footnote to a number a reader has already
    taken — and the number is the thing that is wrong."""
    text = card("qwen3.5-4b-german-xix-v1")

    assert text.index("Known defect") < text.index("## Evaluation")


def test_a_superseded_models_card_points_somewhere(card):
    text = card("qwen3vl-medieval-german-v1")

    assert "qwen3vl-medieval-german-v3" in text


def test_the_card_explains_the_defect_rather_than_only_naming_it(card):
    """A reader who has this page open does not have the issue open."""
    text = card("qwen3.5-2b-german-xix-v1")

    assert "<Word>" in text
    assert "+33.5%" in text

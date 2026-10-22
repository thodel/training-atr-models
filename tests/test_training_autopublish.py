"""When a finished model is uploaded automatically, and when it is not (#88).

Publishing is outward-facing and the hub keeps history, so an unpublish is a
deletion that does not undo the copy someone already pulled. The decision is
therefore separated from the upload, and every refusal states its own reason —
"nothing happened" is the worst possibleanswer for an automatic action.
"""

import pytest

from atr_training.autopublish import TOKEN_ENV, decide

#: Shaped like a real one: `hf_` plus 34 characters. `hf_xxx` is not a token, and
#: since the placeholder incident the code says so.
TOKEN = {"HF_TOKEN": "hf_" + "A" * 34}


class TestTheThreshold:
    def test_a_model_above_the_threshold_is_published(self):
        d = decide(83.0, 80.0, "dh-unibe", TOKEN)
        assert d.publish is True and d.org == "dh-unibe"

    def test_exactly_at_the_threshold_counts(self):
        assert decide(80.0, 80.0, "dh-unibe", TOKEN).publish is True

    def test_below_it_does_not(self):
        d = decide(76.76, 80.0, "dh-unibe", TOKEN)
        assert d.publish is False and "76.76" in d.reason and "80.00" in d.reason

    def test_the_real_vlm_run_would_not_have_been_published(self):
        """qwen3vl-german-medieval-v1 scored 76.76 % (CER 0.2324)."""
        assert decide(76.76, 80.0, "dh-unibe", TOKEN).publish is False


class TestRefusals:
    def test_off_by_default(self):
        """A threshold of 0 must never publish, whatever the score."""
        assert decide(99.9, 0.0, "dh-unibe", TOKEN).publish is False

    def test_a_negative_threshold_does_not_mean_always(self):
        assert decide(10.0, -1.0, "dh-unibe", TOKEN).publish is False

    def test_an_unmeasured_run_is_never_published(self):
        """A model whose score is unknown is not a model that passed."""
        d = decide(None, 80.0, "dh-unibe", TOKEN)
        assert d.publish is False and "no char_accuracy" in d.reason

    def test_a_missing_token_is_reported_not_discovered_mid_upload(self):
        d = decide(95.0, 80.0, "dh-unibe", {})
        assert d.publish is False
        assert "no token" in d.reason and "publish_to_hub.py" in d.reason

    @pytest.mark.parametrize("name", TOKEN_ENV)
    def test_either_token_variable_works(self, name):
        assert decide(95.0, 80.0, "dh-unibe", {name: "hf_" + "A" * 34}).publish is True

    def test_a_blank_token_does_not_count(self):
        assert decide(95.0, 80.0, "dh-unibe", {"HF_TOKEN": "   "}).publish is False


class TestPrivacy:
    def test_an_automatic_publish_is_always_private(self):
        """Automation gets less latitude than the human at the shell, not more:
        making a trained model public stays a deliberate, separate decision."""
        assert decide(99.0, 80.0, "dh-unibe", TOKEN).private is True

    def test_a_refusal_reads_as_a_sentence_because_it_lands_on_the_job_record(self):
        assert str(decide(50.0, 80.0, "o", TOKEN)).startswith("skip: ")
        assert str(decide(90.0, 80.0, "o", TOKEN)).startswith("publish: ")


class TestTokenShape:
    """`HF_TOKEN=...` was really set on the box — the placeholder from a command
    that was pasted without substituting it. "Non-empty" accepted it, and the
    upload would have failed inside huggingface_hub instead of here."""

    @pytest.mark.parametrize("value", ["...", "xxx", "<your-token>", "TODO", "hf_"])
    def test_a_placeholder_is_not_a_token(self, value):
        d = decide(95.0, 80.0, "o", {"HF_TOKEN": value})
        assert d.publish is False and "does not look like" in d.reason

    def test_a_real_looking_token_is_accepted(self):
        assert decide(95.0, 80.0, "o", {"HF_TOKEN": "hf_" + "a" * 34}).publish is True

    def test_the_reason_never_contains_the_value(self):
        """It is logged and written to the job record."""
        secret = "hf_" + "S3CR3T" * 6
        reason = decide(95.0, 80.0, "o", {"HF_TOKEN": secret[:5]}).reason
        assert "S3CR3T" not in reason and secret[:5] not in reason

    def test_a_valid_token_in_the_second_variable_still_works(self):
        env = {"HF_TOKEN": "...", "HUGGINGFACE_HUB_TOKEN": "hf_" + "b" * 34}
        assert decide(95.0, 80.0, "o", env).publish is True


class TestTheStoredToken:
    """The token on asteraix lives where `hf auth login` put it, not in the
    service's environment — and `HfApi()` reads it from there by itself. A check
    that knew only the environment refused every upload with "no token is set"
    on a box that could upload (22.10.2026, #33). The stored token counts, by the
    same shape rule, and its value never appears either."""

    def test_a_stored_token_is_enough(self):
        stored = lambda: "hf_" + "c" * 34  # noqa: E731
        assert decide(95.0, 80.0, "o", {}, stored_token=stored).publish is True

    def test_a_stored_placeholder_is_still_not_a_token(self):
        secret = "hf_S3CR3T"
        d = decide(95.0, 80.0, "o", {}, stored_token=lambda: secret)
        assert d.publish is False
        assert "S3CR3T" not in d.reason

    def test_neither_place_names_both_places(self):
        d = decide(95.0, 80.0, "o", {}, stored_token=lambda: None)
        assert d.publish is False
        assert "hf auth login" in d.reason

    def test_an_explicit_environment_does_not_reach_into_this_machine(self, monkeypatch):
        """A caller that passes its own environment gets exactly that — otherwise
        these tests would pass or fail by what the developer's laptop holds."""
        import atr_training.autopublish as ap

        def boom():
            raise AssertionError("the store was read although an environment was given")
        monkeypatch.setattr(ap, "_hub_stored_token", boom)

        assert decide(95.0, 80.0, "o", {}).publish is False

    def test_the_real_run_consults_the_store(self, monkeypatch):
        """No environment given is the trainer's own call: os.environ, then the store."""
        import atr_training.autopublish as ap

        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
        monkeypatch.setattr(ap, "_hub_stored_token", lambda: "hf_" + "d" * 34)

        assert decide(95.0, 80.0, "o").publish is True

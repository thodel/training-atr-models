"""The suite never writes to the Hugging Face Hub (tests/conftest.py, "the Hub").

On 17.11.2026 a run of the suite on asteraix left 17 private repos under
dh-unibe — qwen3vl-thun-v1, kept-asteraix-completed, qwen3vl-expired-while-
queued, … — each with 1 to 7 bytes of "weights". The path was the runner's
auto-publish: asteraix's `.env` sets ATR_TRAIN_AUTO_PUBLISH_MIN_ACCURACY=79, the
fake runners score 95 %, and `decide()` found the token `hf auth login` stored.
The upload's exception was swallowed by ``_maybe_publish``, so every test stayed
green.

These tests replay that run and show each layer of the guard on its own, and
that an attempt someone swallows still fails the test that made it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from conftest import HF_TOKEN_ENV, NO_HF_TOKEN_FILE, HubCallInTest, hub_write_names

from atr_training import autopublish
from atr_training.jobstore import JobStore
from atr_training.publish import HubUploader
from atr_training.settings import TrainerSettings

#: What asteraix's `.env` said on 17.11.2026 (docs/INFRASTRUCTURE.md, #33).
ASTERAIX_ENV = "ATR_TRAIN_AUTO_PUBLISH_MIN_ACCURACY=79\nATR_TRAIN_AUTO_PUBLISH_ORG=dh-unibe\n"
A_TOKEN = "hf_" + "x" * 34


def _settings(tmp_path: Path) -> TrainerSettings:
    """Built the way test_host_owned_jobs.py built it on 17.11 — no
    ``_env_file=None``, so a `.env` in the working directory is read."""
    (tmp_path / "registry").mkdir()
    return TrainerSettings(
        jobs_root=tmp_path / "training", trained_root=tmp_path / "trained",
        registry_root=tmp_path / "registry", checkpoint_root=tmp_path / "scratch",
        ketos=tmp_path / "ketos", min_free_disk_gb=0, artefact_cache=False,
        artefact_cache_root=tmp_path / "artefacts")


def _run_to_completed(settings: TrainerSettings):
    """test_the_runner_keeps_the_host_when_it_saves, the asteraix arm: the
    test that made dh-unibe/kept-asteraix-completed."""
    from test_train_svc_pipeline import FakeRunner, FakeSource, request_with

    from kraken_train_svc.runner import Pipeline

    store = JobStore(settings.jobs_root)
    job = store.create(request_with(model_id="kept-asteraix-completed"), host="asteraix")
    done = Pipeline(store, settings, runner=FakeRunner(),
                    source=FakeSource({"train": 4, "eval": 2})).execute(job.id)
    assert done.status == "completed", done.error
    assert done.metrics.char_accuracy >= 79   # it would clear asteraix's threshold
    return done


# ── the run of 17.11.2026, replayed ─────────────────────────────────────────
def test_asteraix_s_env_no_longer_turns_auto_publish_on(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(ASTERAIX_ENV, encoding="utf-8")

    settings = _settings(tmp_path)

    assert settings.auto_publish_min_accuracy == 0


def test_the_pipeline_that_made_kept_asteraix_completed_publishes_nothing(
        tmp_path, monkeypatch, hub_attempts):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(ASTERAIX_ENV, encoding="utf-8")
    # The stored token asteraix had: present, well-formed.
    monkeypatch.setattr(autopublish, "_hub_stored_token", lambda: A_TOKEN)

    done = _run_to_completed(_settings(tmp_path))

    assert done.published.startswith("skip: auto-publish is off"), done.published
    assert hub_attempts == []


def test_with_the_threshold_forced_on_the_upload_is_refused_and_caught(
        tmp_path, monkeypatch, hub_attempts):
    """The second layer, with the first one taken away on purpose: a test that
    sets the threshold itself still cannot reach the hub — and the attempt is
    recorded, which is what fails a test that does this by accident."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATR_TRAIN_AUTO_PUBLISH_MIN_ACCURACY", "79")
    monkeypatch.setattr(autopublish, "_hub_stored_token", lambda: A_TOKEN)

    done = _run_to_completed(_settings(tmp_path))

    assert "HubCallInTest" in done.published, done.published
    assert hub_attempts == ["atr_training.publish.HubUploader()"]
    hub_attempts.clear()  # provoked on purpose; see the next test for the rest


def test_an_attempt_that_is_swallowed_still_fails_the_test(tmp_path):
    """``_maybe_publish`` catches BaseException. A refusal alone would leave
    the test green — the teardown check is what makes it red. Run as its own
    pytest, with this suite's conftest, so the red is the inner test's."""
    import subprocess

    here = Path(__file__).resolve().parent
    root = here.parent
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "conftest.py").write_text(
        (here / "conftest.py").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "test_inner.py").write_text(
        "from atr_training.publish import HubUploader\n\n"
        "def test_swallows_it():\n"
        "    try:\n"
        "        HubUploader()\n"
        "    except BaseException:\n"
        "        pass\n", encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(root / "src"), str(root), str(root / "engines")]))

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)

    out = result.stdout + result.stderr
    assert result.returncode == 1, out
    assert "1 passed, 1 error" in out, out
    assert "reached for the Hugging Face Hub: atr_training.publish.HubUploader()" in out, out


# ── each layer on its own ───────────────────────────────────────────────────
def test_a_token_a_test_sets_is_gone_for_the_next(monkeypatch):
    """Set here, and it works here; the next test sees none of it."""
    monkeypatch.setenv("HF_TOKEN", A_TOKEN)
    assert autopublish.decide(95.0, 79.0, "dh-unibe").publish is True


def test_no_token_variable_reaches_a_test():
    for name in HF_TOKEN_ENV:
        assert name not in os.environ, name
    assert os.environ["HF_TOKEN_PATH"] == NO_HF_TOKEN_FILE


def test_the_uploader_refuses_to_be_built(hub_attempts):
    with pytest.raises(HubCallInTest, match="never writes to the hub"):
        HubUploader(token=A_TOKEN)
    assert hub_attempts == ["atr_training.publish.HubUploader()"]
    hub_attempts.clear()


def test_huggingface_hub_has_no_stored_token():
    hf = pytest.importorskip("huggingface_hub")
    from huggingface_hub import constants

    assert constants.HF_TOKEN_PATH == NO_HF_TOKEN_FILE
    assert hf.get_token() is None


def test_every_write_on_the_hub_raises(hub_attempts, tmp_path):
    hf = pytest.importorskip("huggingface_hub")
    from huggingface_hub import HfApi, hf_api

    names = hub_write_names(hf_api.HfApi)
    # The three the auto-publish uses, and the two that would clean up after it.
    for needed in ("create_repo", "upload_folder", "create_commit",
                   "delete_repo", "update_repo_settings"):
        assert needed in names, needed

    api = HfApi(token=A_TOKEN)
    calls = [
        lambda: api.create_repo("dh-unibe/qwen3vl-thun-v1", private=True),
        lambda: api.upload_folder(repo_id="dh-unibe/qwen3vl-thun-v1", folder_path=tmp_path),
        lambda: api.create_commit("dh-unibe/qwen3vl-thun-v1", operations=[],
                                  commit_message="x"),
        lambda: hf.create_repo("dh-unibe/qwen3vl-thun-v1", private=True),
        lambda: hf.upload_folder(repo_id="dh-unibe/qwen3vl-thun-v1", folder_path=tmp_path),
    ]
    for call in calls:
        with pytest.raises(HubCallInTest):
            call()
    assert hub_attempts == ["HfApi.create_repo", "HfApi.upload_folder",
                            "HfApi.create_commit", "HfApi.create_repo",
                            "HfApi.upload_folder"]
    hub_attempts.clear()


def test_reads_are_left_alone():
    """Only writes are refused: a listing leaves nothing behind, and the tests
    that need one fake it already."""
    pytest.importorskip("huggingface_hub")
    from huggingface_hub import hf_api

    names = hub_write_names(hf_api.HfApi)
    for read in ("list_repo_files", "model_info", "dataset_info", "hf_hub_download",
                 "snapshot_download", "repo_exists", "whoami"):
        assert read not in names, read


def test_the_guard_does_not_need_huggingface_hub(monkeypatch, hub_attempts):
    """The suite's venv has no huggingface_hub; the trainer's door is shut there too."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(HubCallInTest):
        HubUploader()
    hub_attempts.clear()

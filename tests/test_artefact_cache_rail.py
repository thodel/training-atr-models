"""The rail conftest.py puts up, checked rather than assumed (#4).

`conftest.py` points every TrainerSettings at a throwaway cache root through the
environment. Until now that was only a fixture — something the suite *does*, not
something it *checks*. A fixture that stops working fails nothing: the tests
still pass, and the developer's home fills up. That is not hypothetical: the
first time the artefact cache ran under the suite it left five directories in
$HOME.

The VLM pipeline is where it matters most. Its settings fixture deliberately
does not set artefact_cache_root, and since the VLM backend joined the cache it
stores a whole compiled corpus there on every run.
"""

from __future__ import annotations

from pathlib import Path

from atr_training.settings import TrainerSettings
from test_vlm_train_pipeline import FakeRunner, FakeSource, run_pipeline  # noqa: F401
from test_vlm_train_pipeline import settings, store  # noqa: F401  (fixtures)

HOME_CACHE = Path.home() / "atr-cache" / "artefacts"


def _listing(path: Path) -> tuple[bool, frozenset[str]]:
    if not path.is_dir():
        return False, frozenset()
    return True, frozenset(p.name for p in path.iterdir())


def test_without_the_rail_the_default_would_be_in_home():
    """The test below is only meaningful if the rail is actually needed."""
    default = TrainerSettings.model_fields["artefact_cache_root"].default
    assert Path(default) == HOME_CACHE


def test_settings_built_in_a_test_point_away_from_home(tmp_path_factory):
    root = TrainerSettings().artefact_cache_root.resolve()
    assert root != HOME_CACHE.resolve()
    assert tmp_path_factory.getbasetemp().resolve() in root.parents


def test_a_full_pipeline_run_stores_its_corpus_under_the_rail(store, settings,  # noqa: F811
                                                              tmp_path_factory):
    """End to end: the cache is really written, and really not in $HOME."""
    before = _listing(HOME_CACHE)

    runner = FakeRunner()
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}), runner)
    assert job.status == "completed", job.error

    train = runner.command("train")
    corpus = Path(train[train.index("--train-jsonl") + 1]).resolve()
    assert tmp_path_factory.getbasetemp().resolve() in corpus.parents, (
        f"the compiled corpus was stored at {corpus}, outside the suite's tmp root")
    assert HOME_CACHE.resolve() not in corpus.parents

    assert _listing(HOME_CACHE) == before, (
        f"the run changed {HOME_CACHE} — the rail in conftest.py is not holding")

"""The gateway fixtures against the gateway's own ``main`` (#151).

Two copies of serving-atr-inference live under ``tests/fixtures/registry/``:
``modelspec_fields.txt`` (what ``Registration`` has to match, field for field)
and ``models.yaml`` (the curated registry the base-model tests resolve against).
Both went stale without a word — the only check was the copy itself. These
tests read the original: the parsing on inline samples everywhere, and the
comparison with ``main`` wherever GitHub is reachable, **skipped** elsewhere
(CI has the network; a laptop on the train does not), never passed in silence.
When they fail, ``scripts/refresh_gateway_fixtures.py`` rewrites both files and
``tests/test_registration.py`` then says what ``Registration`` has to gain.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atr_training.gateway_drift import (
    GatewayUnreachable,
    fetch_gateway_main,
    modelspec_fields,
    registry_entries,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "registry"

SAMPLE_SOURCE = '''
from pydantic import BaseModel, Field

class VramMeasurement(BaseModel):
    host: str
    mib: int = 0

class ModelSpec(BaseModel):
    """The gateway's."""

    model_config = ConfigDict(extra="forbid")

    id: str
    engine: Engine
    description: str | None = None
    vram_mb: int | None = Field(default=None, gt=0)
    vram_measured: VramMeasurement | None = None

    @property
    def is_vllm(self) -> bool:
        return self.engine == "vllm"

    @field_validator("id")
    @classmethod
    def _id(cls, v):
        local: str = "not a field"
        return v
'''

SAMPLE_YAML = """# header
models:
  # - id: commented-out
  #   engine: kraken
  - id: kraken-prima
    engine: kraken   # was vllm
    description: "PRIMA HTR"
    zenodo_id: 10.5281/zenodo.1
    vram_measured:
      host: idhefix
      engine: other   # a nested key of the same name is not the entry's
  - id: 'qwen3vl-german-xix-v2'
    engine: "vllm"
    base_model: Qwen/Qwen3-VL-8B-Instruct
"""


# ── the parsing, offline ─────────────────────────────────────────────────────
def test_the_field_list_is_the_annotated_class_body_in_order():
    assert modelspec_fields(SAMPLE_SOURCE) == \
        ["id", "engine", "description", "vram_mb", "vram_measured"]


def test_another_class_s_fields_are_not_modelspec_s():
    assert modelspec_fields(SAMPLE_SOURCE, "VramMeasurement") == ["host", "mib"]
    with pytest.raises(ValueError, match="no class Missing"):
        modelspec_fields(SAMPLE_SOURCE, "Missing")


def test_registry_entries_are_id_to_engine_with_quotes_and_comments_stripped():
    assert registry_entries(SAMPLE_YAML) == {"kraken-prima": "kraken",
                                             "qwen3vl-german-xix-v2": "vllm"}


def test_a_file_without_entries_is_not_the_registry():
    with pytest.raises(ValueError, match="not the gateway's models.yaml"):
        registry_entries("models: []\n")


def test_the_fixtures_parse_the_way_main_will():
    """The two local files go through the same parsers as the fetched ones, so
    the comparison below compares like with like."""
    fields = [line.strip() for line in
              (FIXTURES / "modelspec_fields.txt").read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.startswith("#")]
    assert "id" in fields and "weights_digest" in fields
    entries = registry_entries((FIXTURES / "models.yaml").read_text(encoding="utf-8"))
    assert entries["kraken-prima"] == "kraken"
    assert all(engine for engine in entries.values()), "an entry without an engine"


# ── the refresh script, on a stubbed main ────────────────────────────────────
def _script():
    import sys
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    import importlib
    return importlib.import_module("scripts.refresh_gateway_fixtures")


def _stub_main(monkeypatch, script, source: str, models_yaml: str):
    from atr_training.gateway_drift import GatewayMain
    monkeypatch.setattr(script, "fetch_gateway_main", lambda: GatewayMain(
        commit="abcdef0123456789", date="2026-10-23T10:00:00Z",
        registry_source=source, models_yaml=models_yaml))


def _as_source(fields: list[str]) -> str:
    return "class ModelSpec(BaseModel):\n" + "".join(f"    {f}: str\n" for f in fields)


def test_check_says_ok_when_both_fixtures_match_main(monkeypatch, capsys, tmp_path):
    script = _script()
    fields = script.current_fields()
    models = (FIXTURES / "models.yaml").read_text(encoding="utf-8")
    _stub_main(monkeypatch, script, _as_source(fields), models)
    assert script.main(["--check"]) == 0
    assert "ok    fixtures match main" in capsys.readouterr().out


def test_check_names_the_drift_and_exits_1(monkeypatch, capsys):
    script = _script()
    fields = script.current_fields() + ["a_new_field"]
    models = (FIXTURES / "models.yaml").read_text(encoding="utf-8").replace(
        "- id: kraken-prima\n", "- id: kraken-prima-renamed\n")
    _stub_main(monkeypatch, script, _as_source(fields), models)
    assert script.main(["--check"]) == 1
    out = capsys.readouterr().out
    assert "DRIFT fields" in out and "a_new_field" in out
    assert "DRIFT registry" in out and "kraken-prima-renamed" in out and "'kraken-prima'" in out


def test_an_unreachable_gateway_is_a_skip_not_a_pass(monkeypatch, capsys):
    script = _script()

    def down():
        raise GatewayUnreachable("no route")
    monkeypatch.setattr(script, "fetch_gateway_main", down)
    assert script.main(["--check"]) == 2
    assert "SKIP" in capsys.readouterr().out


def test_the_rewrite_puts_the_commit_and_date_in_both_headers(monkeypatch, tmp_path):
    script = _script()
    monkeypatch.setattr(script, "FIELDS_FILE", tmp_path / "modelspec_fields.txt")
    monkeypatch.setattr(script, "MODELS_FILE", tmp_path / "models.yaml")
    (tmp_path / "modelspec_fields.txt").write_text("# old\nid\n", encoding="utf-8")
    (tmp_path / "models.yaml").write_text("models:\n  - id: old\n    engine: kraken\n", encoding="utf-8")
    _stub_main(monkeypatch, script, _as_source(["id", "engine"]), SAMPLE_YAML)
    assert script.main([]) == 0
    fields = (tmp_path / "modelspec_fields.txt").read_text(encoding="utf-8")
    assert "at abcdef0 (main, 2026-10-23)" in fields and fields.endswith("id\nengine\n")
    models = (tmp_path / "models.yaml").read_text(encoding="utf-8")
    assert models.startswith("# Copy of serving-atr-inference config/models.yaml at abcdef0 (main, 2026-10-23)")
    assert models.endswith(SAMPLE_YAML), "the gateway's file, verbatim, under the header"
    assert registry_entries(models) == registry_entries(SAMPLE_YAML)


# ── the comparison, where GitHub answers ─────────────────────────────────────
@pytest.fixture(scope="module")
def gateway_main():
    try:
        return fetch_gateway_main()
    except GatewayUnreachable as exc:
        pytest.skip(f"serving-atr-inference/main is not reachable from here: {exc}")


def test_the_pinned_field_list_is_the_gateway_s_current_one(gateway_main):
    """Fails when the gateway's ``ModelSpec`` gains, loses or reorders a field:
    run ``scripts/refresh_gateway_fixtures.py`` and then ``test_registration``
    says what ``Registration`` has to change (#80, #151 — twice unnoticed)."""
    pinned = [line.strip() for line in
              (FIXTURES / "modelspec_fields.txt").read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.startswith("#")]
    live = modelspec_fields(gateway_main.registry_source)
    assert pinned == live, (
        f"modelspec_fields.txt differs from serving-atr-inference@{gateway_main.commit[:7]}: "
        f"missing here {sorted(set(live) - set(pinned))}, gone there "
        f"{sorted(set(pinned) - set(live))} — run scripts/refresh_gateway_fixtures.py")


def test_the_registry_snapshot_names_the_ids_the_gateway_serves(gateway_main):
    """Fails when the curated registry renames, adds or drops an id, so a base
    the tests resolve against is one the gateway still publishes (serving#198
    renamed two that real jobs had used, and nothing here noticed)."""
    pinned = registry_entries((FIXTURES / "models.yaml").read_text(encoding="utf-8"))
    live = registry_entries(gateway_main.models_yaml)
    assert pinned == live, (
        f"models.yaml differs from serving-atr-inference@{gateway_main.commit[:7]}: "
        f"only here {sorted(set(pinned) - set(live))}, only there "
        f"{sorted(set(live) - set(pinned))}, engine changed "
        f"{sorted(k for k in set(pinned) & set(live) if pinned[k] != live[k])} — run "
        "scripts/refresh_gateway_fixtures.py")

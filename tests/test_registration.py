"""Registering a trained model as one file on the share (#14).

The reader of these files is the gateway (serving-atr-inference#138,
``atr_serving/shared_registry.py``). Its rules are restated in
:func:`gateway_problems` rather than imported — the two repos share a file
format, not code — and ``fixtures/registry/trained/example.yaml`` is the
gateway's own example, copied verbatim from serving-atr-inference main
(unchanged since ac1b42f, 16.09.2026). If the gateway changes that file, copy it
again: the contract tests below say what else has to follow.
"""

from __future__ import annotations

import importlib
import io
import os
import threading
from pathlib import Path

import pytest
import yaml

from atr_training import registration
from atr_training.contracts import DatasetSpec, Metrics, TrainRequest
from atr_training.jobstore import JobStore
from atr_training.registration import (
    Registration,
    RegistrationError,
    is_registration_name,
    read_registration,
    registration_path,
    set_enabled,
    trained_dir,
    write_registration,
)
from atr_training.settings import TrainerSettings

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "registry" / "trained" / "example.yaml"
LIVE_ROOT = Path("/mnt/wbkolleg_dh_1/Textrecognition_Training/registry")


def spec(model_id: str = "kraken-thun-v1", **kw) -> dict:
    """What the kraken runner writes, give or take the path."""
    return {
        "id": model_id,
        "engine": "kraken",
        "local_path": f"/mnt/share/trained/{model_id}/{model_id}.mlmodel",
        "enabled": False,
        "task": "htr",
        "level": "page",
        **kw,
    }


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A registry root that exists, as the gateway leaves it: no trained/ yet."""
    r = tmp_path / "registry"
    r.mkdir()
    return r


def gateway_problems(path: Path) -> list[str]:
    """Why the gateway would skip ``path`` — its rules, restated. [] = served.

    ``_is_registration``, then ``_read_one``: a mapping, a valid entry, the file
    named after the id, and an absolute ``local_path``.
    """
    problems = []
    if not is_registration_name(path.name):
        problems.append(f"{path.name} is not a name the gateway reads")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return problems + [f"not ONE mapping but {type(raw).__name__}"]
    try:
        Registration.model_validate(raw)
    except ValueError as exc:
        problems.append(f"not a valid entry: {exc}")
    if raw.get("id") != path.stem:
        problems.append(f"declares id {raw.get('id')!r} but is named {path.name}")
    local = raw.get("local_path")
    if local is not None and not Path(local).is_absolute():
        problems.append(f"relative local_path {local!r}")
    return problems


def listing(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


# ── one file per model ──────────────────────────────────────────────────────
def test_a_registration_is_one_file_per_model(root):
    """Not one shared file rewritten per model: that is what lost updates
    between two writers, and what the gateway no longer reads from the share."""
    for model_id in ("kraken-a-v1", "kraken-b-v1", "kraken-c-v1"):
        write_registration(root, spec(model_id))

    assert listing(trained_dir(root)) == ["kraken-a-v1.yaml", "kraken-b-v1.yaml",
                                          "kraken-c-v1.yaml"]
    for path in trained_dir(root).iterdir():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(raw, dict), f"{path.name} holds {type(raw).__name__}, not ONE model"
        assert "models" not in raw
        assert raw["id"] == path.stem


def test_writing_an_id_again_replaces_its_file_and_no_other(root):
    """The overlay's upsert-by-id, one file per id."""
    write_registration(root, spec("kraken-a-v1"))
    write_registration(root, spec("kraken-b-v1"))
    write_registration(root, spec("kraken-a-v1", languages=["de"]))

    assert listing(trained_dir(root)) == ["kraken-a-v1.yaml", "kraken-b-v1.yaml"]
    assert read_registration(root, "kraken-a-v1").languages == ["de"]
    assert read_registration(root, "kraken-b-v1").languages == []


def test_the_file_name_is_the_model_id(root):
    """The gateway skips a file whose id differs from its name — so the name is
    derived from the id and cannot be passed in."""
    path = write_registration(root, spec("kraken-thun-missiven-v1"))

    assert path == root / "trained" / "kraken-thun-missiven-v1.yaml"
    assert path == registration_path(root, "kraken-thun-missiven-v1")
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["id"] == path.stem
    assert gateway_problems(path) == []


@pytest.mark.parametrize("bad_id", [".hidden", "../escape", "a/b", "Upper", ""])
def test_an_id_that_cannot_be_its_own_file_name_is_refused(root, bad_id):
    """A leading dot is a registration the gateway never reads; a slash is a
    file somewhere else."""
    with pytest.raises(RegistrationError, match="must match"):
        write_registration(root, spec(bad_id))
    assert not trained_dir(root).exists()


# ── atomic ──────────────────────────────────────────────────────────────────
def test_a_registration_is_written_atomically(root, monkeypatch):
    """At no moment is a partial file visible under a name the gateway reads.

    Checked at the one moment it could be: just before the rename. The target
    must still hold the old, complete registration, and the only other file must
    be a sibling tmp whose name the gateway ignores.
    """
    old = write_registration(root, spec("kraken-a-v1", languages=["old"]))
    old_text = old.read_text(encoding="utf-8")
    real_replace = os.replace
    seen = []

    def watching_replace(src, dst):
        src, dst = Path(src), Path(dst)
        if dst.parent == trained_dir(root):
            readable = [n for n in listing(dst.parent) if is_registration_name(n)]
            seen.append({
                "same_dir": src.parent == dst.parent,
                "tmp_ignored": not is_registration_name(src.name),
                "tmp_named": src.name.startswith(".") and src.name.endswith(".tmp"),
                "readable": readable,
                "target_then": dst.read_text(encoding="utf-8"),
                "tmp_complete": yaml.safe_load(src.read_text(encoding="utf-8"))["languages"],
            })
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", watching_replace)
    write_registration(root, spec("kraken-a-v1", languages=["new"]))

    assert len(seen) == 1, "the registration was not written through a rename"
    moment = seen[0]
    assert moment["same_dir"], "a rename across directories is not atomic (or not allowed)"
    assert moment["tmp_ignored"] and moment["tmp_named"]
    assert moment["readable"] == ["kraken-a-v1.yaml"]
    assert moment["target_then"] == old_text
    assert moment["tmp_complete"] == ["new"]
    assert read_registration(root, "kraken-a-v1").languages == ["new"]
    assert listing(trained_dir(root)) == ["kraken-a-v1.yaml"], "a tmp file was left behind"


def test_a_failed_rename_leaves_the_old_registration_and_no_tmp(root, monkeypatch):
    old = write_registration(root, spec("kraken-a-v1", languages=["old"]))
    old_text = old.read_text(encoding="utf-8")

    def failing_replace(src, dst):
        raise OSError(112, "Host is down")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(RegistrationError, match="Host is down"):
        write_registration(root, spec("kraken-a-v1", languages=["new"]))

    assert old.read_text(encoding="utf-8") == old_text
    assert listing(trained_dir(root)) == ["kraken-a-v1.yaml"]


def test_two_concurrent_registrations_do_not_lose_each_other(root, monkeypatch):
    """Two writers, held at the rename until both have written their tmp file.

    The interleaving that loses an update — both read, both write, the second
    replace wins — is forced rather than hoped for. A shared file loses one
    model here; a shared tmp name renames one writer's bytes under the other's
    name and fails the second rename.
    """
    trained_dir(root).mkdir()
    both_written = threading.Barrier(2, timeout=10)
    real_replace = os.replace

    def held_replace(src, dst):
        if Path(dst).parent == trained_dir(root):
            both_written.wait()
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", held_replace)
    errors: list[BaseException] = []

    def register(model_id: str) -> None:
        try:
            write_registration(root, spec(model_id))
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=register, args=(m,))
               for m in ("kraken-a-v1", "kraken-b-v1")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert errors == []
    assert listing(trained_dir(root)) == ["kraken-a-v1.yaml", "kraken-b-v1.yaml"]
    for model_id in ("kraken-a-v1", "kraken-b-v1"):
        assert read_registration(root, model_id).id == model_id


# ── the promotion gate's write ──────────────────────────────────────────────
def test_promotion_rewrites_only_its_own_file(root):
    """The gate proves one model; it may advertise that one and touch nothing
    else — not the other registrations, and not the other fields of this one."""
    write_registration(root, spec("kraken-a-v1", languages=["de"], centuries=[16]))
    other = write_registration(root, spec("kraken-b-v1"))
    # Pushed into the past, so a rewrite would show even on a coarse clock.
    os.utime(other, ns=(1_000_000_000, 1_000_000_000))
    other_before = (other.read_bytes(), other.stat().st_mtime_ns, other.stat().st_ino)

    assert set_enabled(root, "kraken-a-v1", True) is True

    promoted = read_registration(root, "kraken-a-v1")
    assert promoted.enabled is True
    assert promoted.model_dump(exclude={"enabled"}) == Registration.model_validate(
        spec("kraken-a-v1", languages=["de"], centuries=[16])).model_dump(exclude={"enabled"})
    assert (other.read_bytes(), other.stat().st_mtime_ns, other.stat().st_ino) == other_before
    assert listing(trained_dir(root)) == ["kraken-a-v1.yaml", "kraken-b-v1.yaml"]


def test_promoting_a_model_without_a_registration_reports_it(root):
    write_registration(root, spec("kraken-a-v1"))
    assert set_enabled(root, "kraken-ghost-v1", True) is False
    assert listing(trained_dir(root)) == ["kraken-a-v1.yaml"]


def test_promotion_keeps_a_field_written_by_hand(root):
    """Read as it is, not rebuilt: a hand-added field survives the gate."""
    path = write_registration(root, spec("kraken-a-v1"))
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["training_datasets"] = ["10.5281/zenodo.1234567"]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    set_enabled(root, "kraken-a-v1", True)
    assert read_registration(root, "kraken-a-v1").training_datasets == ["10.5281/zenodo.1234567"]


def test_promotion_refuses_a_file_not_named_after_its_id(root):
    path = write_registration(root, spec("kraken-a-v1"))
    path.write_text(yaml.safe_dump(spec("kraken-z-v1")), encoding="utf-8")
    with pytest.raises(RegistrationError, match="declares id 'kraken-z-v1'"):
        set_enabled(root, "kraken-a-v1", True)


# ── refused before writing ──────────────────────────────────────────────────
def test_a_relative_local_path_is_refused_before_writing(root):
    """The gateway skips it — a relative path names a different file on every
    machine. Here it never reaches the share at all, not even as trained/."""
    with pytest.raises(RegistrationError, match="relative"):
        write_registration(root, spec(local_path="trained/kraken-thun-v1/x.mlmodel"))
    assert listing(root) == []


def test_a_spec_still_needs_some_source(root):
    """The gateway's own rule, carried over from the overlay's tests."""
    bare = spec()
    del bare["local_path"]
    with pytest.raises(RegistrationError, match="hf_repo, zenodo_id or local_path"):
        write_registration(root, bare)
    assert listing(root) == []


def test_a_field_the_gateway_does_not_know_is_refused(root):
    """The gateway would drop it without a word; a misspelt `enabled` would be
    a model that is never advertised."""
    with pytest.raises(RegistrationError, match="enabeld"):
        write_registration(root, spec(enabeld=True))
    assert listing(root) == []


def test_a_list_is_not_a_registration(root):
    trained_dir(root).mkdir()
    path = registration_path(root, "kraken-a-v1")
    path.write_text(yaml.safe_dump([spec("kraken-a-v1")]), encoding="utf-8")
    with pytest.raises(RegistrationError, match="ONE model as a mapping"):
        read_registration(root, "kraken-a-v1")


def test_a_missing_registration_is_none_not_an_error(root):
    assert read_registration(root, "kraken-a-v1") is None


# ── where it goes ───────────────────────────────────────────────────────────
def test_trained_is_created_but_never_its_parents(tmp_path):
    """An unmounted share leaves an empty mountpoint. mkdir(parents=True) would
    build the registry on the local disk and register into it, where the
    gateway never looks — a success that is a silent failure."""
    missing = tmp_path / "mnt" / "wbkolleg_dh_1" / "registry"
    with pytest.raises(RegistrationError, match="mounted"):
        write_registration(missing, spec())
    assert listing(tmp_path) == []

    present = tmp_path / "registry"
    present.mkdir()
    write_registration(present, spec())
    assert listing(present) == ["trained"]
    assert listing(present / "trained") == ["kraken-thun-v1.yaml"]


def test_models_config_follows_registry_root(tmp_path, monkeypatch):
    """One setting names the registry; the curated file is found under it.

    Two independent defaults would drift the way ketos and venvs_root once did:
    base models read from one registry, models registered into another.
    """
    for name in ("ATR_TRAIN_REGISTRY_ROOT", "ATR_TRAIN_MODELS_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    moved = tmp_path / "elsewhere" / "registry"

    default = TrainerSettings(_env_file=None)
    assert default.registry_root == LIVE_ROOT
    assert default.models_config == LIVE_ROOT / "models.yaml"

    assert TrainerSettings(_env_file=None, registry_root=moved).models_config == \
        moved / "models.yaml"

    monkeypatch.setenv("ATR_TRAIN_REGISTRY_ROOT", str(moved))
    assert TrainerSettings(_env_file=None).models_config == moved / "models.yaml"
    monkeypatch.setenv("ATR_TRAIN_MODELS_CONFIG", "")          # `=` alone is unset
    assert TrainerSettings(_env_file=None).models_config == moved / "models.yaml"

    pinned = tmp_path / "pinned.yaml"
    assert TrainerSettings(_env_file=None, models_config=pinned).models_config == pinned


@pytest.mark.parametrize("bad", ["registry", "./registry", ""])
def test_a_relative_registry_root_is_refused(bad):
    """Relative to what? A different registry for every working directory."""
    with pytest.raises(ValueError, match="absolute"):
        TrainerSettings(_env_file=None, registry_root=bad)


def test_the_suite_never_registers_on_the_live_share():
    """conftest.py points every settings object in the suite elsewhere."""
    assert not str(TrainerSettings().registry_root).startswith("/mnt/")


# ── by hand, when the share was away ────────────────────────────────────────
def test_the_manual_command_registers_what_it_is_given(root, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(yaml.safe_dump(spec("kraken-a-v1"))))
    assert registration.main(["--root", str(root)]) == 0
    assert read_registration(root, "kraken-a-v1").local_path.endswith("kraken-a-v1.mlmodel")
    assert capsys.readouterr().out.strip() == str(registration_path(root, "kraken-a-v1"))


def test_the_manual_command_refuses_what_the_trainer_would(root, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(yaml.safe_dump(spec(local_path="x.mlmodel"))))
    assert registration.main(["--root", str(root)]) == 1
    assert "relative" in capsys.readouterr().err
    assert listing(root) == []


# ── the contract with the gateway ───────────────────────────────────────────
def test_the_gateways_example_validates_against_the_local_schema():
    """If this fails after copying a new example, the gateway has a field (or a
    rule) this repo's schema does not: update :class:`Registration`."""
    raw = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(raw, dict), "the example is ONE mapping"
    parsed = Registration.model_validate(raw)
    assert parsed.id == FIXTURE.stem
    assert parsed.enabled is False and parsed.disabled_reason is None
    assert gateway_problems(FIXTURE) == []


def _register_through(engine: str, tmp_path: Path) -> tuple[Path, Path]:
    """Run one backend's real ``_register`` and return (registration, weights)."""
    registry_root = tmp_path / "registry"
    registry_root.mkdir()
    settings = TrainerSettings(
        _env_file=None, jobs_root=tmp_path / "jobs", trained_root=tmp_path / "trained",
        registry_root=registry_root, checkpoint_root=tmp_path / "ckpt")
    store = JobStore(settings.jobs_root)
    module = {"kraken": "kraken_train_svc.runner", "trocr": "trocr_train_svc.runner",
              "vllm": "vlm_train_svc.runner"}[engine]
    pipeline = importlib.import_module(module).Pipeline(store, settings)
    model_id = f"{engine}-contract-v1"
    job = store.create(TrainRequest(
        engine=engine, model_id=model_id, force=True,
        dataset=DatasetSpec(hf_repo="dh-unibe/x", train_projects=["p"])))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    if engine == "kraken":
        artefact = scratch / "best_0.9.mlmodel"
        artefact.write_bytes(b"WEIGHTS")
    else:
        artefact = scratch / "adapter"
        artefact.mkdir()
        (artefact / "adapter_model.safetensors").write_bytes(b"WEIGHTS")
    weights = pipeline._register(job, artefact, Metrics(cer=0.1))
    return registration_path(registry_root, model_id), weights


@pytest.mark.parametrize("engine", ["kraken", "trocr", "vllm"])
def test_what_each_runner_writes_is_what_the_gateway_reads(engine, tmp_path):
    """The written file passes the gateway's rules and has the example's shape:
    no key the example's schema does not have, the keys the gate needs present,
    and every shared key of the same kind (a list stays a list)."""
    path, weights = _register_through(engine, tmp_path)
    assert gateway_problems(path) == []

    written = yaml.safe_load(path.read_text(encoding="utf-8"))
    example = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    assert set(written) <= set(Registration.model_fields)
    assert {"id", "engine", "local_path", "enabled"} <= set(written)
    for key in set(written) & set(example):
        assert type(written[key]) is type(example[key]), key

    assert written["engine"] == engine
    assert written["enabled"] is False, "registered is not proven: the gate flips it"
    assert "disabled_reason" not in written, "the gate may ask only for a model without one"
    assert Path(written["local_path"]).is_absolute()
    assert written["local_path"] == str(weights)
    assert Path(written["local_path"]).exists()


# ── the deployment example ──────────────────────────────────────────────────
def test_the_example_env_keeps_weights_and_registry_on_one_mount_and_counts_right():
    """#14: both machines must mount the share at the same path, because a
    registration names its weights by absolute path. And the header's count of
    values that must agree across the machines is what an operator checks
    against — it has to match the markers below it."""
    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")
    values = dict(line.split("=", 1) for line in text.splitlines()
                  if line and not line.startswith("#") and "=" in line)
    assert values["ATR_TRAIN_REGISTRY_ROOT"] == str(LIVE_ROOT)
    assert values["ATR_TRAIN_TRAINED_ROOT"].startswith("/mnt/wbkolleg_dh_1/")
    assert "ATR_TRAIN_MODELS_CONFIG" not in values, "derived from the root; set both and they drift"

    markers = [line for line in text.splitlines()
               if ">>> SHARED <<<" in line and "They are marked" not in line]
    words = {3: "three", 4: "four", 5: "five", 6: "six"}
    assert f"{words[len(markers)]} values have to agree" in text.replace("\n# ", " ")

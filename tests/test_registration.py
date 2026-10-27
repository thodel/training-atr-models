"""Registering a trained model as one file on the share (#14).

The reader of these files is the gateway (serving-atr-inference#138,
``atr_serving/shared_registry.py``). Its rules are restated in
:func:`gateway_problems` rather than imported — the two repos share a file
format, not code — and ``fixtures/registry/trained/example.yaml`` is the
gateway's own example, copied verbatim from serving-atr-inference main
(unchanged since ac1b42f, 16.09.2026). If the gateway changes that file, copy it
again: the contract tests below say what else has to follow.

The example uses a subset of the fields, so it cannot notice a field the gateway
gained — ``max_pixels`` did, twelve minutes before the schema here was written.
``fixtures/registry/modelspec_fields.txt`` pins the whole list — written by
``scripts/refresh_gateway_fixtures.py`` and compared with the gateway's main by
``tests/test_gateway_drift.py`` wherever the network allows (#151).
"""

from __future__ import annotations

import importlib
import io
import json
import os
import threading
from pathlib import Path

import pytest
import yaml

from atr_training import registration
from atr_training.contracts import DatasetSpec, Metrics, TrainRequest
from atr_training.jobstore import JobStore
from atr_training.weights_digest import WeightFile, combined_digest, digest_weights
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

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "registry"
FIXTURE = FIXTURES / "trained" / "example.yaml"
LIVE_ROOT = Path("/mnt/wbkolleg_dh_1/Textrecognition_Training/registry")


def gateway_fields() -> list[str]:
    """The gateway's ModelSpec fields, as copied into the fixture."""
    lines = (FIXTURES / "modelspec_fields.txt").read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


#: One value for every field the gateway knows — what a registration an
#: operator has tuned by hand can look like.
EVERY_FIELD = {
    "hf_repo": "dh-unibe/x", "zenodo_id": "10.5281/zenodo.1234567",
    "disabled_reason": None, "base_model": "10.5281/zenodo.7516057",
    "languages": ["de"], "scripts": ["Kurrent"], "centuries": [16], "vram_mb": 500,
    "max_new_tokens": 2048, "max_pixels": 1_605_632, "residency": "pinned",
    "gpu_affinity": 1, "prompt": "Transcribe the page.",
    #: vLLM-only (the gateway refuses them on another engine), which is why the
    #: tests that use EVERY_FIELD register a vllm model.
    "vllm_venv": "vllm-next", "max_num_seqs": 4,
    "vllm_args": ["--chat-template-content-format", "string"], "startup_timeout_s": 900,
    "training_datasets": ["10.5281/zenodo.1234567"],
    "description": "Qwen3-VL fine-tuned on Thun missives",
    #: Measured on the gateway (serving#130); a runner never writes it.
    "vram_measured": {"total_mib": 9000, "weights_mib": 8000, "host": "srv",
                      "measured_at": "2026-10-21T06:00:00+00:00", "gpu": 1},
    #: An honest one: the combined value follows from its file list, as the
    #: validator demands on both sides (#259).
    "weights_digest": {
        "sha256": combined_digest([WeightFile(name="model.safetensors", sha256="a" * 64,
                                              size_bytes=4)]),
        "files": [{"name": "model.safetensors", "sha256": "a" * 64, "size_bytes": 4}],
        "host": "asteraix", "measured_at": "2026-10-21T06:00:00+00:00",
    },
}


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


def test_promotion_keeps_every_field_the_gateway_knows(root):
    """A registration the gateway serves is one the gate can promote. Before
    ``max_pixels`` was added here, a hand-set pixel budget made set_enabled
    refuse the file ("Extra inputs are not permitted") and the kraken gate
    report "not promoted" for a model that had served."""
    everything = spec("vllm-a-v1", engine="vllm", **EVERY_FIELD)
    assert set(everything) == set(gateway_fields()), "EVERY_FIELD is behind the fixture"
    path = registration_path(root, "vllm-a-v1")
    trained_dir(root).mkdir()
    path.write_text(yaml.safe_dump(everything), encoding="utf-8")

    assert set_enabled(root, "vllm-a-v1", True) is True
    promoted = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert promoted == {**{k: v for k, v in everything.items() if v is not None},
                        "enabled": True}
    assert gateway_problems(path) == []


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
    with pytest.raises(ValueError, match="registry_root must be an absolute"):
        TrainerSettings(_env_file=None, registry_root=bad)


@pytest.mark.parametrize("bad", ["trained", "./trained", ""])
def test_a_relative_trained_root_is_refused_at_startup(bad):
    """Every local_path is under it. Relative, it used to be refused only by
    the registration, after the whole run — with a register-by-hand command
    that carried the same relative path and was refused too."""
    with pytest.raises(ValueError, match="trained_root must be an absolute"):
        TrainerSettings(_env_file=None, trained_root=bad)


def test_the_suite_never_registers_on_the_live_share():
    """conftest.py points every settings object in the suite elsewhere."""
    assert not str(TrainerSettings().registry_root).startswith("/mnt/")


# ── by hand, when the share was away ────────────────────────────────────────
def test_the_manual_command_registers_what_it_is_given(root, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(yaml.safe_dump(spec("kraken-a-v1"))))
    assert registration.main(["--root", str(root)]) == 0
    assert read_registration(root, "kraken-a-v1").local_path.endswith("kraken-a-v1.mlmodel")
    assert capsys.readouterr().out.strip() == str(registration_path(root, "kraken-a-v1"))


def test_the_manual_command_takes_every_field_the_gateway_knows(root, monkeypatch, capsys):
    everything = spec("vllm-a-v1", engine="vllm", **EVERY_FIELD)
    monkeypatch.setattr("sys.stdin", io.StringIO(yaml.safe_dump(everything)))
    assert registration.main(["--root", str(root)]) == 0, capsys.readouterr().err
    written = read_registration(root, "vllm-a-v1")
    assert written.max_pixels == 1_605_632
    assert (written.vllm_venv, written.max_num_seqs) == ("vllm-next", 4)


def test_the_manual_command_refuses_what_the_trainer_would(root, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(yaml.safe_dump(spec(local_path="x.mlmodel"))))
    assert registration.main(["--root", str(root)]) == 1
    assert "relative" in capsys.readouterr().err
    assert listing(root) == []


# ── the contract with the gateway ───────────────────────────────────────────
def test_the_local_schema_has_exactly_the_gateways_fields():
    """Pinned, because the example cannot notice drift: it uses a subset.

    A field only the gateway has makes this repo refuse files the gateway
    serves (extra="forbid"); a field only this repo has is one the gateway
    drops without a word. After refreshing modelspec_fields.txt, this says
    which way :class:`Registration` has to move.
    """
    gateway, local = gateway_fields(), list(Registration.model_fields)
    assert len(gateway) == len(set(gateway)), "a field is listed twice in the fixture"
    assert set(gateway) - set(local) == set(), "the gateway has fields Registration lacks"
    assert set(local) - set(gateway) == set(), "Registration has fields the gateway lacks"


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
    store = JobStore(settings.jobs_root, host_id=settings.host_id)
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
    assert set(written) <= set(gateway_fields())
    assert {"id", "engine", "local_path", "enabled"} <= set(written)
    for key in set(written) & set(example):
        assert type(written[key]) is type(example[key]), key

    assert written["engine"] == engine
    # #151: every trained model says what it is, the way every curated one does.
    assert written["description"].startswith(
        {"kraken": "kraken model trained from scratch", "trocr": "TrOCR fine-tune",
         "vllm": "LoRA"}[engine]), written["description"]
    assert "dh-unibe/x (1 project)" in written["description"]
    assert written["description"].endswith("held-out CER 10.0\u202f%")
    assert written["enabled"] is False, "registered is not proven: the gate flips it"
    assert "disabled_reason" not in written, "the gate may ask only for a model without one"
    assert Path(written["local_path"]).is_absolute()
    assert written["local_path"] == str(weights)
    assert Path(written["local_path"]).exists()


# ── the description a runner writes (#151) ──────────────────────────────────
from atr_training.registration import CorpusCount, describe_training  # noqa: E402


def test_a_description_names_base_corpus_counts_level_and_cer():
    line = describe_training(
        engine="vllm", base_model="Qwen/Qwen3.5-4B",
        corpora=[CorpusCount("dh-unibe/image-text_kurrent-xix-lines", 3, 412, 13870)],
        granularity="line", cer=0.0312)
    assert line == ("LoRA on Qwen/Qwen3.5-4B, trained on dh-unibe/image-text_kurrent-xix-lines "
                    "(3 projects, 412 pages, 13\u202f870 lines), line level; held-out CER 3.1\u202f%")


def test_a_kraken_model_without_a_base_is_trained_from_scratch():
    line = describe_training(engine="kraken", base_model=None,
                             corpora=[CorpusCount("dh-unibe/x", 1, 9, 1)],
                             granularity="page", cer=0.2)
    assert line.startswith("kraken model trained from scratch, trained on dh-unibe/x (1 project, 9 pages, 1 line)")
    fine = describe_training(engine="kraken", base_model="kraken-prima",
                             corpora=[CorpusCount("dh-unibe/x", 1)], granularity=None, cer=0.2)
    assert fine == "kraken fine-tune of kraken-prima, trained on dh-unibe/x (1 project); held-out CER 20.0\u202f%"


def test_an_unmeasured_cer_is_said_not_numbered_and_uncounted_pages_are_omitted():
    line = describe_training(engine="trocr", base_model="microsoft/trocr-base-handwritten",
                             corpora=[CorpusCount("dh-unibe/x", None)], granularity="line", cer=None)
    assert line.endswith("; CER not measured")
    assert "all projects" in line and "pages" not in line and "0" not in line


def test_several_corpora_are_all_named():
    line = describe_training(engine="vllm", base_model="b",
                             corpora=[CorpusCount("dh-unibe/a", 2, 10, 100),
                                      CorpusCount("dh-unibe/b", 5, 20, 200)],
                             granularity="mixed", cer=0.05)
    assert "dh-unibe/a (2 projects, 10 pages, 100 lines) + dh-unibe/b (5 projects, 20 pages, 200 lines)" in line
    with pytest.raises(ValueError, match="at least one corpus"):
        describe_training(engine="vllm", base_model="b", corpora=[], granularity=None, cer=None)


def test_three_or_more_corpora_are_named_briefly_and_their_counts_summed():
    """Four corpora with their counts were 450 characters on asteraix; /models
    shows one line."""
    four = [CorpusCount("dh-unibe/image-text_zh-regierungsratsprotokolle", None, 20000, 816625),
            CorpusCount("dh-unibe/image-text_kurrent-xix", 25, 6071, 141067),
            CorpusCount("dh-unibe/image-text_parlamentsdienste-protokolle", 9, 63, 5703),
            CorpusCount("dh-unibe/image-text_nr-sr-vereinigte-bundesversammlung-xix", 35, 52, 3368)]
    line = describe_training(engine="vllm", base_model="Qwen/Qwen3-VL-4B-Instruct", corpora=four,
                             granularity="line", cer=0.184)
    assert line == ("LoRA on Qwen/Qwen3-VL-4B-Instruct, trained on 4 dh-unibe corpora "
                    "(zh-regierungsratsprotokolle, kurrent-xix, parlamentsdienste-protokolle, "
                    "nr-sr-vereinigte-bundesversammlung-xix; 26\u202f186 pages, 966\u202f763 lines), "
                    "line level; held-out CER 18.4\u202f%")
    assert len(line) < 260
    mixed = four[:2] + [CorpusCount("other/corpus", 1, None, None)]
    line = describe_training(engine="vllm", base_model="b", corpora=mixed, granularity="line", cer=None)
    assert "3 corpora (dh-unibe/image-text_zh-regierungsratsprotokolle, " in line, \
        "no shared prefix: full names"
    assert "counts partly unrecorded" in line and "pages" not in line.split("(")[1].split(";")[1]
    two = describe_training(engine="vllm", base_model="b", corpora=four[:2], granularity="line", cer=None)
    assert " + " in two and "2 dh-unibe corpora" not in two, "two corpora keep the full form"


def test_a_runner_s_spec_may_carry_its_own_description(tmp_path):
    """``setdefault``: a backend that writes a better line keeps it."""
    from vlm_train_svc.runner import Pipeline
    registry_root = tmp_path / "registry"
    registry_root.mkdir()
    settings = TrainerSettings(
        _env_file=None, jobs_root=tmp_path / "jobs", trained_root=tmp_path / "trained",
        registry_root=registry_root, checkpoint_root=tmp_path / "ckpt")
    store = JobStore(settings.jobs_root, host_id=settings.host_id)
    pipeline = Pipeline(store, settings)
    job = store.create(TrainRequest(
        engine="vllm", model_id="vllm-own-words", force=True, base_model="Qwen/Qwen3.5-4B",
        dataset=DatasetSpec(hf_repo="dh-unibe/x", train_projects=["p", "q"])))
    weights = tmp_path / "trained" / "vllm-own-words"
    weights.mkdir(parents=True)
    pipeline._write_registration(job, {"id": "vllm-own-words", "engine": "vllm",
                                       "local_path": str(weights), "enabled": False,
                                       "description": "my own words"}, weights,
                                 metrics=Metrics(cer=0.5))
    assert read_registration(registry_root, "vllm-own-words").description == "my own words"
    # and the derived one, from the job alone, when the spec has none
    derived = pipeline._describe(job, Metrics(cer=0.5))
    assert derived == ("LoRA on Qwen/Qwen3.5-4B, trained on dh-unibe/x (2 projects), "
                       "line level; held-out CER 50.0\u202f%")
    whole = store.create(TrainRequest(
        engine="vllm", model_id="vllm-all", force=True, base_model="b",
        dataset=DatasetSpec(hf_repo="dh-unibe/y", all_projects=True, max_pages=50)))
    assert "dh-unibe/y (all projects)" in pipeline._describe(whole, None)
    assert pipeline._describe(whole, None).endswith("CER not measured")


# ── the registrations written before #151 get their line from metadata.json ──
def _registered(root: Path, tmp_path: Path, model_id: str, engine: str, *, metadata: dict | None,
                description: str | None = None, as_file: bool = False) -> Path:
    weights_dir = tmp_path / "trained" / model_id
    weights_dir.mkdir(parents=True, exist_ok=True)
    local = weights_dir / f"{model_id}.mlmodel" if as_file else weights_dir
    if as_file:
        local.write_bytes(b"W")
    if metadata is not None:
        (weights_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    spec = {"id": model_id, "engine": engine, "local_path": str(local), "enabled": False}
    if description:
        spec["description"] = description
    return write_registration(root, spec)


META = {
    "engine": "vllm", "base_model": "Qwen/Qwen3.5-4B",
    "request": {"datasets": [{"hf_repo": "dh-unibe/image-text_kurrent-xix-lines",
                              "train_projects": ["a", "b", "c"]}],
                "params": {"granularity": "line"}},
    "progress": {"dataset_counts": [{"hf_repo": "dh-unibe/image-text_kurrent-xix-lines",
                                     "pages_written": 412, "lines": 13870}]},
    "metrics": {"cer": 0.0312},
}


def test_description_from_metadata_reads_what_the_runners_wrote():
    assert registration.description_from_metadata(META) == (
        "LoRA on Qwen/Qwen3.5-4B, trained on dh-unibe/image-text_kurrent-xix-lines "
        "(3 projects, 412 pages, 13\u202f870 lines), line level; held-out CER 3.1\u202f%")
    legacy = {"engine": "kraken", "request": {"dataset": {"hf_repo": "dh-unibe/x",
                                                         "train_projects": ["p"], "granularity": "page"}},
              "metrics": {}}
    assert registration.description_from_metadata(legacy) == (
        "kraken model trained from scratch, trained on dh-unibe/x (1 project), page level; "
        "CER not measured")
    with pytest.raises(ValueError, match="names no dataset"):
        registration.description_from_metadata({"engine": "kraken", "request": {}})


def test_describing_fills_only_what_is_missing_and_says_why_it_cannot(root, tmp_path):
    _registered(root, tmp_path, "vllm-old", "vllm", metadata=META)
    _registered(root, tmp_path, "kraken-old", "kraken",
                metadata={**META, "engine": "kraken", "base_model": None}, as_file=True)
    _registered(root, tmp_path, "kraken-by-hand", "kraken", metadata=META, description="my words")
    _registered(root, tmp_path, "kraken-moved", "kraken", metadata=None)

    dry = {r.model_id: r for r in registration.describe_registrations(root)}
    assert dry["vllm-old"].action == "described" and dry["vllm-old"].after.startswith("LoRA on")
    assert dry["kraken-old"].action == "described" and "metadata.json" in dry["kraken-old"].reason, \
        "a kraken local_path is the file; metadata.json sits beside it"
    assert dry["kraken-by-hand"].action == "kept" and dry["kraken-by-hand"].after == "my words"
    assert dry["kraken-moved"].action == "unchanged" and "no metadata.json" in dry["kraken-moved"].reason
    assert not any(r.written for r in dry.values())
    assert read_registration(root, "vllm-old").description is None, "a dry run writes nothing"

    wet = {r.model_id: r for r in registration.describe_registrations(root, write=True)}
    assert wet["vllm-old"].written and wet["kraken-old"].written
    assert read_registration(root, "vllm-old").description.startswith("LoRA on Qwen/Qwen3.5-4B")
    assert read_registration(root, "kraken-old").description.startswith("kraken model trained from scratch")
    assert read_registration(root, "kraken-by-hand").description == "my words"
    assert read_registration(root, "kraken-moved").description is None
    again = {r.model_id: r for r in registration.describe_registrations(root, write=True)}
    assert again["vllm-old"].action == "kept"


def test_the_describe_command_lists_then_writes(root, tmp_path, monkeypatch, capsys):
    _registered(root, tmp_path, "vllm-old", "vllm", metadata=META)
    assert registration.main(["--root", str(root), "--describe"]) == 0
    out = capsys.readouterr().out
    assert "1 would be described" in out and "re-run with --write" in out and "(dry run)" in out
    assert read_registration(root, "vllm-old").description is None
    assert registration.main(["--root", str(root), "--describe", "--write"]) == 0
    assert "1 described" in capsys.readouterr().out
    assert read_registration(root, "vllm-old").description.startswith("LoRA on")
    with pytest.raises(SystemExit):
        registration.main(["--root", str(root), "--write"])


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


# ── the two vLLM launch fields (#80) ────────────────────────────────────────
#
# The gateway gained `vllm_venv` and `max_num_seqs` on 21.09.2026 (serving
# 58f678a) and this schema refuses what it does not know, so for three days a
# model needing a second vLLM venv could not be registered at all — and a
# registration an operator had added one to by hand could not be enabled either,
# because set_enabled re-validates before writing back. The rules travel with
# the fields: the gateway refuses the same values, and a registration this repo
# writes must not be one the gateway then skips.

def test_a_vllm_registration_carries_its_venv_and_batch_size():
    reg = Registration(id="gemma3-medieval-german-v1", engine="vllm",
                       local_path="/mnt/share/trained/g/g.safetensors",
                       vllm_venv="vllm-next", max_num_seqs=4)
    assert reg.vllm_venv == "vllm-next"
    assert reg.max_num_seqs == 4


@pytest.mark.parametrize("field,value", [("vllm_venv", "vllm-next"), ("max_num_seqs", 4)])
def test_the_launch_fields_are_refused_for_another_engine(root: Path, field, value):
    """The gateway's rule: only a vLLM launch has a venv or a batch size."""
    with pytest.raises(RegistrationError, match="only meaningful for engine vllm"):
        write_registration(root, spec("trocr-kurrent-v1", engine="trocr", **{field: value}))


@pytest.mark.parametrize("name", ["../etc", "vllm/next", ".", "..", "", "vllm next"])
def test_a_venv_that_could_climb_out_of_venvs_is_refused(root: Path, name):
    """It is a directory name under the gateway's .venvs/, not a path."""
    with pytest.raises(RegistrationError, match="must be a directory name"):
        write_registration(root, spec("vllm-x-v1", engine="vllm", vllm_venv=name))


def test_max_num_seqs_must_be_at_least_one(root: Path):
    with pytest.raises(RegistrationError, match="max_num_seqs"):
        write_registration(root, spec("vllm-x-v1", engine="vllm", max_num_seqs=0))


def test_a_registration_with_a_venv_can_be_written_and_then_enabled(root: Path):
    """The path out of OPERATIONS.md, end to end: this is what broke.

    write_registration → read_registration → set_enabled all validate the same
    schema, so a field missing here fails on the way in *or* on the way back.
    """
    model_id = "gemma3-medieval-german-v1"
    written = write_registration(root, spec(model_id, engine="vllm",
                                           local_path=f"/mnt/share/trained/{model_id}",
                                           vllm_venv="vllm-next", max_num_seqs=4))
    assert gateway_problems(Path(written)) == []
    set_enabled(root, model_id, True)
    reread = read_registration(root, model_id)
    assert reread.enabled is True
    assert (reread.vllm_venv, reread.max_num_seqs) == ("vllm-next", 4)


def test_the_pinned_field_list_is_the_one_the_gateway_has_today():
    """Both sides at 25 fields, and the fixture says which commit it was copied
    from — the failure mode this pin exists for is two stale copies agreeing.

    The copy from 21.09. had 22 and was a month behind when #259 refreshed it:
    ``description`` (serving#101) and ``vram_measured`` (serving#130) had
    reached the gateway without this repo noticing, exactly the drift the pin
    is for — nothing here compares a fixture against the gateway, only against
    :class:`Registration`."""
    fields = gateway_fields()
    assert len(fields) == 27
    assert fields[fields.index("prompt") + 1:fields.index("training_datasets")] == [
        "vllm_venv", "max_num_seqs", "vllm_args", "startup_timeout_s", "weights_digest"]


def test_the_two_launch_fields_follow_the_gateways_rules(tmp_path):
    """``vllm_args`` and ``startup_timeout_s`` (serving 06ba832, Apertus 1.5):
    vLLM only, and never a flag the launcher owns — a registration the gateway
    would refuse to read is refused here, before it is written."""
    base = {"id": "apertus-test-v1", "local_path": "/mnt/share/trained/apertus-test-v1"}
    ok = Registration.model_validate({**base, "engine": "vllm", "startup_timeout_s": 900,
                                      "vllm_args": ["--chat-template-content-format", "string"]})
    assert ok.startup_timeout_s == 900 and ok.vllm_args[0] == "--chat-template-content-format"
    for bad, why in (({"engine": "kraken", "startup_timeout_s": 900}, "only meaningful"),
                     ({"engine": "kraken", "vllm_args": ["--enforce-eager"]}, "only meaningful"),
                     ({"engine": "vllm", "vllm_args": ["--port=9000"]}, "launcher sets"),
                     ({"engine": "vllm", "startup_timeout_s": 0}, "greater than 0")):
        with pytest.raises(ValueError, match=why):
            Registration.model_validate({**base, **bad})


# ── vram_mb: a number or nothing, never 0 (#225) ────────────────────────────
def test_a_zero_vram_mb_is_read_as_unrecorded_and_never_written(tmp_path):
    """Every registration from before #225 carried ``vram_mb: 0`` from the old
    default (five of six on the share when #242 ran). A 0 is not a measurement;
    it is read as none and dropped on the next rewrite, so a promotion by hand
    neither refuses the file nor keeps the 0."""
    spec = {"id": "kraken-legacy-v1", "engine": "kraken",
            "local_path": str(tmp_path / "w.mlmodel"), "vram_mb": 0}
    assert Registration.model_validate(spec).vram_mb is None
    written = write_registration(tmp_path, spec)
    # Parsed, not grepped: tmp_path itself carries this test's name, "vram_mb" included.
    assert "vram_mb" not in yaml.safe_load(written.read_text(encoding="utf-8"))

    legacy = registration_path(tmp_path, "kraken-old-v1")
    legacy.parent.mkdir(exist_ok=True)
    legacy.write_text("id: kraken-old-v1\nengine: kraken\nlocal_path: /x/w.mlmodel\n"
                      "enabled: false\nvram_mb: 0\n", encoding="utf-8")
    assert set_enabled(tmp_path, "kraken-old-v1") is True
    assert "vram_mb" not in yaml.safe_load(legacy.read_text(encoding="utf-8"))
    assert read_registration(tmp_path, "kraken-old-v1").enabled is True


def test_a_measured_vram_mb_is_kept_and_a_negative_one_refused(tmp_path):
    spec = {"id": "kraken-sized-v1", "engine": "kraken",
            "local_path": str(tmp_path / "w.mlmodel"), "vram_mb": 16}
    written = write_registration(tmp_path, spec)
    assert yaml.safe_load(written.read_text(encoding="utf-8"))["vram_mb"] == 16
    with pytest.raises(RegistrationError, match="vram_mb"):
        write_registration(tmp_path, dict(spec, id="kraken-negative-v1", vram_mb=-1))


def test_weights_mib_sizes_a_file_or_the_weight_files_of_a_directory(tmp_path):
    """What vram_mb means on the gateway: the weights, rounded up to a MiB, and
    nothing else that sits beside them. No weights is None, not 0 (#165)."""
    model = tmp_path / "m.mlmodel"
    model.write_bytes(b"x" * (3 * 2 ** 20 + 1))
    assert registration.weights_mib(model) == 4

    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"x" * 2 ** 20)
    (checkpoint / "optimizer.pt").write_bytes(b"x" * 2 ** 20)
    (checkpoint / "README.md").write_text("x" * 2 ** 20, encoding="utf-8")  # not weights
    assert registration.weights_mib(checkpoint) == 2

    tiny = tmp_path / "tiny.pt"
    tiny.write_bytes(b"x")
    assert registration.weights_mib(tiny) == 1

    assert registration.weights_mib(tmp_path / "missing") is None
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.txt").write_text("x", encoding="utf-8")
    assert registration.weights_mib(empty) is None


@pytest.mark.parametrize("engine, expected", [("kraken", 1), ("trocr", 1), ("vllm", None)])
def test_a_runner_registers_the_size_of_its_weights_or_nothing(engine, expected, tmp_path):
    """kraken and trocr register what they copied; a VLM adapter registers no
    size, because what is served is base plus adapter, merged on idhefix."""
    path, _ = _register_through(engine, tmp_path)
    written = yaml.safe_load(path.read_text(encoding="utf-8"))
    if expected is None:
        assert "vram_mb" not in written
    else:
        assert written["vram_mb"] == expected


# ── sizing what was registered before #225 (#242) ────────────────────────────
#
# Every registration from before #236 carries ``vram_mb: 0``. Both readers take
# the 0 as "unrecorded" by now, but the files are unchanged, and for a kraken or
# trocr model there is a number that could stand there: the size of its weights.

def _legacy(root: Path, model_id: str, **fields) -> Path:
    """A registration as the trainer wrote it before #225: ``vram_mb: 0`` included."""
    raw = {"id": model_id, "engine": "kraken", "local_path": f"/x/{model_id}.mlmodel",
           "enabled": True, "task": "htr", "languages": ["de"], "vram_mb": 0, **fields}
    path = registration_path(root, model_id)
    path.parent.mkdir(exist_ok=True)
    path.write_text("# written by the trainer\n" + yaml.safe_dump(raw, sort_keys=False),
                    encoding="utf-8")
    return path


def _weights(tmp_path: Path, name: str, mib: int, suffix: str = ".mlmodel") -> Path:
    path = tmp_path / "weights" / f"{name}{suffix}"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"\0" * (mib * 2**20 - 7))   # just under: rounded up, never down
    return path


def _raw(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_sizing_replaces_the_legacy_zero_with_the_size_of_the_weights(root, tmp_path):
    weights = _weights(tmp_path, "kraken-old-v1", 3)
    path = _legacy(root, "kraken-old-v1", local_path=str(weights))

    (result,) = registration.size_weights(root, write=True)

    assert (result.action, result.before, result.after, result.written) == ("sized", 0, 3, True)
    assert result.reason.endswith(str(weights))
    after = _raw(path)
    assert after["vram_mb"] == 3
    assert {k: v for k, v in after.items() if k != "vram_mb"} == {
        "id": "kraken-old-v1", "engine": "kraken", "local_path": str(weights),
        "enabled": True, "task": "htr", "languages": ["de"],
        # the writer's own defaults, as every rewrite adds them
        "level": "page", "scripts": [], "centuries": [], "residency": "lazy",
        "training_datasets": [],
    }, "only vram_mb changed"
    assert read_registration(root, "kraken-old-v1").vram_mb == 3
    assert listing(trained_dir(root)) == ["kraken-old-v1.yaml"], "no tmp file left behind"


def test_sizing_sizes_a_directory_of_weights_too(root, tmp_path):
    weights = _weights(tmp_path, "model", 2, suffix=".safetensors").parent
    (weights / "config.json").write_text("{" + " " * 2**20 + "}", encoding="utf-8")
    _legacy(root, "trocr-old-v1", engine="trocr", local_path=str(weights))

    (result,) = registration.size_weights(root, write=True)

    assert (result.action, result.after) == ("sized", 2), "the config is not weights"


def test_sizing_sizes_an_absent_vram_mb_as_well_as_a_zero(root, tmp_path):
    """A hand-made registration without the field is the same case for the gateway."""
    weights = _weights(tmp_path, "kraken-hand-v1", 1)
    path = _legacy(root, "kraken-hand-v1", local_path=str(weights))
    raw = _raw(path)
    del raw["vram_mb"]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    (result,) = registration.size_weights(root, write=True)

    assert (result.action, result.before, result.after) == ("sized", None, 1)
    assert _raw(path)["vram_mb"] == 1


def test_a_vllm_adapter_loses_its_zero_and_gets_no_number(root, tmp_path):
    """What is served is base plus adapter, merged on idhefix; the adapter alone
    would understate it by the base. The merge writes the size (serving#216)."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"\0" * 2**20)
    path = _legacy(root, "qwen3vl-old-v1", engine="vllm", local_path=str(adapter),
                   base_model="Qwen/Qwen3-VL-8B-Instruct", enabled=False)

    (result,) = registration.size_weights(root, write=True)

    assert (result.action, result.before, result.after) == ("dropped", 0, None)
    assert "serving#216" in result.reason
    assert "vram_mb" not in _raw(path)
    assert read_registration(root, "qwen3vl-old-v1").enabled is False


def test_weights_that_cannot_be_sized_here_lose_the_zero_and_are_named(root, tmp_path):
    """"Could not size" is not "needs nothing" (#225): the 0 goes, no number comes."""
    missing = tmp_path / "gone" / "kraken-gone-v1.mlmodel"
    path = _legacy(root, "kraken-gone-v1", local_path=str(missing))

    (result,) = registration.size_weights(root, write=True)

    assert (result.action, result.before, result.after) == ("dropped", 0, None)
    assert str(missing) in result.reason
    assert "vram_mb" not in _raw(path)


def test_a_declared_or_measured_value_is_kept_byte_for_byte(root, tmp_path):
    weights = _weights(tmp_path, "w", 9)
    declared = _legacy(root, "kraken-sized-v1", local_path=str(weights), vram_mb=16)
    measured = _legacy(root, "kraken-meas-v1", local_path=str(weights), vram_mb=0,
                       vram_measured={"total_mib": 900, "host": "idhefix",
                                      "measured_at": "2026-10-01T10:00:00Z"})
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (declared, measured)}

    results = {r.model_id: r for r in registration.size_weights(root, write=True)}

    assert results["kraken-sized-v1"].action == "kept"
    assert (results["kraken-sized-v1"].before, results["kraken-sized-v1"].after) == (16, 16)
    assert results["kraken-meas-v1"].action == "kept"
    assert "measure_vram" in results["kraken-meas-v1"].reason
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (declared, measured)} == before


def test_a_registration_with_nothing_to_size_and_no_zero_is_unchanged(root, tmp_path):
    path = _legacy(root, "kraken-clean-v1", local_path=str(tmp_path / "absent.mlmodel"))
    raw = _raw(path)
    del raw["vram_mb"]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    before = path.read_bytes()

    (result,) = registration.size_weights(root, write=True)

    assert (result.action, result.written) == ("unchanged", False)
    assert path.read_bytes() == before


def test_a_dry_run_says_what_would_happen_and_touches_nothing(root, tmp_path):
    weights = _weights(tmp_path, "kraken-old-v1", 3)
    sized = _legacy(root, "kraken-old-v1", local_path=str(weights))
    dropped = _legacy(root, "qwen3vl-old-v1", engine="vllm", local_path=str(tmp_path),
                      base_model="b")
    before = {p: p.read_bytes() for p in (sized, dropped)}

    results = registration.size_weights(root)

    assert [(r.model_id, r.action, r.after, r.written) for r in results] == [
        ("kraken-old-v1", "sized", 3, False), ("qwen3vl-old-v1", "dropped", None, False)]
    assert all(r.line().endswith("(dry run)") for r in results)
    assert {p: p.read_bytes() for p in (sized, dropped)} == before


def test_sizing_rewrites_only_the_files_that_change_and_atomically(root, tmp_path, monkeypatch):
    weights = _weights(tmp_path, "kraken-old-v1", 3)
    _legacy(root, "kraken-old-v1", local_path=str(weights))
    kept = _legacy(root, "kraken-sized-v1", local_path=str(weights), vram_mb=5)
    os.utime(kept, ns=(1_000_000_000, 1_000_000_000))
    kept_before = (kept.read_bytes(), kept.stat().st_mtime_ns, kept.stat().st_ino)
    real_replace = os.replace
    renames = []

    def watching_replace(src, dst):
        src, dst = Path(src), Path(dst)
        renames.append((src.parent == dst.parent, src.name.startswith("."),
                        src.name.endswith(".tmp"), dst.name))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", watching_replace)
    registration.size_weights(root, write=True)

    assert renames == [(True, True, True, "kraken-old-v1.yaml")]
    assert (kept.read_bytes(), kept.stat().st_mtime_ns, kept.stat().st_ino) == kept_before


def test_a_file_this_module_cannot_read_is_skipped_with_its_reason(root, tmp_path):
    weights = _weights(tmp_path, "kraken-old-v1", 3)
    _legacy(root, "kraken-old-v1", local_path=str(weights))
    odd = registration_path(root, "kraken-odd-v1")
    odd.write_text(yaml.safe_dump(dict(spec("kraken-other-v1"), vram_mb=0)), encoding="utf-8")
    odd_before = odd.read_bytes()

    results = {r.model_id: r for r in registration.size_weights(root, write=True)}

    assert results["kraken-old-v1"].action == "sized"
    assert results["kraken-odd-v1"].action == "skipped"
    assert "declares id" in results["kraken-odd-v1"].reason
    assert odd.read_bytes() == odd_before


def test_sizing_an_empty_or_missing_trained_is_nothing(root):
    assert registration.size_weights(root, write=True) == []
    trained_dir(root).mkdir()
    assert registration.size_weights(root, write=True) == []
    assert listing(root) == ["trained"]


def test_the_size_weights_command_is_a_dry_run_unless_told_to_write(root, tmp_path,
                                                                     monkeypatch, capsys):
    weights = _weights(tmp_path, "kraken-old-v1", 3)
    path = _legacy(root, "kraken-old-v1", local_path=str(weights))
    before = path.read_bytes()
    monkeypatch.setattr("sys.stdin", io.StringIO("id: never-read\n"))

    assert registration.main(["--root", str(root), "--size-weights"]) == 0
    out = capsys.readouterr().out
    assert "kraken-old-v1" in out and "0 -> 3" in out and "sized" in out
    assert "1 would change" in out and "--write" in out
    assert path.read_bytes() == before, "nothing is written without --write"

    assert registration.main(["--root", str(root), "--size-weights", "--write"]) == 0
    out = capsys.readouterr().out
    assert "1 rewritten" in out
    assert _raw(path)["vram_mb"] == 3

    assert registration.main(["--root", str(root), "--size-weights", "--write"]) == 0
    assert "0 rewritten" in capsys.readouterr().out, "a second run has nothing left to do"


def test_the_size_weights_command_reports_a_skipped_file_in_its_exit_code(root, tmp_path,
                                                                          capsys):
    odd = registration_path(root, "kraken-odd-v1")
    odd.parent.mkdir()
    odd.write_text("- not\n- a mapping\n", encoding="utf-8")

    assert registration.main(["--root", str(root), "--size-weights", "--write"]) == 1
    out = capsys.readouterr().out
    assert "skipped" in out and "1 skipped" in out


def test_write_without_size_weights_is_refused(root):
    with pytest.raises(SystemExit):
        registration.main(["--root", str(root), "--write"])


# ── the weights digest the gate writes (#259) ───────────────────────────────
def test_the_gate_writes_the_digest_in_the_same_rewrite_as_enabled(root, tmp_path):
    weights = tmp_path / "kraken-a-v1.mlmodel"
    weights.write_bytes(b"the bytes the gate served")
    write_registration(root, spec("kraken-a-v1", local_path=str(weights)))
    digest = digest_weights(weights, host="asteraix")

    assert set_enabled(root, "kraken-a-v1", True, weights_digest=digest) is True

    written = read_registration(root, "kraken-a-v1")
    assert written.enabled is True
    assert written.weights_digest is not None
    assert written.weights_digest.sha256 == digest.sha256
    assert written.weights_digest.host == "asteraix"
    assert [f.name for f in written.weights_digest.files] == ["kraken-a-v1.mlmodel"]
    assert gateway_problems(registration_path(root, "kraken-a-v1")) == []


def test_disabling_keeps_the_digest_of_the_weights_that_were_served(root, tmp_path):
    """Demotion (a retrain replacing the weights) does not forget what the
    earlier weights were; the gate's next rewrite replaces the digest."""
    weights = tmp_path / "w.mlmodel"
    weights.write_bytes(b"v1")
    write_registration(root, spec("kraken-a-v1", local_path=str(weights)))
    set_enabled(root, "kraken-a-v1", True, weights_digest=digest_weights(weights, host="h"))

    set_enabled(root, "kraken-a-v1", False)

    kept = read_registration(root, "kraken-a-v1")
    assert kept.enabled is False and kept.weights_digest is not None


def test_a_digest_that_does_not_follow_from_its_files_is_refused_before_writing(root):
    """A hand-edited value is worse than none: the gateway refuses to load it,
    so this side refuses to write it — neither the file nor trained/ appears."""
    dishonest = {"sha256": "0" * 64,
                 "files": [{"name": "m.mlmodel", "sha256": "a" * 64, "size_bytes": 1}],
                 "host": "h", "measured_at": "2026-10-21T06:00:00+00:00"}

    with pytest.raises(RegistrationError, match="does not follow"):
        write_registration(root, spec("kraken-a-v1", weights_digest=dishonest))
    assert not trained_dir(root).exists()


def test_a_digest_without_provenance_is_refused_before_writing(root):
    files = [WeightFile(name="m.mlmodel", sha256="a" * 64, size_bytes=1)]
    anonymous = {"sha256": combined_digest(files),
                 "files": [f.model_dump() for f in files]}

    with pytest.raises(RegistrationError, match="host|measured_at"):
        write_registration(root, spec("kraken-a-v1", weights_digest=anonymous))


def test_no_digest_is_no_field_not_an_empty_one(root):
    """Three-valued: absent means nobody measured. ``exclude_none`` keeps the
    key out of the file, so the gateway's own default applies."""
    path = write_registration(root, spec("kraken-a-v1"))
    assert "weights_digest" not in yaml.safe_load(path.read_text(encoding="utf-8"))

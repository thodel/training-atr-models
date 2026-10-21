"""Registering a trained model: one file per model in the shared registry (#14).

Until 16.09.2026 the three runners registered a model by rewriting
``config/models.local.yaml`` (``overlay.upsert_entry``). After the split (#3) that
file sat in *this* checkout, and the gateway on idhefix never read it: every
registration went nowhere, without a word, because a missing overlay was read as
"nothing trained yet".

The handover is now a directory on the research share, read by the gateway
(serving-atr-inference#138, ``atr_serving/shared_registry.py``)::

    <registry_root>/models.yaml        curated, published by the gateway (#5)
    <registry_root>/trained/<id>.yaml  one trained model per file — written here

Its rules, which this module keeps:

* **The file name is the id.** The gateway skips a file whose ``id`` differs, so
  a hand-made copy cannot give one id two sets of weights. The path is derived
  from the id here, never passed in.
* **One mapping, not a list** — the fields of one entry under ``models:``.
* **Atomic.** A tmp file in the *same* directory, then ``os.replace``. The tmp
  name starts with ``.`` and ends with ``.tmp``; the gateway reads neither, so a
  half-written file is never served. The job store has written every
  ``job.json`` on this mount the same way for weeks (``jobstore.py``).
* **One file per model**, because two trainers doing read-modify-write on one
  shared file lose each other's updates (#12) — the job store's answer, one
  level up.
* **``local_path`` is absolute.** The gateway skips a relative one: it names a
  different file on every machine.

What this deliberately does not do on the share (CIFS, ``uid=0``,
``gid=research``, ``forcegid``): no ``chmod``, no ``copy2``, no ``utime`` — a
non-owner gets EPERM, which is how a finished run once failed after training
(see the kraken runner). No rename across filesystems either: the tmp file is a
sibling of its target.

The schema below is this repo's own, not an import of the gateway's
``ModelSpec`` (E3: a thin seam, no shared package). It is kept in step by a
contract test against the gateway's example file, copied to
``tests/fixtures/registry/trained/example.yaml``.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import re
import shlex
import socket
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from atr_training.contracts import MODEL_ID_RE
from atr_training.weights_digest import WeightsDigest

__all__ = [
    "TRAINED_DIRNAME",
    "WEIGHT_SUFFIXES",
    "Registration",
    "RegistrationError",
    "Sizing",
    "VramMeasurement",
    "is_registration_name",
    "manual_registration",
    "read_registration",
    "registration_path",
    "set_enabled",
    "size_weights",
    "trained_dir",
    "weights_mib",
    "write_registration",
]

#: What counts as weights when :func:`weights_mib` sizes a directory. Everything
#: else beside them (``metadata.json``, a README, tokenizer files, the kraken
#: training log) is not what a card has to hold.
WEIGHT_SUFFIXES = frozenset({".mlmodel", ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf"})


def weights_mib(path: str | Path) -> int | None:
    """The size of the weights at ``path`` in MiB, rounded up — or None.

    This is what ``vram_mb`` means on the gateway (``atr_serving/manager.py``):
    the weights, on top of which vLLM adds its KV cache and scratch. It is not
    the training peak from ``StageRecord.peak_gpu_mib``, which measures a
    different thing (#163) and must not be written here.

    A file is sized as it is; a directory as the sum of its weight files
    (:data:`WEIGHT_SUFFIXES`). None when there is nothing to size — a missing
    path, or a directory without weight files — because "could not measure" is
    not "needs nothing" (#165, #225).
    """
    target = Path(path)
    try:
        if target.is_file():
            size = target.stat().st_size
        elif target.is_dir():
            size = sum(f.stat().st_size for f in target.rglob("*")
                       if f.is_file() and f.suffix.lower() in WEIGHT_SUFFIXES)
        else:
            return None
    except OSError:
        return None
    if size <= 0:
        return None
    return max(1, math.ceil(size / 2 ** 20))

#: ``<registry_root>/trained`` — the gateway's ``TRAINED_DIRNAME``.
TRAINED_DIRNAME = "trained"
SUFFIX = ".yaml"

#: A venv directory name under the gateway's ``.venvs/``, and nothing that could
#: climb out of it. The gateway's ``_VENV_NAME``, restated (serving 58f678a).
VENV_NAME_RE = re.compile(r"[A-Za-z0-9._-]+")


class RegistrationError(RuntimeError):
    """A registration could not be written or read. Carries the path and why."""

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason


class VramMeasurement(BaseModel):
    """The gateway's ``VramMeasurement`` (serving#130), field for field.

    Never written by a runner: it is what ``scripts/measure_vram.py`` on the
    gateway records against a card. It is here so a registration an operator
    has measured by hand is one ``set_enabled`` can still rewrite — the same
    reason ``max_pixels`` is here (see :class:`Registration`).
    """

    model_config = ConfigDict(extra="forbid")

    total_mib: int
    weights_mib: int | None = None
    kv_cache_mib: int | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    host: str
    measured_at: str
    gpu: int | None = None
    note: str | None = None


class Registration(BaseModel):
    """One trained model, as the gateway's ``ModelSpec`` reads it.

    The field names, types and defaults are the gateway's (``atr_serving/
    registry.py`` on serving-atr-inference main at 567c125, 21.10.2026). Unknown
    fields are **refused**, the opposite of
    :class:`~atr_training.shared_registry.BaseEntry`: the reader here ignores
    what it does not need, but a writer that emitted a field the gateway does not
    know would have it dropped without a word — and a misspelt ``enabeld: true``
    would be a model that never gets promoted.

    The price of refusing is that a field the gateway gains and this class lacks
    turns a file the gateway serves into one this repo cannot touch. It happened
    within the hour: ``max_pixels`` reached the gateway (serving 3d74ba3) twelve
    minutes before this class was written without it, and ``set_enabled`` then
    refused every registration an operator had given a pixel budget. The field
    list is therefore pinned against a copy of the gateway's
    (``tests/fixtures/registry/modelspec_fields.txt``); refresh that copy when
    the gateway's ``registry.py`` changes, and the test says what to add here.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    engine: Literal["vllm", "trocr", "kraken", "party"]
    #: One line naming the weights, for ``/models`` (serving#101). Written by
    #: hand or from a Zenodo title there; no runner here invents one.
    description: str | None = None
    hf_repo: str | None = None
    zenodo_id: str | None = None
    local_path: str | None = None
    enabled: bool = True
    disabled_reason: str | None = None
    base_model: str | None = None
    task: Literal["ocr", "htr"] = "ocr"
    level: Literal["page", "line"] = "page"
    languages: list[str] = Field(default_factory=list)
    scripts: list[str] = Field(default_factory=list)
    centuries: list[int] = Field(default_factory=list)
    #: The weights in MiB (:func:`weights_mib`), or absent. Never 0: until #225
    #: the default was ``0`` and no register stage set the field, so every
    #: trained registration carried a number where there was no measurement,
    #: and each reader had to guess whether it meant "small" or "unknown" (the
    #: gateway reads ``<= 0`` as missing, the MCP's headroom says "unrecorded,
    #: not zero", ``/models`` shows the 0). A 0 read from an existing file is
    #: that legacy and becomes None; the writer omits None (``exclude_none``),
    #: and the gateway's own default takes over, as it did for the 0.
    vram_mb: int | None = Field(default=None, gt=0)
    #: The measurement behind ``vram_mb`` when the gateway took one (serving#130).
    vram_measured: VramMeasurement | None = None
    max_new_tokens: int | None = None
    #: Pixels one image may carry into the model; None = the level's default on
    #: the gateway. The VLM runner writes the budget the job trained at, because
    #: serving at another scale is a silent distribution shift (serving#140).
    max_pixels: int | None = None
    residency: Literal["pinned", "lazy"] = "lazy"
    gpu_affinity: int | None = None
    prompt: str | None = None
    #: The venv the gateway launches this model's vLLM from, as a directory name
    #: under its ``.venvs/`` — ``vllm-next`` for a model whose vLLM the older venv
    #: cannot serve (serving#132). Written by hand today; a trained model gets one
    #: when the run's base needs it.
    vllm_venv: str | None = None
    #: ``--max-num-seqs`` for that launch: how many sequences the server batches.
    max_num_seqs: int | None = Field(default=None, ge=1)
    #: The weights' identity (#259, serving#200): sha256 over the hashed files'
    #: lines, with ``host`` and ``measured_at``. Measured at the promotion gate
    #: on the file the gate just proved servable, and written in the same
    #: rewrite as ``enabled: true`` (:func:`set_enabled`). Absent means nobody
    #: measured — never "unversioned". The gateway recomputes it from ``files``
    #: on load and refuses one that does not follow, as :class:`WeightsDigest`
    #: does here first.
    weights_digest: WeightsDigest | None = None
    training_datasets: list[str] = Field(default_factory=list)

    @field_validator("vram_mb", mode="before")
    @classmethod
    def _zero_is_unrecorded(cls, value: Any) -> Any:
        # Every registration written before #225 carried ``vram_mb: 0`` from
        # the old default: five of the six files under trained/ when
        # :func:`size_weights` (#242) first ran on asteraix on 14.10.2026 (#225
        # had guessed thirteen). Read as "unrecorded", so a promotion by hand
        # (``set_enabled`` rewrites the file as it is) drops the 0 rather than
        # refusing a file the gateway has been serving for weeks. A 0 can still
        # arrive from an old copy or a hand-written entry; this keeps reading it.
        if value == 0:
            return None
        return value

    @model_validator(mode="after")
    def _check(self) -> "Registration":
        # The id becomes the file name. The contracts rule already keeps job
        # model ids boring; here it also keeps them out of what the gateway never
        # reads — a leading dot would be a registration that is silently ignored.
        if not MODEL_ID_RE.match(self.id):
            raise ValueError(
                f"id {self.id!r} must match {MODEL_ID_RE.pattern}: it is the file name "
                "under trained/")
        # The gateway's own rule (ModelSpec._check_source).
        if not self.hf_repo and not self.zenodo_id and not self.local_path:
            raise ValueError(f"model {self.id!r}: needs one of hf_repo, zenodo_id or local_path")
        # The gateway skips a relative local_path (shared_registry._local_path_is_usable):
        # it names a different file on every machine. Refused here, before writing,
        # rather than discovered as an ERROR line in another machine's log.
        if self.local_path is not None and not Path(self.local_path).is_absolute():
            raise ValueError(
                f"local_path {self.local_path!r} is relative. It must be an absolute path "
                "under the shared mount, valid on both machines.")
        # The gateway's own rules for the two vLLM launch fields
        # (ModelSpec._check_vllm_venv). Kept here so a registration this repo
        # writes cannot be one the gateway refuses to read.
        if self.engine != "vllm":
            for field in ("vllm_venv", "max_num_seqs"):
                if getattr(self, field) is not None:
                    raise ValueError(
                        f"model {self.id!r}: {field} is only meaningful for engine vllm")
        if self.vllm_venv is not None and (
                not VENV_NAME_RE.fullmatch(self.vllm_venv) or self.vllm_venv in {".", ".."}):
            raise ValueError(
                f"model {self.id!r}: vllm_venv {self.vllm_venv!r} must be a directory "
                "name under .venvs/ (letters, digits, '.', '_', '-')")
        return self


# ── paths ───────────────────────────────────────────────────────────────────
def trained_dir(root: str | Path) -> Path:
    return Path(root) / TRAINED_DIRNAME


def registration_path(root: str | Path, model_id: str) -> Path:
    return trained_dir(root) / f"{model_id}{SUFFIX}"


def is_registration_name(name: str) -> bool:
    """The gateway's rule for which names in ``trained/`` it reads at all."""
    return name.endswith(SUFFIX) and not name.startswith(".")


def _tmp_path(target: Path) -> Path:
    # A dot in front AND .tmp at the end: either alone keeps the gateway off it.
    # Host, pid and a random part: the directory is shared by two machines, and a
    # fixed name would let two writers (a register and a promotion, or two
    # trainers) interleave their bytes in one tmp file and rename the mixture.
    unique = f"{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    return target.with_name(f".{target.name}.{unique}.tmp")


# ── validating ──────────────────────────────────────────────────────────────
def _validate(spec: dict[str, Any], where: Path) -> Registration:
    try:
        return Registration.model_validate(spec)
    except ValidationError as exc:
        raise RegistrationError(where, f"not a valid registration: {exc}") from None


def _document(spec: Registration) -> str:
    header = (
        "# One trained model (training-atr-models#14), read by the gateway from the\n"
        "# shared registry (serving-atr-inference#138). The file name is the id.\n"
        f"# Written by the trainer on {socket.gethostname()} at "
        f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}.\n"
    )
    body = yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True),
                          sort_keys=False, allow_unicode=True)
    return header + body


# ── writing ─────────────────────────────────────────────────────────────────
def _replace_atomically(target: Path, text: str) -> None:
    """Write ``text`` to ``target`` via a sibling tmp file. Raises RegistrationError."""
    directory = target.parent
    try:
        # trained/ and nothing above it. An unmounted share leaves an empty
        # mountpoint, and mkdir(parents=True) would quietly build the tree on the
        # local disk and register into it — where the gateway never looks. The
        # gateway's publish_curated makes the same choice for the registry itself.
        directory.mkdir(exist_ok=True)
    except FileNotFoundError:
        raise RegistrationError(
            target, f"{directory.parent} does not exist. Is the share mounted, and is "
                    "ATR_TRAIN_REGISTRY_ROOT the gateway's ATR_REGISTRY_ROOT?") from None
    except OSError as exc:
        raise RegistrationError(
            target, f"cannot create {directory} ({type(exc).__name__}: {exc})") from exc

    tmp = _tmp_path(target)
    try:
        # write_text, not copy2 or chmod: the bytes are all that matter, and on
        # this mount anything that touches mode or times is EPERM for a non-owner.
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
    except OSError as exc:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise RegistrationError(
            target, f"could not be written ({type(exc).__name__}: {exc})") from exc


def write_registration(root: str | Path, spec: dict[str, Any]) -> Path:
    """Register one model: ``<root>/trained/<id>.yaml``, atomically.

    ``spec`` is validated before anything is touched — a refused registration
    creates neither the file nor ``trained/``. Writing the same id again
    replaces that file and no other. Returns the path written.
    """
    where = trained_dir(root)
    registration = _validate(spec, where)
    target = registration_path(root, registration.id)
    if registration.local_path is not None and not os.path.exists(registration.local_path):
        # Not refused: the gateway serves it regardless and says so, because on
        # CIFS the weights and the registration can become visible in either
        # order. Here it is the one cheap moment to notice a wrong path.
        logger.warning("registering {} with local_path {}, which does not exist on {}",
                       registration.id, registration.local_path, socket.gethostname())
    _replace_atomically(target, _document(registration))
    logger.info("registered {} in {} (enabled: {})", registration.id, target,
                registration.enabled)
    return target


def read_registration(root: str | Path, model_id: str) -> Registration | None:
    """The registration for ``model_id``, or None if there is none.

    Raises :class:`RegistrationError` for a file that exists but cannot be used
    — the same checks the gateway makes, so what reads here is what it serves.
    """
    path = registration_path(root, model_id)
    raw = _load(path)
    if raw is None:
        return None
    return _validate(raw, path)


def _load(path: Path) -> dict[str, Any] | None:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise RegistrationError(path, f"cannot be read ({type(exc).__name__}: {exc})") from exc
    if not isinstance(raw, dict):
        raise RegistrationError(
            path, f"expected ONE model as a mapping, got {type(raw).__name__}")
    if raw.get("id") != path.name.removesuffix(SUFFIX):
        raise RegistrationError(
            path, f"declares id {raw.get('id')!r}; the gateway skips a file not named "
                  "after its id")
    return raw


def set_enabled(root: str | Path, model_id: str, enabled: bool = True, *,
                weights_digest: WeightsDigest | None = None) -> bool:
    """Rewrite ``enabled`` in one registration. False if ``model_id`` has none.

    What the promotion gate is allowed to write, and nothing else: it proves a
    model can be served, so it may advertise that model — and, since #259, say
    which bytes it served, the *weights_digest* it measured on the way. One
    rewrite carries both, so the file never says "enabled" with the digest of
    an earlier promotion beside it. Only this one file is rewritten — every
    other registration is left byte for byte, and so is every other field of
    this one (read as it is, not rebuilt from defaults). A digest already in
    the file is kept when none is passed: disabling a model does not forget
    what its weights were.
    """
    path = registration_path(root, model_id)
    raw = _load(path)
    if raw is None:
        return False
    raw["enabled"] = enabled
    if weights_digest is not None:
        raw["weights_digest"] = weights_digest.model_dump(mode="json", exclude_none=True)
    _replace_atomically(path, _document(_validate(raw, path)))
    logger.info("{}: enabled is now {}{}", path, enabled,
                f", weights {weights_digest.short()}" if weights_digest else "")
    return True


# ── sizing what was registered before #225 ─────────────────────────────────
#
# Every registration the trainer wrote before #236 carries ``vram_mb: 0`` from
# the old default. Both readers understand the 0 as "unrecorded" by now, but the
# files are unchanged, and for a kraken or trocr model among them there is a
# number that could stand there since #236: :func:`weights_mib` of its
# ``local_path``, the rule every new registration is written with. Until #242 an
# old model got its number only when somebody rewrote the file for another
# reason, and even then only the 0 went away.

@dataclass(frozen=True)
class Sizing:
    """What :func:`size_weights` did, or would do, to one registration."""

    model_id: str
    engine: str
    #: ``vram_mb`` as the file carries it (None = absent) and as it ends up.
    before: int | None
    after: int | None
    #: ``sized`` (a number written), ``dropped`` (the 0 removed, no number),
    #: ``kept`` (a positive or measured value, untouched), ``unchanged`` (nothing
    #: to size and nothing to drop), ``skipped`` (a file this module cannot
    #: read or write; the reason says why).
    action: str
    reason: str
    #: False on a dry run, and for every action that writes nothing.
    written: bool = False

    def line(self) -> str:
        before = "-" if self.before is None else str(self.before)
        after = "-" if self.after is None else str(self.after)
        mark = "" if self.written or self.action not in {"sized", "dropped"} else " (dry run)"
        return (f"{self.model_id:40s} {self.engine:7s} {before:>6} -> {after:<6} "
                f"{self.action:9s} {self.reason}{mark}")


def size_weights(root: str | Path, *, write: bool = False) -> list[Sizing]:
    """One pass over ``trained/``: replace ``vram_mb: 0`` with the weights' size.

    Per file, with ``vram_mb`` read as the file carries it:

    * a positive value, or a ``vram_measured`` block beside it, is **kept** — a
      number somebody measured on a card beats one weighed on disk, and the
      gateway's ``scripts/measure_vram.py`` owns the measured ones;
    * otherwise (0, or absent) a kraken, trocr or party model whose
      ``local_path`` has weight files here is **sized** to :func:`weights_mib`;
    * otherwise a 0 is **dropped**: for a vLLM adapter because the merge on
      idhefix writes the merged size (serving-atr-inference#216), for the rest
      because "could not size" is not "needs nothing" (#225);
    * a file absent of ``vram_mb`` with nothing to size is **unchanged**.

    Only the one field changes, through the same read, validation and atomic
    write as :func:`set_enabled`; a file this module cannot read or write is
    **skipped** and the reason says why. With ``write`` False nothing is
    touched and each entry says what would happen. Sorted by file name, so two
    runs read the same.
    """
    directory = trained_dir(root)
    try:
        names = sorted(n for n in os.listdir(directory) if is_registration_name(n))
    except FileNotFoundError:
        return []
    results = []
    for name in names:
        path = directory / name
        try:
            raw = _load(path)
        except RegistrationError as exc:
            results.append(Sizing(name.removesuffix(SUFFIX), "?", None, None, "skipped",
                                  exc.reason))
            continue
        if raw is None:  # withdrawn between the listing and the read
            continue
        results.append(_size_one(path, raw, write=write))
    return results


def _size_one(path: Path, raw: dict[str, Any], *, write: bool) -> Sizing:
    model_id = str(raw.get("id"))
    engine = str(raw.get("engine"))
    before = raw.get("vram_mb")
    present = "vram_mb" in raw
    if isinstance(before, int) and before > 0:
        return Sizing(model_id, engine, before, before, "kept", "declared or measured")
    if "vram_measured" in raw:
        return Sizing(model_id, engine, before, before, "kept",
                      "measured (vram_measured); scripts/measure_vram.py owns it")
    before = before if isinstance(before, int) else None

    local_path = raw.get("local_path")
    size = None
    if engine != "vllm" and isinstance(local_path, str):
        size = weights_mib(local_path)

    if size is not None:
        action, after = "sized", size
        reason = f"weights at {local_path}"
    elif engine == "vllm":
        if not present:
            return Sizing(model_id, engine, None, None, "unchanged",
                          "a merge on idhefix writes the merged size (serving#216)")
        action, after = "dropped", None
        reason = "an adapter is not what is served; the merge writes the merged size (serving#216)"
    else:
        where = local_path if isinstance(local_path, str) else "no local_path"
        reason = f"no weight files to size at {where} on {socket.gethostname()}"
        if not present:
            return Sizing(model_id, engine, None, None, "unchanged", reason)
        action, after = "dropped", None

    if not write:
        return Sizing(model_id, engine, before, after, action, reason)
    if after is None:
        raw.pop("vram_mb", None)
    else:
        raw["vram_mb"] = after
    try:
        _replace_atomically(path, _document(_validate(raw, path)))
    except RegistrationError as exc:
        return Sizing(model_id, engine, before, before, "skipped", exc.reason)
    logger.info("{}: vram_mb {} -> {} ({})", path, before, after, reason)
    return Sizing(model_id, engine, before, after, action, reason, written=True)


# ── by hand ─────────────────────────────────────────────────────────────────
def manual_registration(root: str | Path, spec: dict[str, Any]) -> str:
    """A shell command that registers ``spec`` — for the message of a failed register.

    Spelled with this interpreter and this ``src/``, so it can be pasted as it
    stands: the runner's venv has no ``atr_training`` installed, and the unit
    puts ``src/`` on ``PYTHONPATH`` (deploy/systemd/atr-train.service), which an
    operator's shell does not.
    """
    # Plain values only: this runs on the way to failing a job, and a YAML error
    # here would replace the message that says where the weights are.
    plain = {k: v if isinstance(v, (str, int, float, bool, list, type(None))) else str(v)
             for k, v in spec.items()}
    body = yaml.safe_dump(plain, sort_keys=False, allow_unicode=True)
    src = Path(__file__).resolve().parents[1]
    return (f"PYTHONPATH={shlex.quote(str(src))} {shlex.quote(sys.executable)} "
            f"-m atr_training.registration --root {shlex.quote(str(root))} <<'EOF'\n"
            f"{body}EOF")


def main(argv: list[str] | None = None) -> int:
    """Register one model by hand: the spec as YAML on stdin.

    The same validation and the same atomic write as the trainer — a copy made
    with an editor is a half-written file for as long as the editor takes.
    """
    parser = argparse.ArgumentParser(
        description="Write <root>/trained/<id>.yaml from one model spec (YAML) on stdin, "
                    "or, with --size-weights, replace every vram_mb: 0 under trained/ "
                    "with the size of the weights (#242).")
    parser.add_argument("--root", required=True, type=Path,
                        help="the registry root (ATR_TRAIN_REGISTRY_ROOT)")
    parser.add_argument("--size-weights", action="store_true",
                        help="list every registration with what sizing would do to its "
                             "vram_mb; nothing is written without --write")
    parser.add_argument("--write", action="store_true",
                        help="with --size-weights: rewrite the registrations that change")
    args = parser.parse_args(argv)
    if args.write and not args.size_weights:
        parser.error("--write goes with --size-weights")
    if args.size_weights:
        results = size_weights(args.root, write=args.write)
        if not results:
            print(f"nothing registered under {trained_dir(args.root)}")
            return 0
        for result in results:
            print(result.line())
        changed = sum(1 for r in results if r.action in {"sized", "dropped"})
        skipped = sum(1 for r in results if r.action == "skipped")
        if args.write:
            print(f"{changed} rewritten, {skipped} skipped, {len(results)} read")
        else:
            print(f"{changed} would change, {skipped} skipped, {len(results)} read "
                  "— nothing written; re-run with --write")
        return 1 if skipped else 0
    try:
        spec = yaml.safe_load(sys.stdin.read())
    except yaml.YAMLError as exc:
        print(f"not YAML: {exc}", file=sys.stderr)
        return 2
    if not isinstance(spec, dict):
        print("expected ONE model as a YAML mapping on stdin", file=sys.stderr)
        return 2
    try:
        print(write_registration(args.root, spec))
    except RegistrationError as exc:
        print(f"not registered: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

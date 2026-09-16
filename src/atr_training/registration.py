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
import os
import shlex
import socket
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from atr_training.contracts import MODEL_ID_RE

__all__ = [
    "TRAINED_DIRNAME",
    "Registration",
    "RegistrationError",
    "is_registration_name",
    "manual_registration",
    "read_registration",
    "registration_path",
    "set_enabled",
    "trained_dir",
    "write_registration",
]

#: ``<registry_root>/trained`` — the gateway's ``TRAINED_DIRNAME``.
TRAINED_DIRNAME = "trained"
SUFFIX = ".yaml"


class RegistrationError(RuntimeError):
    """A registration could not be written or read. Carries the path and why."""

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason


class Registration(BaseModel):
    """One trained model, as the gateway's ``ModelSpec`` reads it.

    The field names, types and defaults are the gateway's (``atr_serving/
    registry.py`` on serving-atr-inference main at 4158bcf, 16.09.2026). Unknown
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
    vram_mb: int = 0
    max_new_tokens: int | None = None
    #: Pixels one image may carry into the model; None = the level's default on
    #: the gateway. The VLM runner writes the budget the job trained at, because
    #: serving at another scale is a silent distribution shift (serving#140).
    max_pixels: int | None = None
    residency: Literal["pinned", "lazy"] = "lazy"
    gpu_affinity: int | None = None
    prompt: str | None = None
    training_datasets: list[str] = Field(default_factory=list)

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


def set_enabled(root: str | Path, model_id: str, enabled: bool = True) -> bool:
    """Rewrite ``enabled`` in one registration. False if ``model_id`` has none.

    The only thing the promotion gate is allowed to write: it proves a model can
    be served, so it may advertise that model and nothing else. Only this one
    file is rewritten — every other registration is left byte for byte, and so
    is every other field of this one (read as it is, not rebuilt from defaults).
    """
    path = registration_path(root, model_id)
    raw = _load(path)
    if raw is None:
        return False
    raw["enabled"] = enabled
    _replace_atomically(path, _document(_validate(raw, path)))
    logger.info("{}: enabled is now {}", path, enabled)
    return True


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
        description="Write <root>/trained/<id>.yaml from one model spec (YAML) on stdin.")
    parser.add_argument("--root", required=True, type=Path,
                        help="the registry root (ATR_TRAIN_REGISTRY_ROOT)")
    args = parser.parse_args(argv)
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

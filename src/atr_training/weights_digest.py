"""The identity of a trained model's weights, as the gateway defines it (#259).

The gateway's ``ModelSpec`` carries a ``weights_digest`` since
serving-atr-inference#200: ``/models`` hands it out, and a digest that does not
follow from its own file list or lacks its provenance **does not load** there.
What the gateway cannot do is measure one — every self-trained model reaches it
through ``trained/<id>.yaml`` on the share, written here, and the one moment the
weights are in hand is the promotion gate.

This module is that measurement. It is the gateway's ``atr_serving/weights.py``
restated, not imported: ``tests/test_isolation.py`` forbids the import, and the
two implementations are held to one answer by a pinned vector instead —
``tests/fixtures/weights_digest/contract.json``, byte-identical in both repos,
the way ``/host`` and ``/gpu`` are (#13, #250). If
``test_the_pinned_vector_still_produces_the_pinned_digest`` fails, the
definition moved and every recorded value in both repos means something else.

The rule the vector pins
------------------------
* **sha256 over the lines** ``<relative name> <sha256> <size>``, sorted by name
  — not over the concatenated bytes. A re-shard changes the digest even when
  the parameters are identical: the conservative direction, on purpose, because
  #200 exists because two runs over one page were indistinguishable after a
  finetune was rewritten.
* **Relative names**, because the share hangs at different paths on asteraix
  and idhefix and identical bytes must give one answer.
* **Weight files only** (:data:`WEIGHT_SUFFIXES`) — ``config.json`` and
  ``metadata.json`` are not weights. This list is the gateway's, and it is
  narrower than :data:`atr_training.registration.WEIGHT_SUFFIXES` (which sizes
  ``.gguf`` too); the digest follows the gateway, because the gateway recomputes
  it from the file list.
* **CoreML alone** when a directory holds it: that is the file
  ``kraken_loader.resolve_weights`` serves, and a ``.safetensors`` beside it
  would version a file kraken 7.0.2 cannot load.

Three-valued, like everything here: no digest means **nobody measured**, never
"unversioned" and never a digest over nothing. :func:`weight_files` raises
rather than returning ``[]``, and :class:`WeightsDigest` refuses to exist
without files or without ``host``/``measured_at``.
"""

from __future__ import annotations

import hashlib
import socket
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "COREML_SUFFIX",
    "WEIGHT_SUFFIXES",
    "WeightFile",
    "WeightsDigest",
    "WeightsUnreadable",
    "combined_digest",
    "digest_weights",
    "weight_files",
]

#: Files whose bytes are the model — the gateway's list, verbatim.
WEIGHT_SUFFIXES = (".mlmodel", ".safetensors", ".bin", ".pt", ".pth", ".ckpt")

#: Preferred alone when a directory holds it — see the module note on the loader.
COREML_SUFFIX = ".mlmodel"

#: Read in blocks rather than whole: a merged 27B checkpoint is tens of GB.
_BLOCK = 1024 * 1024


class WeightsUnreadable(FileNotFoundError):
    """The path names no weights this host can hash."""


class WeightFile(BaseModel):
    """One hashed file, named relative to the model's directory."""

    name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class WeightsDigest(BaseModel):
    """The weights' identity, with what it was taken over and by whom.

    The same shape the gateway's ``WeightsDigest`` validates: a value that
    reaches ``trained/<id>.yaml`` from here loads there, or it is refused here
    first by the same rule.
    """

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    #: Every file hashed, in the order they were hashed (sorted by name).
    files: list[WeightFile]
    #: Where and when. A digest is about bytes on one machine at one time.
    host: str
    measured_at: str
    #: What was resolved — the path as the trainer saw it. For the reader, not
    #: for comparison: it differs from the gateway's for the same bytes.
    source: str | None = None
    #: Free text: a re-merge, a conversion, anything that explains a change.
    note: str | None = None

    @model_validator(mode="after")
    def _digest_follows_from_its_files(self) -> "WeightsDigest":
        if not self.files:
            raise ValueError(
                "weights_digest lists no files: a digest over nothing is not a "
                "measurement of anything"
            )
        expected = combined_digest(self.files)
        if self.sha256 != expected:
            raise ValueError(
                f"weights_digest {self.sha256[:12]}… does not follow from the "
                f"{len(self.files)} file(s) it lists (they give {expected[:12]}…). "
                "A digest nobody measured reads like one somebody did, which is "
                "worse than none."
            )
        return self

    @property
    def total_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.files)

    def short(self) -> str:
        """The form a log line carries."""
        return self.sha256[:12]


def combined_digest(files: list[WeightFile]) -> str:
    """The digest over *files* — name, hash and size, one line each, sorted."""
    lines = sorted(f"{entry.name} {entry.sha256} {entry.size_bytes}" for entry in files)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def weight_files(path: str | Path) -> list[Path]:
    """The files that *are* this model, in the order they will be hashed.

    Raises :class:`WeightsUnreadable` rather than returning an empty list: "no
    weights here" is a finding, and a digest over zero files would be a valid
    hash of nothing.
    """
    root = Path(path).expanduser()
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise WeightsUnreadable(
            f"{root} does not exist on {socket.gethostname()}: a digest is a "
            "measurement on a file, and there is none here"
        )
    coreml = sorted(p for p in root.rglob(f"*{COREML_SUFFIX}") if p.is_file())
    if coreml:
        return coreml
    found = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix in WEIGHT_SUFFIXES)
    if not found:
        raise WeightsUnreadable(
            f"{root} exists but holds no {', '.join(WEIGHT_SUFFIXES)} file — "
            "nothing here is weights"
        )
    return found


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def digest_weights(path: str | Path, *, note: str | None = None,
                   host: str | None = None, now: datetime | None = None) -> WeightsDigest:
    """Hash the weights at *path* and say what was hashed, where and when.

    The one place a :class:`WeightsDigest` is produced in this repo. Raises
    :class:`WeightsUnreadable` (a ``FileNotFoundError``) when there is nothing
    to hash; the caller writes no field then, never an empty one.
    """
    paths = weight_files(path)
    root = Path(path).expanduser()
    base = root if root.is_dir() else root.parent
    entries = [
        WeightFile(name=str(each.relative_to(base)), sha256=_file_digest(each),
                   size_bytes=each.stat().st_size)
        for each in paths
    ]
    stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    return WeightsDigest(
        sha256=combined_digest(entries), files=entries,
        host=host or socket.gethostname(), measured_at=stamp,
        source=str(root), note=note,
    )

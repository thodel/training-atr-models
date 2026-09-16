"""The model registry, read from the research share (#5).

Until 16.09.2026 the trainer imported the gateway's own `atr_serving.registry`
to resolve a ``base_model`` given as a registry id. After the split that import
is gone, and the registry is **shared as a file**, not as Python: the gateway
publishes the curated ``models.yaml`` to the share (serving-atr-inference#138),
and this module reads it.

Why a file on the share and not the two alternatives:

- **a copy in this repo** runs out of date without a word, and the failure shows
  up weeks later as "unknown base model" on a request that used to work;
- **``GET /models`` on the gateway** filters on ``enabled`` (``routes.py:157``).
  A model can be disabled for serving on idhefix and still be a perfectly good
  starting point for a fine-tune. The trainer must see it.

What this reads is deliberately narrow. The gateway's ``ModelSpec`` has some
twenty fields; resolving a base needs four. Unknown fields are ignored rather
than rejected, so the gateway can grow its schema without breaking the trainer —
the contract that matters is pinned by ``tests/test_shared_registry.py`` against
a snapshot of the published file.

Measured on the 48 job records of 16.09.: only **kraken** ever names a registry
id (12 of 26 kraken jobs — 11 × ``kraken-early_modern_german``, 1 ×
``kraken-medieval_generic_b``). vllm and trocr always name a HuggingFace repo
and never touch this module.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

__all__ = ["BaseEntry", "RegistryUnavailable", "SharedRegistry", "load_shared_registry"]


class RegistryUnavailable(RuntimeError):
    """The registry file could not be read. Carries the path and the reason."""

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason


class BaseEntry(BaseModel):
    """The four fields a base-model lookup needs, and nothing else."""

    model_config = ConfigDict(extra="ignore")

    id: str
    engine: str
    zenodo_id: str | None = None
    local_path: str | None = None
    #: Carried, not filtered on. A disabled model is still a valid base.
    enabled: bool = True


class SharedRegistry:
    """Lookups over the published registry. Duck-types what base_models needs."""

    def __init__(self, entries: list[BaseEntry], path: Path | None = None) -> None:
        self.path = path
        self._by_id = {e.id: e for e in entries}

    def __len__(self) -> int:
        return len(self._by_id)

    def get(self, model_id: str) -> BaseEntry | None:
        return self._by_id.get(model_id)

    def by_engine(self, engine: str) -> list[BaseEntry]:
        return [e for e in self._by_id.values() if e.engine == engine]


def load_shared_registry(path: str | Path) -> SharedRegistry:
    """Read the published ``models.yaml``. Raises :class:`RegistryUnavailable`.

    Raising rather than returning an empty registry is the point. An empty
    registry would turn "the file is not there" into "that id does not exist",
    and the user would go looking for a typo that is not theirs — the same trap
    `load_heldout` sets by returning an empty set for a missing file.
    """
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RegistryUnavailable(p, "no such file — has the gateway published it?") from None
    except (OSError, yaml.YAMLError) as exc:
        raise RegistryUnavailable(p, f"{type(exc).__name__}: {exc}") from exc

    items = raw.get("models") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise RegistryUnavailable(p, "expected a list of models, or a mapping with 'models'")
    entries = []
    for item in items:
        try:
            entries.append(BaseEntry.model_validate(item))
        except ValueError as exc:
            raise RegistryUnavailable(p, f"malformed entry {item!r:.80}: {exc}") from exc
    return SharedRegistry(entries, path=p)

"""A sweep is a file: what to try, on which data, for how long (#113).

Before this, a sweep was a person with a terminal. The first architecture search
produced a ranking over seven heights and four LSTM widths, and when the data it
ran on was deleted on 16.09.2026 — rightly, it predated #89/#90 and the ground
truth correction that touched 28 % of the corpus — nothing recorded what those
numbers had actually measured. The ranking became unreadable, not wrong.

So the data version is a **required field**, not a comment, and it is folded into
every configuration's identity: the same hyperparameters on a different corpus
are a different measurement and must not share a row in a leaderboard.

**``config_id`` is a hash, not an index.** A sweep runs for days, is interrupted,
resumed and extended. If a configuration were identified by its position, adding
one value to one axis would rename every configuration after it, and the
leaderboard would compare things that are not the same. The pattern is the one
:mod:`atr_training.artefact_cache` already uses for compiled corpora: canonical
JSON, sha256, and a version constant so that old ids are never served under new
meaning.

What this module does **not** do is run anything. It reads a file and answers
three questions — which configurations, in which ladder, on which data — and
#114 is what turns that into jobs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from atr_training.contracts import KrakenTrainParams, VlmTrainParams
from atr_training.rungs import DEFAULT_ETA, plan_rungs

__all__ = [
    "SweepError",
    "SWEEP_KEY_VERSION",
    "DataVersion",
    "SweepConfig",
    "SweepRung",
    "Sweep",
    "NOT_MEASURED",
    "config_id_for",
    "effective_params",
    "load_sweep",
]

#: Bumped when the meaning of a ``config_id`` changes, so ids from an older
#: definition are never compared with new ones. Same reasoning as
#: ``artefact_cache.KEY_VERSION``, and the same alternative rejected: a migration
#: nobody runs.
SWEEP_KEY_VERSION = 1

#: Which params model validates a resolved configuration, per engine. Validation
#: at load time is the point: an axis named ``lr`` instead of ``lrate`` would
#: otherwise resolve to a field nothing reads, every configuration would train
#: identically, and the sweep would report a flat ranking as a finding.
_PARAMS: dict[str, Any] = {"kraken": KrakenTrainParams, "vllm": VlmTrainParams}

#: Fields that reach the trainer but do not change what is measured, and are
#: therefore left out of ``config_id``. Moving a sweep to a machine with a
#: different worker count or a second card must not rename every configuration
#: in the leaderboard. The inverse of the artefact cache's rule, and for the same
#: reason: an identity is over what the number depends on, nothing else.
NOT_MEASURED = frozenset({"workers", "device"})


class SweepError(ValueError):
    """Raised when a sweep file cannot be read as a sweep."""


@dataclass(frozen=True)
class DataVersion:
    """One dataset the sweep reads, and the digest that pins it."""

    role: str
    sha256: str
    path: str | None = None
    #: For an evaluation set, its durable name — ``german-medieval-v1`` rather
    #: than a path that differs per machine (see ``docs/EVAL_SETS.md``).
    name: str | None = None

    def describe(self) -> dict:
        return {"role": self.role, "sha256": self.sha256, "name": self.name}


@dataclass(frozen=True)
class SweepConfig:
    """One point of the search space."""

    config_id: str
    #: The full parameter set handed to the trainer.
    params: dict
    #: Only the values that differ across the sweep — what a leaderboard column
    #: needs. Derived, never authored.
    axes: dict


@dataclass(frozen=True)
class SweepRung:
    """One budget level: how many configurations, and how many optimizer steps.

    Steps, never epochs. h256 costs about twelve times what h48 costs per step,
    so a budget counted in epochs hands the expensive configurations less compute
    and returns a ranking of cost. The ladder's *shape* still comes from
    :func:`atr_training.rungs.plan_rungs`, which is tested; only the unit is
    translated here.
    """

    index: int
    configs: int
    steps: int

    def __str__(self) -> str:
        return f"rung {self.index}: {self.configs} configs × {self.steps} steps"


@dataclass(frozen=True)
class Sweep:
    name: str
    engine: str
    data: tuple[DataVersion, ...]
    configs: tuple[SweepConfig, ...]
    rungs: tuple[SweepRung, ...]
    #: The smallest difference this material can resolve, from #115. ``None``
    #: until it has been measured — and a ranking read without it is exactly how
    #: the first sweep went wrong.
    noise_floor: float | None = None
    source: Path | None = field(default=None, compare=False)

    @property
    def data_digest(self) -> str:
        """One digest over every dataset the sweep reads, in role order."""
        payload = json.dumps([d.describe() for d in sorted(self.data, key=lambda d: d.role)],
                             sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def __str__(self) -> str:
        floor = f", noise floor {self.noise_floor}" if self.noise_floor is not None else ""
        return (f"sweep {self.name}: {len(self.configs)} configs, "
                f"{len(self.rungs)} rungs, data {self.data_digest[:12]}{floor}")


def effective_params(params: Mapping[str, Any], engine: str) -> dict:
    """The configuration as the trainer will see it: defaults filled in.

    Hashing what was *written* would make an axis whose value happens to be the
    default a different configuration from leaving it out — same training, two
    rows in the leaderboard. The identity has to be over the effective
    configuration, so every manifest that means the same run produces the same
    id, however it is written.
    """
    model = _PARAMS[engine]
    return model(**dict(params)).model_dump(mode="json")


def config_id_for(params: Mapping[str, Any], data_digest: str, engine: str) -> str:
    """The stable identity of one configuration on one data version.

    Twelve hex characters: enough that a collision inside one sweep is not a
    practical concern, short enough to read in a table.
    """
    effective = effective_params(params, engine)
    payload = json.dumps(
        {
            "version": SWEEP_KEY_VERSION,
            "engine": engine,
            "params": {k: v for k, v in sorted(effective.items()) if k not in NOT_MEASURED},
            "data": data_digest,
        },
        sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _require(raw: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in raw or raw[key] in (None, "", [], {}):
        raise SweepError(f"{where}: {key!r} is required")
    return raw[key]


def _read_data(raw: Any, source: str) -> tuple[DataVersion, ...]:
    """The data block — and the refusal that gives this module its point.

    A sweep without a pinned data version is refused rather than accepted with a
    warning, because the failure it causes is silent and arrives months later: a
    table of numbers whose corpus cannot be named.
    """
    if not isinstance(raw, Mapping) or not raw:
        raise SweepError(f"{source}: 'data' must name at least one dataset")
    out: list[DataVersion] = []
    for role in sorted(raw):
        entry = raw[role]
        if not isinstance(entry, Mapping):
            raise SweepError(f"{source}: data.{role} must be a mapping with a sha256")
        digest = entry.get("sha256")
        if not digest:
            raise SweepError(
                f"{source}: data.{role} has no sha256. The data version is a required "
                "field, not a comment: the first architecture sweep lost the meaning of "
                "its whole ranking when the corpus behind it was deleted."
            )
        out.append(DataVersion(role=role, sha256=str(digest),
                               path=entry.get("path"), name=entry.get("name")))
    return tuple(out)


def _resolve(base: Mapping[str, Any], axes: Mapping[str, Iterable[Any]],
             ) -> list[tuple[dict, dict]]:
    """Cross product of the axes over the base. Returns ``(params, varying)``.

    Axis names are sorted so the enumeration order does not depend on how the
    file was written; ``config_id`` does not depend on order at all, but a stable
    order keeps logs and leaderboards diffable.
    """
    names = sorted(axes)
    value_lists = []
    for name in names:
        values = axes[name]
        if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            raise SweepError(f"axis {name!r} must be a list of values, got {values!r}")
        values = list(values)
        if not values:
            raise SweepError(f"axis {name!r} is empty, so the sweep has no configurations")
        value_lists.append(values)

    # An axis with one value is not a variable: it belongs in the leaderboard's
    # header, not in a column where every row is identical.
    varying_names = {n for n, v in zip(names, value_lists) if len(v) > 1}

    out: list[tuple[dict, dict]] = []
    for combination in product(*value_lists) if names else [()]:
        params = dict(base)
        params.update(dict(zip(names, combination)))
        varying = {n: v for n, v in zip(names, combination) if n in varying_names}
        out.append((params, varying))
    return out


def _validate(params: Mapping[str, Any], engine: str, where: str) -> None:
    model = _PARAMS.get(engine)
    if model is None:
        raise SweepError(f"{where}: unknown engine {engine!r}, expected one of "
                         f"{sorted(_PARAMS)}")
    unknown = sorted(set(params) - set(model.model_fields))
    if unknown:
        # pydantic ignores extra fields, so a typo would train every
        # configuration identically and the sweep would report the resulting flat
        # ranking as a finding.
        raise SweepError(
            f"{where}: {', '.join(unknown)} is not a parameter of {model.__name__}. "
            "A misspelled axis does not fail — it produces a sweep in which every "
            "configuration is the same one."
        )
    try:
        model(**params)
    except Exception as exc:  # pydantic's own message is the useful one
        raise SweepError(f"{where}: {exc}") from None


def load_sweep(path: str | Path) -> Sweep:
    """Read a sweep file. Every refusal here is one that would otherwise surface
    as a wrong number hours or weeks later."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise SweepError(f"{path}: cannot read ({exc})") from None
    except yaml.YAMLError as exc:
        raise SweepError(f"{path}: not valid YAML ({exc})") from None
    if not isinstance(raw, Mapping):
        raise SweepError(f"{path}: expected a mapping at the top level")

    where = str(path)
    name = str(_require(raw, "name", where))
    engine = str(raw.get("engine", "kraken"))
    data = _read_data(_require(raw, "data", where), where)

    base = dict(raw.get("base") or {})
    axes = dict(raw.get("axes") or {})
    if not axes and not base:
        raise SweepError(f"{where}: neither 'base' nor 'axes' — nothing to try")

    budget = dict(raw.get("budget") or {})
    base_steps = budget.get("steps")
    if not base_steps or int(base_steps) < 1:
        raise SweepError(f"{where}: budget.steps is required and counts **optimizer "
                         "steps** for the first rung, not epochs")
    base_steps = int(base_steps)
    eta = int(budget.get("eta", DEFAULT_ETA))

    resolved = _resolve(base, axes)
    digest_payload = json.dumps([d.describe() for d in sorted(data, key=lambda d: d.role)],
                                sort_keys=True, separators=(",", ":")).encode("utf-8")
    data_digest = hashlib.sha256(digest_payload).hexdigest()

    configs: list[SweepConfig] = []
    seen: dict[str, dict] = {}
    for params, varying in resolved:
        _validate(params, engine, where)
        cid = config_id_for(params, data_digest, engine)
        if cid in seen:
            raise SweepError(f"{where}: two configurations resolve identically "
                             f"({cid}); an axis is repeated in 'base'")
        seen[cid] = params
        configs.append(SweepConfig(config_id=cid, params=params, axes=varying))

    ladder = plan_rungs(len(configs), eta=eta, base_epochs=1)
    rungs = tuple(
        SweepRung(index=r.index, configs=r.configs, steps=base_steps * r.epochs)
        for r in ladder
    )

    floor = raw.get("noise_floor")
    return Sweep(name=name, engine=engine, data=data, configs=tuple(configs),
                 rungs=rungs, noise_floor=None if floor is None else float(floor),
                 source=path)

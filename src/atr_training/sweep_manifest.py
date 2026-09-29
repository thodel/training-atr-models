"""A sweep is a file, and a configuration is a hash of what it resolves to (#113).

The step #31 named last ("Next: S1 sweep manifest with stable config_id") and
never reached. A sweep runs over days, is interrupted, resumed and extended, so a
configuration has to stay the same thing across all of that:

    name: kraken-medieval-augment-01
    data:
      train:  <path>
      eval:   german_test
      digest: sha256:…            # from #112 — mandatory, not a comment
    budget:
      steps: 4000                 # optimizer steps, never epochs
      rungs: [12, 4, 1]
    base:
      spec: "[1,48,0,1 Cr3,3,32 …]"
      lrate: 1.0e-4
      schedule: cosine
      batch_size: 256
      augment: false
      normalization: NFD
    axes:
      augment:       [false, true]
      normalization: [NFD, NFC]
      lrate:         [3.0e-5, 1.0e-4, 3.0e-4]

Three things here are not obvious, and each one is a measured failure rather than
a preference.

**Numbers are canonicalised before they are hashed.** PyYAML resolves `1e-4` to
the *string* ``'1e-4'`` — its float pattern requires a decimal point — while
`1.0e-4` and `0.0001` both resolve to the float. So the notation in #113's own
example would have given one configuration two ids depending on how somebody
happened to type it, which is precisely what K1's third acceptance criterion
forbids. :func:`canonical` folds numeric strings to numbers and integral floats
to ints, so `1e-4`, `1.0e-4` and `0.0001` are one value, and `256`, `256.0` and
`"256"` are one value.

**An axis must name a key that ``base`` already carries.** It forces every swept
parameter to have a declared default, and it buys the stability the issue asks
for: adding a value to an axis leaves every other configuration's resolved
parameters untouched, and promoting a `base` value to an axis leaves *that*
configuration's id unchanged. What no scheme can do is keep an id stable when a
parameter appears that was never specified before — a run that did not pin
`normalization` is not the same run as one that pinned it to NFD, and saying so
would be the kind of quiet mislabelling this file exists to prevent.

**The budget is in steps.** An epoch budget hands more optimizer steps to a
smaller batch size and fewer to a larger one, so a sweep with `batch_size` on an
axis would rank by how long each configuration got. A `budget.epochs` key is
refused rather than translated, because the translation depends on the data
version and would be silent.

The id deliberately does **not** include the sweep's name: the same parameters on
the same data are the same experiment, and two sweeps that overlap should be able
to see that they do.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "ManifestError",
    "MANIFEST_VERSION",
    "SweepConfig",
    "SweepManifest",
    "canonical",
    "config_id",
    "load_manifest",
    "parse_manifest",
]

#: Bumped when the meaning of a hashed field changes, so a sweep resumed under a
#: new version gets new ids rather than silently comparing across definitions.
#: Same role as ``artefact_cache.KEY_VERSION``, and the same reason.
MANIFEST_VERSION = 1

#: As long as ``artefact_cache``'s short digests, for the same reason: it is
#: pasted into issues and log lines.
ID_CHARS = 12

#: What may appear at the top level. A typo'd key is refused rather than ignored:
#: `budgets:` silently dropping the budget is the failure this is cheap to avoid.
TOP_LEVEL = frozenset({"name", "data", "budget", "base", "axes", "notes"})

#: Python's ``float()`` also accepts ``nan``, ``inf`` and ``1_000``; YAML does
#: not mean those, so the match is a pattern rather than a ``try``.
NUMERIC = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


class ManifestError(ValueError):
    """A manifest that cannot be read as one experiment."""


def _as_number(text: str) -> int | float | None:
    if not NUMERIC.match(text.strip()):
        return None
    value = float(text)
    return int(value) if value.is_integer() else value


def canonical(value: Any) -> Any:
    """One spelling per value, so equal manifests hash equally.

    ``bool`` is checked before ``int`` on purpose: in Python ``True`` *is* an
    int, and ``float(True)`` is ``1.0``, so an unguarded numeric branch would
    turn ``augment: true`` into ``augment: 1`` and lose the distinction between
    a flag and a count.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, str):
        number = _as_number(value)
        return value if number is None else number
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if value is None:
        return None
    raise ManifestError(f"a manifest cannot carry a {type(value).__name__}: {value!r}")


def _payload(params: Mapping[str, Any], data_digest: str) -> str:
    return json.dumps(
        {"version": MANIFEST_VERSION, "data": data_digest, "params": canonical(params)},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def config_id(params: Mapping[str, Any], *, data_digest: str) -> str:
    """The identity of one configuration: its resolved parameters and its data.

    Not its position in the file, which is what #31 would have had: inserting a
    value into an axis renames every configuration after it, and the leaderboard
    then compares two different things under one name.
    """
    return hashlib.sha256(_payload(params, data_digest).encode("utf-8")).hexdigest()[:ID_CHARS]


@dataclass(frozen=True)
class SweepConfig:
    """One point of the cross product."""

    config_id: str
    #: Everything the trainer needs: ``base`` with this point's axis values on top.
    params: dict
    #: Just the axis values, which is what a leaderboard puts in its columns —
    #: the base is the same for every row and belongs in the table's header.
    axes: dict

    def __str__(self) -> str:
        shown = ", ".join(f"{k}={v}" for k, v in sorted(self.axes.items()))
        return f"{self.config_id}  {shown}" if shown else self.config_id


@dataclass(frozen=True)
class SweepManifest:
    name: str
    train: str
    eval: str
    data_digest: str
    steps: int
    rungs: tuple[int, ...]
    base: dict
    axes: dict[str, tuple]
    source: str | None = None

    def configs(self) -> list[SweepConfig]:
        """The full cross product, in a fixed order.

        Ordered by axis name rather than by the order they were written, so two
        manifests that differ only in how the axes are arranged produce the same
        list in the same sequence — an id that is stable but a *listing* that is
        not would still make two runs look different in every report.
        """
        names = sorted(self.axes)
        out: list[SweepConfig] = []
        for point in itertools.product(*(self.axes[n] for n in names)):
            chosen = dict(zip(names, point))
            params = canonical({**self.base, **chosen})
            out.append(SweepConfig(config_id=config_id(params, data_digest=self.data_digest),
                                   params=params, axes=canonical(chosen)))
        return out


def _require(raw: Mapping, key: str, where: str) -> Any:
    if key not in raw or raw[key] in (None, "", [], {}):
        raise ManifestError(f"{where}: `{key}` is required")
    return raw[key]


def _axis_values(name: str, values: Any) -> tuple:
    if not isinstance(values, (list, tuple)):
        raise ManifestError(f"axes.{name}: expected a list of values, got "
                            f"{type(values).__name__}")
    if not values:
        raise ManifestError(
            f"axes.{name}: an empty axis makes the cross product empty, so the "
            "sweep would run nothing. Remove the axis or give it values.")
    folded = [canonical(v) for v in values]
    seen: dict[str, Any] = {}
    for original, value in zip(values, folded):
        key = json.dumps(value, sort_keys=True)
        if key in seen:
            raise ManifestError(
                f"axes.{name}: {original!r} and {seen[key]!r} are the same value "
                f"once canonicalised ({value!r}). They would produce two "
                "configurations with one id, and the second result would "
                "overwrite the first.")
        seen[key] = original
    return tuple(folded)


def _budget(raw: Mapping, n_configs: int) -> tuple[int, tuple[int, ...]]:
    if "epochs" in raw:
        raise ManifestError(
            "budget.epochs: the budget is in optimizer steps, not epochs. An "
            "epoch budget gives a smaller batch size more steps than a larger "
            "one, so a sweep with batch_size on an axis ranks by how much "
            "training each configuration got (#111, lesson 3). Convert it "
            "yourself — the conversion depends on the data version, and doing "
            "it here would hide that.")

    steps = _require(raw, "steps", "budget")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ManifestError(f"budget.steps: expected a positive whole number, got {steps!r}")

    rungs_raw = raw.get("rungs")
    if rungs_raw is None:
        return steps, ()
    if (not isinstance(rungs_raw, (list, tuple)) or not rungs_raw
            or any(not isinstance(r, int) or isinstance(r, bool) or r < 1 for r in rungs_raw)):
        raise ManifestError(f"budget.rungs: expected positive whole numbers, got {rungs_raw!r}")
    rungs = tuple(rungs_raw)
    for earlier, later in zip(rungs, rungs[1:]):
        if later >= earlier:
            raise ManifestError(
                f"budget.rungs: {rungs} does not narrow — {later} follows "
                f"{earlier}. Successive halving keeps the top fraction at each "
                "rung, so every rung holds fewer configurations than the last.")
    if rungs[-1] != 1:
        raise ManifestError(
            f"budget.rungs: {rungs} ends at {rungs[-1]}, so the sweep stops with "
            "more than one survivor and never names a winner.")
    if rungs[0] > n_configs:
        raise ManifestError(
            f"budget.rungs: rung 0 wants {rungs[0]} configurations and the axes "
            f"define {n_configs}. Widen the axes or lower the ladder — a rung "
            "that starts short of its width eliminates nothing.")
    return steps, rungs


def parse_manifest(raw: Mapping, *, source: str | None = None) -> SweepManifest:
    """Validate and resolve, refusing rather than warning.

    A sweep is days of GPU time; every complaint here costs seconds and every
    one waved through costs a re-run.
    """
    where = source or "manifest"
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{where}: expected a mapping at the top level, got "
                            f"{type(raw).__name__}")
    unknown = sorted(set(raw) - TOP_LEVEL)
    if unknown:
        raise ManifestError(
            f"{where}: unknown top-level {'keys' if len(unknown) > 1 else 'key'} "
            f"{', '.join(unknown)} — known: {', '.join(sorted(TOP_LEVEL))}. A "
            "misspelt key would otherwise be dropped in silence.")

    name = str(_require(raw, "name", where))
    data = _require(raw, "data", where)
    if not isinstance(data, Mapping):
        raise ManifestError(f"{where}: `data` must be a mapping")

    if "digest" not in data or not str(data.get("digest") or "").strip():
        raise ManifestError(
            f"{where}: data.digest is required. The first sweep recorded no data "
            "version, and when the corpus was deleted on 16.09.2026 nobody could "
            "say what its numbers had measured (#111, lesson 4). Get it from "
            "`scripts/restore_eval_split.py` (#112).")

    base = dict(_require(raw, "base", where)) if raw.get("base") else {}
    if not isinstance(base, dict):
        raise ManifestError(f"{where}: `base` must be a mapping")

    axes_raw = raw.get("axes") or {}
    if not isinstance(axes_raw, Mapping):
        raise ManifestError(f"{where}: `axes` must be a mapping of name to values")
    undeclared = sorted(set(axes_raw) - set(base))
    if undeclared:
        raise ManifestError(
            f"{where}: {', '.join(undeclared)} "
            f"{'are axes' if len(undeclared) > 1 else 'is an axis'} with no entry "
            "in `base`. Every swept parameter needs a declared default: without "
            "one there is nothing to compare an extension against, and a "
            "configuration that simply did not pin the parameter cannot be told "
            "apart from one that pinned it to this value.")
    axes = {str(k): _axis_values(str(k), v) for k, v in axes_raw.items()}

    n_configs = 1
    for values in axes.values():
        n_configs *= len(values)
    steps, rungs = _budget(_require(raw, "budget", where), n_configs)

    return SweepManifest(
        name=name,
        train=str(_require(data, "train", f"{where}.data")),
        eval=str(_require(data, "eval", f"{where}.data")),
        data_digest=str(data["digest"]).strip(),
        steps=steps,
        rungs=rungs,
        base=canonical(base),
        axes=axes,
        source=source,
    )


def load_manifest(path: str | Path) -> SweepManifest:
    import yaml

    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ManifestError(f"no sweep manifest at {path}") from None
    except yaml.YAMLError as exc:
        raise ManifestError(f"{path} is not readable as YAML: {exc}") from exc
    return parse_manifest(raw or {}, source=str(path))


def describe(manifest: SweepManifest, configs: Sequence[SweepConfig] | None = None
             ) -> Iterable[str]:
    """Lines for a human about to spend days of GPU on this."""
    configs = list(configs if configs is not None else manifest.configs())
    yield f"sweep   {manifest.name}"
    yield f"data    {manifest.train} / {manifest.eval}"
    yield f"digest  {manifest.data_digest}"
    yield f"budget  {manifest.steps} optimizer steps per configuration"
    if manifest.rungs:
        yield f"rungs   {' → '.join(str(r) for r in manifest.rungs)}"
    yield f"configs {len(configs)} from {len(manifest.axes)} axes"
    yield ""
    for config in configs:
        yield f"  {config}"

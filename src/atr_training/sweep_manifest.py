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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "BASE_AXIS",
    "ManifestError",
    "distinguishing",
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
TOP_LEVEL = frozenset({"name", "data", "budget", "base", "axes", "notes",
                       "noise_floor", "baseline",
                       "engine", "base_model"})

#: Python's ``float()`` also accepts ``nan``, ``inf`` and ``1_000``; YAML does
#: not mean those, so the match is a pattern rather than a ``try``.
NUMERIC = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


#: Axis values longer than this are shown by what tells them apart.
LABEL_MAX = 28

#: The one axis that is not a ketos parameter: what each cell starts from.
#: #118 lists "fine-tuning instead of from scratch" among the untried axes and
#: says why it comes first — "a fine-tune of a fitting base can beat any
#: architecture variant from scratch, and then the whole search space is the
#: wrong question". Its default is the manifest's top-level ``base_model``, and
#: ``null`` in the axis means from scratch.
BASE_AXIS = "base_model"


class ManifestError(ValueError):
    """A manifest that cannot be read as one experiment."""


def distinguishing(values: Sequence[Any]) -> dict[str, str]:
    """Map each value to the shortest text that tells it from its siblings.

    A height axis is three whole VGSL specs — ``ketos`` takes the spec, not a
    height — and printing them in full makes a twelve-row listing unreadable and
    a leaderboard column impossible. They differ in one number, so that is what
    is shown: ``…,64,…`` against ``…,128,…``.

    Only for long strings, and only where a common prefix and suffix actually
    exist; anything else is returned as it stands, because an abbreviation that
    hides the difference is worse than a long cell.
    """
    texts = [str(v) for v in values]
    labels = {text: text for text in texts}
    if len(set(texts)) < 2 or max(len(t) for t in texts) <= LABEL_MAX:
        return labels

    head = 0
    while all(t[head:head + 1] == texts[0][head:head + 1] and head < len(t) for t in texts):
        head += 1
    tail = 0
    while all(tail < len(t) - head and t[len(t) - tail - 1] == texts[0][len(texts[0]) - tail - 1]
              for t in texts):
        tail += 1

    middles = {text: text[head:len(text) - tail] for text in texts}
    if len(set(middles.values())) < len(set(texts)) or any(not m for m in middles.values()):
        return labels                      # the difference is not in one place
    if max(len(m) for m in middles.values()) > LABEL_MAX // 2:
        # The shared prefix and suffix are too short to be worth removing: the
        # label would be nearly the whole value with ellipses stuck on, which is
        # longer than the value and no easier to read.
        return labels
    return {text: f"…{middle}…" for text, middle in middles.items()}


def _as_number(text: str) -> int | float | None:
    if not NUMERIC.match(text.strip()):
        return None
    value = float(text)
    return int(value) if value.is_integer() else value


def _noise_floor(raw: Mapping[str, Any], where: str,
                 data_digest: str) -> tuple[float | None, dict]:
    """The measured resolution of this material, if it has been measured (#115).

    Two shapes, and the difference between them is whether anybody can tell what
    the number was measured on:

        noise_floor: 0.0085                        # a bare number

        noise_floor:                               # measured, and says so
          value: 0.0085
          measured_on: "sha256:0123…"              # must equal data.digest
          seeds: [42, 43, 44, 45]
          commit: "1a429b3…"
          steps: 2000

    A floor is a property of a corpus **and** a budget — #115 measured 0.0085 on
    one configuration and watched another move 0.1924 — so a number carried over
    from other material licenses a ranking it never earned. When the block names
    a `measured_on`, it has to be this sweep's data version or the manifest is
    refused. A bare number is still accepted, because the field was defined that
    way, but it travels with `provenance: "unstated"` so the state file and the
    leaderboard can say that nobody knows where it came from rather than printing
    it as if somebody did.

    Refused rather than coerced when it is not a positive number: a floor of zero
    would mark every cut as decided outside the noise, which is the reassurance
    this field exists to withhold.
    """
    value = raw.get("noise_floor")
    if value is None:
        return None, {}

    provenance: dict = {}
    if isinstance(value, Mapping):
        block = dict(value)
        if "value" not in block:
            raise ManifestError(
                f"{where}: noise_floor is a block without a `value`. Write the "
                "measured spread there, or give the bare number instead.")
        measured_on = str(block.get("measured_on") or "").strip()
        if measured_on and measured_on != data_digest:
            raise ManifestError(
                f"{where}: noise_floor was measured on {measured_on!r} and this "
                f"sweep runs on {data_digest!r}. A floor is a property of the "
                "corpus and the budget it was measured at — #115 measured 0.0085 "
                "on one configuration and 0.1924 on another — so a number from "
                "other material would license a ranking it never earned. Measure "
                "it again on this corpus, or remove the field.")
        provenance = {k: v for k, v in block.items() if k != "value"}
        provenance.setdefault("provenance", "measured" if measured_on else "unstated")
        value = block["value"]
    else:
        provenance = {"provenance": "unstated"}

    try:
        floor = float(value)
    except (TypeError, ValueError):
        raise ManifestError(f"{where}: noise_floor must be a number, got {value!r}") from None
    if not floor > 0:
        raise ManifestError(
            f"{where}: noise_floor must be greater than zero, got {floor}. A floor of "
            "zero marks every cut as decided outside the noise, which is the "
            "reassurance this field exists to withhold.")
    return floor, provenance


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


def _payload(params: Mapping[str, Any], data_digest: str,
             base_model: str | None = None) -> str:
    body: dict[str, Any] = {"version": MANIFEST_VERSION, "data": data_digest,
                            "params": canonical(params)}
    if base_model is not None:
        # Omitted when there is none, so every id written before `base_model`
        # could be swept is unchanged. A cell with no base and a cell from
        # before the field existed are the same configuration — training from
        # scratch — so collapsing them is the right answer and not a silent one.
        body["base_model"] = str(base_model)
    return json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def config_id(params: Mapping[str, Any], *, data_digest: str,
              base_model: str | None = None) -> str:
    """The identity of one configuration: its parameters, its data, its base.

    Not its position in the file, which is what #31 would have had: inserting a
    value into an axis renames every configuration after it, and the leaderboard
    then compares two different things under one name.

    ``base_model`` belongs in the identity even though it is not a parameter:
    a fine-tune of CATMuS and a run from scratch with the same hyperparameters
    are not the same experiment, and before this they would have shared an id —
    so the second result would have overwritten the first (#118).
    """
    return hashlib.sha256(
        _payload(params, data_digest, base_model).encode("utf-8")
    ).hexdigest()[:ID_CHARS]


@dataclass(frozen=True)
class SweepConfig:
    """One point of the cross product."""

    config_id: str
    #: Everything the trainer needs: ``base`` with this point's axis values on top.
    params: dict
    #: Just the axis values, which is what a leaderboard puts in its columns —
    #: the base is the same for every row and belongs in the table's header.
    axes: dict

    #: What this cell starts from: a base to fine-tune, or ``None`` for from
    #: scratch. Resolved, so a cell whose axis chose ``null`` is distinguishable
    #: from one in a sweep that has no base at all — they mean the same thing to
    #: the trainer, and the distinction matters to nobody reading the file.
    base_model: str | None = None

    #: axis name -> value -> how to print it. Set by ``SweepManifest.configs``,
    #: because an abbreviation is only meaningful against an axis's other values.
    labels: dict = field(default_factory=dict)

    def __str__(self) -> str:
        shown = ", ".join(
            f"{k}={self.labels.get(k, {}).get(str(v), v)}"
            for k, v in sorted(self.axes.items()))
        return f"{self.config_id}  {shown}" if shown else self.config_id


@dataclass(frozen=True)
class SweepManifest:
    name: str
    train: str
    eval: str
    data_digest: str
    #: The dataset specs every configuration trains on, verbatim. They are the
    #: data, so they live under `data` — and because every configuration gets the
    #: same list, they all hit one entry of the artefact cache and the corpus is
    #: compiled once for the whole sweep (#109).
    datasets: tuple[dict, ...]
    engine: str
    base_model: str | None
    steps: int
    rungs: tuple[int, ...]
    base: dict
    axes: dict[str, tuple]
    #: The smallest difference this material and this budget can resolve,
    #: measured by repeating one configuration across seeds (#115). Optional,
    #: because it is measured *on* a corpus and a budget and therefore cannot
    #: exist before them. Where it is absent, a ranking from this sweep has no
    #: resolution attached and must not be read as one.
    #: The most optimizer steps this sweep may ever spend, across every rung and
    #: every configuration. ``None`` means no ceiling — fine for a sweep somebody
    #: starts and watches, and wrong for one that runs continuously (#117): a
    #: process without an upper bound is not a process, it is a leak.
    max_total_steps: int | None = None
    noise_floor: float | None = None
    #: Where that floor came from: the seeds, the commit and the data version it
    #: was measured on, or ``{"provenance": "unstated"}`` for a bare number.
    noise_floor_provenance: dict = field(default_factory=dict)
    #: The number this sweep has to beat, and nothing else (#111). Optional,
    #: because a sweep can be run to see what happens; present, it is what turns
    #: "no configuration won" from silence into a finding.
    baseline: float | None = None
    #: Which model, which metric, which measurement set. #111's own opening
    #: paragraph is why this is not just a number: it holds 0.2131 against 0.111
    #: and 0.0680 and says the comparison is **not** established, because the
    #: sets are different.
    baseline_provenance: dict = field(default_factory=dict)
    source: str | None = None

    def configs(self) -> list[SweepConfig]:
        """The full cross product, in a fixed order.

        Ordered by axis name rather than by the order they were written, so two
        manifests that differ only in how the axes are arranged produce the same
        list in the same sequence — an id that is stable but a *listing* that is
        not would still make two runs look different in every report.
        """
        names = sorted(self.axes)
        labels = {name: distinguishing(self.axes[name]) for name in names}
        if BASE_AXIS in labels:
            # `base_model=None` is "from scratch", and a leaderboard column
            # reading `None` invites the reader to wonder what went missing.
            labels[BASE_AXIS] = {**labels[BASE_AXIS], "None": "from scratch"}
        out: list[SweepConfig] = []
        for point in itertools.product(*(self.axes[n] for n in names)):
            chosen = dict(zip(names, point))
            # `base_model` is a request field, not a ketos parameter, so it is
            # swept and shown but never passed as one.
            base = chosen.get(BASE_AXIS, self.base_model)
            params = canonical({k: v for k, v in {**self.base, **chosen}.items()
                                if k != BASE_AXIS})
            out.append(SweepConfig(
                config_id=config_id(params, data_digest=self.data_digest,
                                    base_model=base),
                params=params, axes=canonical(chosen),
                base_model=str(base) if base else None, labels=labels))
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


def _budget(raw: Mapping, n_configs: int) -> tuple[int, tuple[int, ...], int | None]:
    if "epochs" in raw:
        raise ManifestError(
            "budget.epochs: the budget is in optimizer steps, not epochs. An "
            "epoch budget gives a smaller batch size more steps than a larger "
            "one, so a sweep with batch_size on an axis ranks by how much "
            "training each configuration got (#111, lesson 3). Convert it "
            "yourself — the conversion depends on the data version, and doing "
            "it here would hide that.")

    ceiling = raw.get("max_total_steps")
    if ceiling is not None and (not isinstance(ceiling, int) or isinstance(ceiling, bool)
                                or ceiling < 1):
        raise ManifestError(
            f"budget.max_total_steps: expected a positive whole number, got {ceiling!r}")

    steps = _require(raw, "steps", "budget")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ManifestError(f"budget.steps: expected a positive whole number, got {steps!r}")

    rungs_raw = raw.get("rungs")
    if rungs_raw is None:
        return steps, (), ceiling
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
    if rungs[0] != n_configs:
        raise ManifestError(
            f"budget.rungs: rung 0 wants {rungs[0]} configurations and the axes "
            f"define {n_configs}. Rung 0 screens the whole field: a wider ladder "
            "has nothing to eliminate, and a narrower one would leave "
            "configurations unrun without saying which — they would be decided "
            "by the order the cross product happens to come out in.")
    return steps, rungs, ceiling


def _baseline(raw: Mapping[str, Any], where: str) -> tuple[float | None, dict]:
    """The number this sweep has to beat, and what it was measured on (#111).

    Two shapes, as with ``noise_floor``, and the difference is again whether
    anybody can say what the number means:

        baseline: 0.2131                           # a bare number

        baseline:                                  # and what it is
          model: kraken-medieval-german-v2
          value: 0.2131
          metric: benchmark_cer
          measured_on: german-medieval-v1
          chars: 882255
          errors: 188022

    ``metric`` is the field that does the work. #111 opens by putting 0.2131
    against 0.111 and 0.0680 and saying the comparison is **not** established
    because the sets differ — and a sweep ranking on ``cer`` (its own validation
    partition) against a baseline measured as ``benchmark_cer`` (held-out
    documents) is that same mistake one level down. The leaderboard refuses the
    comparison rather than printing a difference nobody can read.

    A bare number is accepted and travels as ``provenance: "unstated"``: it can
    still be shown, and it cannot be compared, because nothing says which metric
    it is.
    """
    value = raw.get("baseline")
    if value is None:
        return None, {}

    provenance: dict = {}
    if isinstance(value, Mapping):
        block = dict(value)
        if "value" not in block:
            raise ManifestError(
                f"{where}: baseline is a block without a `value`. Put the number "
                "there, or give the bare number instead.")
        provenance = {k: v for k, v in block.items() if k != "value"}
        if not str(provenance.get("metric") or "").strip():
            raise ManifestError(
                f"{where}: baseline needs a `metric`. 0.2131 on held-out "
                "documents and 0.2131 on a validation partition that overlaps "
                "the training projects are different claims, and without the "
                "field the leaderboard would compare them as if they were one "
                "(#111).")
        if not str(provenance.get("measured_on") or "").strip():
            raise ManifestError(
                f"{where}: baseline needs a `measured_on` naming the measurement "
                "set. #111's first table exists to show what happens without it.")
        provenance.setdefault("provenance", "stated")
        value = block["value"]
    else:
        provenance = {"provenance": "unstated"}

    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ManifestError(
            f"{where}: baseline {value!r} is not a number") from None
    if not 0 < number:
        raise ManifestError(
            f"{where}: baseline {number!r} must be positive — a target of zero "
            "or less is not one this sweep could miss.")
    return number, provenance


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
    undeclared = sorted(set(axes_raw) - set(base) - {BASE_AXIS})
    if undeclared:
        raise ManifestError(
            f"{where}: {', '.join(undeclared)} "
            f"{'are axes' if len(undeclared) > 1 else 'is an axis'} with no entry "
            "in `base`. Every swept parameter needs a declared default: without "
            "one there is nothing to compare an extension against, and a "
            "configuration that simply did not pin the parameter cannot be told "
            "apart from one that pinned it to this value.")
    if BASE_AXIS in axes_raw and "spec" in axes_raw:
        raise ManifestError(
            f"{where}: `spec` and `{BASE_AXIS}` cannot both be axes. `ketos "
            "train` ignores --spec when --load is given — the loaded network "
            "keeps its own geometry — so a fine-tune cell labelled h192 would "
            "have trained at whatever height the base has, and the leaderboard "
            "would carry a column of heights that half the rows never used. "
            "Sweep the geometry from scratch, or sweep the base; not both.")
    axes = {str(k): _axis_values(str(k), v) for k, v in axes_raw.items()}

    n_configs = 1
    for values in axes.values():
        n_configs *= len(values)
    steps, rungs, ceiling = _budget(_require(raw, "budget", where), n_configs)

    specs = _require(data, "datasets", f"{where}.data")
    if not isinstance(specs, (list, tuple)) or not specs:
        raise ManifestError(f"{where}: data.datasets must be a non-empty list of "
                            "dataset specs — they are what a job is submitted with")
    for index, spec in enumerate(specs):
        if not isinstance(spec, Mapping):
            raise ManifestError(f"{where}: data.datasets[{index}] must be a mapping, "
                                f"got {type(spec).__name__}")

    floor, floor_provenance = _noise_floor(raw, where, str(data["digest"]).strip())
    baseline, baseline_provenance = _baseline(raw, where)
    return SweepManifest(
        name=name,
        train=str(_require(data, "train", f"{where}.data")),
        eval=str(_require(data, "eval", f"{where}.data")),
        data_digest=str(data["digest"]).strip(),
        datasets=tuple(dict(s) for s in specs),
        engine=str(raw.get("engine") or "kraken"),
        base_model=(str(raw["base_model"]) if raw.get("base_model") else None),
        steps=steps,
        rungs=rungs,
        max_total_steps=ceiling,
        base=canonical(base),
        axes=axes,
        noise_floor=floor, noise_floor_provenance=floor_provenance,
        baseline=baseline, baseline_provenance=baseline_provenance,
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
    if BASE_AXIS in manifest.axes:
        shown = ", ".join(str(v) if v else "from scratch"
                          for v in manifest.axes[BASE_AXIS])
        yield f"base    an axis: {shown}"
    elif manifest.base_model:
        yield f"base    {manifest.base_model}"
    yield f"configs {len(configs)} from {len(manifest.axes)} axes"
    yield ""
    for config in configs:
        yield f"  {config}"

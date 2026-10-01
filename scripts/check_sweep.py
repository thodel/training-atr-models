#!/usr/bin/env python3
"""Read a sweep manifest, resolve it, and print what it would run (#113).

Between writing a manifest and spending days of GPU on it there should be a step
that costs seconds. This is that step: it applies every check `load_manifest`
makes and prints the cross product with each configuration's `config_id`, so the
ladder, the data version and the number of runs can be looked at before anything
is submitted.

    scripts/check_sweep.py config/sweeps/kraken-medieval-augment-01.yaml
    scripts/check_sweep.py <manifest> --ids        # ids alone, for a diff
    scripts/check_sweep.py <manifest> --registry config/models.yaml

`--ids` exists for the question a resumed sweep asks: which of these did we
already run? Two manifests whose id lists are equal describe the same
experiment, whatever their axes are called or how their numbers are spelt.

`--registry` answers the question a sweep with a `base_model` raises and cannot
answer itself: what has that base already seen? A fine-tune inherits its base's
training data, so a base that saw the held-out pages leaks the same way #100
did, one step further back where nobody looks. The registry is the only place
that could say, and for most kraken entries it says nothing — which is reported
as nothing, not as clean.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_training.base_models import provenance  # noqa: E402
from atr_training.shared_registry import (  # noqa: E402
    RegistryUnavailable, load_shared_registry,
)
from atr_training.sweep_manifest import (  # noqa: E402
    BASE_AXIS, ManifestError, describe, load_manifest,
)


def report_bases(manifest, registry_path) -> list[str]:
    """What each base this sweep starts from has already seen, if anybody said.

    Empty for a sweep that trains from scratch throughout: there is no base and
    therefore no inherited leak. Never fatal — this is a report, and a sweep
    that cannot reach a registry still has to be checkable.
    """
    bases = sorted({c.base_model for c in manifest.configs() if c.base_model})
    if not bases:
        return []
    registry, error = None, None
    if registry_path is not None:
        try:
            registry = load_shared_registry(registry_path)
        except RegistryUnavailable as exc:
            error = str(exc)
    elif BASE_AXIS in manifest.axes or manifest.base_model:
        error = "no --registry given"

    lines = ["", "bases this sweep fine-tunes from:"]
    for base in bases:
        lines.append("  " + provenance(base, registry, error).describe())
    if any(provenance(b, registry, error).state != "recorded" for b in bases):
        lines += [
            "",
            "  A fine-tune inherits what its base was trained on. Where that is "
            "unrecorded,",
            "  nobody can say the base has not seen the held-out documents — and "
            "a name is",
            "  not evidence: 28 of 43 kraken entries resolve to a different model "
            "than their",
            "  name says (serving-atr-inference#101). Resolve the DOI and record "
            "it before",
            "  a number from this sweep is published (#100).",
        ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--ids", action="store_true",
                        help="print only the config_ids, one per line")
    parser.add_argument("--registry", type=Path,
                        help="a models.yaml, to report what each base has seen")
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        configs = manifest.configs()
    except ManifestError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    if args.ids:
        print("\n".join(c.config_id for c in configs))
        return 0

    print("\n".join(describe(manifest, configs)))
    bases = report_bases(manifest, args.registry)
    if bases:
        print("\n".join(bases))
    if manifest.rungs:
        print(f"\nrung 0 runs {manifest.rungs[0]} of {len(configs)} configurations "
              f"for {manifest.steps} steps each.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

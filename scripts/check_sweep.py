#!/usr/bin/env python3
"""Read a sweep manifest, resolve it, and print what it would run (#113).

Between writing a manifest and spending days of GPU on it there should be a step
that costs seconds. This is that step: it applies every check `load_manifest`
makes and prints the cross product with each configuration's `config_id`, so the
ladder, the data version and the number of runs can be looked at before anything
is submitted.

    scripts/check_sweep.py config/sweeps/kraken-medieval-augment-01.yaml
    scripts/check_sweep.py <manifest> --ids        # ids alone, for a diff

`--ids` exists for the question a resumed sweep asks: which of these did we
already run? Two manifests whose id lists are equal describe the same
experiment, whatever their axes are called or how their numbers are spelt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_training.sweep_manifest import (  # noqa: E402
    ManifestError, describe, load_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--ids", action="store_true",
                        help="print only the config_ids, one per line")
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
    if manifest.rungs:
        print(f"\nrung 0 runs {manifest.rungs[0]} of {len(configs)} configurations "
              f"for {manifest.steps} steps each.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

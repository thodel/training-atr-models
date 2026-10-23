#!/usr/bin/env python3
"""Rewrite the two gateway fixtures from serving-atr-inference/main (#151).

    python scripts/refresh_gateway_fixtures.py          # fetch and rewrite
    python scripts/refresh_gateway_fixtures.py --check  # fetch and compare only

``tests/fixtures/registry/modelspec_fields.txt`` (the gateway's ``ModelSpec``
fields, which ``Registration`` must match exactly) and
``tests/fixtures/registry/models.yaml`` (the curated registry the base-model
tests resolve against) are copies, and copies made by hand went stale twice
without a word. This writes them from the gateway's ``main``, with the commit
and date in the header, so the refresh is one command and the header is never
a claim nobody checked. ``tests/test_gateway_drift.py`` runs the same
comparison on every test run that has the network.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_training.gateway_drift import (  # noqa: E402
    GATEWAY_REPO,
    MODELSPEC_SOURCE,
    REGISTRY_FILE,
    GatewayUnreachable,
    fetch_gateway_main,
    modelspec_fields,
    registry_entries,
)

FIXTURES = ROOT / "tests" / "fixtures" / "registry"
FIELDS_FILE = FIXTURES / "modelspec_fields.txt"
MODELS_FILE = FIXTURES / "models.yaml"


def fields_text(fields: list[str], commit: str, date: str) -> str:
    return (
        "# The fields of the gateway's ModelSpec, in its order: serving-atr-inference\n"
        f"# {MODELSPEC_SOURCE} at {commit[:7]} (main, {date[:10]}).\n"
        "#\n"
        "# Written by scripts/refresh_gateway_fixtures.py — do not edit by hand; re-run it.\n"
        "# tests/test_registration.py then says what atr_training.registration.Registration\n"
        "# has to gain or lose, and tests/test_gateway_drift.py compares this file with\n"
        "# the gateway's main whenever the network allows.\n"
        + "".join(f"{name}\n" for name in fields)
    )


def models_text(models_yaml: str, commit: str, date: str) -> str:
    return (
        f"# Copy of serving-atr-inference {REGISTRY_FILE} at {commit[:7]} (main, {date[:10]}),\n"
        "# written by scripts/refresh_gateway_fixtures.py — do not edit by hand; re-run it.\n"
        "# tests/test_gateway_drift.py compares the ids and engines with the gateway's main.\n"
        + models_yaml
    )


def current_fields() -> list[str]:
    lines = FIELDS_FILE.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="compare only; exit 1 on drift, 2 when the gateway is unreachable")
    args = parser.parse_args(argv)

    try:
        main_ = fetch_gateway_main()
    except GatewayUnreachable as exc:
        print(f"SKIP  {GATEWAY_REPO}@main is unreachable: {exc}")
        return 2
    fields = modelspec_fields(main_.registry_source)
    entries = registry_entries(main_.models_yaml)
    print(f"{GATEWAY_REPO}@{main_.commit[:7]} ({main_.date[:10]}): "
          f"{len(fields)} ModelSpec fields, {len(entries)} registry entries")

    drift = 0
    if current_fields() != fields:
        drift += 1
        print(f"DRIFT fields: fixture {current_fields()} vs main {fields}")
    have = registry_entries(MODELS_FILE.read_text(encoding="utf-8"))
    if have != entries:
        drift += 1
        gone = sorted(set(have) - set(entries))
        new = sorted(set(entries) - set(have))
        moved = sorted(k for k in set(have) & set(entries) if have[k] != entries[k])
        print(f"DRIFT registry: only in fixture {gone}; only on main {new}; engine changed {moved}")
    if args.check:
        print("ok    fixtures match main" if not drift else f"{drift} fixture(s) drifted")
        return 1 if drift else 0

    FIELDS_FILE.write_text(fields_text(fields, main_.commit, main_.date), encoding="utf-8")
    MODELS_FILE.write_text(models_text(main_.models_yaml, main_.commit, main_.date), encoding="utf-8")
    print(f"wrote {FIELDS_FILE} and {MODELS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

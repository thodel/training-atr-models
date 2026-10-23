"""What the gateway's ``main`` says, read from the gateway, not from a copy (#151).

Two files of serving-atr-inference are mirrored here as fixtures, and both have
gone stale without a word — twice for ``modelspec_fields.txt`` (#80, then
``description`` and ``vram_measured`` in #151), once for ``models.yaml`` (the
#198 renames). A copy that is refreshed by hand holds exactly as long as
somebody remembers; the only check that notices by itself is one that reads the
original.

This module reads it: the ``ModelSpec`` field list out of the gateway's
``registry.py`` (parsed, not imported — the gateway is not installed here, and
``tests/test_isolation.py`` keeps it that way) and the ids of the curated
``config/models.yaml``. ``scripts/refresh_gateway_fixtures.py`` uses it to
rewrite the fixtures, and ``tests/test_gateway_drift.py`` to compare them with
``main`` — **skipped** without the network, never passed. Stdlib only, so the
script runs in any interpreter.
"""

from __future__ import annotations

import ast
import json
import urllib.error
import urllib.request
from dataclasses import dataclass

__all__ = [
    "GATEWAY_REPO",
    "MODELSPEC_SOURCE",
    "REGISTRY_FILE",
    "GatewayMain",
    "GatewayUnreachable",
    "fetch_gateway_main",
    "modelspec_fields",
    "registry_entries",
]

GATEWAY_REPO = "thodel/serving-atr-inference"
MODELSPEC_SOURCE = "src/atr_serving/registry.py"
REGISTRY_FILE = "config/models.yaml"
_TIMEOUT_S = 15


class GatewayUnreachable(RuntimeError):
    """The gateway's repository could not be read — not a drift, an absence."""


@dataclass(frozen=True)
class GatewayMain:
    """The two files at one commit of the gateway's ``main``."""

    commit: str
    date: str
    registry_source: str
    models_yaml: str


def modelspec_fields(source: str, class_name: str = "ModelSpec") -> list[str]:
    """The annotated fields of *class_name* in *source*, in declaration order.

    Parsed with :mod:`ast` so the gateway need not be importable. Only
    annotated assignments at the class body's top level count — ``model_config``
    is a plain assignment, methods and validators are not fields — which is the
    same set ``ModelSpec.model_fields`` reports.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [stmt.target.id for stmt in node.body
                    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)]
    raise ValueError(f"no class {class_name} in the source")


def registry_entries(models_yaml: str) -> dict[str, str]:
    """``{id: engine}`` of every entry under ``models:``.

    Read without a YAML parser on purpose: the entries are a flat list of
    mappings whose first two keys are ``id`` and ``engine``, and the fixture is
    compared on exactly those — a value edit (a measured ``vram_mb``) is the
    gateway's business, a renamed or added id is what every test that resolves
    a base against the snapshot needs to know about.
    """
    out: dict[str, str] = {}
    current: str | None = None
    for raw in models_yaml.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("- id:"):
            current = stripped[len("- id:"):].strip().strip("'\"")
            out[current] = ""
        elif current is not None and stripped.startswith("engine:") and not out[current]:
            out[current] = stripped[len("engine:"):].strip().strip("'\"")
    if not out:
        raise ValueError("no `- id:` entries found: not the gateway's models.yaml")
    return out


def _read(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GatewayUnreachable(f"{url}: {exc}") from exc


def fetch_gateway_main(repo: str = GATEWAY_REPO) -> GatewayMain:
    """Both files and the commit they are at, from GitHub.

    The commit comes from the API first and the files from ``raw`` at that
    commit, so the three agree even if ``main`` moves between the requests.
    """
    api = _read(f"https://api.github.com/repos/{repo}/commits/main")
    try:
        head = json.loads(api)
        commit = head["sha"]
        date = head["commit"]["committer"]["date"]
    except (ValueError, KeyError, TypeError) as exc:
        raise GatewayUnreachable(f"unexpected answer for {repo}@main: {exc}") from exc
    base = f"https://raw.githubusercontent.com/{repo}/{commit}/"
    return GatewayMain(
        commit=commit, date=date,
        registry_source=_read(base + MODELSPEC_SOURCE).decode("utf-8"),
        models_yaml=_read(base + REGISTRY_FILE).decode("utf-8"),
    )

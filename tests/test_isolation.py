"""This repo does not import the serving half (#1, #5).

The only guarantee that the seam stays thin. Without it, one convenient
`from atr_serving import ...` six months from now quietly makes the two repos
one again — and nothing would fail until the day someone tried to deploy them
apart.

Checked on the source rather than on sys.modules: `atr_serving` is not
installed here, so an import of it would fail at collection anyway — but only
for the modules a test happens to import. This reads every file.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ROOTS = ("src", "engines", "ubelix", "scripts", "eval")


def _python_files():
    for root in ROOTS:
        base = REPO / root
        if base.is_dir():
            yield from base.rglob("*.py")


def _imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.lineno, node.module


def test_nothing_imports_the_serving_package():
    offenders = [
        f"{path.relative_to(REPO)}:{line}  {module}"
        for path in _python_files()
        for line, module in _imported_modules(path)
        if module == "atr_serving" or module.startswith("atr_serving.")
    ]
    assert not offenders, "the serving half is imported here:\n  " + "\n  ".join(offenders)


@pytest.mark.parametrize("module", ["atr_training.registry", "atr_training.overlay"])
def test_the_temporary_registry_copy_and_the_overlay_are_gone(module):
    """registry.py came over in #3 so the moved code would import; overlay.py
    wrote a file in this checkout that the gateway, on another machine, never
    read. #5 took the reading side off the copy and #14 the writing side: a
    registration is one file on the share (atr_training.registration), with a
    schema of this repo's own. Neither module may come back, nor an import of
    either.
    """
    assert importlib.util.find_spec(module) is None, f"{module} exists again"
    users = [
        f"{path.relative_to(REPO)}:{line}"
        for path in _python_files()
        for line, imported in _imported_modules(path)
        if imported == module or imported.startswith(module + ".")
    ]
    assert not users, f"{module} is imported by: {users}"

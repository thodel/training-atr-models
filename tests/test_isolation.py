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
from pathlib import Path

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


def test_the_temporary_registry_copy_is_only_used_for_writing():
    """registry.py came over in #3 so the moved code would import.

    #5 took the reading side off it — base-model lookups now read the file the
    gateway publishes to the share. What is left is the writing side, and #14
    removes that. Until then, nothing new may start depending on the copy.
    """
    allowed = {
        "src/atr_training/overlay.py",
        "engines/kraken_train_svc/runner.py",
        "engines/trocr_train_svc/runner.py",
        "engines/vlm_train_svc/runner.py",
    }
    users = {
        str(path.relative_to(REPO))
        for path in _python_files()
        for _, module in _imported_modules(path)
        if module == "atr_training.registry"
    }
    assert users <= allowed, f"new users of the temporary copy: {sorted(users - allowed)}"

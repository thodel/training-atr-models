"""The comments name this machine, not the one training used to share (#89 follow-up).

Every module here was written on the shared box and moved on 16.09.2026. What
moved with it was a layer of asides — "GPU 0 is the shared RAG GPU", "~356 GB
free", "a card we share with the serving engines", a pointer to
``docs/asteraix-environment.md`` — each of which reads perfectly and describes a
different machine. `preflight.py` carried one into an error message a user was
meant to act on.

Two sweeps, because nothing else notices when a comment goes out of date:

* the old spelling **asterAIx** always means the serving box, which is called
  idhefix since serving-atr-inference#136; in this repository's code it is
  therefore always either the wrong machine or the wrong name;
* a ``docs/NAME.md`` reference has to resolve **here**. The plans this code grew
  out of stayed in the serving repository, so a reference to one of them is
  written with its repository in front of it.

Neither sweep can tell whether a *number* is still true. They pin the two forms
the staleness took that were mechanical enough to catch.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
#: Code, not documents: the docs may discuss the old name and the other repo's
#: plans by name, and `docs/INFRASTRUCTURE.md` does exactly that.
CODE_DIRS = ("src", "engines", "scripts")
SUFFIXES = {".py", ".txt", ".sh", ".yaml", ".yml", ".toml"}


def code_files() -> list[Path]:
    out = []
    for directory in CODE_DIRS:
        for path in sorted((REPO / directory).rglob("*")):
            if path.is_file() and path.suffix in SUFFIXES and "__pycache__" not in path.parts:
                out.append(path)
    return out


FILES = code_files()
DOC_REF = re.compile(r"(?<![\w/-])docs/([A-Za-z0-9_.-]+\.md)")


def test_there_are_files_to_check():
    """A sweep over an empty list passes and proves nothing."""
    assert len(FILES) > 20


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_no_code_comment_says_asterAIx(path):
    text = path.read_text(encoding="utf-8")
    assert "asterAIx" not in text, (
        f"{path.relative_to(REPO)} uses the serving box's old name. It is idhefix "
        "(serving-atr-inference#136); this machine is asteraix."
    )


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_doc_references_resolve_in_this_repo(path):
    text = path.read_text(encoding="utf-8")
    missing = sorted({name for name in DOC_REF.findall(text)
                      if not (REPO / "docs" / name).exists()})
    assert not missing, (
        f"{path.relative_to(REPO)} points at {missing}, which this repository does "
        "not have. Write the repository in front of it, e.g. "
        "serving-atr-inference/docs/TRAINING_PLAN.md."
    )

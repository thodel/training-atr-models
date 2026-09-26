"""Models that carry the consequences of a corpus defect, and what became of them.

A fix to the corpus does not end a corpus defect. The code is corrected on a
Tuesday and the weights trained before it are still on the share, still in an
overlay, still cited in a table — and their CER still looks like a CER. #35 is
the case this module is built from: `pagexml.line_texts()` kept the first word of
every line and dropped the rest, 28 % of the medieval corpus by characters, and
the models trained on it learned to stop writing after a few characters.

That issue cannot be closed by its fix, and says so: it closes when **every**
model in its table has been retrained or explicitly marked superseded. Until now
that rule lived in the issue's prose and in hand-written ``notes`` on four
HuggingFace cards, so a model could be marked in one place and quoted in another
— which is exactly how the truncation survived long enough to reach three
trainings.

So the rule is data here, and :func:`outstanding` computes it. What the module
does **not** do is decide anything: it does not block a publish, disable a model
or rewrite a metric. The affected weights are real and sometimes the best thing
available for a task; a card that says what is wrong with them is the honest
treatment, not a refusal.

The registry is ``config/corpus_defects.json``, in the repo rather than on the
share, for the reasons :mod:`atr_training.heldout` gives for its own: it has to
be readable when the share is not, it is small, and a change to it is a change
somebody should have to review.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

__all__ = [
    "DEFAULT_REGISTRY",
    "STATUSES",
    "AffectedModel",
    "CorpusDefect",
    "defects_for",
    "load_defects",
    "outstanding",
]

#: Repo-relative, resolved from this file so a service started anywhere finds it.
DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "config" / "corpus_defects.json"

#: ``retrain_pending`` is the only one that keeps an issue open; the other two
#: are terminal. A status outside this set is a typo that would otherwise read as
#: "not pending" and quietly close the rule it was meant to keep open.
STATUSES = frozenset({"retrain_pending", "superseded", "withdrawn"})


@dataclass(frozen=True)
class AffectedModel:
    """One model that carries a defect, and where it stands."""

    model_id: str
    status: str
    successor: str | None = None
    where: str | None = None
    note: str | None = None

    @property
    def pending(self) -> bool:
        return self.status == "retrain_pending"

    def sentence(self) -> str:
        """What the model card says about it, in one line."""
        if self.status == "superseded" and self.successor:
            return f"Superseded by `{self.successor}`, retrained on the corrected corpus."
        if self.status == "superseded":
            return "Superseded by a retrain on the corrected corpus."
        if self.status == "withdrawn":
            return "Withdrawn — these weights are not published."
        return ("**No retrain exists yet.** Scores measured on this model reflect the "
                "defect as much as the model.")


@dataclass(frozen=True)
class CorpusDefect:
    """A defect in the training corpus, and every model that carries it."""

    id: str
    title: str
    issue: str | None = None
    fixed_by: dict = field(default_factory=dict)
    what: str = ""
    evidence: str = ""
    models: dict[str, AffectedModel] = field(default_factory=dict)

    @property
    def pending(self) -> list[AffectedModel]:
        return [m for m in self.models.values() if m.pending]

    def fixed_by_sentence(self) -> str:
        commit, repo = self.fixed_by.get("commit"), self.fixed_by.get("repo")
        date = self.fixed_by.get("date")
        if not commit:
            return "Not fixed yet."
        where = f"{repo}@{commit}" if repo else commit
        return f"Fixed in `{where}`" + (f" on {date}." if date else ".")


def _text(value) -> str:
    """A string, or a list of lines joined — the JSON carries prose either way."""
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    return str(value or "")


def load_defects(registry: str | Path | None = None) -> list[CorpusDefect]:
    """Read the registry. A missing or malformed file yields no defects.

    Tolerant for the same reason :func:`atr_training.heldout.load_heldout` is: a
    publish that dies on an unreadable side file helps nobody, and the shipped
    file's validity is a test's job, not a runtime's. The log line is the part
    that matters — silence here would mean cards quietly losing their warning.
    """
    path = Path(registry) if registry else DEFAULT_REGISTRY
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.debug("no corpus-defect registry at {}", path)
        return []
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("corpus-defect registry {} is unreadable ({}); model cards "
                       "will not carry their warnings", path, exc)
        return []

    defects: list[CorpusDefect] = []
    for entry in raw.get("defects") or []:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        models = {}
        for model_id, record in (entry.get("models") or {}).items():
            record = record if isinstance(record, dict) else {}
            models[model_id] = AffectedModel(
                model_id=model_id,
                status=str(record.get("status") or "retrain_pending"),
                successor=record.get("successor"),
                where=record.get("where"),
                note=record.get("note"),
            )
        defects.append(CorpusDefect(
            id=str(entry["id"]),
            title=str(entry.get("title") or entry["id"]),
            issue=entry.get("issue"),
            fixed_by=entry.get("fixed_by") or {},
            what=_text(entry.get("what")),
            evidence=_text(entry.get("evidence")),
            models=models,
        ))
    return defects


def defects_for(model_id: str, registry: str | Path | None = None
                ) -> list[tuple[CorpusDefect, AffectedModel]]:
    """Every defect this model carries, with its own record in each."""
    return [(defect, defect.models[model_id])
            for defect in load_defects(registry)
            if model_id in defect.models]


def outstanding(registry: str | Path | None = None) -> dict[str, list[str]]:
    """Defect id -> the models still waiting for a retrain.

    The closing rule of #35, as something a test can ask. An empty dict means
    every affected model has reached a terminal state and the issue may close —
    which is a fact about the registry, not about whether anybody remembered.
    """
    return {defect.id: sorted(m.model_id for m in defect.pending)
            for defect in load_defects(registry) if defect.pending}

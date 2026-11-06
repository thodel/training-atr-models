"""Scorers: language models that only score (agentic_historian#491 L1, #110).

A scorer is an artefact whose one interface is a number: ``bits_per_char(text)``.
It generates nothing, decodes nothing and is registered nowhere as a model
somebody could call. The separation sits in the type: this module has no
generation path, and a test holds it there (#110, acceptance criterion 6).

One scorer per **cell** ``(region, document_type, century_band, lang)`` — the
vocabulary of ``agent_a.model_selector.SourceCriteria`` in the historian, not a
second one. Scores of two cells are two scales, so the artefact carries the
distribution of its own held-out text and reports a **z-score** against it:
"2.1 spreads worse than this register's own text" reads across cells, bits/char
does not. A thin cell is worse than none, so a cell below ``min_chars`` is not
built and says so; a question no cell can answer falls back over the matrix in
the order the model selector falls back — century band, then genre, then
region, then the language alone — and the answer names the cell that answered.

Trained **only** on edition text and corrected ground truth, never on an engine's
output (agentic_historian#326, mode 1): a scorer trained on our readings learns
our errors and rewards them. And the same contamination rule as the held-out
guard (#23, #98): a scorer that has seen a document refuses to score it.

The model is a character n-gram with stupid backoff — the pre-experiment of
agentic_historian#491 in artefact form. It trains in seconds on the CPU, next
to the GPU job, and it is L1; a neural scorer is L5 and waits for L2's number.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev

__all__ = [
    "MIN_CHARS",
    "Cell",
    "CharNgramScorer",
    "ContaminationError",
    "Score",
    "ScorerMatrix",
    "TooLittleText",
    "century_band",
    "learning_curve",
]

#: Below this many training characters a cell is not built. An early anchor
#: from the pre-experiment (16 k chars reached 1.62 bits/char on register-like
#: text), **not a measured threshold**: ``learning_curve`` is how to measure it
#: per cell, and the measured value belongs in the artefact that used it.
MIN_CHARS = 16_000

#: Stupid backoff weight (Brants et al. 2007): an unseen context costs this
#: factor per order it falls back.
BACKOFF = 0.4

#: Document boundary marker. A page is the document (#110): lines are scored
#: with their predecessor as context, and the first line of a page starts cold.
BOUNDARY = "\n"


def century_band(century: int | None) -> str | None:
    """The band a century falls in — the coarsest useful cut, two centuries wide.

    15 and 16 are one band (charters and missives of the Confederation, the
    second cell of #110), 19 and 20 another. ``None`` stays ``None``.
    """
    if century is None:
        return None
    lo = century - ((century - 1) % 2)
    return f"{lo}-{lo + 1}"


@dataclass(frozen=True)
class Cell:
    """The key of one scorer. ``None`` in a field means "any" — a fallback cell."""

    lang: str | None = None
    region: str | None = None
    document_type: str | None = None
    century_band: str | None = None

    def key(self) -> str:
        return "|".join(str(v) if v is not None else "*"
                        for v in (self.lang, self.region, self.document_type, self.century_band))

    @property
    def is_generic(self) -> bool:
        return self.region is None and self.document_type is None and self.century_band is None

    def backoff_chain(self) -> list[Cell]:
        """This cell, then the cells to try when it has no scorer — in the model
        selector's order: drop the century band, then the genre, then the
        region, down to the language alone."""
        chain = [self]
        c = self
        for drop in ("century_band", "document_type", "region"):
            if getattr(c, drop) is not None:
                c = Cell(**{**asdict(c), drop: None})
                chain.append(c)
        return chain

    @classmethod
    def from_criteria(cls, criteria) -> Cell:
        """From a ``SourceCriteria``-shaped object or dict (lang, region,
        document_type, century)."""
        get = criteria.get if isinstance(criteria, dict) else lambda k, d=None: getattr(criteria, k, d)
        return cls(lang=get("lang"), region=get("region"), document_type=get("document_type"),
                   century_band=century_band(get("century")))


class TooLittleText(ValueError):
    """The cell has fewer characters than ``min_chars`` — it is not built."""


class ContaminationError(ValueError):
    """The scorer trained on the document it is asked to score (#23, #98)."""


@dataclass
class Score:
    """What ``score`` returns. ``z`` is the number to read; ``cell`` is the cell
    that answered, and ``fallback`` says whether it was the one asked for."""

    z: float
    bits_per_char: float
    cell: str
    asked: str
    fallback: bool
    chars: int


class CharNgramScorer:
    """A character n-gram with stupid backoff that scores and does nothing else."""

    def __init__(self, cell: Cell, order: int = 6, backoff: float = BACKOFF):
        if order < 2:
            raise ValueError("order must be at least 2")
        self.cell = cell
        self.order = order
        self.backoff = backoff
        self._counts: dict[str, Counter] = defaultdict(Counter)   # context -> next char
        self._alphabet: set[str] = set()
        self.train_chars = 0
        self.documents: set[str] = set()
        self.sources: list[str] = []
        self.gt_commit: str = ""
        self.heldout: dict = {}          # mean, std, n, chars — the own distribution

    # ── training ───────────────────────────────────────────────────────────

    def train(self, pages: Iterable[tuple[str, str]], *, min_chars: int = MIN_CHARS,
              sources: Iterable[str] = (), gt_commit: str = "") -> CharNgramScorer:
        """Count n-grams over ``(document, page_text)`` pairs.

        Raises :class:`TooLittleText` — and keeps nothing — when the text is
        below ``min_chars``: a thin cell scores everything badly, and that is
        indistinguishable from "the reading is bad".
        """
        counts: dict[str, Counter] = defaultdict(Counter)
        alphabet: set[str] = set()
        chars = 0
        docs: set[str] = set()
        for document, text in pages:
            text = (text or "").strip()
            if not text:
                continue
            docs.add(document)
            padded = BOUNDARY + text + BOUNDARY
            chars += len(text)
            for i in range(1, len(padded)):
                alphabet.add(padded[i])
                for k in range(self.order):
                    if i - k < 0:
                        break
                    counts[padded[i - k:i]][padded[i]] += 1
        if chars < min_chars:
            raise TooLittleText(
                f"cell {self.cell.key()}: {chars:,} characters of edition text, below the "
                f"{min_chars:,} this cell needs — not built, measure the learning curve "
                f"before lowering the floor")
        self._counts, self._alphabet, self.train_chars = counts, alphabet, chars
        self.documents = docs
        self.sources = list(sources)
        self.gt_commit = gt_commit
        return self

    def calibrate(self, pages: Iterable[tuple[str, str]]) -> dict:
        """The distribution of the cell's own held-out text: what "normal"
        bits/char look like here. Pages that were trained on are refused."""
        values = []
        chars = 0
        for document, text in pages:
            if document in self.documents:
                raise ContaminationError(
                    f"cell {self.cell.key()}: held-out page of document {document!r} is in "
                    f"the training set — the same document on both sides (#23)")
            text = (text or "").strip()
            if not text:
                continue
            values.append(self.bits_per_char(text))
            chars += len(text)
        if len(values) < 2:
            raise ValueError("calibration needs at least two held-out pages")
        std = pstdev(values)
        if std < 1e-6:
            # Pages that score identically are copies of one another, and a z
            # against a zero spread is a number without meaning.
            raise ValueError(
                f"cell {self.cell.key()}: the held-out pages are indistinguishable "
                f"(spread {std:.2e}) — calibrate on varied pages, not copies")
        self.heldout = {"mean": mean(values), "std": std, "n": len(values), "chars": chars}
        return self.heldout

    # ── scoring ────────────────────────────────────────────────────────────

    def _prob(self, context: str, ch: str) -> float:
        weight = 1.0
        for k in range(len(context), -1, -1):
            ctx = context[len(context) - k:] if k else ""
            nxt = self._counts.get(ctx)
            if nxt:
                total = sum(nxt.values())
                if ch in nxt:
                    return weight * nxt[ch] / total
            weight *= self.backoff
        # unseen character: one share of the alphabet plus one
        return weight / (len(self._alphabet) + 1)

    def bits_per_char(self, text: str, previous: str = "") -> float:
        """Mean bits per character of ``text``, scored after ``previous`` (the
        page's preceding line) so a line is not judged cold."""
        text = (text or "").strip()
        if not text:
            raise ValueError("nothing to score")
        if not self._counts:
            raise ValueError(f"cell {self.cell.key()} is not trained")
        padded = BOUNDARY + previous + BOUNDARY + text if previous else BOUNDARY + text
        start = len(padded) - len(text)
        bits = 0.0
        for i in range(start, len(padded)):
            context = padded[max(0, i - self.order + 1):i]
            bits -= math.log2(self._prob(context, padded[i]))
        return bits / len(text)

    def z(self, text: str, previous: str = "") -> tuple[float, float]:
        if not self.heldout:
            raise ValueError(f"cell {self.cell.key()} is not calibrated — no held-out distribution")
        b = self.bits_per_char(text, previous)
        return (b - self.heldout["mean"]) / self.heldout["std"], b

    def refuse_if_contaminated(self, document: str | None) -> None:
        if document is not None and document in self.documents:
            raise ContaminationError(
                f"cell {self.cell.key()}: document {document!r} is in this scorer's training "
                f"set — a perplexity there would describe text it has seen (#23)")

    # ── artefact ───────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "kind": "char-ngram-scorer",        # no generation path; see test
            "cell": asdict(self.cell),
            "order": self.order,
            "backoff": self.backoff,
            "train_chars": self.train_chars,
            "documents": sorted(self.documents),
            "sources": self.sources,
            "gt_commit": self.gt_commit,
            "heldout": self.heldout,
            "alphabet": sorted(self._alphabet),
            "counts": {ctx: dict(c) for ctx, c in self._counts.items()},
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict) -> CharNgramScorer:
        s = cls(Cell(**data["cell"]), order=int(data["order"]), backoff=float(data["backoff"]))
        s._counts = defaultdict(Counter, {ctx: Counter(c) for ctx, c in data["counts"].items()})
        s._alphabet = set(data.get("alphabet", []))
        s.train_chars = int(data.get("train_chars", 0))
        s.documents = set(data.get("documents", []))
        s.sources = list(data.get("sources", []))
        s.gt_commit = data.get("gt_commit", "")
        s.heldout = dict(data.get("heldout", {}))
        return s

    @classmethod
    def load(cls, path: str | Path) -> CharNgramScorer:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class ScorerMatrix:
    """The cells that exist, and the backoff between them."""

    scorers: dict[str, CharNgramScorer] = field(default_factory=dict)

    def add(self, scorer: CharNgramScorer) -> None:
        self.scorers[scorer.cell.key()] = scorer

    def resolve(self, cell: Cell) -> tuple[CharNgramScorer | None, bool]:
        for i, candidate in enumerate(cell.backoff_chain()):
            s = self.scorers.get(candidate.key())
            if s is not None:
                return s, i > 0
        return None, True

    def score(self, text: str, criteria, *, document: str | None = None,
              previous: str = "") -> Score:
        """z-score of ``text`` under the cell ``criteria`` names, or the nearest
        cell that exists. Refuses a document the answering scorer trained on."""
        asked = Cell.from_criteria(criteria)
        scorer, fallback = self.resolve(asked)
        if scorer is None:
            raise LookupError(f"no scorer for cell {asked.key()} nor for any fallback of it")
        scorer.refuse_if_contaminated(document)
        z, bits = scorer.z(text, previous)
        return Score(z=z, bits_per_char=bits, cell=scorer.cell.key(), asked=asked.key(),
                     fallback=fallback, chars=len(text.strip()))

    @classmethod
    def load_dir(cls, directory: str | Path) -> ScorerMatrix:
        m = cls()
        for path in sorted(Path(directory).glob("*.scorer.json")):
            m.add(CharNgramScorer.load(path))
        return m


def learning_curve(train_pages: list[tuple[str, str]], heldout_pages: list[tuple[str, str]],
                   steps: Iterable[int], *, cell: Cell | None = None,
                   order: int = 6) -> list[tuple[int, float]]:
    """bits/char on the held-out pages against the amount of training text — the
    measurement that sets a cell's floor. Returns ``(chars, bits)`` per step;
    the floor is where the curve goes flat."""
    cell = cell or Cell()
    out = []
    for n in steps:
        budget, chosen, used = n, [], 0
        for doc, text in train_pages:
            if used >= budget:
                break
            chosen.append((doc, text))
            used += len(text)
        s = CharNgramScorer(cell, order=order).train(chosen, min_chars=1)
        values = [s.bits_per_char(t) for _, t in heldout_pages if t.strip()]
        out.append((used, mean(values)))
    return out


def pages_from_lines_manifest(path: str | Path, document=None) -> list[tuple[str, str]]:
    """``lines_<role>.jsonl`` (image, text) → (document, page text) pairs, lines of
    one page joined by newlines. The document comes from the image name through
    ``document`` (default: :func:`atr_training.heldout.document_of`)."""
    from atr_training.heldout import document_of
    document = document or document_of
    pages: dict[str, list[str]] = defaultdict(list)
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            doc = document(row["image"]) or row["image"]
            pages[doc].append(row["text"])
    return [(doc, BOUNDARY.join(lines)) for doc, lines in pages.items()]

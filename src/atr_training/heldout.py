"""Documents reserved for evaluation, kept out of every training selection (#98).

An eval set drawn from the same corpora a model trains on only stays an eval set
if something enforces it. Nothing did: 198 of the 200 test documents of
``german-medieval-v1`` were in the training set of
``20260910T110352Z-qwen3vl-german-pages-v3``, and 146 of its 150 validation
documents. A CER measured there would have described hands the model had trained
on — which is worse than having no number, because it looks like progress.

**Documents, not pages.** Pages of one manuscript share a hand, ink and layout,
so holding out pages leaks the hand. The unit is the Transkribus ``docId``, the
same unit :mod:`scripts.make_split` groups by, read out of the materialised page
name ``<pool index>_<docId>_<folio>_<image>.xml``.

**Dropping, not refusing.** The reserved documents live inside the corpora a run
is supposed to train on, so refusing the run would make the eval set unusable for
exactly the training it exists to measure. The pages are removed from the
training manifest and the count is recorded on the job, which is the honest
version of what a selection by project name would have done in advance.

The registry is ``config/heldout_eval_documents.json``, in the repo rather than
on the share: it has to be readable when the share is not, it is small, and a
change to it is a change somebody should have to review.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

__all__ = ["HeldOutDocuments", "DEFAULT_REGISTRY", "Reservation", "document_of",
           "document_of_entry", "load_heldout"]

#: Repo-relative, resolved from this file so a service started anywhere finds it.
DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "config" / "heldout_eval_documents.json"

#: ``<pool index>_<docId>_<folio>_<image>.xml`` — field 1 is the document. Field 0
#: is unique per page and field 2 is the folio, so neither groups anything.
DOC_FIELD = 1


def document_of(path: str | Path) -> str | None:
    """The Transkribus document id in a materialised page name, or None.

    None for anything that is not one — a line-granularity manifest holds crop
    names, and guessing a document from those would drop training data for a
    reason nobody could reconstruct.
    """
    parts = Path(str(path)).name.split("_")
    if len(parts) <= DOC_FIELD or not parts[0].isdigit():
        return None
    document = parts[DOC_FIELD]
    return document or None


def document_of_entry(entry: str) -> str | None:
    """The document of one manifest entry, whatever kind of manifest it is from.

    Three kinds reach the reservation step, and only the first is a page path:

    * a page manifest (``pages_train.lst``) — one materialised page name per line;
    * a **line manifest** (``lines_train.jsonl``) — one JSON object per line, whose
      ``page`` field holds the scan the crop came from;
    * a **chunk plan** (``train_plan.json``) — not a manifest at all, but what
      ``prepare`` returns in chunked mode, so it used to arrive here too.

    Handing a JSON object to :func:`document_of` parses it as a path, and
    ``Path(line).name`` is whatever follows the **last slash in the whole blob**.
    Measured on the shape ``prepare`` writes: with ``image`` and then a
    transcription containing a slash — ``anno 1431 / vigilia``, routine in this
    material — the answer flips from the right document to None, i.e. from
    "reserved" to "kept". Field order and transcription content decided whether a
    held-out page was removed from training.

    So the kind is established first, and only a page path goes to
    :func:`document_of`. A JSON entry with no usable ``page`` yields None, which
    the caller must read as *unidentifiable*, never as "not reserved" — see
    :class:`Reservation`.
    """
    text = entry.strip()
    if text.startswith("{"):
        try:
            sample = json.loads(text)
        except ValueError:
            return None
        page = sample.get("page") if isinstance(sample, dict) else None
        return document_of(page) if isinstance(page, str) and page else None
    return document_of(text)


@dataclass(frozen=True)
class Reservation:
    """What a manifest yielded: the two lists, and whether it could be read.

    ``identified`` is the point. An empty ``reserved`` means two different things
    — the registry was consulted against pages it holds nothing of, or no entry
    named a document it could have matched — and the project has paid for that
    confusion six times over (#119, #595, #631, #633, #649, #251). A caller that
    records a count without looking at ``measurable`` writes a zero that cannot be
    read afterwards, which is what #247 then has to untangle across 54 runs.
    """

    keep: list[str]
    reserved: list[str]
    identified: int
    total: int

    @property
    def measurable(self) -> bool:
        """Did any entry name a document? If not, the count means nothing."""
        return self.identified > 0


@dataclass(frozen=True)
class HeldOutDocuments:
    """The union of every reserved set, and where each document came from."""

    documents: frozenset[str]
    #: set name -> how many of its documents are in ``documents``; for the log.
    sets: dict[str, int]

    def __bool__(self) -> bool:
        return bool(self.documents)

    def split(self, pages: list[str]) -> tuple[list[str], list[str]]:
        """``(keep, reserved)`` — order preserved, so a manifest stays comparable.

        Page paths only. :meth:`reserve` is the one to call on a manifest whose
        kind is not certain, because it also says whether anything *could* be
        matched.
        """
        result = self.reserve(pages, document=document_of)
        return result.keep, result.reserved

    def reserve(self, entries: list[str], document=document_of_entry) -> Reservation:
        """Split *entries* and report how many of them named a document.

        The count of identified entries travels with the split because the caller
        cannot reconstruct it: ``reserved == []`` is the same output for "nothing
        of this corpus is held back" and for "nothing here could be checked".
        """
        keep: list[str] = []
        reserved: list[str] = []
        identified = 0
        for entry in entries:
            name = document(entry)
            if name is not None:
                identified += 1
            (reserved if name in self.documents else keep).append(entry)
        return Reservation(keep, reserved, identified, len(entries))


def load_heldout(registry: str | Path | None = None) -> HeldOutDocuments:
    """Read the registry. A missing or malformed file holds nothing back.

    Deliberately not an error: this runs inside the prepare stage of every job,
    and a run that cannot start because a JSON file is unreadable is a worse
    failure than one that trains on more data than intended — but it is logged at
    warning level, because silently holding nothing back is how the set was lost
    the first time.
    """
    path = Path(registry) if registry is not None else DEFAULT_REGISTRY
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("no held-out registry at {} — nothing is reserved (#98)", path)
        return HeldOutDocuments(frozenset(), {})
    except (OSError, ValueError) as exc:
        logger.warning("held-out registry {} is unreadable ({}) — nothing is reserved "
                       "(#98)", path, exc)
        return HeldOutDocuments(frozenset(), {})

    documents: set[str] = set()
    sets: dict[str, int] = {}
    for entry in raw.get("sets", []):
        names = {str(d) for d in entry.get("test_documents", [])}
        names |= {str(d) for d in entry.get("val_documents", [])}
        if not names:
            continue
        sets[str(entry.get("name", "unnamed"))] = len(names)
        documents |= names
    return HeldOutDocuments(frozenset(documents), sets)

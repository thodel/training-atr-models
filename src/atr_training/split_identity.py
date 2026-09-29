"""What makes a held-out split *the same split* (#112).

`german_test` and `german_val` no longer exist as compiled datasets — the arrows
were removed in the 21.09.2026 cleanup (serving-atr-inference#143). What survived
is the set's identity: the 350 document ids in
``config/heldout_eval_documents.json``. That file is in the repository rather
than on the share for exactly this reason.

K0's acceptance says the restored split's document ids must match that registry,
"checked, not assumed". They will — and that is not enough, which is the reason
this module exists.

``make_split.build_split`` draws documents by popping from a list shuffled once
from ``seed``; no draw consumes the generator, so **which documents** are held
out depends on the pool, the seed and ``val_docs``/``test_docs`` alone. But a
document with more pages than its cap is sampled with ``rng.sample``, and that
*does* consume the generator — so **which pages** of those documents are held out
depends on ``val_per_doc`` and ``test_per_doc`` as well, including across the two
draws. Measured on a synthetic pool: raising ``test_per_doc`` by one left every
val *document* identical and exchanged 63 of 335 val *pages*.

``heldout_eval_documents.json`` records neither cap. Its ``built_by`` reads
``scripts/make_split.py --val-docs 150 --test-docs 200``, so the rest were
whatever the defaults were that day. They were 160 and 220 — the file has two
commits in its history and both carry those values — but that is a thing we
worked out afterwards, not a thing the record says.

Hence two levels of proof, kept apart on purpose:

* **Documents** are provable from the registry today. A restore whose documents
  differ is not this set and must not carry its name.
* **Pages** are provable only against a recorded digest. Until one is recorded,
  a restore that matches on documents may still hold out different pages, and
  saying so is the whole job of :class:`Verdict`.

The digest can still be pinned from the original lists if
``pages_test.lst``/``pages_val.lst`` survived on the share. That is K0's first
question, and the difference between confirming this set and starting a new one
that wears its name.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from atr_training.heldout import DEFAULT_REGISTRY, document_of

#: The parameters ``build_split`` reads. Anything absent from a record makes the
#: split unreproducible from it, which is what ``missing_params`` reports.
PARAM_FIELDS = ("seed", "val_docs", "val_per_doc", "test_docs", "test_per_doc",
                "shard_pages")

#: Same length and construction as ``artefact_cache.heldout_fingerprint`` — short
#: enough to paste into an issue, long enough that a collision is not a concern
#: for a few hundred filenames.
DIGEST_CHARS = 16


def page_key(line: str | Path) -> str:
    """The basename, because a manifest carries whatever path the box used.

    The same split materialised on asteraix and on UBELIX writes different
    prefixes for identical pages; digesting the whole line would call those two
    different sets.
    """
    return Path(str(line).strip()).name


def page_digest(pages: Iterable[str | Path]) -> str:
    """A stable digest of a page list: sorted, deduplicated, basenames only.

    Sorted because a manifest's order is not part of what was held out, and a
    reordered file would otherwise read as a different set.
    """
    keys = sorted({page_key(p) for p in pages if str(p).strip()})
    if not keys:
        return "empty"
    payload = "\n".join(keys).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:DIGEST_CHARS]


def documents_of(pages: Iterable[str | Path]) -> set[str]:
    """The documents a page list covers, by the same field ``heldout`` uses."""
    return {d for d in (document_of(p) for p in pages) if d}


@dataclass(frozen=True)
class Verdict:
    """The outcome of checking a rebuilt split against the registry.

    ``ok`` and ``pages_proven`` are deliberately two questions. A restore can be
    ok — every held-out document accounted for — and still hold out different
    pages than the lost set did, and a caller that collapses the two would report
    a CER against a set it only half recognises.
    """

    complaints: list[str]
    pages_proven: bool

    @property
    def ok(self) -> bool:
        return not self.complaints

    def summary(self) -> str:
        if self.complaints:
            return "\n".join(f"  ✗ {c}" for c in self.complaints)
        if self.pages_proven:
            return "  ✓ documents and pages match the recorded split"
        return ("  ✓ documents match the recorded split\n"
                "  ? pages are UNPROVEN — no digest was ever recorded for this "
                "set, so a different page selection would look exactly like "
                "this. Pin one with --record once you accept this restore.")


@dataclass(frozen=True)
class SplitIdentity:
    """One ``sets`` entry of ``heldout_eval_documents.json``, read for restoring."""

    name: str
    source_job: str | None
    built_by: str | None
    params: dict
    manifests: tuple[str, ...]
    test_documents: frozenset[str]
    val_documents: frozenset[str]
    test_page_digest: str | None
    val_page_digest: str | None

    @property
    def missing_params(self) -> list[str]:
        """Parameters ``build_split`` needs that this record does not carry."""
        return [f for f in PARAM_FIELDS if self.params.get(f) is None]

    @property
    def proves_pages(self) -> bool:
        return bool(self.test_page_digest and self.val_page_digest)

    def verify(self, *, test_pages: Iterable[str | Path],
               val_pages: Iterable[str | Path]) -> Verdict:
        """Compare a rebuilt split against what the registry records."""
        test_pages = list(test_pages)
        val_pages = list(val_pages)
        complaints: list[str] = []

        for side, pages, expected in (("test", test_pages, self.test_documents),
                                      ("val", val_pages, self.val_documents)):
            got = documents_of(pages)
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            if missing:
                complaints.append(
                    f"{side}: {len(missing)} recorded documents are not in the "
                    f"rebuild ({', '.join(missing[:5])}"
                    f"{', …' if len(missing) > 5 else ''})")
            if extra:
                complaints.append(
                    f"{side}: {len(extra)} documents in the rebuild were never "
                    f"in this set ({', '.join(extra[:5])}"
                    f"{', …' if len(extra) > 5 else ''})")

        # Held-out documents on the training side is the failure make_split
        # exists to prevent, and it survives a per-side comparison: a document
        # can be right in `test` and also sitting in train.
        overlap = documents_of(test_pages) & documents_of(val_pages)
        if overlap:
            complaints.append(
                f"{len(overlap)} documents are in BOTH test and val "
                f"({', '.join(sorted(overlap)[:5])}) — one hand cannot be two "
                "independent measurements")

        pages_proven = False
        if self.proves_pages:
            for side, pages, expected in (("test", test_pages, self.test_page_digest),
                                          ("val", val_pages, self.val_page_digest)):
                got = page_digest(pages)
                if got != expected:
                    complaints.append(
                        f"{side}: page digest {got} does not match the recorded "
                        f"{expected}. The documents may still match — the caps "
                        "(--val-per-doc / --test-per-doc) change which pages of "
                        "an over-cap document are drawn without changing which "
                        "documents are drawn.")
            pages_proven = not complaints

        return Verdict(complaints, pages_proven)


def _entry(registry: str | Path | None, name: str | None) -> tuple[dict, dict, Path]:
    path = Path(registry) if registry is not None else DEFAULT_REGISTRY
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    sets = raw.get("sets", [])
    if not sets:
        raise KeyError(f"{path} holds no sets")
    if name is None:
        if len(sets) > 1:
            raise KeyError(
                f"{path} holds {len(sets)} sets "
                f"({', '.join(str(s.get('name')) for s in sets)}) — name one")
        return raw, sets[0], path
    for entry in sets:
        if str(entry.get("name")) == name:
            return raw, entry, path
    raise KeyError(f"{path} has no set named {name!r}")


def load_identity(name: str | None = None,
                  registry: str | Path | None = None) -> SplitIdentity:
    """Read one set.

    Unlike :func:`heldout.load_heldout`, a malformed file raises here. That
    function runs inside every job's prepare stage, where refusing to start is
    worse than reserving nothing; this one runs when somebody is deliberately
    restoring a measurement set, and guessing is the failure mode to avoid.
    """
    _, entry, _ = _entry(registry, name)
    params = dict(entry.get("split_params") or {})
    if "seed" not in params and entry.get("seed") is not None:
        params["seed"] = entry["seed"]          # the older, flatter shape
    digests = dict(entry.get("page_digests") or {})
    return SplitIdentity(
        name=str(entry.get("name", "unnamed")),
        source_job=entry.get("source_job"),
        built_by=entry.get("built_by"),
        params=params,
        manifests=tuple(params.get("manifests") or ()),
        test_documents=frozenset(str(d) for d in entry.get("test_documents", [])),
        val_documents=frozenset(str(d) for d in entry.get("val_documents", [])),
        test_page_digest=digests.get("test"),
        val_page_digest=digests.get("val"),
    )


def record_digests(name: str | None, *, test_pages: Iterable[str | Path],
                   val_pages: Iterable[str | Path],
                   registry: str | Path | None = None) -> dict[str, str]:
    """Pin the page digests of an accepted restore, and return them.

    Refuses to move a digest that is already recorded: the point of pinning is
    that a later restore has something it cannot talk its way past.
    """
    raw, entry, path = _entry(registry, name)
    digests = dict(entry.get("page_digests") or {})
    fresh = {"test": page_digest(test_pages), "val": page_digest(val_pages)}

    for side, value in fresh.items():
        if digests.get(side) and digests[side] != value:
            raise ValueError(
                f"{entry.get('name')} already records a {side} digest "
                f"({digests[side]}) and this restore gives {value}. That is two "
                "different sets under one name; give the new one its own entry "
                "rather than overwriting the old one.")

    entry["page_digests"] = {**digests, **fresh}
    path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return fresh

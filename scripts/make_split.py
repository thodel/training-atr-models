#!/usr/bin/env python3
"""Manuscript-grouped, seeded split of a page manifest into val / test / shards.

Why this exists (#39): the first full-corpus run was recovered by hand with
``split -l``, and the val set that ``prepare`` produced turned out to draw its
pages from documents that were also in train — every one of them. A CER measured
against that set describes recognition of hands the model trained on, which is
not the claim a model card makes.

Two rules, both learned the expensive way:

* **A document never straddles a split.** Pages of one manuscript share a hand,
  ink and layout, so a page-level random split reports an in-manuscript figure.
  Grouping is by the document field of the filename
  (``<page>_<document>_<folio>_<image>.xml``).
* **Pages of a val/test document that exceed the per-document cap are dropped,
  never returned to train.** Recycling them would put the same hand on both
  sides of the split and reintroduce exactly the leakage the grouping prevents.

The cap is what makes the evaluation cover many hands rather than a few
manuscripts deeply: documents here run to ~1,500 pages, so sizing val by page
count alone fills the quota from a handful of manuscripts and the resulting CER
is dominated by which ones were drawn.

Everything is derived from ``--seed``, so a split is reproducible from the
record in ``split.json`` rather than from whatever was typed at a terminal.

A third rule, from the first draw of a second set (#301, 27.10.2026): **a
document another registered set already holds is not in the pool.** The
XVI–XIX page corpus kept a `german-medieval-v1` document on its validation
side, the draw put it into the new set, and `register_eval_set.py` refused the
whole split. The pool is now cut by the registry first, and ``split.json``
names the sets it was cut by — by name, so the draw can be repeated after the
new set has joined the registry, which a "whatever is registered" default
could not.

And a fourth, from the second draw of that set: **only a page that names its
document can be drawn.** The six imported datasets of the XVI–XIX corpus are
materialised as ``<counter>_<original file name>``, so field 1 is whatever their
own names start with — ``PPN643815198`` or ``agtck`` (a work, fine), but also
``00000017.xml`` (page 17 of every book in the set) or ``11`` (a page number).
Drawing those groups would hold out pages, not documents, and registering them
would make the trainer drop page 17 of every such book from every later run.
``--documents transkribus-or-named`` (the default) draws only documents whose
field 1 is a Transkribus id (five or more digits) or a name (letters, then
letters and digits); every other page stays on the training side, counted in
``split.json`` under ``undrawable``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

#: Filenames look like ``495426_171954_0406_6478463.xml``; field 1 (0-based) is
#: the document. Field 0 is unique per page and field 2 is the folio number, so
#: neither groups anything.
DOC_FIELD = 1

#: The held-out registry the trainer reads (``atr_training.heldout``). Read
#: here as plain JSON: this script runs with a login node's python3.
DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "config" / "heldout_eval_documents.json"


def document_of(line: str) -> str:
    return Path(line).name.split("_")[DOC_FIELD]


#: Which documents a draw may take (``--documents``). ``None`` — "any" — is the
#: behaviour every set drawn before 27.10.2026 was drawn with, and what
#: ``restore_eval_split.py`` replays for an entry that records no rule.
DRAWABLE = {
    "transkribus-or-named": re.compile(r"[0-9]{5,}|[A-Za-z][A-Za-z0-9]*"),
}


def drawable(line: str, rule: str | None) -> bool:
    """Whether the page's document may be drawn under ``rule``. The first field
    must be a number, as the trainer's ``heldout.document_of`` demands — a page
    it cannot attribute could never be dropped from training."""
    if rule is None:
        return True
    parts = Path(line).name.split("_")
    if len(parts) <= DOC_FIELD or not parts[0].isdigit():
        return False
    return bool(DRAWABLE[rule].fullmatch(parts[DOC_FIELD]))


def read_manifests(data_dir: Path, names: tuple[str, ...]) -> list[str]:
    """Union of the given manifests, deduplicated and ordered.

    Ordering matters: the seeded shuffle is only reproducible if what it shuffles
    is deterministic, and a filesystem does not promise directory order.
    """
    lines: list[str] = []
    for name in names:
        path = data_dir / name
        if path.exists():
            lines += [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return sorted(set(lines))


def registered_documents(registry: Path, names: list[str] | None) -> tuple[list[str], set[str]]:
    """``(set names, their documents)`` — every set when ``names`` is None.

    A name the registry does not hold is an error, not an empty set: a typo
    would otherwise draw a held-out set that overlaps the one it meant.
    """
    data = json.loads(registry.read_text(encoding="utf-8"))
    sets = {entry.get("name"): entry for entry in data.get("sets", [])}
    chosen = sorted(sets) if names is None else list(names)
    unknown = [n for n in chosen if n not in sets]
    if unknown:
        raise ValueError(f"not in {registry}: {', '.join(unknown)} (it holds: "
                         f"{', '.join(sorted(sets)) or 'nothing'})")
    documents: set[str] = set()
    for name in chosen:
        for side in ("test_documents", "val_documents"):
            documents |= {str(d) for d in sets[name].get(side, [])}
    return chosen, documents


def exclude_documents(lines: list[str], documents: set[str]) -> tuple[list[str], dict]:
    """The pool without the pages of ``documents``, and what was taken out."""
    kept = [ln for ln in lines if document_of(ln) not in documents]
    removed = {document_of(ln) for ln in lines} & documents
    return kept, {"documents": len(removed), "pages": len(lines) - len(kept),
                  "which": sorted(removed)}


def build_split(
    lines: list[str],
    *,
    seed: int,
    val_docs: int,
    val_per_doc: int,
    test_docs: int,
    test_per_doc: int,
    shard_pages: int,
    documents: str | None = None,
) -> tuple[dict[str, list[str]], list[list[str]], dict]:
    rng = random.Random(seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for line in lines:
        groups[document_of(line)].append(line)
    for doc in groups:
        groups[doc].sort()

    # A document is drawable when every one of its pages is; the rest go to
    # train whole and are never shuffled, so the draw over the drawable ones is
    # the same whatever the undrawable pages are.
    undrawable = sorted(doc for doc, pages in groups.items()
                        if not all(drawable(page, documents) for page in pages))
    docs = sorted(set(groups) - set(undrawable))
    rng.shuffle(docs)

    def draw(n_docs: int, per_doc: int) -> tuple[list[str], list[str], int]:
        picked: list[str] = []
        pages: list[str] = []
        dropped = 0
        while docs and len(picked) < n_docs:
            doc = docs.pop()
            available = groups[doc]
            sample = available if len(available) <= per_doc else rng.sample(available, per_doc)
            dropped += len(available) - len(sample)
            picked.append(doc)
            pages += sorted(sample)
        return picked, pages, dropped

    test_ids, test, test_dropped = draw(test_docs, test_per_doc)
    val_ids, val, val_dropped = draw(val_docs, val_per_doc)

    # Train keeps whole documents: a shard boundary is allowed to fall short of
    # the target, but never inside a manuscript.
    shards: list[list[str]] = []
    current: list[str] = []
    for doc in docs + undrawable:
        if current and len(current) + len(groups[doc]) > shard_pages:
            shards.append(current)
            current = []
        current += groups[doc]
    if current:
        shards.append(current)

    train_ids = {document_of(ln) for shard in shards for ln in shard}
    record = {
        "seed": seed,
        "total_pages": len(lines),
        "total_documents": len(groups),
        "test": {"pages": len(test), "documents": len(test_ids), "dropped_pages": test_dropped},
        "val": {"pages": len(val), "documents": len(val_ids), "dropped_pages": val_dropped},
        "train": {
            "pages": sum(len(s) for s in shards),
            "documents": len(train_ids),
            "shard_pages": [len(s) for s in shards],
        },
        # Must be 0. Anything else means a held-out hand is being trained on and
        # the CER from this split cannot be published.
        "leak_documents_into_train": len((set(val_ids) | set(test_ids)) & train_ids),
    }
    if documents is not None:
        record["undrawable"] = {
            "rule": documents, "documents": len(undrawable),
            "pages": sum(len(groups[d]) for d in undrawable),
            "examples": [Path(groups[d][0]).name for d in undrawable[:10]],
        }
    return {"val": val, "test": test}, shards, record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path, help="job data/ directory holding the manifests")
    parser.add_argument("--out", type=Path, default=None, help="default: <data_dir>/split")
    parser.add_argument("--manifests", nargs="+", default=["pages_train.lst", "pages_val.lst"],
                        help="manifests to pool before splitting")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--val-docs", type=int, default=25)
    parser.add_argument("--val-per-doc", type=int, default=160)
    parser.add_argument("--test-docs", type=int, default=35)
    parser.add_argument("--test-per-doc", type=int, default=220)
    parser.add_argument("--shard-pages", type=int, default=50000)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY,
                        help="the held-out registry whose documents leave the pool")
    parser.add_argument("--exclude-sets", nargs="+", default=None, metavar="NAME",
                        help="the registered sets whose documents leave the pool "
                             "(default: every set the registry holds now)")
    parser.add_argument("--documents", choices=[*DRAWABLE, "any"],
                        default="transkribus-or-named",
                        help="which documents may be drawn; 'any' is the rule "
                             "of every set drawn before 27.10.2026")
    args = parser.parse_args(argv)
    rule = None if args.documents == "any" else args.documents

    lines = read_manifests(args.data_dir, tuple(args.manifests))
    if not lines:
        parser.error(f"no pages found in {args.data_dir} from {args.manifests}")
    try:
        exclude_sets, reserved = registered_documents(args.registry, args.exclude_sets)
    except (OSError, ValueError) as exc:
        parser.error(f"cannot read the held-out registry: {exc}")
    lines, excluded = exclude_documents(lines, reserved)

    held, shards, record = build_split(
        lines,
        seed=args.seed,
        val_docs=args.val_docs,
        val_per_doc=args.val_per_doc,
        test_docs=args.test_docs,
        test_per_doc=args.test_per_doc,
        shard_pages=args.shard_pages,
        documents=rule,
    )

    # All six parameters and the manifests, not only the seed: the first set's
    # caps had to be recovered from git history (#112), and
    # scripts/register_eval_set.py refuses a split that does not carry them.
    record["params"] = {
        "seed": args.seed, "val_docs": args.val_docs, "val_per_doc": args.val_per_doc,
        "test_docs": args.test_docs, "test_per_doc": args.test_per_doc,
        "shard_pages": args.shard_pages, "manifests": list(args.manifests),
        "exclude_sets": exclude_sets,
        "documents": rule,
    }
    record["excluded"] = {"sets": exclude_sets, **excluded}

    out = args.out or (args.data_dir / "split")
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("shard_*.lst"):
        stale.unlink()
    for name, pages in held.items():
        (out / f"pages_{name}.lst").write_text("\n".join(pages) + "\n")
    for index, shard in enumerate(shards):
        (out / f"shard_{index:02d}.lst").write_text("\n".join(shard) + "\n")
    (out / "split.json").write_text(json.dumps(record, indent=2) + "\n")

    print(json.dumps(record, indent=2))
    if record["leak_documents_into_train"]:
        print("\nLEAKAGE: held-out documents appear in train — do not train on this split")
        return 1
    print(f"\nwrote {len(shards)} shards + val/test to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

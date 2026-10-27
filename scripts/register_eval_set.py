#!/usr/bin/env python3
"""Register a held-out set drawn with make_split.py — checked, then written (#301).

    python scripts/register_eval_set.py <split dir> --name german-kurrent-xvi-xix-v1 \\
        --source-job <job id> --datasets dh-unibe/a dh-unibe/b …          # prints the entry
    python scripts/register_eval_set.py … --write                          # and adds it

`german-medieval-v1` went into ``config/heldout_eval_documents.json`` by hand,
and three of its six split parameters were not written down — they had to be
recovered from the history of make_split.py (#112). The next set is registered
from what make_split.py wrote instead: ``pages_test.lst``, ``pages_val.lst`` and
``split.json``, whose ``params`` hold all six and the manifests.

Refused, before anything is written:

* a name that is already registered;
* a split whose ``params`` are incomplete, or which reports held-out documents
  in train (``leak_documents_into_train``);
* a page whose filename names no document, or a document on both sides;
* a document that another registered set already holds — two sets sharing a
  document would let one set's CER be measured on the other's hands.

**Registering is not a measurement step, it changes every training run.** The
trainer drops every page of a registered document from the training side of
any job, in every backend (#98). A run trained before the registration on one
of these documents reads ``crossed`` in the held-out audit
(``docs/HELDOUT_BOUNDARY.md``) and has to be trained again to be measured on
this set. That is the price of way c in #301, and it is why ``--write`` is a
separate, deliberate step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_training.heldout import DEFAULT_REGISTRY, document_of  # noqa: E402
from atr_training.split_identity import PARAM_FIELDS, page_digest  # noqa: E402


class RegistrationRefused(RuntimeError):
    """The split cannot be registered as it stands; the message says why."""


def _pages(path: Path) -> list[str]:
    if not path.is_file():
        raise RegistrationRefused(f"no {path.name} in {path.parent}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _documents(pages: list[str], side: str) -> list[str]:
    docs = []
    for page in pages:
        doc = document_of(page)
        if not doc:
            raise RegistrationRefused(f"{side}: {Path(page).name} names no document")
        docs.append(str(doc))
    return sorted(set(docs))


def build_entry(split_dir: Path, *, name: str, source_job: str,
                datasets: list[str], registry: dict) -> dict:
    """The registry entry for this split, or :class:`RegistrationRefused`."""
    existing = registry.get("sets", [])
    if any(entry.get("name") == name for entry in existing):
        raise RegistrationRefused(f"a set named {name!r} is already registered")
    record = json.loads((split_dir / "split.json").read_text(encoding="utf-8"))
    params = dict(record.get("params") or {})
    missing = [field for field in (*PARAM_FIELDS, "manifests") if field not in params]
    if missing:
        raise RegistrationRefused(
            f"split.json does not record {', '.join(missing)} — draw it again with a "
            "make_split.py that writes them, so this set can be drawn again")
    if record.get("leak_documents_into_train"):
        raise RegistrationRefused(
            f"split.json reports {record['leak_documents_into_train']} held-out "
            "document(s) in train")
    if not datasets:
        raise RegistrationRefused("name the datasets the pool was drawn from (--datasets)")

    test_pages, val_pages = _pages(split_dir / "pages_test.lst"), _pages(split_dir / "pages_val.lst")
    test_docs, val_docs = _documents(test_pages, "test"), _documents(val_pages, "val")
    both = set(test_docs) & set(val_docs)
    if both:
        raise RegistrationRefused(f"{len(both)} document(s) on both sides: "
                                  f"{', '.join(sorted(both)[:5])}")
    for entry in existing:
        held = {str(d) for d in entry.get("test_documents", [])}
        held |= {str(d) for d in entry.get("val_documents", [])}
        shared = held & (set(test_docs) | set(val_docs))
        if shared:
            raise RegistrationRefused(
                f"{len(shared)} document(s) already belong to {entry.get('name')!r}: "
                f"{', '.join(sorted(shared)[:5])}")

    flags = " ".join(f"--{field.replace('_', '-')} {params[field]}"
                     for field in PARAM_FIELDS)
    return {
        "name": name,
        "source_job": source_job,
        "seed": params["seed"],
        "built_by": f"scripts/make_split.py {flags} --manifests "
                    + " ".join(params["manifests"])
                    + (" --exclude-sets " + " ".join(params["exclude_sets"])
                       if params.get("exclude_sets") else "")
                    + (f" --documents {params['documents']}"
                       if params.get("documents") else ""),
        "datasets": sorted(datasets),
        "test_documents": test_docs,
        "val_documents": val_docs,
        "split_params": params,
        "page_digests": {"test": page_digest(test_pages), "val": page_digest(val_pages)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("split_dir", type=Path, help="what make_split.py wrote (--out)")
    parser.add_argument("--name", required=True)
    parser.add_argument("--source-job", required=True,
                        help="the job whose pages were split; build_page_benchmark.py "
                             "takes the pages from there")
    parser.add_argument("--datasets", nargs="+", default=[])
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--write", action="store_true",
                        help="add the entry to the registry (it changes every training run)")
    args = parser.parse_args(argv)

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    try:
        entry = build_entry(args.split_dir, name=args.name, source_job=args.source_job,
                            datasets=args.datasets, registry=registry)
    except (RegistrationRefused, OSError, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    print(json.dumps({k: (v if not isinstance(v, list) or len(v) <= 6 else
                          f"{len(v)} entries") for k, v in entry.items()}, indent=1))
    if not args.write:
        print(f"\n{len(entry['test_documents'])} test and {len(entry['val_documents'])} val "
              "documents. Not written — --write adds it, and from then on every training "
              "run drops these documents.")
        return 0
    registry.setdefault("sets", []).append(entry)
    args.registry.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(f"\nadded {args.name} to {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

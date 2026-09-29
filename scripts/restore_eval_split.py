#!/usr/bin/env python3
"""Restore `german_test` / `german_val`, and prove it is the same set (#112).

The compiled arrows were removed on 21.09.2026 (serving-atr-inference#143). The
set's identity was not: `config/heldout_eval_documents.json` holds its 350
document ids, its seed, and now the parameters it was drawn with. This rebuilds
the split from that record and refuses to call the result by the set's name
unless it matches.

Two routes, and which one is available decides what can be proved:

  --from-split-dir DIR
      The original `pages_test.lst` / `pages_val.lst` survived. This is the good
      case: the page digest pinned from them is the real set's, and every later
      restore has something to fail against. Check the share first --- `--locate`
      prints where to look.

  --from-pool DATA_DIR --out DIR
      They did not. The split is drawn again from the page manifests with the
      recorded seed and caps. The documents will match (they are a function of
      pool, seed and doc counts alone) --- and the PAGES are unproven, because the
      caps that decide which pages of an over-cap document are drawn were never
      recorded and are recovered, not read. The script says so and will not
      pretend otherwise; `--accept-unproven-pages` is how an operator takes that
      on the record.

Nothing here compiles anything. It produces the page lists and the digest that
K1's sweep manifest cites; `ketos compile` is the step after, on a box that can
reach the material.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_training.split_identity import (  # noqa: E402
    documents_of, load_identity, page_digest, record_digests,
)


def _make_split():
    """`scripts/` is not a package, so the sibling is loaded by path."""
    spec = importlib.util.spec_from_file_location(
        "make_split", Path(__file__).resolve().parent / "make_split.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_list(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"no such file: {path}")
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def locate(identity) -> int:
    """Print what to look for on the share, with the commands to look with."""
    print(f"set        {identity.name}")
    print(f"source job {identity.source_job}")
    print(f"built by   {identity.built_by}")
    print(f"params     {json.dumps({k: v for k, v in identity.params.items() if not k.startswith('_')})}")
    recovered = identity.params.get("_recovered") or []
    if recovered:
        print(f"           recovered, not recorded: {', '.join(recovered)}")
    print()
    print("The lists to look for, in falling order of what they would prove:")
    print()
    print("  # 1. the original split directory --- pins the real page digest")
    print("  ls -la /mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/jobs/"
          f"{identity.source_job}/data/split/")
    print()
    print("  # 2. an archived copy of the arrows, named in this registry")
    print("  ls -la /mnt/wbkolleg_dh_1/Textrecognition_Training/archive/"
          "idhefix-2026-09/arrows/")
    print()
    print("  # 3. anything else carrying these lists")
    print("  find /mnt/wbkolleg_dh_1/Textrecognition_Training -name 'pages_test.lst' "
          "-o -name 'split.json' 2>/dev/null | head")
    print()
    print("Found (1) or (2) --- run with --from-split-dir DIR --record.")
    print("Found neither    --- run with --from-pool DATA_DIR --out DIR, and read "
          "what it says about the pages.")
    return 0


def _report(identity, test_pages, val_pages) -> int:
    verdict = identity.verify(test_pages=test_pages, val_pages=val_pages)
    print()
    print(f"  test  {len(test_pages):>5} pages  "
          f"{len(documents_of(test_pages)):>4} documents  digest {page_digest(test_pages)}")
    print(f"  val   {len(val_pages):>5} pages  "
          f"{len(documents_of(val_pages)):>4} documents  digest {page_digest(val_pages)}")
    print()
    print(verdict.summary())
    return verdict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", dest="name", default=None,
                        help="which set in the registry (default: the only one)")
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--locate", action="store_true",
                        help="print where to look on the share, and stop")
    parser.add_argument("--from-split-dir", type=Path, default=None,
                        help="a surviving split/ with pages_test.lst and pages_val.lst")
    parser.add_argument("--from-pool", type=Path, default=None,
                        help="a job data/ directory holding the page manifests")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the rebuilt split (--from-pool only)")
    parser.add_argument("--record", action="store_true",
                        help="pin the page digests into the registry")
    parser.add_argument("--accept-unproven-pages", action="store_true",
                        help="proceed although no page digest was ever recorded")
    args = parser.parse_args(argv)

    identity = load_identity(args.name, args.registry)
    if args.locate:
        return locate(identity)

    if bool(args.from_split_dir) == bool(args.from_pool):
        parser.error("give exactly one of --from-split-dir or --from-pool "
                     "(or --locate to find out which)")

    missing = identity.missing_params
    if missing:
        raise SystemExit(
            f"{identity.name} does not record {', '.join(missing)} — the split "
            "cannot be drawn again from it. Fill split_params in the registry "
            "first; a guess here produces a different set under the same name.")

    if args.from_split_dir:
        source = f"the surviving lists in {args.from_split_dir}"
        test_pages = _read_list(args.from_split_dir / "pages_test.lst")
        val_pages = _read_list(args.from_split_dir / "pages_val.lst")
    else:
        if args.out is None:
            parser.error("--from-pool needs --out")
        make_split = _make_split()
        params = identity.params
        lines = make_split.read_manifests(args.from_pool, identity.manifests)
        if not lines:
            raise SystemExit(
                f"no pages in {args.from_pool} from {list(identity.manifests)} — "
                "the pool has to be materialised before the split can be drawn")
        held, shards, record = make_split.build_split(
            lines,
            seed=params["seed"],
            val_docs=params["val_docs"], val_per_doc=params["val_per_doc"],
            test_docs=params["test_docs"], test_per_doc=params["test_per_doc"],
            shard_pages=params["shard_pages"],
        )
        test_pages, val_pages = held["test"], held["val"]
        source = f"a fresh draw over {len(lines)} pages from {args.from_pool}"

        args.out.mkdir(parents=True, exist_ok=True)
        for stale in args.out.glob("shard_*.lst"):
            stale.unlink()
        for side, pages in held.items():
            (args.out / f"pages_{side}.lst").write_text("\n".join(pages) + "\n")
        for index, shard in enumerate(shards):
            (args.out / f"shard_{index:02d}.lst").write_text("\n".join(shard) + "\n")
        record["page_digests"] = {"test": page_digest(test_pages),
                                  "val": page_digest(val_pages)}
        record["split_params"] = {k: v for k, v in params.items()
                                  if not k.startswith("_")}
        (args.out / "split.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote {len(shards)} shards + test/val to {args.out}")

        if record["leak_documents_into_train"]:
            print(f"\nLEAKAGE: {record['leak_documents_into_train']} held-out "
                  "documents are in train — do not measure on this split")
            return 1

    print(f"\n{identity.name} from {source}")
    verdict = _report(identity, test_pages, val_pages)
    if not verdict.ok:
        return 1

    if args.record:
        pinned = record_digests(args.name, test_pages=test_pages,
                                val_pages=val_pages, registry=args.registry)
        print(f"\nrecorded page digests: test {pinned['test']}, val {pinned['val']}")
        if args.from_split_dir:
            print("Pinned from the original lists, so this set is now confirmable.")
        else:
            print("Pinned from a REBUILD: this fixes what the set is from now on, "
                  "it does not show it is what it was.")
        return 0

    if not verdict.pages_proven and not args.accept_unproven_pages:
        print("\nStopping short of success: the documents check out, the pages "
              "cannot. Re-run with --record to pin these pages as the set's, or "
              "--accept-unproven-pages to go on without pinning.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

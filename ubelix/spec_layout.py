#!/usr/bin/env python3
"""Refuse a spec whose datasets do not lie the way the spec selects them (#201).

    spec_layout.py <spec.json>

Runs on the UBELIX login node (python 3.9, no container, stdlib only), from
``submit.sh`` after the spec has validated against the contract. The contract
cannot see the hub: ``all_projects: true`` on a dataset laid out as flat shards
(``data/train-00000-of-00002.parquet``, no project directories) is a valid spec
that dies in ``prepare`` — on 09.10.2026 after 2 h 35 of compute, with two more
such datasets behind it in the same spec. One listing call per dataset here
costs a second.

Per dataset the hub's tree API is asked what lies under ``data/`` (and under
``data/<split>/`` when that is a directory), and the selection is judged:

* ``all_projects`` needs project directories: a flat split is **refused**,
  with the measured depth and the form that reaches it (``whole_split``).
* ``train_projects``/``eval_projects`` need the named directories: a flat split
  is refused; a missing name is refused with the names there are.
* ``whole_split`` takes the split as it lies; a split with nothing is refused.
* A ``granularity: line`` dataset is read whole either way and only has to exist.

Exit 0: submit. Exit 1: do not, with every reason on stderr. A hub that cannot
be asked is **not** a refusal: the dataset is printed as ``not checked`` with
the reason, and the exit stays 0 — "could not look" is not "wrong" (#165).

Two datasets of one spec that **share project directory names** are a warning
with the share, not a refusal (#202): ``zh-regierungsratsprotokolle`` and
``kurrent-xix`` hold 182 projects in common, and a spec selecting both whole
weights that material twice — found only because two capped selections
happened to report byte-identical statistics. The warning compares what each
dataset *selects* (its named projects, or every directory under
``all_projects``), so a spec that gives one of them its own projects only is
clean and says so.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

HUB = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
TIMEOUT_S = 20


class HubUnreachable(RuntimeError):
    """The tree could not be read — not a verdict about the dataset."""


def _token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    path = os.path.expanduser("~/.hf_token")
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


def _next_link(headers) -> Optional[str]:
    """The ``rel="next"`` URL of a paginated tree answer, or None."""
    for part in (headers.get("Link") or "").split(","):
        if 'rel="next"' in part and "<" in part:
            return part.split("<", 1)[1].split(">", 1)[0]
    return None


def fetch_tree(repo: str, revision: Optional[str], path: str) -> List[Dict[str, Any]]:
    """Entries under *path* of the dataset repo, as the hub's tree API lists them.

    Follows the pagination: the API answers 1000 entries a page, and
    ``koenigsfelden-charters-post-1500`` has ~1190 project directories — a
    named project on the second page is not a missing one.
    """
    url = "%s/api/datasets/%s/tree/%s/%s" % (HUB, repo, revision or "main", path)
    token = _token()
    entries: List[Dict[str, Any]] = []
    while url:
        request = urllib.request.Request(url, headers={"User-Agent": "atr-training spec_layout"})
        if token:
            request.add_header("Authorization", "Bearer " + token)
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                entries.extend(json.load(response))
                url = _next_link(response.headers)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return entries
            raise HubUnreachable("%s: HTTP %d" % (url, exc.code)) from exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise HubUnreachable("%s: %s" % (url, exc)) from exc
    return entries


def layout(repo: str, revision: Optional[str], split: str,
           fetch=fetch_tree) -> Dict[str, Any]:
    """``{kind, depth, flat_shards, project_dirs}`` for one split, from one or two calls."""
    top = fetch(repo, revision, "data")
    flat = sorted(e["path"] for e in top
                  if e.get("type") == "file" and e["path"].startswith("data/%s-" % split)
                  and e["path"].endswith(".parquet"))
    dirs: List[str] = []
    files_in_split: List[str] = []
    if any(e.get("type") == "directory" and e["path"] == "data/%s" % split for e in top):
        below = fetch(repo, revision, "data/%s" % split)
        dirs = sorted(e["path"].split("/")[-1] for e in below if e.get("type") == "directory")
        files_in_split = [e["path"] for e in below
                          if e.get("type") == "file" and e["path"].endswith(".parquet")]
    if dirs:
        kind, depth = "nested", 4
    elif flat:
        kind, depth = "flat", 2
    elif files_in_split:
        kind, depth = "split-files", 3
    else:
        kind, depth = "empty", None
    return {"kind": kind, "depth": depth, "flat_shards": flat, "project_dirs": dirs}


def _describe(found: Dict[str, Any], split: str) -> str:
    if found["kind"] == "nested":
        return "%d project directories under data/%s/ (depth 4)" % (len(found["project_dirs"]), split)
    if found["kind"] == "flat":
        return "%d flat shards at depth 2 (%s), no project directories" % (
            len(found["flat_shards"]), found["flat_shards"][0])
    if found["kind"] == "split-files":
        return "parquet directly under data/%s/ (depth 3), no project directories" % split
    return "no parquet for split %r" % split


def judge(dataset: Dict[str, Any], found: Dict[str, Any], alone: bool = True) -> Tuple[str, str]:
    """``("ok" | "refused", message)`` for one dataset against its measured layout.

    *alone* says whether the dataset is the request's only one: a single
    ``granularity: line`` dataset is read whole (``granularity_files``) and its
    projects are never resolved, while in a multi-dataset request every entry
    goes through ``data_files_for`` and its ``all_projects``/``train_projects``
    must match the layout — the dataset-level ``granularity`` is not consulted
    there (the live specs write ``line`` on every page-shaped dataset).
    """
    repo = dataset.get("hf_repo", "?")
    split = dataset.get("split", "train")
    what = _describe(found, split)
    if found["kind"] == "empty":
        return "refused", "%s: %s — nothing to select" % (repo, what)
    if alone and dataset.get("granularity") == "line":
        return "ok", "%s: %s; read whole as a line dataset" % (repo, what)
    if dataset.get("whole_split"):
        return "ok", "%s: %s; whole_split takes it as it lies" % (repo, what)
    if dataset.get("all_projects"):
        if found["kind"] == "nested":
            return "ok", "%s: %s; all_projects selects them" % (repo, what)
        return "refused", ("%s: all_projects=true, but the split holds %s. Select it with "
                           "`whole_split: true` (and `max_pages`) instead" % (repo, what))
    named = list(dataset.get("train_projects") or []) + list(dataset.get("eval_projects") or [])
    if named:
        if found["kind"] != "nested":
            return "refused", ("%s: train_projects/eval_projects name directories, but the "
                               "split holds %s" % (repo, what))
        missing = [p for p in named if p not in found["project_dirs"]]
        if missing:
            shown = ", ".join(found["project_dirs"][:8])
            more = "" if len(found["project_dirs"]) <= 8 else ", … (%d)" % len(found["project_dirs"])
            return "refused", "%s: no project %s under data/%s/; there are: %s%s" % (
                repo, ", ".join(repr(m) for m in missing), split, shown, more)
        return "ok", "%s: %s; %d named project(s) found" % (repo, what, len(named))
    return "ok", "%s: %s; nothing selected here to check" % (repo, what)


def selected_projects(dataset: Dict[str, Any], found: Dict[str, Any]) -> List[str]:
    """The project directories this dataset's selection materialises."""
    if dataset.get("all_projects"):
        return list(found.get("project_dirs") or [])
    return list(dataset.get("train_projects") or []) + list(dataset.get("eval_projects") or [])


def shared_projects(selections: Dict[str, List[str]]) -> List[Tuple[str, str, int, float, float]]:
    """``(a, b, shared, share of a, share of b)`` for every pair of datasets whose
    selections share project names, largest first."""
    names = sorted(k for k, v in selections.items() if v)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(selections[a]) & set(selections[b])
            if shared:
                out.append((a, b, len(shared), len(shared) / len(set(selections[a])),
                            len(shared) / len(set(selections[b]))))
    return sorted(out, key=lambda r: -r[2])


def main(argv: List[str], fetch=fetch_tree) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    with open(argv[0], encoding="utf-8") as fh:
        spec = json.load(fh)
    datasets = spec.get("datasets") or ([spec["dataset"]] if spec.get("dataset") else [])
    if not datasets:
        print("spec layout: no datasets in the spec — nothing to check")
        return 0
    refused = 0
    selections: Dict[str, List[str]] = {}
    unchecked = 0
    for dataset in datasets:
        repo, split = dataset.get("hf_repo", "?"), dataset.get("split", "train")
        try:
            found = layout(repo, dataset.get("revision"), split, fetch=fetch)
        except HubUnreachable as exc:
            unchecked += 1
            print("spec layout: not checked  %s: the hub could not be asked (%s)" % (repo, exc))
            continue
        verdict, message = judge(dataset, found, alone=len(datasets) == 1)
        if verdict == "refused":
            refused += 1
            print("spec layout: REFUSED  %s" % message, file=sys.stderr)
        else:
            print("spec layout: ok  %s" % message)
        selections[repo] = selected_projects(dataset, found)
    for a, b, shared, share_a, share_b in shared_projects(selections):
        print("spec layout: WARNING  %s and %s select %d project director%s of the same name "
              "(%.1f %% of the one, %.1f %% of the other): that material would be weighted "
              "twice (#202)" % (a, b, shared, "y" if shared == 1 else "ies",
                                100 * share_a, 100 * share_b), file=sys.stderr)
    if unchecked and len(datasets) > 1:
        print("spec layout: %d dataset(s) not checked, so shared projects with them are "
              "unknown, not absent" % unchecked)
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

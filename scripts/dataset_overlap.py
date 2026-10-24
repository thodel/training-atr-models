#!/usr/bin/env python3
"""Which dh-unibe datasets share project directories, and how much (#202).

    python scripts/dataset_overlap.py                     # every dataset of dh-unibe
    python scripts/dataset_overlap.py --author dh-unibe --min-shared 1
    python scripts/dataset_overlap.py dh-unibe/image-text_zh-regierungsratsprotokolle \\
                                      dh-unibe/image-text_kurrent-xix

Stdlib only, so it runs on the UBELIX login node as it runs here. One or two
tree calls per dataset (paginated), then every pair compared on the names of
its project directories under ``data/train/``. A dataset that cannot be read
(private without a token, no network) is listed as **not read** and left out
of the pairs — unknown is not "no overlap".

Why names: ``zh-regierungsratsprotokolle`` and ``kurrent-xix`` hold 182
project directories in common (56.7 % of kurrent-xix), and the prepare run
17687225 found it only because both were capped at 2 000 pages from the same
``MM_1_001`` and reported byte-identical statistics. A mix that weights one
corpus twice is not a mix anyone learns from; the preflight
(``ubelix/spec_layout.py``) warns about it per spec since #202, and this script
answers it for the catalogue.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

HUB = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
TIMEOUT_S = 30


def _token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        with open(os.path.expanduser("~/.hf_token"), encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


def _get(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "atr-training dataset_overlap"})
    token = _token()
    if token:
        request.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return json.load(response), response.headers


def _next_link(headers) -> Optional[str]:
    for part in (headers.get("Link") or "").split(","):
        if 'rel="next"' in part and "<" in part:
            return part.split("<", 1)[1].split(">", 1)[0]
    return None


def list_datasets(author: str) -> List[str]:
    url = "%s/api/datasets?author=%s&limit=500" % (HUB, author)
    names: List[str] = []
    while url:
        body, headers = _get(url)
        names.extend(d["id"] for d in body)
        url = _next_link(headers)
    return sorted(names)


def project_dirs(repo: str, split: str = "train") -> List[str]:
    """The project directories under ``data/<split>/``; ``[]`` for a flat split.

    Raises ``urllib.error.URLError``/``HTTPError`` when the repo cannot be read.
    """
    url = "%s/api/datasets/%s/tree/main/data/%s" % (HUB, repo, split)
    dirs: List[str] = []
    while url:
        try:
            body, headers = _get(url)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:        # no data/<split>/ directory: flat or empty
                return []
            raise
        dirs.extend(e["path"].split("/")[-1] for e in body if e.get("type") == "directory")
        url = _next_link(headers)
    return sorted(dirs)


def overlaps(projects: Dict[str, List[str]], min_shared: int = 1
             ) -> List[Tuple[str, str, int, float, float]]:
    """``(a, b, shared, share of a, share of b)`` for every pair sharing names, largest first."""
    names = sorted(projects)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = len(set(projects[a]) & set(projects[b]))
            if shared >= min_shared and shared:
                out.append((a, b, shared, shared / len(projects[a]), shared / len(projects[b])))
    return sorted(out, key=lambda r: -r[2])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="*", help="repo ids; default: every dataset of --author")
    parser.add_argument("--author", default="dh-unibe")
    parser.add_argument("--min-shared", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="print the project lists as JSON instead")
    args = parser.parse_args(argv)

    try:
        names = args.datasets or list_datasets(args.author)
    except (urllib.error.URLError, OSError) as exc:
        print("could not list the datasets of %s: %s" % (args.author, exc), file=sys.stderr)
        return 2
    projects: Dict[str, List[str]] = {}
    unread: Dict[str, str] = {}
    for repo in names:
        try:
            projects[repo] = project_dirs(repo)
        except (urllib.error.URLError, OSError) as exc:
            unread[repo] = "HTTP %d" % exc.code if isinstance(exc, urllib.error.HTTPError) else str(exc)
    if args.json:
        json.dump({"projects": projects, "not_read": unread}, sys.stdout, indent=1)
        return 0

    width = max(len(n) for n in names)
    print("%-*s  projects" % (width, "dataset"))
    for repo in names:
        if repo in unread:
            print("%-*s  not read (%s)" % (width, repo, unread[repo]))
        else:
            count = len(projects[repo])
            print("%-*s  %s" % (width, repo, count if count else "0 (flat or empty split)"))
    rows = overlaps({k: v for k, v in projects.items() if v}, args.min_shared)
    print()
    if not rows:
        print("no two readable datasets share a project directory name")
    else:
        print("shared project directories (name-identical), largest first:")
        for a, b, shared, share_a, share_b in rows:
            print("  %4d  %s (%.1f %%)  <->  %s (%.1f %%)" % (shared, a, 100 * share_a, b, 100 * share_b))
    if unread:
        print("\n%d dataset(s) not read and therefore not compared — unknown is not 'no overlap'"
              % len(unread))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""#201: a spec's datasets are checked against their layout on the login node,
before the queue — not two and a half hours into prepare.

The script is stdlib and runs under the login node's Python 3.9; the hub is
replaced by a fake tree here. What is tested is the judgement: all_projects on
a flat split is refused with the measured depth and the form that reaches it,
named projects are checked against the directories there are, a single line
dataset is read whole, and a hub that cannot be asked is "not checked", never
a refusal.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

UBELIX = Path(__file__).resolve().parents[1] / "ubelix"
_SPEC = importlib.util.spec_from_file_location("ubelix_spec_layout", UBELIX / "spec_layout.py")
sl = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = sl
_SPEC.loader.exec_module(sl)

FLAT = {"data": [{"type": "file", "path": "data/train-00000-of-00002.parquet"},
                 {"type": "file", "path": "data/train-00001-of-00002.parquet"}]}
NESTED = {"data": [{"type": "directory", "path": "data/train"}],
          "data/train": [{"type": "directory", "path": "data/train/A"},
                         {"type": "directory", "path": "data/train/B"},
                         {"type": "file", "path": "data/train/README"}]}
SPLIT_FILES = {"data": [{"type": "directory", "path": "data/train"}],
               "data/train": [{"type": "file", "path": "data/train/x.parquet"}]}


def _fake(trees: dict):
    calls = []

    def fetch(repo, revision, path):
        calls.append((repo, revision, path))
        return trees.get(path, [])
    fetch.calls = calls
    return fetch


def test_the_layout_is_read_from_one_or_two_tree_calls():
    flat = sl.layout("o/flat", "a" * 40, "train", fetch=_fake(FLAT))
    assert flat["kind"] == "flat" and flat["depth"] == 2 and len(flat["flat_shards"]) == 2
    fetch = _fake(NESTED)
    nested = sl.layout("o/nested", None, "train", fetch=fetch)
    assert nested["kind"] == "nested" and nested["depth"] == 4
    assert nested["project_dirs"] == ["A", "B"]
    assert [c[2] for c in fetch.calls] == ["data", "data/train"], "the split's directory is asked once"
    assert sl.layout("o/x", None, "train", fetch=_fake(SPLIT_FILES))["kind"] == "split-files"
    assert sl.layout("o/x", None, "validation", fetch=_fake(FLAT))["kind"] == "empty", \
        "another split's shards do not count"


def test_all_projects_on_a_flat_split_is_refused_with_the_measured_depth():
    found = sl.layout("o/flat", None, "train", fetch=_fake(FLAT))
    verdict, message = sl.judge({"hf_repo": "o/flat", "all_projects": True, "max_pages": 10}, found,
                                alone=False)
    assert verdict == "refused"
    assert "2 flat shards at depth 2 (data/train-00000-of-00002.parquet)" in message
    assert "whole_split: true" in message and "max_pages" in message
    nested = sl.layout("o/nested", None, "train", fetch=_fake(NESTED))
    assert sl.judge({"hf_repo": "o/nested", "all_projects": True}, nested, alone=False)[0] == "ok"


def test_named_projects_are_checked_against_the_directories_there_are():
    nested = sl.layout("o/nested", None, "train", fetch=_fake(NESTED))
    ok = sl.judge({"hf_repo": "o/nested", "train_projects": ["A"], "eval_projects": ["B"]}, nested)
    assert ok[0] == "ok" and "2 named project(s) found" in ok[1]
    verdict, message = sl.judge({"hf_repo": "o/nested", "train_projects": ["A", "Z"]}, nested)
    assert verdict == "refused" and "'Z'" in message and "there are: A, B" in message
    flat = sl.layout("o/flat", None, "train", fetch=_fake(FLAT))
    assert sl.judge({"hf_repo": "o/flat", "train_projects": ["A"]}, flat)[0] == "refused"


def test_whole_split_takes_the_split_as_it_lies_and_an_empty_one_is_refused():
    flat = sl.layout("o/flat", None, "train", fetch=_fake(FLAT))
    assert sl.judge({"hf_repo": "o/flat", "whole_split": True, "max_pages": 1}, flat)[0] == "ok"
    empty = sl.layout("o/none", None, "train", fetch=_fake({}))
    verdict, message = sl.judge({"hf_repo": "o/none", "whole_split": True, "max_pages": 1}, empty)
    assert verdict == "refused" and "nothing to select" in message


def test_a_line_dataset_alone_is_read_whole_but_in_company_its_selection_counts():
    """The live specs write `granularity: line` on every page-shaped dataset; in a
    multi-dataset request that field is not consulted and `all_projects` must
    match the layout. A single line dataset really is read whole."""
    flat = sl.layout("o/towerbooks", None, "train", fetch=_fake(FLAT))
    dataset = {"hf_repo": "o/towerbooks", "granularity": "line", "all_projects": True, "max_pages": 1}
    assert sl.judge(dataset, flat, alone=True)[0] == "ok"
    assert sl.judge(dataset, flat, alone=False)[0] == "refused"


def test_main_refuses_with_every_reason_and_a_hub_that_cannot_be_asked_is_not_a_refusal(tmp_path, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"model_id": "m", "datasets": [
        {"hf_repo": "o/flat", "all_projects": True, "max_pages": 1},
        {"hf_repo": "o/nested", "all_projects": True, "max_pages": 1},
        {"hf_repo": "o/private", "all_projects": True, "max_pages": 1},
        {"hf_repo": "o/flat2", "train_projects": ["A"]},
    ]}), encoding="utf-8")
    trees = {"o/flat": FLAT, "o/nested": NESTED, "o/flat2": FLAT}

    def fetch(repo, revision, path):
        if repo == "o/private":
            raise sl.HubUnreachable("%s: HTTP 401" % repo)
        return trees[repo].get(path, [])
    assert sl.main([str(spec)], fetch=fetch) == 1
    out, err = capsys.readouterr()
    assert err.count("spec layout: REFUSED") == 2 and "o/flat:" in err and "o/flat2:" in err
    assert "spec layout: ok  o/nested" in out
    assert "not checked  o/private" in out and "HTTP 401" in out and "o/private" not in err

    only_private = tmp_path / "p.json"
    only_private.write_text(json.dumps({"datasets": [{"hf_repo": "o/private", "all_projects": True}]}))
    assert sl.main([str(only_private)], fetch=fetch) == 0, "could not look is not wrong"
    nothing = tmp_path / "n.json"
    nothing.write_text(json.dumps({"model_id": "m"}))
    assert sl.main([str(nothing)], fetch=fetch) == 0
    assert "nothing to check" in capsys.readouterr().out
    single = tmp_path / "s.json"
    single.write_text(json.dumps({"dataset": {"hf_repo": "o/flat", "all_projects": True}}))
    assert sl.main([str(single)], fetch=fetch) == 1, "the old singular `dataset` is read too"


def test_the_tree_api_s_pagination_is_followed_and_a_404_is_an_empty_tree(monkeypatch):
    class Response:
        def __init__(self, body, link=None):
            self._body = json.dumps(body).encode("utf-8")
            self.headers = {"Link": link} if link else {}

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    pages = {"https://huggingface.co/api/datasets/o/big/tree/main/data/train":
             Response([{"type": "directory", "path": "data/train/A"}],
                      link='<https://huggingface.co/api/datasets/o/big/tree/main/data/train?cursor=xyz>; rel="next"'),
             "https://huggingface.co/api/datasets/o/big/tree/main/data/train?cursor=xyz":
             Response([{"type": "directory", "path": "data/train/B"}])}
    asked = []

    def urlopen(request, timeout=0):
        asked.append(request.full_url)
        if request.full_url in pages:
            return pages[request.full_url]
        raise urllib.error.HTTPError(request.full_url, 404, "not found", {}, io.BytesIO(b""))
    monkeypatch.setattr(sl.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(sl, "_token", lambda: None)
    entries = sl.fetch_tree("o/big", None, "data/train")
    assert [e["path"] for e in entries] == ["data/train/A", "data/train/B"]
    assert len(asked) == 2
    assert sl.fetch_tree("o/big", None, "data/none") == []
    assert sl._next_link({"Link": '<u1>; rel="prev", <u2>; rel="next"'}) == "u2"
    assert sl._next_link({}) is None

    def down(request, timeout=0):
        raise urllib.error.URLError("no route to host")
    monkeypatch.setattr(sl.urllib.request, "urlopen", down)
    with pytest.raises(sl.HubUnreachable, match="no route"):
        sl.fetch_tree("o/big", None, "data")


def test_the_token_reaches_the_request_but_is_never_printed(monkeypatch, tmp_path):
    seen = {}

    class Response:
        headers = {}

        def read(self):
            return b"[]"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, timeout=0):
        seen["auth"] = request.get_header("Authorization")
        return Response()
    monkeypatch.setattr(sl.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    sl.fetch_tree("o/x", None, "data")
    assert seen["auth"] == "Bearer hf_secret_value"
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".hf_token").write_text("hf_from_file\n")
    sl.fetch_tree("o/x", None, "data")
    assert seen["auth"] == "Bearer hf_from_file"


# ── #202: two datasets of one spec that select the same project names ────────
def test_selected_projects_are_what_the_selection_materialises():
    nested = sl.layout("o/nested", None, "train", fetch=_fake(NESTED))
    assert sl.selected_projects({"all_projects": True}, nested) == ["A", "B"]
    assert sl.selected_projects({"train_projects": ["A"], "eval_projects": ["B"]}, nested) == ["A", "B"]
    assert sl.selected_projects({"train_projects": ["A"]}, nested) == ["A"]
    assert sl.selected_projects({"whole_split": True}, sl.layout("o/f", None, "train", fetch=_fake(FLAT))) == []


def test_shared_projects_are_counted_per_pair_with_both_shares():
    rows = sl.shared_projects({"o/zh": ["MM_1_001", "MM_1_002", "MM_1_094"],
                               "o/kx": ["MM_1_001", "MM_1_002", "MM_2_033", "MM_2_034"],
                               "o/other": ["X"], "o/empty": []})
    assert rows == [("o/kx", "o/zh", 2, 0.5, 2 / 3)]


def test_main_warns_about_shared_projects_but_does_not_refuse(tmp_path, capsys):
    both = {"data": [{"type": "directory", "path": "data/train"}],
            "data/train": [{"type": "directory", "path": "data/train/MM_1_001"},
                           {"type": "directory", "path": "data/train/MM_1_002"}]}
    kx = {"data": [{"type": "directory", "path": "data/train"}],
          "data/train": [{"type": "directory", "path": "data/train/MM_1_001"},
                         {"type": "directory", "path": "data/train/MM_1_002"},
                         {"type": "directory", "path": "data/train/MM_2_033"},
                         {"type": "directory", "path": "data/train/MM_2_034"}]}
    trees = {"o/zh": both, "o/kx": kx}

    def fetch(repo, revision, path):
        if repo == "o/private":
            raise sl.HubUnreachable("401")
        return trees[repo].get(path, [])
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"datasets": [
        {"hf_repo": "o/zh", "all_projects": True, "max_pages": 1},
        {"hf_repo": "o/kx", "all_projects": True, "max_pages": 1},
        {"hf_repo": "o/private", "all_projects": True, "max_pages": 1}]}))
    assert sl.main([str(spec)], fetch=fetch) == 0, "a warning, not a refusal"
    out, err = capsys.readouterr()
    assert "WARNING  o/kx and o/zh select 2 project directories of the same name" in err
    assert "(50.0 % of the one, 100.0 % of the other)" in err and "weighted twice" in err
    assert "1 dataset(s) not checked, so shared projects with them are unknown" in out

    own = tmp_path / "own.json"
    own.write_text(json.dumps({"datasets": [
        {"hf_repo": "o/zh", "all_projects": True, "max_pages": 1},
        {"hf_repo": "o/kx", "train_projects": ["MM_2_033", "MM_2_034"]}]}))
    assert sl.main([str(own)], fetch=fetch) == 0
    out, err = capsys.readouterr()
    assert "WARNING" not in err, "kurrent-xix given its own projects only: clean"

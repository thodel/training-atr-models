"""ubelix/make_line_dataset.py — the defects the sixteen jobs of 05.10.2026 found.

Eight of sixteen failed, and the script had no tests at all. These cover the four
failure modes that were the script's own: a flat dataset layout, holding every crop
in memory, a source with no line coordinates, and a target repo that is public
although `private=True` was asked for.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ubelix_make_line_dataset",
    Path(__file__).resolve().parents[1] / "ubelix" / "make_line_dataset.py")
mld = importlib.util.module_from_spec(_SPEC)
# Registered before exec: @dataclass resolves its annotations through
# sys.modules[cls.__module__], which is None for a module loaded off a path.
sys.modules[_SPEC.name] = mld
_SPEC.loader.exec_module(mld)

REV = "a" * 40


# ── dataset layout ──────────────────────────────────────────────────────────
def test_flat_layout_reads_the_whole_split(monkeypatch):
    """A page-shaped dataset with no project directories must still be readable.

    `koenigsfelden-charters-part-3` and `transkribus-exports-bullinger-handschrift`
    hold `data/train-00000-of-00002.parquet`, so enumerating project directories
    finds nothing and `all_projects` raises. Both jobs died in 3 and 6 seconds.
    """
    monkeypatch.setattr(mld, "list_projects", lambda repo, split, revision: [])
    monkeypatch.setattr(mld, "has_flat_shards", lambda repo, split, revision: True)
    globs = mld.train_globs("dh-unibe/flat", REV, 200, None)
    assert globs == ["data/train-*.parquet"]


def test_empty_split_directory_reads_the_nested_glob(monkeypatch):
    """No project directories and no flat shards: keep the nested glob.

    `data/train/**/*.parquet` is what a `data/train/` directory without project
    subdirectories needs; `data/train-*.parquet` would match nothing there. The
    two spellings are not interchangeable — the flat one is what the 08.10.2026
    jobs 17545204 and 17545206 lacked, this one is what they had.
    """
    monkeypatch.setattr(mld, "list_projects", lambda repo, split, revision: [])
    monkeypatch.setattr(mld, "has_flat_shards", lambda repo, split, revision: False)
    globs = mld.train_globs("dh-unibe/empty-dir", REV, 200, None)
    assert globs == ["data/train/**/*.parquet"]


def test_project_layout_still_enumerates(monkeypatch):
    """The common layout must keep going through the spec, guard and all."""
    monkeypatch.setattr(mld, "list_projects", lambda repo, split, revision: ["A", "B"])
    seen = {}

    def fake_expand(spec):
        seen["all_projects"] = spec.all_projects
        seen["max_pages"] = spec.max_pages
        return spec

    monkeypatch.setattr(mld, "expand_all_projects", fake_expand)
    monkeypatch.setattr(mld, "data_files_for", lambda spec: {"train": ["data/train/A/*.parquet"]})
    globs = mld.train_globs("dh-unibe/nested", REV, 200, None)
    assert globs == ["data/train/A/*.parquet"]
    assert seen == {"all_projects": True, "max_pages": 200}


def test_named_projects_do_not_enumerate(monkeypatch):
    """With `--projects` the hub need not be asked what exists."""
    def boom(*a, **k):
        raise AssertionError("list_projects must not be called when projects are named")

    monkeypatch.setattr(mld, "list_projects", boom)
    monkeypatch.setattr(mld, "expand_all_projects", lambda spec: spec)
    monkeypatch.setattr(mld, "data_files_for",
                        lambda spec: {"train": [f"data/train/{p}/*.parquet"
                                                for p in spec.train_projects]})
    assert mld.train_globs("dh-unibe/x", REV, 200, ["sg-missiven"]) == \
        ["data/train/sg-missiven/*.parquet"]


# ── memory ──────────────────────────────────────────────────────────────────
def built(tmp_path, n=3):
    manifest = []
    for i in range(n):
        rel = f"crops/{i:04d}.jpg"
        (tmp_path / "crops").mkdir(exist_ok=True)
        (tmp_path / rel).write_bytes(b"\xff\xd8" + bytes(10 + i))
        manifest.append((rel, "line %d" % i, "page.xml", "proj"))
    return mld.Built(manifest, tmp_path, "pool: ...", "dh-unibe/src", REV)


def test_records_is_a_generator(tmp_path):
    """Not a list.

    Four jobs were OOM-killed at `Dataset.from_list`, holding every crop's JPEG at
    once — `rats-und-richtebuecher` after cropping 139'707 of 139'708 lines.
    """
    b = built(tmp_path)
    assert inspect.isgenerator(b.records())


def test_counts_do_not_read_the_images(tmp_path, monkeypatch):
    """The report needs lines, characters and megabytes — none of them need bytes."""
    b = built(tmp_path, n=4)

    def no_reading(self, *a, **k):
        raise AssertionError("read_bytes during counting defeats the purpose")

    monkeypatch.setattr(Path, "read_bytes", no_reading)
    assert b.lines == 4
    assert b.chars == sum(len("line %d" % i) for i in range(4))
    assert b.crop_bytes == sum(12 + i for i in range(4))


def test_records_match_the_published_schema(tmp_path):
    """Every row carries exactly the seven declared columns."""
    rows = list(built(tmp_path).records())
    assert [r["line_index"] for r in rows] == [0, 1, 2]
    assert set(rows[0]) == {"image", "text", "page", "line_index", "project",
                            "source_repo", "source_revision"}
    assert rows[0]["image"]["path"] == "0000.jpg"
    assert rows[0]["image"]["bytes"].startswith(b"\xff\xd8")
    assert rows[0]["source_revision"] == REV


def test_a_source_without_coordinates_publishes_nothing(tmp_path, monkeypatch):
    """`sg-missiven` reported 25'814 lines and could crop none of them.

    Its PageXML carries the text in `<TextLine>` with no `<Coords>`, and the pool
    counts a transcribed line from the text alone. The build must stop, not publish
    an empty dataset.
    """
    class Pages:
        xml_paths = [tmp_path / "a.xml"]
        summary = "pool: 3334 pages, 25814 transcribed lines, worst aspect 0:1"

    monkeypatch.setattr(mld, "train_globs", lambda *a, **k: ["data/train/**/*.parquet"])
    monkeypatch.setattr(mld, "HFPageSource", lambda **k: type(
        "S", (), {"stream": lambda self, *a, **k: iter([])})())
    monkeypatch.setattr(mld, "materialize", lambda *a, **k: Pages())
    monkeypatch.setattr(mld, "samples_for", lambda *a, **k: [])

    def boom(*a, **k):
        raise AssertionError("write_crops must not run with no samples")

    monkeypatch.setattr(mld, "write_crops", boom)
    out = mld.build("dh-unibe/sg-missiven", REV, 200, tmp_path)
    assert out.lines == 0
    assert "25814" in out.summary


# ── visibility: public is the normal case for a dataset ─────────────────────
class NotFound(Exception):
    """Stands in for huggingface_hub.errors.RepositoryNotFoundError."""


class FakeApi:
    """Stands in for HfApi. ``state`` is None for "no such repo"."""

    def __init__(self, state, on_create=None, raises=None):
        self.state = state
        self.on_create = on_create
        self.raises = raises
        self.created = None

    def dataset_info(self, repo_id, **kw):
        if self.raises is not None:
            raise self.raises
        if self.state is None:
            raise NotFound("404")
        return type("Info", (), {"private": self.state})()

    def create_repo(self, repo_id, **kw):
        self.created = kw.get("private")
        self.state = self.on_create


@pytest.fixture
def api(monkeypatch):
    """Install a stub `huggingface_hub` for the duration of one test.

    The functions under test import the hub inside the call, the way
    `hf_source._default_list_repo_files` does — so that a venv without the ML
    dependencies can still run this code. The tests honour that: they stub the
    module rather than require it.
    """
    import types

    def install(fake):
        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = lambda *a, **k: fake
        errors = types.ModuleType("huggingface_hub.errors")
        errors.RepositoryNotFoundError = NotFound
        hub.errors = errors
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
        monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
        return fake

    return install


# The default: public, because a line-level copy exists to be usable.
def test_a_new_dataset_is_created_public_by_default(api):
    fake = api(FakeApi(state=None, on_create=False))
    assert mld.settle_visibility("dh-unibe/x-lines", want_private=False) is True
    assert fake.created is False, "create_repo must be asked for a public repo"


def test_an_existing_public_target_is_simply_used(api):
    """No refusal: public is what we wanted."""
    api(FakeApi(state=False))
    assert mld.settle_visibility("dh-unibe/x-lines", want_private=False) is True


def test_an_unreachable_hub_does_not_stop_a_public_run(api, capsys):
    """Nothing is at stake: the result would be public either way."""
    api(FakeApi(state=None, raises=OSError("connection reset")))
    assert mld.settle_visibility("dh-unibe/x-lines", want_private=False) is True
    assert "public is the default" in capsys.readouterr().out


# `--private`: the deliberate exception, and there the old refusals apply.
def test_private_is_requested_at_creation(api):
    fake = api(FakeApi(state=None, on_create=True))
    assert mld.settle_visibility("dh-unibe/x-lines", want_private=True) is True
    assert fake.created is True


def test_a_creation_that_comes_out_public_refuses_when_private_was_asked(api, capsys):
    """`private=True` was not honoured three times on 05.10.2026."""
    api(FakeApi(state=None, on_create=False))
    assert mld.settle_visibility("dh-unibe/x-lines", want_private=True) is False
    assert "refusing to upload" in capsys.readouterr().err


def test_an_existing_public_target_refuses_when_private_was_asked(api, capsys):
    """push_to_hub cannot turn it private, so saying it would be a lie."""
    api(FakeApi(state=False))
    assert mld.settle_visibility("dh-unibe/x-lines", want_private=True) is False
    assert "Set it private in the HF UI" in capsys.readouterr().err


def test_an_unreachable_hub_refuses_when_private_was_asked(api, capsys):
    """"Could not look" is not "is private", and uploading is not undoable."""
    api(FakeApi(state=None, raises=OSError("connection reset")))
    assert mld.settle_visibility("dh-unibe/x-lines", want_private=True) is False
    assert "refusing to upload rather than guess" in capsys.readouterr().err


# An existing repo keeps what it has — the house rule from publish_to_hub.py.
def test_an_existing_private_target_is_kept_private(api, capsys):
    fake = api(FakeApi(state=True))
    assert mld.settle_visibility("dh-unibe/x-lines", want_private=False) is True
    assert fake.created is None, "an existing repo is never re-created"
    assert "kept as it is" in capsys.readouterr().out


def test_confirm_private_reports_a_public_result(api, capsys):
    api(FakeApi(state=False))
    assert mld.confirm_private("dh-unibe/x-lines") is False
    err = capsys.readouterr().err
    assert "readable by anyone" in err
    assert "settings" in err


def test_confirm_private_is_quiet_when_private(api, capsys):
    api(FakeApi(state=True))
    assert mld.confirm_private("dh-unibe/x-lines") is True
    assert capsys.readouterr().err == ""


# ── the licence travels with the crops (#173) ───────────────────────────────
SOURCE_CARD = """---
license: mit
language:
- de
- la
tags:
- htr
- doi:10.57967/hf/8591
dataset_info:
  features:
  - name: image
    dtype: image
  splits:
  - name: train
    num_examples: 3222
---
# Dataset Card for image-text_koenigsfelden-charters-post-1500

Provenance: State Archives Aargau.
"""

#: What push_to_hub writes for the copy: its own features, nothing else.
PUSHED_HEAD = """---
dataset_info:
  features:
  - name: image
    dtype: binary
  - name: text
    dtype: string
  splits:
  - name: train
    num_examples: 50024
---"""


def test_the_licence_and_the_language_are_carried_and_nothing_else():
    assert mld.carried_metadata(SOURCE_CARD) == {"license": "mit", "language": ["de", "la"]}, \
        "the source's DOI, tags and features describe the source, not the copy"


def test_carried_keys_go_on_top_and_the_pushed_block_stays_byte_for_byte():
    merged = mld.with_metadata(PUSHED_HEAD, {"license": "mit", "language": ["de"]})
    assert merged.startswith("---\nlicense: mit\nlanguage:\n- de\ndataset_info:\n")
    inner = PUSHED_HEAD[4:-4]
    assert merged.endswith(inner + "\n---"), "dataset_info is not round-tripped"


def test_a_licence_set_on_the_copy_is_not_overwritten():
    head = "---\nlicense: cc-by-4.0\n---"
    assert mld.with_metadata(head, {"license": "mit"}) == head


def test_a_copy_without_frontmatter_gets_one():
    assert mld.with_metadata("", {"license": "mit"}) == "---\nlicense: mit\n---"


def test_a_source_without_a_licence_carries_nothing_and_the_card_says_so(monkeypatch):
    monkeypatch.setattr(mld, "_source_card", lambda repo, rev: "# no frontmatter\n")
    monkeypatch.setattr(mld, "existing_frontmatter", lambda out: PUSHED_HEAD)
    text = mld.card("dh-unibe/x-lines", "dh-unibe/x", REV, 10, 100, "ok")
    assert "\nlicense:" not in text.split("\n---", 2)[0]
    assert "| Licence | none declared by the source — ask before reusing |" in text


def test_the_card_of_a_new_copy_carries_the_licence(monkeypatch):
    monkeypatch.setattr(mld, "_source_card", lambda repo, rev: SOURCE_CARD)
    monkeypatch.setattr(mld, "existing_frontmatter", lambda out: PUSHED_HEAD)
    text = mld.card("dh-unibe/x-lines", "dh-unibe/x", REV, 50024, 3892784, "ok")
    assert text.startswith("---\nlicense: mit\nlanguage:\n- de\n- la\ndataset_info:")
    assert "| Licence | `mit`, as declared by the source |" in text
    assert "Provenance: State Archives Aargau." in text
    frontmatter = text.split("\n---", 1)[0]
    assert "doi" not in frontmatter and "3222" not in frontmatter, \
        "the source's DOI and split sizes stay the source's"


class Uploads:
    def __init__(self):
        self.files = []

    def upload_file(self, **kw):
        self.files.append(kw)


def _hub_with(monkeypatch, tmp_path, out_card: str):
    import types

    readme = tmp_path / "README.md"
    readme.write_text(out_card, encoding="utf-8")
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda *a, **k: str(readme)
    hub.HfApi = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)


def test_carry_metadata_adds_the_licence_to_an_existing_copy(monkeypatch, tmp_path, capsys):
    _hub_with(monkeypatch, tmp_path, PUSHED_HEAD + "\n\n# x-lines\n\nLine-level variant.\n")
    monkeypatch.setattr(mld, "_source_card", lambda repo, rev: SOURCE_CARD)
    api = Uploads()
    assert mld.carry_metadata("dh-unibe/x-lines", "dh-unibe/x", REV, api=api) == 0
    (upload,) = api.files
    text = upload["path_or_fileobj"].decode("utf-8")
    assert text.startswith("---\nlicense: mit\nlanguage:")
    assert "# x-lines\n\nLine-level variant." in text, "the prose is kept"
    assert upload["path_in_repo"] == "README.md" and upload["repo_type"] == "dataset"
    assert "added license, language" in capsys.readouterr().out


def test_carry_metadata_leaves_a_copy_that_has_it(monkeypatch, tmp_path, capsys):
    _hub_with(monkeypatch, tmp_path, "---\nlicense: mit\nlanguage: de\n---\n\nbody\n")
    monkeypatch.setattr(mld, "_source_card", lambda repo, rev: SOURCE_CARD)
    api = Uploads()
    assert mld.carry_metadata("dh-unibe/x-lines", "dh-unibe/x", REV, api=api) == 0
    assert api.files == [] and "unchanged" in capsys.readouterr().out


def test_carry_metadata_refuses_when_the_source_declares_nothing(monkeypatch, tmp_path):
    _hub_with(monkeypatch, tmp_path, PUSHED_HEAD)
    monkeypatch.setattr(mld, "_source_card", lambda repo, rev: "# nothing\n")
    api = Uploads()
    assert mld.carry_metadata("dh-unibe/x-lines", "dh-unibe/x", REV, api=api) == 1
    assert api.files == []


def test_carry_metadata_builds_nothing_and_needs_no_page_bound(monkeypatch):
    seen = []
    monkeypatch.setattr(mld, "carry_metadata", lambda out, repo, rev: seen.append(out) or 0)
    monkeypatch.setattr(mld, "build", lambda *a, **k: pytest.fail("nothing is built"))
    assert mld.main(["--repo", "dh-unibe/x", "--revision", REV, "--out", "dh-unibe/x-lines",
                     "--carry-metadata"]) == 0
    assert seen == ["dh-unibe/x-lines"]
    assert mld.main(["--repo", "dh-unibe/x", "--revision", REV,
                     "--out", "dh-unibe/x-lines"]) == 2, "a build still needs --max-pages"

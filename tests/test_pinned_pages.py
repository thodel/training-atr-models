"""A pinned page excerpt (``DatasetSpec.pages``): what it keeps and what it refuses.

The first use is a capped excerpt of the Leuven registers in
``image-text_medieval-scripts_xiv-xv-xvi``: ``max_pages`` alone would take each
register's opening pages, not a sample of it.
"""

import pytest

from atr_training.contracts import DatasetSpec
from atr_training.hf_source import keep_pages_for, only_pages

REPO = "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi"


def spec(**kw):
    base = dict(hf_repo=REPO, train_projects=["SAL7307_full"])
    base.update(kw)
    return DatasetSpec(**base)


def rows(*names):
    return [{"filename": n, "project_name": "SAL7307_full"} for n in names]


def test_only_the_pinned_pages_pass_in_stream_order():
    keep = keep_pages_for(spec(pages=["p3.jpg", "p1.jpg"]))
    got = [r["filename"] for r in only_pages(rows("p1.jpg", "p2.jpg", "p3.jpg"), keep)]
    assert got == ["p1.jpg", "p3.jpg"]


def test_no_pinned_pages_means_no_filter():
    assert keep_pages_for(spec()) is None
    source = rows("a.jpg")
    assert only_pages(source, None) is source


def test_a_missing_pinned_page_is_not_an_error():
    keep = keep_pages_for(spec(pages=["gone.jpg", "p1.jpg"]))
    assert [r["filename"] for r in only_pages(rows("p1.jpg"), keep)] == ["p1.jpg"]


def test_pages_need_named_projects():
    with pytest.raises(ValueError, match="train_projects"):
        DatasetSpec(hf_repo=REPO, pages=["p1.jpg"])


@pytest.mark.parametrize("extra", [
    {"eval_projects": ["SAL7308_full"]},
    {"chunk_size": 100},
])
def test_pages_refuse_the_paths_that_would_ignore_them(extra):
    with pytest.raises(ValueError, match="cannot be combined"):
        spec(pages=["p1.jpg"], **extra)


def test_pages_refuse_line_granularity():
    with pytest.raises(ValueError, match="granularity"):
        spec(pages=["p1.jpg"], granularity="line")


def test_a_filename_pinned_twice_is_refused():
    with pytest.raises(ValueError, match="more than once"):
        spec(pages=["p1.jpg", "p1.jpg"])


def test_pages_survive_a_round_trip_through_json():
    s = spec(pages=["p1.jpg", "p2.jpg"], max_pages=2)
    assert DatasetSpec.model_validate_json(s.model_dump_json()).pages == ["p1.jpg", "p2.jpg"]

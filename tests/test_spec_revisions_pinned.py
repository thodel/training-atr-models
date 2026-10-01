"""A spec that several arms share has to pin its datasets (#135).

``revision: null`` means "whatever the hub serves today", and the artefact cache
honours that honestly: an unpinned entry is reusable for
:data:`~atr_training.artefact_cache.UNPINNED_MAX_AGE_DAYS` days and then refuses,
because the data may have moved. For a one-off run that is right. For a series of
arms whose whole purpose is to differ in exactly one field it is a trap — the
corpus can be rebuilt between two arms, and nothing says the second one trained
on the first one's split.

It happened: the fifth medieval page arm found its artefact "built 7.5 days ago
from an unpinned revision (limit 7)" and rebuilt the corpus from scratch. That
rebuild happened to be identical, which was luck rather than a guarantee — the
four datasets had last moved in March and April 2026.

The rule here is therefore scoped to what it protects: a spec with **more than
one** dataset is a comparison series, and every one of its datasets carries a
revision. The single-dataset experiment specs (``exp*``, ``resumable*``,
``smoke``) are finished one-offs and are left alone.

    pytest tests/test_spec_revisions_pinned.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

SPECS = sorted((Path(__file__).resolve().parents[1] / "ubelix" / "specs").glob("*.json"))
SERIES = [p for p in SPECS if len(json.loads(p.read_text(encoding="utf-8")).get("datasets") or []) > 1]
SHA = re.compile(r"[0-9a-f]{40}")


def test_there_are_series_specs_to_check():
    """Guards the guard: a glob that matches nothing passes every test below."""
    assert SERIES, f"no multi-dataset spec found among {len(SPECS)} specs"


@pytest.mark.parametrize("spec", SERIES, ids=lambda p: p.name)
def test_every_dataset_of_a_series_spec_pins_a_revision(spec: Path):
    missing = [d["hf_repo"] for d in json.loads(spec.read_text(encoding="utf-8"))["datasets"]
               if not d.get("revision")]
    assert not missing, (
        f"{spec.name} is a comparison series and leaves {missing} unpinned; an "
        "unpinned artefact expires after seven days and the next arm may rebuild "
        "the corpus, which re-derives the split")


@pytest.mark.parametrize("spec", SERIES, ids=lambda p: p.name)
def test_a_pinned_revision_is_a_full_commit_sha(spec: Path):
    """Not a branch name and not an abbreviation: ``main`` would pin nothing, and
    a short sha is ambiguous as the dataset grows."""
    for d in json.loads(spec.read_text(encoding="utf-8"))["datasets"]:
        assert SHA.fullmatch(d["revision"]), f"{spec.name}: {d['hf_repo']} -> {d['revision']!r}"


def test_the_arms_of_one_series_pin_the_same_revisions():
    """The five medieval page arms differ in model_id and base_model and in
    nothing else — including which bytes they train on. If two of them pinned
    different revisions they would share a name and not a corpus."""
    page = {}
    for spec in SERIES:
        d = json.loads(spec.read_text(encoding="utf-8"))
        if d.get("params", {}).get("granularity") != "page" or "medieval" not in spec.name:
            continue
        page[spec.name] = {x["hf_repo"]: x["revision"] for x in d["datasets"]}
    assert len(page) >= 2, f"expected several medieval page arms, found {sorted(page)}"
    first, *rest = page.items()
    for name, pins in rest:
        assert pins == first[1], f"{name} pins different revisions than {first[0]}"

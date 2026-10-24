"""Every spec under ubelix/specs/ is a TrainRequest the trainer would accept.

A spec that does not validate cannot be submitted — submit.sh validates it in
the container first — and lying around it reads like one that could.
``german-xix-xvi-xviii-v1.json`` did exactly that for two weeks: fifteen of its
seventeen datasets set ``all_projects`` without ``max_pages`` (#201). The
datasets' layout is not checked here (that needs the hub: ``ubelix/spec_layout.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atr_training.contracts import TrainRequest

SPECS = sorted((Path(__file__).resolve().parents[1] / "ubelix" / "specs").glob("*.json"))


@pytest.mark.parametrize("path", SPECS, ids=[p.name for p in SPECS])
def test_the_spec_validates_as_a_train_request(path: Path):
    request = TrainRequest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    assert request.datasets, path.name
    for dataset in request.datasets:
        assert dataset.train_projects or dataset.all_projects or dataset.whole_split \
            or dataset.granularity == "line", f"{path.name}: {dataset.hf_repo} selects nothing"


def test_there_is_at_least_one_spec():
    assert SPECS

"""A model card keeps what was measured after training (EVALUATION.md).

The card is rebuilt from ``metadata.json`` on every publish, so a figure
measured later — the BLB page models read through the gateway on 02.11.2026 —
had nowhere to live: edited into the README on the hub, it would be gone on the
next push. A file beside the weights is uploaded with them and spliced in.
"""

from __future__ import annotations

import json
from pathlib import Path

from atr_training.publish import (
    CARD_FILENAME,
    EVALUATION_FILENAME,
    METADATA_FILENAME,
    model_card,
    publish_one,
    plan,
    scan_trained,
)
from test_training_publish import VLM_META, FakeUploader, trained_dir

TEXT = """# Through the gateway

| set | pages | median CER |
|---|---:|---:|
| lassberg_eval | 23 | 4.4 % |

## A caution

Fluent German that is not on the page.
"""


def _model(tmp_path: Path, with_file: bool = True):
    directory = trained_dir(tmp_path, VLM_META, weights="adapter_model.safetensors")
    if with_file:
        (directory / EVALUATION_FILENAME).write_text(TEXT, encoding="utf-8")
    return scan_trained(tmp_path).models[0]


def test_the_file_is_a_section_of_the_card(tmp_path):
    model = _model(tmp_path)
    card = model_card(model, "dh-unibe/qwen3vl-8b-thun")
    assert "## Measured after training" in card
    assert "| lassberg_eval | 23 | 4.4 % |" in card
    # its own headings sit below the section, never beside it
    assert "\n### Through the gateway\n" in card and "\n#### A caution\n" in card
    assert card.index("## Measured after training") < card.index("## Training data")
    assert card.index("## Evaluation") < card.index("## Measured after training")


def test_without_the_file_the_card_is_as_before(tmp_path):
    card = model_card(_model(tmp_path, with_file=False), "dh-unibe/qwen3vl-8b-thun")
    assert "Measured after training" not in card


def test_the_file_is_not_a_weight(tmp_path):
    model = _model(tmp_path)
    assert [p.name for p in model.weights] == ["adapter_model.safetensors"]
    assert EVALUATION_FILENAME not in model_card(model, "x/y").split("| weights |")[1].split("\n")[0]


def test_a_republish_keeps_it(tmp_path):
    model = _model(tmp_path)
    uploader = FakeUploader()
    publication = plan([model])[0]
    publish_one(publication, uploader)
    # a second push rewrites the card from metadata.json; the section survives
    model = scan_trained(tmp_path).models[0]
    publish_one(plan([model], force=True)[0], uploader)
    card = (model.directory / CARD_FILENAME).read_text(encoding="utf-8")
    assert card.count("## Measured after training") == 1
    assert "4.4 %" in card
    meta = json.loads((model.directory / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert "published" in meta, "the trainer's record is still the one that records the push"

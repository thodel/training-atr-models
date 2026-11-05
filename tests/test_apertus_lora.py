"""Apertus 1.5 LoRA on enlarged strips (#316, #317): what the trainer needs to know.

Apertus 1.5 reads an image as one discrete code per 16 x 16 pixels, and a letter
smaller than a cell does not survive (#314). Zero-shot on the 166 settled GT
pages, four-line strips enlarged 2x read at 22.4 % CER, native strips at 30.4 %,
the whole page at 31.1 % (#317). The probe trains at that scale, and these tests
pin the five things that make it the same scale and the same model everywhere:

1. the base runs in its own venv (swiss-ai's transformers fork),
2. its chat template takes the answer only as a string,
3. crops are enlarged as serving enlarges them, within the budget,
4. a page sample is never enlarged,
5. the registration launches it the way the gateway's Apertus entries launch.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from atr_training.backends import APERTUS_VENV, needs_apertus_fork, runner_python, venv_for
from atr_training.contracts import TrainRequest, VlmTrainParams
from atr_training.cropping import enlarge, write_crops
from atr_training.registration import (
    APERTUS_VLLM_ARGS,
    Registration,
    launch_problems,
    vllm_launch_defaults,
)
from atr_training.vlm_dataset import Sample, assistant_takes_parts, example_for

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = Path(__file__).parent / "fixtures" / "chat_templates"
APERTUS = (TEMPLATES / "apertus-v1.5-8b.jinja").read_text(encoding="utf-8")
QWEN35 = (TEMPLATES / "qwen3.5-4b.jinja").read_text(encoding="utf-8")
SPEC = ROOT / "ubelix" / "specs" / "apertus-1.5-8b-kurrent-xix-strip4-x2-probe.json"


# ── 1. venv ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("base", ["swiss-ai/Apertus-v1.5-8B", "swiss-ai/apertus-v1.5-70b"])
def test_an_apertus_base_runs_in_its_own_venv(base):
    assert needs_apertus_fork(base)
    assert venv_for("vllm", base) == APERTUS_VENV
    assert runner_python("vllm", "/r/.venvs", base) == Path("/r/.venvs/vlm-train-apertus/bin/python")


@pytest.mark.parametrize("base", ["swiss-ai/Apertus-8B-Instruct-2509", "Qwen/Qwen3.5-4B", None])
def test_nothing_else_moves_to_it(base):
    assert venv_for("vllm", base) != APERTUS_VENV
    assert venv_for("kraken", "swiss-ai/Apertus-v1.5-8B") == "kraken-train"


# ── 2. the chat template ───────────────────────────────────────────────────

class TemplateProcessor:
    """`apply_chat_template` as transformers renders a Jinja template."""

    def __init__(self, template: str) -> None:
        self.chat_template = template
        self.tokenizer = SimpleNamespace(chat_template=None)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        jinja2 = pytest.importorskip("jinja2")
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=["jinja2.ext.loopcontrols"])
        env.globals["raise_exception"] = raise_exception
        env.filters["tojson"] = lambda value, **k: json.dumps(value, **k)
        return env.from_string(self.chat_template).render(
            messages=messages, add_generation_prompt=add_generation_prompt,
            bos_token="<s>", **kw)


def test_apertus_takes_the_answer_as_a_string_only():
    """A list of parts raises `Invalid assistant content`: the first batch of
    the probe would have died on it."""
    processor = TemplateProcessor(APERTUS)
    assert assistant_takes_parts(processor) is False
    rendered = processor.apply_chat_template(
        example_for(processor, "Transcribe.", "Item ontfaen\nvan Janne"))
    assert "<|user_start|><|image|>Transcribe.<|user_end|>" in rendered
    assert rendered.endswith("<|assistant_start|>Item ontfaen\nvan Janne<|assistant_end|>")


def test_the_generation_prompt_ends_where_the_answer_starts():
    """The collator masks everything up to the assistant header; the header in
    the training render and in the generation prompt must be the same tokens."""
    processor = TemplateProcessor(APERTUS)
    prompt = processor.apply_chat_template(example_for(processor, "Transcribe."),
                                           add_generation_prompt=True)
    training = processor.apply_chat_template(example_for(processor, "Transcribe.", "X"))
    assert prompt.endswith("<|assistant_start|>")
    assert training.startswith(prompt)


def test_qwen_keeps_the_render_it_was_measured_on():
    processor = TemplateProcessor(QWEN35)
    assert assistant_takes_parts(processor) is True
    turn = example_for(processor, "Transcribe.", "X")[-1]
    assert turn["content"] == [{"type": "text", "text": "X"}]


# ── 3./4. scale ────────────────────────────────────────────────────────────

def test_enlarge_follows_serving_rule():
    """`min(scale, sqrt(budget / area))`, as serving's crop_strip (#241, #243)."""
    from PIL import Image

    small = Image.new("RGB", (400, 100))
    assert enlarge(small, 2.0, 1_000_000).size == (800, 200)
    wide = Image.new("RGB", (5000, 800))                 # an Aarau strip
    out = enlarge(wide, 2.0, 1_000_000)
    assert out.width * out.height <= 1_000_000 and out.width < 5000


def test_write_crops_enlarges_within_the_kind_budget(tmp_path):
    from PIL import Image

    (tmp_path / "pages").mkdir()
    Image.new("RGB", (1000, 800), "white").save(tmp_path / "pages" / "p.jpg")
    samples = [Sample(image="pages/p.jpg", text="a\nb", source_type="block",
                      bbox=[0, 0, 400, 100], page="pages/p.xml")]
    out = write_crops(samples, tmp_path, tmp_path / "crops", scale=2.0,
                      budgets={"block": 1_000_000})
    assert Image.open(tmp_path / out[0].image).size == (800, 200)
    native = write_crops(samples, tmp_path, tmp_path / "native")
    assert Image.open(tmp_path / native[0].image).size == (400, 100)


def test_a_page_sample_is_never_enlarged():
    with pytest.raises(ValidationError, match="page sample"):
        VlmTrainParams(granularity="page", crop_scale=2.0)
    assert VlmTrainParams(granularity="block", crop_scale=2.0).crop_scale == 2.0


# ── 5. registration ────────────────────────────────────────────────────────

def test_a_trained_apertus_launches_like_the_gateways_apertus():
    defaults = vllm_launch_defaults("vllm", "swiss-ai/Apertus-v1.5-8B")
    assert defaults["vllm_venv"] == "vllm-apertus"
    assert defaults["vllm_args"] == APERTUS_VLLM_ARGS
    assert "--kv-cache-memory-bytes" in defaults["vllm_args"]       # serving#239
    assert defaults["startup_timeout_s"] == 900


def test_an_apertus_registration_without_its_venv_is_refused():
    spec = Registration(id="m", engine="vllm", local_path="/x",
                        base_model="swiss-ai/Apertus-v1.5-8B")
    assert any("vllm-apertus" in p for p in launch_problems(spec))
    ok = spec.model_copy(update=vllm_launch_defaults("vllm", spec.base_model))
    assert launch_problems(ok) == []


# ── the probe's spec ───────────────────────────────────────────────────────

def test_the_probe_spec_trains_at_the_scale_it_was_measured_at():
    raw = json.loads(SPEC.read_text(encoding="utf-8"))
    request = TrainRequest(**{k: v for k, v in raw.items() if k != "notes"})
    params = request.params
    assert request.base_model == "swiss-ai/Apertus-v1.5-8B"
    assert params.granularity == "block" and params.block_lines == 4
    assert params.crop_scale == 2.0 and params.max_pixels == 1_000_000
    assert params.load_in_4bit is False
    assert "vision_tokenizer" in params.exclude_modules
    assert "gate_proj" not in params.target_modules                  # Apertus has none


def test_the_smoke_script_parses_the_trainers_own_argv_for_the_probe():
    """scripts/apertus_lora_smoke.py drives the trainer with the argv the runner
    would build; that argv has to parse, or the smoke test proves nothing."""
    import sys
    sys.path.insert(0, str(ROOT / "engines"))
    from atr_training.vlm_cmd import train_cmd
    from vlm_train_svc.train_qlora import parse_args

    raw = json.loads(SPEC.read_text(encoding="utf-8"))
    request = TrainRequest(**{k: v for k, v in raw.items() if k != "notes"})
    argv = train_cmd("python", params=request.params, base_model=request.base_model,
                     train_jsonl="t", val_jsonl="v", data_root=".", output_dir="o")[3:]
    job = parse_args(argv)
    assert job.max_pixels == 1_000_000 and job.load_in_4bit is False
    assert "vision_tokenizer" in job.exclude_modules

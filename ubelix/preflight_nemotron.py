"""Stage A of the Llama-Nemotron feasibility package (#135, F5): no weights, no GPU.

F5 asks two questions the other families never raised — does the vendor's remote
code load **offline** in our container, and does the image processor have a budget
knob **at all** — and the eight points of the family pre-flight still apply. Every
one of those answers lives in the *small* files: config, remote code, tokenizer,
image processor. The weights are 17 GB for the 8B and 27 GB for the 12B, and
fetching them before the answer is known is the expensive mistake this stage
exists to avoid.

Two of the eight points are *not* covered here, because they need the weights:
the module tree behind ``target_modules`` (point 5) and the 4-bit forward and
evaluation pass (point 8). Those are stage B, and only worth paying for if this
stage comes back clean.

    python preflight_nemotron.py nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1

Run it **twice** — once to populate the cache, then again with ``HF_HUB_OFFLINE=1``
— because "it loaded" and "it loads without the internet" are different answers,
and the second one is the only one a compute node can use. These models reach for
repos that are named nowhere in their own file listing (the vision tower, and for
the 12B the language model too), so the offline pass is the real test of whether a
pre-download was complete.

Reads two sample corpora for the measured tiling; writes nothing but the HF cache.
"""
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"
JOBS = Path(os.environ.get("PREFLIGHT_JOBS", "/scratch/network/users/th19c587/runs/jobs"))
LINE_JOB = os.environ.get("PREFLIGHT_LINE_JOB", "20260925T060317Z-ladder-med-gemma4-e4b")
PAGE_JOB = os.environ.get("PREFLIGHT_PAGE_JOB",
                          "20260923T202356Z-qwen3vl-medieval-german-page-v1")
PROMPT = "Transcribe the handwritten text in this image exactly as written."
TILE_CANDIDATES = (1, 2, 4, 8, 12)


def head(label, title):
    print("\n== %s %s" % (label, title), flush=True)


def fail(exc, limit=200):
    print("   -> %s: %s" % (type(exc).__name__, str(exc)[:limit]), flush=True)


def sample(job, kind):
    """The first row of a job's evaluation draw, as (PIL image, reference text)."""
    from PIL import Image
    root = JOBS / job
    row = json.loads((root / "data" / "val_eval.jsonl").read_text().splitlines()[0])
    image = Image.open(root / row["image"]).convert("RGB")
    print("   %-5s %-28s %s px" % (kind, Path(row["image"]).name, "x".join(map(str, image.size))))
    return image, row["text"]


print("base:        ", BASE)
print("HF_HUB_OFFLINE:", os.environ.get("HF_HUB_OFFLINE", "(unset)"))
print("HF_HOME:     ", os.environ.get("HF_HOME", "(unset)"))

import torch                                                          # noqa: E402
import transformers                                                   # noqa: E402
from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, AutoTokenizer  # noqa: E402
from transformers import AutoModelForImageTextToText                   # noqa: E402

print("transformers:", transformers.__version__, "| torch:", torch.__version__)

from atr_training.contracts import VLM_PIXEL_BUDGET, VlmTrainParams    # noqa: E402
from atr_training.vlm_dataset import (CHAT_TEMPLATE_KWARGS, apply_visual_budget,  # noqa: E402
                                      chat_example, fit_pixels)
from vlm_train_svc.evaluate_qlora import stop_token_ids                # noqa: E402
from vlm_train_svc.train_qlora import assistant_header_ids             # noqa: E402

# ---------------------------------------------------------------- 1. container
head("1.", "container: does transformers know this architecture, and offline?")
cfg = None
try:
    cfg = AutoConfig.from_pretrained(BASE, trust_remote_code=True)
    print("   model_type:   ", cfg.model_type)
    print("   architectures:", getattr(cfg, "architectures", None))
    auto_map = getattr(cfg, "auto_map", {}) or {}
    print("   auto_map:     ", dict(auto_map))
    print("   AutoModelForImageTextToText in auto_map:",
          "AutoModelForImageTextToText" in auto_map)
    print("   resolves in the static mapping:",
          AutoModelForImageTextToText._model_mapping.get(type(cfg), "NOT IN MAPPING"))
    # The repos these models pull that their own file listing does not name. Read
    # from the raw config, not from the parsed objects: a sub-config keeps or drops
    # auto_map depending on its class, and what matters is what the file asks for.
    from transformers.utils import cached_file
    raw = json.loads(Path(cached_file(BASE, "config.json")).read_text())
    foreign = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "auto_map" and isinstance(value, dict):
                    for ref in value.values():
                        if "--" in str(ref):
                            foreign.add(str(ref).split("--")[0])
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(raw)
    print("   foreign code repos required:", sorted(foreign) or "none")
except Exception as exc:
    fail(exc, 400)
    traceback.print_exc()

# ------------------------------------------------- 2. the training forward path
head("2.", "the resolved model class: can our trainer call its forward?")
try:
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.utils import cached_file
    raw_cfg = json.loads(Path(cached_file(BASE, "config.json")).read_text())
    ref = (raw_cfg.get("auto_map") or {}).get("AutoModel")
    cls = get_class_from_dynamic_module(ref, BASE)
    print("   class:", cls.__name__, "from", ref)
    params = inspect.signature(cls.forward).parameters
    print("   forward accepts:", ", ".join(p for p in params if p != "self"))
    for needed in ("pixel_values", "input_ids", "labels", "image_flags", "num_patches"):
        print("     %-14s %s" % (needed, "yes" if needed in params else "NO"))
    src = inspect.getsource(cls.forward)
    print("   forward dereferences image_flags unconditionally:",
          "image_flags.squeeze" in src and "if image_flags" not in src)
    print("   forward calls torch.distributed.get_rank():",
          "torch.distributed.get_rank()" in src)
except Exception as exc:
    fail(exc, 400)
    # The class would not import. The file is still on disk, and the two questions
    # that decide whether our trainer can call it are answerable from the text.
    try:
        from transformers.utils import cached_file
        src = Path(cached_file(BASE, "modeling.py")).read_text()
        # The file defines several forwards — a squared ReLU and an RMSNorm come
        # first. Taking the first one measures the wrong function and answers no
        # to both questions, which is how this check lied on its first run.
        name = (raw_cfg["auto_map"]["AutoModel"]).rsplit(".", 1)[-1]
        klass = src.split("class %s(" % name, 1)[1]
        body = klass.split("    def forward(", 1)[1].split("\n    def ", 1)[0]
        print("   read from the cached modeling.py instead (class %s):" % name)
        print("     forward signature names image_flags:",
              "image_flags" in body.split("):", 1)[0])
        print("     forward dereferences image_flags unconditionally:",
              "image_flags.squeeze" in body and "if image_flags" not in body)
        print("     forward calls torch.distributed.get_rank():",
              "torch.distributed.get_rank()" in body)
    except Exception as exc2:
        fail(exc2)

# ---------------------------------------------------------------- 3. processor
head("3.", "processor: is there one our collator can drive?")
proc = ip = tok = None
try:
    proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
    ip = getattr(proc, "image_processor", None)
    tok = getattr(proc, "tokenizer", None)
    print("   AutoProcessor:  ", type(proc).__name__)
except Exception as exc:
    print("   AutoProcessor:  unavailable")
    fail(exc)
    try:
        ip = AutoImageProcessor.from_pretrained(BASE, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
        print("   fallback:        AutoImageProcessor + AutoTokenizer "
              "(our collator calls processor(text=..., images=...) and has no such path)")
    except Exception as exc2:
        fail(exc2)
print("   image_processor:", type(ip).__name__ if ip is not None else None)
for attr in ("image_size", "max_num_tiles", "use_thumbnail", "num_image_token",
             "size", "max_pixels", "min_pixels", "patch_size", "merge_size",
             "max_soft_tokens", "pooling_kernel_size", "image_mean", "image_std"):
    if ip is not None and hasattr(ip, attr):
        print("     %-18s %s" % (attr, getattr(ip, attr)))

# ------------------------------------------------------------- 4. budget knob
head("4.", "pixel budget: what apply_visual_budget finds")
for kind, px in sorted(VLM_PIXEL_BUDGET.items(), key=lambda kv: kv[1]):
    try:
        fresh = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
        print("   %-6s %8d px -> %s" % (kind, px, apply_visual_budget(fresh, px)))
    except Exception as exc:
        print("   %-6s %8d px -> %s: %s" % (kind, px, type(exc).__name__, str(exc)[:180]))

# --------------------------------------------- 5. measured tiling, not arithmetic
head("5.", "cell size and tokens: measured through the real image processor")
images = {}
for kind, job in (("line", LINE_JOB), ("page", PAGE_JOB)):
    try:
        images[kind] = sample(job, kind)[0]
    except Exception as exc:
        fail(exc)
if ip is not None and hasattr(ip, "max_num_tiles"):
    tile_px = int(getattr(ip, "image_size", 512)) ** 2
    per_tile = getattr(ip, "num_image_token", None)
    source = "image_processor.num_image_token"
    if per_tile is None:
        from transformers.utils import cached_file
        raw_cfg = json.loads(Path(cached_file(BASE, "config.json")).read_text())
        side = int(raw_cfg["force_image_size"]) // int(raw_cfg["patch_size"])
        per_tile = int(side ** 2 * float(raw_cfg["downsample_ratio"]) ** 2)
        source = ("config: (force_image_size/patch_size)^2 * downsample_ratio^2 = "
                  "(%d/%d)^2 * %s^2" % (raw_cfg["force_image_size"], raw_cfg["patch_size"],
                                        raw_cfg["downsample_ratio"]))
    print("   one tile = %d px; tokens per tile = %s (%s)" % (tile_px, per_tile, source))
    print("   so one visual token carries %d px -> an effective cell of %d px "
          "(Qwen3-VL: 32, olmOCR: 28)" % (tile_px // per_tile, int((tile_px / per_tile) ** 0.5)))
    print("   budget in whole tiles: " + ", ".join(
        "%s %d" % (k, max(1, round(v / tile_px)))
        for k, v in sorted(VLM_PIXEL_BUDGET.items(), key=lambda kv: kv[1])))
    original = ip.max_num_tiles
    # A line strip at the line budget is the interesting case: one tile is a square,
    # so the grid has no way to keep an 18:1 aspect ratio. The block budget is shown
    # beside it because it is the cheapest grid that can.
    for kind, budget_kind in (("line", "line"), ("line", "block"), ("page", "page")):
        image = images.get(kind)
        if image is not None:
            px = VLM_PIXEL_BUDGET[budget_kind]
            fitted = fit_pixels(image, px)
            print("   %s crop fitted to the %s budget (%s px):"
                  % (kind, budget_kind, "x".join(map(str, fitted.size))))
            for tiles in TILE_CANDIDATES:
                ip.max_num_tiles = tiles
                try:
                    out = ip(images=[fitted])
                    patches = list(out["num_patches"])
                    shape = tuple(out["pixel_values"].shape)
                    tokens = patches[0] * per_tile if per_tile else None
                    print("     max_num_tiles=%-3d patches=%-3d pixel_values=%-22s "
                          "tokens=%-5s pixels seen=%d"
                          % (tiles, patches[0], shape, tokens, patches[0] * tile_px))
                except Exception as exc:
                    print("     max_num_tiles=%-3d %s: %s"
                          % (tiles, type(exc).__name__, str(exc)[:90]))
    ip.max_num_tiles = original
else:
    print("   no max_num_tiles on this image processor — nothing to measure here")

# -------------------------------------------------------- 6. image list form
head("6.", "image list form: one list per text, or one flat list per batch")
if proc is not None and images:
    image = fit_pixels(images.get("line") or next(iter(images.values())),
                       VLM_PIXEL_BUDGET["line"])
    try:
        text = proc.apply_chat_template(chat_example(PROMPT), tokenize=False,
                                        add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS)
        print("   rendered prompt contains '<image>':", "<image>" in text)
        print("   render:", repr(text[:160]))
    except Exception as exc:
        text = None
        print("   apply_chat_template failed — our collator renders every sample this way")
        fail(exc)
    if text is not None:
        for label, arg in (("nested [[img]]", [[image]]), ("flat [img]", [image])):
            try:
                out = proc(text=[text], images=arg, return_tensors="pt", padding=True)
                print("   %-16s OK  keys=%s input_ids=%s pixel_values=%s"
                      % (label, sorted(out.keys()), tuple(out["input_ids"].shape),
                         tuple(out["pixel_values"].shape)))
            except Exception as exc:
                print("   %-16s %s: %s" % (label, type(exc).__name__, str(exc)[:120]))
else:
    print("   skipped: no processor, or no sample images")

# ------------------------------------------- 7. header, stop tokens, exclusion
head("7.", "assistant header, stop tokens, and what our module names would hit")
if proc is not None:
    try:
        header = assistant_header_ids(proc, PROMPT)
        rendered = proc.apply_chat_template(chat_example(PROMPT, "ZZSAMPLE"), tokenize=False,
                                            add_generation_prompt=False, **CHAT_TEMPLATE_KWARGS)
        ids = proc.tokenizer(rendered, add_special_tokens=False).input_ids
        n = len(header)
        print("   header ids:", header, "->", repr(proc.tokenizer.decode(header)))
        print("   present in a real training render:",
              any(ids[i:i + n] == header for i in range(len(ids) - n + 1)))
    except Exception as exc:
        fail(exc)
    try:
        stops = stop_token_ids(proc.tokenizer, proc)
        print("   stop ids:", stops, "->", [proc.tokenizer.decode([s]) for s in stops])
    except Exception as exc:
        fail(exc)
print("   target_modules:  ", VlmTrainParams().target_modules)
print("   exclude_modules: ", VlmTrainParams().exclude_modules)
print("   note: this family names its tower 'vision_model', which that regex does "
      "not match. Whether anything is hit there is stage B.")

print("\n== done", flush=True)

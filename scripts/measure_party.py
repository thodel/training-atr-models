#!/usr/bin/env python3
"""How well does the party base model read our held-out material? (#432)

party (mittagessen/party, a Swin encoder with a Llama decoder) reads a page and
is prompted with one line at a time, by its baseline curve. Behind the gateway it
reads what kraken ``blla`` segmented (serving-atr-inference
``engines/party_svc/app.py``), and nobody has measured it on our sets: it has
only ever run as a second opinion (serving-atr-inference#265). Whether to
fine-tune it on our page corpora, or to stay with the VLM page route (#59),
depends on that number.

Two readings, so that segmentation and recognition show apart:

    gt      the ground truth's own baselines (``XMLPage.to_container()``), scored
            per line and per page: recognition alone, comparable to the line models
    blla    kraken ``blla`` first, as the gateway does, scored per page: what a
            caller gets today

Inputs:

    --page-set DIR   a frozen page draw (``eval_sets/<set>/vlm-pages-<side>/``),
                     checked against its ``benchmark.json`` like eval_granularity.py
    --lines FILE     a line set whose rows name a PageXML per line (the Federal
                     Council lines, ``eval_sets/federal-minutes/data/val.jsonl``);
                     ``gt`` only, since one line is not a page to segment

The settings are the gateway's: ``precision 32-true``, batch 1, the prompt mode
taken from the segmentation, ``max_generated_tokens`` = min(512, the decoder's
``max_seq_len``). Texts are compared with whitespace collapsed and scored with
``textmetrics.score_pairs``, as every other number on these sets.

Runs in serving-atr-inference's party venv, with this repository's ``src`` on the
path for the metric (it needs pydantic and the stdlib only):

    cd ~/Repo/training-atr-models
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \\
      ~/Repo/serving-atr-inference/.venvs/party/bin/python scripts/measure_party.py \\
        --model ~/atr-cache/party/model.safetensors \\
        --page-set /mnt/wbkolleg_dh_1/Textrecognition_Training/eval_sets/german-kurrent-xvi-xix-v1/vlm-pages-val \\
        --modes gt,blla --out party-base.xvi-xix-val.json

The model is a foreign base: whether its training data hold our documents is not
known, so the result is a reading, not a held-out number.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from atr_training.textmetrics import score_pairs  # noqa: E402
from eval_granularity import PageSetError, flat, read_page_set  # noqa: E402

#: What party's own default asks for (serving-atr-inference party_svc, #149).
REQUESTED_TOKENS = 512
MODES = ("gt", "blla")


def summarise(level: str, items: list[dict]) -> dict:
    """Corpus CER with the two failure shapes a mean hides, as eval_granularity does."""
    score = score_pairs((flat(i["hyp"]), flat(i["ref"])) for i in items)
    return {
        "level": level,
        "n": len(items),
        "ref_chars": score.chars,
        "cer": round(score.cer, 4) if items else None,
        "length_ratio": round(score.length_ratio, 3) if items else None,
        "collapsed<1/3": sum(1 for i in items if len(flat(i["hyp"])) < len(flat(i["ref"])) / 3),
        "runaway>1.5x": sum(1 for i in items if len(flat(i["hyp"])) > 1.5 * len(flat(i["ref"]))),
        "failed": sum(1 for i in items if i.get("error")),
        "sec_per_item": round(sum(i.get("sec", 0.0) for i in items) / max(len(items), 1), 2),
    }


def page_items(page: str, ref_lines: list[str], hyp_lines: list[str], sec: float,
               error: str | None = None) -> tuple[list[dict], dict]:
    """Line rows (paired by position) and the page row for one page read on its own lines.

    With the ground truth's segmentation party reads exactly the reference's
    lines, so position pairs them; a missing hypothesis counts as empty.
    """
    lines = [{"page": page, "line": n, "ref": ref, "hyp": hyp_lines[n] if n < len(hyp_lines) else "",
              "error": error}
             for n, ref in enumerate(ref_lines)]
    page_row = {"page": page, "ref": "\n".join(ref_lines), "hyp": "\n".join(hyp_lines),
                "lines": len(ref_lines), "sec": sec, "error": error}
    return lines, page_row


class Reader:
    """party loaded once, read the way the gateway reads it. Torch, kraken and party
    are imported here, so the scoring above stays testable without them."""

    def __init__(self, model_path: Path, device: str) -> None:
        import torch  # noqa: PLC0415
        from kraken.tasks import RecognitionTaskModel  # noqa: PLC0415
        from party.configs import PartyRecognitionInferenceConfig  # noqa: PLC0415

        self.device = device
        self.model = RecognitionTaskModel.load_model(model_path)
        try:
            limit = int(self.model.net.nn["decoder"].max_seq_len)
        except (AttributeError, KeyError, TypeError, ValueError):
            limit = REQUESTED_TOKENS
        self.max_tokens = min(REQUESTED_TOKENS, limit)
        cuda = device.startswith("cuda") and torch.cuda.is_available()
        self.config = PartyRecognitionInferenceConfig(
            accelerator="cuda" if cuda else "cpu", device=1, precision="32-true",
            num_threads=1, batch_size=1, prompt_mode=None,
            max_generated_tokens=self.max_tokens, add_lang_token=True, raise_on_error=False,
        )

    def read(self, image, segmentation) -> list[str]:
        records = self.model.predict(im=image, segmentation=segmentation, config=self.config)
        return [getattr(r, "prediction", None) or "" for r in records]

    def gt(self, xml: Path, image_path: Path | None = None):
        from kraken.lib.xml import XMLPage  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        seg = XMLPage(xml).to_container()
        lines = [ln for ln in seg.lines if (ln.text or "").strip()]
        seg.lines = lines
        img = Image.open(image_path or seg.imagename).convert("RGB")
        return [ln.text for ln in lines], self.read(img, seg)

    def blla(self, image_path: Path) -> list[str]:
        from kraken import blla  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        img = Image.open(image_path).convert("RGB")
        return self.read(img, blla.segment(img, device=self.device))


def _timed(fn, *args):
    t0 = time.perf_counter()
    try:
        return fn(*args), round(time.perf_counter() - t0, 2), None
    except Exception as exc:  # noqa: BLE001 - one bad page must not end a 500-page reading
        return None, round(time.perf_counter() - t0, 2), f"{type(exc).__name__}: {exc}"[:300]


def measure_page_set(reader, path: Path, modes: tuple[str, ...], max_pages: int | None,
                     raw) -> dict:
    pages, identity = read_page_set(path)  # refuses a draw that is not the frozen one
    draw = json.loads((path / "benchmark.json").read_text(encoding="utf-8")).get("draw",
                                                                                  "pages.jsonl")
    rows = {row["page"]: row for row in (json.loads(line) for line in (path / draw).read_text(
        encoding="utf-8").splitlines() if line.strip())}
    pages = pages[:max_pages] if max_pages else pages
    out: dict[str, list[dict]] = {"gt-line": [], "gt-page": [], "blla-page": []}
    for n, page in enumerate(pages, 1):
        row = rows[page]
        xml, image = path / page, path / row["image"]
        if "gt" in modes:
            got, sec, err = _timed(reader.gt, xml, image)
            refs, hyps = got if got else ([], [])
            lines, page_row = page_items(page, refs, hyps, sec, err)
            out["gt-line"] += lines
            out["gt-page"].append(page_row)
            raw.write(json.dumps({"mode": "gt", **page_row, "hyp_lines": hyps},
                                 ensure_ascii=False) + "\n")
        if "blla" in modes:
            got, sec, err = _timed(reader.blla, image)
            item = {"page": page, "ref": row["text"], "hyp": "\n".join(got or []),
                    "lines": len(got or []), "sec": sec, "error": err}
            out["blla-page"].append(item)
            raw.write(json.dumps({"mode": "blla", **item}, ensure_ascii=False) + "\n")
        raw.flush()
        print(f"{n}/{len(pages)} {page}", flush=True)
    return {"set": identity, "levels": [summarise(k, v) for k, v in out.items() if v]}


def measure_lines(reader, lines_file: Path, data_root: Path, max_lines: int | None, raw) -> dict:
    rows = [json.loads(line) for line in lines_file.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    rows = rows[:max_lines] if max_lines else rows
    items = []
    for n, row in enumerate(rows, 1):
        got, sec, err = _timed(reader.gt, data_root / row["page"])
        refs, hyps = got if got else ([], [])
        # The row's own text is the reference, as for every other reading of the set.
        item = {"page": row["page"], "ref": row["text"], "hyp": " ".join(hyps),
                "pagexml_lines": len(refs), "sec": sec, "error": err}
        items.append(item)
        raw.write(json.dumps(item, ensure_ascii=False) + "\n")
        if n % 100 == 0:
            raw.flush()
            print(f"{n}/{len(rows)}", flush=True)
    return {"lines_file": str(lines_file), "levels": [summarise("gt-line", items)]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", type=Path, required=True, help="party model file (.safetensors)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--page-set", type=Path)
    src.add_argument("--lines", type=Path, help="a line set whose rows name a PageXML per line")
    ap.add_argument("--data-root", type=Path, help="root of --lines (default: its parent's parent)")
    ap.add_argument("--modes", default="gt,blla", help="gt, blla or both (page sets only)")
    ap.add_argument("--max-items", type=int, default=None, help="first N pages or lines (smoke)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    if not set(modes) <= set(MODES):
        ap.error(f"--modes: pick from {MODES}")
    reader = Reader(args.model, args.device)
    raw_path = args.out.with_suffix(".raw.jsonl")
    with open(raw_path, "w", encoding="utf-8") as raw:
        try:
            if args.page_set:
                report = measure_page_set(reader, args.page_set, modes, args.max_items, raw)
            else:
                root = args.data_root or args.lines.parent.parent
                report = measure_lines(reader, args.lines, root, args.max_items, raw)
        except PageSetError as exc:
            print(exc, file=sys.stderr)
            return 2
    report.update(model=str(args.model), max_generated_tokens=reader.max_tokens,
                  precision="32-true", raw=str(raw_path),
                  note="party is a foreign base: a reading, not a held-out number (#432)")
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for level in report["levels"]:
        print(json.dumps(level, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    from atr_training.measurement_record import measurement  # noqa: E402

    # A record next to the job store, so live() and results() see this run (#385).
    with measurement(Path(__file__).stem):
        raise SystemExit(main())

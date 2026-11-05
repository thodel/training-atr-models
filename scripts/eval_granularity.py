#!/usr/bin/env python3
"""How does a VLM read single lines, paragraphs and whole pages — against ground truth?

A model trained on line crops is served per line (kraken segments, one call per
line) or per page (one call for the whole scan). Whether the page shape works is a
question about the model, and a number on line crops does not answer it: on
2026-09-22 `qwen3vl-medieval-german-v3` read held-out lines at CER 0.111 and every
one of 14 whole pages as the two characters "de" (#159, serving-atr-inference/docs/VLM_TRAINING.md).

This script asks the model four ways over the same PageXML pages:

    line                  every transcribed line crop, at the line pixel budget
    block@block_budget    runs of --block-lines consecutive lines, cropped as one
                          image, the unit `granularity: block` trains on
                          (training-atr-models#57)
    region@page_budget    every TextRegion (paragraph), cropped from the page,
                          at the page budget (never upscaled)
    region@line_budget    the same crops squeezed into the line budget
    page@page_budget      the whole scan at the page budget, as /recognize does
    page@recognize        (--recognize) the whole scan through the gateway's
                          /recognize, i.e. exactly what a caller gets

Ground truth and geometry come from the training code (`line_boxes`,
`page_sample`), the metric from `textmetrics.score_pairs` — the one behind every
stored CER. Texts are compared with whitespace collapsed, so a paragraph read as
one long line is not penalised for its missing line breaks.

    # against a vLLM you started yourself — the normal case
    .venvs/gateway/bin/python scripts/eval_granularity.py \\
        --root <job dir with data/<jsonl> and data/pages/> --jsonl val_heldout.jsonl \\
        --model qwen3vl-medieval-german-v3 --out report.json \\
        --base-url http://127.0.0.1:8299 --no-auth

    # against the production gateway (ATR_API_KEY from .env) — only when it is idle
    ... --base-url http://127.0.0.1:8200 [--recognize]

**Do not point this at the production gateway while it serves anyone.** Card 1 of
idhefix holds the engines and one VLM; asking for a second model evicts the one a
caller is using, and two launches racing for the same free memory both fail. On
2026-09-22 a run for `qwen3vl-german-xix-v2` cost tei's live requests to
`qwen3.5-4b-german-xix-v2` three 502s and a 503 within one minute. Serve the model
on a spare card instead — asteraix when it is not training — bound to 127.0.0.1.

`--jsonl` rows need `image` (a line crop, relative to --root), `text` and `page`
(the PageXML beside its .jpg). `--max-pages` draws that many pages, spread over the
sources the file names start with.

A frozen held-out page set (`eval_sets/<set>/vlm-pages-<side>/`, written by
`scripts/build_page_benchmark.py`) is read with `--page-set` instead of
`--root`/`--jsonl`. It holds pages only, so the line level is cut from the same
PageXML with the rule training cuts it by (`line_samples`): every level then
reads the same pages (#376). The draw's md5 is checked against its
`benchmark.json` first, and the set's identity goes into the report.

    ... --page-set <share>/eval_sets/german-kurrent-xvi-xix-v1/vlm-pages-test

The prompt and the budgets default to the Qwen line contract the script was
written for (`VLM_PROMPT`, `VLM_PIXEL_BUDGET`). A model trained on another
contract is asked on its own terms — an olmOCR px4m-native page arm with its
spec's prompt and 4 MP:

    ... --prompt "$(jq -r .params.prompt ubelix/specs/<spec>.json)" \\
        --page-pixels 4194304

`--page-pixels` is the budget of both `@page_budget` levels, `--line-pixels`
of `line` and `region@line_budget`, `--max-tokens-page` the generation cap of
the region and page levels. The report records the prompt and every level's
budget, so two reports say themselves whether they asked the same question.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_training.contracts import VLM_PIXEL_BUDGET, VLM_PROMPT  # noqa: E402
from atr_training.pagexml import _localname, line_boxes  # noqa: E402
from atr_training.textmetrics import score_pairs  # noqa: E402
from atr_training.vlm_dataset import line_samples, page_sample  # noqa: E402

LINE_PX, BLOCK_PX, PAGE_PX = (VLM_PIXEL_BUDGET[k] for k in ("line", "block", "page"))
#: Generation caps per call. Line and block are fixed; the region and page
#: levels share one, ``--max-tokens-page``.
LINE_TOKENS, BLOCK_TOKENS, PAGE_TOKENS = 512, 1024, 4096
REGION_PAD = 12
DEFAULT_BLOCK_LINES = 6
LEVELS = ("line", "block@block_budget", "region@page_budget", "region@line_budget",
          "page@page_budget")


class Budgets(NamedTuple):
    """The question every call asks: the prompt, and per level pixels and tokens.

    The defaults are the Qwen line contract (`VLM_PROMPT`, `VLM_PIXEL_BUDGET`)
    and the caps this script always used, so a run without the flags sends the
    requests it sent before they existed. A NamedTuple, not a dataclass: the
    script is also loaded without a module entry (`tests/test_moved_scripts.py`),
    which a dataclass under `from __future__ import annotations` cannot survive.
    """

    prompt: str = VLM_PROMPT
    line_pixels: int = LINE_PX
    block_pixels: int = BLOCK_PX
    page_pixels: int = PAGE_PX
    max_tokens_page: int = PAGE_TOKENS

    def levels(self) -> dict[str, tuple[int, int]]:
        """``(max_pixels, max_tokens)`` per level, in `LEVELS` order."""
        return {
            "line": (self.line_pixels, LINE_TOKENS),
            "block@block_budget": (self.block_pixels, BLOCK_TOKENS),
            "region@page_budget": (self.page_pixels, self.max_tokens_page),
            "region@line_budget": (self.line_pixels, self.max_tokens_page),
            "page@page_budget": (self.page_pixels, self.max_tokens_page),
        }

    def record(self) -> dict:
        """What the report keeps of it."""
        return {"prompt": self.prompt,
                "budgets": {k: {"max_pixels": px, "max_tokens": tok}
                            for k, (px, tok) in self.levels().items()}}


class PageSetError(RuntimeError):
    """The directory is not a frozen page draw, or not the one its record names."""


def read_page_set(path: Path) -> tuple[list[str], dict]:
    """The pages of a draw `build_page_benchmark.py` wrote, and its identity.

    Refuses a draw whose md5 is not the one its ``benchmark.json`` recorded: the
    set's name is what every number from it cites, so an edited draw must not
    carry it.
    """
    record_path = path / "benchmark.json"
    if not record_path.is_file():
        raise PageSetError(f"{path} has no benchmark.json; build it with "
                           "scripts/build_page_benchmark.py")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    draw = path / record.get("draw", "pages.jsonl")
    md5 = hashlib.md5(draw.read_bytes()).hexdigest()
    if md5 != record.get("draw_md5"):
        raise PageSetError(f"{draw} has md5 {md5}, benchmark.json records "
                           f"{record.get('draw_md5')}: not the frozen draw")
    with open(draw, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    pages = sorted({r["page"] for r in rows})
    identity = {k: record.get(k) for k in
                ("set", "side", "documents", "pages", "page_digest", "samples", "draw_md5")}
    return pages, identity


def page_lines(root: Path, page: str, page_size: tuple[int, int]) -> list[dict]:
    """The page's line rows — ``image``, ``bbox``, ``text``, ``page`` — cut as training cuts them.

    `line_samples` is the rule `granularity: line` trains on: padded line
    boxes clamped to the page, implausible and empty lines dropped.
    """
    return [{"image": s.image, "bbox": s.bbox, "text": s.text, "page": page}
            for s in line_samples(root / page, root=root, page_size=page_size)]


def flat(text: str) -> str:
    """Whitespace collapsed: line breaks are layout, not transcription."""
    return re.sub(r"\s+", " ", text).strip()


def fit(img, max_pixels: int):
    """Scale down so width*height <= max_pixels; never scale up."""
    from PIL import Image

    w, h = img.size
    if w * h <= max_pixels:
        return img
    s = (max_pixels / (w * h)) ** 0.5
    return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)


def region_boxes(xml_text: str, page_size: tuple[int, int], pad: int = REGION_PAD):
    """``(region_id, n_lines, (l, t, r, b), text)`` per TextRegion with transcribed lines.

    The box is the union of the region's line boxes, padded and clamped to the
    page — not the region's own Coords, which in Transkribus exports often cover
    margins and neighbouring text. The text is its lines joined by newlines, read
    with the same `_own_text` rule the training corpus uses (#125).
    """
    root = ET.fromstring(xml_text)
    width, height = page_size
    for region in root.iter():
        if _localname(region.tag) != "TextRegion":
            continue
        boxes = line_boxes(ET.tostring(region, encoding="unicode"))
        if not boxes:
            continue
        box = (
            max(min(b.left for b in boxes) - pad, 0),
            max(min(b.top for b in boxes) - pad, 0),
            min(max(b.right for b in boxes) + pad, width),
            min(max(b.bottom for b in boxes) + pad, height),
        )
        yield region.get("id"), len(boxes), box, "\n".join(b.text for b in boxes)


def line_region_ids(xml_text: str) -> list[str | None]:
    """The innermost TextRegion id per TextLine, in the order `line_boxes` counts.

    A copy of `atr_training.pagexml.line_regions`, which lives in the training
    repo: this script has to build exactly the blocks that
    `granularity: block` trains on, or the number it reports is about a
    different crop than the model saw.
    """
    root = ET.fromstring(xml_text)
    out: list[str | None] = []
    anonymous = 0

    def walk(el, region):
        nonlocal anonymous
        name = _localname(el.tag)
        if name == "TextRegion":
            if el.get("id"):
                region = el.get("id")
            else:
                anonymous += 1
                region = f"#region{anonymous}"
        elif name == "TextLine":
            out.append(region)
        for child in el:
            walk(child, region)

    walk(root, None)
    return out


def block_boxes(xml_text: str, page_size: tuple[int, int], block_lines: int = DEFAULT_BLOCK_LINES,
                pad: int = REGION_PAD):
    """``(index, n_lines, (l, t, r, b), text)`` per block of consecutive lines.

    Same rule as training: a run never crosses a region boundary, and never a
    line that is on the image without a usable transcription — that line would be
    visible in the crop and missing from the target.
    """
    regions = line_region_ids(xml_text)
    width, height = page_size
    runs: list[list] = []
    previous = None
    for box in line_boxes(xml_text):
        region = regions[box.index] if box.index < len(regions) else None
        joins = previous is not None and box.index == previous[0] + 1 and region == previous[1]
        if not joins:
            runs.append([])
        runs[-1].append(box)
        previous = (box.index, region)

    n = 0
    for run in runs:
        for start in range(0, len(run), block_lines):
            block = run[start:start + block_lines]
            box = (max(min(b.left for b in block) - pad, 0),
                   max(min(b.top for b in block) - pad, 0),
                   min(max(b.right for b in block) + pad, width),
                   min(max(b.bottom for b in block) + pad, height))
            yield n, len(block), box, "\n".join(b.text for b in block)
            n += 1


def summarise(level: str, items: list[dict]) -> dict:
    """Corpus CER plus the two failure shapes a mean hides: stopping and looping."""
    score = score_pairs((flat(i["hyp"]), flat(i["ref"])) for i in items)
    return {
        "level": level,
        "n": len(items),
        "ref_chars": score.chars,
        "cer": round(score.cer, 4),
        "length_ratio": round(score.length_ratio, 3),
        "collapsed<1/3": sum(1 for i in items if len(flat(i["hyp"])) < len(flat(i["ref"])) / 3),
        "runaway>1.5x": sum(1 for i in items if len(flat(i["hyp"])) > 1.5 * len(flat(i["ref"]))),
        "finish": dict(Counter(i.get("finish") for i in items)),
        "sec_per_item": round(sum(i["sec"] for i in items) / max(len(items), 1), 2),
    }


def by_line_count(items: list[dict]) -> dict[str, dict]:
    """Share of the reference text returned, by how many lines a region has."""
    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for i in items:
        n = i["lines"]
        key = "1" if n == 1 else "2-3" if n <= 3 else "4-10" if n <= 10 else ">10"
        buckets[key][0] += 1
        buckets[key][1] += len(flat(i["ref"]))
        buckets[key][2] += len(flat(i["hyp"]))
    return {k: {"n": n, "ref_chars": r, "hyp_chars": h, "ratio": round(h / r, 3) if r else None}
            for k, (n, r, h) in buckets.items()}


def spread_pages(pages: list[str], limit: int | None) -> list[str]:
    """``limit`` pages taken round-robin over sources (the file-name prefix)."""
    if not limit or limit >= len(pages):
        return pages
    by_source: dict[str, list[str]] = defaultdict(list)
    for p in pages:
        by_source[Path(p).name.split("_")[0]].append(p)
    out: list[str] = []
    queues = [sorted(v) for _, v in sorted(by_source.items())]
    while len(out) < limit and any(queues):
        for q in queues:
            if q and len(out) < limit:
                out.append(q.pop(0))
    return sorted(out)


class Client:
    def __init__(self, base_url: str, model: str, api_key: str | None,
                 prompt: str = VLM_PROMPT) -> None:
        self.base_url, self.model, self.prompt = base_url.rstrip("/"), model, prompt
        self.headers = {"X-API-Key": api_key} if api_key else {}

    def payload(self, url: str, max_tokens: int) -> dict:
        """The chat request for one image, given as a data URL."""
        return {"model": self.model, "temperature": 0.0, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": self.prompt}]}]}

    def chat(self, img, max_tokens: int) -> tuple[str, str, float]:
        import httpx

        buf = io.BytesIO()
        img.convert("RGB").save(buf, "PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        payload = self.payload(url, max_tokens)
        t0 = time.time()
        resp = httpx.post(f"{self.base_url}/v1/chat/completions", json=payload,
                          headers=self.headers, timeout=900)
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        return (choice["message"]["content"] or "").strip(), choice["finish_reason"], time.time() - t0

    def recognize(self, image_path: Path) -> tuple[str, str, float]:
        import httpx

        t0 = time.time()
        with open(image_path, "rb") as fh:
            resp = httpx.post(f"{self.base_url}/recognize", headers=self.headers, timeout=900,
                              files={"image": (image_path.name, fh, "image/jpeg")},
                              data={"model": self.model})
        resp.raise_for_status()
        body = resp.json()
        return (body.get("text") or "").strip(), body.get("finish_reason") or "n/a", time.time() - t0




def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", type=Path, help="a job dir with data/<jsonl> and data/pages/")
    source.add_argument("--page-set", type=Path,
                        help="a frozen held-out draw, eval_sets/<set>/vlm-pages-<side>/")
    ap.add_argument("--jsonl", default="val_heldout.jsonl", help="with --root only")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-url", required=True,
                    help="a vLLM of your own (normal), or the gateway only when it is idle")
    ap.add_argument("--no-auth", action="store_true", help="no X-API-Key (a bare vLLM)")
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--block-lines", type=int, default=DEFAULT_BLOCK_LINES,
                    help="lines per block; must match the model's block_lines")
    ap.add_argument("--recognize", action="store_true",
                    help="also send each page through the gateway's /recognize")
    ap.add_argument("--prompt", default=VLM_PROMPT,
                    help="the model's own prompt (default: VLM_PROMPT, the Qwen line contract)")
    ap.add_argument("--line-pixels", type=int, default=LINE_PX,
                    help="budget of line and region@line_budget")
    ap.add_argument("--block-pixels", type=int, default=BLOCK_PX, help="budget of the blocks")
    ap.add_argument("--page-pixels", type=int, default=PAGE_PX,
                    help="budget of region@page_budget and page@page_budget")
    ap.add_argument("--max-tokens-page", type=int, default=PAGE_TOKENS,
                    help="generation cap of the region and page levels")
    return ap


def budgets_of(args: argparse.Namespace) -> Budgets:
    return Budgets(prompt=args.prompt, line_pixels=args.line_pixels,
                   block_pixels=args.block_pixels, page_pixels=args.page_pixels,
                   max_tokens_page=args.max_tokens_page)


def main() -> int:
    args = build_parser().parse_args()
    budgets = budgets_of(args)
    levels = budgets.levels()

    from PIL import Image

    client = Client(args.base_url, args.model, None if args.no_auth else os.environ.get("ATR_API_KEY"),
                    prompt=budgets.prompt)
    page_set = None
    if args.page_set:
        root = args.page_set
        pages, page_set = read_page_set(root)
        pages = spread_pages(pages, args.max_pages)
        rows = []
        for p in pages:
            with Image.open((root / p).with_suffix(".jpg")) as im:
                rows.extend(page_lines(root, p, im.size))
    else:
        root = args.root
        with open(root / "data" / args.jsonl, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        pages = spread_pages(sorted({r["page"] for r in rows}), args.max_pages)
        rows = [r for r in rows if r["page"] in set(pages)]
    results: dict[str, list[dict]] = {k: [] for k in LEVELS}
    if args.recognize:
        results["page@recognize"] = []

    px, tok = levels["line"]
    opened = (None, None)
    for r in rows:
        if opened[0] != r["image"]:      # a page set's lines come page by page
            opened = (r["image"], Image.open(root / r["image"]))
        img = opened[1].crop(tuple(r["bbox"])) if r.get("bbox") else opened[1]
        hyp, fin, sec = client.chat(fit(img, px), tok)
        results["line"].append({"page": r["page"], "ref": r["text"], "hyp": hyp, "finish": fin, "sec": sec})
    print(f"lines: {len(rows)}", flush=True)

    for p in pages:
        xml = root / p
        img = Image.open(xml.with_suffix(".jpg"))
        img.load()
        xml_text = xml.read_text(encoding="utf-8")
        px, tok = levels["block@block_budget"]
        for bid, nlines, box, ref in block_boxes(xml_text, img.size, args.block_lines):
            hyp, fin, sec = client.chat(fit(img.crop(box), px), tok)
            results["block@block_budget"].append({"page": p, "block": bid, "lines": nlines,
                                                  "size": (box[2] - box[0], box[3] - box[1]),
                                                  "ref": ref, "hyp": hyp, "finish": fin, "sec": sec})
        for rid, nlines, box, ref in region_boxes(xml_text, img.size):
            crop = img.crop(box)
            for level in ("region@page_budget", "region@line_budget"):
                px, tok = levels[level]
                hyp, fin, sec = client.chat(fit(crop, px), tok)
                results[level].append({"page": p, "region": rid, "lines": nlines, "size": crop.size,
                                       "ref": ref, "hyp": hyp, "finish": fin, "sec": sec})
        ref = page_sample(xml).text
        px, tok = levels["page@page_budget"]
        hyp, fin, sec = client.chat(fit(img, px), tok)
        results["page@page_budget"].append({"page": p, "size": img.size, "ref": ref, "hyp": hyp,
                                            "finish": fin, "sec": sec})
        if args.recognize:
            hyp, fin, sec = client.recognize(xml.with_suffix(".jpg"))
            results["page@recognize"].append({"page": p, "ref": ref, "hyp": hyp, "finish": fin, "sec": sec})
        print(f"page {p}", flush=True)

    summary = [summarise(k, v) for k, v in results.items()]
    report = {"model": args.model, "jsonl": None if page_set else args.jsonl,
              "page_set": page_set and {"path": str(args.page_set), **page_set},
              **budgets.record(), "pages": pages, "summary": summary,
              "block_lines": args.block_lines,
              "regions_by_line_count": {k: by_line_count(results[k])
                                        for k in ("block@block_budget", "region@page_budget",
                                                  "region@line_budget")},
              "items": results}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for s in summary:
        print(json.dumps(s, ensure_ascii=False))
    print(json.dumps(report["regions_by_line_count"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""scripts/eval_granularity.py: the pure parts — regions, the summary, the page draw."""

from __future__ import annotations

import hashlib
import json

import pytest

from atr_training.vlm_dataset import page_sample, write_jsonl
from scripts.eval_granularity import (
    Budgets,
    Client,
    PageSetError,
    block_boxes,
    budgets_of,
    build_parser,
    by_line_count,
    flat,
    line_region_ids,
    page_lines,
    read_page_set,
    region_boxes,
    spread_pages,
    summarise,
)

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"


def _line(lid: str, x0: int, y0: int, x1: int, y1: int, text: str, words: str = "") -> str:
    return (f'<TextLine id="{lid}"><Coords points="{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"/>'
            f"{words}<TextEquiv><Unicode>{text}</Unicode></TextEquiv></TextLine>")


def _page(*regions: str) -> str:
    return f'<PcGts xmlns="{NS}"><Page imageFilename="p.jpg" imageWidth="1000" imageHeight="800">' \
           + "".join(regions) + "</Page></PcGts>"


def _region(rid: str, *lines: str) -> str:
    return f'<TextRegion id="{rid}"><Coords points="0,0 1000,0 1000,800 0,800"/>' + "".join(lines) + "</TextRegion>"


def test_region_box_is_the_union_of_its_lines_padded_and_clamped():
    xml = _page(_region("r1", _line("l1", 100, 100, 500, 140, "eins"), _line("l2", 90, 150, 520, 190, "zwei")),
                _region("r2", _line("l3", 5, 780, 990, 799, "unten")))
    got = list(region_boxes(xml, (1000, 800), pad=12))
    assert got[0] == ("r1", 2, (78, 88, 532, 202), "eins\nzwei")
    assert got[1] == ("r2", 1, (0, 768, 1000, 800), "unten")        # clamped to the page


def test_region_text_is_the_lines_own_text_not_the_first_word():
    words = "<Word><TextEquiv><Unicode>der</Unicode></TextEquiv></Word>"
    xml = _page(_region("r1", _line("l1", 0, 0, 100, 20, "der gantze satz", words)))
    assert next(iter(region_boxes(xml, (1000, 800))))[3] == "der gantze satz"    # #125


def test_region_without_transcribed_lines_is_skipped():
    xml = _page(_region("empty"), _region("r1", _line("l1", 0, 0, 100, 20, "text")))
    assert [r[0] for r in region_boxes(xml, (1000, 800))] == ["r1"]


def test_summary_names_collapse_and_runaway():
    items = [
        {"ref": "eine ganze Zeile", "hyp": "eine ganze Zeile", "finish": "stop", "sec": 1.0},
        {"ref": "eine ganze Seite\nmit zwei Zeilen", "hyp": "de", "finish": "stop", "sec": 1.0},
        {"ref": "kurz", "hyp": "und er und er und er", "finish": "length", "sec": 2.0},
    ]
    s = summarise("page", items)
    assert (s["n"], s["collapsed<1/3"], s["runaway>1.5x"]) == (3, 1, 1)
    assert s["finish"] == {"stop": 2, "length": 1}


def test_line_breaks_are_not_errors():
    s = summarise("region", [{"ref": "a b\nc d", "hyp": "a b c d", "finish": "stop", "sec": 0}])
    assert s["cer"] == 0.0 and flat("a\n  b") == "a b"


def test_by_line_count_buckets():
    items = [{"lines": 1, "ref": "abcd", "hyp": "abcd"}, {"lines": 12, "ref": "x" * 100, "hyp": "x"}]
    got = by_line_count(items)
    assert got["1"]["ratio"] == 1.0 and got[">10"]["ratio"] == 0.01


def test_spread_pages_takes_every_source_before_repeating_one():
    pages = [f"data/pages/{s}_{i}.xml" for s in ("aaa", "bbb", "ccc") for i in range(5)]
    got = spread_pages(pages, 4)
    assert len(got) == 4 and {p.split("/")[-1].split("_")[0] for p in got} == {"aaa", "bbb", "ccc"}
    assert spread_pages(pages, None) == pages


# ── block level (training-atr-models#57) ────────────────────────────────────
def _block_page(*regions: str) -> str:
    return _page(*regions)


def test_line_region_ids_cover_every_textline_transcribed_or_not():
    xml = _page(_region("r1", _line("l1", 0, 0, 100, 20, "eins"), _line("l2", 0, 30, 100, 50, "")),
                _region("r2", _line("l3", 0, 60, 100, 80, "drei")))
    assert line_region_ids(xml) == ["r1", "r1", "r2"]


def test_blocks_are_runs_of_consecutive_lines_within_one_region():
    lines = [_line(f"l{i}", 0, i * 30, 100, i * 30 + 20, f"z{i}") for i in range(5)]
    xml = _page(_region("r1", *lines[:4]), _region("r2", lines[4]))
    got = list(block_boxes(xml, (1000, 800), block_lines=3, pad=0))
    assert [(n, text) for _, n, _, text in got] == [
        (3, "z0\nz1\nz2"), (1, "z3"), (1, "z4")]
    assert got[0][2] == (0, 0, 100, 80)                  # union of the first three


def test_a_line_without_transcription_breaks_the_run():
    """It is on the image; a target without it would teach the model to skip text."""
    xml = _page(_region("r1",
                        _line("l1", 0, 0, 100, 20, "eins"),
                        _line("l2", 0, 30, 100, 50, ""),
                        _line("l3", 0, 60, 100, 80, "drei")))
    assert [t for _, _, _, t in block_boxes(xml, (1000, 800), block_lines=6)] == ["eins", "drei"]


def test_one_line_blocks_are_the_lines():
    xml = _page(_region("r1", _line("l1", 0, 0, 100, 20, "eins"), _line("l2", 0, 30, 100, 50, "zwei")))
    assert [t for _, _, _, t in block_boxes(xml, (1000, 800), block_lines=1)] == ["eins", "zwei"]


# ── prompt and budgets (#376) ───────────────────────────────────────────────
#: What the script sent before the flags existed, copied from its last version
#: rather than from the constants that now feed it — so a drift in either the
#: contract or the defaults fails here.
OLD_PROMPT = "Transcribe the handwritten text in this image exactly as written."
OLD_LEVELS = {
    "line": (256 * 32 * 32, 512),
    "block@block_budget": (1024 * 32 * 32, 1024),
    "region@page_budget": (2048 * 32 * 32, 4096),
    "region@line_budget": (256 * 32 * 32, 4096),
    "page@page_budget": (2048 * 32 * 32, 4096),
}
REQUIRED = ["--model", "m", "--out", "r.json", "--base-url", "http://127.0.0.1:8299"]


def test_without_the_flags_every_level_asks_what_it_asked_before():
    budgets = budgets_of(build_parser().parse_args(["--root", "job", *REQUIRED]))
    assert budgets == Budgets()
    assert budgets.prompt == OLD_PROMPT
    assert budgets.levels() == OLD_LEVELS


def test_without_the_flags_the_request_is_byte_identical():
    url = "data:image/png;base64,AAAA"
    old = {"model": "m", "temperature": 0.0, "max_tokens": 512,
           "messages": [{"role": "user", "content": [
               {"type": "image_url", "image_url": {"url": url}},
               {"type": "text", "text": OLD_PROMPT}]}]}
    new = Client("http://127.0.0.1:8299/", "m", None).payload(url, 512)
    assert json.dumps(new).encode() == json.dumps(old).encode()


def test_flags_set_the_prompt_and_each_budget():
    args = build_parser().parse_args(["--page-set", "set", *REQUIRED, "--prompt", "Lies.\nGanz.",
                                      "--line-pixels", "1", "--block-pixels", "2",
                                      "--page-pixels", "4194304", "--max-tokens-page", "3072"])
    budgets = budgets_of(args)
    assert budgets.levels() == {"line": (1, 512), "block@block_budget": (2, 1024),
                                "region@page_budget": (4194304, 3072),
                                "region@line_budget": (1, 3072),
                                "page@page_budget": (4194304, 3072)}
    assert Client("http://x", "m", None, prompt=budgets.prompt).payload("u", 1)[
        "messages"][0]["content"][1]["text"] == "Lies.\nGanz."
    record = budgets.record()
    assert record["prompt"] == "Lies.\nGanz."
    assert record["budgets"]["page@page_budget"] == {"max_pixels": 4194304, "max_tokens": 3072}


def test_root_and_page_set_exclude_each_other():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--root", "a", "--page-set", "b", *REQUIRED])
    with pytest.raises(SystemExit):
        build_parser().parse_args(REQUIRED)


# ── frozen page sets (#376) ─────────────────────────────────────────────────
def _page_set(tmp_path, xml: str):
    """A draw laid out as scripts/build_page_benchmark.py writes one."""
    from PIL import Image

    pages = tmp_path / "data" / "pages"
    pages.mkdir(parents=True)
    (pages / "aaa_1.xml").write_text(xml, encoding="utf-8")
    Image.new("RGB", (1000, 800), "white").save(pages / "aaa_1.jpg")
    write_jsonl(tmp_path / "pages.jsonl", [page_sample(pages / "aaa_1.xml", root=tmp_path)])
    record = {"set": "toy-v1", "side": "test", "documents": 1, "pages": 1,
              "page_digest": "0123456789abcdef", "samples": 1, "draw": "pages.jsonl",
              "draw_md5": hashlib.md5((tmp_path / "pages.jsonl").read_bytes()).hexdigest()}
    (tmp_path / "benchmark.json").write_text(json.dumps(record), encoding="utf-8")
    return tmp_path


def test_page_set_reader_names_the_pages_and_the_set(tmp_path):
    root = _page_set(tmp_path, _page(_region("r1", _line("l1", 100, 100, 500, 140, "eins"))))
    pages, identity = read_page_set(root)
    assert pages == ["data/pages/aaa_1.xml"]
    assert identity["set"] == "toy-v1" and identity["side"] == "test"
    assert identity["page_digest"] == "0123456789abcdef"


def test_page_set_reader_refuses_an_edited_draw(tmp_path):
    root = _page_set(tmp_path, _page(_region("r1", _line("l1", 100, 100, 500, 140, "eins"))))
    with open(root / "pages.jsonl", "a", encoding="utf-8") as fh:
        fh.write("\n")
    with pytest.raises(PageSetError, match="not the frozen draw"):
        read_page_set(root)


def test_page_set_reader_refuses_a_directory_without_a_record(tmp_path):
    with pytest.raises(PageSetError, match="benchmark.json"):
        read_page_set(tmp_path)


def test_lines_are_cut_from_the_pagexml_as_training_cuts_them(tmp_path):
    xml = _page(_region("r1", _line("l1", 100, 100, 500, 140, "eins"),
                        _line("l2", 0, 0, 0, 0, "zu klein"),          # no legible crop
                        _line("l3", 2, 780, 990, 799, "unten")))
    root = _page_set(tmp_path, xml)
    rows = page_lines(root, "data/pages/aaa_1.xml", (1000, 800))
    assert [r["text"] for r in rows] == ["eins", "unten"]
    assert rows[0] == {"image": "data/pages/aaa_1.jpg", "bbox": [92, 92, 508, 148],
                       "text": "eins", "page": "data/pages/aaa_1.xml"}
    assert rows[1]["bbox"] == [0, 772, 998, 800]                   # padded by 8, clamped


def test_line_and_page_references_come_from_the_same_page(tmp_path):
    xml = _page(_region("r1", _line("l1", 100, 100, 500, 140, "eins"),
                        _line("l2", 90, 150, 520, 190, "zwei")))
    root = _page_set(tmp_path, xml)
    rows = page_lines(root, "data/pages/aaa_1.xml", (1000, 800))
    assert "\n".join(r["text"] for r in rows) == page_sample(root / "data/pages/aaa_1.xml").text


# ── off the image, and a report that survives a crash (asteraix, 21.11.2026) ──

def test_a_region_or_block_off_the_image_has_no_box():
    """The PageXML put a region's lines beyond the image edge; clamping each side
    from one direction only gave right < left, and PIL ended a 17-hour run."""
    xml = _page(_region("r1", _line("l1", 100, 100, 500, 140, "eins")),
                _region("off", _line("l2", 1200, 100, 1400, 140, "draussen")))
    regions = {rid: box for rid, _, box, _ in region_boxes(xml, (1000, 800), pad=12)}
    assert regions == {"r1": (88, 88, 512, 152), "off": None}
    blocks = [box for _, _, box, _ in block_boxes(xml, (1000, 800), 6, pad=12)]
    assert blocks == [(88, 88, 512, 152), None]


def test_clamp_box_keeps_what_is_on_the_image():
    from scripts.eval_granularity import clamp_box
    assert clamp_box(-5, -5, 50, 60, 100, 100) == (0, 0, 50, 60)
    assert clamp_box(90, 90, 150, 150, 100, 100) == (90, 90, 100, 100)
    assert clamp_box(120, 10, 150, 20, 100, 100) is None
    assert clamp_box(10, 10, 10, 20, 100, 100) is None


def test_an_empty_level_summarises_to_nothing_measured():
    s = summarise("page@page_budget", [])
    assert s["n"] == 0 and s["cer"] is None and s["length_ratio"] is None


class _FakeClient:
    """Answers every crop with the same word; `fail_at` raises on that call."""

    calls = 0
    fail_at: int | None = None
    on_call = None

    def __init__(self, *a, **kw):
        pass

    def chat(self, img, max_tokens):
        type(self).calls += 1
        if self.on_call:
            self.on_call(type(self).calls)
        if self.fail_at and type(self).calls >= self.fail_at:
            raise RuntimeError("the server went away")
        return "eins", "stop", 0.0


def _run_main(monkeypatch, root, out, client, *extra):
    import sys

    import scripts.eval_granularity as eg
    client.calls = 0
    monkeypatch.setattr(eg, "Client", client)
    monkeypatch.setattr(sys, "argv", ["eval_granularity.py", "--page-set", str(root), "--model", "m",
                                      "--out", str(out), "--base-url", "http://x", "--no-auth", *extra])
    return eg.main()


def test_a_crop_off_the_image_is_skipped_and_said(tmp_path, monkeypatch):
    xml = _page(_region("r1", _line("l1", 100, 100, 500, 140, "eins")),
                _region("off", _line("l2", 1200, 100, 1400, 140, "draussen")))
    root = _page_set(tmp_path / "set", xml)
    out = tmp_path / "report.json"

    class Client(_FakeClient):
        fail_at = None
    assert _run_main(monkeypatch, root, out, Client) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["complete"] is True and report["error"] is None
    assert {(s["level"], s["id"]) for s in report["skipped"]} == {("region", "off"), ("block", 1)}
    assert len(report["items"]["region@page_budget"]) == 1
    assert not (tmp_path / "report.json.part").exists()


def test_a_crash_leaves_a_partial_report_with_its_error(tmp_path, monkeypatch):
    xml = _page(_region("r1", _line("l1", 100, 100, 500, 140, "eins"),
                        _line("l2", 90, 150, 520, 190, "zwei")))
    root = _page_set(tmp_path / "set", xml)
    out = tmp_path / "report.json"

    class Client(_FakeClient):
        fail_at = 3                     # two lines read, the first block fails
    with pytest.raises(RuntimeError, match="went away"):
        _run_main(monkeypatch, root, out, Client)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["complete"] is False
    assert report["error"] == "RuntimeError: the server went away"
    assert len(report["items"]["line"]) == 2 and report["pages_done"] == 0
    line = next(s for s in report["summary"] if s["level"] == "line")
    assert line["n"] == 2 and line["cer"] == 0.5          # "zwei" read as "eins"


def test_the_lines_are_on_disk_before_the_first_block_is_asked(tmp_path, monkeypatch):
    """A SIGKILL runs no except clause: the checkpoint after the line level is
    what a killed run leaves."""
    xml = _page(_region("r1", _line("l1", 100, 100, 500, 140, "eins")))
    root = _page_set(tmp_path / "set", xml)
    out = tmp_path / "report.json"
    seen = {}

    class Client(_FakeClient):
        fail_at = None

        def on_call(self, n):
            if n == 2:                  # the first block: the lines must be written
                seen.update(json.loads(out.read_text(encoding="utf-8")))
    assert _run_main(monkeypatch, root, out, Client) == 0
    assert seen["complete"] is False and len(seen["items"]["line"]) == 1


# ── the usable line: a reading above 15 % CER is useless (26.11.2026) ───────

from scripts.eval_granularity import USABLE_CER, page_interval, verdict  # noqa: E402


def _items(per_page: dict[str, list[tuple[str, str]]]) -> list[dict]:
    return [{"page": p, "ref": r, "hyp": h, "finish": "stop", "sec": 0}
            for p, pairs in per_page.items() for r, h in pairs]


@pytest.mark.parametrize("cer, ci, want", [
    (0.388, [0.36, 0.42], "useless"),        # page-v1, regions at the page budget
    (0.10, [0.08, 0.13], "usable"),
    (0.15, [0.15, 0.15], "usable"),
    (0.14, [0.10, 0.19], "undecided"),
    (0.16, None, "useless"),
    (0.15, None, "usable"),                  # one page: the point decides
    (None, None, "nothing measured"),
])
def test_the_verdict_asks_the_whole_interval(cer, ci, want):
    assert USABLE_CER == 0.15
    assert verdict(cer, ci) == want


def test_a_level_says_its_verdict_and_how_many_items_are_usable():
    items = _items({"p1": [("abcdefghij", "abcdefghij"), ("abcdefghij", "abcdefghiX")],
                    "p2": [("abcdefghij", "abcdefgXXX"), ("abcdefghijklmnopqrst", "abcdefghijklmnopqXXX")]})
    s = summarise("line", items)
    assert s["usable_items"] == 3 and s["usable_share"] == 0.75   # 0.0, 0.1, 0.15 — not 0.3
    assert s["cer_ci_pages"] is not None and s["verdict"] in ("usable", "undecided", "useless")


def test_the_interval_resamples_pages_and_is_reproducible():
    good = {f"g{i}": [("abcdefghij", "abcdefghij")] for i in range(5)}
    bad = {f"b{i}": [("abcdefghij", "XXXXXXXXXX")] for i in range(5)}
    items = _items(good | bad)
    lo, hi = page_interval(items)
    assert lo < 0.5 < hi and page_interval(items) == [lo, hi]
    assert page_interval(_items({"only": [("ab", "ab")]})) is None


def test_a_useless_level_is_called_useless():
    items = _items({f"p{i}": [("abcdefghij", "abcdeXXXXX")] for i in range(4)})   # 0.5 everywhere
    s = summarise("page@page_budget", items)
    assert s["cer"] == 0.5 and s["cer_ci_pages"] == [0.5, 0.5] and s["verdict"] == "useless"
    assert s["usable_items"] == 0 and s["usable_share"] == 0.0


def test_an_empty_level_has_no_verdict():
    s = summarise("block@block_budget", [])
    assert s["verdict"] == "nothing measured" and s["usable_share"] is None and s["cer_ci_pages"] is None


def test_the_summary_uses_the_page_interval_not_the_point():
    """Three clean pages and one bad one: the corpus CER is under the line,
    the pages allow far more — undecided, not usable."""
    items = _items({"a": [("abcdefghij", "abcdefghij")], "b": [("abcdefghij", "abcdefghij")],
                    "c": [("abcdefghij", "abcdefghij")], "d": [("abcdefghij", "abcdeXXXXX")]})
    s = summarise("page@page_budget", items)
    assert s["cer"] == 0.125 and s["cer_ci_pages"][1] > 0.15
    assert s["verdict"] == "undecided"

"""Sample building, PageXML geometry and the corpus-level metrics.

All pure logic, so it runs in the repo venv with no torch, no GPU and no images
beyond the tiny JPEGs the tests write themselves.
"""

import collections
import json

import pytest

from atr_training.contracts import (
    DEFAULT_GRANULARITY_MIX,
    VLM_MAX_NEW_TOKENS,
    VLM_MAX_SAMPLE_CHARS,
    VLM_MAX_SEQ_LEN,
    VLM_PIXEL_BUDGET,
    VlmTrainParams,
)
from atr_training.pagexml import line_boxes, line_regions, parse_points
from atr_training.textmetrics import cer, score_pairs, wer
from atr_training.vlm_dataset import (
    Sample,
    VlmDatasetError,
    block_samples,
    chat_example,
    fit_pixels,
    mix_samples,
    drop_long_samples,
    drop_short_samples,
    line_samples,
    page_sample,
    read_jsonl,
    samples_for,
    write_jsonl,
)

PAGE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="000001_p.jpg" imageWidth="1600" imageHeight="1067">
    <TextRegion id="r1">
      <TextLine id="l1">
        <Coords points="10,20 800,20 800,90 10,90"/>
        <TextEquiv><Unicode>Item ontfaen van Janne</Unicode></TextEquiv></TextLine>
      <TextLine id="l2">
        <Coords points="12,100 790,100 790,170 12,170"/>
        <TextEquiv><Unicode>van der Straten</Unicode></TextEquiv></TextLine>
      <TextLine id="l3">
        <Coords points="12,200 790,200 790,270 12,270"/>
        <TextEquiv><Unicode>  </Unicode></TextEquiv></TextLine>
    </TextRegion>
  </Page>
</PcGts>
"""

BASELINE_ONLY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="000002_p.jpg" imageWidth="1600" imageHeight="1067">
    <TextLine id="b1">
      <Baseline points="30,400 900,400"/>
      <TextEquiv><Unicode>eine Zeile ohne Coords</Unicode></TextEquiv></TextLine>
  </Page>
</PcGts>
"""


@pytest.fixture
def page(tmp_path):
    """A materialized page: <stem>.xml next to <stem>.jpg, as prepare writes it."""
    xml = tmp_path / "pages" / "000001_p.xml"
    xml.parent.mkdir(parents=True)
    xml.write_text(PAGE_XML, encoding="utf-8")
    (xml.with_suffix(".jpg")).write_bytes(b"\xff\xd8notreallyajpeg")
    return xml


# ── PageXML geometry ────────────────────────────────────────────────────────
def test_parse_points_skips_malformed_pairs():
    assert parse_points("10,20 oops 30,40") == [(10, 20), (30, 40)]
    assert parse_points("") == []


def test_line_boxes_returns_transcribed_lines_only():
    boxes = line_boxes(PAGE_XML)
    assert [b.text for b in boxes] == ["Item ontfaen van Janne", "van der Straten"]
    assert (boxes[0].left, boxes[0].top, boxes[0].right, boxes[0].bottom) == (10, 20, 800, 90)
    assert boxes[0].line_id == "l1"


def test_a_baseline_only_line_still_gets_a_box():
    """Baseline-only exports are real; a flat polyline has no height of its own,
    so one is derived rather than dropping the line."""
    boxes = line_boxes(BASELINE_ONLY_XML)
    assert len(boxes) == 1
    assert boxes[0].height > 0
    assert boxes[0].bottom == 400


def test_padding_is_clamped_to_the_page():
    box = line_boxes(PAGE_XML)[0]
    padded = box.padded(50, width=1600, height=1067)
    assert (padded.left, padded.top) == (0, 0)  # clamped, not negative
    assert padded.right == 850


# ── samples ─────────────────────────────────────────────────────────────────
def test_page_sample_joins_every_line(page, tmp_path):
    sample = page_sample(page, root=tmp_path)
    assert sample.source_type == "page"
    assert sample.text == "Item ontfaen van Janne\nvan der Straten"
    assert sample.bbox is None
    assert sample.image == "pages/000001_p.jpg"  # relative to the job root


def test_line_samples_carry_one_box_each(page, tmp_path):
    samples = line_samples(page, root=tmp_path, pad=0)
    assert [s.text for s in samples] == ["Item ontfaen van Janne", "van der Straten"]
    assert all(s.source_type == "line" for s in samples)
    assert samples[0].bbox == [10, 20, 800, 90]
    assert all(s.page == "pages/000001_p.xml" for s in samples)


def test_a_page_without_its_jpeg_is_an_error(tmp_path):
    """prepare writes the pair together, so one without the other means the page
    directory was touched afterwards — better said than silently skipped."""
    xml = tmp_path / "orphan.xml"
    xml.write_text(PAGE_XML, encoding="utf-8")
    with pytest.raises(VlmDatasetError, match="no sibling"):
        line_samples(xml)


def test_an_untranscribed_page_yields_no_page_sample(tmp_path):
    xml = tmp_path / "empty.xml"
    xml.write_text(PAGE_XML.replace("Item ontfaen van Janne", "")
                   .replace("van der Straten", ""), encoding="utf-8")
    xml.with_suffix(".jpg").write_bytes(b"\xff\xd8")
    assert page_sample(xml) is None
    assert line_samples(xml) == []


def test_samples_for_rejects_an_unknown_granularity(page):
    with pytest.raises(VlmDatasetError, match="granularity"):
        samples_for([page], "paragraph")


def test_samples_for_dispatches_on_granularity(page, tmp_path):
    assert len(samples_for([page], "page", root=tmp_path)) == 1
    assert len(samples_for([page], "line", root=tmp_path)) == 2
    assert len(samples_for([page], "block", root=tmp_path)) == 1
    assert set(VLM_PIXEL_BUDGET) == {"line", "block", "page"}


# ── granularity: block (#57) ────────────────────────────────────────────────
def _line(lid: str, top: int, text: str | None, left: int = 10, right: int = 800) -> str:
    equiv = f"<TextEquiv><Unicode>{text}</Unicode></TextEquiv>" if text is not None else ""
    return (f'<TextLine id="{lid}"><Coords points="{left},{top} {right},{top} '
            f'{right},{top + 60} {left},{top + 60}"/>{equiv}</TextLine>')


BLOCK_XML = f"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="000003_p.jpg" imageWidth="1600" imageHeight="2000">
    <TextRegion id="r1">
      {_line("a1", 100, "eins")}{_line("a2", 170, "zwei")}{_line("a3", 240, "drei")}
      {_line("a4", 310, "vier")}{_line("a5", 380, None)}{_line("a6", 450, "sechs")}
      {_line("a7", 520, "sieben")}{_line("a8", 590, "acht")}
    </TextRegion>
    <TextRegion id="r2">
      {_line("b1", 900, "neun")}{_line("b2", 970, "zehn")}
    </TextRegion>
    {_line("c1", 1500, "draussen")}
  </Page>
</PcGts>
"""


@pytest.fixture
def block_page(tmp_path):
    xml = tmp_path / "pages" / "000003_p.xml"
    xml.parent.mkdir(parents=True)
    xml.write_text(BLOCK_XML, encoding="utf-8")
    xml.with_suffix(".jpg").write_bytes(b"\xff\xd8notreallyajpeg")
    return xml


def test_line_regions_are_aligned_with_every_textline():
    """Untranscribed lines count too — the list is indexed by TextLineBox.index."""
    assert line_regions(BLOCK_XML) == ["r1"] * 8 + ["r2", "r2", None]
    assert [b.index for b in line_boxes(BLOCK_XML)] == [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]


def test_anonymous_regions_do_not_merge():
    xml = BLOCK_XML.replace('<TextRegion id="r1">', "<TextRegion>").replace(
        '<TextRegion id="r2">', "<TextRegion>")
    regions = line_regions(xml)
    assert regions[0] != regions[8] and regions[0] is not None


def test_blocks_stay_in_their_region_and_break_at_an_untranscribed_line(block_page, tmp_path):
    """a5 has no transcription but is on the image, so no block may span it."""
    got = block_samples(block_page, root=tmp_path, pad=0, block_lines=3)
    assert [s.text for s in got] == [
        "eins\nzwei\ndrei", "vier",            # run a1-a4, cut at 3
        "sechs\nsieben\nacht",                 # run a6-a8
        "neun\nzehn",                           # region r2
        "draussen",                            # outside any region
    ]
    assert all(s.source_type == "block" and s.page == "pages/000003_p.xml" for s in got)


def test_a_block_box_is_the_union_of_its_lines(block_page):
    first = block_samples(block_page, pad=0, block_lines=3)[0]
    assert first.bbox == [10, 100, 800, 300]            # a1 top .. a3 bottom


def test_an_implausible_line_breaks_the_block_too(tmp_path):
    xml = tmp_path / "wide.xml"
    xml.write_text(BLOCK_XML.replace(_line("a2", 170, "zwei"),
                                     _line("a2", 170, "zwei", right=9000)), encoding="utf-8")
    xml.with_suffix(".jpg").write_bytes(b"\xff\xd8")
    texts = [s.text for s in block_samples(xml, pad=0, block_lines=6)]
    assert "zwei" not in "\n".join(texts)
    assert texts[0] == "eins" and texts[1] == "drei\nvier"


def test_one_line_blocks_are_the_line_samples(block_page, tmp_path):
    blocks = block_samples(block_page, root=tmp_path, block_lines=1)
    lines = line_samples(block_page, root=tmp_path)
    assert [(b.text, b.bbox) for b in blocks] == [(s.text, s.bbox) for s in lines]


def test_block_lines_below_one_is_refused(block_page):
    with pytest.raises(VlmDatasetError, match="block_lines"):
        block_samples(block_page, block_lines=0)


def test_samples_for_passes_block_lines_on(block_page, tmp_path):
    assert len(samples_for([block_page], "block", root=tmp_path, block_lines=2)) == 6   # 2+2+1+1
    assert len(samples_for([block_page], "block", root=tmp_path, block_lines=16)) == 4


def test_block_has_a_budget_in_every_table():
    for table in (VLM_PIXEL_BUDGET, VLM_MAX_SEQ_LEN, VLM_MAX_NEW_TOKENS, VLM_MAX_SAMPLE_CHARS):
        assert table["line"] < table["block"] < table["page"]


def test_block_params():
    params = VlmTrainParams(granularity="block", block_lines=4)
    assert params.pixel_budget() == VLM_PIXEL_BUDGET["block"]
    assert params.generation_budget() == VLM_MAX_NEW_TOKENS["block"]
    with pytest.raises(ValueError):
        VlmTrainParams(granularity="block", block_lines=0)
    with pytest.raises(ValueError):
        VlmTrainParams(granularity="block", block_lines=17)


# ── jsonl round trip ────────────────────────────────────────────────────────
def test_jsonl_round_trip(tmp_path):
    samples = [Sample(image="a.jpg", text="ä ö ü", source_type="line", bbox=[1, 2, 3, 4]),
               Sample(image="b.jpg", text="zwei", source_type="line")]
    path = tmp_path / "train.jsonl"
    assert write_jsonl(path, samples) == 2
    assert list(read_jsonl(path)) == samples
    # non-ASCII stays readable in the file, so a sample set can be eyeballed
    assert "ä ö ü" in path.read_text(encoding="utf-8")


def test_a_malformed_jsonl_line_raises(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"image": "a.jpg", "text": "ok", "source_type": "line"}\nnot json\n',
                    encoding="utf-8")
    with pytest.raises(VlmDatasetError, match="not JSON"):
        list(read_jsonl(path))


def test_a_sample_without_text_raises(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"image": "a.jpg"}) + "\n", encoding="utf-8")
    with pytest.raises(VlmDatasetError, match="missing 'text'"):
        list(read_jsonl(path))


# ── the conversation ────────────────────────────────────────────────────────
def test_chat_example_without_text_is_the_inference_shape():
    turns = chat_example("Transcribe.")
    assert len(turns) == 1 and turns[0]["role"] == "user"
    assert turns[0]["content"][0] == {"type": "image"}


def test_chat_example_with_text_adds_the_assistant_turn():
    turns = chat_example("Transcribe.", "das Ergebnis")
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[1]["content"][0]["text"] == "das Ergebnis"


# ── metrics ─────────────────────────────────────────────────────────────────
def test_cer_and_wer_are_edit_distances():
    assert cer("abc", "abc") == 0.0
    assert cer("abd", "abc") == pytest.approx(1 / 3)
    assert wer("ein zwei", "ein drei") == pytest.approx(0.5)


def test_score_pairs_is_corpus_level_not_a_mean_of_rates():
    """One wrong character in a 3-char line and none in a 97-char line is a CER
    of 1 %, not of 16.7 % — which is what averaging the two rates would say."""
    long_ref = "x" * 97
    score = score_pairs([("abd", "abc"), (long_ref, long_ref)])
    assert score.chars == 100 and score.errors == 1
    assert score.cer == pytest.approx(0.01)
    assert score.samples == 2


def test_an_empty_reference_does_not_divide():
    score = score_pairs([("", ""), ("abc", "abc")])
    assert score.chars == 3 and score.errors == 0
    assert score.cer == 0.0
    assert score.samples == 2


def test_score_with_no_characters_reports_no_rate():
    """A CER of None is what makes the job fail rather than complete at 0.0."""
    score = score_pairs([("", "")])
    assert score.cer is None and score.wer is None
    assert score.as_report()["cer"] is None


# ── apply_visual_budget (#86) ───────────────────────────────────────────────
from atr_training.vlm_dataset import (  # noqa: E402
    SOFT_TOKEN_STEPS,
    VisualBudgetError,
    apply_visual_budget,
    has_stepped_budget,
)


class FakeImageProcessor:
    """A Qwen3-VL image processor: `size` in pixel *areas*, patch 16 x merge 2."""

    def __init__(self, size=None, max_pixels=None, patch_size=16, merge_size=2):
        if size is not None:
            self.size = size
        if max_pixels is not None:
            self.max_pixels = max_pixels
        if patch_size is not None:
            self.patch_size = patch_size
        if merge_size is not None:
            self.merge_size = merge_size


class FakeProcessor:
    def __init__(self, image_processor):
        if image_processor is not None:
            self.image_processor = image_processor


def qwen3() -> FakeProcessor:
    """What Qwen/Qwen3-VL-8B-Instruct's preprocessor_config.json actually declares."""
    return FakeProcessor(FakeImageProcessor(
        size={"longest_edge": 16777216, "shortest_edge": 65536}))


class TestApplyVisualBudget:
    """The budget was passed as `max_pixels=` and silently dropped.

    Job 20260814T192904Z-qwen3vl-german-medieval-v1 trained at the model default
    of 16384 visual tokens instead of 256, and died at step 2 of 774 when
    truncation cut a 600-token image out of a 512-token sequence.
    """

    def test_qwen3_is_bounded_through_size_not_max_pixels(self):
        processor = qwen3()
        applied = apply_visual_budget(processor, 256 * 32 * 32)
        assert applied.knob == "size.longest_edge"
        assert processor.image_processor.size["longest_edge"] == 256 * 32 * 32

    def test_the_default_line_budget_really_is_256_tokens_on_qwen3(self):
        """The old constant used 28² — Qwen2-VL's grid — and bought 196, not 256."""
        from atr_training.contracts import VLM_PIXEL_BUDGET

        applied = apply_visual_budget(qwen3(), VLM_PIXEL_BUDGET["line"])
        assert applied.cell_px == 32 and applied.grid_known
        assert applied.visual_tokens == 256

    def test_the_untouched_default_would_have_been_16384_tokens(self):
        """Why this matters: what the run was actually training at."""
        default = qwen3().image_processor.size["longest_edge"]
        assert default // (32 * 32) == 16384

    def test_shortest_edge_is_left_alone(self):
        processor = qwen3()
        apply_visual_budget(processor, 256 * 32 * 32)
        assert processor.image_processor.size["shortest_edge"] == 65536

    def test_a_qwen2_style_processor_still_works(self):
        """max_pixels is not wrong, just not Qwen3-VL's. Both are supported."""
        processor = FakeProcessor(FakeImageProcessor(
            max_pixels=1280 * 28 * 28, patch_size=14, merge_size=2))
        applied = apply_visual_budget(processor, 256 * 28 * 28)
        assert applied.knob == "max_pixels"
        assert applied.cell_px == 28 and applied.visual_tokens == 256

    def test_a_processor_with_both_knobs_gets_both(self):
        """CHURRO's processor, and the reason R4 measured nothing (#128).

        Qwen2.5-VL declares ``size={"longest_edge", "shortest_edge"}`` *and*
        ``max_pixels``, and ``smart_resize`` reads ``max_pixels``. Setting only
        the first knob left the budget at the model's default while the read-back
        confirmed the write: R4 asked for 2,097,152 pixels against R1's
        4,014,080 and the two runs produced byte-identical output.
        """
        processor = FakeProcessor(FakeImageProcessor(
            size={"longest_edge": 4014080, "shortest_edge": 401408},
            max_pixels=4014080, patch_size=14, merge_size=2))
        applied = apply_visual_budget(processor, 2097152)
        assert processor.image_processor.max_pixels == 2097152
        assert processor.image_processor.size["longest_edge"] == 2097152
        assert applied.knob == "size.longest_edge+max_pixels"

    def test_a_processor_with_no_knob_is_refused(self):
        processor = FakeProcessor(FakeImageProcessor(patch_size=16, merge_size=2))
        with pytest.raises(VisualBudgetError, match="no knob here"):
            apply_visual_budget(processor, 4096)

    def test_a_processor_with_no_image_processor_is_refused(self):
        with pytest.raises(VisualBudgetError, match="no image_processor"):
            apply_visual_budget(FakeProcessor(None), 4096)

    def test_a_budget_that_does_not_stick_is_refused(self):
        """The failure mode this whole helper exists for: it looked set, it wasn't."""
        class Stubborn(FakeImageProcessor):
            @property
            def size(self):
                return {"longest_edge": 16777216, "shortest_edge": 65536}

            @size.setter
            def size(self, value):
                pass                      # accepts, discards — as the kwarg did

        with pytest.raises(VisualBudgetError, match="did not take"):
            apply_visual_budget(FakeProcessor(Stubborn()), 4096)

    def test_an_unknown_grid_falls_back_and_says_so(self):
        processor = FakeProcessor(FakeImageProcessor(
            size={"longest_edge": 1}, patch_size=None, merge_size=None))
        applied = apply_visual_budget(processor, 256 * 32 * 32)
        assert applied.grid_known is False
        assert "ASSUMED" in str(applied)

    def test_str_is_readable_because_it_is_printed_into_the_job_log(self):
        assert str(apply_visual_budget(qwen3(), 256 * 32 * 32)) == (
            "size.longest_edge=262144 -> ~256 visual tokens (32px cell)")


# ── Gemma 4: a budget in soft tokens, on five steps ─────────────────────────
class FakeGemmaImageProcessor:
    """What google/gemma-4-31B-it's processor_config.json declares, and what a
    real ``AutoProcessor.from_pretrained`` reported on transformers 5.17.0:
    ``max_soft_tokens`` and no ``size``/``max_pixels`` at all."""

    def __init__(self, max_soft_tokens=280, patch_size=16, pooling_kernel_size=3):
        self.max_soft_tokens = max_soft_tokens
        self.image_processor_type = "Gemma4ImageProcessor"
        if patch_size is not None:
            self.patch_size = patch_size
        if pooling_kernel_size is not None:
            self.pooling_kernel_size = pooling_kernel_size


def gemma4() -> FakeProcessor:
    return FakeProcessor(FakeGemmaImageProcessor())


class TestSteppedVisualBudget:
    """Gemma 4 is not budget-less, it is budgeted differently (#86 follow-up).

    The five steps and the 48px cell are measured, not assumed: transformers
    5.17.0 raises ``ValueError: `max_soft_tokens` must be one of (70, 140, 280,
    560, 1120)`` for anything else, and at step 280 the patch count before
    pooling is 2520 = 280 x 3 x 3 with patch_size 16.
    """

    def test_a_request_lands_on_the_cheapest_step_that_carries_it(self):
        # 48px cell: a step of n tokens carries n * 48 * 48 pixels.
        for asked, step in ((262144, 140), (1048576, 560), (2097152, 1120)):
            applied = apply_visual_budget(gemma4(), asked)
            assert applied.visual_tokens == step, (asked, applied)
            assert applied.max_pixels == step * 48 * 48
            assert applied.max_pixels >= asked, "a step below the request starves it"

    def test_the_step_is_written_onto_the_processor(self):
        processor = gemma4()
        apply_visual_budget(processor, 262144)
        assert processor.image_processor.max_soft_tokens == 140

    def test_the_line_budget_does_not_land_on_the_default(self):
        """280 is the processor's default, so a line landing there would look
        applied and be untouched — the #86 failure in a new shape."""
        assert apply_visual_budget(gemma4(), VLM_PIXEL_BUDGET["line"]).visual_tokens != 280

    def test_a_request_above_every_step_takes_the_largest(self):
        applied = apply_visual_budget(gemma4(), 99_000_000)
        assert applied.visual_tokens == max(SOFT_TOKEN_STEPS)

    def test_every_chosen_step_is_one_the_processor_accepts(self):
        for asked in (1, 262144, 1048576, 2097152, 10**9):
            assert apply_visual_budget(gemma4(), asked).visual_tokens in SOFT_TOKEN_STEPS

    def test_it_says_what_was_asked_for_when_that_is_not_what_it_got(self):
        applied = apply_visual_budget(gemma4(), 262144)
        assert applied.stepped is True
        assert applied.asked_pixels == 262144
        assert str(applied) == (
            "max_soft_tokens=322560 -> ~140 visual tokens (48px cell) "
            "(asked for 262144, rounded up to a step)")

    def test_a_processor_that_will_not_say_its_grid_is_refused(self):
        """Without patch_size x pooling_kernel_size there is no way to know which
        step carries the pixels, and guessing is what #86 was about."""
        processor = FakeProcessor(FakeGemmaImageProcessor(pooling_kernel_size=None))
        with pytest.raises(VisualBudgetError, match="patch_size and pooling_kernel_size"):
            apply_visual_budget(processor, 262144)

    def test_a_step_that_does_not_stick_is_refused(self):
        class Stubborn(FakeGemmaImageProcessor):
            @property
            def max_soft_tokens(self):
                return 280

            @max_soft_tokens.setter
            def max_soft_tokens(self, value):
                pass

        with pytest.raises(VisualBudgetError, match="did not take"):
            apply_visual_budget(FakeProcessor(Stubborn()), 262144)

    def test_the_predicate_separates_the_two_families(self):
        """``evaluate_qlora`` has a processor and no budget object; it asks this."""
        assert has_stepped_budget(gemma4()) is True
        assert has_stepped_budget(qwen3()) is False
        assert has_stepped_budget(None) is False

    def test_qwen_is_not_stepped_and_still_reports_the_budget_it_was_given(self):
        applied = apply_visual_budget(qwen3(), VLM_PIXEL_BUDGET["line"])
        assert applied.stepped is False
        assert applied.asked_pixels is None

# ── dropping samples too long to afford (#110) ──────────────────────────────
def _sample(chars: int, name: str = "p") -> Sample:
    return Sample(image=f"{name}.jpg", text="x" * chars, source_type="page")


def test_a_sample_over_the_cap_is_dropped():
    result = drop_long_samples([_sample(100), _sample(9000), _sample(200)], 8000)
    assert result.dropped == 1
    assert [len(s.text) for s in result.kept] == [100, 200]


def test_the_cap_is_inclusive():
    # 8000 chars is affordable; the cap is the last length that is.
    assert drop_long_samples([_sample(8000)], 8000).dropped == 0
    assert drop_long_samples([_sample(8001)], 8000).dropped == 1


def test_the_longest_is_measured_before_the_drop():
    # The outlier is the finding. A record showing only what survived would hide
    # the 32,477-character page that cost eleven hours.
    result = drop_long_samples([_sample(100), _sample(32477)], 8000)
    assert result.max_chars == 32477
    assert result.dropped == 1


def test_the_page_that_killed_the_german_run_would_be_dropped():
    # 20260908T101611Z-qwen3vl-german-pages-v1: one validation page of 32,477
    # characters tokenized to 14,411 tokens and asked cross-entropy for 8.16 GiB,
    # at step 785 of 2355, 11h24m in.
    corpus = [_sample(383)] * 100 + [_sample(32477)]
    result = drop_long_samples(corpus, VLM_MAX_SAMPLE_CHARS["page"])
    assert result.dropped == 1
    assert all(len(s.text) <= 8000 for s in result.kept)


def test_nothing_is_dropped_from_an_ordinary_corpus():
    # Median 383, p90 ~2,200, p99 ~5,000 — the shape the cap was chosen against.
    corpus = [_sample(n) for n in (383, 371, 2183, 2275, 4969, 5438)]
    result = drop_long_samples(corpus, VLM_MAX_SAMPLE_CHARS["page"])
    assert result.dropped == 0
    assert result.max_chars == 5438


def test_a_mis_segmented_line_is_dropped_at_line_granularity():
    # A "line" of 1,200 characters is a block the segmenter merged, not a line.
    result = drop_long_samples([_sample(13), _sample(1200)],
                               VLM_MAX_SAMPLE_CHARS["line"])
    assert result.dropped == 1


def test_the_filter_says_what_it_did():
    assert "no sample over the cap" in str(drop_long_samples([_sample(10)], 8000))
    assert "dropped 1 sample(s)" in str(drop_long_samples([_sample(9000)], 8000))


# ── the visual budget across transformers versions (#86, 5.x) ───────────────
class _SizeDict:
    """transformers 5.x's `SizeDict`: attributes, not mapping access."""

    def __init__(self, longest_edge, shortest_edge):
        self.longest_edge = longest_edge
        self.shortest_edge = shortest_edge


class _Processor:
    def __init__(self, image_processor):
        self.image_processor = image_processor


class _ImageProcessor:
    def __init__(self, size):
        self.size = size
        self.patch_size = 16
        self.merge_size = 2


def test_budget_applies_to_a_4x_dict_size():
    from atr_training.vlm_dataset import apply_visual_budget

    ip = _ImageProcessor({"longest_edge": 16777216, "shortest_edge": 65536})
    applied = apply_visual_budget(_Processor(ip), 262144)
    assert ip.size["longest_edge"] == 262144
    assert applied.visual_tokens == 262144 // (16 * 2) ** 2


def test_budget_applies_to_a_5x_sizedict():
    """transformers 5.17 made `size` an object; the knob is still longest_edge.

    Before this, the guard fell through both branches and refused outright —
    correctly, since the alternative was training at 16,384 visual tokens an
    image against an intended 256 (#86). Caught on job 14717192.
    """
    from atr_training.vlm_dataset import apply_visual_budget

    ip = _ImageProcessor(_SizeDict(longest_edge=16777216, shortest_edge=65536))
    applied = apply_visual_budget(_Processor(ip), 262144)
    assert ip.size.longest_edge == 262144
    assert applied.visual_tokens == 262144 // (16 * 2) ** 2


def test_budget_still_refuses_when_there_is_no_knob():
    from atr_training.vlm_dataset import VisualBudgetError, apply_visual_budget

    ip = _ImageProcessor(None)
    with pytest.raises(VisualBudgetError):
        apply_visual_budget(_Processor(ip), 262144)


def test_warmup_ratio_is_converted_for_transformers_5x():
    """5.x dropped warmup_ratio and kept warmup_steps; the schedule must not move."""
    from vlm_train_svc.train_qlora import warmup_kwarg

    # Injected rather than probed, so this runs in the gateway venv, which has
    # no transformers at all.
    assert warmup_kwarg(0.05, 1000, supports_ratio=True) == {"warmup_ratio": 0.05}
    assert warmup_kwarg(0.05, 1000, supports_ratio=False) == {"warmup_steps": 50}
    # A ratio too small to reach a whole step still warms up for one.
    assert warmup_kwarg(0.0001, 100, supports_ratio=False) == {"warmup_steps": 1}


def test_no_warmup_asks_for_neither():
    from vlm_train_svc.train_qlora import warmup_kwarg

    assert warmup_kwarg(0.0, 1000, supports_ratio=False) == {}


# ── generation must stop at the end of the turn (Qwen3.5 ships no gen config) ─
class _Tok:
    def __init__(self, vocab, unk=0):
        self.vocab, self.unk_token_id = vocab, unk

    def convert_tokens_to_ids(self, name):
        return self.vocab.get(name, self.unk_token_id)


def test_stop_ids_come_from_each_models_own_tokenizer():
    """Looked up by name: the two families disagree on every id."""
    from vlm_train_svc.evaluate_qlora import stop_token_ids

    qwen3vl = _Tok({"<|im_end|>": 151645, "<|endoftext|>": 151643})
    qwen35 = _Tok({"<|im_end|>": 248046, "<|endoftext|>": 248044})
    assert stop_token_ids(qwen3vl) == [151645, 151643]
    assert stop_token_ids(qwen35) == [248046, 248044]


def test_a_tokenizer_without_stop_tokens_is_refused():
    """Better to fail than to let every prediction run to max_new_tokens."""
    from vlm_train_svc.evaluate_qlora import stop_token_ids

    with pytest.raises(RuntimeError):
        stop_token_ids(_Tok({}))


# ── dropping samples too short to teach anything but stopping ───────────────
def test_a_sample_under_the_floor_is_dropped():
    result = drop_short_samples([_sample(2), _sample(45), _sample(1)], 4)
    assert result.dropped == 2
    assert [len(s.text) for s in result.kept] == [45]


def test_the_floor_is_exclusive():
    # min_train_chars=4 means "at least 4 chars": 4 survives, 3 does not.
    assert drop_short_samples([_sample(4)], 4).dropped == 0
    assert drop_short_samples([_sample(3)], 4).dropped == 1


def test_a_floor_of_zero_drops_nothing():
    corpus = [_sample(1), _sample(2), _sample(45)]
    assert drop_short_samples(corpus, 0).dropped == 0


def test_the_shortest_is_measured_before_the_drop():
    # Same reason as the long filter: the record should show what the corpus
    # contained, not what survived it.
    result = drop_short_samples([_sample(1), _sample(45)], 4)
    assert result.min_chars == 1
    assert result.dropped == 1


def test_the_medieval_short_tail_is_what_the_floor_removes():
    # The measured medieval distribution: median 12 chars, 20.9% at 1-3 chars
    # (folio numbers, column figures, marginalia: "dat", "16", "B VI", "190").
    # A model trained on that emits end-of-turn after the first word and scores
    # 0.14 output/reference on real 16-40 char lines.
    corpus = [_sample(n) for n in (3, 2, 16, 1, 45, 12, 3, 64, 2, 104)]
    result = drop_short_samples(corpus, 4)
    assert result.dropped == 5
    assert [len(s.text) for s in result.kept] == [16, 45, 12, 64, 104]


def test_the_nineteenth_century_corpus_is_barely_touched():
    # Median 30 chars, 0.8% at <=3 — which is why it reached 1.0% CER on the
    # same code, prompt and evaluator that gave medieval 0.53.
    corpus = [_sample(n) for n in (23, 27, 30, 30, 32, 35, 28, 31)]
    assert drop_short_samples(corpus, 4).dropped == 0


# ── granularity: mixed (#59) ────────────────────────────────────────────────
def _samples(kind: str, n: int) -> list[Sample]:
    return [Sample(image=f"{kind}/{i}.jpg", text=f"{kind} {i}", source_type=kind) for i in range(n)]


def test_a_mix_is_drawn_in_the_asked_shares():
    by_kind = {"line": _samples("line", 1000), "block": _samples("block", 200),
               "page": _samples("page", 100)}
    got = mix_samples(by_kind, {"line": 0.5, "block": 0.3, "page": 0.2}, seed=1)
    counts = collections.Counter(s.source_type for s in got)
    assert counts["page"] == 100                      # the scarcest kind, used in full
    assert counts["line"] == 250 and counts["block"] == 150
    assert len(got) == 500


def test_the_scarcest_kind_bounds_the_total_so_nothing_repeats():
    by_kind = {"line": _samples("line", 10), "page": _samples("page", 1)}
    got = mix_samples(by_kind, {"line": 0.9, "page": 0.1}, seed=1)
    assert len(got) == len(set(s.image for s in got))
    assert collections.Counter(s.source_type for s in got) == {"line": 9, "page": 1}


def test_the_same_seed_draws_the_same_corpus():
    by_kind = {"line": _samples("line", 100), "page": _samples("page", 20)}
    mix = {"line": 0.5, "page": 0.5}
    assert [s.image for s in mix_samples(by_kind, mix, 7)] == [
        s.image for s in mix_samples(by_kind, mix, 7)]
    assert [s.image for s in mix_samples(by_kind, mix, 7)] != [
        s.image for s in mix_samples(by_kind, mix, 8)]


def test_a_kind_the_corpus_cannot_supply_is_an_error():
    with pytest.raises(VlmDatasetError, match="no such samples"):
        mix_samples({"line": _samples("line", 5)}, {"line": 0.5, "page": 0.5}, seed=1)


def test_samples_for_mixed_builds_every_kind(page, tmp_path):
    got = samples_for([page], "mixed", root=tmp_path, mix={"line": 0.5, "page": 0.5}, seed=3)
    assert {s.source_type for s in got} == {"line", "page"}


def test_mixed_params_budget_by_kind():
    params = VlmTrainParams(granularity="mixed")
    assert params.mix() == DEFAULT_GRANULARITY_MIX
    assert params.kinds() == ("line", "block", "page")
    # the processor ceiling is the largest kind; each kind keeps its own budget
    assert params.pixel_budget() == VLM_PIXEL_BUDGET["page"]
    assert params.kind_budgets() == {k: VLM_PIXEL_BUDGET[k] for k in ("line", "block", "page")}
    assert params.sequence_budget() == VLM_MAX_SEQ_LEN["page"]
    assert params.generation_budget() == VLM_MAX_NEW_TOKENS["page"]


def test_a_mix_is_normalised_and_may_leave_a_kind_out():
    params = VlmTrainParams(granularity="mixed", granularity_mix={"line": 3, "page": 1})
    assert params.mix() == {"line": 0.75, "page": 0.25}
    assert params.kinds() == ("line", "page")
    assert params.pixel_budget() == VLM_PIXEL_BUDGET["page"]


def test_a_mix_on_a_single_granularity_is_refused():
    """Silently ignoring it would train something other than what was written."""
    with pytest.raises(ValueError, match="only meaningful at granularity: mixed"):
        VlmTrainParams(granularity="line", granularity_mix={"line": 1.0})
    with pytest.raises(ValueError, match="unknown kind"):
        VlmTrainParams(granularity="mixed", granularity_mix={"paragraph": 1.0})
    with pytest.raises(ValueError, match="greater than 0"):
        VlmTrainParams(granularity="mixed", granularity_mix={"line": 1.0, "page": 0})


def test_a_single_granularity_is_its_own_mix():
    params = VlmTrainParams(granularity="block")
    assert params.mix() == {"block": 1.0} and params.kinds() == ("block",)
    assert params.kind_budgets() == {"block": VLM_PIXEL_BUDGET["block"]}


class _Image:
    def __init__(self, size): self.size = size
    def resize(self, size): return _Image(size)


def test_fit_pixels_shrinks_and_never_grows():
    assert fit_pixels(_Image((2000, 1500)), 2000 * 1500).size == (2000, 1500)
    assert fit_pixels(_Image((100, 100)), 10_000_000).size == (100, 100)
    out = fit_pixels(_Image((4000, 3000)), 1_000_000).size
    assert out[0] * out[1] <= 1_000_000 and abs(out[0] / out[1] - 4 / 3) < 0.01

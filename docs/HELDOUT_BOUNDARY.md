# Which published numbers cross the held-out boundary?

The held-out boundary (#98, `docs/BASE_MODEL_LADDER.md` §1f) is proven for the
medieval ladder: the September corpus held 306,582 training lines, the October
one 219,261, because `_reserve_eval_documents` keeps 3,173 of 11,277 pages for
`german-medieval-v1`. **Which other published numbers cross the same boundary**
is the question of #247.

**Answer, measured 2026-10-17 on UBELIX over both stores:** of 58 runs whose
record carries a CER, **6 crossed**, 43 are clean, 8 are undecidable and 1 is
uncheckable. The six are the medieval September corpus and nothing else. Two
more published figures are crossed whose CER is not in any record — the
medieval Gemma line arms — and only `--all` finds them. Details in
[Results](#results).

## `reserved_pages` cannot answer the question

Over both stores, 14.10.2026: 120 job records, 54 of them completed and
carrying a CER. `progress.reserved_pages` takes four values:

| `reserved_pages` | runs | what it was taken to mean |
|---|---:|---|
| `None` | 22 | nobody looked |
| `0` | 20 | registry consulted, it held nothing back |
| `3173` | 10 | medieval corpus, after the boundary |
| `1318` | 2 | Kurrent corpus, after the boundary |

So it is a **corpus fingerprint, not a boundary marker** — the number hangs on
the dataset cut, not on the date — and 42 runs carry no usable answer.

**And the 20 zeros cannot all be read as "checked".** Two of them were measured
while building this, and in both cases the registry was never consulted at all:

### A line manifest was parsed as a path

`_reserve_eval_documents` received `lines_train.jsonl` — one JSON object per
line — and handed each line to `document_of`, which reads a *page filename*.
`Path(line).name` is everything after the **last slash in the whole JSON blob**,
so the answer depended on field order and on the transcription:

```
image zuletzt                      document_of=12345   reserved=1
ein zweites Feld mit Slash danach   document_of=None    reserved=0
Text enthält einen Slash            document_of=None    reserved=0
```

`anno 1431 / vigilia` is routine in this material. A held-out page stayed in
training when the line it came from happened to contain a slash — and the record
said `reserved_pages: 0`, which is also what a real check that found nothing
says.

Fixed: the document now comes from the sample's `page` field (`heldout.document_of_entry`),
which is the unit the registry is written in. A line corpus cropped from
materialised pages is therefore checked like any other; one whose pages are
foreign scans has no `docId` the registry could match, and the record now says
that instead of `0`.

### A chunk plan was parsed as a page list

In chunked mode `prepare` does not write a page manifest at all — it writes
`train_plan.json` and defers the train side to `compile`. That plan arrived at
the reservation step and was read as a nine-line page list:

```
als Seitenliste gelesen: 9 'Seiten', reserviert 0
document_of je Zeile: [None × 9]
```

So **the boundary was not enforced for chunked runs**, and the record said `0`.
Latent — `chunk_pages` defaults to `0`, chunking is kraken-only and needs
explicit `eval_projects` — but armed: it is the corpus-scale run that turns
chunking on, and that is the run the boundary matters most for.

**Enforced since #255.** The pages of a chunked run exist for one chunk at a
time, inside the compile loop, so that is where the registry is applied now:
reserved pages are dropped from each chunk's manifest, the counts are summed
across chunks, and the sum lands on the record as `compile (per chunk)`. A
resume reads each adopted chunk's count back out of its `.done.json`; a record
written before that key existed makes the run's total `None` rather than a sum
that quietly omits it. `prepare` still records `not checked: chunked prepare
defers the train side to compile`, which `compile` then overwrites with the real
number.

Runs compiled **before** that change are the ones this census reads as
`uncheckable`: their train side was materialised per chunk and discarded, so
there is nothing left to apply the registry to.

## The method

For each run whose record carries a CER (58 on 2026-10-17): apply **today's** registry
(`config/heldout_eval_documents.json`) to the manifest the run actually trained
on, and count how many pages *would* fall. No GPU, no network, minutes.

```
python scripts/heldout_boundary.py --markdown   # the Results table, to paste below
python scripts/heldout_boundary.py --json | jq '.summary'
python scripts/heldout_boundary.py --all        # also runs whose CER is not in the record
```

On UBELIX it runs inside the training image, which has the dependencies the
plain `python3` there lacks:

```
apptainer exec --bind /storage --bind /scratch --bind /rs_scratch \
  ~/ubelix/vlm-train-tf5.sif env PYTHONPATH=$PWD/src:$PWD/engines \
  python3 scripts/heldout_boundary.py --markdown
```

It also names a third store, `local`, as unreadable there: that is this box's
`TrainerSettings.jobs_root`, which on UBELIX points at a directory that does not
exist. It holds no runs; the message is the script refusing to assume so.

Both stores by default, because they do not hold the same runs — 61 in the share
against 58 in scratch — and a run purged from one may still be whole in the
other.

### Four states, and only two of them are verdicts about a number

| state | meaning |
|---|---|
| `clean` | entries name documents, and none is reserved today — the number stands |
| `crossed` | trained on material reserved today; the number is optimistic |
| `uncheckable` | nothing in the manifest names a document (line corpus of foreign scans, or a chunked run) — **nothing was ruled out** |
| `gone` | no manifest left (scratch purge after 30 days) — undecidable |

#247 named three states and warned that the third must not fall into the first.
`uncheckable` is the fourth, and it exists because of the two findings above.
A run that cannot be checked is never counted as unaffected: that is the one
direction in which a census like this goes quietly optimistic, and it is the
direction the records themselves already failed in.

The table prints the recorded `reserved_pages` **beside** the measurement rather
than instead of it. Where the two disagree, that disagreement is the finding.

### A run that adopted its corpus has its builder's manifest

The first pass left **25** runs `gone`, one of them not a run at all (below).
Sixteen of the other 24 had never had a manifest
of their own to lose: they adopted a compiled corpus from the artefact cache,
and their record says from whom — `progress.artefact = "<key> reused, built by
<job>"`. An adopter trained on the builder's bytes, so the builder's manifest is
its manifest. `resolve_adoptions` takes the builder's verdict over, after every
store is read, because the builder can sit in the other store and need not have
a CER of its own (`qwen3vl-german-pages-v5-asteraix` was adopted from a
cancelled first attempt).

**The key is the identity, the name is a label.** The builder must confirm the
key in its own record (`"<key> built by this job"`, since #238 followed by
`", stored in N min"`). That is not pedantry:
`gemma4-12b-medieval-german-page-v1` adopted `2439ac90f727`, its prepare log
names `qwen3vl-medieval-german-page-v1` as the builder, and its record names
`20261001T090507Z-olmocr2-7b-medieval-german-page-v1` — an abandoned olmOCR
attempt that rebuilt the same key and re-registered the cache entry while the
Gemma run was being requeued. Both confirm the key and both hold an 8,104-page
manifest, so here the answer is the same either way; a builder that does not
confirm leaves the adopter `gone`, never `clean`.

**And adoption is a third way the record says `0` without checking.** An adopter
skips prepare, so the reservation never runs and `reserved_pages` keeps what the
job started with. Three records say `0` where the builder had held pages back:
`qwen3.5-4b-medieval-german-page-v1` and `gemma4-e4b-medieval-german-page-v1`
(builder: 3,173) and `qwen3vl-german-pages-v5-asteraix` (builder: 3,187). All
three are clean — but not because of the `0`.

A directory in a store is not necessarily a run: `evalset-federal-minutes`
holds the published-benchmark reports beside the jobs. Without a `job.json` and
without a job-shaped name it is skipped, rather than counted as a purged run.

## Results

Measured 2026-10-17 on UBELIX, both stores, today's registry (350 documents,
all from `german-medieval-v1`).

| verdict | runs | |
|---|---:|---|
| `clean` | **43** | 27 from their own manifest, 16 through their builder |
| `crossed` | **6** | 15,349 training pages reserved today |
| `uncheckable` | 1 | `kraken-corpus-thun-ft-v1`, chunked |
| `gone` | 8 | all from 10.–14.09., the first generation |
| **runs with a CER** | **58** | |

### The six crossed runs

| run | CER | pages | reserved today |
|---|---:|---:|---:|
| `qwen3vl-medieval-german-v3` | 0.1120 | 11,448 | **3,091** (27.0 %) |
| `ladder-med-qwen35-27b` | **0.0955** | 11,448 | **3,091** |
| `ladder-med-qwen35-9b` | 0.1213 | 11,448 | **3,091** |
| `ladder-med-olmocr2-7b` | 0.7465 | 11,448 | **3,091** |
| `qwen3vl-german-pages-v3` | 0.9756 | 12,557 | **2,979** (23.7 %) |
| `qwen3vl-german-medieval-v1` | 0.2324 | 279 | 6 (2.2 %) |

**They are exactly the four numbers the medieval ladder stood on** — including
the 9.55 % this repository carried as its best medieval line result until
14.10. — and their corrected counterparts are in `docs/BASE_MODEL_LADDER.md`
§1f-bis. The other two do not matter in practice and are listed so the list is
complete: `qwen3vl-german-pages-v3` is a collapse at 97.56 %, and
`qwen3vl-german-medieval-v1` loses 6 of 279 pages.

### Two more published figures, whose CER is in no record

The census counts runs whose *record* carries a CER, as a proxy for "was
published". The proxy has a hole. The medieval Gemma line arms —
`ladder-med-gemma4-e4b` (six attempts) and `ladder-med-gemma4-12b` (two) — are
all `failed` with no metrics: most attempts are the drop-in failures of §6 on
the published page, which died within minutes, and the ones that trained
crashed in their evaluation (items 6 and 7 there), so their CER was recovered by
re-scoring the stored adapter. **All eight attempts are crossed**, 3,091 of
11,448 pages each, because they share one corpus. That puts two more published
figures on the wrong side: Gemma-E4B 16.9 % and Gemma-12B 12.2 % on medieval
lines. The comparison they were
published in still holds *within itself* — `qwen3vl-medieval-german-v3`, the
Qwen arm beside them, trained on the same corpus — but no figure in it is
comparable with anything measured after the boundary.

`--all` lists them (121 runs: 56 clean, 34 crossed, 1 uncheckable, 30 gone);
the default does not, and says so in its help text.

### Two cross-checks

**Two implementations agree.** A first survey for #247 (PR #261, first
version) applied `HeldOutDocuments.split` to each `pages_train.lst` with its own
loop; this module uses `HeldOutDocuments.reserve` through `check_run`. They name
the same six crossed runs with the same counts — 3,091, 2,979 and 6.

**A positive record is right where it can be checked.** Every run that records
`reserved_pages: 3173` and still holds a manifest shows 8,104 pages in it —
11,277 − 3,173 exactly — and comes out clean: four of four. A *positive* count
has so far never been wrong; it is the `0` that cannot be trusted (above).

### What the census decided

**The 19th-century series is clean — with one exception that shows why the
name of a run is no evidence.** Every `german-xix` arm shows 0 reserved pages of
23,971, and so does `kraken-german-xix-v1` (14,093). But
`qwen3vl-german-pages-v3`, filed with the XIX work, trained on 12,557 pages of
which 2,979 are reserved today.

**The `MM_2_*` worry was unfounded.** #247 asked whether `kurrent-xix` was
structurally immune to the reservation because `document_of` cannot read its
file names. Measured: both Kurrent arms have **13,606 of 13,606** pages with a
document id, and no run in either store is `uncheckable` for that reason.

**Every number in the current tables is clean.** The post-boundary ladder
(4B/9B/27B, Qwen3.8-27B), the five olmOCR page knobs, both Kurrent arms, every
medieval page arm and every 19th-century arm — directly or through the run that
built their corpus.

### The eight undecidable runs

All from 10.–14.09.: `qwen3vl-medieval-german-v1`, `-v2`, the three
`qwen3.5-*-medieval-german-v1`, `qwen3.5-{2b,4b}-german-xix-v1` and
`resume-probe-a100`. Their manifests went with the 30-day scratch purge, their
records carry `reserved_pages: None`, and none of them adopted a corpus. These
are the first-generation numbers that the parser fix replaced; nothing current
rests on them. **They stay open, and that is the result, not a missing one.**

### The first held-out numbers (#301)

`german-medieval-v1`, part test: **695 pages from 200 documents**, held out by
document, as a VLM draw under
`eval_sets/german-medieval-v1/vlm-pages-test/` (`scripts/build_page_benchmark.py`).
Scored with `ubelix/rescore_on_shared_draw.py --heldout`, each adapter at its
own pixel budget and with its own prompt. Both reports say `checked: true` and
`shared: []`: no document of the draw is among the 1,180 the corpus trained on.
Read off UBELIX on 26.10.2026 (`eval_report.german-medieval-v1-test.json`).

| arm | validation CER (164 pages, drawn by page) | **held-out CER** | `truncated_cer` | `length_ratio` | at the token cap |
|---|---:|---:|---:|---:|---:|
| `px4m-native` | 0.2659 | **0.3674** | 0.3475 | 0.898 | 38 of 695 |
| `px4m-native-r16` | 0.3214 | **0.4186** | 0.3780 | 0.919 | 37 of 695 |

Two readings. The ranking holds held out: `px4m-native` ahead by ~5 points
either way. And both lose ~10 points from validation to held-out, which is the
measure of how much the by-page validation split flattered them: unknown hands
are harder than the pages beside the training pages. Under
`convention_normalized` the held-out CERs are 0.3433 and 0.3782.

**The two XVI–XIX page arms on the same 695 pages** (way a of #301, read off
UBELIX on 27.10.2026, Slurm 20187794/20187795 on code `448bef6`). Both reports
say `checked: true` and `shared: []` against the corpus builder's training
manifest (`20261013T100501Z-qwen3.5-4b-german-kurrent-xvi-xix-page-v1`, 13,606
entries, 1,310 documents).

| arm | validation CER (171 pages, drawn by page) | **held-out CER** | `truncated_cer` | `length_ratio` | at the token cap |
|---|---:|---:|---:|---:|---:|
| XVI–XIX `px4m-native` | 0.1571 | **0.3681** | 0.3354 | 0.937 | 32 of 695 |
| XVI–XIX `qwen3.5-4b-…-page-v1` | 0.2704 | **0.5405** | 0.4168 | 1.011 | 99 of 695 |

Three readings, none of them about the XVI–XIX target:

* The XVI–XIX olmOCR arm reads these early pages as well as the medieval one
  (0.368 against 0.367), though only four of its twenty datasets reach into
  the XVI century.
* The gap to validation (21 and 27 points) is **not** the measure of how much
  the by-page validation flattered them, as it is for the medieval arms: this
  draw sits at the early edge of the XVI–XIX corpus, so the gap mixes unknown
  hands with another period.
* The Qwen arm hits `max_new_tokens=1536` on 99 pages — one in seven — so
  part of its CER is cut-off text; `truncated_cer` (8 points behind olmOCR,
  not 17) is the fairer reading of the reading itself. Under
  `convention_normalized`: 0.3452 and 0.4910.

The XVI–XIX target itself needs a Kurrent page set of its own: way c in
[EVAL_SETS.md](EVAL_SETS.md).

### The full table

<details>
<summary>58 runs, as printed by <code>--markdown</code></summary>

| Lauf | Store | CER | `reserved_pages` | Verdikt | gefallen | Grund |
|---|---|---:|---:|---|---:|---|
| `20261019T180118Z-olmocr2-7b-medieval-page-px4m-native-r16` | scratch | 0.3214 | ? | clean | 0 | added 26.10. (#301), not printed by the audit: the held-out rescore compared the draw with the training manifest of its builder `20261001T093909Z-olmocr2-7b-medieval-german-page-v1` (8104 entries, 1180 documents) and found no shared document; `reserved_pages` not read |
| `20261014T194417Z-ladder-med-qwen35-4b-v2` | scratch | 0.1594 | 3173 | clean | 0 | adopted f7188cd0b798 from 20261013T043436Z-ladder-med-qwen35-27b-v2 (scratch): 8104 of 8104 entries name a document; none reserved |
| `20261014T193118Z-ladder-med-qwen35-9b-v2` | scratch | 0.1375 | 3173 | clean | 0 | adopted f7188cd0b798 from 20261013T043436Z-ladder-med-qwen35-27b-v2 (scratch): 8104 of 8104 entries name a document; none reserved |
| `20261013T121310Z-olmocr2-7b-medieval-page-px4m-native` | scratch | 0.2659 | 3173 | clean | 0 | adopted 2ad500ee8414 from 20261001T093909Z-olmocr2-7b-medieval-german-page-v1 (scratch): 8104 of 8104 entries name a document; none reserved |
| `20261013T100501Z-qwen3.5-4b-german-kurrent-xvi-xix-page-v1` | scratch | 0.2704 | 1318 | clean | 0 | 13606 of 13606 entries name a document; none reserved |
| `20261013T100004Z-ladder-med-qwen38-27b-v3` | scratch | 0.1215 | 3173 | clean | 0 | adopted f7188cd0b798 from 20261013T043436Z-ladder-med-qwen35-27b-v2 (scratch): 8104 of 8104 entries name a document; none reserved |
| `20261013T043436Z-ladder-med-qwen35-27b-v2` | scratch | 0.1224 | 3173 | clean | 0 | 8104 of 8104 entries name a document; none reserved |
| `20261010T111013Z-ladder-med-qwen38-27b-v2` | scratch | 0.1215 | 3173 | clean | 0 | 8104 of 8104 entries name a document; none reserved |
| `20261009T174425Z-olmocr2-7b-medieval-page-px4m` | scratch | 0.2775 | 3173 | clean | 0 | adopted 2ad500ee8414 from 20261001T093909Z-olmocr2-7b-medieval-german-page-v1 (scratch): 8104 of 8104 entries name a document; none reserved |
| `20261009T174425Z-olmocr2-7b-medieval-page-nativeprompt` | scratch | 0.3438 | 3173 | clean | 0 | adopted 2ad500ee8414 from 20261001T093909Z-olmocr2-7b-medieval-german-page-v1 (scratch): 8104 of 8104 entries name a document; none reserved |
| `20261009T150138Z-qwen3.5-4b-german-kurrent-xvi-xix-v1` | scratch | 0.0882 | 1318 | clean | 0 | 13606 of 13606 entries name a document; none reserved |
| `20261008T103636Z-olmocr2-7b-medieval-page-px1288` | scratch | 0.4114 | 3173 | clean | 0 | adopted 2ad500ee8414 from 20261001T093909Z-olmocr2-7b-medieval-german-page-v1 (scratch): 8104 of 8104 entries name a document; none reserved |
| `20261008T103636Z-olmocr2-7b-medieval-page-cap3072` | scratch | 0.4674 | 3173 | clean | 0 | adopted 2ad500ee8414 from 20261001T093909Z-olmocr2-7b-medieval-german-page-v1 (scratch): 8104 of 8104 entries name a document; none reserved |
| `20261001T093909Z-olmocr2-7b-medieval-german-page-v1` | scratch | 0.3492 | 3173 | clean | 0 | 8104 of 8104 entries name a document; none reserved |
| `20261001T085321Z-ladder-med-olmocr2-7b` | scratch | 0.7465 | None | crossed | 3091 | 3091 of 11448 training pages are reserved today (345 documents) |
| `20260930T211455Z-ladder-med-qwen35-9b` | scratch | 0.1213 | None | crossed | 3091 | 3091 of 11448 training pages are reserved today (345 documents) |
| `20260930T210559Z-gemma4-12b-medieval-german-page-v1` | scratch | 0.9620 | 3173 | clean | 0 | adopted 2439ac90f727 from 20261001T090507Z-olmocr2-7b-medieval-german-page-v1 (scratch): 8104 of 8104 entries name a document; none reserved |
| `20260930T170315Z-ladder-xix-gemma4-12b` | scratch | 0.2122 | 0 | clean | 0 | 23971 of 23971 entries name a document; none reserved |
| `20260929T160031Z-gemma4-e4b-medieval-german-page-v1` | scratch | 0.5184 | 0 | clean | 0 | adopted 2439ac90f727 from 20260923T202356Z-qwen3vl-medieval-german-page-v1 (scratch): 8104 of 8104 entries name a document; none reserved |
| `20260926T080902Z-ladder-xix-gemma4-e4b` | scratch | 0.1110 | 0 | clean | 0 | 23971 of 23971 entries name a document; none reserved |
| `20260926T073357Z-qwen3.5-4b-medieval-german-page-v1` | scratch | 0.4160 | 0 | clean | 0 | adopted 2439ac90f727 from 20260923T202356Z-qwen3vl-medieval-german-page-v1 (scratch): 8104 of 8104 entries name a document; none reserved |
| `20260925T045500Z-ladder-med-qwen35-27b` | scratch | 0.0955 | 0 | crossed | 3091 | 3091 of 11448 training pages are reserved today (345 documents) |
| `20260924T124211Z-qwen3vl-german-xix-mixed-v1` | scratch | 0.2558 | 0 | clean | 0 | 23971 of 23971 entries name a document; none reserved |
| `20260923T202356Z-qwen3vl-medieval-german-page-v1` | scratch | 0.4265 | 3173 | clean | 0 | 8104 of 8104 entries name a document; none reserved |
| `20260923T005341Z-qwen3vl-german-xix-page-v1` | scratch | 0.0801 | 0 | clean | 0 | 23971 of 23971 entries name a document; none reserved |
| `20260922T181814Z-qwen3vl-german-xix-block-v1` | scratch | 0.0754 | 0 | clean | 0 | 23971 of 23971 entries name a document; none reserved |
| `20260918T070304Z-qwen3.5-2b-german-xix-v2` | scratch | 0.0549 | 0 | clean | 0 | 23971 of 23971 entries name a document; none reserved |
| `20260918T070304Z-qwen3.5-0.8b-german-xix-v2` | scratch | 0.0704 | 0 | clean | 0 | 23971 of 23971 entries name a document; none reserved |
| `20260918T070300Z-qwen3.5-4b-german-xix-v2` | scratch | 0.0478 | 0 | clean | 0 | 23971 of 23971 entries name a document; none reserved |
| `20260916T090417Z-qwen3vl-german-xix-v2` | scratch | 0.0533 | 0 | clean | 0 | 23971 of 23971 entries name a document; none reserved |
| `20260915T055400Z-qwen3vl-medieval-german-v3` | scratch | 0.1120 | None | crossed | 3091 | 3091 of 11448 training pages are reserved today (345 documents) |
| `20260914T055236Z-qwen3vl-medieval-german-v2` | scratch | 0.4995 | None | gone | 0 | no train manifest in data/ |
| `20260911T154340Z-qwen3.5-4b-german-xix-v1` | scratch | 0.0107 | None | gone | 0 | no train manifest in data/ |
| `20260911T154340Z-qwen3.5-2b-german-xix-v1` | scratch | 0.0141 | None | gone | 0 | no train manifest in data/ |
| `20260911T114344Z-qwen3vl-german-xix-v1` | scratch | 0.0100 | None | clean | 0 | 24097 of 24097 entries name a document; none reserved |
| `20260911T053821Z-qwen3.5-4b-medieval-german-v1` | scratch | 0.5806 | None | gone | 0 | no train manifest in data/ |
| `20260911T053821Z-qwen3.5-2b-medieval-german-v1` | scratch | 6.1581 | None | gone | 0 | no train manifest in data/ |
| `20260911T053821Z-qwen3.5-0.8b-medieval-german-v1` | scratch | 6.9842 | None | gone | 0 | no train manifest in data/ |
| `20260910T202205Z-qwen3vl-medieval-german-v1` | scratch | 0.5322 | None | gone | 0 | no train manifest in data/ |
| `20260910T180413Z-resume-probe-a100` | scratch | 0.3057 | None | gone | 0 | no train manifest in data/ |
| `20261014T064653Z-vlm-xix-effective-batch-01-r0-3de4c2e3aef1` | share | 0.1455 | 0 | clean | 0 | adopted 1ab0a28006d1 from 20261012T135706Z-vlm-xix-effective-batch-01-noise-s42 (share): 23971 of 23971 entries name a document; none reserved |
| `20261012T135715Z-vlm-xix-effective-batch-01-noise-s45` | share | 0.1842 | 0 | clean | 0 | adopted 1ab0a28006d1 from 20261012T135706Z-vlm-xix-effective-batch-01-noise-s42 (share): 23971 of 23971 entries name a document; none reserved |
| `20261012T135712Z-vlm-xix-effective-batch-01-noise-s44` | share | 0.1877 | 0 | clean | 0 | adopted 1ab0a28006d1 from 20261012T135706Z-vlm-xix-effective-batch-01-noise-s42 (share): 23971 of 23971 entries name a document; none reserved |
| `20261012T135709Z-vlm-xix-effective-batch-01-noise-s43` | share | 0.1837 | 0 | clean | 0 | adopted 1ab0a28006d1 from 20261012T135706Z-vlm-xix-effective-batch-01-noise-s42 (share): 23971 of 23971 entries name a document; none reserved |
| `20261012T135706Z-vlm-xix-effective-batch-01-noise-s42` | share | 0.3010 | 0 | clean | 0 | 23971 of 23971 entries name a document; none reserved |
| `20261007T131819Z-kraken-german-xix-v1` | share | 0.0312 | 0 | clean | 0 | 14093 of 14093 entries name a document; none reserved |
| `20260916T205123Z-qwen3vl-german-pages-v5-asteraix` | share | 0.5235 | 0 | clean | 0 | adopted 82db328c96d7 from 20260916T115939Z-qwen3vl-german-pages-v5-asteraix (share): 9457 of 9457 entries name a document; none reserved |
| `20260916T115539Z-trocr-thun-smoke-asteraix-v2` | share | 0.9757 | 0 | clean | 0 | 127 of 127 entries name a document; none reserved |
| `20260914T074401Z-trocr-thun-smoke-v3` | share | 0.9585 | None | clean | 0 | 127 of 127 entries name a document; none reserved |
| `20260910T110352Z-qwen3vl-german-pages-v3` | share | 0.9756 | None | crossed | 2979 | 2979 of 12557 training pages are reserved today (345 documents) |
| `20260903T105603Z-kraken-corpus-thun-ft-v1` | share | 0.2054 | None | uncheckable | 0 | chunked run: the train side was materialised per chunk and discarded |
| `20260903T104406Z-qwen3vl-sg-missiven-v1` | share | 0.2785 | None | clean | 0 | 1500 of 1500 entries name a document; none reserved |
| `20260821T163926Z-qwen3vl-german-medieval-v1` | share | 0.2324 | None | crossed | 6 | 6 of 279 training pages are reserved today (3 documents) |
| `20260813T144649Z-kraken-thun-kurrent-v2` | share | 0.2180 | None | clean | 0 | 127 of 127 entries name a document; none reserved |
| `20260813T141927Z-kraken-thun-kurrent-v1` | share | 0.2350 | None | clean | 0 | 127 of 127 entries name a document; none reserved |
| `20260813T090059Z-kraken-thun-finetune-v1` | share | 0.3921 | None | clean | 0 | 127 of 127 entries name a document; none reserved |
| `20260808T080206Z-qwen3vl-thun-smoke` | share | 0.4656 | None | clean | 0 | 40 of 40 entries name a document; none reserved |
| `20260807T212539Z-kraken-medieval-scripts-v1` | share | 0.7074 | None | clean | 0 | 4121 of 4121 entries name a document; none reserved |
| `20260807T161137Z-kraken-thun-missiven-v1` | share | 0.9838 | None | clean | 0 | 127 of 127 entries name a document; none reserved |

</details>

## Still open

* **The 171 pages.** The three medieval September runs each carry 11,448 pages
  in `data/pages_train.lst`, including `qwen3vl-medieval-german-v3`, the 4B
  reference of the §1b table — against 11,277 in October, *before* the 3,173 are
  deducted. The census sharpens this rather than answering it:

  | | selected | reserved | remaining |
  |---|---:|---:|---:|
  | September (the crossed arms) | 11,448 | 3,091 | 8,357 *(hypothetical)* |
  | October (everything since) | 11,277 | 3,173 | 8,104 |
  | difference | **−171** | +82 | **−253** |

  So the October corpus is not "September minus reserved". The selection moved
  by 171 pages, and 171 pages *fewer* produced 82 reserved pages *more*: the
  composition changed, not only the size. The registry was the same for both
  measurements, so it is not the registry. It is a property of two
  materialisations, and not answerable from the repository alone.
* **The published page.** `docs/vlm-finetuning.md` in
  `thodel/agentic-historian-outputs` carries the affected numbers: its §8a
  already corrects the ladder, but §6 (`qwen3vl-medieval-german-v3`, both Gemma
  line arms) is crossed and not yet marked, nor is the olmOCR line figure (74.6 %, §4 and §8) and the 9 B line
figure beside it in §4. The reach is now known; correcting it is #248.

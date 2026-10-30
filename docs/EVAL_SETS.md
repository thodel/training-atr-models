# Evaluation sets

An evaluation set is not a file. A file gets deleted — `german_test.arrow` and
`german_val.arrow` were compiled on 06.09.2026, produced the only trustworthy
German CER this project has, and were gone by 21.09. when idhefix was cleaned up
(serving-atr-inference#143). What survived was the set's **identity**: the 350
Transkribus document ids in
[`config/heldout_eval_documents.json`](../config/heldout_eval_documents.json),
which is in the repo precisely so that it outlives any machine.

This document is the other half: how each set is rebuilt from that identity, and
what number proves the rebuild is the same corpus rather than a similar one.

---

## `german-medieval-v1`

Medieval German, four corpora, held out **by document** so no hand appears on
both sides of the split.

| | test | val |
|---|---:|---:|
| pages | 695 | 769 |
| documents | 200 | 150 |
| characters (as `ketos test` counts them, NFD) | 882,255 | — |

Source corpora: `image-text_rats-und-richtebuecher_xv-xvi`,
`image-text_bullinger-autoren`,
`image-text_koenigsfelden-charters-post-1500`, `image-text_aaeb-xiv-xvii`.

**Built by** `scripts/make_split.py` (seed **20260810**) from the page pool of job
`20260905T190759Z-kraken-german-eval-pool-v1` — 12,301 pages in 1,640 documents,
`leak_documents_into_train: 0`. The per-document cap is what makes the set cover
many hands instead of a few manuscripts deeply; pages of a held-out document
beyond the cap are **dropped, never returned to train** (86 for test, 1,773 for
val), because recycling them would put the same hand on both sides.

### Where it lives

| | |
|---|---|
| definition | `<pool job>/data/split/{pages_test,pages_val}.lst` + `split.json` |
| page pool | `<pool job>/data/pages/` — 24,602 files (XML + JPG) |
| compiled, durable | `/mnt/wbkolleg_dh_1/Textrecognition_Training/eval_sets/german-medieval-v1/` |
| compiled, local on asteraix | `~/atr-cache/arrows/german_{test,val}.arrow` |

The share copy is **outside** the jobs tree on purpose: job directories are
cleaned (#183), and that is how the first copy was lost.

### Rebuilding it

`scripts/restore_eval_split.py` is the supported route: it rebuilds the split from
the registry and **refuses to call the result by this set's name** unless it
matches. Prefer it to the raw commands below, and read
`atr_training.split_identity` for why matching document ids is necessary but not
sufficient — `val_per_doc` / `test_per_doc` also decide *which pages* of a capped
document are drawn, and the registry did not originally record them.

The compile itself, for when the split lists are already in hand:

```bash
S=/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/jobs/20260905T190759Z-kraken-german-eval-pool-v1/data/split
A=$HOME/atr-cache/arrows
K=$HOME/Repo/training-atr-models/.venvs/kraken-train/bin/ketos
mkdir -p $A
for role in test val; do
  $K --device cpu --workers 4 compile --format-type page \
     --files $S/pages_${role}.lst --output $A/german_${role}.arrow \
     --skip-empty-lines
done
```

That argv is what `atr_training.ketos_cmd.compile_cmd` produces, so it stays
correct when the builder changes. `--device cpu` on purpose: a compile needs no
GPU and should not claim one.

The compile logs warnings of the form *"polygon outside of image bounds"* and
skips those lines. That is the known mis-segmentation class (#90), not a fault of
the rebuild.

### The acceptance test — and why a digest is not one

Arrow files are not byte-reproducible. The rebuild of 29.09.2026 came out 464
bytes **smaller** for test and 1,000 bytes **larger** for val than the files of
06.09., on 2.8 and 3.1 GB. A digest comparison would have called that a different
corpus; it is the same one.

What does settle it is the number of characters the set contains, and the error
counts a known model produces on it. `kraken-medieval-german-v2`, downloaded from
the Hub and scored on the rebuilt test set:

```
$K --device cuda:0 --workers 8 test --model kraken-medieval-german-v2.mlmodel \
   --test-data german_test_bin.lst --format-type binary --normalization NFD
```

| | recorded in the model's `metadata.json` (06.09.) | rebuilt set (29.09.) |
|---|---:|---:|
| Characters | 882,255 | **882,255** |
| Errors | 188,022 | **188,022** |
| Insertions / Deletions / Substitutions | — | 94,452 / 12,568 / 81,002 |
| Character Accuracy | 78.69 % | **78.69 %** |
| Word Accuracy | 40.30 % | **40.30 %** |
| CER | 0.2131 | **0.213117** |

Not close — identical. Any future rebuild is checked the same way: score
`kraken-medieval-german-v2` and expect 882,255 characters and 188,022 errors. A
rebuild that misses those numbers is a different set and must not carry this name.

### Digests of the 29.09.2026 build

```
b96679b559cdc861f1ec30cab1e1c2bc2f295c21cb222ff07ba1ad98ddd5f93f  german_test.arrow
c69e74941589207db714c3a111210d64e596f073073646dd14b1077c12c16d7b  german_val.arrow
```

Compiled on asteraix with kraken **7.0.2** (`.venvs/kraken-train`), repo at
`fad3f89`; `compile_cmd` is unchanged between that commit and current main, so
the argv above is the one that ran.

A sweep manifest pins the set by this digest and refuses to load without one
(`atr_training.sweep_manifest`, #113):

```yaml
data:
  train: /home/tobias/atr-cache/arrows/sweep_train.arrow
  eval: german_test
  digest: sha256:b96679b559cdc861f1ec30cab1e1c2bc2f295c21cb222ff07ba1ad98ddd5f93f
```

The digest is folded into every configuration's `config_id`, so the same
hyperparameters measured on a different corpus cannot share a leaderboard row
with these.

### The reference number, and what it is not

**CER 0.2131** is the only trustworthy German CER a Kraken model in this project
has. It is not comparable with the VLM numbers (`qwen3vl-medieval-german-v3` at
0.111 on medieval held-out lines, the 19th-century models at 0.0765 / 0.0680 on
the Bundesprotokoll benchmark) — different sets, different units. Producing one
comparable number is part of #111, not an assumption of it.

Two caveats that travel with the number: the run was stopped by hand at epoch 44
before its own test stage ran, and insertions outnumber deletions 7.5 : 1 under
this project's convention — where an insertion is a reference character the
hypothesis did *not* produce. So the model omits rather than over-generates, and
had not finished converging when it was stopped.

### Checking the direction instead of trusting it

That convention is the inverse of the usual ASR one, so it is the kind of thing a
reasonable reader "corrects" and thereby breaks. It never has to be taken on
trust, because it implies an identity:

    hypothesis_chars == chars - insertions + deletions

`ketos_cmd` derives kraken's `length_ratio` from it, and
`tests/test_edit_convention.py` pins it at both the pair and the corpus level
(#55). **Recompute it before building on any statement about the direction** —
three such statements in this repository were backwards, and the identity caught
all three on numbers already published in them:

- `Score`'s own docstring said a model that fails to stop produces insertions
  (it produces deletions) — `8126cb9`.
- `docs/BASE_MODEL_LADDER.md` said the Gemma 12B arm "invents 15 % of the text",
  while 114 960 − 17 027 + 4 783 = 102 716 reproduces the `length_ratio` of
  0.894 printed two tables above it: the arm omits 10.7 %.
- `atr_training.eval_material` was built on serving-atr-inference#52's reading
  that every model "outputs substantially more characters than the reference".
  The same issue's numbers give 11 566 − 11 191 + 2 = **377** characters of
  hypothesis against a reference of 11 566 — a blank collapse, the opposite
  direction — so the module attributed the symptom to the wrong tail of its own
  distribution and did not count the right one at all (`fa02f2d`).

### Measuring a model on it

A kraken run can score it itself, as a benchmark beside its own validation split
(#124):

```json
{"params": {"benchmarks": [{"hf_repo": "dh-unibe/image-text_…",
                            "project": "…",
                            "label": "german-medieval-v1"}]}}
```

That fills `benchmark_cer`, `benchmark_wer` and `measured_on` on the job record,
and the model card renders them. Before #124 the benchmark stage existed only in
the VLM backend, which is why the 0.2131 above had to be measured by hand and its
`measured_on` written in by hand with it.

The run **refuses** rather than scores if the benchmark's documents appear in the
training corpus: a CER measured there is a memory test, not an error rate. The
check is by document, not by page, because pages of one manuscript share a hand.

#### A finished VLM page arm, without retraining (#301)

A VLM page arm reads a draw: one JSON sample per page, not an arrow. The set's
test pages become one, once, and outside the jobs tree:

```bash
POOL=<jobs root>/20260905T190759Z-kraken-german-eval-pool-v1/data
OUT=<share>/eval_sets/german-medieval-v1/vlm-pages-test
python scripts/build_page_benchmark.py --set german-medieval-v1 --side test \
    --pages-list $POOL/split/pages_test.lst --pages-dir $POOL/pages --out $OUT
```

It **refuses** unless the list's documents are the registry's 200 test documents
and its page digest is the recorded `7a3f839ff97da4fd`. Then it copies the XML
and JPEG of every page to `$OUT/data/pages/`, writes `$OUT/pages.jsonl`, and
writes `$OUT/benchmark.json` with what it checked, how many samples and reference
characters there are, and the draw's md5. A page without text is named there,
not silently dropped.

Scoring an arm on it:

```bash
JOB=<id> DRAW_FILE=$OUT/pages.jsonl DATA_ROOT=$OUT LABEL=german-medieval-v1-test \
    HELDOUT=1 ubelix/submit.sh ubelix/rescore_on_shared_draw.sbatch -- --time=<…>
```

`HELDOUT=1` (`--heldout`) compares the draw's documents with the run's training
manifest **before** scoring, and refuses on one shared document. A run that
adopted its corpus has no manifest of its own; the check reads the cache entry's,
then the builder's. When none is left, or nothing on one side names a document,
it refuses too; `--allow-unchecked` overrides and says so. The outcome is kept
beside the report as `eval_report.<label>.heldout.json`. The score itself goes to
`eval_report.<label>.json`; the job record is not touched.

**Training manifests outlive their job directories here, not on scratch.** The
check needs the manifest of the corpus a run trained on, and for an adopting run
that is often only in the builder's job directory, which scratch purges after 30
days. Copies are kept under
`<share>/archive/train-manifests/<builder job id>/` (the builder's `data/*.lst`,
its `job.json` and `SHA256SUMS`), and `TRAIN_MANIFEST=<that>/pages_train.lst`
(`--train-manifest`) points the check at one. The sidecar then records
`train_manifest_given: true`. The first to be kept there is
`20261001T093909Z-olmocr2-7b-medieval-german-page-v1`, the builder of
`2ad500ee8414`, on which every medieval olmOCR page arm trained (8,104 pages,
1,180 documents; #301 records when the copy was made and its checksums).

### Keeping it out of training

The 350 document ids are reserved in `config/heldout_eval_documents.json`, and
`_prepare` drops their pages from the training manifest of every run
(serving-atr-inference#98). It has fired in earnest: job
`20260916T041001Z-qwen3vl-german-pages-v5` recorded `reserved_pages: 3187`.
Beware the ambiguity `reserved_pages: 0` still carries — on a run that adopted a
cached artefact it means *prepare was skipped*, not *nothing was reserved*
(#119).

---

## The 19th-century corpus, measured before it was trained on (#164)

Not an evaluation set — the geometry of the **training** corpora of
`qwen3vl-german-xix-v2`, measured with
[`scripts/measure_corpus_geometry.py`](../scripts/measure_corpus_geometry.py) on
05.10.2026 so a kraken run on the same data could be planned against numbers
rather than against the medieval corpus's habits.

| corpus | pages | transcribed lines | untranscribed | box aspect median / p99 / max | over 60:1 | `aspect_per_char` p10 |
|---|---:|---:|---:|---|---:|---:|
| `nr-sr-vereinigte-bundesversammlung-xix` | 60 | 4,097 | 1 | 10.1 / 17.6 / 21.7 | 0 | 0.2034 |
| `parlamentsdienste-protokolle` | 120 | 4,206 | 10 | 10.3 / 18.0 / 24.0 | 0 | 0.1613 |
| `zh-regierungsratsprotokolle` | 300 | 9,235 | **5,719** | 6.3 / 9.1 / 12.1 | 0 | 0.1778 |
| `kurrent-xix` (the v2 `TRAIN_*` projects) | 300 | 6,316 | 203 | 9.8 / 32.3 / **58.9** | 0 | 0.1719 |

**The 60:1 ceiling does not bind.** Not on any of the four, against 215.6:1 on the
medieval sweep corpus where it cut 11,318 lines (#145, #113). `kurrent-xix` is the
one to watch: its p99 is 32.3 and its widest box 58.9, and #145 is open precisely
because kraken pads the **extracted** line image, which on the medieval corpus
reached 177:1 where the box said 60. A box tail under the ceiling is therefore not
a verdict — but a box tail at 12:1, as `zh-regierungsratsprotokolle` has, leaves no
room for one.

**These hands are denser than the medieval ones**, and that decides the
architecture before the first run. `aspect_per_char` is width/(height × characters)
at the p10, scale-free (#91):

| material | `aspect_per_char` p10 | kraken+ (height 64, stride 8) | kraken default (height 120, stride 4) |
|---|---:|---|---|
| medieval (the project's reference) | 0.246 | 1.97 frames/char — **warn** | 3.69 — ok |
| `nr-sr…` | 0.2034 | 1.63 — **warn** | 3.05 — ok |
| `zh-regierungsratsprotokolle` | 0.1778 | 1.42 — **warn** | 2.67 — ok |
| `kurrent-xix` | 0.1719 | 1.38 — **warn** | 2.58 — ok |
| `parlamentsdienste-protokolle` | 0.1613 | 1.29 — **warn** | 2.42 — ok |

**Correction, 08.10.2026.** An earlier version of this section said "all five
completed kraken runs here use the kraken+ spec". That was read off the `spec`
field of their job records, and in four of the five it had no effect: `ketos
train` **ignores `--spec` when `--load` is given**, so a fine-tune inherits its
base's architecture. Every base this project has fine-tuned from is height 120 —
measured on the weights, not on the registry:

| run | base | height | own split CER |
|---|---|---:|---:|
| `kraken-medieval-scripts-v1` | none, **from scratch** | **64** | 0.7074 |
| `kraken-thun-finetune-v1` | `10.5281/zenodo.15366732` (Bifrost) | 120 | 0.3921 |
| `kraken-thun-kurrent-v1` | `kraken-early_modern_german` (CATMuS Medieval) | 120 | 0.2350 |
| `kraken-thun-kurrent-v2` | the same | 120 | 0.2180 |
| `kraken-corpus-thun-ft-v1` | a checkpoint of `kraken-medieval-german-v2` | 120 | 0.2054 |

So the height-64 architecture the guard warns about has been used **once**, by the
only from-scratch run, and that run is also the worst by a factor of three. The
fine-tunes were never exposed to the geometry problem; a from-scratch run on this
material would be. Those four CERs are **not** an architecture comparison either —
they differ in base and in data.

The only controlled architecture comparison the project has is
`docs/KRAKEN_PLUS.md` in the serving repo: three runs on one corpus
(`shard_00`), one validation set, one held-out test set of 6,186 pages in 35
unseen documents.

| run | spec | height | val acc | test CER | word acc |
|---|---|---:|---:|---:|---:|
| run 2 | without the final layer | 64 | 0.7809 | 0.1812 | 51.8 % |
| kraken+ | `Cr1,1,85,1,1` | 64 | 0.7927 | 0.1655 | 55.0 % |
| **run 3** | kraken default | **120** | **0.8226** | **0.1335** | **62.5 %** |

**The input height is what separates them** — 0.032 CER between run 3 and the
better of the two height-64 runs, against 0.0157 between the two height-64 runs
themselves — and it matches the frames-per-character arithmetic above (1.97
against 3.69 on that material). The 85-channel final layer helps slightly rather
than bottlenecking.

That comparison is about the architecture and not about German: all three trained
and scored on the Flemish `medieval-scripts` corpus, so their absolute numbers
transfer to nothing.

**`zh-regierungsratsprotokolle` is 38 % untranscribed** (5,719 of 14,954
`TextLine`s in the sample). `ketos compile --skip-empty-lines` drops them, so its
usable yield is ~31 transcribed lines per page, not the ~50 the page count
suggests. A `max_pages` chosen from page counts alone over-promises here.

## The XVI–XIX page arms: three ways to a held-out number (#301, part B)

The XVI–XIX page corpus (20 datasets) has no held-out set of its own. Its
`reserved_pages: 1318` only means the pages of `german-medieval-v1` documents
were taken out of training. #301 names three ways, and they measure three
different things. None of them stands in for another.

| way | what it measures | price |
|---|---|---|
| a | the early edge only: `german-medieval-v1`'s documents come from the four datasets that reach into the XVI century | none, the arms are rescored |
| b | the XIX century on the Federal Council lines, a page model read on **lines** | none, but not comparable with a page CER |
| c | the target, Kurrent pages held out by document | every XVI–XIX **and** XIX arm must be trained again |

### a) `german-medieval-v1/test`, the draw that already exists

The two XVI–XIX arms have no report on it yet (26.10.2026). `--heldout`
checks their training manifests against the draw's 200 documents first. They
should be clean, because these documents were reserved before the corpus was
built, but the check is what says so.

```bash
D=/storage/research/wbkolleg_dh_1/Textrecognition_Training/eval_sets/german-medieval-v1/vlm-pages-test
for JOB in 20261019T180118Z-olmocr2-7b-german-kurrent-xvi-xix-page-px4m-native \
           20261013T100501Z-qwen3.5-4b-german-kurrent-xvi-xix-page-v1; do
  JOB=$JOB DRAW_FILE=$D/pages.jsonl DATA_ROOT=$D LABEL=german-medieval-v1-test HELDOUT=1 \
    ubelix/submit.sh ubelix/rescore_on_shared_draw.sbatch -- --time=08:00:00
done
```

Compare the result with the medieval arms on the same 695 pages
(`docs/HELDOUT_BOUNDARY.md`), not with the XVI–XIX validation CER.

**Measured 27.10.2026** (Slurm 20187794/20187795, both `checked: true`,
`shared: []`): XVI–XIX `px4m-native` **0.3681** (`truncated_cer` 0.3354, 32
pages at the token cap), XVI–XIX Qwen 3.5 4B **0.5405** (`truncated_cer`
0.4168, 99 at the cap). The medieval `px4m-native` reads the same pages at
0.3674. The table and what it does and does not say:
[HELDOUT_BOUNDARY.md](HELDOUT_BOUNDARY.md#the-first-held-out-numbers-301).

### b) The Federal Council lines

`score_federal_minutes.sbatch` asks every adapter with the **line** prompt at the
line pixel budget, which is right for a line arm and wrong for a page arm. The
olmOCR arm was trained on the olmOCR page prompt. `rescore_on_shared_draw.py`
keeps each run's own prompt and budget, so b goes through it, with the
Federal Council draw as a file:

```bash
E=/scratch/network/users/$USER/runs/jobs/evalset-federal-minutes
for JOB in 20261019T180118Z-olmocr2-7b-german-kurrent-xvi-xix-page-px4m-native \
           20261013T100501Z-qwen3.5-4b-german-kurrent-xvi-xix-page-v1; do
  JOB=$JOB DRAW_FILE=$E/data/val.jsonl DATA_ROOT=$E LABEL=federal-minutes-lines \
    ubelix/submit.sh ubelix/rescore_on_shared_draw.sbatch -- --time=04:00:00
done
```

Without `--heldout`: the 2,751 lines come from Zenodo (Hodel & Schoch 2021,
doi:10.5281/zenodo.4746342), not from a dh-unibe dataset, so no manifest names
their documents. Two of the 20 training datasets are Swiss federal records
(`parlamentsdienste-protokolle`, `nr-sr-vereinigte-bundesversammlung-xix`).
Nothing has shown that they share no pages with the Council minutes. Until
something does, the number is a reading on known material, not a held-out
number.

**Measured 27.10.2026** (Slurm 20187796/20187797, `heldout: null`, as it has
to be):

| arm | CER | `truncated_cer` | `length_ratio` | at the token cap |
|---|---:|---:|---:|---:|
| XVI–XIX `px4m-native` | 0.2848 | 0.2024 | 1.129 | 2 of 2,751 |
| XVI–XIX `qwen3.5-4b-…-page-v1` | 0.4363 | 0.2396 | 1.250 | 12 of 2,751 |

Both write more than the line holds (`length_ratio` above 1), and
`truncated_cer` lies far closer together than CER: most of the distance between
the arms is text past the line, not misread characters. A page model on lines,
on material that may be known — a ranking, not a held-out CER.

### c) A Kurrent page set of its own

Drawn by document from the corpus the XVI–XIX arms trained on, then
registered. From then on the trainer drops those documents from every run.

**Started 27.10.2026.** Before step 1, the checkout on UBELIX has to be on
`origin/main` (`submit.sh` refuses otherwise, and the scripts below are
read from it): `cd ~/training-atr-models && git pull`.

1. **Draw** (UBELIX, login node). The pool is the page manifests of a run that
   built the XVI–XIX page corpus with its own manifest
   (`20261013T100501Z-qwen3.5-4b-german-kurrent-xvi-xix-page-v1`, 13,606
   entries, `docs/HELDOUT_BOUNDARY.md`). Its job directory is on scratch, and
   **scratch purges after 30 days**, so this has to happen before mid-November
   2026, or the corpus has to be built again.

   ```bash
   J=/scratch/network/users/$USER/runs/jobs/20261013T100501Z-qwen3.5-4b-german-kurrent-xvi-xix-page-v1
   OUT=/storage/research/wbkolleg_dh_1/Textrecognition_Training/eval_sets/german-kurrent-xvi-xix-v1/split
   python3 scripts/make_split.py $J/data --out $OUT --seed 20261026 \
       --test-docs 100 --val-docs 60 --test-per-doc 220 --val-per-doc 160
   ```

   The printed record gives the page counts. `german-medieval-v1/test` has 695
   pages. If this draw is far off, change `--test-docs` and draw again; the
   seed keeps every draw reproducible. `make_split.py` now writes all six
   parameters into `split.json`.

   **First draw, 27.10.2026:** a pool of 15,315 pages in 1,483 documents; test
   406 pages from 100 documents (about four per document, none dropped), val
   587 from 60 (207 dropped at the cap). The registration was **refused**:
   `german-medieval-v1` document `034854` sat on the corpus's validation
   side — the trainer only takes reserved documents out of *train* — and the
   draw had put it into the new set. Since then `make_split.py` cuts the pool
   by the registry before drawing (`--registry`, `--exclude-sets`, default
   every registered set) and records the sets by name in `split.json`, so the
   draw repeats after the new set has joined the registry. Draw again with the
   same command; at about four pages per document, `--test-docs 170` would
   bring the test side near the medieval set's 695 pages, at the price of 70
   more documents leaving training.

   **Second draw, 27.10.2026 (170 test documents):** the cut removed **26**
   `german-medieval-v1` documents (284 pages), all on the corpus's validation
   side; test 656 pages, val 639. Not registered either: **48 of the 230
   "documents" were not documents.** The six imported datasets are
   materialised as `<counter>_<their own file name>`, so field 1 is whatever
   that name starts with:

   | field 1 | example page | a document? |
   |---|---|---|
   | a Transkribus id | `…_018187_0004_…` | yes |
   | a work's name | `018786_PPN643815198_0002.xml`, `018204_agtck_1834_02_00002.xml` | yes, coarse |
   | the page's own file name | `018426_00000017.xml`, `019000_813172837-0007.xml` | **no** — page 17 of every book |
   | a page number | `018821_11_7175e_default.xml` | **no** |
   | a year | `018116_1871_104_0417.xml` | not known |

   In the whole pool 327 of 1,483 "documents" (4,156 of 15,315 pages) are of
   the last three kinds. Holding them out would hold out pages, not hands, and
   registering them would make the trainer drop page 17 of every such book
   from every later run. `make_split.py --documents transkribus-or-named` (the
   default since) draws only Transkribus ids and names; the other pages stay
   on the training side, counted under `undrawable` in `split.json`. A
   held-out number for those six datasets needs a document key from their
   source, not from the materialised name. `restore_eval_split.py` replays
   both the cut and the rule from the registry entry.

   **Registered, 27.10.2026, third draw:** with the rule, 170 documents came to
   1,710 test pages (a real document averages ten pages, the page-named groups
   one to five), so the size was cut to **70 test documents: 733 pages**
   (141 dropped at the cap) and **60 val documents: 509 pages** (257 dropped).
   Train keeps 13,391 of 15,031 pages; 310 page-named groups (699 pages) stay on
   it as `undrawable`. The entry in `config/heldout_eval_documents.json` is the
   one `register_eval_set.py` wrote on UBELIX into a copy of the registry
   (`page_digests` test `22814f9ebb7b58cc`, val `d7e8b9a5bab890ea`), taken into
   the repository by PR.

   **What it costs, from the merge on.** The trainer drops these 130
   documents from every run whose datasets hold them — the XVI–XIX corpus and
   also the medieval one, which shares `aaeb-xiv-xvii`, `bullinger-autoren`,
   the Königsfelden sets and `rats-und-richtebuecher_xv-xvi`. Every XIX,
   XVI–XIX and medieval model trained before reads `crossed` against this set
   in the held-out audit if it trained on one of them. And because the
   artefact cache key fingerprints every reserved document, every compiled
   corpus is built again on its next use; the two medieval sweep manifests
   carry the new digest.

2. **Register.** `scripts/register_eval_set.py` turns the split into a
   registry entry. It refuses a taken name, missing parameters, a leak into
   train, a document on both sides, or a document another set holds. Without
   `--write` it only prints.

   ```bash
   # The 20 datasets, read from the run's own request rather than typed.
   DS=$(python3 -c "import json,sys; print(' '.join(d['hf_repo'] for d in json.load(open(sys.argv[1]))['request']['datasets']))" $J/job.json)
   # register_eval_set.py imports the trainer (loguru); the login node's
   # python3 has none, the image has it.
   apptainer exec --bind /storage/research --bind /scratch \
     --env PYTHONPATH=$PWD/src ~/ubelix/vlm-train.sif \
     python scripts/register_eval_set.py $OUT --name german-kurrent-xvi-xix-v1 \
       --source-job 20261013T100501Z-qwen3.5-4b-german-kurrent-xvi-xix-page-v1 \
       --datasets $DS
   ```

   `--write` is the step that costs something. These documents leave the
   training side of **every** later run in every backend, and the XIX corpora
   share their datasets (`zh-regierungsratsprotokolle`, `kurrent-xix`, the
   two federal ones). Every XIX and XVI–XIX model trained before the
   registration reads `crossed` against this set in the held-out audit. The
   entry belongs in the repository through a PR, not as a dirty checkout on
   UBELIX (`submit.sh` refuses one).

3. **Freeze the draw**, as for the medieval set. It imports the trainer, so it
   runs in the image — and needs `/rs_scratch` bound as well: on UBELIX
   `/scratch/network/users/$USER` resolves there, the page lists name
   `/rs_scratch/…`, and without the bind every page reads as missing (the
   first attempt on 28.10. refused all 733 that way).

   ```bash
   J=/scratch/network/users/$USER/runs/jobs/20261013T100501Z-qwen3.5-4b-german-kurrent-xvi-xix-page-v1
   OUT=/storage/research/wbkolleg_dh_1/Textrecognition_Training/eval_sets/german-kurrent-xvi-xix-v1
   for SIDE in test val; do
     apptainer exec --bind /storage/research --bind /scratch --bind /rs_scratch \
       --env PYTHONPATH=$PWD/src ~/ubelix/vlm-train.sif \
       python scripts/build_page_benchmark.py --set german-kurrent-xvi-xix-v1 --side $SIDE \
         --pages-list $OUT/split/pages_$SIDE.lst --pages-dir $J/data/pages \
         --out $OUT/vlm-pages-$SIDE
   done
   ```

   **Frozen, 28.10.2026**, under
   `eval_sets/german-kurrent-xvi-xix-v1/vlm-pages-{test,val}/`, so the draw
   no longer depends on the scratch job directory:

   | side | documents | pages = samples | reference characters | page digest | `draw_md5` |
   |---|---:|---:|---:|---|---|
   | test | 70 | 733 | 977,707 | `22814f9ebb7b58cc` | `ae8bf1fd315e3a06eee272bb3ca7c7ad` |
   | val | 60 | 509 | 760,695 | `d7e8b9a5bab890ea` | `4ade0622afbfe4e5a836d63772d6f3ad` |

   Every page carries a transcription, and both digests are the registry's.

4. **Train again, then measure** with `rescore_on_shared_draw.py --heldout`,
   `DRAW_FILE=$OUT/vlm-pages-test/pages.jsonl DATA_ROOT=$OUT/vlm-pages-test
   LABEL=german-kurrent-xvi-xix-v1-test`. The two XVI–XIX page arms of
   October trained on these documents and read `crossed`; only a run whose
   corpus was built after #319 can be measured here.

   The two arms again (#341):
   `ubelix/specs/olmocr2-7b-german-kurrent-xvi-xix-page-px4m-native-v2.json` and
   `ubelix/specs/qwen3.5-4b-german-kurrent-xvi-xix-page-v2.json`, the October
   recipes with a new `model_id` and `max_new_tokens` 3072. Both describe the
   same corpus, so one prepare builds it and the other adopts it: submit the
   second prepare only after the first has finished, or both build. Between
   prepare and train, one check per run, minutes on the login node instead of
   GPU days later:

   ```bash
   apptainer exec --bind /storage/research --bind /scratch --bind /rs_scratch \
     --env PYTHONPATH=$PWD/src ~/ubelix/vlm-train.sif \
     python scripts/heldout_boundary.py --job /scratch/network/users/$USER/runs/jobs/<id>
   ```

   It applies today's registry to the run's `pages_train.lst` (or its
   builder's, for the adopting arm) and exits 0 only on `clean`; `crossed`,
   `uncheckable` and `gone` are all reasons not to submit the GPU stage.
   `reserved_pages` in the record should have grown past the October corpus's
   1318, by the pages of the 130 new documents.

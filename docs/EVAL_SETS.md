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

All five completed kraken runs here use the kraken+ spec, which the guard warns
about on *every* one of these corpora — and on the medieval one too. The best
kraken CER the project has, **0.1335** (run 3), came from the height-120
architecture (`docs/KRAKEN_PLUS.md` §run 3 in the serving repo). That comparison
is about the architecture and not about German: run 3 trained and scored on the
Flemish `medieval-scripts` corpus, so its absolute number transfers to nothing.

**`zh-regierungsratsprotokolle` is 38 % untranscribed** (5,719 of 14,954
`TextLine`s in the sample). `ketos compile --skip-empty-lines` drops them, so its
usable yield is ~31 transcribed lines per page, not the ~50 the page count
suggests. A `max_pages` chosen from page counts alone over-promises here.

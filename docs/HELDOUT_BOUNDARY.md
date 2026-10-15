# Which published numbers cross the held-out boundary?

The held-out boundary (#98, `docs/BASE_MODEL_LADDER.md` §1f) is proven for the
medieval ladder: the September corpus held 306,582 training lines, the October
one 219,261, because `_reserve_eval_documents` keeps 3,173 of 11,277 pages for
`german-medieval-v1`. **Which other published numbers cross the same boundary is
not yet known.** This page is where the answer goes (#247).

It is written before the measurement because the measurement needs the job
stores, and the stores are on the research share and in UBELIX scratch. What is
settled here is the method, the states, and two findings that change how the
existing records must be read.

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

So **the boundary is not enforced for chunked runs**, and the record said `0`.
Latent today — `chunk_pages` defaults to `0`, chunking is kraken-only and needs
explicit `eval_projects` — but armed: it is the corpus-scale run that turns
chunking on, and that is the run the boundary matters most for. The pages exist
only inside the compile loop, so enforcement belongs there; until it is built,
the record says `not checked: chunked prepare defers the train side to compile`
and a warning is logged. **Enforcement is #255.**

## The method

For each of the 54 runs: apply **today's** registry
(`config/heldout_eval_documents.json`) to the manifest the run actually trained
on, and count how many pages *would* fall. No GPU, no network, minutes.

```
python scripts/heldout_boundary.py --markdown > docs/HELDOUT_BOUNDARY.md
python scripts/heldout_boundary.py --json | jq '.summary'
```

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

## Results

Not yet measured. The run needs a box that can read the job stores; the script
refuses to produce a table from stores it cannot see, and names each one it
could not read:

```
store share (/storage/research/.../training_folder/jobs) is not readable from
here — its runs are absent from this table, not clean

0 Läufe: 0 unberührt, 0 betroffen, 0 nicht prüfbar, 0 unentscheidbar.
```

| Lauf | Store | CER | `reserved_pages` | Verdikt | gefallen | Grund |
|---|---|---:|---:|---|---:|---|
| _(pending the run on asteraix)_ | | | | | | |

## Still open

* **The 171 pages.** The three medieval September runs each carry 11,448 pages
  in `data/pages_train.lst`, including `qwen3vl-medieval-german-v3`, the 4B
  reference of the §1b table — against 11,277 in October, *before* the 3,173 are
  deducted. Where the difference comes from is unanswered, and it is not
  answerable from the repository alone: it is a property of two materialisations.
* **The published page.** `docs/vlm-finetuning.md` in
  `thodel/agentic-historian-outputs` carries the affected numbers. Correcting it
  once with the full reach is better than correcting it twice — that is #248,
  and it waits for the table above.

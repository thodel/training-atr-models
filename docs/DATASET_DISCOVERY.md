# Dataset discovery: what changed in the training material

Every early medieval run drew on a dataset whose German content was about 291
pages, while 9,885 pages of Zurich council books sat unqueried in the same org
for a month. `scripts/plan_corpus.py` answers "what is the best corpus *now*"
when someone thinks to run it; nothing noticed when the answer changed. This is
the part that notices (#32).

## What runs, where

A systemd user timer on **asteraix**, daily at 06:40 Europe/Zurich (±10 min):

| file | role |
|---|---|
| `deploy/systemd/atr-discovery.timer` | the schedule; `Persistent=true`, so a pass missed while the box was down runs at the next boot |
| `deploy/systemd/atr-discovery.service` | one pass: `scripts/discover_datasets.py` in the `kraken-train` venv |
| `scripts/install_discovery_timer.sh` | installs both, checks the venv and that a Hub token exists (`--run-now` takes the baseline at once) |

It runs on asteraix because the dh-unibe datasets are **gated**: `hf auth login`
is done there and its token is what `huggingface_hub` reads. Nothing copies the
token anywhere. One pass over `dh-unibe` — 49 datasets, three Hub calls each —
took 29 s on 22.10.2026; the Hub allows 1,000 calls per five minutes.

## What it reports

Changes, never an inventory. Against the last pass it reports:

| kind | when |
|---|---|
| `new` | the Hub lists a dataset it did not list before (also when its card cannot be read yet) |
| `gone` | a dataset is no longer listed |
| `projects` | the card's *Projects Included* gained or lost entries |
| `size` | the page count moved by at least 100 pages **and** 5 % |
| `metadata` | period, languages, document type, or the card's statement that its text is machine output changed |
| `unreadable` | a dataset that was readable cannot be read this time — reported once, its last known state kept |

Each change carries its **consequence**, for every target profile (default the
medieval German one, 1300–1600; `--period FROM TO` adds more): the dataset's score
before and after, and where `plan_corpus` would put it — entering the plan at
which rank, leaving it, moving within it. For example, from a rehearsal on the
real catalogue with one card's period edited:

```
## `image-text_zh-regierungsratsprotokolle`
- **metadata** — period (1803, 1883) → (1500, 1600)
  - against 1300–1600: score 0.00 → 1.00; **would enter the plan at rank 1** (152,786 pages)
```

Scores and ranks are estimates from card metadata, exactly as in
`plan_corpus.py`; only `prepare` knows the real counts.

## Where it keeps its memory

`~/atr-cache/discovery/` on asteraix, local disk:

| file | content |
|---|---|
| `snapshot.json` | the last plausible pass, reduced to what the planner reads |
| `reports/YYYY-MM-DD.md` | written **only** on days something changed |
| `runs.jsonl` | one line per run: readable and unreadable counts, number of changes, note |

The first run records a baseline and reports nothing. An unchanged Hub writes no
report.

## Two refusals

**An unreadable card is not a deletion.** A dataset the Hub lists but whose card
cannot be read (a 503, a revoked token, no README) keeps its last known state.
It is reported as `unreadable` once, not as `gone` — and when it is readable
again, its projects are not reported as new.

**An outage is not remembered.** When the Hub lists fewer than half the datasets
of the last pass, or fewer than half the cards are readable, the run exits 1 and
keeps the old snapshot. The service then shows as failed
(`systemctl --user --failed`), which is the signal to look.

## Reading it

```bash
ls ~/atr-cache/discovery/reports/
tail -5 ~/atr-cache/discovery/runs.jsonl
systemctl --user list-timers atr-discovery.timer
journalctl --user -u atr-discovery.service -n 20 --no-pager
```

How the report reaches anyone beyond this box is serving-atr-inference#116
(`GET /train/discoveries`), not decided yet.

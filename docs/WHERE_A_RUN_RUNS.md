# Where a run runs — asteraix or UBELIX

The rule asked for in [#157](https://github.com/thodel/training-atr-models/issues/157),
for every engine: `vllm` (VLM QLoRA), `kraken`, `trocr`.

**UBELIX is the default. asteraix is the exception, and the exception is about
waiting, not about size.** That is the opposite of the intuition the question
started from, and §3 says why with the numbers.

Decided 05.10.2026: both A40s are usable for training and measurement — asteraix
is no longer held free for urgent work. Something urgent displaces a run by hand.

## 1. The rule

```
                     ┌─ > 44.42 GiB peak? ──── yes ──→ UBELIX, and only UBELIX
                     │                                 (one A40 cannot hold it;
                     │                                  two buy capacity, no speed)
  a run arrives ─────┤
                     │                     ┌─ a re-evaluation, or < ~2 h? ─ yes ─→ asteraix
                     └─ fits one A40? ─────┤
                                           │                      ┌─ > (r − 1) · T ──→ asteraix
                                           └─ UBELIX delay is … ──┤
                                                                  └─ ≤ (r − 1) · T ──→ UBELIX

  T = the run's expected H100 hours;  r = A40/H100 time ratio (§1, measured, #162)
```

Read top to bottom; the first branch that matches decides.

**1. Does it fit on one A40?** 44.42 GiB usable (measured, #137 — not the 46 068 MiB
nvidia-smi prints, and not the 92 GB the two-card epic assumed). If the peak does
not fit, the run goes to UBELIX and nothing below applies. Two A40s do hold
88.8 GiB, but only through `device_map="auto"`, which is naive model-parallel: it
distributes the peak and buys **no** speed (249.7 s against 247.6 s on the same
work, #137). Use that to make a run *possible*, never to make it faster.

The answer comes from the peak table below, and only from it (#163). **A run
whose point is missing goes to UBELIX**, and so does a run whose only point is a
lower bound — a lower bound may stand in the table, but it is not a limit. A
point is (engine, base, granularity, batch per device, quantisation); gradient
accumulation does not move it (21,466–22,486 MiB at accumulation 1, 2 and 4 below).

**The peak table.** The training stage's `own_mib`: what nvidia-smi showed
for the processes descending from the stage's pid, polled every 2 s
(`gpu.PeakSampler`, recorded on `stages[train].peak_gpu_mib` of every job since
07.10.2026). The caching allocator keeps what it reserved, so a spike shorter
than the poll is still seen. 45,486 MiB is what an A40 makes usable.

| Engine | Base | Gran. | Batch × acc. | Quant. | Peak own | Of an A40 | Card | Source | Job |
|---|---|---|---|---|---:|---:|---|---|---|
| vllm | Qwen3.5-4B | line | 2 × 8 | 4-bit | 9,954 | 21.9 % | A40 | **lower bound**: 400 steps, 6,400 of 221,523 lines | `20261017T114213Z-throughput-qwen35-4b-a40` |
| vllm | Qwen3-VL-4B | line | 16 × 1/2/4 | bf16 | 21,466–22,486 | 47.2–49.4 % | A40 | **lower bound**: 750–4,500 steps of 881,902 lines | `20261012T135706Z-vlm-xix-effective-batch-01-noise-s42` and 7 more, 12.–17.10. |
| vllm | Qwen3.5-4B | line | 16 × 1 | 4-bit | 34,742 | 76.4 % | H100 | full run, 221,523 lines | `20261014T194417Z-ladder-med-qwen35-4b-v2` |
| vllm | Qwen3.5-4B | line | 16 × 1 | bf16 | 36,414–36,498 | 80.1–80.2 % | H100 | full runs, 221,523 and 473,549 lines | `20261023T194329Z-ladder-med-qwen35-4b-bf16-v2`, `20261009T150138Z-qwen3.5-4b-german-kurrent-xvi-xix-v1` |
| vllm | Qwen3.5-9B | line | 16 × 1 | 4-bit | 44,412 | **97.6 %** | H100 | full run, 221,523 lines | `20261014T193118Z-ladder-med-qwen35-9b-v2` |
| vllm | Qwen3.5-27B, Qwen3.8-27B | line | 16 × 1 | 4-bit | 62,324 | **137 %** | H100 | full runs, 221,523 lines | `20261013T043436Z-ladder-med-qwen35-27b-v2`, `20261010T111013Z-ladder-med-qwen38-27b-v2` |
| vllm | Qwen3-VL-8B | page | 1 × 16 | bf16 | 27,763 | 61.0 % | A40 | **lower bound**: head of the file, 96 of 9,441 samples (#137) | hand-run |
| vllm | Qwen3-VL-8B | page | 1 × 16 | bf16 | 28,612 | 62.9 % | A40 | **lower bound**: 100 steps, 1,600 of 99,933 pages | `20261104T200338Z-throughput-qwen3vl-8b-page-a40` |
| vllm | Qwen3-VL-8B | page | 1 × 16 | bf16 | 29,116 | 64.0 % | H100 | **lower bound**: the same 100 steps | `20261026T205257Z-throughput-qwen3vl-8b-page-h100` |
| vllm | Qwen3.5-4B | page | 2 × 1 | bf16 | 49,916–50,054 | **109.7–110.0 %** | H100 | full runs, 415,545–511,909 lines | `20261013T100501Z-qwen3.5-4b-german-kurrent-xvi-xix-page-v1` and 2 more |
| vllm | olmOCR-2-7B, `max_pixels` 1288² | page | 2 × 8 | bf16 | 42,608 | 93.7 % | H100 | full run | `20261008T103636Z-olmocr2-7b-medieval-page-px1288` |
| vllm | olmOCR-2-7B, cap 3072 | page | 2 × 8 | bf16 | 49,542 | **109 %** | H100 | full run | `20261008T103636Z-olmocr2-7b-medieval-page-cap3072` |
| vllm | olmOCR-2-7B, px4m | page | 2 × 8 | bf16 | 63,728–65,558 | **140–144 %** | H100 | full runs | `20261013T121310Z-olmocr2-7b-medieval-page-px4m-native` and 5 more |
| kraken | `KRAKEN_PLUS_SPEC`, from scratch | line | 256 | — | 43,536 | **95.7 %** | A40 | full epoch, `german_val.arrow`, no ceiling (hand-run, 07.10.2026) | `scripts/measure_kraken_arms.py` |
| kraken | `KRAKEN_PLUS_SPEC`, from scratch | line | 256 | — | 8,894 | 19.6 % | A40 | full run, 197,083 lines, ceiling 60:1 (cut 4.8 %) | `20261027T072036Z-kraken-collapse-probe-131` |
| kraken | `KRAKEN_PLUS_SPEC`, from scratch | line | 128 | — | 3,970–4,046 | 8.9 % | RTX 4090 (UBELIX) | full runs, 1,786 lines, widest 22:1 | `20261105T193111Z-kraken-bern-hertwig-scratch-ubelix-v2`, `20261106T064743Z-…-v3` |
| kraken | fine-tune of `kraken-german-xix-v1` | line | 32 | — | 18,400 | 40.5 % | A40 | full run, 38,360 lines, ceiling as dry run only | `20261026T193646Z-kraken-german-xix-blb-v1` |
| kraken | fine-tune of `cursive_inzigkofen` and its offspring | line | 16 | — | 3,952–6,930 | 8.7–15.2 % | A40 | full runs, 1,786–4,897 lines | `20261022T110104Z-kraken-bern-hertwig-v1` and 3 more |
| trocr | `trocr-base-handwritten`, and `trocr-kurrent-XVI-XVII` built on it | line | 4 or 8 | bf16 | 6,830–6,832 | 15.0 % | A40 | full runs, 4,005 to 473,549 lines | `20261024T213142Z-trocr-german-kurrent-xvi-xix-v1` and 5 more |

How to read it:

- **Only a full run says "fits".** A step-bounded run saw a shuffled draw, and
  the first peak #137 ever measured came from the head of the file; both are
  floors. The three Qwen3-VL-8B page rows agree within 1,353 MiB, which says
  the floor is stable, not that the tail of 99,933 pages stays under it.
- **An H100 figure transfers below the line, not above it.** The same 100 steps
  peaked at 29,116 MiB on the H100 and 28,612 MiB on the A40, 1.8 % apart. A
  full run that stayed under 45,486 on an H100 fits an A40. One that went over
  does not fit *as measured*: on an 80 GB card the allocator is never under
  pressure, so it may have held cache an A40 would have had to give back. Such a
  run goes to UBELIX until an A40 says otherwise.
- **A row closer to the line than that 1.8 % does not fit.** Qwen3.5-9B lines in
  4-bit (97.6 %, and 98.2 % for the whole card) is that row.
- **What drives the peak is batch per device and pixels, not the corpus.**
  Qwen3.5-4B lines in 4-bit: 9,954 MiB at batch 2 against 34,742 at batch 16.
  olmOCR-2-7B pages at the same batch: 42,608, 49,542 and up to 65,558 MiB as the
  pixel budget grows. kraken pads a batch to its widest line, which is why the
  60:1 ceiling of #145 matters. The corpora differ, so the two batch-256 rows
  are not a clean before/after, but they are a factor of five apart. trocr's
  processor resizes every line image to one fixed shape (384 × 384): 6,830 MiB over 53,008 lines and over
  473,549, at batch 4 and at batch 8.
- **Not a row: what the card held.** `card_mib` adds everything else on the card.
  The first A40 throughput run (`20261026T205256Z`) died at 26,950 MiB of its own
  (`torch.OutOfMemoryError`) because a kraken job held 18,400 MiB on the same
  card; that was the
  card-assignment bug fixed in 8b4c2b9, not a peak.

**2. Is it a measurement rather than a training?** A re-evaluation is generation
with no optimizer steps: minutes to about two hours, run repeatedly, and usually
wanted the same day. Those go to asteraix, where there is no queue. This is the
branch that would have saved the two rescoring jobs that died on a walltime
guess (`17310472`, `17310473`, fixed in #155 after two lost allocations) — on
asteraix there would have been no walltime to guess.

**3. Otherwise: how long is the UBELIX delay, against what the A40 costs?**
`squeue --start` gives an estimate of the wait, and the preemption record gives
the rest (§4). A run that needs **T** hours on an H100 needs **r · T** on an A40,
so asteraix finishes first only when the expected UBELIX delay exceeds
**(r − 1) · T**. Below that, UBELIX — the H100 is the faster card and the queue is
the price. Above it, asteraix, if the run fits.

r is measured on identical work, one run on each card (#162):

| load | H100 | A40 | r | asteraix wins above |
|---|---:|---:|---:|---|
| Qwen3.5-4B, lines, 4-bit, batch 2 × 8 (point 1, 17.10.2026) | 4.37 s/step | 6.61 s/step | **1.51** | 0.51 · T |
| Qwen3-VL-8B, pages, bf16, batch 1 × 16 (point 2, 04.11.2026) | 11.28 s/step | 41.20 s/step | **3.65** | 2.65 · T |

Worked through: the 70-H100-hour line run of §4 would take ~106 h on an A40, so
it belonged on asteraix once the delay passed **36 h** — an estimate, since that
run was a Gemma 4 12B and the line row is a 4B. A page run of 10 H100-hours
takes ~36.5 h on an A40 and belongs there only past **26.5 h** of delay; a
40-hour page run never does in practice (106 h of delay).

The two points differ in three things at once — granularity, model size and
quantisation (4-bit under transformers 5.17 against bf16 under 4.57.6) — so two
points give two ratios for these two loads, not a curve. For anything in between,
use the nearer row and say which; for a page run, the A40 is the much slower card.

What the size costs, so the choice is priced and not only placed: on the medieval
line ladder **a CER point costs about 0.3 kWh up to 9B and 3.5 kWh above it**
(4B → 9B: +0.60 kWh for −2.19 points; 9B → 27B: +5.63 kWh for −1.60; measured
records, `FOOTPRINT.md` §"What a CER point costs", #333). That does not move the
rule above — #138 settled that a footprint prices a default and does not shift it
— but a 27B arm asked for "because it is better" is asked for at thirteen times the
energy per point of the 9B step.

### Why the line is a delay and not a queue position

Until #162 measured r the line was a judgement: **48 hours**. It was close for
lines (36 h for the §4 run) and wrong for most page runs: 48 hours of delay
justify an A40 only for a page run under ~18 H100-hours (2.65 · 18 ≈ 48).

What carries over is why the delay is counted in wall-clock: a chunked run pays
the wait more than once. A run longer than the 24 h
preemptable wall is interrupted *by design*, and each interruption costs a fresh
wait: the live case in §4 paid 17 h 11 and 9 h 43 between its three attempts. One
day of queue is the normal cost of a free H100. Two days means the run is being
scheduled against, not scheduled.

## 2. What the two places are

| | asteraix | UBELIX |
|---|---|---|
| cards | 2 × A40, **44.42 GiB usable** each | H100 80 GB |
| how many at once | both, no queue | `job_gratis` **one**; `job_gpu_preemptable` four |
| wall limit | none | 96 h on `job_gratis`, **24 h** preemptable |
| interruption | none | preemption at any time on the preemptable QoS |
| CPU cap | none | `job_gratis`: **cpus × minutes ≤ 11 520**, over RUNNING jobs and their *remaining* time, GPU jobs included. A pending job is freed in place with `scontrol update JobId=<id> TimeLimit=<hh:mm:ss>`, which keeps the id the train stage depends on (#212) |
| job store | **none** — `find ~/atr-cache -name job.json` finds nothing, so a run there appears in no report (#156) | the shared store on the research share |
| state 05.10. 14:53 CEST | both cards **0 MiB, 0 %**; last checkpoint 16.09., last measurement 03.10. | one run, third attempt, 91.2 % after three days |

Parallelism, measured in the `vlm-train` venv rather than read from a library's
documentation (#137): `device_map="auto"` works, FSDP is installed but needs a
launcher and is unmeasured, DeepSpeed ZeRO-3 is **not installed**, tensor-parallel
has no plan for our model classes. There is no DDP anywhere. **One run uses one
card**, except when a model does not fit on one.

## 3. Size pushes a run to UBELIX, not away from it

The question in #157 assumed that size is what sends a run to asteraix. The
measurements say the reverse, and it is worth stating plainly because the whole
rule hangs on it:

- One A40 holds **44.42 GiB**. One H100 holds **80 GB**. The H100 is the bigger
  card by 1.8×, and `job_gratis` grants exactly one of them per user.
- asteraix exceeds an H100 only by using **both** cards — 88.8 GiB — and only for
  capacity, not speed.
- That leaves a band from 80 to 88.8 GiB where asteraix can do something UBELIX
  cannot. **No model we have is in it.** A 27B in bf16 is ~56 GB and a 31B ~62 GB;
  both fit one H100. At 4-bit a 27B is ~16 GB and fits one A40.

So "asteraix when the size requires it" describes a case that does not currently
occur. The rule above therefore reads size the other way: **asteraix when the size
permits it** and the queue is the real cost. If a model ever needs more than 80 GB,
§1's first branch sends it to UBELIX and the two-card path on asteraix becomes the
only option — at which point #12 stops being optional.

One measured consequence worth carrying: on an A40, **bf16 is 22 % faster than
4-bit** (247.6 s against 301.5 s, #137), because nf4 dequantises every weight on
every pass while the card does bf16 matmuls natively. Quantisation bought memory
with time; on a card the run owns alone that is no longer a trade worth making.
`load_in_4bit` still defaults to `True` because what it does to a *trained model*
is #138, and a footprint may not move that default.

## 4. The live case, and the correction it forces

`20260930T170315Z-ladder-xix-gemma4-12b`, Slurm `16830477`, Gemma 4 12B on the
19th-century corpus, lines:

| attempt | start | end | GPU time |
|---|---|---|---:|
| 1 | 01.10. 20:41, gnode27 | 02.10. 14:36 — `DUE TO PREEMPTION` | 17 h 55 |
| 2 | 03.10. 07:47, gnode28 | 04.10. 07:45 — wall | 23 h 58 |
| 3 | 04.10. 17:28, gnode25 | wall ~05.10. 17:28 | 21 h 25 at 14:53 |

At 14:53 on 05.10 it stands at **step 49 800 of 54 592 — 91.2 %**, loss 0.2026,
after **63 hours of GPU time** and **3 days 20 hours** of wall-clock since its
first start.

**#157 calls this "five days of wall-clock for a 24-hour run". That is not what it
is, and the difference changes the rule.** 63 GPU-hours for 49 800 steps is
4.58 s/step, so the full 54 592 steps need about **70 hours**. It was never a
24-hour run. It is a 70-hour run on a 24-hour queue, and three chunks plus two
waits is the *correct* cost of that, not a malfunction. Nothing was lost to
preemption beyond the steps since each last checkpoint, which `save_steps: 200`
bounds.

What the case does show is that **a run longer than the wall pays the queue once
per chunk**, which is why §1 measures the wait in wall-clock delay rather than in
queue position. It is also the strongest argument for asteraix that exists today —
*if* an A40 can finish 70 H100-hours of this work in less than the four to five
days UBELIX takes. Measured since (#162, point 1): r = 1.51 for lines (on a 4B,
so an estimate for this 12B), so ~106 h — 4.4 days — on asteraix. Close to a draw against four to five
days of UBELIX, and a clear win only past 36 h of delay (§1).

Two further frictions, both already fixed, both worth remembering as the kind of
cost the queue adds: a walltime guess carried over from the line case killed two
page rescorings at 150/164 and 75/164 samples (#155), and a resume ran the
training stage on `ff9effd` while the job had been created with `0870e50` — the
pin holds across the queue but not across that resume.

## 5. What is not measured, and what each gap blocks

- **The A40-to-H100 ratio for anything but the two measured loads.** #162
  measured lines (4B, 4-bit) at 1.51 and pages (8B, bf16) at 3.65; §1 carries
  both. Not measured: which of granularity, size and quantisation moves the
  ratio, a 12B or 27B on either card, and kraken or trocr at all. A third point
  that changes one variable at a time (8B lines in bf16, say) would say which.
- **Peak memory: what the table in §1 does not have yet.** It has full-run
  points for every engine (#163), filled from the job records rather than
  hand-run: every stage samples the card while its subprocess runs and records
  the peak on `StageRecord.peak_gpu_mib`, with the job's maximum on
  `Progress.peak_gpu_mib`. Two numbers per card: `own_mib`, the compute apps
  whose ancestor chain contains the pid the stage spawned, and `card_mib`, what
  the card reported in use at the same instant. VLM runs add the allocator's own
  marks, `reserved_mib` and `allocated_mib` — a floor, because the CUDA context
  lies outside them. Until #163 the VLM runner *assigned* those marks to the job
  after the train stage, which wiped the sampled training peak and left the test
  stage's in its place (28,612 MiB trained, 18,872 recorded); and it named them
  after torch's card index instead of nvidia-smi's. Both are fixed; the stage
  records were never affected, and they are what the table reads.

  Missing:
  - **A full VLM run on an A40.** Every A40 VLM row is step-bounded. The first
    full run there turns its lower bound into a point.
  - **An A40 figure for the H100 rows above the line.** Qwen3.5-4B pages at batch
    2 (109.7–110.0 %) and olmOCR at cap 3072 (109 %) are near enough that an
    allocator under pressure might fit them; nobody has tried.
  - **kraken before and after the ceiling on one corpus.** The two batch-256 rows
    differ in corpus as well as ceiling.
  - **trocr beyond `trocr-base`.** Every trocr point is the base model; a large
    one has none, and goes to UBELIX.

  The kraken batch sweep on the uncut arrow, 07.10.2026 (#171):

  | Batch | Peak own | Of an A40 (45,486) | Of an RTX 4090 (23,028) |
  |---:|---:|---:|---:|
  | 32 | 12,220 MiB | 26.9 % | 53.1 % |
  | 64 | 13,766 MiB | 30.3 % | 59.8 % |
  | 128 | **20,134 MiB** | 44.3 % | **87.4 %** |
  | 256 | 43,536 MiB | 95.7 % | **does not fit** |

  Memory grows far faster than the batch: a batch is padded to its widest
  line, and the larger it is, the likelier it holds one. So a kraken run fits
  UBELIX' free cards — RTX 4090, not the H100s — at **batch 128 or less**, and
  `ubelix/kraken_train.sbatch` refuses a larger one before anything trains
  (`ubelix/kraken_batch_fits.py`, for `KRAKEN_PLUS_SPEC` from scratch; another
  network is "not checked"). The path there is `ubelix/kraken-train.def`
  (ketos 7.0.2, as on asteraix) and the two stages in `ubelix/README.md`.
  The first peak on UBELIX itself (5./6.11.2026, Hertwig, batch 128) is
  3,970–4,046 MiB: a fifth of the 20,134 the sweep gives for 128, because its
  widest line is 22:1 where the sweep's arrow reached 135:1. So the sweep is a
  safe upper bound, not a forecast. Still missing: a UBELIX peak on the sweep's
  own arrow, the like-for-like comparison. What the
  image ceiling of #145 does at batch 256 is the collapse-probe row in §1.
- **Whether a run on asteraix can be reported.** It writes no job record, so it
  appears in no report and no comparison (#156). A rule that sends work there and
  loses its provenance is a bad trade; this gap bounds how much work should move
  before it is closed.
- **FSDP in the `vlm-train` venv.** Installed, unmeasured, and the only path that
  would make two A40s buy speed rather than only capacity.

## 6. Doing it today

UBELIX, the default — long runs on the preemptable queue, which chunks and
resumes by itself:

```bash
JOB_ID=<job id> SIF=$HOME/ubelix/vlm-train-tf5.sif ubelix/submit.sh ubelix/train_resumable.sbatch
```

A short measurement, which belongs on asteraix, goes through the trainer service
on the box and needs no queue at all. Until #156 gives it a job record, note the
run by hand wherever its number is going to be quoted.

Before choosing, two commands answer the third branch:

```bash
ssh ubelix "squeue -u \$USER --start -o '%.10i %.20S %.8Q'; sinfo -h -o '%N %t %G %C' -p gpu | grep h100"
```

```bash
ssh asteraix "nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader"
```

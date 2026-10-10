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
                                           │                      ┌─ ≥ 48 h ──→ asteraix
                                           └─ UBELIX wait is … ───┤
                                                                  └─ < 48 h ──→ UBELIX
```

Read top to bottom; the first branch that matches decides.

**1. Does it fit on one A40?** 44.42 GiB usable (measured, #137 — not the 46 068 MiB
nvidia-smi prints, and not the 92 GB the two-card epic assumed). If the peak does
not fit, the run goes to UBELIX and nothing below applies. Two A40s do hold
88.8 GiB, but only through `device_map="auto"`, which is naive model-parallel: it
distributes the peak and buys **no** speed (249.7 s against 247.6 s on the same
work, #137). Use that to make a run *possible*, never to make it faster.

**2. Is it a measurement rather than a training?** A re-evaluation is generation
with no optimizer steps: minutes to about two hours, run repeatedly, and usually
wanted the same day. Those go to asteraix, where there is no queue. This is the
branch that would have saved the two rescoring jobs that died on a walltime
guess (`17310472`, `17310473`, fixed in #155 after two lost allocations) — on
asteraix there would have been no walltime to guess.

**3. Otherwise: how long is the UBELIX wait?** `squeue --start` gives an estimate,
and the preemption record gives the rest (§4). **48 hours of expected
wall-clock delay is the line.** Below it, UBELIX — an H100 is the faster card and
the queue is the price. Above it, asteraix, if the run fits.

### Why 48 hours and not a day

Because a chunked run pays the wait more than once. A run longer than the 24 h
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
days UBELIX takes. Nobody has measured that, which is §5.

Two further frictions, both already fixed, both worth remembering as the kind of
cost the queue adds: a walltime guess carried over from the line case killed two
page rescorings at 150/164 and 75/164 samples (#155), and a resume ran the
training stage on `ff9effd` while the job had been created with `0870e50` — the
pin holds across the queue but not across that resume.

## 5. What is not measured, and what each gap blocks

- **The A40-to-H100 throughput ratio on identical work.** Without it §1's third
  branch cannot be computed: "48 hours of UBELIX delay" is only worth paying if
  the A40 finishes sooner than that, and the A40 is the slower card. This is the
  one measurement that makes the rule quantitative instead of a judgement, and it
  is cheap: the same corpus, the same step budget, one short run on each.
- **Peak memory per (engine, model size, granularity).** §1's first branch is a
  number nobody has tabulated. #137 measured one point — an 8B VLM on page
  samples, 27 763 MiB peak — from 96 of 9 441 samples, and said so: that figure
  is a lower bound, not a guarantee. kraken and trocr have no such figure at all.

  **Since 07.10.2026 the gap is the measurement, not the means (#163).** Every
  stage now samples the card while its subprocess runs and records the peak on
  `StageRecord.peak_gpu_mib`, with the job's maximum on `Progress.peak_gpu_mib`.
  Two numbers per card: `own_mib`, the compute apps whose ancestor chain contains
  the pid the stage spawned, and `card_mib`, what the card reported in use at the
  same instant. VLM runs add the allocator's own marks, `reserved_mib` and
  `allocated_mib`, into the same row — a floor, because the CUDA context lies
  outside them and `expandable_segments:True` accounts the reserve differently
  from the driver.

  This is how kraken gets a figure at all: it drives `ketos` as an external CLI
  and imports no torch, and its log carries no numbers because ketos renders
  through `rich`. Where the ancestry cannot be traced — a PID namespace would do
  it — `own_mib` is omitted rather than written as 0, because "could not look" is
  not "used nothing" (#165).

  **kraken's first point, measured 07.10.2026 on asteraix** with
  `scripts/measure_kraken_arms.py`: `german_val.arrow` as training data,
  `KRAKEN_PLUS_SPEC`, batch 256 — the project default — one full epoch in ~215 s.

  | Engine | Spec | Batch | Peak own | Peak card | Of 45,486 usable | Source |
  |---|---|---:|---:|---:|---:|---|
  | kraken | `KRAKEN_PLUS_SPEC` | 256 | **43,536 MiB** | 43,579 MiB | **95.7 %** | full epoch over the arrow, asteraix 07.10.2026 |

  So §1's first branch, for kraken: **it fits one A40 and very nearly does
  not** — 1,950 MiB of headroom at the default batch. The limit for this engine
  is the batch geometry, not the model size; a kraken model is ~16 MB. And the
  card must be empty: while a colleague's job held 740 MiB of gpu0, `card_mib`
  rose to 44,319 while `own_mib` stayed at 43,536.

  Not yet measured: batch 128, 64, 32, which is where the headroom is. Until
  those exist, a kraken run that cannot have a card to itself belongs on UBELIX.

  **trocr still has no number, and gets one by sampling rather than by
  plumbing.** Its runner does not read its own `training_summary.json`, so torch
  marks written there would be a number nobody reads; the external sampler covers
  it through `_run` instead. No trocr model has ever been trained (#37), so the
  first point arrives with the first run — and until then the tree sends it to
  UBELIX.

  `scripts/measure_vlm_arms.py --worst-case` is the other half: it replaces the
  head of the file with the longest transcriptions and the largest crops, which
  is where the peak lives. Without it a figure is a lower bound however carefully
  it was sampled.
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

# What a training run costs in electricity (#184)

The energy of a run is **measured**: `gpu.PeakSampler` integrates `power.draw`
next to every stage, and the job record carries the sum as `progress.energy_wh`.
The carbon is **computed** from it in `atr_training.footprint`, with the factor
kept beside the result. The model card prints both, plus the caveats. This page
says what the measured figure covers, what it misses, and what the programme
has cost so far.

## What is measured, and what is not

| | in the figure | why |
|---|---|---|
| GPU draw of every stage | yes | `nvidia-smi power.draw`, trapezoid over the real time between samples |
| attempts that were preempted or requeued | **yes, since this page** — before it, no | see below |
| CPU, memory, storage, network | no | not metered; 3 022 CPU-hours against 163.5 GPU-hours on 08.10.2026 says it is not nothing |
| data-centre overhead (PUE) | no | unknown, asked in #185; `ATR_TRAIN_PUE` takes it once known |
| manufacture of the hardware | **not by default** | needs a lifetime and an embodied figure nobody here has fixed (#185, point 4); `ATR_TRAIN_EMBODIED_G_PER_GPU_HOUR` takes the quotient once known, see below |
| runs started by hand outside the runner | no | measurement scripts, rescorings run outside a job |

Every figure is therefore a **floor**, and the card says so in words.

## Interrupted attempts were lost until 24.10.2026

The sampler integrated in memory, and the total reached the record only when a
stage ended in order. A preempted stage, or one requeued at the 24-hour wall,
dies before that, so an interrupted run kept **only the energy of its last
attempt**. Measured on UBELIX, 24.10.2026:

| run | GPU time, all attempts | recorded | the recorded figure matches |
|---|---:|---:|---|
| `olmocr2-7b-medieval-page-px4m-native-r16` | ~46 h (14 preemptions) | 880 Wh | its last attempt, 2.5 h at 288 W |
| `olmocr2-7b-german-kurrent-xvi-xix-page-px4m-native` | ~59 h (11 preemptions) | 4 001 Wh | its last attempt, 13.2 h at 291 W |

Since then the sampler writes its running total every 60 s to
`energy-pending.<stage>.<pid>.json` in the job directory. The next attempt adds
whatever a dead one left to `progress.energy_wh` and reports that part separately as
`progress.energy_recovered_wh`. An interruption now loses at most one minute of
draw, about 6 Wh on an H100. A stage that ends in order removes its ledger
before the record is saved: a kill in between loses that stage's energy rather
than counting it twice, so the error can only point down.

## What a card draws while training

From the eight records with a measured figure (UBELIX, H100, train stage,
last attempt; 13.–24.10.2026):

| run | train stage | mean draw |
|---|---:|---:|
| Qwen3.5-4B, lines, 4-bit | 6.08 h | 389 W |
| Qwen3.5-4B, lines, bf16 | 5.38 h | 389 W |
| Qwen3.5-9B, lines, 4-bit | 7.56 h | 392 W |
| Qwen3.8-27B, lines, 4-bit | 21.86 h | 393 W |
| Qwen3.5-4B, pages | 5.58 h | 359 W |
| olmOCR-2-7B, pages, 4 MP (three runs) | 2.5–13.2 h | 288–292 W |
| test stage, every run | 0.06–1.5 h | 110–150 W |

Qwen training sits at **~390 W regardless of size or quantisation**: the card is
saturated, and a larger model takes longer rather than drawing more. olmOCR at a
4 MP budget draws a quarter less per hour, but its steps take 2.4× as long as at the default budget (69 s against 29).

## The baseline: the programme since 01.09.2026

`sacct`, every attempt of every job of this user from 01.09. to 24.10.2026:

| card | GPU-hours | attempts |
|---|---:|---:|
| H100 | 726.5 | 136 |
| RTX 4090 | 59.8 | 44 |
| A100 | 7.3 | 14 |
| **total** | **793.5** | 194 |

Of the 194 attempts, 64 were preempted, 33 failed, 10 requeued, 3 cancelled and
2 hit the wall. The figure that gets this programme its energy is the share of
the work that was redone, not the efficiency of a step.

**Modelled, not measured**, because most of these hours predate the sampler.
The low end assumes every hour drew what a test stage draws, the high end what
Qwen training draws:

| card | assumed draw | energy |
|---|---|---:|
| H100 | 150–393 W | 109–286 kWh |
| RTX 4090 | 150–450 W (the high end is its TDP; nothing measured) | 9–27 kWh |
| A100 | 150–400 W | 1–3 kWh |
| **total** | | **119–315 kWh** |

At the factors `footprint` uses (#185) that is **2.4–6.3 kg CO2e market-based**
(20 g/kWh, the university's renewable procurement) and **14–36 kg
location-based** (115 g/kWh, Swiss consumer mix). PUE, CPUs and asteraix are
not in it. The **measured** part, eight records since 13.10., is 24.0 kWh.

asteraix has no scheduler and so no `sacct`. Its runs are in the job store with
a measured `energy_wh` where the runner ran them, and absent where a
measurement was run by hand.

## Method and framing

How the figures above relate to the published frameworks, and how to describe
them in a paper or a card. None of the frameworks replaces the method; they
name it.

**What the method is.** Activity-based, bottom-up accounting of operational
electricity: GPU draw is measured per stage and per attempt, carbon is computed
from it with factors kept as parameters. It is reported twice, as the GHG
Protocol Scope 2 Guidance asks when a renewable supply is claimed: market-based
at 20 g CO2e/kWh (the university's 100 % renewable procurement; the factor is
our order-of-magnitude choice, see `footprint.py`) and location-based at
115 g CO2e/kWh (Swiss consumer mix).

**Software Carbon Intensity (SCI, ISO/IEC 21031), Green Software Foundation.**
SCI is a rate, `(E × I + M) per R`. Mapped onto this repository:

| term | here |
|---|---|
| E, energy | `progress.energy_wh`, × PUE once known (#185) |
| I, carbon intensity | **115 g/kWh only**: SCI is location-based and does not let renewable purchases lower the score, so the 20 g figure belongs in the dual report, not in an SCI value |
| M, embodied hardware | card-hours (`progress.gpu_hours`) × `ATR_TRAIN_EMBODIED_G_PER_GPU_HOUR`; **0 until that rate is set** (#185, point 4) |
| R, functional unit | one trained model; for inference, 1 000 lines or pages transcribed |

While the rate is unset, say "follows the structure of SCI, without its
embodied term", not "SCI-conformant".

**Setting M.** The rate is the card's embodied emissions divided by its
expected lifetime in hours, apportioned by time as the SCI does. Card-hours
are counted by the sampler over the same intervals as the energy, interrupted
attempts included, so the two halves of `E × I + M` cover the same work.
Records written before 28.10.2026 carry 0 card-hours and keep the
"not apportioned" caveat even after a rate is set. For scale only, not as a
value: #185's own range of 1–2 t for an H100, over four years of continuous
use, is 29–57 g per card-hour. At ~390 W and 115 g/kWh, operation costs about
45 g per card-hour, so M is plausibly the same size as E × I and not a rounding
term. Both the embodied figure and the lifetime need a source before the rate
is set; the card prints M and an "SCI per trained model" row once it is. R is the useful part: per-model and per-page rates
make a 4B and a 27B run comparable on cost per CER point.

**Green Algorithms** (University of Cambridge; the computing calculator listed
on SPARKHub). The same equation, runtime × power × PUE × carbon intensity, with
power *estimated* from the hardware's rated draw and core usage instead of
measured. The modelled baseline above (119–315 kWh) is an estimate of exactly
that kind; the run records are measured. Its CPU and memory terms are the
obvious way to fill the gap in the first table without a new meter.

**Green DiSC** (Software Sustainability Institute, hosted on SPARKHub). A
certification for research groups, not a calculation method. Measured per-run
energy with stated caveats is evidence for its reporting criteria; whether it
satisfies a given level has to be read against the current criteria.

**DPC Carbon Footprint Toolkit** (Digital Preservation Coalition, 2026). Scoped
to digital preservation: on-premise and cloud storage, networking, travel. It
does not fit training, but it fits what this page leaves out: artefacts on the
research storage, the Hugging Face copies, the corpus caches (#185, point 4).

A sentence that can stand in a paper:

> We report the operational electricity of model training as a measured,
> bottom-up figure: GPU power is sampled during every attempt of every run,
> including preempted and requeued ones. Carbon is reported market-based
> (20 g CO2e/kWh, the university's renewable procurement) and location-based
> (115 g CO2e/kWh, Swiss consumer mix), following the GHG Protocol Scope 2
> Guidance, and per trained model following the structure of the Software
> Carbon Intensity specification (ISO/IEC 21031) without its embodied term.
> Hours not covered by measurement are estimated as in Green Algorithms
> (runtime × assumed draw). CPU, memory, storage, data-centre overhead and
> hardware manufacture are excluded, so all figures are lower bounds.

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
| manufacture of the hardware | no | needs a lifetime and a utilisation nobody here has fixed (#185, point 4) |
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

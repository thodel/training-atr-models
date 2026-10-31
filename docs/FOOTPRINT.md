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
| CPU, memory | **modelled**, apart (#331) | RAPL is root-only on both boxes, so every stage carries `energy_cpu_modelled` (allocated cores × W per core at TDP × hours; memory at 3 W per 8 GB allocated) in its own fields, never summed into `energy_wh`; see *CPU and memory* below |
| storage, network | no | not metered |
| data-centre overhead (PUE) | no | unknown, asked in #185; `ATR_TRAIN_PUE` takes it once known |
| manufacture of the hardware | **yes, since 31.10.2026**, apart from the operational figures | 13.6 g CO2e per card-hour by default: H100 164 kg plus 312 kg server share over four years (see "What a card cost to make", #332; sourced band 2.4–20 g); decided in #185, point 4. `ATR_TRAIN_EMBODIED_G_PER_GPU_HOUR` overrides it, 0 switches it off. Records written before 28.10.2026 carry no card-hours and keep the "not apportioned" caveat |
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

## What a CER point costs on the ladder (#333)

Both axes are measured — `progress.energy_wh` by the sampler, `metrics.cer` by
the test stage — and `scripts/energy_per_cer.py` puts them side by side, under
the rule the results MCP already enforces: two CER values compare only when they
were **scored on the same draw** (md5 of `data/val_eval.jsonl`) **and trained on
the same corpus** (`reserved_pages`, #98). Pairing goes by the record, not by the
arm's name — the draft of #333 had put the Qwen3.8-27B arm's kilowatt-hours
next to the Qwen3.5-27B arm's CER, and the Qwen3.5-27B record carries **no energy
figure at all**: it ran before the sampler reached UBELIX. The Qwen3.8 arm stands
in for it; §1f of `BASE_MODEL_LADDER.md` measured the two 0.09 points apart.

The medieval line ladder (draw `9fe26816`, 200 lines, `reserved_pages` 3173, all
H100 on UBELIX, 30.10.2026), **4-bit arms only** — the ladder in which only the
size differs:

| arm | job | train | kWh (job) | CER | ΔkWh | ΔCER | kWh per CER point |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B, 4-bit | `20261014T194417Z-ladder-med-qwen35-4b-v2` | 6.08 h | 2.38 | 15.94 % | | | |
| Qwen3.5-9B, 4-bit | `20261014T193118Z-ladder-med-qwen35-9b-v2` | 7.56 h | 2.97 | 13.75 % | +0.60 | −2.19 | **0.27** |
| Qwen3.8-27B, 4-bit | `20261013T100004Z-ladder-med-qwen38-27b-v3` | 21.86 h | 8.61 | 12.15 % | +5.63 | −1.60 | **3.52** |

**A CER point costs 0.3 kWh up to 9B and 3.5 kWh above it** — the 27B step pays
thirteen times as much energy per point gained as the 9B step. That is the curve
the tutorial draws (Wu et al. 2022: quality ∝ energy^−k with k ≈ 0.002–0.004),
on our records. In carbon: 0.27 kWh is 34 g location-based, 3.52 kWh is 450 g —
per CER point, per run.

With **every arm** of the draw the picture sharpens rather than changes. The
bf16 4B arm (`20261023T194329Z-ladder-med-qwen35-4b-bf16-v2`, 5.38 h, 2.10 kWh,
14.74 %) is both cheaper and better than the 4-bit 4B, which the script lists as
*dominated*: 4-bit buys memory, not energy, and on a card that holds the bf16 model
it buys nothing. Along that frontier the step to 9B is +0.87 kWh for −0.99 points
(0.88 kWh per point) and the step to 27B stays at 3.52.

Three things the table does not say. The **noise floor** of this draw is not
measured (#115 measured kraken and the XIX sweep, where it was 0.117 points); a
ΔCER of a few tenths may be seed noise, and energy per noise is infinite — the
script takes `--floor` and prints ∞ for a step inside it. **Lines against pages**
(#333, point 3) has no comparable pair: the line ladder and the page runs are
scored on different draws, so the issue's 13.65 % against 37.61 % is two
measurements on two draws, not one step. And the only page ladder with two measured runs is a
different corpus, Kurrent XVI–XIX (draw `ecef5cb8`, 171 pages, `reserved_pages`
1318): olmOCR-2-7B at 4 MP reads it at 15.71 % for 4.00 kWh against Qwen3.5-4B's
27.04 % for 2.17 kWh — 1.83 kWh for 11.3 points, 0.16 kWh per point, the cheapest
point on any ladder here.

A footprint does not move the rule in `WHERE_A_RUN_RUNS.md` (#138 decided that);
it prices it. Reproduce the tables on the login node, no container needed:

    python3 scripts/energy_per_cer.py --markdown --filter host=ubelix
    python3 scripts/energy_per_cer.py --markdown --filter load_in_4bit=true

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

At the factors `footprint` uses (#185, sources below) that is **2.4–6.3 kg CO2e
market-based** (20 g/kWh, the university's renewable procurement) and
**15–40 kg location-based** (127 g/kWh, Swiss consumer mix 2021–2023). PUE and
asteraix are not in it, and the CPUs only as the model below. The **measured** part is a running counter,
not a hand sum: `atr-results` `energy()` (#335) adds up `energy_wh` over both
job stores per host, names every record whose GPU stage ran without a
measurement (the sum is a floor until that list is empty), and gives the share
recovered from preempted attempts. The daily report prints its line from
there. (When this page was first written, 24.10.2026, the counter stood at
24.0 kWh over eight records since 13.10.; that figure is history, not a
number to update here.)

asteraix has no scheduler and so no `sacct`. Its runs are in the job store with
a measured `energy_wh` where the runner ran them, and absent where a
measurement was run by hand.

## CPU and memory: modelled, apart (#331)

**Measured first, 30.10.2026.** The kernel's RAPL counters
(`/sys/class/powercap/intel-rapl*/energy_uj`) are `0400 root` on asteraix, on the
UBELIX login node and on an epyc2 compute node (`bnode`, read from an `srun`).
Without root the CPU energy cannot be read on either box. So it is modelled, the
model is written into every stage record beside its parameters, and it is **never
added to `energy_wh`**: a measurement and a model do not sum, and the card prints
"CPU and memory, modelled, not in the figures above" as its own row.

The model is CodeCarbon's own fallback. Watts per allocated core = the
processor's TDP / its cores, from CodeCarbon's table
([cpu_power.csv](https://github.com/mlco2/codecarbon/blob/master/codecarbon/data/hardware/cpu_power.csv));
cores = `SLURM_CPUS_PER_TASK` under Slurm, else the request's `workers`; memory =
`SLURM_MEM_PER_NODE` × 3 W per 8 GB (CodeCarbon's assumption), off Slurm not
counted. The cores are taken as fully loaded for the whole stage — right for a
CPU-bound `prepare`, an upper band for the 16 cores idling beside an H100.

| box | processor (lscpu, 30.10.2026) | TDP | W per core | a prepare stage (8 cores, 16 GB) |
|---|---|---:|---:|---:|
| UBELIX epyc2 (`bnode`) | 2 × AMD EPYC 7742, 64 cores each | 225 W | **3.5** | 28 W + 6 W memory → 34 W, 2 h = 68 Wh |
| UBELIX H100 nodes (`gnode`) | 2 × 64 cores; *assumed* the same EPYC 7742 — the record keeps the name it read, so another model shows as unmodelled | 225 W | 3.5 | train: 16 cores, 90 GB → 56 W + 34 W |
| asteraix | AMD Ryzen Threadripper PRO 5965WX, 24 cores | 280 W | **11.7** | `workers` 8 → 93 W, memory not allocated → not counted |

An unknown processor leaves the stage **unmodelled** (`cpu_wh: null`), which is
not 0 Wh; `ATR_TRAIN_CPU_W_PER_CORE` overrides the rate for one.

**The band since 01.09.2026.** `sacct` over every attempt of this user, 01.09.–30.10.:
1 222 CPU-hours on the CPU partitions (prepare, inventories) and 4 944 CPU-hours
allocated beside a GPU, 6 166 in all (#184 counted 3 022 on 08.10.). At 3.5 W per
core that is **4.3 kWh for the CPU-only jobs and 17.4 kWh beside the cards — 21.7
kWh at full load, 10.8 kWh at CodeCarbon's 50 % default**: against 50.9 kWh
measured at the cards (`energy()`, 30.10.), the unmetered share is a fifth to two
fifths of the measured one, not larger than it as the issue feared with 10–15 W per
core. The cores beside the cards are the uncertain part; the prepare jobs are the
part the model fits.

**Calibration, once, 30.10.2026 22:10 CET on asteraix** (issue point 3): a 160 s
bf16 matmul load on card 1, watched by our `PeakSampler` on the load's pid and by
CodeCarbon 3.3.2 (`OfflineEmissionsTracker`, `tracking_mode="machine"`,
`gpu_ids=[1]`, 2 s samples) at the same time — beside ours, not instead of it.

| signal | ours | CodeCarbon | note |
|---|---:|---:|---|
| card 1, energy | **13.30 Wh** (77 readings, 0 failures) | **13.25 Wh** (295.9 W mean) | same `nvidia-smi`/NVML signal, 0.4 % apart: our GPU figure is what CodeCarbon would publish |
| card 0, idle beside it | 1.05 Wh (≈ 24 W) | not tracked | what an idle A40 costs per hour: 24 Wh |
| CPU | model: 8 workers × 11.7 W × 0.044 h = **4.15 Wh** | **1.24 Wh** (28 W mean, "CPU load mode": TDP × measured load, RAPL refused) | a GPU-bound stage keeps the cores near idle; the static model overstates it 3.3× there and is the ceiling it claims to be. For `prepare` the load is the TDP |
| memory | not counted (no Slurm allocation) | 3.10 Wh (70 W for the box's 251 GB) | CodeCarbon prices the whole box's DIMMs; by allocation on UBELIX (16 GB → 6 W) we count a fifth of that per prepare job |
| CPU + memory against the card | — | 4.34 Wh against 13.25 Wh = **+33 %** | the box-level unmetered share during a GPU stage, with a nearly idle CPU |

CodeCarbon's own log confirms point 1: `RAPL - Permission denied … falling back
on estimation based on TDP`. Both tools model the cores on this box; neither
measures them. (The CodeCarbon venv stays at `~/.venvs/codecarbon` on asteraix for
a repeat; nothing runs from it.)

Not done here: the energy of a corpus materialisation on the **dataset** card
(issue point 4) — the modelled prepare figure is in the job record now, and the
dataset card can read it once the dataset publisher learns to (#331, follow-up).

## Where the factors come from (#336)

Every factor on a card has a source and a named mix; a number without either
cannot be corrected, only replaced. Until 30.10.2026 the grid factor was 115 g,
"a round figure from published ranges". It is now read from a document.

| factor | value | what it is | source |
|---|---:|---|---|
| location-based (`GRID_FACTOR_G_PER_KWH`) | **127 g CO2e/kWh** | Swiss **consumer mix** 2021–2023: domestic production plus commercial trade, hour by hour, at the low-voltage socket; the mix Swiss accounting rules (KBOB) prescribe when the product is unknown | Frischknecht (treeze) for the BFE, *Energieetikette für Personenwagen: Umweltkennwerte 2025 der Strom- und Treibstoffbereitstellung*, Bern, 4 June 2025, section 4.5.2 — [pubdb.bfe.admin.ch/…/12224](https://pubdb.bfe.admin.ch/de/publication/download/12224) |
| market-based (`DEFAULT_FACTOR_G_PER_KWH`) | **20 g CO2e/kWh** | our choice for the university's procurement (100 % renewable: hydro, biomass, solar, predominantly Swiss — [klima.unibe.ch](https://klima.unibe.ch/betrieb/energie/index_ger.html)), anchored on the KBOB life-cycle factors and rounded **up** | treeze for KBOB/BAFU/AHB, *Umweltkennwerte und Primärenergiefaktoren von Energiesystemen*, 9 March 2022, Tab. 2.1 — [ecobau.ch/…/KBOB_BAFU_AHB_2022_Energiesysteme_v1_0.pdf](https://www.ecobau.ch/resources/uploads/Oekobilanzdaten/Hintergrundberichte/KBOB_BAFU_AHB_2022_Energiesysteme_v1_0.pdf) |

The KBOB table (kg CO2-eq per MJ, at the building's socket) gives the pieces the
20 g rounds from: hydro 0.003 (≈ 11 g/kWh), photovoltaics 0.013–0.015
(47–54 g/kWh), the certified renewable product mix 0.004 (≈ 14 g/kWh), the
certificate-based supplier mix 0.015 (54 g/kWh, 2018 basis). The BFE's 2025
figure for the certificate-based supplier mix 2023 is 29 g CO2e/kWh. The
university's portfolio is not that mix — it buys renewables only — but its
biomass and solar shares are unpublished, so 20 is an anchored upper rounding
of 14, not a measurement. It stays overridable (`ATR_TRAIN_CO2_G_PER_KWH`).

**Why the grid factor is not 35 g.** The Swiss *production* mix — what the
country's plants emit, imports excluded — is 0.009 kg/MJ (≈ 32 g/kWh) in the
same KBOB table, and CodeCarbon's country file
(`codecarbon/data/private_infra/global_energy_mix.json`, checked 30.10.2026)
carries 34.843 g CO2e/kWh for Switzerland, year 2023, derived from production
by source. Both are "location-based" in a loose sense and they differ from 127
by about 3.6×. We use the consumer mix because the electricity is consumed in
Bern, not produced there, and because it is what Swiss rules prescribe. The
difference is a choice of mix, not a measurement: a run measured with
CodeCarbon on the same card publishes about a quarter of our location-based
figure, and the card says so.

## Method and framing

How the figures above relate to the published frameworks, and how to describe
them in a paper or a card. None of the frameworks replaces the method; they
name it.

**What the method is.** Activity-based, bottom-up accounting of operational
electricity: GPU draw is measured per stage and per attempt, carbon is computed
from it with factors kept as parameters. It is reported twice, as the GHG
Protocol Scope 2 Guidance asks when a renewable supply is claimed: market-based
at 20 g CO2e/kWh (the university's 100 % renewable procurement; the factor is
our anchored choice, see below) and location-based at 127 g CO2e/kWh (Swiss
consumer mix 2021–2023).

**Software Carbon Intensity (SCI, ISO/IEC 21031), Green Software Foundation.**
SCI is a rate, `(E × I + M) per R`. Mapped onto this repository:

| term | here |
|---|---|
| E, energy | `progress.energy_wh`, × PUE once known (#185) |
| I, carbon intensity | **127 g/kWh only**: SCI is location-based and does not let renewable purchases lower the score, so the 20 g figure belongs in the dual report, not in an SCI value |
| M, embodied hardware | card-hours (`progress.gpu_hours`) × 13.6 g by default (`DEFAULT_EMBODIED_G_PER_GPU_HOUR`, decided 31.10.2026 in #185, point 4; `ATR_TRAIN_EMBODIED_G_PER_GPU_HOUR` overrides) |
| R, functional unit | one trained model; for inference, 1 000 lines or pages transcribed |

With the rate set, say "follows the structure of SCI with M at a sourced
rate", still not "SCI-conformant" (E lacks the PUE, CPU and memory are
modelled apart). A record without card-hours, every one before 28.10.2026,
is still "without its embodied term".

**Setting M.** The rate is the card's embodied emissions divided by its
expected lifetime in hours, apportioned by time as the SCI does. Card-hours
are counted by the sampler over the same intervals as the energy, interrupted
attempts included, so the two halves of `E × I + M` cover the same work.
Records written before 28.10.2026 carry 0 card-hours and keep the
"not apportioned" caveat even after a rate is set. The sources for the
embodied figure and the band they give are under "What a card cost to make"
below (#332); the card prints M with its source and an "SCI per trained model"
row for every record that has card-hours. R is the useful part: per-model and per-page rates make a 4B and a 27B
run comparable on cost per CER point.

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
> (127 g CO2e/kWh, Swiss consumer mix 2021–2023 per the Federal Office of
> Energy), following the GHG Protocol Scope 2
> Guidance, and per trained model following the structure of the Software
> Carbon Intensity specification (ISO/IEC 21031), with hardware manufacture
> apportioned by time at 13.6 g CO2e per card-hour (NVIDIA's product carbon
> footprint of the HGX H100, 1 312 kg for eight cards, plus a server share
> after Luccioni et al. 2023, over four years). Hours not covered by
> measurement are estimated as in Green Algorithms (runtime × assumed draw).
> CPU and memory are modelled from allocated cores and reported apart;
> storage and data-centre overhead are excluded, so the operational figures
> are lower bounds.

## What a card cost to make (#332)

The rate M is `C ÷ L`: the embodied emissions C of the card, with its share of
the node it sits in, divided by its lifetime L in hours. Every hour on the card
carries 1/L of its manufacture, whoever else uses it; idle hours are the data
centre's, not a run's (the SCI convention; BLOOM instead divided by an 85 %
occupancy, which would raise every rate below by 18 %).

**The card**, cradle to gate:

| card | C, kg CO2e | source |
|---|---:|---|
| H100 SXM | **164** | NVIDIA, *Product Carbon Footprint Summary for NVIDIA HGX H100* (2025): **1 312 kg for one HGX H100 baseboard with eight H100 SXM**, NVSwitch, thermals and board included; ISO 14067, critically reviewed, performed by WSP. 1 312 ÷ 8, as Morand et al. (2026) and Falk et al. (2025) divide it |
| H100 | 225 | Morand, Névéol, Ligozat, *The Rising Unsustainability of AI Graphics Cards Production*, LIMITS '26, arXiv:2607.01258, Table 4: their own model of one H100 card; most of the gap to NVIDIA is the memory |
| A100 SXM4 40 GB | **127.6** | Falk et al., *More than carbon: cradle-to-grave environmental impacts of GenAI training on the Nvidia A100 GPU*, arXiv:2509.00093 (v3, 18.12.2025); 141 kg with distribution and disposal |
| A100 | 150 | Luccioni, Viguier, Ligozat, *Estimating the Carbon Footprint of BLOOM*, JMLR 24 (2023), §4.1: "a lower bound" from secondary data (Davy 2021), before any measured figure existed |
| A40 (asteraix) | **above 82.5, below 128** | not published. The nearest sourced card is Morand et al., Table 3: RTX A4500 (2021), **82.5 kg**, the same GA102 die with 20 GB GDDR6. The A40 carries 48 GB, and memory is the largest term in every model above (42 % of NVIDIA's baseboard), so the A40 sits above the A4500 and below the A100's 123–128 |
| RTX 4090 | not published | no vendor footprint, not in Morand's table. The band below applies the data-centre figures to its hours, which this page says rather than hides |

**The node.** A card does not run without CPUs, memory, board, power supply and
chassis. BLOOM took HPE's footprint of a ProLiant DL345 Gen10 Plus server,
**about 2 500 kg without GPUs** (HPE 2021), as the proxy for an eight-GPU Apollo
6500 node: **~312 kg per GPU slot**. How many cards share a UBELIX node is not
recorded here; at four per node the share doubles.

**Why #185 said 1–2 t per H100.** That is the size of the eight-card baseboard
(1 312 kg), not of one card. Read as per-card, it put the rate at 29–57 g per
card-hour, which this page printed "for scale" until 30.10.2026; per card it is
six to twelve times lower.

**The rate**, lifetime 4 years (35 040 h, the usual convention) to 6 years
(52 560 h, what IDRIS told the BLOOM authors):

| C | 4 years | 6 years |
|---|---:|---:|
| A100 card only, 127.6 kg | 3.6 g/card-h | 2.4 |
| H100 card only, 164 kg | 4.7 | 3.1 |
| H100 card + node share, 476 kg | **13.6** | 9.1 |
| H100 (Morand) + node share, 538 kg | 15.3 | 10.2 |

Falk et al. assume three years for the A100, not four; at three years the top
of the band is 20 g. So **M is 2.4–15 g per card-hour** (20 g at a three-year
life), against operation at ~390 W of 7.8 g
market-based and 50 g location-based. Location-based, manufacture adds 5–30 %;
market-based, it is about as large as the operation or twice as large. A
footprint that reports only the 20 g figure leaves out what may be its largest
term, which is why the card keeps the "not apportioned" caveat while M is 0.

**For the programme since 01.09.2026** (793.5 card-hours, the baseline above):
**1.9–12.2 kg CO2e** of manufacture, 10.8 kg at the H100-with-node rate over
four years, beside 2.4–6.3 kg operation market-based and 15–40 kg
location-based. For one run, the 4-bit Qwen3.5-4B line arm (6.08 h): 15–93 g,
beside its 47 g market-based and 300 g location-based.

**Set on 31.10.2026** (#185, point 4): the default is 13.6 g per card-hour,
the H100 with its node share over four years, because most card-hours are
UBELIX H100 hours and a card is never made without a node. It is the one
decision on this page, not a lookup; `ATR_TRAIN_EMBODIED_G_PER_GPU_HOUR`
overrides it and 0 switches it off. For the asteraix A40 hours the rate
overstates the card itself (82.5–128 kg against 164) and the node share is
the same guess. C is now known for the H100 and A100, bounded by an analogue for the
A40 and unknown for the RTX 4090, hence a band rather than a point. Every
figure in this section was read against its primary document on 31.10.2026
(NVIDIA PCF summary, JUL 25; BLOOM §4.1; Morand Tables 3–4; Falk v3).

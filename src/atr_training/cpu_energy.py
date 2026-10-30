"""CPU and memory energy of a stage: modelled, because it cannot be measured (#331).

`gpu.PeakSampler` integrates the cards' `power.draw`. A stage without a card —
`prepare` materialises a corpus for hours on CPUs — therefore records 0 Wh, which
`Footprint` reads correctly as "not measured". That means the CPU share of the
programme was not a floor in the figures: it was absent.

**Measured on 30.10.2026, before anything was built:** the kernel's RAPL counters
(``/sys/class/powercap/intel-rapl*/energy_uj``) are ``0400 root`` on asteraix,
on the UBELIX login node and on an epyc2 compute node (``bnode``, checked from an
``srun``). They cannot be read without root on either box, so the CPU energy
cannot be measured here; it can only be modelled, and the model has to say so.

The model is CodeCarbon's fallback (its TDP table is where the wattages below
come from): the processor's TDP divided by its cores gives watts per allocated
core, times the cores the scheduler gave the stage, times the hours the stage
ran; memory at 3 W per 8 GB allocated. Two assumptions are stated rather than
hidden. The cores are taken as fully loaded for the whole stage, which is what a
CPU-bound ``prepare`` does and what the 16 cores beside an H100 do *not* — so the
figure is an upper band for the GPU stages and about right for prepare. And
memory is counted by allocation, not by use.

Calibrated once against CodeCarbon on asteraix (30.10.2026, docs/FOOTPRINT.md):
the two read the card within 0.4 % of each other; for the cores CodeCarbon's
load-based estimate was a third of this model's on a GPU-bound stage, which is
the direction a ceiling should err in.

The modelled figure is kept in its **own fields** and never added to
``energy_wh``: a measurement and a model do not sum, and the card prints "CPU
modelled at N W/core" beside the measured draw.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

#: CodeCarbon's memory assumption, 3 W per 8 GB — stated as an estimate there too.
#: https://mlco2.github.io/codecarbon/methodology.html#ram
W_PER_GB_RAM = 0.375

#: TDP in watts and physical cores, from CodeCarbon's table
#: (codecarbon/data/hardware/cpu_power.csv, read 30.10.2026) and the vendors'
#: specifications. The key is the model name as ``/proc/cpuinfo`` prints it.
#: UBELIX's epyc2 nodes: two EPYC 7742 (lscpu on bnode, 30.10.2026). Its H100
#: nodes report 2 × 64 cores as well and are *assumed* to carry the same
#: processor; the record keeps the name it actually read, so a different one
#: shows up as unmodelled rather than mis-modelled. asteraix: lscpu 30.10.2026.
CPU_TDP_W: dict[str, tuple[float, int]] = {
    "AMD EPYC 7742 64-Core Processor": (225.0, 64),
    "AMD EPYC 7763 64-Core Processor": (280.0, 64),
    "AMD EPYC 7713 64-Core Processor": (225.0, 64),
    "AMD Ryzen Threadripper PRO 5965WX 24-Cores": (280.0, 24),
    "AMD Ryzen Threadripper PRO 5975WX 32-Cores": (280.0, 32),
}
TDP_SOURCE = ("https://github.com/mlco2/codecarbon/blob/master/codecarbon/data/hardware/"
              "cpu_power.csv")

#: Override for a processor not in the table (watts per allocated core).
ENV_W_PER_CORE = "ATR_TRAIN_CPU_W_PER_CORE"


def cpu_model_name(cpuinfo: Path = Path("/proc/cpuinfo")) -> str | None:
    """The processor's name as the kernel prints it, or None off Linux."""
    try:
        text = cpuinfo.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def w_per_core(model: str | None, env: dict[str, str] | None = None) -> float | None:
    """Watts per allocated core at TDP, or None when the processor is unknown.

    None, never 0: an unknown processor leaves the stage unmodelled, which the
    record says, instead of entering a total as if it cost nothing (#165).
    """
    env = os.environ if env is None else env
    raw = env.get(ENV_W_PER_CORE)
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    if model and model in CPU_TDP_W:
        tdp, cores = CPU_TDP_W[model]
        return tdp / cores
    return None


def allocated_cpus(fallback: int | None, env: dict[str, str] | None = None) -> int | None:
    """Cores the scheduler gave this process: Slurm's count, else the caller's.

    Under Slurm ``SLURM_CPUS_PER_TASK`` (``--cpus-per-task``) is what the stage
    may use; off Slurm there is no allocation, and the request's ``workers`` is
    the only honest count of what the stage keeps busy.
    """
    env = os.environ if env is None else env
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        raw = env.get(name)
        if raw and raw.isdigit() and int(raw) > 0:
            return int(raw)
    return fallback if fallback and fallback > 0 else None


def allocated_ram_gb(env: dict[str, str] | None = None) -> float | None:
    """Memory the scheduler allocated, in GB; None off Slurm (not 0)."""
    env = os.environ if env is None else env
    raw = env.get("SLURM_MEM_PER_NODE")
    if raw and raw.isdigit() and int(raw) > 0:
        return int(raw) / 1024.0
    return None


@dataclass(frozen=True)
class CpuModel:
    """One stage's modelled CPU and memory energy, with every parameter beside it."""
    hours: float
    cpus: int | None
    w_per_core: float | None
    cpu_model: str | None
    ram_gb: float | None

    @property
    def cpu_wh(self) -> float | None:
        if self.cpus is None or self.w_per_core is None:
            return None
        return round(self.cpus * self.w_per_core * self.hours, 2)

    @property
    def ram_wh(self) -> float | None:
        if self.ram_gb is None:
            return None
        return round(self.ram_gb * W_PER_GB_RAM * self.hours, 2)

    @property
    def modelled(self) -> bool:
        return self.cpu_wh is not None

    def as_record(self) -> dict:
        """What the stage record keeps: the figures and the assumptions that made them."""
        return {
            "cpu_wh": self.cpu_wh, "ram_wh": self.ram_wh, "hours": round(self.hours, 4),
            "cpus": self.cpus, "w_per_core": (round(self.w_per_core, 3)
                                              if self.w_per_core is not None else None),
            "cpu_model": self.cpu_model, "ram_gb": self.ram_gb, "w_per_gb_ram": W_PER_GB_RAM,
            "assumption": "allocated cores at full per-core TDP for the whole stage; "
                          "memory by allocation (CodeCarbon's fallback); modelled, "
                          "not measured — RAPL is root-only here",
        }


def model_stage(hours: float, *, workers: int | None,
                env: dict[str, str] | None = None,
                cpuinfo: Path = Path("/proc/cpuinfo")) -> CpuModel:
    model = cpu_model_name(cpuinfo)
    return CpuModel(hours=hours, cpus=allocated_cpus(workers, env),
                    w_per_core=w_per_core(model, env), cpu_model=model,
                    ram_gb=allocated_ram_gb(env))

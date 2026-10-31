"""Energy and carbon of a training run (#184).

The energy is **measured**: `gpu.PeakSampler` integrates `power.draw` against the
real time between samples, next to every stage. The carbon is **computed** from
that, and computation is where the honesty has to live — so everything the number
rests on travels with it, and the card prints the parameters beside the result.

**What the electricity is.** The University of Bern has procured electricity from
100 % renewable sources since 2016 — hydro, biomass, solar, predominantly produced
in Switzerland (`klima.unibe.ch/betrieb/energie`). That page states the *source*
and no emission factor, so the factor below is our choice and is marked as such.

**100 % renewable is not 0 g.** Lifecycle emissions of renewable electricity are
small but real: building and maintaining dams, growing and burning biomass,
manufacturing panels. The KBOB life-cycle factors for Switzerland (treeze for
KBOB/BAFU/AHB, 2022, Tab. 2.1, at the building's low-voltage socket) put hydro
at about 11 g CO2e/kWh, rooftop photovoltaics at 47–54, and the certified
renewable product mix sold in Switzerland at about 14. The university's portfolio
adds biomass and solar to hydro in shares it does not publish, and both lie above
hydro, which is why :data:`DEFAULT_FACTOR_G_PER_KWH` rounds *up* to 20 and why it
is a *parameter*: an anchored choice, not a measurement, and a run's record keeps
the figure it used so a later correction can be applied without re-running
anything.

**Market-based, and the other number is larger.** Procuring certified renewable
electricity is a market instrument. The GHG Protocol asks for dual reporting when
one is used: the supplier's factor (what we pay for) *and* the grid factor where
the consumption physically happens (what the wires did). For the grid figure the
Swiss federal rule (KBOB accounting rules, applied by the BFE) is the **consumer
mix**: domestic production plus commercial imports and exports, hour by hour, at
the low-voltage socket. The BFE's 2025 figures give 127 g CO2e/kWh for 2021–2023,
so a location-based figure is about six times the market-based one.
:func:`Footprint.both` reports both rather than letting the flattering one stand
alone.

**Which "location-based" — the mix has to be named.** The Swiss *production*
mix (what the country's plants emit, imports excluded) is about 32 g CO2e/kWh in
the same KBOB table, and CodeCarbon's country file carries 34.8 g for Switzerland
(2023). A tool that takes the production mix publishes a figure a quarter of ours
for the same run; the difference is the choice of mix, not a measurement. The
consumer mix is used here because the electricity is *consumed* in Bern, not
produced there, and because it is the figure Swiss accounting prescribes when
the specific product is unknown — which, for the grid half of the dual report,
it is by definition.

**What is missing, and in which direction.** Three gaps, all of which make the
number an *under*-estimate:

* **PUE is unknown.** Cooling and power conversion in the data centre are not in
  `power.draw`. Typical values are 1.1 to 1.6. :data:`DEFAULT_PUE` is 1.0 — not
  because it is right, but because inventing 1.3 would dress a guess as a fact.
  The card says the figure excludes it.
* **Only the GPUs are metered.** CPU, memory, storage and network draw power that
  `power.draw` does not see. For a GPU training run the cards dominate, but
  3,022 CPU-hours against 163.5 GPU-hours (#184) says it is not nothing.
* **Hardware manufacture is counted, apportioned by time, since 31.10.2026**
  (#185, point 4, decided; sources in #332). The Software Carbon Intensity
  specification calls this term M: the card's embodied emissions divided by its
  expected lifetime in hours, times the hours this run held it.
  :data:`DEFAULT_EMBODIED_G_PER_GPU_HOUR` is that quotient at 13.6 g: an H100 at
  164 kg CO2e (NVIDIA's HGX H100 product carbon footprint, 1 312 kg for the
  eight-card baseboard, ISO 14067) plus 312 kg for its share of the server
  (the HPE proxy Luccioni et al. used for BLOOM), over four years (35 040 h).
  The sourced band is 2.4–20 g; the rate is deliberately the H100-with-node
  value, because most card-hours are UBELIX H100 hours and a card is never
  made without a node. ``ATR_TRAIN_EMBODIED_G_PER_GPU_HOUR`` overrides it, and
  0 switches it off. The card hours it multiplies are in the record
  (``progress.gpu_hours``); records written before 28.10.2026 have none and
  keep the "not apportioned" caveat.

So a published figure is a floor, and the card states that in words rather than
leaving a reader to assume completeness.
"""
from __future__ import annotations

from dataclasses import dataclass

#: g CO2e per kWh for the university's procured mix — 100 % renewable, mostly
#: Swiss. A choice anchored on :data:`FACTOR_SOURCE`, rounded up; see the module
#: docstring. Overridable with ``ATR_TRAIN_CO2_G_PER_KWH`` so a documented
#: correction needs no code change.
DEFAULT_FACTOR_G_PER_KWH = 20.0
#: What the market-based factor rests on, and why it is 20 and not 14.
FACTOR_DESCRIPTION = ("certified renewable product mix, Switzerland, about 14 g "
                      "CO2e/kWh in the KBOB 2022 life-cycle factors, rounded up "
                      "for the biomass and solar shares of the university's "
                      "portfolio")
FACTOR_SOURCE = ("https://www.ecobau.ch/resources/uploads/Oekobilanzdaten/"
                 "Hintergrundberichte/KBOB_BAFU_AHB_2022_Energiesysteme_v1_0.pdf")
#: The Swiss consumer mix, for the location-based half of the dual report:
#: domestic production plus commercial trade, 2021–2023, at the low-voltage
#: socket, as the BFE publishes it (treeze for BFE, 4 June 2025, section 4.5.2).
#: The production mix (imports excluded) is about 32 g in the KBOB 2022 table and
#: 34.8 g in CodeCarbon's country file (2023); see the module docstring for why
#: the consumer mix is the one used here.
GRID_FACTOR_G_PER_KWH = 127.0
GRID_FACTOR_DESCRIPTION = ("Swiss consumer mix 2021–2023, production plus "
                           "commercial trade, at the low-voltage socket")
GRID_FACTOR_SOURCE = "https://pubdb.bfe.admin.ch/de/publication/download/12224"
#: 1.0 = the data centre's own overhead is NOT included, because nobody has given
#: us its PUE. Overridable with ``ATR_TRAIN_PUE``.
DEFAULT_PUE = 1.0
#: g CO2e of hardware manufacture per card-hour: embodied emissions ÷ expected
#: lifetime in hours (the SCI's M, apportioned by time). 476 kg ÷ 35 040 h, see
#: ``EMBODIED_DESCRIPTION``; set on 31.10.2026 (#185, point 4). Overridable
#: with ``ATR_TRAIN_EMBODIED_G_PER_GPU_HOUR``; 0 = not counted.
DEFAULT_EMBODIED_G_PER_GPU_HOUR = 13.6
#: What the rate is made of, phrased to follow "at 13.6 g/card-hour".
EMBODIED_DESCRIPTION = ("H100 164 kg CO2e, NVIDIA's HGX H100 footprint of 1 312 kg "
                        "for eight cards, plus 312 kg server share after Luccioni "
                        "et al. 2023, over four years")
#: Where the card figure comes from (cradle-to-gate, ISO 14067, by WSP, 2025).
EMBODIED_SOURCE = ("https://images.nvidia.com/aem-dam/Solutions/documents/"
                   "HGX-H100-PCF-Summary.pdf")

#: Where the claim about the electricity comes from.
MIX_SOURCE = "https://klima.unibe.ch/betrieb/energie/index_ger.html"
#: Phrased to read correctly inside "procured electricity from … since 2016":
#: the year belongs to the sentence, not to this constant, or the card says it
#: twice — which it did until the rendered text was actually read.
MIX_DESCRIPTION = ("100 % renewable sources (hydro, biomass, solar), "
                   "predominantly Swiss")
#: The sentence every model trained on the cluster must carry, verbatim.
UBELIX_CREDIT = ("Model training was performed on UBELIX "
                 "(https://www.id.unibe.ch/hpc), the HPC cluster at the "
                 "University of Bern.")


@dataclass(frozen=True)
class Footprint:
    """What a run drew, what that is worth in carbon, and under which assumptions.

    Constructed from a measured energy figure. Every parameter is carried so the
    result is reproducible and recomputable: a number whose factor is implicit
    cannot be corrected later, only replaced.
    """

    energy_wh: float
    factor_g_per_kwh: float = DEFAULT_FACTOR_G_PER_KWH
    pue: float = DEFAULT_PUE
    #: Card-hours the run held, over the same intervals as ``energy_wh``.
    gpu_hours: float = 0.0
    embodied_g_per_gpu_hour: float = DEFAULT_EMBODIED_G_PER_GPU_HOUR
    #: True when a card's wattage was shared with another process, so the energy
    #: is the card's and not only this job's.
    shared: bool = False
    #: False when no wattage could be read at all — then this is not a floor, it
    #: is nothing, and must not be published as 0.
    measured: bool = True
    #: CPU and memory energy as **modelled** (#331): allocated cores at per-core
    #: TDP, memory at 3 W per 8 GB. Reported beside the measured figure and never
    #: inside it — ``kwh`` and the carbon figures stay GPU-only. 0 = not modelled.
    cpu_wh_modelled: float = 0.0
    ram_wh_modelled: float = 0.0
    cpu_w_per_core: float | None = None

    @property
    def modelled_kwh(self) -> float:
        """What the cores and the memory would add, by the model, at the wall."""
        return (self.cpu_wh_modelled + self.ram_wh_modelled) * self.pue / 1000.0

    @property
    def gco2e_grid_modelled(self) -> float:
        """Location-based carbon of the modelled CPU and memory share, apart."""
        return self.modelled_kwh * GRID_FACTOR_G_PER_KWH

    @property
    def kwh(self) -> float:
        """Energy at the wall, including the PUE multiplier."""
        return self.energy_wh * self.pue / 1000.0

    @property
    def gco2e(self) -> float:
        """Market-based: the factor of the electricity actually procured."""
        return self.kwh * self.factor_g_per_kwh

    @property
    def gco2e_grid(self) -> float:
        """Location-based: what the local grid mix would have cost."""
        return self.kwh * GRID_FACTOR_G_PER_KWH

    @property
    def counts_embodied(self) -> bool:
        """Both halves of M are known: a rate, and hours to apply it to."""
        return self.embodied_g_per_gpu_hour > 0 and self.gpu_hours > 0

    @property
    def embodied_g(self) -> float:
        """M: the share of the hardware's manufacture this run used up."""
        return self.gpu_hours * self.embodied_g_per_gpu_hour

    @property
    def sci_g(self) -> float:
        """E × I + M for one run, the SCI numerator with R = one trained model.

        Location-based only: the SCI does not let a renewable purchase lower the
        score, so the market-based factor never enters it.
        """
        return self.gco2e_grid + self.embodied_g

    def both(self) -> str:
        """The dual report, in one line, with the larger figure visible."""
        return (f"{self.gco2e:.0f} g CO2e market-based "
                f"({self.factor_g_per_kwh:.0f} g/kWh, {MIX_DESCRIPTION}); "
                f"{self.gco2e_grid:.0f} g location-based "
                f"({GRID_FACTOR_G_PER_KWH:.0f} g/kWh, {GRID_FACTOR_DESCRIPTION})")

    def caveats(self) -> list[str]:
        """Why the figure is a floor, in the order that matters most."""
        out = []
        if self.pue == 1.0:
            out.append("excludes data-centre overhead (PUE unknown; typical "
                       "values 1.1–1.6 would raise it by that factor)")
        if self.cpu_wh_modelled > 0:
            rate = (f" at {self.cpu_w_per_core:.1f} W per allocated core (TDP / cores)"
                    if self.cpu_w_per_core else "")
            out.append("measures GPU draw only — CPU and memory are modelled, not "
                       f"metered: {self.cpu_wh_modelled + self.ram_wh_modelled:.0f} Wh"
                       f"{rate}, kept apart from the figures above; storage and "
                       "network are not counted")
        else:
            out.append("counts GPU draw only — CPU, memory, storage and network are "
                       "not metered")
        if not self.counts_embodied:
            out.append("counts operation only — manufacture of the hardware is "
                       "not apportioned")
        if self.shared:
            out.append("a card was shared with another process, so part of this "
                       "energy is not this run's")
        return out


def from_energy(energy_wh: float, *, shared: bool = False,
                factor_g_per_kwh: float | None = None,
                pue: float | None = None, gpu_hours: float = 0.0,
                embodied_g_per_gpu_hour: float | None = None,
                cpu_wh_modelled: float = 0.0, ram_wh_modelled: float = 0.0,
                cpu_w_per_core: float | None = None) -> Footprint:
    """A footprint from a sampled energy figure, with the configured parameters.

    ``energy_wh`` of 0 yields ``measured=False``: `power.draw` answers ``[N/A]``
    on some cards and drivers, and a zero there means "not read", not "drew
    nothing". The distinction is #165's rule in its third place.
    """
    import os

    def _env(name: str, fallback: float) -> float:
        raw = os.environ.get(name)
        if not raw:
            return fallback
        try:
            return float(raw)
        except ValueError:
            return fallback

    return Footprint(
        energy_wh=energy_wh,
        factor_g_per_kwh=(factor_g_per_kwh if factor_g_per_kwh is not None
                          else _env("ATR_TRAIN_CO2_G_PER_KWH", DEFAULT_FACTOR_G_PER_KWH)),
        pue=pue if pue is not None else _env("ATR_TRAIN_PUE", DEFAULT_PUE),
        gpu_hours=gpu_hours,
        embodied_g_per_gpu_hour=(
            embodied_g_per_gpu_hour if embodied_g_per_gpu_hour is not None
            else _env("ATR_TRAIN_EMBODIED_G_PER_GPU_HOUR",
                      DEFAULT_EMBODIED_G_PER_GPU_HOUR)),
        shared=shared,
        measured=energy_wh > 0,
        cpu_wh_modelled=cpu_wh_modelled or 0.0,
        ram_wh_modelled=ram_wh_modelled or 0.0,
        cpu_w_per_core=cpu_w_per_core,
    )

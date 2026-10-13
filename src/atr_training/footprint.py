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
manufacturing panels. Published lifecycle figures put Swiss hydro at roughly
4–12 g CO2e/kWh, roof photovoltaics at roughly 40–60, biomass higher and far more
variable. A portfolio of the three lands in the low tens, which is why
:data:`DEFAULT_FACTOR_G_PER_KWH` is 20 and why it is a *parameter*: it is an
order-of-magnitude statement, not a measurement, and a run's record keeps the
figure it used so a later correction can be applied without re-running anything.

**Market-based, and the other number is larger.** Procuring certified renewable
electricity is a market instrument. The GHG Protocol asks for dual reporting when
one is used: the supplier's factor (what we pay for) *and* the grid factor where
the consumption physically happens (what the wires did). The Swiss consumer mix
is roughly 100–130 g CO2e/kWh, so a location-based figure is about five times the
market-based one. :func:`Footprint.both` reports both rather than letting the
flattering one stand alone.

**What is missing, and in which direction.** Three gaps, all of which make the
number an *under*-estimate:

* **PUE is unknown.** Cooling and power conversion in the data centre are not in
  `power.draw`. Typical values are 1.1 to 1.6. :data:`DEFAULT_PUE` is 1.0 — not
  because it is right, but because inventing 1.3 would dress a guess as a fact.
  The card says the figure excludes it.
* **Only the GPUs are metered.** CPU, memory, storage and network draw power that
  `power.draw` does not see. For a GPU training run the cards dominate, but
  3,022 CPU-hours against 163.5 GPU-hours (#184) says it is not nothing.
* **Hardware manufacture is not counted.** Scope 3 for an H100 is on the order of
  a tonne; apportioning it needs a lifetime and a utilisation assumption nobody
  here has made yet.

So a published figure is a floor, and the card states that in words rather than
leaving a reader to assume completeness.
"""
from __future__ import annotations

from dataclasses import dataclass

#: g CO2e per kWh for the university's procured mix — 100 % renewable, mostly
#: Swiss. A choice, not a measurement; see the module docstring. Overridable with
#: ``ATR_TRAIN_CO2_G_PER_KWH`` so a documented correction needs no code change.
DEFAULT_FACTOR_G_PER_KWH = 20.0
#: The Swiss consumer grid mix, for the location-based half of the dual report.
#: Also a round figure from published ranges (~100–130), not a cited point value.
GRID_FACTOR_G_PER_KWH = 115.0
#: 1.0 = the data centre's own overhead is NOT included, because nobody has given
#: us its PUE. Overridable with ``ATR_TRAIN_PUE``.
DEFAULT_PUE = 1.0

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
    #: True when a card's wattage was shared with another process, so the energy
    #: is the card's and not only this job's.
    shared: bool = False
    #: False when no wattage could be read at all — then this is not a floor, it
    #: is nothing, and must not be published as 0.
    measured: bool = True

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

    def both(self) -> str:
        """The dual report, in one line, with the larger figure visible."""
        return (f"{self.gco2e:.0f} g CO2e market-based "
                f"({self.factor_g_per_kwh:.0f} g/kWh, {MIX_DESCRIPTION}); "
                f"{self.gco2e_grid:.0f} g location-based "
                f"({GRID_FACTOR_G_PER_KWH:.0f} g/kWh, Swiss consumer mix)")

    def caveats(self) -> list[str]:
        """Why the figure is a floor, in the order that matters most."""
        out = []
        if self.pue == 1.0:
            out.append("excludes data-centre overhead (PUE unknown; typical "
                       "values 1.1–1.6 would raise it by that factor)")
        out.append("counts GPU draw only — CPU, memory, storage and network are "
                   "not metered")
        out.append("counts operation only — manufacture of the hardware is not "
                   "apportioned")
        if self.shared:
            out.append("a card was shared with another process, so part of this "
                       "energy is not this run's")
        return out


def from_energy(energy_wh: float, *, shared: bool = False,
                factor_g_per_kwh: float | None = None,
                pue: float | None = None) -> Footprint:
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
        shared=shared,
        measured=energy_wh > 0,
    )

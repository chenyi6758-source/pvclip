"""Techno-economics for a DC/AC sizing decision.

All money is in generic currency units; the example in ``examples/`` uses
USD with explicitly stated, illustrative cost assumptions — not market
data. Energy in year ``t`` (1-based) degrades geometrically at
``degradation`` per year. LCOE and NPV both discount at ``discount_rate``.

Definitions
-----------
* CAPEX = ``dc_cost_per_kw * Pdc + ac_cost_per_kw * Pac + fixed_cost``
* Annual O&M = ``opex_per_kw_dc * Pdc`` (flat, not discounted in the
  payback calculation, discounted in LCOE/NPV).
* Revenue in year ``t`` = ``annual_energy_kwh * (1 - degradation) ** (t - 1)
  * electricity_price``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Economics:
    dc_cost_per_kw: float = 350.0  # modules + DC BOS per kWdc
    ac_cost_per_kw: float = 120.0  # inverter per kWac
    fixed_cost: float = 0.0
    opex_per_kw_dc: float = 8.0  # per kWdc per year
    lifetime_years: int = 25
    discount_rate: float = 0.05
    electricity_price: float = 0.12  # per kWh
    degradation: float = 0.005  # fractional energy loss per year

    def __post_init__(self):
        if self.dc_cost_per_kw < 0 or self.ac_cost_per_kw < 0 or self.fixed_cost < 0:
            raise ValueError("costs must be non-negative")
        if self.opex_per_kw_dc < 0:
            raise ValueError("opex must be non-negative")
        if self.lifetime_years < 1:
            raise ValueError("lifetime_years must be >= 1")
        if not 0.0 <= self.discount_rate < 1.0:
            raise ValueError("discount_rate must be in [0, 1)")
        if self.electricity_price < 0:
            raise ValueError("electricity_price must be non-negative")
        if not 0.0 <= self.degradation < 1.0:
            raise ValueError("degradation must be in [0, 1)")

    def capex(self, pdc_kw: float, pac_kw: float) -> float:
        if pdc_kw < 0 or pac_kw < 0:
            raise ValueError("capacities must be non-negative")
        return (
            self.dc_cost_per_kw * pdc_kw
            + self.ac_cost_per_kw * pac_kw
            + self.fixed_cost
        )

    def _discount_factor(self, year: int) -> float:
        return 1.0 / (1.0 + self.discount_rate) ** year

    def discounted_energy(self, annual_energy_kwh: float) -> float:
        """Present value of lifetime energy (kWh), with degradation."""
        return sum(
            annual_energy_kwh * (1.0 - self.degradation) ** (t - 1) * self._discount_factor(t)
            for t in range(1, self.lifetime_years + 1)
        )

    def discounted_opex(self, pdc_kw: float) -> float:
        annual = self.opex_per_kw_dc * pdc_kw
        return sum(annual * self._discount_factor(t) for t in range(1, self.lifetime_years + 1))

    def discounted_revenue(self, annual_energy_kwh: float) -> float:
        return self.discounted_energy(annual_energy_kwh) * self.electricity_price


def lcoe(annual_energy_kwh: float, pdc_kw: float, pac_kw: float, econ: Economics) -> float:
    """Levelised cost of energy (currency/kWh). Infinite if no energy."""
    energy_pv = econ.discounted_energy(annual_energy_kwh)
    if energy_pv <= 0:
        return math.inf
    return (econ.capex(pdc_kw, pac_kw) + econ.discounted_opex(pdc_kw)) / energy_pv


def npv(annual_energy_kwh: float, pdc_kw: float, pac_kw: float, econ: Economics) -> float:
    """Net present value over the system lifetime."""
    return (
        econ.discounted_revenue(annual_energy_kwh)
        - econ.capex(pdc_kw, pac_kw)
        - econ.discounted_opex(pdc_kw)
    )


def payback_years(
    annual_energy_kwh: float, pdc_kw: float, pac_kw: float, econ: Economics
) -> float:
    """Discounted payback in years; ``math.inf`` if never repaid."""
    capex = econ.capex(pdc_kw, pac_kw)
    annual_opex = econ.opex_per_kw_dc * pdc_kw
    cumulative = -capex
    for t in range(1, econ.lifetime_years + 1):
        energy_t = annual_energy_kwh * (1.0 - econ.degradation) ** (t - 1)
        cash = (energy_t * econ.electricity_price - annual_opex) * econ._discount_factor(t)
        previous = cumulative
        cumulative += cash
        if cumulative >= 0 and cash > 0:
            # Linear interpolation within the payback year.
            return (t - 1) + (-previous / cash)
    return math.inf


def evaluate(annual_energy_kwh: float, pdc_kw: float, pac_kw: float, econ: Economics) -> dict:
    """One-row economics summary for a sizing candidate."""
    capex = econ.capex(pdc_kw, pac_kw)
    return {
        "pdc_kw": float(pdc_kw),
        "pac_kw": float(pac_kw),
        "capex": capex,
        "annual_energy_kwh": float(annual_energy_kwh),
        "lcoe": lcoe(annual_energy_kwh, pdc_kw, pac_kw, econ),
        "npv": npv(annual_energy_kwh, pdc_kw, pac_kw, econ),
        "payback_years": payback_years(annual_energy_kwh, pdc_kw, pac_kw, econ),
    }

"""Ratio sweeps and time-resolution bias.

A DC/AC sweep answers the design question directly: fix the inverter,
grow the array, and watch annual energy rise with diminishing returns
while clipped energy grows faster than linearly. The optimum is whatever
minimises LCOE (or maximises NPV) under stated costs — the literature
finds it anywhere from ~1.2 to ~1.9 depending on climate and prices,
which is why the sweep is over *your* series and *your* costs, not a
universal rule of thumb.

The resolution check quantifies a known modelling trap: averaging a
spiky DC series to hourly resolution smooths away the peaks that cause
clipping, so hourly models understate clipped energy relative to
5-minute data. :func:`resolution_bias` measures that gap on your series.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .clipping import clip_series
from .economics import Economics, evaluate


def sweep_ratios(
    dc_per_kwp_kw,
    ratios: Sequence[float],
    pac_kw: float = 1.0,
    dt_h: float = 1.0,
    efficiency=None,
    econ: Optional[Economics] = None,
) -> pd.DataFrame:
    """Sweep DC/AC ratios for a fixed inverter size.

    Parameters
    ----------
    dc_per_kwp_kw : array-like
        DC power per kWp of array rating at each timestep (kW/kWp). Scale
        it by the array size under each ratio.
    ratios : DC/AC ratios to evaluate, each > 0.
    pac_kw : inverter AC rating, kW.
    econ : Economics; defaults to :class:`Economics` defaults.
    """
    per_kwp = np.asarray(dc_per_kwp_kw, dtype=float).ravel()
    if per_kwp.size == 0:
        raise ValueError("dc_per_kwp_kw must not be empty")
    if np.any(~np.isfinite(per_kwp)) or np.any(per_kwp < 0):
        raise ValueError("dc_per_kwp_kw must be finite and non-negative")
    if pac_kw <= 0:
        raise ValueError("pac_kw must be positive")
    econ = econ if econ is not None else Economics()

    rows = []
    for ratio in ratios:
        ratio = float(ratio)
        if ratio <= 0:
            raise ValueError(f"ratios must be positive, got {ratio}")
        pdc = ratio * pac_kw
        res = clip_series(per_kwp * pdc, pac_kw, dt_h=dt_h, efficiency=efficiency)
        m = res.metrics
        eco = evaluate(m["ac_energy_kwh"], pdc, pac_kw, econ)
        rows.append(
            {
                "ratio": ratio,
                "pdc_kw": pdc,
                "pac_kw": float(pac_kw),
                "ac_energy_kwh": m["ac_energy_kwh"],
                "specific_yield_kwh_per_kwp": m["ac_energy_kwh"] / pdc,
                "clipped_kwh": m["clipped_kwh"],
                "clipped_fraction": m["clipped_fraction"],
                "hours_clipped": m["hours_clipped"],
                "capacity_factor": m["capacity_factor"],
                "capex": eco["capex"],
                "lcoe": eco["lcoe"],
                "npv": eco["npv"],
                "payback_years": eco["payback_years"],
            }
        )
    return pd.DataFrame(rows)


def optimum(table: pd.DataFrame, criterion: str = "lcoe") -> pd.Series:
    """Best row of a sweep table: minimum LCOE or maximum NPV."""
    if criterion == "lcoe":
        return table.loc[table["lcoe"].idxmin()]
    if criterion == "npv":
        return table.loc[table["npv"].idxmax()]
    raise ValueError(f"criterion must be 'lcoe' or 'npv', got {criterion!r}")


def resolution_bias(
    dc_kw,
    ac_rating_kw: float,
    dt_h: float = 1.0,
    factors: Sequence[int] = (1, 2, 4, 12),
    efficiency=None,
) -> pd.DataFrame:
    """Clipped fraction at coarser time resolutions (block-averaged DC).

    ``factors`` are aggregation counts relative to the input series: factor
    12 on a 5-minute series is hourly. Block-averaging preserves total DC
    energy (up to a trimmed tail) but removes peaks, so the clipped
    fraction normally falls as resolution coarsens. ``underestimate_pct``
    is measured against the finest resolution in the table.
    """
    dc = np.asarray(dc_kw, dtype=float).ravel()
    rows = []
    for factor in factors:
        factor = int(factor)
        if factor < 1:
            raise ValueError("factors must be >= 1")
        n = (dc.size // factor) * factor
        if n == 0:
            raise ValueError("series shorter than aggregation factor")
        coarse = dc[:n].reshape(-1, factor).mean(axis=1)
        res = clip_series(coarse, ac_rating_kw, dt_h=dt_h * factor, efficiency=efficiency)
        rows.append(
            {
                "factor": factor,
                "dt_h": dt_h * factor,
                "clipped_kwh": res.metrics["clipped_kwh"],
                "clipped_fraction": res.metrics["clipped_fraction"],
            }
        )
    table = pd.DataFrame(rows)
    finest = table["clipped_fraction"].iloc[0]
    if finest > 0:
        table["underestimate_pct"] = (
            (finest - table["clipped_fraction"]) / finest * 100.0
        )
    else:
        table["underestimate_pct"] = 0.0
    return table

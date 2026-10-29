"""Recover clipped energy with a DC-coupled battery.

The battery charges *only* from DC power the inverter refused
(``dc_excess_kw`` from :func:`pvclip.clip_series`) and discharges whenever
no power is being refused — a causal, greedy rule with no foresight and no
optimiser. That makes the result an achievable lower bound for a
DC-coupled retrofit, not an optimised dispatch: a controller that
anticipates the evening peak could do better, never worse, on recovered
energy alone (it could do worse on battery wear, which this module does
not model — see the README honesty notes).

Round-trip efficiency is split symmetrically: ``eta_c = eta_d = sqrt(RTE)``.
State of charge is stored energy (kWh) and is kept within ``[0, cap]`` by
construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class CaptureResult:
    recovered_kwh: float
    excess_kwh: float
    capture_fraction: float  # recovered / refused DC energy
    soc_kwh: np.ndarray
    charged_kw: np.ndarray  # DC power into the battery
    delivered_kw: np.ndarray  # power delivered back (AC-equivalent)
    dt_h: float
    metrics: dict = field(default_factory=dict)


def capture_clipped(
    dc_excess_kw,
    battery_kwh: float,
    battery_kw: float,
    rte: float = 0.90,
    dt_h: float = 1.0,
    discharge_mask=None,
) -> CaptureResult:
    """Simulate DC-coupled capture of refused (clipped) DC power.

    Parameters
    ----------
    dc_excess_kw : array-like, kW, >= 0
        DC power the inverter refused at each timestep.
    battery_kwh, battery_kw : float, > 0
        Usable energy capacity and charge/discharge power limit.
    rte : float in (0, 1]
        Round-trip efficiency, split equally between charge and discharge.
    dt_h : float, hours, > 0.
    discharge_mask : array-like of bool, optional
        Timesteps where discharging is allowed. Defaults to "whenever no
        power is being refused" (``dc_excess_kw == 0``).
    """
    excess = np.asarray(dc_excess_kw, dtype=float).ravel()
    if excess.size == 0:
        raise ValueError("dc_excess_kw must not be empty")
    if np.any(~np.isfinite(excess)) or np.any(excess < 0):
        raise ValueError("dc_excess_kw must be finite and non-negative")
    if not np.isfinite(battery_kwh) or battery_kwh <= 0:
        raise ValueError(f"battery_kwh must be positive, got {battery_kwh}")
    if not np.isfinite(battery_kw) or battery_kw <= 0:
        raise ValueError(f"battery_kw must be positive, got {battery_kw}")
    if not 0.0 < rte <= 1.0:
        raise ValueError(f"rte must be in (0, 1], got {rte}")
    if not np.isfinite(dt_h) or dt_h <= 0:
        raise ValueError(f"dt_h must be positive, got {dt_h}")
    if discharge_mask is None:
        can_discharge = excess == 0
    else:
        can_discharge = np.asarray(discharge_mask, dtype=bool).ravel()
        if can_discharge.shape != excess.shape:
            raise ValueError("discharge_mask must match dc_excess_kw in length")

    eta = float(np.sqrt(rte))
    soc = 0.0
    soc_series = np.zeros_like(excess)
    charged = np.zeros_like(excess)
    delivered = np.zeros_like(excess)
    for i, ex in enumerate(excess):
        if ex > 0:
            headroom_kw = (battery_kwh - soc) / (eta * dt_h) if eta > 0 else 0.0
            charge = min(float(ex), float(battery_kw), max(headroom_kw, 0.0))
            charged[i] = charge
            soc += charge * eta * dt_h
        elif can_discharge[i] and soc > 0:
            from_storage = min(float(battery_kw), soc / dt_h)
            delivered[i] = from_storage * eta
            soc -= from_storage * dt_h
        soc = float(np.clip(soc, 0.0, battery_kwh))
        soc_series[i] = soc

    recovered = float(np.sum(delivered) * dt_h)
    excess_kwh = float(np.sum(excess) * dt_h)
    capture_fraction = recovered / excess_kwh if excess_kwh > 0 else 0.0
    metrics = {
        "recovered_kwh": recovered,
        "excess_kwh": excess_kwh,
        "capture_fraction": capture_fraction,
        "max_soc_kwh": float(np.max(soc_series)),
        "final_soc_kwh": float(soc_series[-1]),
        "charged_kwh": float(np.sum(charged) * dt_h),
    }
    return CaptureResult(
        recovered_kwh=recovered,
        excess_kwh=excess_kwh,
        capture_fraction=capture_fraction,
        soc_kwh=soc_series,
        charged_kw=charged,
        delivered_kw=delivered,
        dt_h=float(dt_h),
        metrics=metrics,
    )


def capture_sweep(
    dc_excess_kw,
    sizes_kwh: Sequence[float],
    battery_kw: float,
    rte: float = 0.90,
    dt_h: float = 1.0,
    discharge_mask=None,
) -> pd.DataFrame:
    """Recovered energy vs. battery size, for one refused-power series."""
    rows = []
    for size in sizes_kwh:
        res = capture_clipped(
            dc_excess_kw, float(size), battery_kw, rte=rte, dt_h=dt_h,
            discharge_mask=discharge_mask,
        )
        rows.append(
            {
                "battery_kwh": float(size),
                "recovered_kwh": res.recovered_kwh,
                "excess_kwh": res.excess_kwh,
                "capture_fraction": res.capture_fraction,
            }
        )
    return pd.DataFrame(rows)


def battery_for_capture(
    dc_excess_kw,
    sizes_kwh: Sequence[float],
    battery_kw: float,
    target_fraction: float = 0.90,
    rte: float = 0.90,
    dt_h: float = 1.0,
    discharge_mask=None,
) -> Optional[float]:
    """Smallest battery size reaching ``target_fraction`` capture, or None."""
    if not 0.0 < target_fraction <= 1.0:
        raise ValueError("target_fraction must be in (0, 1]")
    table = capture_sweep(
        dc_excess_kw, sizes_kwh, battery_kw, rte=rte, dt_h=dt_h,
        discharge_mask=discharge_mask,
    )
    hit = table.loc[table["capture_fraction"] >= target_fraction]
    if hit.empty:
        return None
    return float(hit["battery_kwh"].iloc[0])

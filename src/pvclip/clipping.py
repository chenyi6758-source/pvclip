"""Inverter clipping: cap a DC power series at the inverter AC rating.

Energy accounting (per timestep, ``eta`` = DC-to-AC efficiency at that load):

* ``ac_uncapped = dc * eta`` — what the inverter would deliver without a limit.
* ``ac = min(ac_uncapped, ac_rating)`` — what it actually delivers.
* ``clipped_ac = ac_uncapped - ac`` — the clipping loss, at the AC meter.
* ``dc_used = ac / eta`` and ``dc_excess = dc - dc_used`` — the DC power the
  array could have delivered but the inverter refused. Identity:
  ``clipped_ac == eta * dc_excess`` at every timestep.

The AC cap follows the Sandia grid-connected inverter model as implemented
in pvlib's ``snlinverter``: modelled AC power is clipped at the rated output
``Pac0``. pvclip does not re-implement the Sandia model itself; the efficiency
curves here are transparent stand-ins (constant, or a quadratic partial-load
shape) and any user callable ``eta(load_fraction)`` is accepted, e.g. a curve
digitised from an inverter datasheet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

EfficiencyFn = Callable[[np.ndarray], np.ndarray]


def efficiency_constant(eta: float = 0.97) -> EfficiencyFn:
    """Flat efficiency, independent of load. ``eta`` must be in (0, 1]."""
    if not 0.0 < eta <= 1.0:
        raise ValueError(f"eta must be in (0, 1], got {eta}")

    def _fn(load: np.ndarray) -> np.ndarray:
        return np.full(np.shape(load), float(eta), dtype=float)

    return _fn


def efficiency_quadratic(
    eta_peak: float = 0.98,
    peak_load: float = 0.5,
    low_load_loss: float = 0.08,
) -> EfficiencyFn:
    """Quadratic partial-load efficiency shape.

    Efficiency peaks at ``eta_peak`` when the load fraction equals
    ``peak_load``, and falls quadratically towards both ends: at zero load it
    is ``eta_peak - low_load_loss``. This is an illustrative shape for
    sensitivity studies, not a fitted datasheet curve — pass your own
    callable for a real inverter.
    """
    if not 0.0 < eta_peak <= 1.0:
        raise ValueError(f"eta_peak must be in (0, 1], got {eta_peak}")
    if not 0.0 < peak_load < 1.0:
        raise ValueError(f"peak_load must be in (0, 1), got {peak_load}")
    if not 0.0 <= low_load_loss < eta_peak:
        raise ValueError(f"low_load_loss must be in [0, eta_peak), got {low_load_loss}")
    a = low_load_loss / peak_load**2

    def _fn(load: np.ndarray) -> np.ndarray:
        load = np.asarray(load, dtype=float)
        eta = eta_peak - a * (load - peak_load) ** 2
        return np.clip(eta, 0.05, 1.0)

    return _fn


@dataclass
class ClipResult:
    """Per-timestep clipping outcome plus summary metrics."""

    ac_kw: np.ndarray
    ac_uncapped_kw: np.ndarray
    clipped_kw: np.ndarray  # AC-equivalent clipped power (kW)
    dc_excess_kw: np.ndarray  # DC power refused by the inverter (kW)
    efficiency: np.ndarray
    dt_h: float
    ac_rating_kw: float
    metrics: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """Return a copy of the summary metrics dictionary."""
        return dict(self.metrics)


def clip_series(
    dc_kw,
    ac_rating_kw: float,
    dt_h: float = 1.0,
    efficiency: Optional[EfficiencyFn] = None,
    dc_rated_kw: Optional[float] = None,
) -> ClipResult:
    """Clip a DC power series at an inverter AC rating.

    Parameters
    ----------
    dc_kw : array-like, kW, >= 0
        DC power delivered by the array at each timestep.
    ac_rating_kw : float, kW, > 0
        Inverter rated AC output (``Pac0``).
    dt_h : float, hours, > 0
        Timestep length, used only for energy metrics.
    efficiency : callable, optional
        ``eta(load_fraction)`` with ``load_fraction = dc_kw / dc_rated_kw``.
        Defaults to a constant 97% efficiency.
    dc_rated_kw : float, optional
        DC power at which the inverter reaches its AC rating. Defaults to
        ``ac_rating_kw / 0.97``. Only used to normalise the load fraction
        passed to the efficiency function.

    Returns
    -------
    ClipResult
        ``ac_kw`` never exceeds ``ac_rating_kw``; ``clipped_kw`` is zero
        whenever the uncapped output is below the rating.
    """
    dc = np.asarray(dc_kw, dtype=float).ravel()
    if dc.size == 0:
        raise ValueError("dc_kw must not be empty")
    if np.any(~np.isfinite(dc)) or np.any(dc < 0):
        raise ValueError("dc_kw must be finite and non-negative")
    if not np.isfinite(ac_rating_kw) or ac_rating_kw <= 0:
        raise ValueError(f"ac_rating_kw must be positive, got {ac_rating_kw}")
    if not np.isfinite(dt_h) or dt_h <= 0:
        raise ValueError(f"dt_h must be positive, got {dt_h}")
    if dc_rated_kw is None:
        dc_rated_kw = ac_rating_kw / 0.97
    if not np.isfinite(dc_rated_kw) or dc_rated_kw <= 0:
        raise ValueError(f"dc_rated_kw must be positive, got {dc_rated_kw}")

    eta_fn = efficiency if efficiency is not None else efficiency_constant()
    load = dc / dc_rated_kw
    eta = np.asarray(eta_fn(load), dtype=float).ravel()
    if eta.shape != dc.shape:
        raise ValueError("efficiency function must return one value per timestep")
    if np.any(~np.isfinite(eta)) or np.any(eta <= 0) or np.any(eta > 1.0):
        raise ValueError("efficiency values must be finite and in (0, 1]")

    ac_uncapped = dc * eta
    ac = np.minimum(ac_uncapped, ac_rating_kw)
    clipped = ac_uncapped - ac
    dc_excess = dc - ac / eta
    # Guard against -0.0 / fp noise on unclipped steps.
    dc_excess = np.where(clipped > 0, dc_excess, 0.0)

    total_h = dt_h * dc.size
    ac_energy = float(np.sum(ac) * dt_h)
    uncapped_energy = float(np.sum(ac_uncapped) * dt_h)
    clipped_energy = float(np.sum(clipped) * dt_h)
    metrics = {
        "ac_energy_kwh": ac_energy,
        "dc_energy_kwh": float(np.sum(dc) * dt_h),
        "uncapped_ac_energy_kwh": uncapped_energy,
        "clipped_kwh": clipped_energy,
        "clipped_fraction": clipped_energy / uncapped_energy if uncapped_energy > 0 else 0.0,
        "hours_clipped": float(np.sum(clipped > 0) * dt_h),
        "n_steps_clipped": int(np.sum(clipped > 0)),
        "max_clip_kw": float(np.max(clipped)) if clipped.size else 0.0,
        "capacity_factor": ac_energy / (ac_rating_kw * total_h) if total_h > 0 else 0.0,
        "mean_ac_kw": float(np.mean(ac)),
    }
    return ClipResult(
        ac_kw=ac,
        ac_uncapped_kw=ac_uncapped,
        clipped_kw=clipped,
        dc_excess_kw=dc_excess,
        efficiency=eta,
        dt_h=float(dt_h),
        ac_rating_kw=float(ac_rating_kw),
        metrics=metrics,
    )

"""Array DC power from plane-of-array irradiance and temperature.

Two temperature conventions, kept separate on purpose:

* :func:`dc_power_from_cell_temp` takes the *cell* temperature directly.
  At STC (1000 W/m2, 25 C cell, no extra losses) it returns exactly the
  rated DC power — the identity the tests pin down.
* :func:`dc_power` takes the *ambient* temperature and estimates the cell
  temperature with the NOCT model. At 1000 W/m2 it therefore returns less
  than rated power, because a real cell under full sun runs hotter than
  25 C. That is physics, not a bug.

The power model is the standard linear form used across the clipping
literature (e.g. the temperature-corrected model in the bifacial clipping
study, Kaewnukultorn et al. 2024): ``P = P_stc * (G / 1000) *
(1 + gamma * (T_cell - 25)) * (1 - losses)``.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


def cell_temperature(poa_wm2, temp_amb_c, noct: float = 45.0) -> np.ndarray:
    """NOCT cell-temperature estimate: ``T_cell = T_amb + (NOCT-20)/800 * G``."""
    poa = np.asarray(poa_wm2, dtype=float)
    tamb = np.asarray(temp_amb_c, dtype=float)
    if np.any(poa < 0):
        raise ValueError("poa_wm2 must be non-negative")
    return tamb + (float(noct) - 20.0) / 800.0 * poa


def dc_power_from_cell_temp(
    poa_wm2,
    temp_cell_c,
    pdc_kw: float,
    gamma_per_c: float = -0.004,
    losses: float = 0.02,
) -> np.ndarray:
    """DC power from POA irradiance and cell temperature (kW)."""
    poa = np.asarray(poa_wm2, dtype=float)
    tcell = np.asarray(temp_cell_c, dtype=float)
    if np.any(poa < 0):
        raise ValueError("poa_wm2 must be non-negative")
    if pdc_kw <= 0:
        raise ValueError(f"pdc_kw must be positive, got {pdc_kw}")
    if not 0.0 <= losses < 1.0:
        raise ValueError(f"losses must be in [0, 1), got {losses}")
    power = (
        pdc_kw
        * (poa / 1000.0)
        * (1.0 + gamma_per_c * (tcell - 25.0))
        * (1.0 - losses)
    )
    return np.maximum(power, 0.0)


def dc_power(
    poa_wm2,
    temp_amb_c,
    pdc_kw: float,
    gamma_per_c: float = -0.004,
    noct: float = 45.0,
    losses: float = 0.02,
) -> np.ndarray:
    """DC power from POA irradiance and ambient temperature (kW)."""
    tcell = cell_temperature(poa_wm2, temp_amb_c, noct=noct)
    return dc_power_from_cell_temp(
        poa_wm2, tcell, pdc_kw, gamma_per_c=gamma_per_c, losses=losses
    )


def dc_power_mix(
    poa_by_orientation: Mapping[str, object],
    shares: Mapping[str, float],
    temp_amb_c,
    pdc_kw: float,
    **kwargs,
) -> np.ndarray:
    """DC power of a mixed-orientation array.

    ``poa_by_orientation`` maps an orientation label to its POA series and
    ``shares`` maps the same labels to capacity fractions summing to 1.
    East/west splits flatten the midday peak, which is exactly why mixed
    roofs clip less at the same DC/AC ratio.
    """
    labels = set(poa_by_orientation) | set(shares)
    if set(poa_by_orientation) != set(shares):
        raise ValueError("poa_by_orientation and shares must have the same keys")
    total_share = sum(float(v) for v in shares.values())
    if abs(total_share - 1.0) > 1e-9:
        raise ValueError(f"shares must sum to 1, got {total_share}")
    if any(v < 0 for v in shares.values()):
        raise ValueError("shares must be non-negative")
    total = None
    for label in poa_by_orientation:
        part = dc_power(
            poa_by_orientation[label], temp_amb_c, pdc_kw * float(shares[label]), **kwargs
        )
        total = part if total is None else total + part
    return total

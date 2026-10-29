"""Weather inputs: a seeded synthetic year, CSV loading, and an optional
pvlib clear-sky path.

The synthetic generator exists so every example and test runs offline and
reproducibly. It is a *first-order* clear-sky bell curve with seeded cloud
variability — good enough to exercise clipping dynamics, not a substitute
for measured or transposed irradiance. For real sites, bring your own POA
series via :func:`load_series_csv` or the optional pvlib clear-sky helper,
which needs pvlib installed and performs proper solar-position and
transposition calculations.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def synthetic_year(
    days: int = 365,
    dt_h: float = 1.0,
    latitude: float = 31.2,
    azimuth_deg: float = 180.0,
    peak_poa: float = 1000.0,
    cloudiness: float = 0.25,
    annual_mean_temp: float = 17.0,
    seasonal_temp_amp: float = 11.0,
    diurnal_temp_amp: float = 4.0,
    seed: int = 42,
    start: str = "2025-01-01",
) -> pd.DataFrame:
    """Seeded synthetic hourly POA irradiance and ambient temperature.

    Parameters
    ----------
    days, dt_h : year length and timestep.
    latitude : degrees, used only for the day-length (declination) curve.
    azimuth_deg : 180 = south (northern hemisphere). East/west arrays get
        an earlier/later, slightly lower peak — a first-order stand-in for
        orientation, not a transposition model.
    peak_poa : clear-sky POA peak at the summer solstice, W/m2.
    cloudiness : 0 = always clear, 1 = heavily attenuated. Scales the
        seeded cloud factor.
    seed : identical seeds give identical series.

    Returns a DataFrame indexed by timestamp with ``poa_wm2`` and
    ``temp_c`` (ambient) columns.
    """
    if days <= 0:
        raise ValueError("days must be positive")
    if dt_h <= 0:
        raise ValueError("dt_h must be positive")
    if not 0.0 <= cloudiness <= 1.0:
        raise ValueError("cloudiness must be in [0, 1]")
    rng = np.random.default_rng(seed)

    n_steps = int(round(days * 24 / dt_h))
    times = pd.date_range(start, periods=n_steps, freq=pd.Timedelta(hours=dt_h))
    doy = times.dayofyear.to_numpy(dtype=float)
    hour = times.hour.to_numpy(dtype=float) + times.minute.to_numpy(dtype=float) / 60.0

    # Solar declination (Cooper) and day length from latitude.
    decl = np.radians(-23.44 * np.cos(2 * np.pi * (doy + 10) / 365.0))
    lat = np.radians(latitude)
    cos_omega = -np.tan(lat) * np.tan(decl)
    cos_omega = np.clip(cos_omega, -1.0, 1.0)
    day_length = 2 * np.degrees(np.arccos(cos_omega)) / 15.0  # hours
    sunrise = 12.0 - day_length / 2.0
    sunset = 12.0 + day_length / 2.0

    # Orientation: shift and scale the bell curve.
    shift = float(np.clip((180.0 - azimuth_deg) / 45.0, -2.5, 2.5))
    peak_scale = 0.90 + 0.10 * math.cos(math.radians(azimuth_deg - 180.0))
    seasonal = 0.80 + 0.20 * np.cos(2 * np.pi * (doy - 172) / 365.0)
    frac = (hour + shift - sunrise) / np.maximum(day_length, 1e-9)
    bell = np.where((frac > 0) & (frac < 1), np.sin(np.pi * frac), 0.0)
    clear_poa = peak_poa * seasonal * peak_scale * bell

    # Seeded cloud variability: smoothed normal noise, hourly scale.
    window = max(1, int(round(3.0 / dt_h)))
    noise = rng.standard_normal(n_steps)
    kernel = np.ones(window) / window
    smooth = np.convolve(noise, kernel, mode="same")
    smooth = smooth / (np.std(smooth) + 1e-12)
    cloud = np.clip(1.0 - cloudiness * np.clip(0.45 + 0.55 * smooth, 0.0, 1.7), 0.05, 1.0)
    poa = np.maximum(clear_poa * cloud, 0.0)
    poa = np.where(bell > 0, poa, 0.0)

    daily_mean = annual_mean_temp + seasonal_temp_amp * np.cos(
        2 * np.pi * (doy - 200) / 365.0
    )
    diurnal = diurnal_temp_amp * np.sin(2 * np.pi * (hour - 9.0) / 24.0)
    temp = daily_mean + diurnal + rng.normal(0.0, 0.5, n_steps)

    return pd.DataFrame(
        {"poa_wm2": poa, "temp_c": temp},
        index=pd.DatetimeIndex(times),
    )


def load_series_csv(
    path,
    time_col: str = "time",
    poa_col: str = "poa_wm2",
    temp_col: str = "temp_c",
) -> Tuple[pd.DataFrame, float]:
    """Load a user POA/temperature series from CSV.

    Returns ``(df, dt_h)`` where ``df`` has a DatetimeIndex and
    ``poa_wm2``/``temp_c`` columns, sorted by time, and ``dt_h`` is the
    median timestep in hours.
    """
    df = pd.read_csv(Path(path))
    missing = [c for c in (time_col, poa_col, temp_col) if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    df = df.rename(columns={time_col: "time", poa_col: "poa_wm2", temp_col: "temp_c"})
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").set_index("time")
    if (df["poa_wm2"] < 0).any():
        raise ValueError("poa_wm2 must be non-negative")
    diffs = df.index.to_series().diff().dt.total_seconds().dropna() / 3600.0
    if len(diffs) == 0:
        raise ValueError("CSV must contain at least two rows")
    dt_h = float(diffs.median())
    if dt_h <= 0:
        raise ValueError("timestamps must be strictly increasing")
    return df[["poa_wm2", "temp_c"]].copy(), dt_h


def pvlib_clearsky(
    latitude: float,
    longitude: float,
    surface_tilt: float,
    surface_azimuth: float,
    start: str = "2025-01-01",
    days: int = 365,
    dt_h: float = 1.0,
    temp_amb_c: float = 20.0,
    wind_speed_ms: float = 1.0,
    tz: str = "UTC",
) -> pd.DataFrame:
    """Clear-sky POA irradiance and cell temperature via pvlib (optional).

    Requires pvlib. Uses the Ineichen clear-sky model with pvlib's bundled
    Linke turbidity lookup, solar position, and isotropic transposition,
    and the PVSyst cell-temperature model. Returns ``poa_wm2``,
    ``temp_amb_c`` and ``temp_cell_c`` columns — feed ``temp_cell_c`` to
    :func:`pvclip.dc_power_from_cell_temp`.
    """
    try:
        import pvlib
    except ImportError as exc:  # pragma: no cover - pvlib is optional
        raise ImportError(
            "pvlib_clearsky requires pvlib: pip install pvclip[pvlib]"
        ) from exc

    loc = pvlib.location.Location(latitude, longitude, tz=tz)
    times = pd.date_range(
        start, periods=int(round(days * 24 / dt_h)), freq=pd.Timedelta(hours=dt_h), tz=tz
    )
    clearsky = loc.get_clearsky(times)
    solpos = loc.get_solarposition(times)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt,
        surface_azimuth,
        solpos["apparent_zenith"],
        solpos["azimuth"],
        clearsky["dni"],
        clearsky["ghi"],
        clearsky["dhi"],
    )
    temp_amb = np.full(len(times), float(temp_amb_c))
    wind = np.full(len(times), float(wind_speed_ms))
    tcell = pvlib.temperature.pvsyst_cell_temperature(
        poa["poa_global"], temp_amb, wind
    )
    out = pd.DataFrame(
        {
            "poa_wm2": np.asarray(poa["poa_global"], dtype=float),
            "temp_amb_c": temp_amb,
            "temp_cell_c": np.asarray(tcell, dtype=float),
        },
        index=times,
    )
    out["poa_wm2"] = out["poa_wm2"].clip(lower=0.0)
    return out

import numpy as np
import pandas as pd
import pytest

from pvclip import load_series_csv, synthetic_year


def test_synthetic_year_shape_and_bounds():
    df = synthetic_year(days=365, dt_h=1.0, seed=7)
    assert len(df) == 8760
    assert list(df.columns) == ["poa_wm2", "temp_c"]
    assert (df["poa_wm2"] >= 0).all()
    assert df["poa_wm2"].max() <= 1000.0 + 1e-9
    assert df.index.is_monotonic_increasing
    # Nights are dark: midnight POA is zero all year.
    assert (df.between_time("00:00", "00:00")["poa_wm2"] == 0).all()
    # Summer days beat winter days on average.
    assert df.loc["2025-07"]["poa_wm2"].mean() > df.loc["2025-01"]["poa_wm2"].mean()


def test_synthetic_year_reproducible_with_seed():
    a = synthetic_year(days=30, seed=42)
    b = synthetic_year(days=30, seed=42)
    pd.testing.assert_frame_equal(a, b)
    c = synthetic_year(days=30, seed=43)
    assert not c.equals(a)


def test_synthetic_year_clear_sky_hits_peak():
    df = synthetic_year(days=365, dt_h=1.0, cloudiness=0.0, seed=1)
    assert df["poa_wm2"].max() == pytest.approx(1000.0, rel=1e-6)


def test_synthetic_year_subhourly():
    df = synthetic_year(days=2, dt_h=0.25, seed=3)
    assert len(df) == 192


def test_orientation_shifts_peak():
    east = synthetic_year(days=60, cloudiness=0.0, azimuth_deg=90.0, seed=1)
    west = synthetic_year(days=60, cloudiness=0.0, azimuth_deg=270.0, seed=1)
    # Mean profile: east peaks earlier than west.
    east_hour = east.groupby(east.index.hour)["poa_wm2"].mean().idxmax()
    west_hour = west.groupby(west.index.hour)["poa_wm2"].mean().idxmax()
    assert east_hour < west_hour


def test_invalid_weather_inputs():
    with pytest.raises(ValueError):
        synthetic_year(days=0)
    with pytest.raises(ValueError):
        synthetic_year(cloudiness=1.5)


def test_load_series_csv_roundtrip(tmp_path):
    times = pd.date_range("2025-06-01", periods=48, freq="30min")
    raw = pd.DataFrame(
        {"time": times, "poa_wm2": np.linspace(0, 800, 48), "temp_c": 25.0}
    )
    path = tmp_path / "series.csv"
    raw.to_csv(path, index=False)
    df, dt_h = load_series_csv(path)
    assert dt_h == pytest.approx(0.5)
    assert len(df) == 48
    assert df.index[0] == times[0]


def test_load_series_csv_missing_column(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"time": ["2025-01-01"], "poa_wm2": [0.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_series_csv(path)


def test_pvlib_clearsky_smoke():
    pytest.importorskip("pvlib", reason="pvlib is an optional dependency")
    from pvclip.weather import pvlib_clearsky

    df = pvlib_clearsky(31.2, 120.0, 25.0, 180.0, days=3, dt_h=1.0)
    assert len(df) == 72
    assert (df["poa_wm2"] >= 0).all()
    assert df["poa_wm2"].max() > 0

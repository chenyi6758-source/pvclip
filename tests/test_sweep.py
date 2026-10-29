import numpy as np
import pytest

from pvclip import optimum, resolution_bias, sweep_ratios
from pvclip import clip_series


def _flat_day_dc_per_kwp():
    # 12 daylight steps at 0.8 kW/kWp, 12 night steps at 0.
    return np.array([0.0] * 6 + [0.8] * 12 + [0.0] * 6)


def test_sweep_clipped_fraction_monotonic_in_ratio():
    table = sweep_ratios(_flat_day_dc_per_kwp(), [1.0, 1.3, 1.6, 2.0], pac_kw=1.0)
    fracs = table["clipped_fraction"].to_numpy()
    assert np.all(np.diff(fracs) >= -1e-12)
    assert fracs[0] == 0.0  # 0.8 kW/kWp * 1.0 < 1/0.97 cap, no clipping
    assert fracs[-1] > 0.3


def test_sweep_ac_energy_diminishing_returns():
    # Bell-shaped day: low steps keep gaining at every ratio, high steps
    # saturate early, so total gains shrink but stay positive.
    profile = np.array([0.2, 0.4, 0.6, 0.8, 0.95, 1.0, 0.95, 0.8, 0.6, 0.4, 0.2])
    table = sweep_ratios(profile, [1.0, 1.3, 1.6, 2.0], pac_kw=1.0)
    energies = table["ac_energy_kwh"].to_numpy()
    assert np.all(np.diff(energies) > 0)
    gains = np.diff(energies)
    assert np.all(np.diff(gains) <= 1e-9)  # each extra kWp adds less


def test_sweep_specific_yield_falls_with_ratio():
    table = sweep_ratios(_flat_day_dc_per_kwp(), [1.0, 1.5, 2.0], pac_kw=1.0)
    yields = table["specific_yield_kwh_per_kwp"].to_numpy()
    assert np.all(np.diff(yields) < 0)


def test_sweep_optimum_interior_for_balanced_costs():
    # With DC capacity cheap relative to energy value, optimum > 1;
    # with DC capacity expensive, optimum pulls back. Check both move.
    from pvclip import Economics, synthetic_year, dc_power

    weather = synthetic_year(days=60, seed=5)
    dc_per_kwp = dc_power(weather["poa_wm2"], weather["temp_c"], pdc_kw=1.0)
    cheap_dc = Economics(dc_cost_per_kw=50.0, ac_cost_per_kw=200.0,
                         electricity_price=0.20)
    pricey_dc = Economics(dc_cost_per_kw=900.0, ac_cost_per_kw=50.0,
                          electricity_price=0.05)
    ratios = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    t_cheap = sweep_ratios(dc_per_kwp, ratios, pac_kw=5.0, econ=cheap_dc)
    t_pricey = sweep_ratios(dc_per_kwp, ratios, pac_kw=5.0, econ=pricey_dc)
    assert optimum(t_cheap, "lcoe")["ratio"] > optimum(t_pricey, "lcoe")["ratio"]
    assert optimum(t_cheap, "npv")["ratio"] >= optimum(t_pricey, "npv")["ratio"]


def test_optimum_validation():
    table = sweep_ratios(_flat_day_dc_per_kwp(), [1.0, 1.5], pac_kw=1.0)
    with pytest.raises(ValueError):
        optimum(table, criterion="payback")


def test_sweep_invalid_inputs():
    with pytest.raises(ValueError):
        sweep_ratios([], [1.0])
    with pytest.raises(ValueError):
        sweep_ratios([1.0], [0.0])
    with pytest.raises(ValueError):
        sweep_ratios([-1.0], [1.0])


def test_resolution_bias_spiky_series():
    # 5-minute series: 11 quiet steps then a 1-step spike, repeated.
    # Hourly averaging erases the spike, so clipping must fall.
    pattern = np.array([0.2] * 11 + [3.0])
    dc = np.tile(pattern, 24)
    table = resolution_bias(dc, ac_rating_kw=1.0, dt_h=1 / 12, factors=(1, 12))
    assert table["clipped_fraction"].iloc[1] < table["clipped_fraction"].iloc[0]
    assert table["underestimate_pct"].iloc[1] > 0
    assert table["underestimate_pct"].iloc[0] == 0.0


def test_resolution_bias_no_clipping_no_bias():
    dc = np.full(24, 0.5)
    table = resolution_bias(dc, ac_rating_kw=5.0, dt_h=1.0, factors=(1, 2, 4))
    assert (table["clipped_fraction"] == 0).all()
    assert (table["underestimate_pct"] == 0).all()

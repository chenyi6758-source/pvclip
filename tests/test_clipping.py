import math

import numpy as np
import pytest

from pvclip import clip_series, efficiency_constant, efficiency_quadratic


def test_no_clipping_below_rating():
    res = clip_series([0.0, 1.0, 2.0], ac_rating_kw=5.0)
    np.testing.assert_allclose(res.ac_kw, [0.0, 0.97, 1.94])
    assert res.metrics["clipped_kwh"] == 0.0
    assert res.metrics["hours_clipped"] == 0.0
    assert res.metrics["clipped_fraction"] == 0.0


def test_ac_never_exceeds_rating_and_clipped_is_exact():
    dc = np.array([0.0, 3.0, 5.0, 6.0, 10.0])
    res = clip_series(dc, ac_rating_kw=4.0)
    assert np.all(res.ac_kw <= 4.0 + 1e-12)
    # uncapped = dc * 0.97; clipped steps: 5*0.97-4=0.85, 6*0.97-4=1.82,
    # 10*0.97-4=5.7
    np.testing.assert_allclose(res.clipped_kw, [0, 0, 0.85, 1.82, 5.7], atol=1e-9)
    assert res.metrics["hours_clipped"] == 3.0


def test_energy_identity_clipped_equals_eta_times_dc_excess():
    rng = np.random.default_rng(0)
    dc = rng.uniform(0, 8, size=500)
    res = clip_series(dc, ac_rating_kw=4.0)
    lhs = res.clipped_kw
    rhs = res.efficiency * res.dc_excess_kw
    np.testing.assert_allclose(lhs, rhs, atol=1e-9)
    # DC energy = converted DC + refused DC
    dc_kwh = res.metrics["dc_energy_kwh"]
    converted_kwh = float(np.sum(res.ac_kw / res.efficiency) * res.dt_h)
    refused_kwh = float(np.sum(res.dc_excess_kw) * res.dt_h)
    assert dc_kwh == pytest.approx(converted_kwh + refused_kwh, rel=1e-9)


def test_constant_efficiency_validation():
    with pytest.raises(ValueError):
        efficiency_constant(0.0)
    with pytest.raises(ValueError):
        efficiency_constant(1.5)


def test_quadratic_efficiency_shape():
    fn = efficiency_quadratic(eta_peak=0.98, peak_load=0.5, low_load_loss=0.08)
    eta = fn(np.array([0.0, 0.5, 1.0]))
    assert eta[1] == pytest.approx(0.98)
    assert eta[0] == pytest.approx(0.90)
    assert eta[0] < eta[1] and eta[2] < eta[1]
    assert np.all((eta > 0) & (eta <= 1.0))


def test_quadratic_efficiency_reduces_clipping_vs_constant_at_low_load():
    dc = np.array([0.5, 0.5, 0.5])
    const = clip_series(dc, 5.0, efficiency=efficiency_constant(0.98))
    quad = clip_series(dc, 5.0, efficiency=efficiency_quadratic())
    # At low load the quadratic curve is below peak efficiency.
    assert quad.metrics["ac_energy_kwh"] < const.metrics["ac_energy_kwh"]


def test_custom_efficiency_callable_is_used():
    res = clip_series([2.0], ac_rating_kw=5.0, efficiency=lambda load: np.full_like(load, 0.5))
    assert res.ac_kw[0] == pytest.approx(1.0)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        clip_series([-1.0, 2.0], 5.0)
    with pytest.raises(ValueError):
        clip_series([np.nan, 2.0], 5.0)
    with pytest.raises(ValueError):
        clip_series([], 5.0)
    with pytest.raises(ValueError):
        clip_series([1.0], 0.0)
    with pytest.raises(ValueError):
        clip_series([1.0], 5.0, dt_h=0.0)
    with pytest.raises(ValueError):
        clip_series([1.0], 5.0, efficiency=lambda load: np.array([1.5]))


def test_metrics_capacity_factor():
    # Constant 4 kW DC, 97% efficiency, 4 kW AC rating, 2 hours.
    res = clip_series([4.0, 4.0], ac_rating_kw=4.0, dt_h=1.0)
    # AC = min(3.88, 4) = 3.88 each hour -> CF = 3.88/4 = 0.97
    assert res.metrics["capacity_factor"] == pytest.approx(0.97)
    assert res.metrics["ac_energy_kwh"] == pytest.approx(7.76)
    # At the rating with DC well above it, CF -> 1.
    res2 = clip_series([10.0, 10.0], ac_rating_kw=4.0, dt_h=1.0)
    assert res2.metrics["capacity_factor"] == pytest.approx(1.0)
    assert math.isfinite(res2.metrics["clipped_fraction"])

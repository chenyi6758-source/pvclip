import numpy as np
import pytest

from pvclip import battery_for_capture, capture_clipped, capture_sweep


def test_no_excess_no_recovery():
    res = capture_clipped(np.zeros(24), battery_kwh=10.0, battery_kw=5.0)
    assert res.recovered_kwh == 0.0
    assert res.capture_fraction == 0.0
    assert np.all(res.soc_kwh == 0.0)


def test_soc_bounds_never_violated():
    rng = np.random.default_rng(1)
    excess = rng.uniform(0, 6, size=8760)
    res = capture_clipped(excess, battery_kwh=5.0, battery_kw=2.0, dt_h=1.0)
    assert np.all(res.soc_kwh >= -1e-9)
    assert np.all(res.soc_kwh <= 5.0 + 1e-9)


def test_capture_fraction_bounded_by_round_trip_efficiency():
    # One clipping hour (10 kW refused), then empty hours to discharge.
    excess = np.array([10.0] + [0.0] * 23)
    res = capture_clipped(excess, battery_kwh=100.0, battery_kw=100.0, rte=0.90)
    # Stored = 10 * sqrt(0.9); delivered = stored * sqrt(0.9) = 10 * 0.9.
    assert res.recovered_kwh == pytest.approx(9.0, rel=1e-9)
    assert res.capture_fraction == pytest.approx(0.9, rel=1e-9)


def test_capture_increases_with_battery_size():
    excess = np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    table = capture_sweep(excess, [1.0, 3.0, 10.0, 100.0], battery_kw=5.0, rte=1.0)
    fracs = table["capture_fraction"].to_numpy()
    assert np.all(np.diff(fracs) >= -1e-12)
    assert fracs[-1] == pytest.approx(1.0)


def test_small_battery_recovers_little():
    excess = np.array([10.0, 10.0, 0.0, 0.0])
    tiny = capture_clipped(excess, battery_kwh=0.5, battery_kw=10.0, rte=1.0)
    assert tiny.recovered_kwh == pytest.approx(0.5, rel=1e-9)


def test_power_limit_binds():
    # Battery can only take 1 kW during a 10 kW clipping hour.
    excess = np.array([10.0, 0.0, 0.0])
    res = capture_clipped(excess, battery_kwh=100.0, battery_kw=1.0, rte=1.0)
    assert res.recovered_kwh == pytest.approx(1.0)


def test_battery_for_capture_returns_smallest_qualifying_size():
    excess = np.array([4.0, 4.0, 0.0, 0.0, 0.0])
    size = battery_for_capture(excess, [1.0, 4.0, 8.0, 20.0], battery_kw=4.0,
                               target_fraction=0.9, rte=1.0)
    # 8 kWh stores both 4 kW hours fully; 4 kWh stores only half.
    assert size == 8.0


def test_battery_for_capture_none_when_target_unreachable():
    excess = np.array([4.0, 0.0])
    assert battery_for_capture(excess, [1.0, 2.0], battery_kw=0.5,
                               target_fraction=0.9, rte=1.0) is None


def test_invalid_capture_inputs():
    with pytest.raises(ValueError):
        capture_clipped([-1.0, 0.0], 5.0, 5.0)
    with pytest.raises(ValueError):
        capture_clipped([1.0], 0.0, 5.0)
    with pytest.raises(ValueError):
        capture_clipped([1.0], 5.0, 5.0, rte=1.5)
    with pytest.raises(ValueError):
        capture_clipped([1.0, 0.0], 5.0, 5.0, discharge_mask=[True])

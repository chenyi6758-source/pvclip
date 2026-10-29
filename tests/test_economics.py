import math

import pytest

from pvclip import Economics, evaluate, lcoe, npv, payback_years


def test_lcoe_hand_calculation_no_discount_no_degradation():
    econ = Economics(
        dc_cost_per_kw=100.0,
        ac_cost_per_kw=0.0,
        opex_per_kw_dc=0.0,
        lifetime_years=10,
        discount_rate=0.0,
        electricity_price=0.0,
        degradation=0.0,
    )
    # CAPEX 100, lifetime energy 10 * 1000 kWh -> LCOE 0.01
    assert lcoe(1000.0, pdc_kw=1.0, pac_kw=1.0, econ=econ) == pytest.approx(0.01)
    assert npv(1000.0, 1.0, 1.0, econ) == pytest.approx(-100.0)


def test_npv_positive_with_high_price():
    econ = Economics(electricity_price=1.0, discount_rate=0.0, degradation=0.0,
                     lifetime_years=10, opex_per_kw_dc=0.0,
                     dc_cost_per_kw=100.0, ac_cost_per_kw=0.0)
    assert npv(1000.0, 1.0, 1.0, econ) == pytest.approx(10 * 1000.0 - 100.0)


def test_lcoe_infinite_without_energy():
    econ = Economics()
    assert lcoe(0.0, 1.0, 1.0, econ) == math.inf


def test_discounting_reduces_present_value():
    no_disc = Economics(discount_rate=0.0, degradation=0.0, lifetime_years=10)
    disc = Economics(discount_rate=0.10, degradation=0.0, lifetime_years=10)
    assert disc.discounted_energy(1000.0) < no_disc.discounted_energy(1000.0)
    assert disc.discounted_energy(1000.0) == pytest.approx(
        sum(1000.0 / 1.10**t for t in range(1, 11))
    )


def test_degradation_reduces_lifetime_energy():
    fresh = Economics(degradation=0.0, discount_rate=0.0, lifetime_years=5)
    aged = Economics(degradation=0.10, discount_rate=0.0, lifetime_years=5)
    assert aged.discounted_energy(1000.0) < fresh.discounted_energy(1000.0)
    assert aged.discounted_energy(1000.0) == pytest.approx(
        sum(1000.0 * 0.9 ** (t - 1) for t in range(1, 6))
    )


def test_payback_never_without_revenue():
    econ = Economics(electricity_price=0.0)
    assert payback_years(1000.0, 1.0, 1.0, econ) == math.inf


def test_payback_interpolation():
    # CAPEX 100, net cash 60/yr undiscounted -> payback 100/60 = 1.667 yr.
    econ = Economics(
        dc_cost_per_kw=100.0, ac_cost_per_kw=0.0, opex_per_kw_dc=0.0,
        lifetime_years=10, discount_rate=0.0, degradation=0.0,
        electricity_price=0.06,
    )
    assert payback_years(1000.0, 1.0, 1.0, econ) == pytest.approx(100.0 / 60.0)


def test_evaluate_keys_and_capex():
    econ = Economics(dc_cost_per_kw=200.0, ac_cost_per_kw=100.0, fixed_cost=50.0)
    out = evaluate(1000.0, pdc_kw=2.0, pac_kw=1.0, econ=econ)
    assert out["capex"] == pytest.approx(200 * 2 + 100 * 1 + 50)
    assert {"lcoe", "npv", "payback_years"} <= set(out)


def test_economics_validation():
    with pytest.raises(ValueError):
        Economics(dc_cost_per_kw=-1.0)
    with pytest.raises(ValueError):
        Economics(discount_rate=1.0)
    with pytest.raises(ValueError):
        Economics(lifetime_years=0)
    with pytest.raises(ValueError):
        Economics(degradation=-0.1)

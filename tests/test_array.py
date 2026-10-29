import numpy as np
import pytest

from pvclip import (
    cell_temperature,
    dc_power,
    dc_power_from_cell_temp,
    dc_power_mix,
)


def test_stc_identity_cell_temp():
    p = dc_power_from_cell_temp([1000.0], [25.0], pdc_kw=5.0, losses=0.0)
    assert p[0] == pytest.approx(5.0)


def test_temperature_derating_sign():
    cold = dc_power_from_cell_temp([1000.0], [0.0], pdc_kw=5.0, losses=0.0)
    hot = dc_power_from_cell_temp([1000.0], [65.0], pdc_kw=5.0, losses=0.0)
    assert cold[0] > 5.0 > hot[0]
    # gamma = -0.004: at 65 C, factor = 1 - 0.004*40 = 0.84
    assert hot[0] == pytest.approx(5.0 * 0.84)


def test_noct_cell_temperature():
    tcell = cell_temperature([800.0], [25.0], noct=45.0)
    assert tcell[0] == pytest.approx(25.0 + 25.0 / 800.0 * 800.0)


def test_dc_power_ambient_hot_cell_below_stc():
    # At full sun the NOCT model puts the cell well above 25 C.
    p = dc_power([1000.0], [25.0], pdc_kw=5.0, losses=0.0)
    assert p[0] < 5.0
    assert p[0] == pytest.approx(5.0 * (1 - 0.004 * (56.25 - 25.0)))


def test_dc_power_zero_at_night_and_nonnegative():
    p = dc_power([0.0, 0.0], [20.0, 20.0], pdc_kw=5.0)
    np.testing.assert_allclose(p, [0.0, 0.0])


def test_losses_scale_output():
    a = dc_power_from_cell_temp([1000.0], [25.0], pdc_kw=5.0, losses=0.0)
    b = dc_power_from_cell_temp([1000.0], [25.0], pdc_kw=5.0, losses=0.1)
    assert b[0] == pytest.approx(0.9 * a[0])


def test_invalid_inputs():
    with pytest.raises(ValueError):
        dc_power([-10.0], [20.0], pdc_kw=5.0)
    with pytest.raises(ValueError):
        dc_power([500.0], [20.0], pdc_kw=0.0)
    with pytest.raises(ValueError):
        dc_power_from_cell_temp([500.0], [25.0], pdc_kw=5.0, losses=1.0)


def test_mix_shares_must_sum_to_one():
    with pytest.raises(ValueError):
        dc_power_mix(
            {"east": [500.0], "west": [500.0]},
            {"east": 0.5, "west": 0.4},
            [20.0],
            pdc_kw=5.0,
        )
    with pytest.raises(ValueError):
        dc_power_mix(
            {"east": [500.0]},
            {"east": 0.5, "west": 0.5},
            [20.0],
            pdc_kw=5.0,
        )


def test_mix_splits_capacity():
    # Two identical sub-arrays at 50/50 must equal the single array.
    single = dc_power([800.0], [25.0], pdc_kw=4.0, losses=0.0)
    mixed = dc_power_mix(
        {"a": [800.0], "b": [800.0]},
        {"a": 0.5, "b": 0.5},
        [25.0],
        pdc_kw=4.0,
        losses=0.0,
    )
    np.testing.assert_allclose(mixed, single)

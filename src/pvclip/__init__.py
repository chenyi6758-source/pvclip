"""PV inverter clipping and DC/AC ratio optimisation."""

from .array import cell_temperature, dc_power, dc_power_from_cell_temp, dc_power_mix
from .capture import CaptureResult, battery_for_capture, capture_clipped, capture_sweep
from .clipping import ClipResult, clip_series, efficiency_constant, efficiency_quadratic
from .economics import Economics, evaluate, lcoe, npv, payback_years
from .sweep import optimum, resolution_bias, sweep_ratios
from .weather import load_series_csv, synthetic_year

__version__ = "0.1.0"

__all__ = [
    "CaptureResult",
    "ClipResult",
    "Economics",
    "battery_for_capture",
    "capture_clipped",
    "capture_sweep",
    "cell_temperature",
    "clip_series",
    "dc_power",
    "dc_power_from_cell_temp",
    "dc_power_mix",
    "efficiency_constant",
    "efficiency_quadratic",
    "evaluate",
    "lcoe",
    "load_series_csv",
    "npv",
    "optimum",
    "payback_years",
    "resolution_bias",
    "sweep_ratios",
    "synthetic_year",
]

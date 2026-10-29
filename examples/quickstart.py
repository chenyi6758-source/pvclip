"""Quickstart: DC/AC ratio sweep for a synthetic Shanghai year.

Runs offline (seeded synthetic weather, no pvlib needed), prints the
sweep table and the clipped-energy capture sizing, and saves two figures
plus the sweep table under ``examples/output/``.

Cost assumptions are illustrative USD values stated in the code, not
market data — replace them with your own quotes before deciding anything.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pvclip import (
    Economics,
    battery_for_capture,
    capture_sweep,
    clip_series,
    dc_power,
    optimum,
    resolution_bias,
    sweep_ratios,
    synthetic_year,
)

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

PAC_KW = 5.0
RATIOS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

# Illustrative USD assumptions (see module docstring).
ECON = Economics(
    dc_cost_per_kw=350.0,
    ac_cost_per_kw=120.0,
    opex_per_kw_dc=8.0,
    lifetime_years=25,
    discount_rate=0.05,
    electricity_price=0.12,
    degradation=0.005,
)


def main() -> None:
    weather = synthetic_year(days=365, dt_h=1.0, latitude=31.2, seed=42)
    dc_per_kwp = dc_power(weather["poa_wm2"], weather["temp_c"], pdc_kw=1.0)

    table = sweep_ratios(dc_per_kwp, RATIOS, pac_kw=PAC_KW, dt_h=1.0, econ=ECON)
    table.to_csv(OUT / "sweep.csv", index=False)
    print(table.round(3).to_string(index=False))
    best_lcoe = optimum(table, "lcoe")
    best_npv = optimum(table, "npv")
    print(
        f"\nLCOE optimum: ratio {best_lcoe['ratio']:.1f} "
        f"(LCOE ${best_lcoe['lcoe']:.4f}/kWh, NPV ${best_lcoe['npv']:.0f})"
    )
    print(
        f"NPV optimum:  ratio {best_npv['ratio']:.1f} "
        f"(NPV ${best_npv['npv']:.0f}, LCOE ${best_npv['lcoe']:.4f}/kWh)"
    )

    # Clipped-energy capture at the LCOE-optimal ratio.
    ratio = float(best_lcoe["ratio"])
    clipped = clip_series(dc_per_kwp * ratio * PAC_KW, PAC_KW, dt_h=1.0)
    print(
        f"\nAt ratio {ratio:.1f}: clipped "
        f"{clipped.metrics['clipped_kwh']:.0f} kWh/yr "
        f"({100 * clipped.metrics['clipped_fraction']:.1f}% of uncapped AC), "
        f"{clipped.metrics['hours_clipped']:.0f} clipping hours."
    )
    sizes = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
    cap_table = capture_sweep(
        clipped.dc_excess_kw, sizes, battery_kw=PAC_KW, rte=0.90, dt_h=1.0
    )
    print(cap_table.round(3).to_string(index=False))
    need = battery_for_capture(
        clipped.dc_excess_kw, sizes, battery_kw=PAC_KW,
        target_fraction=0.85, rte=0.90, dt_h=1.0,
    )
    print(f"Battery for 85% capture of refused energy: {need} kWh")
    print("(Capture fraction can never exceed the 0.90 round-trip efficiency.)")

    # Resolution bias on 15-minute summer data, at a ratio that clips.
    weather15 = synthetic_year(days=60, dt_h=0.25, latitude=31.2, seed=42,
                               start="2025-06-01")
    dc15 = dc_power(weather15["poa_wm2"], weather15["temp_c"], pdc_kw=1.8 * PAC_KW)
    bias = resolution_bias(dc15, PAC_KW, dt_h=0.25, factors=(1, 2, 4))
    print("\nResolution bias (15-min -> hourly):")
    print(bias.round(3).to_string(index=False))

    # Figure 1: energy and clipping vs ratio.
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(table["ratio"], table["ac_energy_kwh"], "o-", label="AC energy")
    ax1.set_xlabel("DC/AC ratio")
    ax1.set_ylabel("Annual AC energy (kWh)")
    ax2 = ax1.twinx()
    ax2.plot(
        table["ratio"], 100 * table["clipped_fraction"], "s--", color="tab:red",
        label="Clipped share",
    )
    ax2.set_ylabel("Clipped share of uncapped AC (%)", color="tab:red")
    ax1.set_title(f"5 kW inverter, synthetic Shanghai year (ratio sweep)")
    fig.tight_layout()
    fig.savefig(OUT / "sweep.png", dpi=150)
    plt.close(fig)

    # Figure 2: the worst clipping day at the optimal ratio.
    daily_clipped = (
        clipped.clipped_kw * 1.0
    )
    day_index = weather.index.date
    totals = {}
    for d in sorted(set(day_index)):
        mask = day_index == d
        totals[d] = float(np.sum(daily_clipped[mask]))
    worst_day = max(totals, key=totals.get)
    mask = day_index == worst_day
    hours = weather.index[mask].hour + weather.index[mask].minute / 60.0
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hours, dc_per_kwp[mask] * ratio * PAC_KW, label="DC power")
    ax.plot(hours, clipped.ac_kw[mask], label="AC output")
    ax.fill_between(hours, clipped.ac_kw[mask], clipped.ac_uncapped_kw[mask],
                    color="tab:red", alpha=0.3, label="Clipped")
    ax.axhline(PAC_KW, color="grey", linestyle=":", label="AC rating")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Power (kW)")
    ax.set_title(f"Worst clipping day ({worst_day}) at ratio {ratio:.1f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "worst_day.png", dpi=150)
    plt.close(fig)
    print(f"\nFigures saved to {OUT}")


if __name__ == "__main__":
    main()

"""Draw the figures in results/ from results/results.csv.

Run:  python experiments/make_figures.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Fixed colour per regime (colour follows the regime in every figure) plus a
# distinct marker, so the figures still read in greyscale or for colour-blind readers.
STYLE = {
    "competitive": dict(color="#2a78d6", marker="o", label="Competitive (bid = cost)"),
    "nash": dict(color="#eb6834", marker="s", label="One-shot Nash equilibrium"),
    "learned": dict(color="#1baf7a", marker="D", label="Q-learning algorithms"),
    "cartel": dict(color="#eda100", marker="^", label="Joint-profit maximum"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def load(path):
    data = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for r in csv.DictReader(f):
            data[(r["scenario"], r["regime"])]["carbon"].append(float(r["carbon_price"]))
            for k in ("price", "price_sd", "profit_A", "profit_B", "emissions", "consumer_surplus"):
                v = r.get(k, "")
                data[(r["scenario"], r["regime"])][k].append(float(v) if v not in ("", None) else 0.0)
    return data


def style_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def fig_prices(data):
    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=150)
    for regime, st in STYLE.items():
        d = data[("baseline", regime)]
        ax.plot(d["carbon"], d["price"], lw=2, ms=6, **st)
        if regime == "learned":
            lo = [p - s for p, s in zip(d["price"], d["price_sd"])]
            hi = [p + s for p, s in zip(d["price"], d["price_sd"])]
            ax.fill_between(d["carbon"], lo, hi, color=st["color"], alpha=0.15, lw=0)
    style_axes(ax, "Carbon price (EUR/tCO2)", "Electricity price (EUR/MWh)")
    ax.set_title("Electricity price by market conduct", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.text(0.01, 0.01, "Shaded band: +/- 1 s.d. across learning sessions.", color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(RESULTS / "fig1_prices.png")
    plt.close(fig)


def fig_emissions(data):
    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=150)
    for regime, st in STYLE.items():
        d = data[("baseline", regime)]
        ax.plot(d["carbon"], d["emissions"], lw=2, ms=6, **st)
    style_axes(ax, "Carbon price (EUR/tCO2)", "Emissions (tCO2 per period)")
    ax.set_title("Emissions by market conduct", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig2_emissions.png")
    plt.close(fig)


def fig_abatement(data):
    """Gain in firm A's operating profit from abating its coal unit."""
    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=150)
    for regime, st in STYLE.items():
        base, abated = data[("baseline", regime)], data[("abated", regime)]
        gain = [a - b for a, b in zip(abated["profit_A"], base["profit_A"])]
        ax.plot(base["carbon"], gain, lw=2, ms=6, **st)
    ax.axhline(0, color=MUTED, lw=1)
    style_axes(ax, "Carbon price (EUR/tCO2)", "Change in firm A profit (EUR per period)")
    ax.set_title("Private value of abatement for the coal-heavy firm", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig3_abatement_value.png")
    plt.close(fig)


if __name__ == "__main__":
    data = load(RESULTS / "results.csv")
    fig_prices(data)
    fig_emissions(data)
    fig_abatement(data)
    print("figures saved to", RESULTS)

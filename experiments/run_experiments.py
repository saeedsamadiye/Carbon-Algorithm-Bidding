"""Main experiment: carbon prices, market conduct and the value of abatement.

For each carbon price and each scenario (baseline / firm A abates its coal unit)
we compute four outcomes:

  competitive : both firms bid marginal cost (markup 0)
  nash        : pure-strategy Nash equilibrium of the one-shot game
                (if there are several, we report the one with the highest
                total profit, which is the conservative benchmark for collusion)
  cartel      : the profile that maximises joint profit
  learned     : average long-run play of independent Q-learning bidders

Run:  python experiments/run_experiments.py            (full run, ~12 minutes)
      python experiments/run_experiments.py --quick    (smoke test, ~20 seconds)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carbonbid.game import (build_tables, collusion_index,  # noqa: E402
                            joint_profit_maximum, pure_nash_equilibria)
from carbonbid.market import default_market  # noqa: E402
from carbonbid.qlearning import QLearningConfig, run_sessions  # noqa: E402

CARBON_PRICES = [0, 25, 50, 75, 100, 125]          # EUR/tCO2
print = partial(print, flush=True)  # show progress immediately in logs

MARKUPS = np.arange(0.0, 141.0, 10.0)              # 15 actions, EUR/MWh
SCENARIOS = {"baseline": 0.95, "abated": 0.70}     # firm A coal intensity, tCO2/MWh


def profile_row(tables, i, j):
    return {
        "price": tables.price[i, j],
        "profit_A": tables.profits[i, j, 0],
        "profit_B": tables.profits[i, j, 1],
        "emissions": tables.emissions[i, j],
        "consumer_surplus": tables.consumer_surplus[i, j],
    }


def learned_row(tables, results, tail: int = 20):
    """Average the last `tail` periods of greedy play, then average over sessions."""
    path = results.actions_path[:, -tail:, :]
    i, j = path[..., 0], path[..., 1]
    per_session = {
        "price": tables.price[i, j].mean(axis=1),
        "profit_A": tables.profits[i, j, 0].mean(axis=1),
        "profit_B": tables.profits[i, j, 1].mean(axis=1),
        "emissions": tables.emissions[i, j].mean(axis=1),
        "consumer_surplus": tables.consumer_surplus[i, j].mean(axis=1),
    }
    row = {k: float(v.mean()) for k, v in per_session.items()}
    row["price_sd"] = float(per_session["price"].std())
    row["converged_share"] = float(results.converged.mean())
    return row, per_session


def main(quick: bool) -> None:
    cfg = QLearningConfig(n_periods=40_000 if quick else 1_000_000,
                          n_sessions=5 if quick else 20,
                          check_every=5_000 if quick else 25_000,
                          beta=2e-4 if quick else 2e-5)
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    rows = []
    start = time.time()
    for scenario, intensity in SCENARIOS.items():
        market = default_market(coal_intensity_firm0=intensity)
        for cp in CARBON_PRICES:
            tables = build_tables(market, cp, MARKUPS)
            nash = pure_nash_equilibria(tables.profits)
            if not nash:
                raise RuntimeError(f"no pure Nash equilibrium at carbon price {cp}")
            total = tables.profits.sum(axis=2)
            ne = max(nash, key=lambda ij: total[ij])
            cartel = joint_profit_maximum(tables.profits)

            outcomes = {
                "competitive": profile_row(tables, 0, 0),
                "nash": profile_row(tables, *ne),
                "cartel": profile_row(tables, *cartel),
            }
            results = run_sessions(tables.profits, cfg)
            outcomes["learned"], _ = learned_row(tables, results)

            pi_n, pi_m = total[ne], total[cartel]
            for regime, o in outcomes.items():
                o.update(scenario=scenario, carbon_price=cp, regime=regime,
                         collusion_index=collusion_index(o["profit_A"] + o["profit_B"], pi_n, pi_m),
                         n_nash=len(nash))
                rows.append(o)
            print(f"[{time.time() - start:6.0f}s] {scenario:8s} carbon={cp:3d}  "
                  f"comp={outcomes['competitive']['price']:6.1f}  nash={outcomes['nash']['price']:6.1f}  "
                  f"learned={outcomes['learned']['price']:6.1f}  cartel={outcomes['cartel']['price']:6.1f}  "
                  f"converged={outcomes['learned']['converged_share']:.0%}")

    fields = ["scenario", "carbon_price", "regime", "price", "price_sd", "profit_A", "profit_B",
              "emissions", "consumer_surplus", "collusion_index", "converged_share", "n_nash"]
    name = "results_quick.csv" if quick else "results.csv"
    with open(out_dir / name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"saved {out_dir / name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="short smoke-test run")
    main(parser.parse_args().quick)

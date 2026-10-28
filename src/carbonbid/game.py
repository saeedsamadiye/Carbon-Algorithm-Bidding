
"""The one-shot bidding game and its theoretical benchmarks.

Each firm chooses a markup from a discrete grid and adds it to the marginal
cost of every unit it owns. Since the grid is relatively small, we can evaluate
all possible combinations of bids in advance and store the results in tables.

These tables serve two purposes:
  * They help us calculate the competitive, Nash, and joint-profit benchmarks.
  * They let the learning algorithms look up market outcomes quickly, without
    having to clear the market again in every period.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .market import Market, clear


@dataclass(frozen=True)
class GameTables:
    """Store market outcomes for every possible pair of firm markups."""

    markups: np.ndarray        # Available markups, in EUR/MWh above marginal cost
    profits: np.ndarray        # Profit of each firm for every markup pair
    price: np.ndarray          # Market-clearing price for each markup pair
    emissions: np.ndarray      # Total emissions for each markup pair
    consumer_surplus: np.ndarray  # Consumer surplus for each markup pair
    quantities: np.ndarray     # Output of each firm for every markup pair


def build_tables(
    market: Market,
    carbon_price: float,
    markups: np.ndarray,
) -> GameTables:
    """Calculate and store market outcomes for every pair of firm markups."""
    if market.n_firms != 2:
        raise ValueError("build_tables is written for two strategic firms")

    n = len(markups)

    profits = np.zeros((n, n, 2))
    price = np.zeros((n, n))
    emissions = np.zeros((n, n))
    cs = np.zeros((n, n))
    qty = np.zeros((n, n, 2))

    # Evaluate every possible combination of the two firms' markups.
    for i, j in product(range(n), range(n)):
        firm_markups = np.array([markups[i], markups[j]])
        bids = market.unit_bids(firm_markups, carbon_price)
        out = clear(market, bids, carbon_price)

        profits[i, j] = out.firm_profits
        price[i, j] = out.price
        emissions[i, j] = out.emissions
        cs[i, j] = out.consumer_surplus

        # Add up unit-level output to get the total output of each firm.
        qty[i, j] = np.bincount(
            market.owners,
            weights=out.quantities,
            minlength=2,
        )

    return GameTables(
        np.asarray(markups, float),
        profits,
        price,
        emissions,
        cs,
        qty,
    )


def pure_nash_equilibria(
    profits: np.ndarray,
    tol: float = 1e-9,
) -> list[tuple[int, int]]:
    """Find all pure-strategy Nash equilibria in the two-firm game.

    A markup pair (i, j) is a Nash equilibrium if neither firm can increase
    its profit by changing its own markup while the other firm's markup
    stays fixed.
    """
    p1, p2 = profits[..., 0], profits[..., 1]

    # Firm 1's best responses to each possible markup chosen by firm 2.
    best1 = p1 >= p1.max(axis=0, keepdims=True) - tol

    # Firm 2's best responses to each possible markup chosen by firm 1.
    best2 = p2 >= p2.max(axis=1, keepdims=True) - tol

    return [
        tuple(map(int, ij))
        for ij in np.argwhere(best1 & best2)
    ]


def joint_profit_maximum(profits: np.ndarray) -> tuple[int, int]:
    """Find the markup pair that maximises the firms' combined profit."""
    total = profits.sum(axis=2)
    return tuple(
        int(k)
        for k in np.unravel_index(np.argmax(total), total.shape)
    )


def collusion_index(
    profit: float,
    profit_nash: float,
    profit_cartel: float,
) -> float:
    """Measure profit relative to the Nash and joint-profit benchmarks.

    The index follows Calvano et al. (2020): 0 corresponds to Nash profit,
    while 1 corresponds to the joint-profit maximum.
    """
    if np.isclose(profit_cartel, profit_nash):
        return float("nan")

    return (profit - profit_nash) / (profit_cartel - profit_nash)

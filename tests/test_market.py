
"""Tests for the carbon bidding model.

Run the test suite from the repository root with:
    python -m pytest
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carbonbid.game import build_tables, joint_profit_maximum, pure_nash_equilibria  # noqa: E402
from carbonbid.lp_clearing import clear_lp  # noqa: E402
from carbonbid.market import clear, default_market  # noqa: E402
from carbonbid.qlearning import QLearningConfig, run_sessions  # noqa: E402


def test_competitive_price_without_carbon_is_gas_cost():
    # Without a carbon price, coal costs 30 EUR/MWh and gas costs 55.
    # Renewables and coal together provide 90 MW, which is not enough
    # to meet demand at 55 EUR/MWh, so gas sets the market price.
    market = default_market()
    outcome = clear(
        market,
        market.unit_bids([0, 0], 0.0),
        0.0,
    )

    assert outcome.price == pytest.approx(55.0)
    assert outcome.total_quantity == pytest.approx(market.demand(55.0))


def test_carbon_price_switches_merit_order():
    # At a carbon price of 100 EUR/tCO2, coal costs 30 + 95 = 125 EUR/MWh,
    # while gas costs 55 + 37 = 92 EUR/MWh. Gas becomes cheaper, so coal
    # units should not generate electricity.
    market = default_market()
    outcome = clear(
        market,
        market.unit_bids([0, 0], 100.0),
        100.0,
    )

    coal_units = [
        k for k, unit in enumerate(market.units)
        if unit.name.startswith("coal")
    ]

    assert np.allclose(outcome.quantities[coal_units], 0.0)
    assert outcome.price == pytest.approx(92.0)


def test_supply_equals_demand_and_price_on_demand_curve():
    market = default_market()
    rng = np.random.default_rng(1)

    # Check that market-clearing conditions hold for a range of random bids.
    for _ in range(200):
        bids = rng.uniform(0, 260, size=len(market.units))
        outcome = clear(market, bids, 50.0)

        # Total generation must match demand at the clearing price.
        assert outcome.total_quantity == pytest.approx(
            market.demand(outcome.price),
            abs=1e-9,
        )

        capacities = np.array([unit.capacity for unit in market.units])
        assert np.all(outcome.quantities <= capacities + 1e-9)

        # A unit cannot generate if its offer price is above the market price.
        assert np.all(
            outcome.quantities[bids > outcome.price + 1e-9] == 0
        )


def test_lp_agrees_with_direct_algorithm():
    market = default_market()
    rng = np.random.default_rng(2)

    # Compare the LP-based clearing method with the direct algorithm.
    # The LP method uses a finite grid, so small numerical differences are expected.
    for _ in range(25):
        bids = rng.uniform(0, 200, size=len(market.units))

        exact = clear(market, bids, 0.0)
        lp_price, lp_q = clear_lp(market, bids, n_steps=4000)

        step = market.demand_intercept / market.demand_slope / 4000

        assert lp_q == pytest.approx(
            exact.total_quantity,
            abs=2 * step,
        )
        assert lp_price == pytest.approx(
            exact.price,
            abs=2 * market.demand_slope * step + 1e-6,
        )


def test_nash_and_cartel_on_textbook_game():
    # In this prisoner's dilemma, defecting is the best response to either
    # action, so (defect, defect) is the unique Nash equilibrium.
    # Both firms earn more together by choosing (cooperate, cooperate).
    profits = np.zeros((2, 2, 2))
    profits[0, 0] = [3, 3]
    profits[0, 1] = [0, 5]
    profits[1, 0] = [5, 0]
    profits[1, 1] = [1, 1]

    assert pure_nash_equilibria(profits) == [(1, 1)]
    assert joint_profit_maximum(profits) == (0, 0)


def test_market_game_benchmarks_are_ordered():
    market = default_market()
    tables = build_tables(
        market,
        50.0,
        np.arange(0.0, 141.0, 10.0),
    )

    total_profit = tables.profits.sum(axis=2)
    nash_equilibria = pure_nash_equilibria(tables.profits)

    # The discretised game should have at least one pure-strategy Nash equilibrium.
    assert nash_equilibria, "the discretised game should have a pure Nash equilibrium"

    cartel = joint_profit_maximum(tables.profits)

    # Joint profit at the cartel outcome should be at least as high as at
    # any Nash equilibrium, which should in turn be no worse than zero markups.
    assert (
        total_profit[cartel]
        >= max(total_profit[ij] for ij in nash_equilibria)
        >= total_profit[0, 0] - 1e-9
    )


def test_qlearning_runs_and_returns_valid_actions():
    market = default_market()
    tables = build_tables(
        market,
        50.0,
        np.arange(0.0, 41.0, 10.0),
    )

    config = QLearningConfig(
        n_periods=2000,
        n_sessions=3,
        check_every=500,
        beta=1e-3,
    )

    results = run_sessions(
        tables.profits,
        config,
        eval_periods=10,
    )

    # Check the output dimensions and make sure all actions are in range.
    assert results.actions_path.shape == (3, 10, 2)
    assert results.actions_path.min() >= 0
    assert results.actions_path.max() < 5

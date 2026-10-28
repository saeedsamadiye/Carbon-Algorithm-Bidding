"""Market clearing as a linear programme (an independent cross-check).

The market operator maximises total welfare: the value of electricity to
consumers minus the offered cost of producing it. With a linear demand curve
the objective is quadratic, so we approximate demand by many small steps
("demand bids"). Each step k has width w and value v_k = a - b * (k - 0.5) * w.

    max   sum_k v_k * x_k  -  sum_i bid_i * q_i
    s.t.  sum_k x_k  =  sum_i q_i          (energy balance)
          0 <= x_k <= w,   0 <= q_i <= capacity_i

The market price is the shadow price (dual value) of the energy-balance
constraint. As the steps get smaller, the LP result converges to the exact
algorithm in market.clear(); tests/test_market.py checks this on random cases.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .market import Market


def clear_lp(market: Market, bids: np.ndarray, n_steps: int = 5000) -> tuple[float, float]:
    """Return (price, total quantity) from the welfare-maximising LP."""
    a, b = market.demand_intercept, market.demand_slope
    q_max = a / b                        # demand is zero above this quantity
    width = q_max / n_steps
    values = a - b * (np.arange(n_steps) + 0.5) * width

    offer_prices = np.concatenate(([0.0], np.asarray(bids, dtype=float)))
    offer_caps = np.concatenate(
        ([market.renewable_capacity], [u.capacity for u in market.units])
    )
    n_supply = len(offer_prices)

    # Decision vector: [q_0 .. q_n (supply), x_1 .. x_K (demand steps)]
    # linprog minimises, so we minimise (offer cost - consumer value).
    c = np.concatenate((offer_prices, -values))
    a_eq = np.concatenate((-np.ones(n_supply), np.ones(n_steps)))[None, :]
    bounds = [(0.0, cap) for cap in offer_caps] + [(0.0, width)] * n_steps

    res = linprog(c, A_eq=a_eq, b_eq=[0.0], bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    total_q = float(res.x[n_supply:].sum())
    # The dual of the balance constraint is the marginal value of one more MW
    # of demand being served, i.e. the market-clearing price (up to sign).
    price = float(abs(res.eqlin.marginals[0]))
    return price, total_q

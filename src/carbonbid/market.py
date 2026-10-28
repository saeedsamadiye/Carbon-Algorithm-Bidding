"""A small electricity spot market with a carbon price.

The market has
  * a competitive fringe of renewable capacity that always offers at zero cost,
  * strategic firms, each owning several thermal units (here: coal and gas),
  * price-responsive demand with linear inverse demand  P(Q) = a - b * Q.

Each thermal unit's marginal cost is  fuel_cost + emission_intensity * carbon_price,
so a carbon price changes both the level of costs and, possibly, the merit order.

The market clears at a single price where the offer curve meets the demand curve
(a standard pay-as-clear spot market). Clearing is NOT the research question here;
it is the environment in which the bidding behaviour is studied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Unit:
    """A thermal generating unit owned by a strategic firm."""

    name: str
    owner: int                 # Index of the firm that owns the unit
    capacity: float            # Available capacity in MW
    fuel_cost: float           # Fuel cost in EUR/MWh, excluding carbon
    emission_intensity: float  # Emissions in tCO2 per MWh

    def marginal_cost(self, carbon_price: float) -> float:
        """Calculate marginal generation cost, including carbon costs."""
        return self.fuel_cost + self.emission_intensity * carbon_price

@dataclass(frozen=True)
class Market:
    """Market primitives: demand curve, renewable fringe and strategic units."""

    demand_intercept: float   # a in P = a - bQ (EUR/MWh)
    demand_slope: float       # b in P = a - bQ (EUR/MWh per MW)
    renewable_capacity: float  # MW, offered at 0 EUR/MWh
    units: tuple[Unit, ...] = field(default_factory=tuple)

    def demand(self, price: float) -> float:
        """Quantity demanded at a given price."""
        return max(0.0, (self.demand_intercept - price) / self.demand_slope)

    def marginal_costs(self, carbon_price: float) -> np.ndarray:
        return np.array([u.marginal_cost(carbon_price) for u in self.units])

    @property
    def n_firms(self) -> int:
        return max(u.owner for u in self.units) + 1

    @property
    def owners(self) -> np.ndarray:
        return np.array([u.owner for u in self.units])

    def unit_bids(self, markups: np.ndarray, carbon_price: float) -> np.ndarray:
        """Each firm adds its markup to the marginal cost of every unit it owns."""
        return self.marginal_costs(carbon_price) + np.asarray(markups, float)[self.owners]


@dataclass(frozen=True)
class Outcome:
    """Everything we want to know about one market clearing."""

    price: float
    quantities: np.ndarray      # dispatch of each strategic unit (MW)
    renewable_quantity: float   # dispatch of the renewable fringe (MW)
    total_quantity: float
    profits: np.ndarray         # (price - marginal cost) * quantity, per unit
    firm_profits: np.ndarray    # unit profits summed by owner
    emissions: float            # tCO2
    consumer_surplus: float


def clear(market: Market, bids: np.ndarray, carbon_price: float) -> Outcome:
    """Clear the market for given offer prices of the strategic units.

    Algorithm: walk up the offer curve from the cheapest offer. At each offer
    price level, compare demand with the capacity offered so far.

      * If demand at this level is below what is already dispatched, the demand
        curve crosses the vertical part of the offer curve: the price is
        a - b * (capacity dispatched so far).
      * If demand falls inside this level's capacity, the price equals the
        offer price and units at this level share the residual demand pro rata.
      * Otherwise dispatch the whole level and continue.
    """
    bids = np.asarray(bids, dtype=float)
    caps = np.array([u.capacity for u in market.units], dtype=float)
    a, b = market.demand_intercept, market.demand_slope

    # Offer levels: renewable fringe at 0, then strategic units at their bids.
    offer_prices = np.concatenate(([0.0], bids))
    offer_caps = np.concatenate(([market.renewable_capacity], caps))
    dispatch = np.zeros_like(offer_caps)

    levels = np.unique(offer_prices)  # sorted ascending
    dispatched = 0.0
    price = None
    for level in levels:
        at_level = offer_prices == level
        cap_level = offer_caps[at_level].sum()
        demand_here = market.demand(level)
        if demand_here <= dispatched:
            price = a - b * dispatched  # crossing on a vertical segment
            break
        if demand_here <= dispatched + cap_level:
            price = level
            share = (demand_here - dispatched) / cap_level
            dispatch[at_level] = offer_caps[at_level] * share
            dispatched = demand_here
            break
        dispatch[at_level] = offer_caps[at_level]
        dispatched += cap_level
    if price is None:  # demand exceeds all offered capacity at the top offer
        price = a - b * dispatched

    q_units = dispatch[1:]
    mc = market.marginal_costs(carbon_price)
    profits = (price - mc) * q_units
    emissions = float(sum(u.emission_intensity * q for u, q in zip(market.units, q_units)))
    total_q = float(dispatch.sum())
    consumer_surplus = 0.5 * b * total_q**2  # triangle under a linear demand curve

    firm_profits = np.bincount(market.owners, weights=profits, minlength=market.n_firms)

    return Outcome(
        price=float(price),
        quantities=q_units,
        renewable_quantity=float(dispatch[0]),
        total_quantity=total_q,
        profits=profits,
        firm_profits=firm_profits,
        emissions=emissions,
        consumer_surplus=float(consumer_surplus),
    )


def default_market(coal_intensity_firm0: float = 0.95) -> Market:
    """Stylised parameters: two firms, each owning a coal and a gas unit.

    Firm 0 is coal-heavy and firm 1 is gas-heavy, so a carbon price shifts
    cost advantage between them and, above roughly 43 EUR/t, switches the merit
    order from coal-before-gas to gas-before-coal. The numbers are illustrative,
    not calibrated to a real market.

    coal_intensity_firm0 lets experiments give firm 0 an abated (cleaner) coal unit.
    """
    return Market(
        demand_intercept=250.0,
        demand_slope=2.0,
        renewable_capacity=40.0,
        units=(
            Unit("coal_A", owner=0, capacity=35.0, fuel_cost=30.0,
                 emission_intensity=coal_intensity_firm0),
            Unit("gas_A", owner=0, capacity=15.0, fuel_cost=55.0, emission_intensity=0.37),
            Unit("coal_B", owner=1, capacity=15.0, fuel_cost=30.0, emission_intensity=0.95),
            Unit("gas_B", owner=1, capacity=35.0, fuel_cost=55.0, emission_intensity=0.37),
        ),
    )

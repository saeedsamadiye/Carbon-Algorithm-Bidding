# Walkthrough: how the model works, file by file

This document explains every part of the repository in the order the code runs.
It is written so that someone with an economics background (but not a computer
science background) can follow, re-run and extend the model.

---

## 1. The economic question

When a carbon price is introduced, electricity producers' marginal costs rise.
How much of that cost increase reaches the electricity price (the *pass-through*)
depends on how firms bid:

* under perfect competition, firms bid their marginal cost;
* in a one-shot strategic game (Nash equilibrium), firms add a markup;
* firms that bid with **learning algorithms** may learn, without communicating,
  to keep markups above the Nash level (tacit "algorithmic collusion",
  Calvano et al., 2020).

The second question links to empirical work on how markets value firms'
carbon-abatement investment: **is abatement worth more or less to a firm when
prices are set by algorithms rather than by competition?**

---

## 2. `src/carbonbid/market.py`: the market

**Primitives.**

* Demand is linear: `P = a - b*Q` with `a = 250`, `b = 2`.
* 40 MW of renewables are always offered at 0 EUR/MWh.
* Two firms each own one coal unit and one gas unit:

| Unit   | Owner           | Capacity | Fuel cost | tCO2/MWh |
|--------|-----------------|---------:|----------:|---------:|
| coal_A | Firm A (coal-heavy) | 35 MW | 30 | 0.95 |
| gas_A  | Firm A          | 15 MW | 55 | 0.37 |
| coal_B | Firm B (gas-heavy)  | 15 MW | 30 | 0.95 |
| gas_B  | Firm B          | 35 MW | 55 | 0.37 |

**Marginal cost** = fuel cost + emission intensity × carbon price. Coal is
cheaper without a carbon price. Above about 43 EUR/t, gas becomes cheaper
(`30 + 0.95c = 55 + 0.37c` gives `c ≈ 43`). This is the merit-order switch.

**Clearing (`clear`).** Offers are sorted from cheapest to most expensive and
walked up one by one:

1. If demand at the current offer price is already covered by what has been
   dispatched, demand crosses the offer curve on a *vertical* segment. The price
   is then read off the demand curve: `P = a - b * dispatched`.
2. If demand falls inside the current offer's capacity, that offer is *marginal*
   and sets the price. Units offering the same price share the residual demand
   in proportion to their capacity.
3. Otherwise, the whole offer is dispatched and the walk continues.

The function returns an `Outcome`: price, dispatch, profits per unit and per
firm, emissions and consumer surplus (`b*Q²/2`, the triangle under linear
demand).

## 3. `src/carbonbid/lp_clearing.py`: the same market as a linear programme

This is an independent check on step 2, written as the optimisation problem a
market operator solves: **maximise welfare** (consumer value minus offered
cost) subject to the energy balance and capacity limits.

* Linear demand makes the objective quadratic. To keep it a *linear*
  programme, demand is split into 5,000 small steps, each with its own value.
* SciPy's `linprog` with the HiGHS solver finds the optimal dispatch.
* The **price is the dual value (shadow price) of the energy-balance
  constraint**: how much welfare would change if one more MW had to be
  balanced. This is the textbook link between optimisation and market prices.

`tests/test_market.py::test_lp_agrees_with_direct_algorithm` checks on random
bids that the LP and the direct algorithm give the same price and quantity, up
to the step size.

## 4. `src/carbonbid/game.py`: the bidding game and its benchmarks

**Actions.** Each firm chooses one markup from {0, 10, …, 140} EUR/MWh and adds
it to the marginal cost of every unit it owns. That gives 15 × 15 = 225
action profiles.

**Payoff tables (`build_tables`).** The market is cleared once for each of
the 225 profiles, and profits, price, emissions and consumer surplus are stored
in arrays. Later steps look results up instead of re-clearing the market, which
makes learning fast.

**Benchmarks.**

* *Competitive*: both markups are 0.
* *Nash equilibrium (`pure_nash_equilibria`)*: a profile where neither firm can
  gain by changing its own markup. The code marks each firm's best replies and
  keeps the cells where both are best replies. With a discrete grid there can
  be several. The experiments use the one with the highest total profit, which
  makes it harder, not easier, to call an outcome "collusive".
* *Joint-profit maximum (`joint_profit_maximum`)*: what a perfect cartel would
  choose.
* *Collusion index*: `(π − π_Nash) / (π_cartel − π_Nash)`. It is 0 at the Nash
  profit and 1 at the cartel profit.

## 5. `src/carbonbid/qlearning.py`: learning algorithms

Each firm runs **Q-learning**, a basic reinforcement-learning algorithm. It
keeps a table `Q[state, action]` that estimates the discounted future profit of
choosing `action` in `state`.

* **State**: last period's pair of markups. Firms can therefore react to what
  happened yesterday, which is what allows "punish a deviation" behaviour to be
  learned.
* **Choice**: with probability `ε_t = exp(−β t)` the firm picks a random markup
  (exploration). Otherwise it picks the markup with the highest Q-value
  (exploitation). Exploration fades over time.
* **Update** after observing profit `π`:
  `Q[s,a] ← Q[s,a] + α (π + δ max Q[s',·] − Q[s,a])`,
  with `α = 0.15` and `δ = 0.95`.
* **Convergence**: a session counts as converged when every firm's greedy
  strategy has not changed over the last 100,000 periods.
* **Speed trick**: 20 independent sessions run at the same time by adding a
  leading "session" dimension to all arrays (NumPy vectorisation). One pass of
  the Python loop updates all 20 sessions.

After one million periods, firms play greedily for 100 periods. The outcome is
the average of the last 20 periods, which also covers cases where play cycles.

## 6. `experiments/run_experiments.py`: the experiment

For carbon prices 0, 25, 50, 75, 100 and 125 EUR/t, and for two scenarios, it
computes the four regimes (competitive, Nash, cartel, learned) and saves
`results/results.csv`. The two scenarios are:

* *baseline*;
* *abated*: Firm A's coal unit emits 0.70 instead of 0.95 tCO2/MWh.

**The value of abatement** for Firm A is its operating profit in the abated
scenario minus its profit in the baseline. It is computed separately for each
regime.

## 7. `experiments/make_figures.py`: figures

This script reads `results.csv` and draws three figures: price, emissions and
the value of abatement, each by regime. Every regime keeps the same colour and
marker in all figures.

## 8. `tests/`: automated checks

Run them with `python -m pytest`. They check:

* known prices in simple cases, for example 55 EUR/MWh when gas is marginal at
  zero carbon price;
* that supply always equals demand and no unit bidding above the price is
  dispatched;
* that the LP and the direct algorithm agree;
* that the Nash and cartel finders get the Prisoner's Dilemma right;
* that benchmarks are ordered (cartel ≥ Nash ≥ competitive profit) and that
  learning returns valid actions.

---

## Limitations, and how the project could be extended

* There is one market per period, with no hours, block bids or batteries. The
  natural next step is a day of 24 hourly markets with a storage unit that buys
  low and sells high. Its schedule is an LP when it takes prices as given, and
  a game when it is large enough to move prices.
* Parameters are illustrative, not calibrated. Calibration would use data such
  as ENTSO-E prices and generation by fuel type.
* Tabular Q-learning only handles small, discrete action spaces. Larger
  problems would need deep reinforcement learning.
* Results depend on learning parameters (`α`, `β`, `δ`). A full study would
  vary them.

## References

* Calvano, E., Calzolari, G., Denicolò, V., and Pastorello, S. (2020).
  Artificial Intelligence, Algorithmic Pricing, and Collusion.
  *American Economic Review*, 110(10), 3267–3297.
* Sutton, R. S., and Barto, A. G. (2018). *Reinforcement Learning: An
  Introduction* (2nd ed.). MIT Press.

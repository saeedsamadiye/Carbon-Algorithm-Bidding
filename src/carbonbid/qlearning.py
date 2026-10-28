"""Independent Q-learning bidders in the repeated market (vectorised).

Set-up follows the design of Calvano, Calzolari, Denicolo and Pastorello (2020,
American Economic Review), adapted to the electricity market in market.py:

  * Two firms play the bidding game every period, forever.
  * The state is last period's pair of actions, so a firm can condition its
    bid on what happened one period ago (this is what makes reward-punishment
    behaviour, and therefore tacit collusion, possible).
  * Each firm keeps its own Q-table Q[state, action] and updates it with
        Q <- (1 - alpha) * Q + alpha * (profit + delta * max_a' Q[next_state, a'])
  * Exploration is epsilon-greedy with epsilon_t = exp(-beta * t).

Many independent sessions (different random seeds) are simulated at once by
adding a leading "session" dimension to every array. This keeps the Python
loop short: one iteration of the loop updates all sessions together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QLearningConfig:
    alpha: float = 0.15          # Learning rate
    delta: float = 0.95          # Discount factor
    beta: float = 2e-5           # Rate at which exploration declines
    n_periods: int = 1_000_000   # Number of learning periods per session
    n_sessions: int = 20         # Number of independent sessions
    check_every: int = 25_000    # Frequency of policy stability checks
    seed: int = 0                # Random seed for reproducibility



@dataclass(frozen=True)
class SessionResults:
    actions_path: np.ndarray     # Greedy actions after learning, by session and period
    converged: np.ndarray        # Whether the policy remained stable across checks
    final_q: np.ndarray          # Final Q-tables for both firms in every session

def _initial_q(profits: np.ndarray, delta: float) -> np.ndarray:
    """Initialise Q-values using expected payoffs against a random rival."""
    n = profits.shape[0]
    q0 = np.empty((2, n * n, n))
    q0[0] = profits[:, :, 0].mean(axis=1) / (1 - delta)   # firm 1 averages over j
    q0[1] = profits[:, :, 1].mean(axis=0) / (1 - delta)   # firm 2 averages over i
    return q0


def run_sessions(profits: np.ndarray, cfg: QLearningConfig, eval_periods: int = 100) -> SessionResults:
    """Simulate cfg.n_sessions independent learning sessions."""
    rng = np.random.default_rng(cfg.seed)
    n = profits.shape[0]
    S = cfg.n_sessions
    sess = np.arange(S)

    # Scale profits so Q-values are of order 1-100 (numerically comfortable).
    scale = max(np.abs(profits).max(), 1.0)
    reward = profits / scale

    Q = np.broadcast_to(_initial_q(reward, cfg.delta), (S, 2, n * n, n)).copy()
    state = rng.integers(0, n * n, size=S)

    last_policy = None
    stable_checks = np.zeros(S, dtype=int)

    for t in range(cfg.n_periods):
        eps = np.exp(-cfg.beta * t)
        greedy = Q[sess, :, state, :].argmax(axis=-1)            # (S, 2)
        explore = rng.random((S, 2)) < eps
        random_a = rng.integers(0, n, size=(S, 2))
        a = np.where(explore, random_a, greedy)                  # (S, 2)

        r = reward[a[:, 0], a[:, 1]]                              # (S, 2)
        next_state = a[:, 0] * n + a[:, 1]

        for i in (0, 1):
            future = Q[sess, i, next_state, :].max(axis=-1)
            old = Q[sess, i, state, a[:, i]]
            Q[sess, i, state, a[:, i]] = old + cfg.alpha * (r[:, i] + cfg.delta * future - old)
        state = next_state

        if (t + 1) % cfg.check_every == 0:
            policy = Q.argmax(axis=-1)                            # (S, 2, n_states)
            if last_policy is not None:
                same = (policy == last_policy).all(axis=(1, 2))
                stable_checks = np.where(same, stable_checks + 1, 0)
            last_policy = policy

    # After learning: play greedily (no exploration) and record the path.
    path = np.empty((S, eval_periods, 2), dtype=int)
    for t in range(eval_periods):
        a = Q[sess, :, state, :].argmax(axis=-1)
        path[:, t] = a
        state = a[:, 0] * n + a[:, 1]

    converged = stable_checks >= 4  # unchanged over the last 4 checks
    return SessionResults(actions_path=path, converged=converged, final_q=Q)

"""Value iteration, used only as an optimal baseline for comparison."""

from __future__ import annotations

import numpy as np

from .envs import make_env


def transition_model(noise_p: float = 0.0):
    """Next state, reward and done for every (s, a), plus the noise matrix."""
    base = make_env().unwrapped
    S, A = base.observation_space.n, base.action_space.n
    nxt = np.zeros((S, A), dtype=int)
    rew = np.zeros((S, A))
    term = np.zeros((S, A), dtype=bool)
    # Taxi stores its rules in env.P[state][action] it's deterministic so one outcome each
    for s in range(S):
        for a in range(A):
            (_, s2, r, done), = base.P[s][a]
            nxt[s, a], rew[s, a], term[s, a] = s2, r, done
    # a, b = chance of actually doing b when the agent picks a for the noise wrapper
    M = np.full((A, A), noise_p / (A - 1))
    np.fill_diagonal(M, 1.0 - noise_p)
    return nxt, rew, term, M


def value_iteration(noise_p: float = 0.0, gamma: float = 1.0, tol: float = 1e-9, max_iter: int = 10_000):
    """Optimal Q-values (undiscounted)."""
    nxt, rew, term, M = transition_model(noise_p)
    S, A = rew.shape
    Q = np.zeros((S, A))
    for _ in range(max_iter):
        V = Q.max(axis=1)                                     # best value of each state
        exec_val = rew + gamma * np.where(term, 0.0, V[nxt])  # value of each executed action
        Q_new = exec_val @ M.T                                # average over noise
        if np.abs(Q_new - Q).max() < tol:                     # stops once nothing changes
            return Q_new
        Q = Q_new
    return Q


def reachable_states() -> np.ndarray:
    """The 400 states that can occur mid-episode."""
    out = []
    for row in range(5):
        for col in range(5):
            for pas in range(5):
                for dest in range(4):
                    if pas == 4 or pas != dest:  # skip passenger already at destination
                        out.append(((row * 5 + col) * 5 + pas) * 4 + dest)
    return np.array(out)


def policy_agreement(Q: np.ndarray, Q_star: np.ndarray, tol: float = 1e-6) -> float:
    """Share of states where the greedy action is optimal."""
    states = reachable_states()
    greedy = Q[states].argmax(axis=1)
    return float(np.mean(Q_star[states, greedy] >= Q_star[states].max(axis=1) - tol))

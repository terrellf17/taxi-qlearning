"""Tabular Q-learning agent."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class QConfig:
    """Hyperparameters for one run."""

    alpha: float = 0.1  # learning rate
    gamma: float = 0.99  # discount factor
    epsilon_start: float = 1.0  # random
    epsilon_min: float = 0.05 # dont explore less than that
    epsilon_decay: float = 0.999
    schedule: str = "exp"
    episodes: int = 5000
    q_init: float = 0.0 #starting value

    def to_dict(self) -> dict:
        return asdict(self)


class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, cfg: QConfig, seed: int | None = None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        # one row per state, one column per action
        self.Q = np.full((n_states, n_actions), cfg.q_init, dtype=np.float64)
        self.epsilon = cfg.epsilon_start

    def greedy_action(self, state: int) -> int:
        """Best action; ties broken randomly."""
        q = self.Q[state]
        best = np.flatnonzero(q == q.max()) # all actions tied for the top value
        return int(best[0]) if len(best) == 1 else int(self.rng.choice(best))

    def select_action(self, state: int) -> int:
        """Epsilon-greedy action."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))   # explore
        return self.greedy_action(state)                     # exploit

    def update(self, s: int, a: int, r: float, s_next: int, terminated: bool) -> float:
        """Q-learning update. No future value if the episode terminated."""
        target = r if terminated else r + self.cfg.gamma * self.Q[s_next].max()
        td_error = target - self.Q[s, a] # how wrong the estimate is
        self.Q[s, a] += self.cfg.alpha * td_error
        return td_error

    def decay_epsilon(self, episode: int) -> None:
        """Update epsilon after an episode."""
        c = self.cfg
        if c.schedule == "constant":
            self.epsilon = c.epsilon_start
        elif c.schedule == "exp":
            self.epsilon = max(c.epsilon_min, c.epsilon_start * c.epsilon_decay ** (episode + 1))
        elif c.schedule == "linear":
            frac = min(1.0, (episode + 1) / c.epsilon_decay)
            self.epsilon = c.epsilon_start + frac * (c.epsilon_min - c.epsilon_start)
        else:
            raise ValueError(f"unknown schedule {c.schedule!r}")

    def save(self, path: str) -> None:
        np.save(path, self.Q)

    @classmethod
    def from_q(cls, Q: np.ndarray, cfg: QConfig | None = None, seed: int | None = 0) -> "QLearningAgent":
        agent = cls(Q.shape[0], Q.shape[1], cfg or QConfig(), seed)
        agent.Q = Q.copy()
        agent.epsilon = 0.0 # loaded agents will act greedily
        return agent

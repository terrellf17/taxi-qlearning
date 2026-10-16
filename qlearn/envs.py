"""Taxi environment and action noise wrapper"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


def taxi_id() -> str:
    """Taxi-v4 if available, otherwise Taxi-v3."""
    # newer gymnasium versions
    return "Taxi-v4" if "Taxi-v4" in gym.envs.registry else "Taxi-v3"


class ActionNoiseWrapper(gym.ActionWrapper):
    """With probability p, swap the chosen action for a random one."""

    def __init__(self, env: gym.Env, p: float = 0.0, seed: int | None = None):
        super().__init__(env)
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        self.p = p  # chance an action will get swapped
        self.n = env.action_space.n # number of actioms
        self.noise_rng = np.random.default_rng(seed) # seperate rng
        self.replaced = 0  # the number of swapped actions

    def action(self, action):
        # called automatically 
        if self.p > 0 and self.noise_rng.random() < self.p:
            # pick from the other 5 actions, skips what the agent chose
            other = int(self.noise_rng.integers(self.n - 1))
            self.replaced += 1
            return other if other < action else other + 1
        return action


def make_env(noise_p: float = 0.0, seed: int | None = None,
             render_mode: str | None = None) -> gym.Env:
    # taxt ends an episode after a certain amount of steps
    env = gym.make(taxi_id(), render_mode=render_mode)
    if noise_p > 0:
        env = ActionNoiseWrapper(env, p=noise_p, seed=seed)
        # only add wrapper if noise is turned on
    return env
"""Training and evaluation. Example: python -m qlearn.train --alpha 0.5"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .agent import QConfig, QLearningAgent
from .envs import make_env

CHECKPOINT_EVERY = 50         # test the greedy policy 
CHECKPOINT_EPISODES = 20      # episodes per test
CHECKPOINT_SEED = 50_000      # fixed start states for those tests


def run_greedy(Q: np.ndarray, n_episodes: int, seed0: int, noise_p: float = 0.0) -> dict:
    """Run the greedy policy (no exploration) and return stats."""
    env = make_env(noise_p=noise_p, seed=seed0 + 7)
    returns, steps, success, illegal = [], [], [], []
    for i in range(n_episodes):
        s, _ = env.reset(seed=seed0 + i)  # same seeds and same start states for every policy
        total, n, n_illegal, done, delivered = 0.0, 0, 0, False, False
        while not done:
            a = int(np.argmax(Q[s]))  # always the best action
            s, r, terminated, truncated, _ = env.step(a)
            total += r
            n += 1
            n_illegal += r == -10  # bad pickup or dropoff
            delivered = delivered or terminated
            done = terminated or truncated
        returns.append(total)
        steps.append(n)
        success.append(delivered)
        illegal.append(n_illegal)
    env.close()
    returns, steps, success = np.array(returns), np.array(steps), np.array(success)
    return {
        "mean_reward": float(returns.mean()),
        "std_reward": float(returns.std()),
        "success_rate": float(success.mean()),
        "mean_steps": float(steps.mean()),
        "mean_steps_success": float(steps[success].mean()) if success.any() else float("nan"),
        "illegal_actions_per_ep": float(np.mean(illegal)),
        "returns": returns.tolist(),
        "steps": steps.tolist(),
    }


def train(cfg: QConfig, seed: int = 0, noise_p: float = 0.0, checkpoints: bool = True):
    """Train one agent. Returns (agent, history)."""
    env = make_env(noise_p=noise_p, seed=seed + 1000)
    agent = QLearningAgent(env.observation_space.n, env.action_space.n, cfg, seed=seed)

    # per episode logs for the plots
    ep_return = np.zeros(cfg.episodes)
    ep_length = np.zeros(cfg.episodes, dtype=int)
    ep_success = np.zeros(cfg.episodes, dtype=bool)
    ep_epsilon = np.zeros(cfg.episodes)
    ckpt_ep, ckpt_reward, ckpt_success = [], [], []

    s, _ = env.reset(seed=seed)
    for ep in range(cfg.episodes):
        if ep > 0:
            s, _ = env.reset()
        ep_epsilon[ep] = agent.epsilon
        total, n, done = 0.0, 0, False
        while not done:
            a = agent.select_action(s)
            s_next, r, terminated, truncated, _ = env.step(a)
            # a timeout is not a real terminal state
            agent.update(s, a, r, s_next, terminated)
            s = s_next
            total += r
            n += 1
            done = terminated or truncated
        ep_return[ep], ep_length[ep], ep_success[ep] = total, n, terminated
        agent.decay_epsilon(ep)

        # check how good the greedy policy is 
        if checkpoints and (ep + 1) % CHECKPOINT_EVERY == 0:
            m = run_greedy(agent.Q, CHECKPOINT_EPISODES, CHECKPOINT_SEED, noise_p)
            ckpt_ep.append(ep + 1)
            ckpt_reward.append(m["mean_reward"])
            ckpt_success.append(m["success_rate"])
    env.close()

    history = {
        "returns": ep_return,
        "lengths": ep_length,
        "success": ep_success,
        "epsilon": ep_epsilon,
        "ckpt_episode": np.array(ckpt_ep),
        "ckpt_reward": np.array(ckpt_reward),
        "ckpt_success": np.array(ckpt_success),
    }
    return agent, history


def main():
    # every hyperparameter can be changed from the command line
    p = argparse.ArgumentParser(description="Train a tabular Q-learning agent on Taxi.")
    d = QConfig()
    p.add_argument("--alpha", type=float, default=d.alpha)
    p.add_argument("--gamma", type=float, default=d.gamma)
    p.add_argument("--epsilon-start", type=float, default=d.epsilon_start)
    p.add_argument("--epsilon-min", type=float, default=d.epsilon_min)
    p.add_argument("--epsilon-decay", type=float, default=d.epsilon_decay)
    p.add_argument("--schedule", choices=["exp", "linear", "constant"], default=d.schedule)
    p.add_argument("--episodes", type=int, default=d.episodes)
    p.add_argument("--q-init", type=float, default=d.q_init)
    p.add_argument("--noise", type=float, default=0.0, help="action-noise probability p (part H)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--out", default="results/single_run")
    a = p.parse_args()

    cfg = QConfig(a.alpha, a.gamma, a.epsilon_start, a.epsilon_min, a.epsilon_decay,
                  a.schedule, a.episodes, a.q_init)
    agent, hist = train(cfg, seed=a.seed, noise_p=a.noise)
    m = run_greedy(agent.Q, a.eval_episodes, seed0=20_000, noise_p=a.noise)

    # save the Q-table training history and final scores
    os.makedirs(a.out, exist_ok=True)
    agent.save(os.path.join(a.out, "q_table.npy"))
    np.savez(os.path.join(a.out, "history.npz"), **hist)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump({"config": cfg.to_dict(), "seed": a.seed, "noise_p": a.noise,
                   "eval": {k: v for k, v in m.items() if k not in ("returns", "steps")}}, f, indent=2)

    last = hist["returns"][-100:].mean()
    print(f"config: {cfg.to_dict()}  noise p={a.noise}")
    print(f"mean training return, last 100 episodes: {last:.2f}")
    print(f"greedy eval over {a.eval_episodes} episodes: mean reward {m['mean_reward']:.2f} "
          f"+/- {m['std_reward']:.2f}, success {m['success_rate']:.0%}, steps {m['mean_steps']:.1f}")
    print(f"saved Q-table, history and summary to {a.out}/")


if __name__ == "__main__":
    main()

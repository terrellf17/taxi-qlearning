"""Runs all experiments (parts C-H). Usage: python experiments.py [--seeds 3]"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

from qlearn.agent import QConfig
from qlearn.optimal import policy_agreement, value_iteration
from qlearn.plotting import INK2, MUTED, SERIES, band, moving_average, plt, save
from qlearn.train import CHECKPOINT_EPISODES, CHECKPOINT_SEED, run_greedy, train

VAL_SEED0, TEST_SEED0 = 10_000, 20_000  # validation / test episodes
VAL_EPISODES, TEST_EPISODES = 100, 100
NOISE_EVAL_EPISODES = 200
NOISE_LEVELS = [0.0, 0.10, 0.25, 0.40]
OUT = "results"
PLOTS = os.path.join(OUT, "plots")

BASE = QConfig(alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_min=0.05,
               epsilon_decay=0.999, schedule="exp", episodes=5000, q_init=0.0)

SWEEPS = {
    "alpha": ("alpha", [0.01, 0.1, 0.5, 0.9]),
    "gamma": ("gamma", [0.5, 0.7, 0.9, 0.99]),
    "epsilon_decay": ("epsilon_decay", [0.99, 0.995, 0.999, 0.9995]),
}
SYMBOL = {"alpha": "α", "gamma": "γ", "epsilon_decay": "ε-decay"}
GRID_ALPHA = [0.1, 0.5, 0.9]
GRID_DECAY = [0.99, 0.995, 0.999, 0.9995]
H_ALPHAS = [0.1, 0.5, 0.9]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# running
def _job(args):
    cfg_dict, seed, noise_p = args
    cfg = QConfig(**cfg_dict)
    agent, hist = train(cfg, seed=seed, noise_p=noise_p)
    val = run_greedy(agent.Q, VAL_EPISODES, VAL_SEED0, noise_p)
    return {"Q": agent.Q, "hist": hist, "val": val}


class Runner:
    """Trains agents in parallel and caches results."""

    def __init__(self, seeds, procs, cache_file=None):
        self.seeds, self.pool, self.cache, self.cache_file = list(seeds), Pool(procs), {}, cache_file
        if cache_file and os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.cache = pickle.load(f)
            print(f"  loaded {len(self.cache)} cached runs from {cache_file}")

    @staticmethod
    def key(cfg, seed, p):
        return (tuple(sorted(cfg.to_dict().items())), seed, p)

    def run(self, cfgs_and_noise):
        todo = []
        for cfg, p in cfgs_and_noise:
            for s in self.seeds:
                k = self.key(cfg, s, p)
                if k not in self.cache and k not in [t[0] for t in todo]:
                    todo.append((k, (cfg.to_dict(), s, p)))
        if todo:
            t0 = time.time()
            for (k, _), res in zip(todo, self.pool.imap(_job, [t[1] for t in todo])):
                self.cache[k] = res
            print(f"  trained {len(todo)} agents in {time.time() - t0:.0f}s")
            if self.cache_file:
                os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
                with open(self.cache_file, "wb") as f:
                    pickle.dump(self.cache, f)
        return [[self.cache[self.key(cfg, s, p)] for s in self.seeds] for cfg, p in cfgs_and_noise]


# metrics
def summarize(runs, label, q_star, opt_ckpt):
    """Average metrics over seeds."""
    def first_episode(r, cond):
        idx = np.flatnonzero(cond(r))
        return float(r["hist"]["ckpt_episode"][idx[0]]) if len(idx) else float("nan")

    near_opt = [first_episode(r, lambda r: r["hist"]["ckpt_reward"] >= opt_ckpt - 0.5) for r in runs]
    solved = [first_episode(r, lambda r: r["hist"]["ckpt_success"] >= 1.0) for r in runs]
    last500 = [r["hist"]["returns"][-500:].mean() for r in runs]
    last500_std = [r["hist"]["returns"][-500:].std() for r in runs]
    ckpt_late_std = [r["hist"]["ckpt_reward"][-40:].std() for r in runs]
    val = [r["val"]["mean_reward"] for r in runs]
    ma = [moving_average(r["hist"]["returns"], 100) for r in runs]
    cross0 = [float(np.flatnonzero(m >= 0)[0] + 1) if (m >= 0).any() else np.nan for m in ma]
    return {
        "label": label,
        "train_return_first100": float(np.mean([r["hist"]["returns"][:100].mean() for r in runs])),
        "episode_ma100_reaches_0": float(np.nanmean(cross0)) if not np.all(np.isnan(cross0)) else None,
        "mean_length_first100": float(np.mean([r["hist"]["lengths"][:100].mean() for r in runs])),
        "mean_length_last500": float(np.mean([r["hist"]["lengths"][-500:].mean() for r in runs])),
        "episodes_to_100pct_success": float(np.nanmean(solved)) if not np.all(np.isnan(solved)) else None,
        "seeds_reaching_near_optimal": int(np.sum(~np.isnan(near_opt))),
        "episodes_to_near_optimal": float(np.nanmean(near_opt)) if not np.all(np.isnan(near_opt)) else None,
        "train_return_last500": float(np.mean(last500)),
        "train_return_last500_std_within_run": float(np.mean(last500_std)),
        "greedy_ckpt_std_last2000": float(np.mean(ckpt_late_std)),
        "val_mean_reward": float(np.mean(val)),
        "val_mean_reward_std_across_seeds": float(np.std(val)),
        "val_success_rate": float(np.mean([r["val"]["success_rate"] for r in runs])),
        "val_mean_steps": float(np.mean([r["val"]["mean_steps"] for r in runs])),
        "policy_agreement_with_optimal": float(np.mean([policy_agreement(r["Q"], q_star) for r in runs])),
    }


def rank_key(s):
    """Highest validation reward, then fastest."""
    speed = s["episodes_to_near_optimal"] if s["episodes_to_near_optimal"] is not None else 1e9
    return (-round(s["val_mean_reward"] / 0.05), speed)


# plots
def plot_baseline(runs, path):
    eps = np.arange(1, BASE.episodes + 1)
    R = np.array([r["hist"]["returns"] for r in runs])
    fig, axes = plt.subplots(3, 1, figsize=(8, 8.2), height_ratios=[3, 3, 1.4], sharex=True)
    ax = axes[0]
    ax.plot(eps, R.mean(0), color=SERIES[0], alpha=0.25, linewidth=0.6, label="per-episode reward (mean of seeds)")
    band(ax, eps, R, SERIES[0], "100-episode moving average")
    ax.set_ylabel("Training reward")
    ax.set_title("C. Baseline: reward per training episode")
    ax.legend(loc="lower right")
    ax = axes[1]
    band(ax, eps, R, SERIES[0], "training reward (100-ep MA, ε-greedy)")
    ck = runs[0]["hist"]["ckpt_episode"]
    C = np.array([r["hist"]["ckpt_reward"] for r in runs])
    ax.plot(ck, C.mean(0), color=SERIES[1], marker="o", markersize=3.5, linewidth=1.5,
            label="greedy policy (ε = 0), 20 test episodes")
    ax.set_ylim(-60, 15)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_ylabel("Reward (zoomed)")
    ax.legend(loc="lower right")
    ax = axes[2]
    ax.plot(eps, runs[0]["hist"]["epsilon"], color=INK2, linewidth=1.5)
    ax.set_ylabel("ε")
    ax.set_xlabel("Training episode")
    save(fig, path)


def plot_sweep(name, values, runs_per_value, path, title):
    eps = np.arange(1, BASE.episodes + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, (v, runs) in enumerate(zip(values, runs_per_value)):
        lab = f"{SYMBOL[name]} = {v}"
        band(axes[0], eps, [r["hist"]["returns"] for r in runs], SERIES[i], lab)
        ck = runs[0]["hist"]["ckpt_episode"]
        C = np.array([r["hist"]["ckpt_reward"] for r in runs])
        axes[1].fill_between(ck, C.mean(0) - C.std(0), C.mean(0) + C.std(0), color=SERIES[i], alpha=0.15, linewidth=0)
        axes[1].plot(ck, C.mean(0), color=SERIES[i], label=lab)
    axes[0].set_ylim(-250, 15)
    axes[0].set_title("Training reward (100-episode moving avg.)")
    axes[1].set_ylim(-210, 15)
    axes[1].set_title("Greedy-policy reward during training")
    for ax in axes:
        ax.set_xlabel("Training episode")
        ax.axhline(0, color=MUTED, linewidth=0.8)
    axes[0].set_ylabel("Reward")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", color="#0b0b0b")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(values), bbox_to_anchor=(0.5, 0.93))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_eval(best_runs, base_runs, best_test, base_test, opt_test, best_cfg, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), width_ratios=[1.7, 1])
    ax = axes[0]
    for i, (lab, runs) in enumerate([("baseline (α=0.1, ε-decay=0.999)", base_runs),
                                     (f"best (α={best_cfg.alpha}, ε-decay={best_cfg.epsilon_decay})", best_runs)]):
        ck = runs[0]["hist"]["ckpt_episode"]
        C = np.array([r["hist"]["ckpt_reward"] for r in runs])
        ax.fill_between(ck, C.mean(0) - C.std(0), C.mean(0) + C.std(0), color=SERIES[i], alpha=0.15, linewidth=0)
        ax.plot(ck, C.mean(0), color=SERIES[i], label=lab)
    ax.set_xlim(0, 3500)
    ax.set_ylim(-210, 15)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Greedy-policy reward")
    ax.set_title("Greedy policy during training (mean ± std, 10 seeds)")
    ax.legend(loc="lower right")
    ax = axes[1]
    labels = ["Baseline", "Best", "Optimal\n(value iter.)"]
    means = [base_test["mean_reward"], best_test["mean_reward"], opt_test["mean_reward"]]
    errs = [base_test["std_mean_reward_across_agents"], best_test["std_mean_reward_across_agents"], 0]
    ax.bar(labels, means, yerr=errs, color=SERIES[:3], width=0.55, capsize=3)
    for i, v in enumerate(means):
        ax.text(i, v + 0.25, f"{v:.2f}", ha="center", color=INK2)
    ax.set_ylim(0, max(means) + 1.8)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Test reward, ε = 0 ({TEST_EPISODES} held-out episodes)")
    save(fig, path)


def plot_grid(grid_summ, opt_val, path):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    E = np.array([s["episodes_to_near_optimal"] or np.nan for s in grid_summ]).reshape(len(GRID_ALPHA), len(GRID_DECAY))
    V = np.array([s["val_mean_reward"] for s in grid_summ]).reshape(E.shape)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    im = ax.imshow(E, cmap=cmap, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(GRID_DECAY)), [f"{d}\n(ε→0.05 by ep {int(np.log(0.05) / np.log(d))})" for d in GRID_DECAY], fontsize=8.5)
    ax.set_yticks(range(len(GRID_ALPHA)), [f"α = {a}" for a in GRID_ALPHA])
    ax.set_xlabel("ε-decay per episode")
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            dark = E[i, j] > np.nanmean(E)
            ax.text(j, i, f"{E[i, j]:.0f} eps\nval {V[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="#fcfcfb" if dark else "#0b0b0b")
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("Episodes to near-optimal", color=INK2)
    ax.set_title(f"G. Joint α × ε-decay search (γ = 0.99); optimal val reward = {opt_val:.2f}")
    save(fig, path)


def plot_alpha_noise(own, opt_cross, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.array(NOISE_LEVELS)
    for i, a in enumerate(H_ALPHAS):
        m, s = own[a].mean(1), own[a].std(1)
        ax.errorbar(x + (i - 1) * 0.006, m, yerr=s, color=SERIES[i], marker="o", markersize=5,
                    capsize=3, linewidth=2, label=f"α = {a}")
    ax.plot(x, opt_cross, color=INK2, linestyle="--", linewidth=1.2, label="optimal (VI)")
    ax.set_xticks(x)
    ax.set_xlabel("Noise level p (train = test)")
    ax.set_ylabel("Mean greedy reward")
    ax.set_title("H. Learning rate under action noise")
    ax.legend(loc="lower left")
    save(fig, path)


def plot_noise(train_runs, cross, opt_cross, path, alpha):
    eps = np.arange(1, BASE.episodes + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for i, (p, runs) in enumerate(zip(NOISE_LEVELS, train_runs)):
        band(axes[0], eps, [r["hist"]["returns"] for r in runs], SERIES[i], f"trained with p = {p}")
    axes[0].set_ylim(-250, 15)
    axes[0].set_title(f"H. Training reward under noise (α = {alpha})")
    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel("Reward")
    axes[0].axhline(0, color=MUTED, linewidth=0.8)
    ax = axes[1]
    x = np.array(NOISE_LEVELS)
    for i, p in enumerate(NOISE_LEVELS):
        m = cross[:, i, :].mean(0)
        s = cross[:, i, :].std(0)
        ax.errorbar(x + (i - 1.5) * 0.006, m, yerr=s, color=SERIES[i], marker="o", markersize=5,
                    capsize=3, linewidth=2, label=f"policy trained with p = {p}")
    ax.plot(x, opt_cross, color=INK2, linestyle="--", linewidth=1.2, label="optimal for each test p (VI)")
    ax.set_xticks(x)
    ax.set_xlabel("Noise level p at test time")
    ax.set_ylabel("Mean greedy reward")
    ax.set_title("Robustness: trained policies under each noise level")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8.5)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8.5)
    save(fig, path)


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--cache", action="store_true",
                    help="keep trained runs in results/.cache/ so re-running only redraws plots")
    args = ap.parse_args()
    os.makedirs(PLOTS, exist_ok=True)
    runner = Runner(range(args.seeds), args.procs,
                    os.path.join(OUT, ".cache", f"runs_{args.seeds}seeds.pkl") if args.cache else None)
    t_start = time.time()

    q_star = value_iteration(0.0)
    opt_ckpt = run_greedy(q_star, CHECKPOINT_EPISODES, CHECKPOINT_SEED)["mean_reward"]
    opt_val = run_greedy(q_star, VAL_EPISODES, VAL_SEED0)["mean_reward"]
    results = {"seeds": args.seeds, "baseline_config": BASE.to_dict(),
               "optimal": {"checkpoint_mean": opt_ckpt, "val_mean": opt_val}}
    all_summaries = []

    # C. baseline
    print("C: baseline")
    (base_runs,) = runner.run([(BASE, 0.0)])
    results["baseline"] = summarize(base_runs, "baseline", q_star, opt_ckpt)
    plot_baseline(base_runs, os.path.join(PLOTS, "C_baseline.png"))

    # D-F. one-at-a-time sweeps
    titles = {"alpha": "D. Learning-rate experiment (γ = 0.99, ε-decay = 0.999)",
              "gamma": "E. Discount-factor experiment (α = 0.1, ε-decay = 0.999)",
              "epsilon_decay": "F. Exploration-decay experiment (α = 0.1, γ = 0.99, ε: 1.0 → 0.05)"}
    part = {"alpha": "D", "gamma": "E", "epsilon_decay": "F"}
    winners = {}
    for name, (field, values) in SWEEPS.items():
        print(f"{part[name]}: sweep {name} {values}")
        cfgs = [QConfig(**{**BASE.to_dict(), field: v}) for v in values]
        runs = runner.run([(c, 0.0) for c in cfgs])
        summ = [summarize(r, f"{name}={v}", q_star, opt_ckpt) | {"value": v} for v, r in zip(values, runs)]
        results[name] = summ
        all_summaries += [(c, s) for c, s in zip(cfgs, summ)]
        winners[field] = min(summ, key=rank_key)["value"]
        plot_sweep(name, values, runs, os.path.join(PLOTS, f"{part[name]}_{name}.png"), titles[name])

    # G. grid search over alpha and epsilon decay
    print(f"G: grid alpha {GRID_ALPHA} x epsilon_decay {GRID_DECAY}, gamma={winners['gamma']}")
    grid_cfgs = [[QConfig(**{**BASE.to_dict(), "gamma": winners["gamma"], "alpha": a, "epsilon_decay": d})
                  for d in GRID_DECAY] for a in GRID_ALPHA]
    flat = [c for row in grid_cfgs for c in row]
    grid_runs = runner.run([(c, 0.0) for c in flat])
    grid_summ = [summarize(r, f"alpha={c.alpha},decay={c.epsilon_decay}", q_star, opt_ckpt)
                 for c, r in zip(flat, grid_runs)]
    all_summaries += list(zip(flat, grid_summ))
    plot_grid(grid_summ, opt_val, os.path.join(PLOTS, "G_grid.png"))
    best_cfg, best_sum = min(all_summaries, key=lambda cs: rank_key(cs[1]))
    (best_runs,) = runner.run([(best_cfg, 0.0)])

    def test_metrics(runs):
        ms = [run_greedy(r["Q"], TEST_EPISODES, TEST_SEED0) for r in runs]
        agg = {k: float(np.mean([m[k] for m in ms])) for k in
               ("mean_reward", "success_rate", "mean_steps", "illegal_actions_per_ep")}
        agg["std_reward_within_agent"] = float(np.mean([m["std_reward"] for m in ms]))
        agg["std_mean_reward_across_agents"] = float(np.std([m["mean_reward"] for m in ms]))
        agg["min_agent_mean_reward"] = float(min(m["mean_reward"] for m in ms))
        agg["policy_agreement_with_optimal"] = float(np.mean([policy_agreement(r["Q"], q_star) for r in runs]))
        agg["returns"] = ms[int(np.argmax([m["mean_reward"] for m in ms]))]["returns"]
        return agg, ms

    best_test, best_ms = test_metrics(best_runs)
    base_test, _ = test_metrics(base_runs)
    opt_test = run_greedy(q_star, TEST_EPISODES, TEST_SEED0)
    # agent used for the video
    rep = int(np.argmax([r["val"]["mean_reward"] for r in best_runs]))
    np.save(os.path.join(OUT, "best_q_table.npy"), best_runs[rep]["Q"])
    np.save(os.path.join(OUT, "baseline_q_table.npy"), base_runs[0]["Q"])
    results["G"] = {
        "sweep_winners": winners, "grid_alpha": GRID_ALPHA, "grid_decay": GRID_DECAY,
        "grid": grid_summ, "best_config": best_cfg.to_dict(), "best_summary_validation": best_sum,
        "best_test": {k: v for k, v in best_test.items() if k != "returns"},
        "baseline_test": {k: v for k, v in base_test.items() if k != "returns"},
        "optimal_test": {k: v for k, v in opt_test.items() if k not in ("returns", "steps")},
        "best_per_agent_test_mean": [m["mean_reward"] for m in best_ms],
        "representative_seed": rep, "test_episodes": TEST_EPISODES, "agents": args.seeds,
    }
    plot_eval(best_runs, base_runs, best_test, base_test, opt_test, best_cfg, os.path.join(PLOTS, "G_evaluation.png"))
    with open(os.path.join(OUT, "best_config.json"), "w") as f:
        json.dump(best_cfg.to_dict(), f, indent=2)

    # H. stochastic environment
    print("H: action noise", NOISE_LEVELS, "alphas", H_ALPHAS)
    q_star_p = {p: value_iteration(p) for p in NOISE_LEVELS}
    opt_cross = [run_greedy(q_star_p[p], NOISE_EVAL_EPISODES, TEST_SEED0, p)["mean_reward"] for p in NOISE_LEVELS]
    # compare learning rates under noise
    own = {}
    for a in H_ALPHAS:
        cfg_a = QConfig(**{**best_cfg.to_dict(), "alpha": a})
        runs_a = runner.run([(cfg_a, p) for p in NOISE_LEVELS])
        own[a] = np.array([[run_greedy(r["Q"], NOISE_EVAL_EPISODES, TEST_SEED0, p)["mean_reward"] for r in runs]
                           for p, runs in zip(NOISE_LEVELS, runs_a)])
    h_alpha = max(H_ALPHAS, key=lambda a: own[a][1:].mean())
    plot_alpha_noise(own, opt_cross, os.path.join(PLOTS, "H_alpha_vs_noise.png"))
    # train at each p, test at every p
    h_cfg = QConfig(**{**best_cfg.to_dict(), "alpha": h_alpha})
    noise_runs = runner.run([(h_cfg, p) for p in NOISE_LEVELS])
    cross =np.zeros((args.seeds, len(NOISE_LEVELS), len(NOISE_LEVELS)))
    cross_success = np.zeros_like(cross)
    cross_steps = np.zeros_like(cross)
    for i, runs in enumerate(noise_runs):
        for k, r in enumerate(runs):
            for j, p_test in enumerate(NOISE_LEVELS):
                m = run_greedy(r["Q"], NOISE_EVAL_EPISODES, TEST_SEED0, p_test)
                cross[k, i, j], cross_success[k, i, j], cross_steps[k, i, j] = (
                    m["mean_reward"], m["success_rate"], m["mean_steps"])
    results["H"] = {
        "noise_levels": NOISE_LEVELS,
        "alpha_vs_noise_mean": {str(a): own[a].mean(1).tolist() for a in H_ALPHAS},
        "alpha_vs_noise_std": {str(a): own[a].std(1).tolist() for a in H_ALPHAS},
        "h_config": h_cfg.to_dict(),
        "train_summaries": [summarize(r, f"p={p}", q_star_p[p],
                                      run_greedy(q_star_p[p], CHECKPOINT_EPISODES, CHECKPOINT_SEED, p)["mean_reward"])
                            for p, r in zip(NOISE_LEVELS, noise_runs)],
        "cross_mean_reward": cross.mean(0).tolist(),
        "cross_std_reward_across_agents": cross.std(0).tolist(),
        "cross_success_rate": cross_success.mean(0).tolist(),
        "cross_mean_steps": cross_steps.mean(0).tolist(),
        "optimal_under_each_p": opt_cross,
        "agreement_with_deterministic_optimal": [
            float(np.mean([policy_agreement(r["Q"], q_star) for r in runs])) for runs in noise_runs],
        "eval_episodes": NOISE_EVAL_EPISODES,
    }
    plot_noise(noise_runs, cross, opt_cross, os.path.join(PLOTS, "H_noise.png"), h_alpha)

    # summary table
    with open(os.path.join(OUT, "summary.csv"), "w") as f:
        cols = ["alpha", "gamma", "epsilon_decay", "episodes_to_near_optimal", "seeds_reaching_near_optimal",
                "train_return_last500", "greedy_ckpt_std_last2000", "val_mean_reward",
                "val_mean_reward_std_across_seeds", "val_success_rate", "val_mean_steps",
                "policy_agreement_with_optimal"]
        f.write(",".join(cols) + "\n")
        seen = set()
        for c, s in [(BASE, results["baseline"])] + all_summaries:
            key = (c.alpha, c.gamma, c.epsilon_decay)
            if key in seen:
                continue
            seen.add(key)
            row = [c.alpha, c.gamma, c.epsilon_decay] + [s[k] for k in cols[3:]]
            f.write(",".join("" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in row) + "\n")

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"done in {time.time() - t_start:.0f}s -> {OUT}/results.json, {PLOTS}/")


if __name__ == "__main__":
    main()

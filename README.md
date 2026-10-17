# Taxi Q-Learning

CPSC 4420 Assignment 2 - Terrell Fleming

Tabular Q-learning on the Gymnasium Taxi environment. The agent is written from scratch with numpy.
I trained it with different learning rates, discount factors, and epsilon decay rates, picked the best
setup, and then tested how it does when random action noise is added.

- Report: [report.pdf](report.pdf)
- Video of the trained agent: [results/trained_agent.mp4](results/trained_agent.mp4)
- Plots: [results/plots](results/plots)

## Setup

Needs Python 3.10+.

```
git clone https://github.com/terrellf17/taxi-qlearning.git
cd taxi-qlearning
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Mac/Linux use `source .venv/bin/activate` instead of the second-to-last line.

If PowerShell says scripts are disabled, run this once and try again:

```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## How to run

Train a single agent (defaults: alpha 0.1, gamma 0.99, epsilon decay 0.999, 5000 episodes):

```
python -m qlearn.train
```

You can change any hyperparameter from the command line, for example:

```
python -m qlearn.train --alpha 0.5 --gamma 0.9 --epsilon-decay 0.995
python -m qlearn.train --noise 0.25
```

Run all the experiments for parts C through H. This trains 10 agents per setting so it takes a while
(around 15-30 min). It saves the plots to `results/plots/` and the numbers to `results/results.json`
and `results/summary.csv`.

```
python experiments.py --cache
```

Make the video of the best agent:

```
python record_video.py --renderer gym
```

## Files

```
qlearn/
  envs.py       makes the Taxi env + the action noise wrapper for part H
  agent.py      Q-learning agent (Q-table, epsilon-greedy, update rule, epsilon decay)
  train.py      training loop and greedy evaluation
  optimal.py    value iteration, only used to compare against the optimal policy
  plotting.py   helpers for the graphs
experiments.py  runs parts C-H and makes all the plots
record_video.py records the trained agent to mp4
results/        plots, results.json, summary.csv, saved Q-tables, video
report.pdf      the report
```

## Results

| | alpha | gamma | eps decay | episodes to learn | test reward |
|---|---|---|---|---|---|
| Baseline | 0.1 | 0.99 | 0.999 | ~2370 | 7.58 |
| Best | 0.9 | 0.99 | 0.9995 | ~435 | 7.58 |
| Optimal (value iteration) | | | | | 7.58 |

Both the baseline and the best setup end up at the optimal policy, the best one just gets there a lot faster.
Low gamma (0.5, 0.7) doesn't work because the +20 reward gets discounted away before the agent can
see it. With action noise, a high learning rate falls apart and alpha = 0.1 works best.
More details are in the report.

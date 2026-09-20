"""How long would on-stage training take? Offline test of OnlineLearner: brain responses are sampled from the saved
response bank (data/bank-real.npz) instead of running the simulator, so this needs no GPU and runs in seconds.

Run: .venv/Scripts/python scripts/live_learning_test.py [--flies 16] [--rate 0.02]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain.connectome import DATA
from flybrain.readout import OnlineLearner
from flybrain.snake import ALL_STATES, Arena

parser = argparse.ArgumentParser()
parser.add_argument("--flies", type=int, nargs="+", default=[1, 16])
parser.add_argument("--rate", type=float, nargs="+", default=[0.005, 0.02, 0.08])
parser.add_argument("--rounds", type=int, default=1500, help="moves per fly")
parser.add_argument("--seconds-per-round", type=float, default=0.25, help="measured wall time of one batched brain window")
parser.add_argument("--bank", default="bank-real")
parser.add_argument("--seeds", type=int, default=1, help="independent repeats (different boards, exploration and brain-response samples)")
args = parser.parse_args()

bank = torch.as_tensor(np.load(DATA / f"{args.bank}.npz")["counts"])  # [24 states, trials, R]
state_index = {state: i for i, state in enumerate(ALL_STATES)}

for flies in args.flies:
    for rate in args.rate:
        finals, reached = [], []
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            arenas = [Arena(seed=1000 * seed + i) for i in range(flies)]
            learner = OnlineLearner(bank.shape[2], rate=rate, seed=seed)
            finished, milestones = [], {}
            for round_index in range(args.rounds):
                rows = [bank[state_index[a.state()], rng.integers(bank.shape[1])] for a in arenas]
                actions, _ = learner.act(torch.stack(rows, dim=1))
                rewards = [a.step({0: int(action)}).get(0, 0.0) for a, action in zip(arenas, actions)]
                learner.learn(torch.tensor(rewards))
                finished += [a.snakes[0].last_score for a in arenas if not a.snakes[0].alive]
                recent = np.mean(finished[-20:]) if len(finished) >= 10 else 0
                for target in (3, 8, 15):
                    if recent >= target and target not in milestones:
                        milestones[target] = round_index * args.seconds_per_round
            finals.append(np.mean(finished[-20:]))
            reached.append(milestones)
        summary = "  ".join(f"avg>={t}: {np.mean([m[t] for m in reached if t in m]):5.0f}s in {sum(t in m for m in reached)}/{args.seeds}" if any(t in m for m in reached)
                            else f"avg>={t}: never" for t in (3, 8, 15))
        print(f"flies {flies:3d}  rate {rate:<6}  {summary}   final last-20 average: {np.mean(finals):5.1f} +- {np.std(finals):4.1f}  (per seed: {', '.join(f'{f:.1f}' for f in finals)})")

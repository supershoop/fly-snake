"""Fair wiring control: the nothing-trained policy (turn toward the side whose DNa02/DNa01 fire more) on the real
wiring vs several independently scrambled networks. Nothing is fitted to any of them.

Run: .venv/Scripts/python scripts/untrained_control.py [--games 32] [--shuffles 3]
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain.brain import Brain
from flybrain.channels import build_channels
from flybrain.connectome import load_connectome
from flybrain.readout import HardwiredPolicy
from flybrain.snake import Snake

parser = argparse.ArgumentParser()
parser.add_argument("--games", type=int, default=32)
parser.add_argument("--shuffles", type=int, default=3)
parser.add_argument("--max-moves", type=int, default=300)
args = parser.parse_args()

connectome = load_connectome()
channels = build_channels(connectome)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stim_index, readout_index = channels.stim_index.to(device), channels.readout_index.to(device)
policy = HardwiredPolicy(channels.steer_sign)
steering = channels.steer_sign.to(device) != 0

for label, options in [("real wiring", {})] + [(f"scrambled #{i}", {"shuffled": True, "shuffle_seed": i}) for i in range(args.shuffles)]:
    games = [Snake(seed=i) for i in range(args.games)]
    moves, steering_spikes = np.zeros(args.games), 0.0
    brain = Brain(connectome, batch=args.games, seed=1, **options)
    for _ in range(args.max_moves):
        if not any(g.alive for g in games):
            break
        levels = channels.levels(torch.as_tensor(np.stack([g.encode() for g in games]), device=device).T)
        counts = brain.run(100, stim_index, levels, readout_index)
        steering_spikes += counts[steering].sum().item()
        for i, (game, action) in enumerate(zip(games, policy.act(counts)[0].tolist())):
            if game.alive:
                game.step(action)
                moves[i] += 1
    scores = np.array([g.score for g in games])
    print(f"{label:14s} mean score {scores.mean():5.2f}  max {scores.max():2d}  games with any food {np.mean(scores > 0):4.0%}  "
          f"mean moves survived {moves.mean():5.1f}  steering-neuron spikes per move {steering_spikes / max(1, moves.sum()):6.1f}")

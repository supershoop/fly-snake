"""Why did the scrambled-wiring control score 0 live although its readout was 93.8% accurate on isolated responses?
Hypothesis: activity carries over between moves in the scrambled network, so live features differ from the bank.

Test 1: after 100 ms of stimulation, how many neurons still fire during the next 100 ms with NO input?
Test 2: play with the brain reset before every move (exactly the bank's condition) vs the normal carry-over.

Run: .venv/Scripts/python scripts/scrambled_check.py [--games 16] [--max-moves 300]
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
from flybrain.readout import Policy
from flybrain.snake import ALL_STATES, Snake, encode_state

parser = argparse.ArgumentParser()
parser.add_argument("--games", type=int, default=16)
parser.add_argument("--max-moves", type=int, default=300)
args = parser.parse_args()

connectome = load_connectome()
channels = build_channels(connectome)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stim_index, readout_index = channels.stim_index.to(device), channels.readout_index.to(device)
state_levels = channels.levels(torch.as_tensor(np.stack([encode_state(s) for s in ALL_STATES]), device=device).T)

for wiring in ("real", "shuffled"):
    brain = Brain(connectome, batch=len(ALL_STATES), shuffled=wiring == "shuffled")
    during = brain.run(100, stim_index, state_levels)
    after = brain.run(100)
    later = brain.run(100)
    print(f"[{wiring}] neurons active: during stimulus {(during > 0).sum(0).float().mean():7.0f} · next 100 ms, no input "
          f"{(after > 0).sum(0).float().mean():7.0f} · 100-200 ms after {(later > 0).sum(0).float().mean():7.0f}"
          f" · descending neurons still firing {(after[readout_index] > 0).sum(0).float().mean():5.0f}")

    policy = Policy.load(f"readout-{wiring}", device)
    for reset_each_move in (False, True):
        games = [Snake(seed=i) for i in range(args.games)]
        brain = Brain(connectome, batch=args.games, shuffled=wiring == "shuffled", seed=1)
        for _ in range(args.max_moves):
            if not any(g.alive for g in games):
                break
            if reset_each_move:
                brain.reset()
            levels = channels.levels(torch.as_tensor(np.stack([g.encode() for g in games]), device=device).T)
            actions = policy.act(brain.run(100, stim_index, levels, readout_index))[0].tolist()
            for game, action in zip(games, actions):
                game.step(action)
        scores = np.array([g.score for g in games])
        print(f"[{wiring}] trained readout, {'brain reset every move' if reset_each_move else 'activity carries over  '}: mean score {scores.mean():5.2f}  max {scores.max()}")

"""Play Snake through the retinotopic encoder (flybrain/vision.py): collect descending-neuron responses from real
games, fit the linear readout to the teacher's moves, then score it and an untrained policy with the live brain.

Run: .venv/Scripts/python scripts/train_vision.py [--collect-moves 150] [--games 32]
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
from flybrain.readout import Policy, fit
from flybrain.snake import Arena, Snake, teacher
from flybrain.vision import build_retina

parser = argparse.ArgumentParser()
parser.add_argument("--collect-games", type=int, default=48)
parser.add_argument("--collect-moves", type=int, default=150)
parser.add_argument("--games", type=int, default=32)
parser.add_argument("--max-moves", type=int, default=300)
parser.add_argument("--window", type=float, default=100)
args = parser.parse_args()

connectome = load_connectome()
retina = build_retina(connectome)
channels = build_channels(connectome)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stim_index, readout_index = retina.index.to(device), channels.readout_index.to(device)
readout = connectome.neurons.loc[channels.readout_index.numpy()]
sign = lambda kind: torch.as_tensor(np.where(readout["type"].eq(kind) & readout["side"].eq("L"), 1.0,
                                             np.where(readout["type"].eq(kind) & readout["side"].eq("R"), -1.0, 0.0)), dtype=torch.float32, device=device)
TOWARD, AWAY = sign("DNa02"), sign("DNp01")  # turn toward the food side, away from the side the giant fiber fires more


def see(arenas):
    return torch.as_tensor(np.stack([retina.render(a) for a in arenas], axis=1), device=device)


def untrained(dn_counts, threshold=2.0):
    """No learning: left-minus-right steering neuron spikes, minus left-minus-right giant fiber spikes."""
    drive = TOWARD @ dn_counts - 0.5 * (AWAY @ dn_counts)
    return torch.where(drive > threshold, 0, torch.where(drive < -threshold, 2, 1))


# 1. collect: half the games follow the teacher, half follow the untrained policy, all labelled by the teacher
arenas = [Arena(seed=i) for i in range(args.collect_games)]
brain = Brain(connectome, batch=len(arenas))
features, labels = [], []
for move in range(args.collect_moves):
    counts = brain.run(args.window, stim_index, see(arenas), readout_index)
    wanted = [teacher(a.state()) for a in arenas]
    own = untrained(counts).tolist()
    features.append(counts.T.clone())
    labels += wanted
    for i, arena in enumerate(arenas):
        arena.step({0: wanted[i] if i % 2 == 0 else own[i]})
features, labels = torch.cat(features), torch.as_tensor(labels, device=device)
shuffle = torch.randperm(len(labels), device=device)
split = len(labels) * 4 // 5
policy = fit(features[shuffle[:split]], labels[shuffle[:split]])
held = shuffle[split:]
accuracy = (policy.act(features[held].T)[0] == labels[held]).float().mean().item()
print(f"collected {len(labels):,} moves; readout matches teacher on held-out moves: {accuracy:.1%}")
policy.save("readout-vision")
del brain


# 2. score with the live brain in the loop
def play(controller, label):
    games = [Snake(seed=1000 + i) for i in range(args.games)]
    brain = Brain(connectome, batch=args.games, seed=1)
    for _ in range(args.max_moves):
        if not any(g.alive for g in games):
            break
        actions = controller(brain.run(args.window, stim_index, see(games), readout_index)).tolist()
        for game, action in zip(games, actions):
            game.step(action)
    scores = np.array([g.score for g in games])
    print(f"[vision] {label:34s} mean score {scores.mean():5.2f}  median {np.median(scores):4.1f}  max {scores.max()}")


play(lambda counts: policy.act(counts)[0], "retina + trained readout")
play(untrained, "retina, nothing trained")

"""Lesion table: silence a few neuron types and measure how Snake play breaks, with the live brain in the loop.
Every lesion runs in the same batch (different brains in one simulation), for both the trained readout and the
nothing-trained instinct policy (pursuit steering + giant-fiber veto and dodge).

Run: .venv/Scripts/python scripts/lesion_scores.py [--games 8] [--max-moves 300]
Relay cells were found by path search in the connectome: LC10 -> AOTU025/012/015 -> DNa02 (no direct synapses);
LC4 -> giant fiber DNp01 directly; LC4 -> PVLP141/PVLP137 -> contralateral DNa01.
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
from flybrain.readout import InstinctPolicy, Policy
from flybrain.snake import Snake

parser = argparse.ArgumentParser()
parser.add_argument("--games", type=int, default=8, help="games per lesion")
parser.add_argument("--max-moves", type=int, default=300)
parser.add_argument("--window", type=float, default=100)
parser.add_argument("--seed-offset", type=int, default=0, help="first board seed, to add games to an earlier run without repeating boards")
args = parser.parse_args()

LESIONS = {  # label -> regexes on annotation `type` (full match); "random:N" = N random central-brain neurons
    "intact": [],
    "AOTU relays (025, 012, 015)": ["AOTU025", "AOTU012", "AOTU015"],
    "AOTU025 only": ["AOTU025"],
    "steering DN: DNa02": ["DNa02"],
    "steering DNs: DNa02 + DNa01": ["DNa02", "DNa01"],
    "threat relays PVLP141 + PVLP137": ["PVLP141", "PVLP137"],
    "giant fiber DNp01": ["DNp01"],
    "control: 12 random neurons": ["random:12"],
    "control: 2000 random neurons": ["random:2000"],
}

connectome = load_connectome()
channels = build_channels(connectome)
neurons = connectome.neurons
kind = neurons["type"].fillna("")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stim_index, readout_index = channels.stim_index.to(device), channels.readout_index.to(device)
rng = np.random.default_rng(0)
central = np.flatnonzero(neurons["superclass"].eq("cb_intrinsic").to_numpy())

batch = len(LESIONS) * args.games
mask = np.zeros((connectome.n, batch), dtype=bool)
sizes = {}
for row, (label, patterns) in enumerate(LESIONS.items()):
    silenced = np.zeros(connectome.n, dtype=bool)
    for pattern in patterns:
        if pattern.startswith("random:"):
            silenced[rng.choice(central, int(pattern.split(":")[1]), replace=False)] = True
        else:
            silenced |= kind.str.fullmatch(pattern).to_numpy()
    sizes[label] = int(silenced.sum())
    mask[:, row * args.games:(row + 1) * args.games] = silenced[:, None]

readout = neurons.loc[channels.readout_index.numpy()]
groups = {f"{name}_{s}": torch.as_tensor(np.flatnonzero((readout["type"].eq(name) & readout["side"].eq(s)).to_numpy()), device=device)
          for name in ("DNa02", "DNa01", "DNp01") for s in "LR"}
policies = {"trained readout": Policy.load("readout-real", device), "nothing trained (instinct)": InstinctPolicy(groups, args.window)}
results = {}
for policy_name, policy in policies.items():
    games = [Snake(seed=args.seed_offset + i % args.games) for i in range(batch)]  # the same boards for every lesion
    moves = np.zeros(batch)
    brain = Brain(connectome, batch=batch, seed=1)
    brain.set_lesion(torch.as_tensor(mask))
    for _ in range(args.max_moves):
        if not any(g.alive for g in games):
            break
        levels = channels.levels(torch.as_tensor(np.stack([g.encode() for g in games]), device=device).T)
        actions = policy.act(brain.run(args.window, stim_index, levels, readout_index))[0].tolist()
        for i, (game, action) in enumerate(zip(games, actions)):
            if game.alive:
                game.step(action)
                moves[i] += 1
    scores = np.array([g.score for g in games]).reshape(len(LESIONS), args.games)
    results[policy_name] = (scores.mean(axis=1), moves.reshape(len(LESIONS), args.games).mean(axis=1), scores.std(axis=1) / np.sqrt(args.games))

print(f"\n{args.games} games per lesion, max {args.max_moves} moves, live brain in the loop. score = food eaten, moves = survival\n")
print(f"{'lesion':36s} {'cells':>6s} | " + " | ".join(f"{name:>26s}" for name in policies))
print(f"{'':36s} {'':>6s} | " + " | ".join(f"{'score +- sem':>16s} {'moves':>9s}" for _ in policies))
for row, label in enumerate(LESIONS):
    print(f"{label:36s} {sizes[label]:6d} | " + " | ".join(f"{results[p][0][row]:9.2f} +-{results[p][2][row]:4.2f} {results[p][1][row]:9.0f}" for p in policies))

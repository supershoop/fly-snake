"""Build the response bank, fit the readout, and score it with the live brain in the loop.

Run: .venv/Scripts/python scripts/train_readout.py [--shuffled] [--trials 48] [--games 32]
Bank: for each of the 24 game situations, `trials` independent 100 ms brain runs -> descending-neuron spike counts.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain.brain import Brain
from flybrain.channels import build_channels
from flybrain.connectome import DATA, load_connectome
from flybrain.readout import HardwiredPolicy, Policy, fit
from flybrain.snake import ALL_STATES, Snake, encode_state, teacher

parser = argparse.ArgumentParser()
parser.add_argument("--shuffled", action="store_true", help="control: scrambled wiring")
parser.add_argument("--trials", type=int, default=48)
parser.add_argument("--window", type=float, default=100, help="ms of brain time per move")
parser.add_argument("--games", type=int, default=32)
parser.add_argument("--max-moves", type=int, default=400)
parser.add_argument("--rebuild", action="store_true")
args = parser.parse_args()
name = "shuffled" if args.shuffled else "real"

connectome = load_connectome()
channels = build_channels(connectome)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stim_index, readout_index = channels.stim_index.to(device), channels.readout_index.to(device)
state_levels = torch.as_tensor(np.stack([encode_state(s) for s in ALL_STATES]), device=device).T  # [C, 24]
labels = torch.as_tensor([teacher(s) for s in ALL_STATES], device=device)

bank_path = DATA / f"bank-{name}.npz"
if bank_path.exists() and not args.rebuild:
    bank = torch.as_tensor(np.load(bank_path)["counts"], device=device)
else:
    start, runs = time.time(), []
    brain = Brain(connectome, batch=len(ALL_STATES) * 8, shuffled=args.shuffled)
    levels = channels.levels(state_levels.repeat_interleave(8, dim=1))
    for trial in range(args.trials // 8):
        brain.reset()
        runs.append(brain.run(args.window, stim_index, levels, readout_index).reshape(len(readout_index), len(ALL_STATES), 8))
    bank = torch.cat(runs, dim=2).permute(1, 2, 0).contiguous()  # [24 states, trials, R]
    np.savez_compressed(bank_path, counts=bank.cpu().numpy())
    print(f"bank built in {time.time() - start:.0f}s: {tuple(bank.shape)}")
    del brain

trials = bank.shape[1]
split = trials * 3 // 4
flat = lambda part: (part.reshape(-1, part.shape[-1]), labels.repeat_interleave(part.shape[1]))
policy = fit(*flat(bank[:, :split]))
held_features, held_labels = flat(bank[:, split:])
accuracy = (policy.act(held_features.T)[0] == held_labels).float().mean().item()
print(f"[{name}] readout matches teacher on held-out brain responses: {accuracy:.1%}")
policy.save(f"readout-{name}")


def play(controller, label):
    """`games` snakes at once, each driven by its own copy of the brain (state carries over between moves)."""
    games = [Snake(seed=i) for i in range(args.games)]
    brain = Brain(connectome, batch=args.games, shuffled=args.shuffled, seed=1)
    for _ in range(args.max_moves):
        if not any(g.alive for g in games):
            break
        levels = channels.levels(torch.as_tensor(np.stack([g.encode() for g in games]), device=device).T)
        counts = brain.run(args.window, stim_index, levels, readout_index)
        actions = controller(counts)
        for game, action in zip(games, actions.tolist()):
            game.step(action)
    scores = np.array([g.score for g in games])
    print(f"[{name}] {label:28s} mean score {scores.mean():5.2f}  median {np.median(scores):4.1f}  max {scores.max()}")


play(lambda counts: policy.act(counts)[0], "brain + trained readout")
play(lambda counts: HardwiredPolicy(channels.steer_sign).act(counts)[0], "brain, hardwired DNa02/DNa01")
play(lambda counts: torch.randint(0, 3, (counts.shape[1],)), "random actions (no brain)")

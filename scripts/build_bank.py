"""Response bank for the offline learning tests: for each of the 24 game situations, `trials` independent 100 ms brain runs
from rest -> descending-neuron spike counts. Same recipe as the original data/bank-real.npz and bank-shuffled.npz.

Run: .venv/Scripts/python scripts/build_bank.py --shuffle-seed 1 --out bank-shuffled-1
     .venv/Scripts/python scripts/build_bank.py --out bank-real-check          (real wiring)
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
from flybrain.snake import ALL_STATES, encode_state

parser = argparse.ArgumentParser()
parser.add_argument("--shuffle-seed", type=int, default=None, help="scramble the wiring with this seed; omit for the real wiring")
parser.add_argument("--trials", type=int, default=48)
parser.add_argument("--out", required=True, help="file name inside data/, without .npz")
args = parser.parse_args()

connectome = load_connectome()
channels = build_channels(connectome)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stim_index, readout_index = channels.stim_index.to(device), channels.readout_index.to(device)
state_levels = torch.as_tensor(np.stack([encode_state(s) for s in ALL_STATES]), device=device).T  # [C, 24]
options = {} if args.shuffle_seed is None else {"shuffled": True, "shuffle_seed": args.shuffle_seed}
brain = Brain(connectome, batch=len(ALL_STATES) * 8, **options)
levels = channels.levels(state_levels.repeat_interleave(8, dim=1))
start, runs = time.time(), []
for _ in range(args.trials // 8):
    brain.reset()
    runs.append(brain.run(100, stim_index, levels, readout_index).reshape(len(readout_index), len(ALL_STATES), 8))
bank = torch.cat(runs, dim=2).permute(1, 2, 0).contiguous()  # [24 situations, trials, descending neurons]
np.savez_compressed(DATA / f"{args.out}.npz", counts=bank.cpu().numpy())
print(f"{args.out}: {tuple(bank.shape)} in {time.time() - start:.0f}s ({'real wiring' if args.shuffle_seed is None else f'scramble #{args.shuffle_seed}'})")

"""At full drive, a threat ahead saturates both giant fibers (DNp01 ~380 Hz) and hides which side is also blocked.
Probe: giant-fiber and steering rates for every blocked pattern at several threat intensities, food on the left.

Run: .venv/Scripts/python scripts/threat_intensity_probe.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain.brain import Brain
from flybrain.channels import CHANNEL_NAMES, build_channels
from flybrain.connectome import load_connectome

LEVELS = [1.0, 0.5, 0.3, 0.2, 0.1]
PATTERNS = ["...", "X..", ".X.", "..X", "XX.", ".XX", "X.X", "XXX"]  # blocked left / ahead / right
TRIALS = 4

connectome = load_connectome()
channels = build_channels(connectome)
neurons = connectome.neurons
kind, side = neurons["type"].fillna(""), neurons["side"]
columns, labels = [], []
for level in LEVELS:
    for pattern in PATTERNS:
        drive = dict.fromkeys(CHANNEL_NAMES, 0.0)
        drive["food_L"] = 1.0
        drive["danger_L"], drive["danger_ahead"], drive["danger_R"] = (level * (c == "X") for c in pattern)
        columns += [[drive[name] for name in CHANNEL_NAMES]] * TRIALS
        labels.append((level, pattern))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
levels = channels.levels(torch.as_tensor(np.array(columns, dtype=np.float32).T, device=device))
brain = Brain(connectome, batch=len(columns))
rates = (brain.run(100, channels.stim_index.to(device), levels) * 10).cpu().numpy().reshape(connectome.n, len(labels), TRIALS).mean(axis=2)


def rate(name, which):
    return rates[np.flatnonzero((kind.eq(name) & side.eq(which)).to_numpy())].mean(axis=0)


table = {f"{name}_{s}": rate(name, s) for name in ("DNp01", "DNa02", "DNa01") for s in "LR"}
print("food on the LEFT in every row. Rates in Hz over 100 ms, mean of 4 trials.")
print(f"{'threat':>6s} {'blocked':>8s} | {'DNp01 L':>8s} {'DNp01 R':>8s} {'L-R':>6s} | {'DNa02 L':>8s} {'DNa02 R':>8s} | {'DNa01 L':>8s} {'DNa01 R':>8s}")
for i, (level, pattern) in enumerate(labels):
    print(f"{level:6.1f} {pattern:>8s} | {table['DNp01_L'][i]:8.0f} {table['DNp01_R'][i]:8.0f} {table['DNp01_L'][i] - table['DNp01_R'][i]:6.0f} | "
          f"{table['DNa02_L'][i]:8.0f} {table['DNa02_R'][i]:8.0f} | {table['DNa01_L'][i]:8.0f} {table['DNa01_R'][i]:8.0f}")

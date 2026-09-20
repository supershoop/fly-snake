"""Brain to body: does the steering signal reach the nerve cord's motor neurons, and is it lateralised?
The 708 vnc_motor neurons are simulated but so far unused. 500 ms of each game stimulus, 4 trials.

Run: .venv/Scripts/python scripts/motor_probe.py
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

connectome = load_connectome()
channels = build_channels(connectome)
neurons = connectome.neurons
motor = neurons["superclass"].eq("vnc_motor")
print("vnc motor neurons by side:", neurons[motor].groupby("side", dropna=False).size().to_dict())
print("most common motor types:", neurons[motor]["type"].value_counts().head(12).to_dict())
STIMULI = {"nothing": {}, "food left": {"food_L": 1}, "food right": {"food_R": 1}, "threat left": {"danger_L": 1}, "threat right": {"danger_R": 1},
           "threat ahead": {"danger_ahead": 1}, "food left + threat left": {"food_L": 1, "danger_L": 1}}
TRIALS = 4
levels = np.zeros((len(CHANNEL_NAMES), len(STIMULI) * TRIALS), dtype=np.float32)
for i, stimulus in enumerate(STIMULI.values()):
    for name, value in stimulus.items():
        levels[CHANNEL_NAMES.index(name), i * TRIALS:(i + 1) * TRIALS] = value
brain = Brain(connectome, batch=levels.shape[1])
device = brain.device
rates = (brain.run(500, channels.stim_index.to(device), channels.levels(torch.as_tensor(levels, device=device))) * 2).cpu().numpy()
left, right = np.flatnonzero((motor & neurons["side"].eq("L")).to_numpy()), np.flatnonzero((motor & neurons["side"].eq("R")).to_numpy())
print(f"\n{'stimulus':26s} {'motor L active':>14s} {'motor R active':>14s} {'L mean Hz':>10s} {'R mean Hz':>10s} {'asymmetry':>10s}")
for i, name in enumerate(STIMULI):
    block = rates[:, i * TRIALS:(i + 1) * TRIALS].mean(axis=1)
    l, r = block[left], block[right]
    print(f"{name:26s} {(l > 0).sum():14d} {(r > 0).sum():14d} {l.mean():10.2f} {r.mean():10.2f} {(l.sum() - r.sum()) / (l.sum() + r.sum() + 1e-9):10.2f}")
food_left = rates[:, TRIALS:2 * TRIALS].mean(axis=1)
top = np.argsort(-np.where(motor.to_numpy(), food_left, 0))[:10]
print("\nmost active motor neurons for 'food left':", [(neurons['type'].iloc[i], neurons['side'].iloc[i], round(float(food_left[i]))) for i in top if food_left[i] > 0])

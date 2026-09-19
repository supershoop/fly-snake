"""Does the brain steer by where things are on its retina? Place food, or an obstacle, at different angles around the
fly and read the steering neurons (DNa02, DNa01) and the giant fiber (DNp01).

Run: .venv/Scripts/python scripts/vision_tuning.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain.brain import Brain
from flybrain.connectome import load_connectome
from flybrain.vision import build_retina

connectome = load_connectome()
retina = build_retina(connectome)
neurons = connectome.neurons
kinds = neurons["type"].fillna("").to_numpy()[retina.index.numpy()]
print("retina cells on the horizon band:")
for label in ("LC10", "LC4", "LPLC2"):
    chosen = np.char.startswith(kinds.astype(str), label)
    left = retina.azimuth[chosen] < 0
    histogram = np.histogram(np.abs(retina.azimuth[chosen]), bins=[-15, 0, 30, 60, 90, 120, 150])[0]
    print(f"  {label:6s} {chosen.sum():4d} cells (L {left.sum()}, R {(~left).sum()}); by |azimuth| -15..0..30..60..90..120..150: {histogram.tolist()}")

ANGLES = [-135, -90, -60, -30, -10, 0, 10, 30, 60, 90, 135]
DISTANCES = {"far (4 cells)": 4.0, "near (1.5 cells)": 1.5}
conditions = [(what, name, distance, angle) for what in ("food", "obstacle") for name, distance in DISTANCES.items() for angle in ANGLES]
drive = np.zeros((len(retina.index), len(conditions)), dtype=np.float32)
for b, (what, _, distance, angle) in enumerate(conditions):
    forward, rightward = distance * np.cos(np.radians(angle)), distance * np.sin(np.radians(angle))
    retina.paint(drive[:, b], forward, rightward, retina.is_food if what == "food" else ~retina.is_food, 1.0 if what == "food" else min(1.0, 1.0 / distance))

brain = Brain(connectome, batch=len(conditions))
rates = brain.run(300, retina.index, torch.as_tensor(drive)) / 0.3
side = neurons["side"]


def rate(kind, which_side, b):
    index = np.flatnonzero((neurons["type"].eq(kind) & side.eq(which_side)).to_numpy())
    return rates[torch.as_tensor(index, device=rates.device), b].mean().item()


for what in ("food", "obstacle"):
    for name in DISTANCES:
        print(f"\n{what}, {name}   (negative angle = left of the fly)")
        print(f"  {'angle':>6s} {'cells lit':>9s} | {'DNa02 L':>8s} {'DNa02 R':>8s} | {'DNa01 L':>8s} {'DNa01 R':>8s} | {'DNp01 L':>8s} {'DNp01 R':>8s}")
        for b, (w, n, _, angle) in enumerate(conditions):
            if (w, n) == (what, name):
                print(f"  {angle:6d} {int((drive[:, b] > 0).sum()):9d} | {rate('DNa02', 'L', b):8.0f} {rate('DNa02', 'R', b):8.0f} | "
                      f"{rate('DNa01', 'L', b):8.0f} {rate('DNa01', 'R', b):8.0f} | {rate('DNp01', 'L', b):8.0f} {rate('DNp01', 'R', b):8.0f}")

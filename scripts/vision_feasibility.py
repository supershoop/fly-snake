"""Can the fly see the board through its own eye? Paint patches onto the LEFT eye's column grid (assignedOlHex1/2)
by stimulating columnar optic-lobe neurons, and measure the object detector (LC10), looming detectors (LC4, LPLC2),
steering neurons (DNa02) and giant fiber (DNp01). What we hope for: left-biased responses, LC10 preferring small
patches, LC4/LPLC2 preferring large ones, and different patch positions recruiting different LC10 cells (retinotopy).

Run: .venv/Scripts/python scripts/vision_feasibility.py
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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ol-gain", type=float, default=1.0, help="extra gain on synapses made BY optic-lobe intrinsic neurons")
parser.add_argument("--entries", nargs="*", default=None, help="substring filter on entry names")
args = parser.parse_args()
connectome = load_connectome()
if args.ol_gain != 1.0:
    from_optic_lobe = connectome.neurons["superclass"].eq("ol_intrinsic").to_numpy()[connectome.pre]
    connectome.weight = connectome.weight * np.where(from_optic_lobe, args.ol_gain, 1.0).astype(np.float32)
    print(f"optic-lobe gain x{args.ol_gain} on {from_optic_lobe.mean():.0%} of connections")
neurons = connectome.neurons
kind, side = neurons["type"].fillna(""), neurons["side"]
h1, h2 = neurons["assignedOlHex1"].to_numpy(), neurons["assignedOlHex2"].to_numpy()
left_eye = side.eq("L").to_numpy() & ~np.isnan(h1)

ENTRIES = {  # which columnar cells carry the "pixel"
    "L2 (lamina OFF)": ["L2"], "Mi1 (ON)": ["Mi1"], "Tm1+Tm2+Tm9 (OFF)": ["Tm1", "Tm2", "Tm9"],
    "L1+L2+L3 (lamina)": ["L1", "L2", "L3"], "all Tm/Mi": ["Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm9", "Tm20", "Tm4"],
}
columns = np.unique(np.stack([h1[left_eye], h2[left_eye]], axis=1), axis=0)
order = np.argsort(columns.sum(axis=1))
CENTERS = {"A": columns[order[len(order) // 5]], "B": columns[order[len(order) // 2]], "C": columns[order[4 * len(order) // 5]]}
RADII = {"7 cols": 1, "37 cols": 3, "~200 cols": 8}


def patch(center, radius):
    d1, d2 = h1 - center[0], h2 - center[1]
    return left_eye & (np.maximum.reduce([np.abs(d1), np.abs(d2), np.abs(d1 - d2)]) <= radius)


conditions = {}
if args.entries:
    ENTRIES = {k: v for k, v in ENTRIES.items() if any(e in k for e in args.entries)}
for entry, types in ENTRIES.items():
    cells = kind.isin(types).to_numpy()
    for size, radius in RADII.items():
        for name, center in CENTERS.items():
            conditions[(entry, size, name)] = cells & patch(center, radius)
    conditions[(entry, "whole eye", "-")] = cells & left_eye

stimulated = np.flatnonzero(np.any(list(conditions.values()), axis=0))
level = torch.as_tensor(np.stack([mask[stimulated] for mask in conditions.values()], axis=1), dtype=torch.float32)
brain = Brain(connectome, batch=len(conditions))
rates = brain.run(500, torch.as_tensor(stimulated), level) / 0.5


def group(pattern, which_side):
    return torch.as_tensor(np.flatnonzero(kind.str.fullmatch(pattern).to_numpy() & side.eq(which_side).to_numpy()), device=rates.device)


WATCH = {"LC10": "LC10.*", "LC11": "LC11", "LC4": "LC4", "LPLC2": "LPLC2", "DNa02": "DNa02", "DNp01": "DNp01"}
index = {(name, s): group(pattern, s) for name, pattern in WATCH.items() for s in "LR"}
print(f"{'entry':20s} {'size':10s} {'pos':3s} {'n stim':>6s} {'active':>7s} " + " ".join(f"{name + ' L/R':>13s}" for name in WATCH))
lc10_vectors = {}
for b, (key, mask) in enumerate(conditions.items()):
    column = rates[:, b]
    cells = " ".join(f"{column[index[(name, 'L')]].mean().item():6.1f}/{column[index[(name, 'R')]].mean().item():<6.1f}" for name in WATCH)
    print(f"{key[0]:20s} {key[1]:10s} {key[2]:3s} {mask.sum():6d} {(column > 0).sum().item():7d} {cells}")
    lc10_vectors[key] = column[index[("LC10", "L")]].cpu().numpy()

print("\nRetinotopy: correlation between the LEFT LC10 population's response to patch A vs patch C (low = different cells = retinotopic)")
for entry in ENTRIES:
    for size in RADII:
        a, c = lc10_vectors[(entry, size, "A")], lc10_vectors[(entry, size, "C")]
        if a.std() > 0 and c.std() > 0:
            print(f"  {entry:20s} {size:10s} r = {np.corrcoef(a, c)[0, 1]:5.2f}   LC10 cells active: A {int((a > 0).sum())}, C {int((c > 0).sum())}")
        else:
            print(f"  {entry:20s} {size:10s} LC10 silent for A or C")

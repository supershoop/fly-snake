"""What would the brain do if eating meant tasting sugar and crashing meant pain? 100 ms of each candidate stimulus.

Run: .venv/Scripts/python scripts/event_probe.py
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

connectome = load_connectome()
neurons = connectome.neurons
kind, cls, sup = neurons["type"].fillna(""), neurons["class"].fillna(""), neurons["superclass"].fillna("")
atlas = Path(__file__).resolve().parents[1] / "public/data/brain-atlas"
drawn_ids = np.fromfile(atlas / "ids.bin", dtype="<u4")[np.fromfile(atlas / "groups.bin", dtype="u1") < 3]
drawn = np.isin(neurons["bodyId"].to_numpy(), drawn_ids)

CANDIDATES = {
    "sugar taste (LB3 labellar)": kind.str.startswith("LB3"),
    "sugar taste + taste pegs": kind.str.startswith("LB3") | kind.eq("claw_tpGRN"),
    "bitter taste (LB1)": kind.str.startswith("LB1"),
    "touch (tactile bristles)": cls.eq("mechanosensory_tactile"),
    "antennal mechanosensors (JO)": kind.str.startswith("JO"),
    "heat / humidity sensors": cls.isin(["thermosensory", "hygrosensory"]),
}
masks = {name: mask.to_numpy() for name, mask in CANDIDATES.items()}
stimulated = np.flatnonzero(np.any(list(masks.values()), axis=0))
level = torch.as_tensor(np.stack([mask[stimulated] for mask in masks.values()], axis=1), dtype=torch.float32)
brain = Brain(connectome, batch=len(masks))
rates = (brain.run(100, torch.as_tensor(stimulated), level) * 10).cpu().numpy()


def mean_rate(mask, column):
    index = np.flatnonzero(mask)
    return rates[index, column].mean() if len(index) else 0.0


print(f"{'stimulus':32s} {'cells':>6s} {'active':>7s} {'drawn+active':>13s} {'MN9 feeding':>12s} {'dopamine PAM':>13s} {'PPL1':>6s} {'giant fiber':>12s} {'DNa02':>7s}")
for column, (name, mask) in enumerate(masks.items()):
    active = rates[:, column] > 0
    print(f"{name:32s} {mask.sum():6d} {active.sum():7d} {(active & drawn).sum():13d} {mean_rate(kind.eq('MN9').to_numpy(), column):10.0f} Hz "
          f"{mean_rate(kind.str.startswith('PAM').to_numpy(), column):10.1f} Hz {mean_rate(kind.str.startswith('PPL1').to_numpy(), column):4.1f} "
          f"{mean_rate(kind.eq('DNp01').to_numpy(), column):9.0f} Hz {mean_rate(kind.eq('DNa02').to_numpy(), column):5.0f}")

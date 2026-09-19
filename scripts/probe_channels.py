"""Which sensory groups steer? Stimulate each candidate group on the LEFT only and measure how strongly and how
asymmetrically descending neurons (the brain's output cables) respond.

Run: .venv/Scripts/python scripts/probe_channels.py --scale 0.4
asym = (ipsilateral DN spikes - contralateral DN spikes) / total, for left-side stimulation. |asym| near 0 = useless
for steering; large |asym| = the brain turns this input into a left/right difference.
"""
import argparse
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain.brain import Brain
from flybrain.connectome import load_connectome

parser = argparse.ArgumentParser()
parser.add_argument("--scale", type=float, nargs="+", default=[0.4])
parser.add_argument("--ms", type=float, default=500)
parser.add_argument("--watch", nargs="*", default=["DNa02", "DNa01", "DNp01", "MN9"])
args = parser.parse_args()

connectome = load_connectome()
neurons = connectome.neurons
kind, cls = neurons["type"].fillna(""), neurons["class"].fillna("")
CANDIDATES = {  # name -> boolean mask over neurons (side is applied below)
    "sugar_labellar_LB3": kind.str.startswith("LB3"),
    "bitter_labellar_LB1": kind.str.startswith("LB1"),
    "taste_peg_claw": kind.eq("claw_tpGRN"),
    "leg_taste_LgLG": kind.str.startswith("LgLG"),
    "leg_taste_LgAG": kind.str.startswith("LgAG"),
    "wing_taste_WG": kind.str.startswith("WG"),
    "olfactory_all": cls.eq("olfactory"),
    "mechano_JO": kind.str.startswith("JO"),
    "mechano_tactile": cls.eq("mechanosensory_tactile"),
    "thermo_hygro": cls.isin(["thermosensory", "hygrosensory"]),
    "looming_LPLC2": kind.eq("LPLC2"),
    "looming_LC4": kind.eq("LC4"),
    "looming_LPLC1": kind.eq("LPLC1"),
    "object_LC10": kind.str.startswith("LC10"),
    "object_LC11": kind.eq("LC11"),
    "photoreceptors_R1-6": kind.str.match(r"^R[1-6]$|^R1-6$"),
}
left = neurons["side"].eq("L")
groups = {name: (mask & left).to_numpy() for name, mask in CANDIDATES.items()}
groups = {name: mask for name, mask in groups.items() if mask.any()}
stimulated = torch.as_tensor(neurons.index[[any(m[i] for m in groups.values()) for i in range(len(neurons))]].to_numpy().copy())
level = torch.stack([torch.as_tensor(mask[stimulated.numpy()], dtype=torch.float32) for mask in groups.values()], dim=1)

descending = neurons["superclass"].eq("descending_neuron")
dn_left = torch.as_tensor(neurons.index[descending & left].to_numpy().copy()).cuda()
dn_right = torch.as_tensor(neurons.index[descending & neurons["side"].eq("R")].to_numpy().copy()).cuda()

for scale in args.scale:
    brain = Brain(connectome, batch=len(groups), dt=0.5, weight_scale=scale)
    rates = brain.run(args.ms, stimulated.to(brain.device), level) / (args.ms / 1000)
    print(f"\n=== weight_scale {scale} · {args.ms:.0f} ms · LEFT-side stimulation ===")
    header = f"{'group':22s} {'n':>5s} {'active':>7s} {'DN_L on':>7s} {'DN_R on':>7s} {'DN_L Hz':>8s} {'DN_R Hz':>8s} {'asym':>6s}"
    print(header + "".join(f" {w + ' L/R':>14s}" for w in args.watch))
    for b, (name, mask) in enumerate(groups.items()):
        column = rates[:, b]
        l, r = column[dn_left], column[dn_right]
        asym = ((l.sum() - r.sum()) / (l.sum() + r.sum() + 1e-9)).item()
        line = (f"{name:22s} {mask.sum():5d} {(column > 0).sum().item():7d} {(l > 0).sum().item():7d} {(r > 0).sum().item():7d}"
                f" {l.mean().item():8.2f} {r.mean().item():8.2f} {asym:6.2f}")
        for watched in args.watch:
            rows = neurons[neurons["type"].eq(watched)]
            pair = [column[torch.as_tensor(rows.index[rows["side"].eq(s)].to_numpy().copy()).cuda()].mean().item() for s in "LR"]
            line += f" {pair[0]:6.0f}/{pair[1]:<6.0f} "
        print(line)

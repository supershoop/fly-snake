"""Could learning happen INSIDE the brain (mushroom body) instead of in the readout? Two questions:
1. Do Kenyon cells (KC) respond to our game stimuli, or to a food smell (olfactory sensory neurons)?
2. Does mushroom-body output (MBON) reach the steering neurons DNa02 / DNa01?

Run: .venv/Scripts/python scripts/mushroom_body_check.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch  # no scipy here: scipy.sparse + torch sparse in one process crashes on Windows (duplicate OpenMP runtimes)

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain.brain import Brain
from flybrain.channels import CHANNEL_NAMES, build_channels
from flybrain.connectome import load_connectome

connectome = load_connectome()
channels = build_channels(connectome)
neurons = connectome.neurons
cls, kind, side = neurons["class"].fillna(""), neurons["type"].fillna(""), neurons["side"]


def cells(mask):
    return np.flatnonzero(mask.to_numpy() if hasattr(mask, "to_numpy") else mask)


groups = {"KC": cells(cls.eq("Kenyon_Cell")), "MBON": cells(cls.eq("MBON")), "DAN": cells(cls.eq("DAN")),
          "PAM": cells(kind.str.startswith("PAM")), "PPL1": cells(kind.str.startswith("PPL1"))}
print({name: len(index) for name, index in groups.items()})

game = {name: channels.stim_index.numpy()[channels.stim_matrix[:, i].numpy() > 0] for i, name in enumerate(CHANNEL_NAMES)}
olfactory_left, olfactory_right = cells(cls.eq("olfactory") & side.eq("L")), cells(cls.eq("olfactory") & side.eq("R"))
CONDITIONS = {
    "food_L (LC10 L)": game["food_L"], "danger_L (LC4 L)": game["danger_L"], "danger_ahead (LPLC2)": game["danger_ahead"],
    "smell left": olfactory_left, "smell both": np.concatenate([olfactory_left, olfactory_right]),
    "food_L + smell left": np.concatenate([game["food_L"], olfactory_left]),
    "MBONs left, direct": cells(cls.eq("MBON") & side.eq("L")), "MBONs right, direct": cells(cls.eq("MBON") & side.eq("R")),
}
stimulated = np.unique(np.concatenate(list(CONDITIONS.values())))
level = torch.as_tensor(np.stack([np.isin(stimulated, index) for index in CONDITIONS.values()], axis=1), dtype=torch.float32)
brain = Brain(connectome, batch=len(CONDITIONS))
rates = (brain.run(500, torch.as_tensor(stimulated), level) / 0.5).cpu().numpy()

watch = {f"{name} {s}": cells(kind.eq(name) & side.eq(s)) for name in ("DNa02", "DNa01") for s in "LR"}
print(f"\n{'condition':24s} {'KC active':>10s} {'KC mean Hz':>10s} {'MBON active':>11s} {'MBON Hz':>8s} {'DAN active':>10s} " + " ".join(f"{w:>8s}" for w in watch))
for b, name in enumerate(CONDITIONS):
    column = rates[:, b]
    print(f"{name:24s} {(column[groups['KC']] > 0).sum():10d} {column[groups['KC']].mean():10.2f} {(column[groups['MBON']] > 0).sum():11d} "
          f"{column[groups['MBON']].mean():8.2f} {(column[groups['DAN']] > 0).sum():10d} " + " ".join(f"{column[index].mean():8.1f}" for index in watch.values()))

pre, post, weight = connectome.pre, connectome.post, connectome.weight
from_mbon = np.isin(pre, groups["MBON"])
print("\nWiring from MBONs to steering neurons (signed synapse counts):")
for target in ("DNa02", "DNa01"):
    to_target = np.isin(post, cells(kind.eq(target)))
    direct = weight[from_mbon & to_target].sum()
    into = np.bincount(post[from_mbon], np.abs(weight[from_mbon]), minlength=connectome.n)  # synapses received from MBONs
    out = np.bincount(pre[to_target], weight[to_target], minlength=connectome.n)            # signed synapses given to target
    relay = pd.DataFrame({"type": kind, "path": into * out})
    top = relay[relay.path != 0].groupby("type")["path"].sum().sort_values(key=abs, ascending=False).head(5)
    print(f"  MBON -> {target}: direct {direct:.0f}; strongest 2-hop relays: " + ", ".join(f"{t} ({v:,.0f})" for t, v in top.items()))
plastic = np.isin(pre, groups["KC"]) & np.isin(post, groups["MBON"])
print(f"\nPlastic site: {plastic.sum():,} KC->MBON connections carrying {int(np.abs(weight[plastic]).sum()):,} synapses")

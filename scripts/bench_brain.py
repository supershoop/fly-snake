"""Sanity checks and speed of the LIF brain. Run: .venv/Scripts/python scripts/bench_brain.py [--dt 0.5] [--batch 1]"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain.brain import Brain
from flybrain.connectome import load_connectome

parser = argparse.ArgumentParser()
parser.add_argument("--dt", type=float, default=0.5)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--ms", type=float, default=500)
parser.add_argument("--stim-type", default=None, help="annotation `type` to stimulate, e.g. a sugar GRN type")
parser.add_argument("--watch", nargs="*", default=["MN9"], help="annotation types whose firing rate to report")
args = parser.parse_args()

start = time.time()
connectome = load_connectome()
print(f"{connectome.n:,} neurons, {len(connectome.weight):,} connections (>= 5 synapses), "
      f"{(connectome.weight < 0).mean():.0%} inhibitory, loaded in {time.time() - start:.1f}s")
if args.stim_type is None:
    gustatory = connectome.select(**{"class": "gustatory"})
    print("\ngustatory types (count):\n", gustatory.groupby(["type", "side"], dropna=False).size().unstack(fill_value=0).to_string())
    print("\nbrain motor types:\n", connectome.select(superclass="cb_motor")["type"].value_counts().to_string())

brain = Brain(connectome, batch=args.batch, dt=args.dt)
seconds = args.ms / 1000


def report(label, counts):
    rates = counts[:, 0] / seconds
    print(f"\n[{label}] active neurons: {(rates > 0).sum().item():,}  mean rate {rates.mean().item():.3f} Hz  max {rates.max().item():.0f} Hz")
    for watched in args.watch:
        rows = connectome.select(type=watched)
        for side, group in rows.groupby("side", dropna=False):
            print(f"   {watched} {side}: {rates[torch.as_tensor(group.index.to_numpy(), device=rates.device)].mean().item():.1f} Hz (n={len(group)})")


torch.cuda.synchronize() if brain.device.type == "cuda" else None
start = time.time()
report("rest", brain.run(args.ms))
torch.cuda.synchronize() if brain.device.type == "cuda" else None
elapsed = time.time() - start
print(f"\nspeed: {args.ms:.0f} ms of brain x{args.batch} in {elapsed:.2f}s wall = {elapsed / seconds:.2f}x real time (dt={args.dt} ms, {brain.device})")

if args.stim_type:
    brain.reset()
    stimulated = connectome.select(type=args.stim_type)
    index = torch.as_tensor(stimulated.index.to_numpy(), device=brain.device)
    print(f"\nstimulating {len(index)} x {args.stim_type} at full drive")
    report(f"stim {args.stim_type}", brain.run(args.ms, index, torch.ones(len(index), args.batch)))

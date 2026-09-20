"""Compare compiled and original kernels on the real whole connectome, including learned strengths."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain.brain import Brain
from flybrain.channels import build_channels
from flybrain.connectome import load_connectome
from flybrain.synaptic import SynapticAdapter, SynapticSites


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("outputs/brain-kernel-verification.json"))
    parser.add_argument("--direct-board", action="store_true", help="Verify full-board inputs and their plastic pathways")
    args = parser.parse_args()
    torch.set_num_threads(2)
    c = load_connectome()
    channels, sites = build_channels(c), SynapticSites.from_connectome(c)
    board_encoder = None
    if args.direct_board:
        from flybrain.board_encoder import BoardEncoder, board_sites
        from flybrain.snake import Snake
        board_encoder = BoardEncoder.from_connectome(c)
        sites = board_sites(c, board_encoder)
    brains = [Brain(c, device="cpu", seed=819, compiled=compiled) for compiled in (False, True)]
    adapters = [SynapticAdapter(brain, sites) for brain in brains]
    # Warm the compiler outside the timing measurement, then restore states/RNG.
    brains[1].run(.5)
    for brain in brains:
        brain.reset()
        brain.rng.manual_seed(819)
    rng = np.random.default_rng(914)
    times = [0., 0.]
    for move in range(24):
        if move in (8, 16):
            parameters = rng.uniform(-.7, .7, len(sites.labels)) if move == 8 else np.zeros(len(sites.labels))
            for adapter in adapters:
                adapter.apply(parameters)
        if board_encoder is None:
            stimulus = channels.stim_index
            levels = channels.levels(torch.tensor(rng.integers(0, 2, (5, 1)), dtype=torch.float32))
        else:
            stimulus = torch.as_tensor(board_encoder.stimulus)
            levels = torch.as_tensor(board_encoder.encode(Snake(seed=4000000 + move)))[:, None]
        counts = []
        for i, brain in enumerate(brains):
            started = time.perf_counter()
            counts.append(brain.run(100., stimulus, levels))
            times[i] += time.perf_counter() - started
        torch.testing.assert_close(counts[0], counts[1], rtol=0, atol=0)
        for name in ("v", "g", "refractory", "pending"):
            torch.testing.assert_close(getattr(brains[0], name), getattr(brains[1], name), rtol=1e-5, atol=1e-5)
    report = {"passed": True, "moves": 24, "neurons": c.n, "connections": len(c.pre),
              "comparison": "All neuron spike counts exactly equal; full state within 1e-5; includes changed and restored synapses",
              "original_seconds": times[0], "compiled_seconds": times[1], "warm_speedup": times[0] / times[1],
              "connectome_sha256": sites.fingerprint}
    if board_encoder is not None:
        report["encoder"] = board_encoder.metadata()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

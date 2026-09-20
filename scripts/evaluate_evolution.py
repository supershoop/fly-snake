"""Paired continuous-brain check, with the SAME sensing for both readouts.

The brain is reset only between games, never between moves. This is slower
than bank estimates; unfinished games at --max-moves are explicitly censored.
"""
import argparse
import hashlib
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
from flybrain.readout import Policy
from flybrain.snake import Snake


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/readout-evolved.npz"))
    parser.add_argument("--baseline", type=Path, default=Path("models/readout-real.npz"))
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--max-moves", type=int, default=250)
    parser.add_argument("--game-seed", type=int, default=70000)
    parser.add_argument("--brain-seed", type=int, default=1729)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--report", type=Path, default=Path("models/readout-evolved-live.json"))
    args = parser.parse_args()
    if min(args.games, args.max_moves, args.threads) < 1:
        parser.error("Games, max-moves and threads must be positive")
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    connectome = load_connectome()
    channels = build_channels(connectome)
    brain = Brain(connectome, batch=1, device=device)
    stim, readout = channels.stim_index.to(device), channels.readout_index.to(device)
    report = {"evaluation": "continuous brain simulation; paired seeds and identical tail-escape sensing",
              "device": device, "max_moves": args.max_moves, "games_per_model": args.games,
              "complete": False, "results": {}, "model_hashes": {},
              "limitations": "Small censored smoke comparison, not a full-game performance guarantee."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for label, path in (("candidate", args.model), ("baseline", args.baseline)):
        with np.load(path, allow_pickle=False) as saved:
            policy = Policy(torch.as_tensor(saved["weight"], device=device), torch.as_tensor(saved["bias"], device=device))
        report["model_hashes"][label] = hashlib.sha256(path.read_bytes()).hexdigest()
        records = []
        for index in range(args.games):
            seed, noise = args.game_seed + index, args.brain_seed + index
            game = Snake(seed=seed)
            brain.reset()
            brain.rng.manual_seed(noise)
            for move in range(args.max_moves):
                levels = channels.levels(torch.as_tensor(game.encode(), device=device)[:, None])
                counts = brain.run(100., stim, levels, readout)
                game.step(int(policy.act(counts)[0][0]))
                if not game.alive:
                    break
                if (move + 1) % 50 == 0:
                    print(f"{label} seed {seed}, move {move + 1}: food {game.score}, {time.perf_counter()-started:.1f}s", flush=True)
            records.append({"game_seed": seed, "brain_seed": noise, "food": game.score, "moves": move + 1,
                            "end_reason": game.snakes[0].end_reason, "alive_at_limit": game.alive})
            report["results"][label] = {"games": records, "mean_score": float(np.mean([r["food"] for r in records])),
                                        "collisions": sum(r["end_reason"] == "collision" for r in records),
                                        "starved": sum(r["end_reason"] == "starvation" for r in records),
                                        "alive_at_limit": sum(r["alive_at_limit"] for r in records)}
            report["elapsed_seconds"] = time.perf_counter() - started
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"{label} seed {seed} finished: {records[-1]}", flush=True)
    report["complete"] = True
    report["elapsed_seconds"] = time.perf_counter() - started
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

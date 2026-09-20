"""Compare the saved survival readout with the original in a continuous live brain.

Example: .venv/Scripts/python scripts/evaluate_survival.py --games 4 --max-moves 400
This performs fresh simulation. It does not sample the response bank or reset
the brain between moves. Alive-at-limit games are unfinished, not wins.
"""
import argparse
import json
from pathlib import Path

import torch

from train_readout import play_live
from flybrain.channels import build_channels
from flybrain.connectome import load_connectome
from flybrain.readout import Policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--max-moves", type=int, default=400)
    parser.add_argument("--game-seed", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--model", default="readout-real")
    parser.add_argument("--baseline", default="readout-real-legacy")
    parser.add_argument("--report", type=Path, default=Path("outputs/live-survival-evaluation.json"))
    args = parser.parse_args()
    if min(args.games, args.max_moves, args.threads) < 1:
        parser.error("games, max-moves and threads must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    connectome = load_connectome()
    channels = build_channels(connectome)
    seeds = list(range(args.game_seed, args.game_seed + args.games))
    results = play_live(connectome, channels, [(Policy.load(args.baseline, device), False),
                                             (Policy.load(args.model, device), True)],
                        seeds, args.max_moves, 100., device, seed=args.seed)
    report = {"evaluation": "continuous brain simulation", "game_seeds": seeds, "brain_seed": args.seed,
              "max_moves": args.max_moves, "baseline": args.baseline, "model": args.model,
              "baseline_result": results[0], "candidate_result": results[1]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

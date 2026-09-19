"""Test a frozen evolutionary winner on fresh neural responses and game seeds.

This estimates performance by sampling a response bank; it is NOT live-brain
simulation. Run evaluate_evolution.py for the independent continuous check.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_evolution import digest, load_bank, policy_actions, write_json
from flybrain.evolution_game import score_actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path("models/readout-real.npz"))
    parser.add_argument("--bank", type=Path, default=Path("data/bank-real-evolution-test.npz"))
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--max-moves", type=int, default=1600)
    parser.add_argument("--game-seed", type=int, default=81000)
    parser.add_argument("--response-seed", type=int, default=180071)
    parser.add_argument("--report", type=Path, default=Path("models/readout-evolved-test.json"))
    args = parser.parse_args()
    if min(args.games, args.max_moves) < 1 or args.game_seed < 80000:
        parser.error("Positive game/move limits and fresh game seeds >= 80000 required")
    training = json.loads(args.training_report.read_text())
    if digest(args.model) != training["model_sha256"]:
        parser.error("Model differs from the frozen training winner")
    if digest(args.bank) == training["config"]["bank_sha256"]:
        parser.error("Use a fresh response bank, separate from training and validation")
    if digest(args.baseline) != training["config"]["baseline_sha256"]:
        parser.error("The comparison baseline has changed since training")
    torch.set_num_threads(1)
    bank = load_bank(args.bank)
    with np.load(args.bank, allow_pickle=False) as saved:
        neural_seed = int(saved["seed"])
    with np.load(Path(training["config"]["bank"]), allow_pickle=False) as saved:
        if neural_seed == int(saved["seed"]):
            parser.error("Final response bank must use a different neural seed")
    seeds = list(range(args.game_seed, args.game_seed + args.games))
    results, hashes = {}, {}
    for label, path in (("initial_random", args.initial), ("baseline", args.baseline), ("candidate", args.model)):
        results[label] = score_actions(policy_actions(path, bank), seeds, args.max_moves, response_seed=args.response_seed)
        hashes[label] = digest(path)
    delta = np.asarray(results["candidate"]["scores"]) - results["baseline"]["scores"]
    bootstrap = np.random.default_rng(901).choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
    report = {"evaluation": "fresh sampled response-bank estimate; NOT continuous-brain performance",
              "encoder": "tail-escape-v1 for all models", "game_seeds": seeds, "max_moves": args.max_moves,
              "response_seed": args.response_seed, "neural_bank_seed": neural_seed,
              "response_bank_sha256": digest(args.bank), "response_bank_shape": list(bank.shape),
              "model_hashes": hashes, "results": results,
              "paired_mean_food_improvement": float(delta.mean()),
              "paired_bootstrap_95_interval": np.quantile(bootstrap, [.025, .975]).tolist(),
              "saved_as_default": False,
              "limitations": ["One training run and one fresh neural response bank.",
                              "CI covers variation across these game seeds, not model-training or neural-bank uncertainty.",
                              "Random bank draws cannot recreate continuous action-dependent neural history.",
                              "Games still alive at the move cap are censored, not wins."]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

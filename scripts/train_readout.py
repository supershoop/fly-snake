"""Train survival targets on long games, fit a DN readout, and compare with the old model.

CPU example (bank estimates, not live-brain scores):
  .venv/Scripts/python scripts/train_readout.py --trials 32 --evaluation bank
GPU / continuous-brain evaluation:
  .venv/Scripts/python scripts/train_readout.py --evaluation live

Only the linear readout is fitted. Connectome weights never change. The spatial
encoder reports threatened escape routes; no action filter overrides the brain.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain.brain import Brain
from flybrain.channels import build_channels
from flybrain.connectome import DATA, load_connectome
from flybrain.readout import MODELS, Policy, fit
from flybrain.response_bank import response_bank
from flybrain.snake import Snake
from flybrain.training import optimize_targets, score_targets


def play_live(connectome, channels, policies, seeds, max_moves, window, device, *, shuffled=False, seed=77):
    """Compare (policy, lookahead) configurations in one continuously running batch.

    Games that are alive at the move limit are censored, not counted as wins.
    Every configuration plays the same game seeds. Brain noise is independent.
    """
    games = [Snake(seed=int(s)) for _ in policies for s in seeds]
    brain = Brain(connectome, batch=len(games), shuffled=shuffled, seed=0, device=str(device))
    brain.rng.manual_seed(seed)
    width = len(seeds)
    steps = np.zeros(len(games), dtype=int)
    for move in range(max_moves):
        inputs = np.stack([game.encode(lookahead=policies[i // width][1]) for i, game in enumerate(games)])
        counts = brain.run(window, channels.stim_index.to(device),
                           channels.levels(torch.as_tensor(inputs, device=device).T), channels.readout_index.to(device))
        actions = torch.cat([policy.act(counts[:, i*width:(i+1)*width])[0]
                             for i, (policy, _) in enumerate(policies)]).tolist()
        for i, (game, action) in enumerate(zip(games, actions)):
            if game.alive:
                steps[i] += 1
                game.step(action)
        if (move + 1) % 25 == 0:
            print(f"live move {move+1}: scores {[g.score for g in games]}; alive {sum(g.alive for g in games)}", flush=True)
        if not any(game.alive for game in games):
            break
    results = []
    for i in range(len(policies)):
        group = games[i*width:(i+1)*width]
        scores = [game.score for game in group]
        results.append({"mean_score": float(np.mean(scores)), "scores": scores,
                        "collisions": sum(g.snakes[0].end_reason == "collision" for g in group),
                        "starved": sum(g.snakes[0].end_reason == "starvation" for g in group),
                        "alive_at_limit": sum(g.alive for g in group),
                        "mean_moves": float(steps[i*width:(i+1)*width].mean())})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shuffled", action="store_true")
    parser.add_argument("--trials", type=int, default=48)
    parser.add_argument("--window", type=float, default=100.)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--max-moves", type=int, default=1600)
    parser.add_argument("--train-games", type=int, default=12)
    parser.add_argument("--target-passes", type=int, default=2)
    parser.add_argument("--train-seed", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42, help="brain-response generation seed")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--threads", type=int, default=8, help="CPU torch threads")
    parser.add_argument("--evaluation", choices=("bank", "live"), default="live")
    parser.add_argument("--baseline", help="model name to compare (uses legacy adjacent-cell sensing)")
    parser.add_argument("--output", help="model name; defaults to readout-real / readout-shuffled")
    parser.add_argument("--report", type=Path, default=Path("outputs/training-report.json"))
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if min(args.games, args.train_games, args.max_moves, args.epochs, args.threads) < 1 or args.trials < 8 or args.target_passes < 0:
        parser.error("games, train-games, moves, epochs and threads must be positive; trials >= 8; passes >= 0")
    if args.window != 100:
        parser.error("The live server uses 100 ms windows; train its readout with --window 100")
    train_seeds = list(range(args.train_seed, args.train_seed + args.train_games))
    eval_seeds = list(range(args.eval_seed, args.eval_seed + args.games))
    if set(train_seeds) & set(eval_seeds):
        parser.error("Training and evaluation game seeds must not overlap")
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = "shuffled" if args.shuffled else "real"
    output = args.output or f"readout-{name}"
    baseline_name = args.baseline or ("readout-shuffled" if args.shuffled else "readout-real-legacy")
    baseline = Policy.load(baseline_name, device)
    connectome = load_connectome()
    channels = build_channels(connectome)
    bank = response_bank(connectome, channels, DATA / f"bank-{name}-transition.npz", trials=args.trials,
                         window=args.window, shuffled=args.shuffled, seed=args.seed, device=device,
                         rebuild=args.rebuild, progress=lambda message: print(message, flush=True))
    targets, training = optimize_targets(train_seeds, args.max_moves, args.target_passes,
                                        progress=lambda message: print(message, flush=True))
    split = args.trials * 3 // 4
    labels = torch.as_tensor(targets, device=device)
    policy = fit(bank[:, :split].reshape(-1, bank.shape[-1]), labels.repeat_interleave(split),
                 epochs=args.epochs, weight_decay=1e-4)
    held = bank[:, split:]
    predicted = policy.act(held.reshape(-1, bank.shape[-1]).T)[0].reshape(len(targets), -1)
    accuracy = float((predicted == labels[:, None]).float().mean())
    print(f"Held-out DN-response target accuracy: {accuracy:.1%}", flush=True)
    if args.evaluation == "bank":
        old_actions = baseline.act(held.reshape(-1, bank.shape[-1]).T)[0].reshape(len(targets), -1).cpu().numpy()
        results = [score_targets(actions, eval_seeds, args.max_moves, lookahead=lookahead, response_seed=99)
                   for actions, lookahead in ((old_actions, False), (old_actions, True), (predicted.cpu().numpy(), True))]
        evaluation_kind = "sampled response-bank estimate; not continuous-brain performance"
    else:
        results = play_live(connectome, channels, [(baseline, False), (policy, True)], eval_seeds,
                            args.max_moves, args.window, device, shuffled=args.shuffled)
        evaluation_kind = "continuous brain simulation"
    report = {"evaluation": evaluation_kind, "model": output, "baseline": baseline_name, "encoder": "tail-escape-v1",
              "environment": {"torch": str(torch.__version__), "numpy": np.__version__, "device": str(device)},
              "response_bank_sha256": hashlib.sha256((DATA / f"bank-{name}-transition.npz").read_bytes()).hexdigest(),
              "baseline_sha256": hashlib.sha256((MODELS / f"{baseline_name}.npz").read_bytes()).hexdigest(),
              "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "train_seeds": train_seeds, "eval_seeds": eval_seeds, "targets": targets.tolist(),
              "training": training, "held_out_response_accuracy": accuracy,
              "baseline_result": results[0], "candidate_result": results[-1]}
    if args.evaluation == "bank":
        report["sensing_only_result"] = results[1]
    # Only save candidates that improve the explicitly reported baseline.
    accepted = (results[-1]["mean_score"] > results[0]["mean_score"]
                and results[-1]["collisions"] <= results[0]["collisions"])
    report["saved"] = accepted
    if accepted:
        policy.save(output)
        report["model_sha256"] = hashlib.sha256((MODELS / f"{output}.npz").read_bytes()).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not accepted:
        raise SystemExit("Candidate did not improve the baseline; existing model preserved.")
    print(f"Saved {MODELS / (output + '.npz')}", flush=True)


if __name__ == "__main__":
    main()

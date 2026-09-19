"""Evolve the DN readout from game rewards; all scores here are bank estimates.

python scripts/train_evolution.py --generations 2000 --run outputs/evolution-2000
python scripts/train_evolution.py --generations 2000 --run outputs/evolution-2000 --resume
Requires the transition bank from train_readout.py and optional numba dependency.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain.brain import MALECNS_WEIGHT_SCALE, SHUFFLE_SEED
from flybrain.connectome import load_neurons
from flybrain.evolution import Evaluator, actions_for, breed, feature_basis, features, fitness, weights_for
from flybrain.evolution_game import score_actions
from flybrain.readout import Policy
from flybrain.response_bank import BANK_VERSION


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_policy(path, parameters, basis):
    weight, bias = weights_for(parameters, basis)
    np.savez(path, weight=weight, bias=bias)


def load_bank(path):
    with np.load(path, allow_pickle=False) as saved:
        expected = dict(version=BANK_VERSION, window=100., dt=.5, shuffled=False,
                        weight_scale=MALECNS_WEIGHT_SCALE, wiring_seed=SHUFFLE_SEED)
        if any(key not in saved or saved[key].item() != value for key, value in expected.items()):
            raise ValueError("Incompatible transition bank; regenerate with train_readout.py")
        counts, ids = saved["counts"], saved["readout_body_ids"]
    neurons = load_neurons()
    expected_ids = neurons.loc[neurons.superclass.eq("descending_neuron"), "bodyId"].to_numpy()
    if not np.array_equal(ids, expected_ids):
        raise ValueError("Response-bank neuron IDs/order differ from the current connectome")
    if (counts.ndim != 3 or counts.shape[0] != 24 or counts.shape[1] < 16
            or counts.shape[1] % 8 or counts.shape[2] != len(ids)
            or not np.isfinite(counts).all() or (counts < 0).any()):
        raise ValueError("Need finite nonnegative counts [24, trials, neurons], trials >= 16 divisible by 8")
    return counts


def policy_actions(path, counts):
    with np.load(path, allow_pickle=False) as model:
        policy = Policy(torch.as_tensor(model["weight"]), torch.as_tensor(model["bias"]))
    flat = torch.as_tensor(counts.reshape(-1, counts.shape[-1]))
    return policy.act(flat.T)[0].reshape(counts.shape[:2]).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=2000)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--elites", type=int, default=4)
    parser.add_argument("--train-games", type=int, default=8)
    parser.add_argument("--validation-games", type=int, default=32)
    parser.add_argument("--test-games", type=int, default=128)
    parser.add_argument("--max-moves", type=int, default=600)
    parser.add_argument("--evaluation-moves", type=int, default=1600)
    parser.add_argument("--interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=100., help="Regularize mutation directions to reduce noise overfitting")
    parser.add_argument("--validation-only", action="store_true", help="Tune on validation without opening the final test partition")
    parser.add_argument("--bank", type=Path, default=Path("data/bank-real-transition.npz"))
    parser.add_argument("--baseline", type=Path, default=Path("models/readout-real.npz"))
    parser.add_argument("--output", type=Path, default=Path("models/readout-evolved.npz"))
    parser.add_argument("--run", type=Path, default=Path("outputs/evolution-2000"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.generations, args.train_games, args.validation_games, args.test_games, args.max_moves,
           args.evaluation_moves, args.interval, args.workers, args.elites) < 1 or args.population <= args.elites:
        parser.error("Positive limits and population > elites are required")
    if not np.isfinite(args.ridge) or args.ridge <= 0:
        parser.error("Ridge must be finite and positive")
    # Fixed disjoint seed ranges; reject requests that would cross partitions.
    if ((args.generations + args.interval - 1) // args.interval * args.train_games > 10000
            or args.validation_games > 10000 or args.test_games > 10000):
        parser.error("Requested games would overlap the reserved seed partitions")
    if args.output.resolve() == args.baseline.resolve():
        parser.error("Output must be a separate candidate; preserve the baseline")
    if args.run.exists() and not args.resume:
        parser.error("Run directory exists; choose a new --run or use --resume")
    if args.resume and (args.run / "report.json").exists():
        parser.error("This run has already used its final test set; start a separate experiment")
    torch.set_num_threads(1)
    started = time.perf_counter()
    counts = load_bank(args.bank)
    split = counts.shape[1] // 2
    # Boundaries fall at the bank's reset-every-four-trials boundaries.
    train, validation, test = counts[:, :split], counts[:, split:split + split // 2], counts[:, split + split // 2:]
    basis = feature_basis(train, args.ridge)
    train_features, validation_features = features(train, basis), features(validation, basis)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
              if key not in ("resume", "generations")}
    config.update(bank_sha256=digest(args.bank), baseline_sha256=digest(args.baseline),
                  encoder="tail-escape-v1", method="elitist mutation-only genetic algorithm, random initialization",
                  train_trials=[0, split], validation_trials=[split, split + split // 2],
                  test_trials=[split + split // 2, counts.shape[1]])
    rng = np.random.default_rng(args.seed)
    population = rng.normal(0, 1, (args.population, 3, basis.shape[1]))
    best_parameters, best_validation = population[0].copy(), None
    begin, previous_seconds, prior_games, prior_evaluations, prior_hits = 1, 0., 0, 0, 0
    history = []
    args.run.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if json.loads((args.run / "config.json").read_text()) != config:
            parser.error("Resume configuration or input hashes changed")
        with np.load(args.run / "state.npz", allow_pickle=False) as saved:
            population, best_parameters = saved["population"], saved["best_parameters"]
            state = json.loads(str(saved["state"]))
        rng.bit_generator.state = state["rng"]
        best_validation, begin = state["best_validation"], state["generation"] + 1
        previous_seconds = state["seconds"]
        prior_games, prior_evaluations, prior_hits = state["games"], state["evaluations"], state["cache_hits"]
        history = [row for row in json.loads((args.run / "history.json").read_text()) if row["generation"] < begin]
        if args.generations < begin:
            parser.error("Generations must extend the saved checkpoint")
    else:
        write_json(args.run / "config.json", config)
        save_policy(args.run / "initial.npz", population[0], basis)
    # Compile before using threads; each rollout receives its own RNG objects.
    score_actions(np.ones((24, train.shape[1]), dtype=np.int64), [40000], 1)
    evaluator = Evaluator(args.workers)
    validation_seeds = list(range(50000, 50000 + args.validation_games))
    try:
        for generation in range(begin, args.generations + 1):
            first_seed = 40000 + (generation - 1) // args.interval * args.train_games
            train_seeds = list(range(first_seed, first_seed + args.train_games))
            actions = [actions_for(p, train_features) for p in population]
            results = evaluator.evaluate(actions, train_seeds, args.max_moves, args.seed)
            order = sorted(range(len(results)), key=lambda i: fitness(results[i]), reverse=True)
            # Keep diverse behaviors as elites when available.
            selected, seen = [], set()
            for i in order:
                key = actions[i].tobytes()
                if key not in seen:
                    selected.append(i)
                    seen.add(key)
                if len(selected) == args.elites:
                    break
            for i in order:
                if len(selected) == args.elites:
                    break
                if i not in selected:
                    selected.append(i)
            winner = order[0]
            row = {"generation": generation, "train_mean_food": results[winner]["mean_score"],
                   "train_collisions": results[winner]["collisions"], "train_seed_start": first_seed}
            checkpoint = generation == 1 or generation % args.interval == 0 or generation == args.generations
            if checkpoint:
                for i in selected:
                    validation_result = score_actions(actions_for(population[i], validation_features), validation_seeds,
                                                      args.evaluation_moves, response_seed=args.seed + 1)
                    if best_validation is None or fitness(validation_result) > fitness(best_validation):
                        best_validation, best_parameters = validation_result, population[i].copy()
                        save_policy(args.run / f"winner-{generation:06d}.npz", best_parameters, basis)
                        row["new_validation_winner"] = True
                row["best_validation_mean_food"] = best_validation["mean_score"]
            history.append(row)
            population = breed(population[selected], args.population, rng)
            if checkpoint:
                seconds = previous_seconds + time.perf_counter() - started
                state = {"generation": generation, "rng": rng.bit_generator.state, "best_validation": best_validation,
                         "seconds": seconds, "games": prior_games + evaluator.games,
                         "evaluations": prior_evaluations + evaluator.evaluations,
                         "cache_hits": prior_hits + evaluator.cache_hits}
                write_json(args.run / "history.json", history)
                temporary = args.run / "state.tmp.npz"
                np.savez(temporary, population=population, best_parameters=best_parameters, state=json.dumps(state))
                os.replace(temporary, args.run / "state.npz")
                print(f"generation {generation}/{args.generations}: train {row['train_mean_food']:.2f}, "
                      f"best validation {best_validation['mean_score']:.2f}, "
                      f"{state['games']:,} actual training games, {seconds:.1f}s", flush=True)
    finally:
        evaluator.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_policy(args.output, best_parameters, basis)
    if args.validation_only:
        report = {"evaluation": "validation only; sampled response-bank estimate; test partition unused",
                  "config": config, "generations_completed": args.generations,
                  "candidate_evaluations_requested": args.generations * args.population,
                  "actual_training_evaluations": prior_evaluations + evaluator.evaluations,
                  "actual_training_games": prior_games + evaluator.games,
                  "cached_evaluations": prior_hits + evaluator.cache_hits,
                  "elapsed_seconds": previous_seconds + time.perf_counter() - started,
                  "validation_result": best_validation, "model_sha256": digest(args.output), "saved_as_default": False}
        write_json(args.run / "validation.json", report)
        write_json(args.output.with_suffix(".json"), report)
        print(json.dumps({"output": str(args.output), "validation_mean_food": best_validation["mean_score"],
                          "seconds": report["elapsed_seconds"]}), flush=True)
        return
    test_seeds = list(range(60000, 60000 + args.test_games))
    # Test the exported float32 Policy, rather than the optimizer's float64 basis.
    candidate = score_actions(policy_actions(args.output, test), test_seeds, args.evaluation_moves, response_seed=args.seed + 2)
    baseline = score_actions(policy_actions(args.baseline, test), test_seeds, args.evaluation_moves, response_seed=args.seed + 2)
    initial = score_actions(policy_actions(args.run / "initial.npz", test), test_seeds, args.evaluation_moves, response_seed=args.seed + 2)
    differences = np.asarray(candidate["scores"]) - baseline["scores"]
    bootstrap = np.random.default_rng(901).choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
    report = {"evaluation": "sampled response-bank estimate; NOT continuous-brain performance", "config": config,
              "generations_completed": args.generations, "candidate_evaluations_requested": args.generations * args.population,
              "actual_training_evaluations": prior_evaluations + evaluator.evaluations,
              "actual_training_games": prior_games + evaluator.games, "cached_evaluations": prior_hits + evaluator.cache_hits,
              "elapsed_seconds": previous_seconds + time.perf_counter() - started,
              "environment": {"numpy": np.__version__, "torch": str(torch.__version__), "device": "cpu"},
              "validation_result": best_validation, "test_game_seeds": test_seeds,
              "initial_random_result": initial, "baseline_result": baseline, "candidate_result": candidate,
              "paired_food_improvement": float(differences.mean()),
              "paired_bootstrap_95_interval": np.quantile(bootstrap, [.025, .975]).tolist(),
              "model_sha256": digest(args.output), "saved_as_default": False,
              "limitations": ["Only one training seed/run; not a population-level algorithm comparison.",
                              "Bank draws do not reproduce continuous, action-dependent neural history.",
                              "The engineered encoder supplies only 24 distinct situations.",
                              "Alive-at-limit games are censored, not wins."]}
    write_json(args.run / "report.json", report)
    write_json(args.output.with_suffix(".json"), report)
    print(json.dumps({"output": str(args.output), "report": str(args.output.with_suffix('.json')),
                      "test_mean_food": candidate["mean_score"], "baseline": baseline["mean_score"],
                      "initial_random": initial["mean_score"], "seconds": report["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

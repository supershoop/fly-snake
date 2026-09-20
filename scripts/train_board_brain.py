"""CMA-ES training of internal fly synapses from full-board Snake input.

Starts from the original connectome; input mapping and motor decoder are fixed.
Every action runs 100 ms of the whole continuous brain. No teacher or bank.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import cma
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain.board_experiment import BoardRunner, objective
from flybrain.synaptic import LIMIT


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def optimizer(dimension, config):
    rng = np.random.default_rng(config["seed"])
    return cma.CMAEvolutionStrategy(np.zeros(dimension), config["sigma"], {
        "bounds": [-LIMIT, LIMIT], "popsize": config["population"], "CMA_diagonal": True,
        "randn": lambda *shape: rng.standard_normal(shape), "seed": config["seed"], "verbose": -9})


def evaluate_population(runners, candidates, seeds, limit, generation):
    """Independent CPU brains; no shared mutable neural state or noise RNG."""
    def chunk(worker):
        results = []
        for candidate in range(worker, len(candidates), len(runners)):
            result = runners[worker].evaluate(candidates[candidate], seeds, limit)
            results.append((candidate, result))
            print(f"generation {generation} candidate {candidate + 1}/{len(candidates)}: "
                  f"food {result['mean_food']:.2f}, moves {result['mean_moves']:.1f}", flush=True)
        return results
    with ThreadPoolExecutor(max_workers=len(runners)) as executor:
        records = [record for group in executor.map(chunk, range(len(runners))) for record in group]
    return [result for _, result in sorted(records)]


def bootstrap_difference(original, trained):
    difference = np.array([b["food"] - a["food"] for a, b in zip(original["games"], trained["games"])])
    rng = np.random.default_rng(91741)
    means = rng.choice(difference, size=(20000, len(difference)), replace=True).mean(axis=1)
    return {"mean_food_difference": float(difference.mean()),
            "paired_bootstrap_95_percent": np.quantile(means, [.025, .975]).tolist(),
            "wins": int((difference > 0).sum()), "ties": int((difference == 0).sum()),
            "losses": int((difference < 0).sum()), "bootstrap_seed": 91741, "resamples": 20000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/board-brain-1"))
    parser.add_argument("--generations", type=int, default=24)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--train-games", type=int, default=2)
    parser.add_argument("--max-moves", type=int, default=100)
    parser.add_argument("--validation-games", type=int, default=6)
    parser.add_argument("--validation-moves", type=int, default=120)
    parser.add_argument("--validate-every", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=.4)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--seed-offset", type=int, default=8000000)
    parser.add_argument("--food-sigma", type=float, default=3.)
    parser.add_argument("--obstacle-sigma", type=float, default=.8)
    parser.add_argument("--gain", type=float, default=1.)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=1,
                        help="Independent CPU candidate simulations; does not change seeds or optimizer")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test-games", type=int, default=32)
    parser.add_argument("--test-moves", type=int, default=250)
    parser.add_argument("--test-seed-index", type=int, default=0,
                        help="Start within the reserved 1000-seed test block; skip seeds used by implementation checks")
    args = parser.parse_args()
    counts = (args.generations, args.train_games, args.max_moves, args.validation_games,
              args.validation_moves, args.validate_every, args.test_games, args.test_moves)
    if min(counts) < 1 or args.population < 4 or not 0 < args.sigma <= LIMIT:
        parser.error("Positive counts, population >=4, and 0 < sigma <= log(4) are required")
    if max(args.train_games, args.validation_games, args.test_games) > 1000:
        parser.error("Use at most 1000 games per split")
    if args.test_seed_index < 0 or args.test_seed_index + args.test_games > 1000:
        parser.error("Test seed index and count must stay within the reserved 1000-seed block")
    if args.seed_offset < 0 or args.seed_offset % 1000000:
        parser.error("Seed offset must be a nonnegative multiple of one million")
    if args.test and args.resume:
        parser.error("Choose --test or --resume")
    if args.device == "cuda" and args.workers != 1:
        parser.error("Multiple workers are supported only on CPU")
    checkpoint, final = args.run / "state.json", args.run / "test.json"
    if args.test or args.resume:
        if not checkpoint.exists():
            parser.error("No checkpoint at this run")
        if final.exists():
            parser.error("This run is locked after its final test starts; use a fresh run for more training")
    elif args.run.exists():
        parser.error("Run exists; use --resume or choose a fresh directory")
    config = {key: getattr(args, key) for key in ("population", "train_games", "max_moves", "validation_games",
              "validation_moves", "validate_every", "sigma", "seed", "seed_offset", "food_sigma", "obstacle_sigma", "gain", "device")}
    if args.test or args.resume:
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if args.resume and (state["config"] != config or state["cma_version"] != cma.__version__):
            parser.error("Resume needs the identical configuration, device, and CMA version")
        if args.test:
            config = state["config"]
    torch.set_num_threads(2)
    runner = BoardRunner(device=config["device"], food_sigma=config["food_sigma"],
                         obstacle_sigma=config["obstacle_sigma"], gain=config["gain"])
    if config["device"] != "cpu" and args.workers != 1:
        parser.error("The saved GPU configuration supports one worker")
    args.run.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    zero = np.zeros(len(runner.sites.labels))
    validation_seeds = range(config["seed_offset"] + 200000,
                             config["seed_offset"] + 200000 + config["validation_games"])
    if args.test:
        parameters, metadata = runner.load(args.run / "winner.npz")
        report = {"complete": False, "evaluation": "paired unseen games; continuous whole connectome; fixed motor decoder",
                  "model_sha256": hashlib.sha256((args.run / "winner.npz").read_bytes()).hexdigest(),
                  "metadata": metadata, "device": config["device"], "max_moves": args.test_moves,
                  "test_seed_start": config["seed_offset"] + 300000 + args.test_seed_index,
                  "results": {}}
        used = {game["seed"] for item in state["history"] for result in item["training"] for game in result["games"]}
        used.update(game["seed"] for game in state["baseline_validation"]["games"])
        if used.intersection(range(report["test_seed_start"], report["test_seed_start"] + args.test_games)):
            parser.error("Test seeds overlap training or validation")
        write_json(final, report)  # irreversible test lock before any test outcome
        seeds = range(report["test_seed_start"], report["test_seed_start"] + args.test_games)
        conditions = [("original", zero, None), ("trained", parameters, None),
                      ("trained_food_only", parameters, "food_only"),
                      ("trained_no_sensory_input", parameters, "no_input"),
                      ("original_24_pattern_reference", zero, "legacy_input")]
        runners = [runner] + [BoardRunner(device=config["device"], food_sigma=config["food_sigma"],
                              obstacle_sigma=config["obstacle_sigma"], gain=config["gain"], connectome=runner.connectome)
                              for _ in range(args.workers - 1)]
        def evaluate_conditions(worker):
            results = []
            for index in range(worker, len(conditions), len(runners)):
                name, vector, ablation = conditions[index]
                result = runners[worker].evaluate(vector, seeds, args.test_moves, ablation=ablation,
                    progress=lambda game: print(name, json.dumps(game), flush=True))
                results.append((name, result))
            return results
        with ThreadPoolExecutor(max_workers=len(runners)) as executor:
            futures = [executor.submit(evaluate_conditions, worker) for worker in range(len(runners))]
            for future in as_completed(futures):
                report["results"].update(future.result())
                report["elapsed_seconds"] = time.perf_counter() - started
                write_json(final, report)
        report["paired_comparison"] = bootstrap_difference(report["results"]["original"], report["results"]["trained"])
        report["trained_vs_24_pattern"] = bootstrap_difference(report["results"]["original_24_pattern_reference"], report["results"]["trained"])
        report["full_board_vs_food_only"] = bootstrap_difference(report["results"]["trained_food_only"], report["results"]["trained"])
        report["simulated_moves"] = sum(r.moves for r in runners)
        report["complete"] = True
        write_json(final, report)
        print("paired_comparison", json.dumps(report["paired_comparison"]), flush=True)
        return
    strategy = optimizer(len(zero), config)
    if args.resume:
        if state["encoder"] != runner.encoder.metadata() or state["connectome_sha256"] != runner.sites.fingerprint:
            parser.error("Checkpoint anatomy or encoder has changed")
        # Replaying the small ask/tell history restores CMA without unpickling
        # executable objects or silently restarting optimizer adaptation.
        for item in state["history"]:
            candidates = strategy.ask()
            if not np.allclose(candidates, item["parameters"], rtol=0, atol=1e-12):
                raise ValueError("CMA resume replay differs from saved candidates")
            strategy.tell(candidates, item["losses"])
    else:
        baseline = runner.evaluate(zero, validation_seeds, config["validation_moves"])
        state = {"generation": 0, "config": config, "encoder": runner.encoder.metadata(),
                 "connectome_sha256": runner.sites.fingerprint, "cma_version": cma.__version__,
                 "baseline_validation": baseline, "best_validation": None, "best_generation": None,
                 "plastic_connections": len(runner.sites.pre), "parameter_groups": len(zero),
                 "initial_parameters": zero.tolist(), "history": [], "elapsed_seconds": 0., "simulated_moves": 0}
        write_json(checkpoint, state)
        print("baseline", json.dumps(baseline), flush=True)
    previous_elapsed, previous_moves = state["elapsed_seconds"], state["simulated_moves"]
    runners = [runner] + [BoardRunner(device=config["device"], food_sigma=config["food_sigma"],
                          obstacle_sigma=config["obstacle_sigma"], gain=config["gain"], connectome=runner.connectome)
                          for _ in range(args.workers - 1)]
    for generation in range(state["generation"] + 1, args.generations + 1):
        start_seed = config["seed_offset"] + 100000 + ((generation - 1) % 80) * 1000
        seeds = range(start_seed, start_seed + config["train_games"])
        candidates = strategy.ask()
        scores = evaluate_population(runners, candidates, seeds, config["max_moves"], generation)
        losses = [-objective(result) for result in scores]
        strategy.tell(candidates, losses)
        selected = int(np.argmin(losses))
        item = {"generation": generation, "parameters": [p.tolist() for p in candidates],
                "losses": losses, "training": scores, "selected": selected, "workers": args.workers}
        if generation == 1 or generation % config["validate_every"] == 0:
            validation = runner.evaluate(candidates[selected], validation_seeds, config["validation_moves"])
            item["validation"] = validation
            if state["best_validation"] is None or objective(validation) > objective(state["best_validation"]):
                state["best_validation"], state["best_generation"] = validation, generation
                runner.save(args.run / "winner.npz", candidates[selected], generation=generation, validation=validation,
                            algorithm="diagonal CMA-ES on bounded existing internal synaptic gains")
        runner.save(args.run / "last-candidate.npz", candidates[selected], generation=generation)
        state["history"].append(item)
        state.update(generation=generation, elapsed_seconds=previous_elapsed + time.perf_counter() - started,
                     simulated_moves=previous_moves + sum(r.moves for r in runners))
        write_json(checkpoint, state)
        print(f"GENERATION {generation}: best validation food {state['best_validation']['mean_food']:.3f}; "
              f"{state['simulated_moves']} whole-brain moves; {state['elapsed_seconds']:.1f}s", flush=True)
    print("Saved", args.run / "winner.npz", flush=True)


if __name__ == "__main__":
    main()

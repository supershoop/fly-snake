"""Train EXISTING fly-brain synapses from Snake rewards, keeping the decoder fixed.

Every move runs the continuous whole connectome. No response bank or teacher.
CPU: optional compiled kernel. GPU: existing torch kernel (--device cuda).
Resumable runs retain seeds, RNG, anatomical identity, validation and models.
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
from flybrain.readout import HardwiredPolicy
from flybrain.snake import Snake
from flybrain.synaptic import DECODER, SynapticAdapter, SynapticSites, perturb


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class Runner:
    def __init__(self, device="cpu", compiled=True):
        connectome = load_connectome()
        self.channels = build_channels(connectome)
        self.sites = SynapticSites.from_connectome(connectome)
        self.brain = Brain(connectome, device=device, compiled=compiled and device == "cpu")
        self.adapter = SynapticAdapter(self.brain, self.sites)
        self.policy = HardwiredPolicy(self.channels.steer_sign, threshold=DECODER["threshold"])
        self.stimulus = self.channels.stim_index.to(device)
        self.readout = self.channels.readout_index.to(device)
        self.moves = 0

    def evaluate(self, parameters, seeds, limit, progress=None):
        self.adapter.apply(parameters)
        records = []
        for seed in seeds:
            game = Snake(seed=int(seed))
            self.brain.reset()  # retain state THROUGHOUT each game
            self.brain.rng.manual_seed(int(seed) + 1000000)
            reward_sum = 0.
            for move in range(limit):
                levels = self.channels.levels(torch.as_tensor(game.encode(), device=self.brain.device)[:, None])
                counts = self.brain.run(100., self.stimulus, levels, self.readout)
                action = int(self.policy.act(counts)[0][0])
                reward_sum += game.step(action)
                self.moves += 1
                if not game.alive:
                    break
            records.append({"seed": int(seed), "brain_seed": int(seed) + 1000000, "food": game.score,
                            "moves": move + 1, "reward": reward_sum, "end_reason": game.snakes[0].end_reason,
                            "alive_at_limit": game.alive})
            if progress is not None:
                progress(records[-1])
        return {"games": records, "mean_food": float(np.mean([g["food"] for g in records])),
                "mean_reward": float(np.mean([g["reward"] for g in records])),
                "collisions": sum(g["end_reason"] == "collision" for g in records),
                "alive_at_limit": sum(g["alive_at_limit"] for g in records)}


def fitness(result):
    return result["mean_food"], result["mean_reward"], -result["collisions"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/synaptic-1"))
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--train-games", type=int, default=2)
    parser.add_argument("--max-moves", type=int, default=120)
    parser.add_argument("--validation-games", type=int, default=4)
    parser.add_argument("--validation-moves", type=int, default=200)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--seed-offset", type=int, default=0, help="Use a fresh multiple of 1000000 for a new research run")
    parser.add_argument("--initial", type=Path, help="Warm-start a NEW run from saved brain synapses, keeping its decoder fixed")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-kernel", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test", action="store_true", help="freeze winner; evaluate on untouched paired seeds")
    parser.add_argument("--test-games", type=int, default=16)
    parser.add_argument("--test-moves", type=int, default=250)
    args = parser.parse_args()
    if min(args.generations, args.train_games, args.max_moves, args.validation_games,
           args.validation_moves, args.validate_every, args.test_games, args.test_moves) < 1:
        parser.error("Counts and move limits must be positive")
    if max(args.train_games, args.validation_games, args.test_games) > 1000:
        parser.error("At most 1000 games per split to keep seed ranges disjoint")
    if args.seed_offset < 0 or args.seed_offset % 1000000 != 0:
        parser.error("Seed offset must be a nonnegative multiple of 1000000")
    if args.initial and (args.resume or args.test):
        parser.error("--initial is only for a new training run; resume already restores its parameters")
    torch.set_num_threads(2)
    checkpoint = args.run / "state.json"
    final = args.run / "test.json"
    if args.test:
        if not checkpoint.exists():
            parser.error("Train a run before testing")
        if final.exists():
            parser.error("This run has already started its final test; use the saved report")
    elif args.resume:
        if not checkpoint.exists():
            parser.error("No checkpoint to resume")
        if final.exists():
            parser.error("A tested run is frozen; start a new run for further research")
    elif args.run.exists():
        parser.error("Run already exists; use --resume or a new directory")
    runner = Runner(args.device, not args.torch_kernel)
    args.run.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    config = {key: getattr(args, key) for key in ("train_games", "max_moves", "validation_games", "validation_moves", "validate_every", "seed", "seed_offset")}
    started = time.perf_counter()
    if args.test:
        run_state = json.loads(checkpoint.read_text(encoding="utf-8"))
        test_seed = 300000 + run_state["config"].get("seed_offset", 0)
        model = args.run / "winner.npz"
        parameters, metadata = runner.sites.load(model)
        report = {"complete": False, "evaluation": "continuous whole connectome; fixed hardwired readout; paired unseen seeds",
                  "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(), "metadata": metadata,
                  "device": args.device, "compiled": runner.brain.compiled, "max_moves": args.test_moves,
                  "test_seed_start": test_seed, "results": {}}
        write_json(final, report)  # lock before observing any test result
        for name, vector in (("unmodified", np.zeros(len(parameters))), ("trained_synapses", parameters)):
            report["results"][name] = runner.evaluate(vector, range(test_seed, test_seed + args.test_games), args.test_moves,
                                                     progress=lambda record: print(name, json.dumps(record), flush=True))
            report["elapsed_seconds"] = time.perf_counter() - started
            write_json(final, report)
            print(name, json.dumps(report["results"][name]), flush=True)
        report["complete"] = True
        write_json(final, report)
        return
    if args.resume:
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        state["config"].setdefault("seed_offset", 0)
        if state["config"] != config or state["connectome_sha256"] != runner.sites.fingerprint:
            parser.error("Resume requires the original configuration and anatomy")
        parameters = runner.sites.validate(state["parameters"])
        rng.bit_generator.state = state["rng"]
        if args.device != state["device"]:
            # CPU and CUDA do not promise identical stochastic trajectories.
            # Recalibrate both comparators instead of comparing across devices.
            winner, metadata = runner.sites.load(args.run / "winner.npz")
            validation_seeds = range(200000 + args.seed_offset, 200000 + args.seed_offset + args.validation_games)
            state.setdefault("device_changes", []).append({"after_generation": state["generation"],
                                                          "from": state["device"], "to": args.device})
            state["baseline_validation"] = runner.evaluate(np.zeros(len(parameters)), validation_seeds, args.validation_moves)
            state["best_validation"] = runner.evaluate(winner, validation_seeds, args.validation_moves)
            runner.sites.save(args.run / "winner.npz", winner, generation=metadata["generation"], validation=state["best_validation"])
    else:
        parameters = np.zeros(len(runner.sites.labels))
        validation_seeds = range(200000 + args.seed_offset, 200000 + args.seed_offset + args.validation_games)
        baseline = runner.evaluate(parameters, validation_seeds, args.validation_moves)
        state = {"generation": 0, "config": config, "connectome_sha256": runner.sites.fingerprint,
                 "baseline_validation": baseline, "best_validation": baseline, "best_generation": 0,
                 "history": [], "elapsed_seconds": 0., "simulated_moves": 0}
        runner.sites.save(args.run / "winner.npz", parameters, generation=0, validation=baseline)
        if args.initial:
            parameters, _ = runner.sites.load(args.initial)
            initial_validation = runner.evaluate(parameters, validation_seeds, args.validation_moves)
            state["initial_sha256"] = hashlib.sha256(args.initial.read_bytes()).hexdigest()
            state["initial_validation"] = initial_validation
            if fitness(initial_validation) > fitness(baseline):
                state["best_validation"] = initial_validation
                runner.sites.save(args.run / "winner.npz", parameters, generation=0, validation=initial_validation,
                                  initial_sha256=state["initial_sha256"])
        state.update(parameters=parameters.tolist(), rng=rng.bit_generator.state,
                     device=args.device, compiled=runner.brain.compiled)
        write_json(checkpoint, state)
        print("baseline", json.dumps(baseline), flush=True)
    previous_elapsed, previous_moves = state["elapsed_seconds"], state["simulated_moves"]
    for generation in range(state["generation"] + 1, args.generations + 1):
        seeds = range(100000 + args.seed_offset + ((generation - 1) // 4 % 64) * 1000,
                      100000 + args.seed_offset + ((generation - 1) // 4 % 64) * 1000 + args.train_games)
        proposals = [parameters, *perturb(parameters, rng)]
        scores = [runner.evaluate(vector, seeds, args.max_moves) for vector in proposals]
        selected = max(range(len(scores)), key=lambda i: fitness(scores[i]))
        parameters = proposals[selected]
        item = {"generation": generation, "training": scores, "selected": selected, "device": args.device}
        if generation % args.validate_every == 0 or generation == 1:
            validation = runner.evaluate(parameters, range(200000 + args.seed_offset, 200000 + args.seed_offset + args.validation_games), args.validation_moves)
            item["validation"] = validation
            if fitness(validation) > fitness(state["best_validation"]):
                state["best_validation"], state["best_generation"] = validation, generation
                runner.sites.save(args.run / "winner.npz", parameters, generation=generation, validation=validation)
        state["history"].append(item)
        state.update(generation=generation, parameters=parameters.tolist(), rng=rng.bit_generator.state,
                     elapsed_seconds=previous_elapsed + time.perf_counter() - started,
                     simulated_moves=previous_moves + runner.moves, device=args.device, compiled=runner.brain.compiled)
        write_json(checkpoint, state)
        print(f"generation {generation}: food {scores[selected]['mean_food']:.2f}; best validation "
              f"{state['best_validation']['mean_food']:.2f}; {state['simulated_moves']} brain moves; "
              f"{state['elapsed_seconds']:.1f}s", flush=True)
    print(f"Saved brain synapses to {args.run / 'winner.npz'}; readout unchanged.", flush=True)


if __name__ == "__main__":
    main()

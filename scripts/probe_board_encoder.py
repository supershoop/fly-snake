"""Test direct-board stimulation on the original, untrained whole brain."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain.board_experiment import BoardRunner
from flybrain.snake import Snake


def cases():
    result = {}
    for name, food, cells in [
        ("food_left_near", (4, 6), [(6, 6), (6, 7), (6, 8)]),
        ("food_left_far", (1, 6), [(6, 6), (6, 7), (6, 8)]),
        ("food_right_near", (8, 6), [(6, 6), (6, 7), (6, 8)]),
        ("food_ahead", (6, 3), [(6, 6), (6, 7), (6, 8)]),
        ("different_body", (4, 6), [(6, 6), (6, 7), (5, 7), (5, 8), (6, 8)]),
        ("near_left_wall", (0, 6), [(2, 6), (2, 7), (2, 8)]),
    ]:
        game = Snake(seed=0)
        game.snakes[0].cells, game.snakes[0].heading = cells, 0
        game.foods = [food]
        result[name] = game
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/board-probe.json"))
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--food-sigma", type=float, default=3.)
    parser.add_argument("--obstacle-sigma", type=float, default=.8)
    parser.add_argument("--gain", type=float, default=1.)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("Use at least two repetitions")
    torch.set_num_threads(2)
    runner = BoardRunner(food_sigma=args.food_sigma, obstacle_sigma=args.obstacle_sigma, gain=args.gain)
    report = {"encoder": runner.encoder.metadata(), "connectome_sha256": runner.sites.fingerprint,
              "plastic_connections": len(runner.sites.pre), "parameter_groups": len(runner.sites.labels),
              "brain": "original unmodified connectome", "repeats": args.repeats,
              "window_ms": 100, "brain_seeds": list(range(7000000, 7000000 + args.repeats)), "cases": {}}
    started = time.perf_counter()
    responses, levels = {}, {}
    for name, game in cases().items():
        repetitions, actions, active = [], [], []
        levels[name] = runner.encoder.encode(game)
        for repeat in range(args.repeats):
            runner.brain.reset()
            runner.brain.rng.manual_seed(7000000 + repeat)
            counts = runner.brain.run(100., runner.stimulus, torch.as_tensor(levels[name])[:, None])
            dn = counts[runner.readout]
            repetitions.append(dn[:, 0].numpy().copy())
            actions.append(int(runner.policy.act(dn)[0][0]))
            active.append(int((counts > 0).sum()))
        responses[name] = np.array(repetitions)
        steer = responses[name] @ runner.channels.steer_sign.numpy()
        report["cases"][name] = {"old_24_state": [int(value) for value in game.state()],
                                  "mean_active_neurons": float(np.mean(active)),
                                  "mean_active_descending": float(np.mean((responses[name] > 0).sum(axis=1))),
                                  "steering_differences": steer.tolist(), "actions": actions,
                                  "stimulus_nonzero": int(np.count_nonzero(levels[name]))}
        print(name, json.dumps(report["cases"][name]), flush=True)
    reference = "food_left_near"
    report["paired_changes_from_food_left_near"] = {}
    for name in responses:
        if name == reference:
            continue
        paired = responses[name] - responses[reference]
        # Descriptive paired differences, not a decoder trained on probe labels.
        report["paired_changes_from_food_left_near"][name] = {
            "same_old_input": report["cases"][name]["old_24_state"] == report["cases"][reference]["old_24_state"],
            "changed_input_neurons": int(np.count_nonzero(levels[name] != levels[reference])),
            "mean_changed_descending_neurons": float(np.mean((paired != 0).sum(axis=1))),
            "mean_absolute_dn_spike_difference": float(np.abs(paired).mean()),
            "mean_steering_difference_change": float(np.mean(paired @ runner.channels.steer_sign.numpy()))}
    report["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.output.with_suffix(".npz"), **responses)
    print("saved", args.output, report["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()

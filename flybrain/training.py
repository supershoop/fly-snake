"""Train action targets from long-rollout rewards, then fit them to real DN responses.

The target table is an offline training tool, never an inference controller.
Coordinate search starts from the old food-seeking teacher and keeps changes
only when they improve food scores on the training seeds within the move limit.
"""
import numpy as np

from .snake import ALL_STATES, Snake, teacher

STATE_INDEX = {state: i for i, state in enumerate(ALL_STATES)}


def score_targets(targets, seeds, max_moves=1600, *, lookahead=True, response_seed=0):
    """Score a target table [24], or sampled response-bank actions [24, trials]."""
    scores, collisions, starved, alive, moves = [], 0, 0, 0, []
    for seed in seeds:
        game = Snake(seed=int(seed))
        rng = np.random.default_rng([response_seed, int(seed)])
        for step in range(max_moves):
            state = STATE_INDEX[game.state(lookahead=lookahead)]
            action = targets[state] if targets.ndim == 1 else targets[state, rng.integers(targets.shape[1])]
            game.step(int(action))
            if not game.alive:
                break
        scores.append(game.score)
        moves.append(step + 1)
        collisions += game.snakes[0].end_reason == "collision"
        starved += game.snakes[0].end_reason == "starvation"
        alive += game.alive
    return {"mean_score": float(np.mean(scores)), "scores": scores, "collisions": collisions,
            "starved": starved, "alive_at_limit": alive, "mean_moves": float(np.mean(moves))}


def optimize_targets(seeds, max_moves=1600, passes=2, progress=print):
    """Use food collected over long games as the training objective.

    Only safe alternatives are explored during target search. This restriction
    is not applied to readout actions during evaluation or live play.
    """
    targets = np.array([teacher(state) for state in ALL_STATES], dtype=np.int64)
    best = score_targets(targets, seeds, max_moves)
    progress(f"target training: initial mean score {best['mean_score']:.2f}")
    for epoch in range(passes):
        changed = False
        for index, (_, *danger) in enumerate(ALL_STATES):
            for action in range(3):
                if danger[action] or action == targets[index]:
                    continue
                candidate = targets.copy()
                candidate[index] = action
                result = score_targets(candidate, seeds, max_moves)
                if (result["mean_score"], -result["collisions"]) > (best["mean_score"], -best["collisions"]):
                    targets, best, changed = candidate, result, True
                    progress(f"target training pass {epoch+1}: state {index} -> {action}; mean score {best['mean_score']:.2f}")
        if not changed:
            break
    return targets, best

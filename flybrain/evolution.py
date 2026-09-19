"""Reward-driven evolution of the linear descending-neuron readout.

No teacher labels, action masks, or trainable connectome synapses. A fixed,
label-free feature basis reduces neutral mutations; it is folded into the
ordinary linear weights when saving a policy.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib

import numpy as np

from .evolution_game import score_actions


def feature_basis(train_counts, ridge=100.):
    """24 response-mean directions, computed only from training responses."""
    means = np.log1p(train_counts.astype(np.float64)).mean(axis=1)
    means = np.column_stack((means, np.ones(len(means))))
    return np.linalg.solve(means @ means.T + ridge * np.eye(len(means)), means).T


def features(counts, basis):
    values = np.concatenate((np.log1p(counts.astype(np.float64)), np.ones((*counts.shape[:2], 1))), axis=2)
    return values @ basis


def actions_for(parameters, projected):
    return np.argmax(projected @ parameters.T, axis=-1)


def weights_for(parameters, basis):
    combined = parameters @ basis.T
    return combined[:, :-1].astype(np.float32), combined[:, -1].astype(np.float32)


def fitness(result):
    # Food dominates. For ties prefer fewer crashes/timeouts, then more moves.
    return (result["mean_score"], -result["collisions"], -result["starved"], result["mean_moves"])


def breed(elites, population, rng):
    """Elitist mutation-only genetic algorithm; occasional new random entrants."""
    children = [parent.copy() for parent in elites]
    while len(children) < population:
        if rng.random() < .04:
            child = rng.normal(0, 1, elites[0].shape)
        else:
            child = elites[int(rng.integers(len(elites)))].copy()
            count = int(rng.choice([1, 2, 4, 12, child.size], p=[.35, .3, .2, .1, .05]))
            index = rng.choice(child.size, count, replace=False)
            scale = float(rng.choice([.1, .3, 1., 3.]))
            child.flat[index] += rng.normal(0, scale, count)
        # Equal changes to every action's logit have no behavioral effect.
        child -= child.mean(axis=0, keepdims=True)
        children.append(child)
    return np.stack(children)


class Evaluator:
    """Cache identical action tables, never scores across different game seeds."""
    def __init__(self, workers=4):
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.cache = {}
        self.context = None
        self.evaluations = self.cache_hits = self.games = 0

    def close(self):
        self.pool.shutdown()

    def evaluate(self, actions, seeds, max_moves, response_seed):
        context = (tuple(seeds), max_moves, response_seed)
        if context != self.context:
            self.context, self.cache = context, {}
        keys = [hashlib.sha256(np.asarray(table, dtype=np.uint8).tobytes()).digest() for table in actions]
        missing = {}
        for key, table in zip(keys, actions):
            if key not in self.cache and key not in missing:
                missing[key] = table
        def run(table):
            return score_actions(table, seeds, max_moves, response_seed=response_seed)
        for key, result in zip(missing, self.pool.map(run, missing.values())):
            self.cache[key] = result
        self.evaluations += len(missing)
        self.games += len(missing) * len(seeds)
        self.cache_hits += len(keys) - len(missing)
        return [self.cache[key] for key in keys]

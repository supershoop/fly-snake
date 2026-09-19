"""Compiled solo Snake rollouts for offline response-bank estimates.

This is an optimization of Snake + tail-escape sensing, not a different game.
Differential tests compare it with the canonical Python implementation. It does
not simulate neurons and is never used as a live motor controller.
"""
from functools import lru_cache

import numpy as np

try:
    from numba import njit
except ImportError as exc:
    raise ImportError("Evolution training needs numba: python -m pip install numba") from exc


@lru_cache(maxsize=8)
def neighbours(size):
    result = np.full((size * size, 4), -1, dtype=np.int64)
    for x in range(size):
        for y in range(size):
            for direction, (dx, dy) in enumerate(((0, -1), (1, 0), (0, 1), (-1, 0))):
                if 0 <= x + dx < size and 0 <= y + dy < size:
                    result[x * size + y, direction] = (x + dx) * size + y + dy
    return result


@njit(cache=True)
def _state(body, length, heading, food, size, adjacent, blocked, visited, queue):
    head = body[0]
    fx, fy = food // size - head // size, food % size - head % size
    hx, hy = ((0, -1), (1, 0), (0, 1), (-1, 0))[heading]
    forward, rightward = fx * hx + fy * hy, fx * -hy + fy * hx
    if rightward == 0:
        side = 1 if forward > 0 else 0
    elif forward > abs(rightward):
        side = 1
    else:
        side = 2 if rightward > 0 else 0
    state = side * 8
    for action in range(3):
        destination = adjacent[head, (heading + action - 1) % 4]
        blocked[:] = False
        for i in range(length - 1):
            blocked[body[i]] = True
        danger = destination < 0 or blocked[destination]
        if not danger:
            growing = destination == food
            tail = body[length - 1 if growing else length - 2]
            if not growing:
                blocked[tail] = False
            visited[:] = False
            visited[destination] = True
            queue[0] = destination
            read, write = 0, 1
            reached = destination == tail
            while read < write and not reached:
                cell = queue[read]
                read += 1
                for direction in range(4):
                    other = adjacent[cell, direction]
                    if other >= 0 and not blocked[other] and not visited[other]:
                        if other == tail:
                            reached = True
                            break
                        visited[other] = True
                        queue[write] = other
                        write += 1
            danger = not reached
        if danger:
            state += 1 << (2 - action)
    return state


@njit(cache=True)
def _food(body, length, occupied, rng):
    occupied[:] = False
    for i in range(length):
        occupied[body[i]] = True
    free = len(occupied) - length
    if free == 0:
        return -1
    wanted = rng.integers(0, free)
    for cell in range(len(occupied)):
        if not occupied[cell]:
            if wanted == 0:
                return cell
            wanted -= 1
    return -1


@njit(cache=True, nogil=True)
def _rollout(actions, rng, response_rng, size, max_moves, max_idle, adjacent):
    body = np.empty(size * size, dtype=np.int64)
    occupied = np.zeros(size * size, dtype=np.bool_)
    visited = np.zeros(size * size, dtype=np.bool_)
    queue = np.empty(size * size, dtype=np.int64)
    x, y = rng.integers(2, size - 3), rng.integers(1, size - 1)
    for i in range(3):
        body[i] = (x - i) * size + y
    length, heading, idle, score = 3, 1, 0, 0
    food = _food(body, length, occupied, rng)
    for step in range(max_moves):
        state = _state(body, length, heading, food, size, adjacent, occupied, visited, queue)
        action = actions[state, response_rng.integers(0, actions.shape[1])]
        direction = (heading + action - 1) % 4
        head = adjacent[body[0], direction]
        collision = head < 0
        for i in range(length - 1):
            collision = collision or head == body[i]
        if collision:
            return score, 1, step + 1
        heading = direction
        growing = head == food
        if growing:
            length += 1
        for i in range(length - 1, 0, -1):
            body[i] = body[i - 1]
        body[0] = head
        if growing:
            food = _food(body, length, occupied, rng)
            score += 1
            idle = 0
            if food < 0:
                return score, 3, step + 1
        else:
            idle += 1
            if idle >= max_idle:
                return score, 2, step + 1
    return score, 0, max_moves


def score_actions(actions, seeds, max_moves=1600, *, response_seed=0, size=12, max_idle=150):
    """Same RNG streams, initial boards and food ordering as score_targets()."""
    actions = np.asarray(actions)
    if actions.ndim == 1:
        actions = actions[:, None]
    if (actions.ndim != 2 or actions.shape[0] != 24 or actions.shape[1] < 1
            or not np.isin(actions, (0, 1, 2)).all()):
        raise ValueError("Expected actions [24, trials] with values 0, 1, 2")
    seeds = list(seeds)
    if not seeds or min(max_moves, max_idle) < 1 or size < 6:
        raise ValueError("Need seeds, positive move limits and size >= 6")
    actions = np.ascontiguousarray(actions, dtype=np.int64)
    rows = np.array([_rollout(actions, np.random.default_rng(int(seed)),
                             np.random.default_rng([response_seed, int(seed)]),
                             size, max_moves, max_idle, neighbours(size)) for seed in seeds])
    return {"mean_score": float(rows[:, 0].mean()), "scores": rows[:, 0].tolist(),
            "collisions": int((rows[:, 1] == 1).sum()), "starved": int((rows[:, 1] == 2).sum()),
            "filled": int((rows[:, 1] == 3).sum()), "alive_at_limit": int((rows[:, 1] == 0).sum()),
            "mean_moves": float(rows[:, 2].mean())}


def state_index(game):
    """Adapter used for differential tests of arbitrary canonical game states."""
    if len(game.snakes) != 1 or len(game.foods) != 1:
        raise ValueError("The compiled evaluator supports one snake and one food")
    snake = game.snakes[0]
    body = np.array([x * game.size + y for x, y in snake.cells], dtype=np.int64)
    food = game.foods[0][0] * game.size + game.foods[0][1]
    n = game.size ** 2
    return _state(body, len(body), snake.heading, food, game.size, neighbours(game.size),
                  np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_), np.empty(n, dtype=np.int64))

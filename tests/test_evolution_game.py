import unittest

import numpy as np

try:
    import numba  # noqa: F401
except ImportError:
    raise unittest.SkipTest("Install requirements-training.txt for evolution tests")

from flybrain.evolution_game import score_actions, state_index
from flybrain.snake import ALL_STATES, Snake, teacher
from flybrain.training import score_targets


class EvolutionGameTests(unittest.TestCase):
    def test_rollouts_match_canonical_game(self):
        rng = np.random.default_rng(717)
        teacher_actions = np.array([teacher(s) for s in ALL_STATES])
        policies = [teacher_actions, np.ones(24, dtype=int), np.zeros(24, dtype=int)]
        for _ in range(5):
            policy = np.tile(teacher_actions[:, None], (1, 4))
            changes = rng.random(policy.shape) < .12
            policy[changes] = rng.integers(3, size=changes.sum())
            policies.append(policy)
        for actions in policies:
            expected = score_targets(actions, range(12), 900, response_seed=71)
            actual = score_actions(actions, range(12), 900, response_seed=71)
            self.assertEqual(expected, {k: actual[k] for k in expected})

    def test_states_match_during_long_games(self):
        rng = np.random.default_rng(77)
        for size in (6, 8, 12):
            for seed in range(8):
                game = Snake(size=size, seed=seed)
                for _ in range(600):
                    if not game.alive:
                        break
                    state = game.state()
                    self.assertEqual(state_index(game), ALL_STATES.index(state))
                    action = teacher(state) if rng.random() > .01 else rng.integers(3)
                    game.step(int(action))

    def test_invalid_actions_and_limits(self):
        for actions in (np.ones((23, 4)), np.ones((24, 0)), np.full(24, 3), np.full(24, np.nan)):
            with self.assertRaises(ValueError):
                score_actions(actions, [1])
        with self.assertRaises(ValueError):
            score_actions(np.ones(24), [1], max_moves=0)

    def test_growth_and_tail_traps_match_reference(self):
        game = Snake(size=6)
        snake = game.snakes[0]
        snake.cells = [(2, 1), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (1, 2), (0, 2), (0, 3)]
        snake.heading = 2
        game.foods = [(2, 2)]
        self.assertEqual(state_index(game), ALL_STATES.index(game.state()))
        snake.cells = [(3, 2), (3, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5), (4, 4), (4, 3),
                       (4, 2), (4, 1), (5, 1), (5, 0), (4, 0), (3, 0), (2, 0), (2, 1)]
        snake.heading = 0
        for food in ((0, 5), (3, 1)):
            game.foods = [food]
            self.assertEqual(state_index(game), ALL_STATES.index(game.state()))
        snake.cells, snake.heading = [(2, 2), (2, 3), (1, 3), (1, 2)], 0
        game.foods = [(5, 5)]
        self.assertEqual(state_index(game), ALL_STATES.index(game.state()))


if __name__ == "__main__":
    unittest.main()

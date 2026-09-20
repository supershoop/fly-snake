import unittest

import numpy as np

from flybrain.navigation import dangers
from flybrain.snake import Arena, Snake, LEFT, STRAIGHT, RIGHT, teacher
from flybrain.training import score_targets


class EscapeSensingTests(unittest.TestCase):
    def test_open_cell_can_close_a_box(self):
        game = Snake(size=6)
        snake = game.snakes[0]
        snake.cells = [(2, 1), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (1, 2), (0, 2), (0, 3)]
        snake.heading = 2
        game.foods = [(2, 2)]
        self.assertFalse(game.blocked(0, STRAIGHT))
        self.assertEqual(dangers(game), (True, True, False))
        self.assertEqual(teacher(game.state()), RIGHT)
        self.assertEqual(teacher(game.state(lookahead=False)), STRAIGHT)

    def test_growth_can_close_the_escape_route(self):
        game = Snake(size=6)
        snake = game.snakes[0]
        snake.cells = [(3, 2), (3, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5), (4, 4), (4, 3),
                       (4, 2), (4, 1), (5, 1), (5, 0), (4, 0), (3, 0), (2, 0), (2, 1), (1, 1)]
        snake.heading = 0
        game.foods = [(0, 5)]
        self.assertFalse(dangers(game)[STRAIGHT])
        game.foods = [(3, 1)]
        self.assertTrue(dangers(game)[STRAIGHT])
        self.assertFalse(dangers(game)[LEFT])

    def test_following_a_moving_tail_is_allowed(self):
        game = Snake(size=6)
        game.snakes[0].cells = [(2, 2), (2, 3), (1, 3), (1, 2)]
        game.snakes[0].heading = 0
        game.foods = [(5, 5)]
        self.assertFalse(game.blocked(0, LEFT))
        self.assertFalse(dangers(game)[LEFT])
        game.step(LEFT)
        self.assertTrue(game.alive)
        self.assertEqual(len(set(game.snakes[0].cells)), 4)

    def test_other_snakes_are_obstacles(self):
        game = Arena(size=12, kinds=("fly", "human"))
        game.snakes[0].cells = [(3, 3), (3, 4), (3, 5)]
        game.snakes[0].heading = 0
        game.snakes[1].cells = [(3, 2), (4, 2), (5, 2)]
        self.assertTrue(dangers(game)[STRAIGHT])
        game.snakes[1].alive = False
        self.assertFalse(dangers(game)[STRAIGHT])

    def test_sensing_does_not_move_or_choose_for_the_snake(self):
        game = Snake(seed=7)
        before = (list(game.snakes[0].cells), list(game.foods), game.snakes[0].heading)
        game.encode()
        self.assertEqual(before, (game.snakes[0].cells, game.foods, game.snakes[0].heading))
        self.assertTrue(np.isin(game.encode(), (0., 1.)).all())
        self.assertEqual(game.encode().shape, (5,))

    def test_filling_board_completes_and_can_respawn(self):
        game = Arena(size=6)
        cycle = [(x, 0) for x in range(6)]
        for y in range(1, 6):
            cycle += [(x, y) for x in (range(5, 0, -1) if y % 2 else range(1, 6))]
        cycle += [(0, y) for y in range(5, 0, -1)]
        game.snakes[0].cells = cycle[1:]
        game.snakes[0].heading = 3
        game.foods = [cycle[0]]
        self.assertEqual(game.step({0: STRAIGHT})[0], 1.)
        self.assertEqual(game.snakes[0].end_reason, "filled")
        self.assertFalse(game.snakes[0].alive)
        self.assertTrue(np.isfinite(game.encode()).all())
        game.step({})
        self.assertTrue(game.snakes[0].alive)
        self.assertEqual(len(game.foods), 1)

    def test_training_scores_are_repeatable(self):
        from flybrain.snake import ALL_STATES
        targets = np.array([teacher(state) for state in ALL_STATES])
        self.assertEqual(score_targets(targets, [41, 42], max_moves=50),
                         score_targets(targets, [41, 42], max_moves=50))


if __name__ == "__main__":
    unittest.main()

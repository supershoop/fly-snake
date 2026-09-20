import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import torch

from flybrain.board_encoder import BoardEncoder
from flybrain.board_experiment import BoardRunner
from flybrain.snake import Snake


def encoder():
    indices = np.arange(3 * 23 * 23).reshape(3, 23, 23)
    return BoardEncoder(12, indices, indices + 10000)


def game():
    arena = Snake(seed=1)
    arena.snakes[0].cells = [(6, 6), (6, 7), (6, 8)]
    arena.snakes[0].heading = 0
    arena.foods = [(4, 6)]
    return arena


class BoardEncoderTests(unittest.TestCase):
    def test_every_square_and_complete_body_order_survive_raw_encoding(self):
        e, g = encoder(), game()
        p = e.planes(g)
        expected_inside = 144
        self.assertEqual(int((p[1] == 0).sum()) + len(g.snakes[0].cells), expected_inside)
        self.assertEqual(p[0, 11, 9], 1.)
        for i in range(3):
            self.assertAlmostEqual(float(p[2, 11 + i, 11]), (i + 1) / 144)
        self.assertEqual(int(np.count_nonzero(p[2])), 3)
        self.assertEqual(p[1, 0, 0], 1.)

    def test_same_legacy_situation_has_distinct_food_body_and_wall_inputs(self):
        e, a = encoder(), game()
        for modification in ("food", "body", "wall"):
            b = copy.deepcopy(a)
            if modification == "food":
                b.foods = [(1, 6)]
            elif modification == "body":
                b.snakes[0].cells = [(6, 6), (6, 7), (5, 7), (5, 8), (6, 8)]
            else:
                b.snakes[0].cells = [(2, 6), (2, 7), (2, 8)]
                b.foods = [(0, 6)]
            self.assertEqual(a.state(), b.state())
            self.assertFalse(np.array_equal(e.planes(a), e.planes(b)))
            self.assertFalse(np.array_equal(e.encode(a), e.encode(b)))

    def test_rotation_preserves_relative_input(self):
        e, a = encoder(), game()
        for turns in range(1, 4):
            b = copy.deepcopy(a)
            for _ in range(turns):
                b.snakes[0].cells = [(11 - y, x) for x, y in b.snakes[0].cells]
                b.foods = [(11 - y, x) for x, y in b.foods]
                b.snakes[0].heading = (b.snakes[0].heading + 1) % 4
            np.testing.assert_array_equal(e.planes(a), e.planes(b))
            np.testing.assert_array_equal(e.encode(a), e.encode(b))

    def test_does_not_call_legacy_sensing_and_idle_is_visible(self):
        e, g = encoder(), game()
        for method in ("encode", "state", "blocked", "food_side", "_food_distance"):
            setattr(g, method, Mock(side_effect=AssertionError("No heuristic sensing")))
        before = e.encode(g)
        g.snakes[0].idle += 1
        after = e.encode(g)
        self.assertEqual(int(np.count_nonzero(before != after)), 1)
        self.assertEqual(before.shape, (1587,))
        self.assertTrue(np.isfinite(before).all())
        self.assertTrue(((before >= 0) & (before <= 1)).all())

    def test_body_order_is_distinct_even_with_same_occupied_squares(self):
        e, a = encoder(), game()
        a.snakes[0].cells = [(6, 6), (6, 7), (7, 7), (7, 6)]
        b = copy.deepcopy(a)
        b.snakes[0].cells = [(6, 6), (7, 6), (7, 7), (6, 7)]
        np.testing.assert_array_equal(e.planes(a)[:2], e.planes(b)[:2])
        self.assertFalse(np.array_equal(e.encode(a), e.encode(b)))

    def test_mapping_is_repeatable_disjoint_and_part_of_checkpoint_identity(self):
        rows = [(kind, side) for kind in ("LC10a", "LC4", "Tm3") for side in ("L", "R") for _ in range(300)]
        neurons = pd.DataFrame(rows, columns=["type", "side"])
        neurons["bodyId"] = np.arange(len(rows)) + 10000
        c = SimpleNamespace(neurons=neurons)
        a = BoardEncoder.from_connectome(c)
        b = BoardEncoder.from_connectome(c)
        self.assertEqual(a.metadata(), b.metadata())
        self.assertEqual(len(np.unique(a.indices)), 1587)
        b.food_sigma += .1
        self.assertNotEqual(a.metadata()["sha256"], b.metadata()["sha256"])
        with self.assertRaises(ValueError):
            BoardEncoder.from_connectome(c, size=20)
        for options in ({"food_sigma": np.nan}, {"obstacle_sigma": np.inf}, {"gain": 0.}):
            with self.assertRaises(ValueError):
                BoardEncoder.from_connectome(c, **options)

    def test_wrong_encoder_checkpoint_is_rejected(self):
        runner = BoardRunner.__new__(BoardRunner)
        runner.encoder = encoder()
        metadata = {"encoder": {**runner.encoder.metadata(), "gain": .5}}
        runner.sites = SimpleNamespace(load=lambda path: (np.zeros(3), metadata))
        with self.assertRaisesRegex(ValueError, "encoder"):
            runner.load("unused")

    def test_rejects_wrong_board_size_or_multisnake(self):
        e, g = encoder(), game()
        g.size = 10
        with self.assertRaises(ValueError):
            e.encode(g)
        g.size = 12
        g.snakes.append(copy.deepcopy(g.snakes[0]))
        with self.assertRaises(ValueError):
            e.encode(g)

    def test_ablation_controls_remove_only_the_declared_inputs(self):
        runner = BoardRunner.__new__(BoardRunner)
        runner.encoder = encoder()
        runner.stimulus = torch.as_tensor(runner.encoder.stimulus)
        runner.readout = torch.tensor([0])
        runner.moves = 0
        captured = []
        def run(ms, indices, levels, readout):
            captured.append(levels.numpy().copy())
            return torch.zeros(1, 1)
        runner.brain = SimpleNamespace(device=torch.device("cpu"), run=run)
        runner.policy = SimpleNamespace(act=lambda counts: (torch.tensor([1]), None))
        runner.step(game())
        runner.step(game(), ablation="food_only")
        runner.step(game(), ablation="no_input")
        np.testing.assert_array_equal(captured[0][:529], captured[1][:529])
        self.assertTrue(np.any(captured[0][529:]))
        self.assertFalse(np.any(captured[1][529:]))
        self.assertFalse(np.any(captured[2]))
        self.assertEqual(runner.moves, 3)


if __name__ == "__main__":
    unittest.main()

"""Synthetic decisions verify feedback plumbing, not biological performance."""
import json
from collections import deque
from threading import Lock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from flybrain.feedback import HumanFeedback
from flybrain.readout import OnlineLearner, Policy
from flybrain.server import Experiment


class HumanFeedbackTests(unittest.TestCase):
    def test_positive_encourages_and_negative_discourages_the_chosen_move(self):
        for value in (1, -1, .25, -.25):
            with self.subTest(value=value):
                learner = OnlineLearner(2)
                counts = torch.tensor([[5.], [2.]])
                actions, before = learner.act(counts)
                feedback = HumanFeedback()
                feedback.remember(1, learner, [True])
                feedback.apply({"feedback": value, "fly": 0, "move": 1}, learner)
                after = learner.logits(counts).softmax(1)
                self.assertGreater(float((after[0, actions[0]] - before[0, actions[0]]) * value), 0)
                self.assertEqual(learner.moves, 0)
                self.assertEqual(feedback.last["status"], "applied")

    def test_delayed_feedback_uses_displayed_inputs_not_next_action(self):
        learner = OnlineLearner(2)
        feedback = HumanFeedback()
        learner.act(torch.tensor([[4.], [0.]]))
        feedback.remember(10, learner, [True])
        learner.act(torch.tensor([[0.], [7.]]))
        feedback.remember(11, learner, [True])
        feedback.apply({"feedback": 1, "move": 10}, learner)
        self.assertGreater(float(learner.weight[:, 0].abs().sum()), 0)
        self.assertEqual(float(learner.weight[:, 1].abs().sum()), 0)
        self.assertEqual(feedback.last["move"], 10)

    def test_selected_fly_and_all_flies_exclude_respawns(self):
        for target, expected in ((0, [1., 0., 0.]), (None, [1., 1., 0.])):
            with self.subTest(target=target):
                learner = OnlineLearner(3)
                learner.act(torch.eye(3) * 5)
                reference = OnlineLearner(3)
                reference.learn(torch.tensor(expected), experience=learner.last, count_moves=False)
                feedback = HumanFeedback()
                feedback.remember(1, learner, [True, True, False])
                feedback.apply({"feedback": 1, "fly": target}, learner)
                torch.testing.assert_close(learner.weight, reference.weight)
                torch.testing.assert_close(learner.bias, reference.bias)
                self.assertEqual(feedback.positive, 1)

    def test_invalid_and_expired_feedback_cannot_corrupt_weights(self):
        learner = OnlineLearner(2)
        learner.act(torch.ones(2, 1))
        feedback = HumanFeedback(capacity=1)
        feedback.remember(1, learner, [True])
        feedback.remember(2, learner, [True])
        invalid = [{"feedback": value} for value in (None, True, "1", 0, 2, -2, float("nan"), float("inf"), 10**400)]
        invalid += [{"feedback": 1, "fly": fly} for fly in (-1, 1, "0", True, [])]
        invalid += [{"feedback": 1, "move": move} for move in (1, 999, None, [], True)]
        for message in invalid:
            with self.subTest(message=message):
                feedback.apply(message, learner)
                self.assertEqual(feedback.last["status"], "rejected")
                self.assertEqual(float(learner.weight.abs().sum()), 0)
                json.dumps(feedback.state(), allow_nan=False)
        feedback.clear()
        feedback.apply({"feedback": 1, "move": 2}, learner)
        self.assertEqual(feedback.last["status"], "rejected")
        feedback.apply({"feedback": 1}, None)
        self.assertEqual(feedback.last["status"], "rejected")
        self.assertEqual(feedback.positive + feedback.negative, 0)

    def test_no_feedback_for_a_respawn_frame(self):
        learner = OnlineLearner(2)
        learner.act(torch.ones(2, 1))
        feedback = HumanFeedback()
        feedback.remember(1, learner, [False])
        feedback.apply({"feedback": -1}, learner)
        self.assertEqual(feedback.last["status"], "rejected")
        self.assertEqual(float(learner.weight.abs().sum()), 0)

    def test_feedback_cannot_train_a_replacement_readout(self):
        original = OnlineLearner(2)
        original.act(torch.ones(2, 1))
        feedback = HumanFeedback()
        feedback.remember(1, original, [True])
        replacement = OnlineLearner(2)
        feedback.apply({"feedback": 1, "move": 1}, replacement)
        self.assertEqual(feedback.last["status"], "rejected")
        self.assertEqual(float(replacement.weight.abs().sum()), 0)
        self.assertEqual(float(original.weight.abs().sum()), 0)

    def test_retrain_copies_the_existing_readout(self):
        source = Policy(torch.ones(3, 2), torch.tensor([0., .1, .2]))
        learner = OnlineLearner.from_policy(source)
        counts = torch.ones(2, 1)
        torch.testing.assert_close(source.logits(counts), learner.logits(counts))
        learner.act(counts)
        learner.learn(torch.ones(1))
        torch.testing.assert_close(source.weight, torch.ones(3, 2))
        torch.testing.assert_close(source.bias, torch.tensor([0., .1, .2]))
        self.assertFalse(torch.equal(source.weight, learner.weight))


def small_experiment():
    """Use real arenas/readouts, with explicit synthetic spikes instead of connectome data."""
    experiment = Experiment.__new__(Experiment)
    experiment.device = torch.device("cpu")
    experiment.brains = {"real": Mock()}
    experiment.brains["real"].run.side_effect = lambda *args: torch.tensor([[4.], [0.]])
    experiment.policies = {("learning", "real"): OnlineLearner(2),
                           ("trained", "real"): Policy(torch.ones(3, 2), torch.zeros(3))}
    experiment.wiring, experiment.policy_name = "real", "learning"
    experiment.paused, experiment.override, experiment.clock = False, None, 0.
    experiment.sensor, experiment.sensor_seen = {}, 0.
    experiment.feedback, experiment.move = HumanFeedback(), 0
    experiment.inbox = []
    experiment.human_moves, experiment.human_move_lock = deque(), Lock()
    experiment.stim_index = torch.tensor([0])
    experiment.readout_index = experiment.visible = torch.arange(2)
    experiment.visible_ids = np.array([101, 102])
    experiment.channels = SimpleNamespace(levels=lambda levels: levels, readout_body_ids=np.array([101, 102]))
    experiment.connectome = SimpleNamespace(n=2)
    experiment.steer = {}
    experiment.set_layout("solo")
    return experiment


class ExperimentFeedbackTests(unittest.TestCase):
    def test_tick_applies_human_feedback_before_choosing_next_move(self):
        experiment = small_experiment()
        frame = experiment.tick()
        learner = experiment.policy()
        before = learner.weight.clone()
        # The next decision activates another feature. Only human feedback can now change feature 0.
        experiment.brains["real"].run.side_effect = lambda *args: torch.tensor([[0.], [4.]])
        experiment.inbox.append({"feedback": 1, "move": frame["move"], "fly": 0})
        updated = experiment.tick()
        self.assertFalse(torch.equal(before[:, 0], learner.weight[:, 0]))
        self.assertEqual(updated["learning"]["feedback"]["last"]["move"], frame["move"])
        self.assertEqual(updated["learning"]["feedback"]["positive"], 1)
        self.assertEqual(updated["learning"]["moves"], 2)
        json.dumps(updated, allow_nan=False)

    def test_layout_policy_wiring_and_reset_invalidate_decisions(self):
        for command in ({"layout": "swarm"}, {"policy": "trained"}, {"wiring": "shuffled"},
                        {"learning": "reset"}, {"learning": "pretrained"}):
            with self.subTest(command=command):
                experiment = small_experiment()
                experiment.tick()
                experiment.handle(command)
                self.assertFalse(experiment.feedback.decisions)

    def test_manual_senses_do_not_train_or_capture_game_decisions(self):
        experiment = small_experiment()
        experiment.tick()
        learner = experiment.policy()
        before, moves = learner.weight.clone(), learner.moves
        experiment.handle({"stimulate": {"food_L": 1}})
        frame = experiment.tick()
        torch.testing.assert_close(learner.weight, before)
        self.assertEqual(learner.moves, moves)
        self.assertFalse(experiment.feedback.decisions)
        self.assertFalse(frame["flies"][0]["feedbackEligible"])

    def test_retrain_command_starts_learning_without_editing_saved_policy(self):
        experiment = small_experiment()
        original = experiment.policies[("trained", "real")]
        experiment.handle({"learning": "pretrained"})
        self.assertEqual(experiment.policy_name, "learning")
        torch.testing.assert_close(experiment.policy().weight, original.weight)
        experiment.tick()
        torch.testing.assert_close(original.weight, torch.ones(3, 2))


if __name__ == "__main__":
    unittest.main()

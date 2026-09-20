"""Verify event choices reach sensory drive without relabeling game or learning rewards."""
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from flybrain.event_stimuli import EventStimuli
from test_feedback import small_experiment


class EventStimuliTests(unittest.TestCase):
    def test_defaults_and_partial_changes(self):
        settings = EventStimuli()
        self.assertEqual(settings.state(), {"food": "positive", "death": "negative"})
        settings.update({"food": "negative"})
        self.assertEqual(settings.cue(1), "pain")
        self.assertEqual(settings.cue(-1), "pain")
        for reward in (0, .1, -.1):
            self.assertIsNone(settings.cue(reward))

    def test_invalid_updates_are_atomic(self):
        for invalid in (None, [], {}, {"food": True}, {"food": []}, {"food": "unknown"},
                        {"food": "none", "unexpected": "positive"}):
            settings = EventStimuli()
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                settings.update(invalid)
            self.assertEqual(settings.state(), {"food": "positive", "death": "negative"})


class EventStimuliIntegrationTests(unittest.TestCase):
    def experiment(self, event, choice):
        experiment = small_experiment()
        experiment.event_matrix = torch.eye(2)
        experiment.all_stim_index = torch.arange(7)  # five game channels, two synthetic event receptors
        experiment.handle({"eventStimuli": {event: choice}})
        # Deterministic straight action, retaining the actual OnlineLearner experience/reward flow.
        learner = experiment.policy()
        learner.bias.copy_(torch.tensor([-100., 100., -100.]))
        learner.learn = Mock(wraps=learner.learn)
        arena = experiment.arenas[0]
        snake = arena.snakes[0]
        snake.heading = 1
        snake.cells = [(3, 3), (2, 3), (1, 3)] if event == "food" else [(11, 3), (10, 3), (9, 3)]
        arena.foods = [(4, 3)]
        return experiment

    def test_each_choice_drives_the_selected_receptors_and_preserves_rewards(self):
        for event, reward in (("food", 1), ("death", -1)):
            for choice, cue, drive in (("positive", "taste", [1., 0.]), ("negative", "pain", [0., 1.]), ("none", None, [0., 0.])):
                with self.subTest(event=event, choice=choice):
                    experiment = self.experiment(event, choice)
                    frame = experiment.tick()
                    self.assertEqual(frame["flies"][0]["reward"], reward)
                    self.assertEqual(frame["eventStimuli"][event], choice)
                    torch.testing.assert_close(experiment.policy().learn.call_args.args[0], torch.tensor([float(reward)]))
                    snake = experiment.arenas[0].snakes[0]
                    self.assertEqual(snake.score, 1 if event == "food" else 0)
                    self.assertEqual(snake.alive, event == "food")
                    next_frame = experiment.tick()
                    self.assertEqual(next_frame["flies"][0]["event"], cue)
                    torch.testing.assert_close(experiment.brains["real"].run.call_args.args[2][-2:, 0], torch.tensor(drive))
                    reset_mask = experiment.brains["real"].reset_brains.call_args.args[0]
                    self.assertEqual(bool(reset_mask[0]), choice == "negative")

    def test_changed_choice_cancels_pending_stimulus_without_resetting_learning(self):
        experiment = self.experiment("food", "positive")
        experiment.tick()
        self.assertTrue(experiment.pending_events.any())
        learner, move, feedback = experiment.policy(), experiment.move, experiment.feedback
        weights = learner.weight.clone()
        experiment.handle({"eventStimuli": {"food": "none"}})
        self.assertFalse(experiment.pending_events.any())
        self.assertIs(experiment.policy(), learner)
        self.assertIs(experiment.feedback, feedback)
        self.assertEqual(experiment.move, move)
        torch.testing.assert_close(learner.weight, weights)
        self.assertIsNone(experiment.tick()["flies"][0]["event"])

    def test_invalid_command_reports_error_and_keeps_pending_cue(self):
        experiment = self.experiment("food", "positive")
        experiment.tick()
        pending = experiment.pending_events.copy()
        experiment.handle({"eventStimuli": {"food": "none", "death": "bogus"}})
        self.assertIsNotNone(experiment.event_stimulus_error)
        np.testing.assert_array_equal(experiment.pending_events, pending)
        self.assertEqual(experiment.event_stimuli.state()["food"], "positive")
        experiment.handle({"eventStimuli": {"food": "none"}})
        self.assertIsNone(experiment.event_stimulus_error)

    def test_legacy_master_switch_preserves_choices_and_clears_pending_cues(self):
        experiment = self.experiment("food", "negative")
        experiment.tick()
        experiment.handle({"events": False})
        self.assertFalse(experiment.pending_events.any())
        self.assertEqual(experiment.tick()["eventStimuli"], {"food": "none", "death": "none"})
        experiment.handle({"events": True})
        self.assertEqual(experiment.event_stimuli.state()["food"], "negative")

    def test_death_animation_uses_selected_cue_without_an_extra_game_move(self):
        for choice, cue in (("positive", "taste"), ("negative", "pain"), ("none", None)):
            with self.subTest(choice=choice):
                experiment = self.experiment("death", choice)
                frame = experiment.tick()
                felt = experiment.event_frame(frame)
                if cue is None:
                    self.assertIsNone(felt)
                else:
                    self.assertTrue(felt["eventOnly"])
                    self.assertEqual(felt["flies"][0]["event"], cue)
                    self.assertEqual(felt["arenas"], frame["arenas"])
                    self.assertEqual(felt["move"], frame["move"])
                self.assertFalse(experiment.pending_events.any())
                self.assertEqual(experiment.policy().learn.call_count, 1)


if __name__ == "__main__":
    unittest.main()

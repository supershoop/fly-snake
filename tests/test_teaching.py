"""Absolute D-pad teaching: heading mapping, crowd-shared weighting, and refusals.

Synthetic decisions only; these make no claim about biological performance.
"""
import asyncio
import unittest

import torch

from flybrain.audience import TEACH_BUDGET, AudienceFeedback
from flybrain.feedback import HumanFeedback
from flybrain.readout import OnlineLearner
from flybrain.snake import HEADING_NAMES, LEFT, RIGHT, STRAIGHT
from test_feedback import small_experiment


def decide(learner, features=2):
    """One recorded decision for a single fly."""
    learner.act(torch.ones(features, 1))


class DirectionMappingTests(unittest.TestCase):
    """An absolute press must become the correct relative turn for every heading."""

    def setUp(self):
        self.learner = OnlineLearner(2)
        decide(self.learner)

    def teach(self, heading_name, pressed):
        feedback = HumanFeedback()
        feedback.remember(1, self.learner, [True], [HEADING_NAMES[heading_name]])
        # Capture the turn handed to the readout without changing any weights.
        seen = {}
        original = self.learner.teach
        def spy(desired, weights, **kwargs):
            seen["turn"] = int(desired[0])
            seen["weight"] = float(weights[0])
            return original(desired, weights, **kwargs)
        self.learner.teach = spy
        receipt = feedback.teach({"direction": pressed, "fly": 0, "move": 1}, self.learner, 0.5)
        self.learner.teach = original
        return receipt, seen

    def test_every_heading_maps_to_the_right_turn(self):
        # facing, pressed -> relative turn the fly should have taken
        cases = [
            ("up", "up", STRAIGHT), ("up", "right", RIGHT), ("up", "left", LEFT),
            ("right", "right", STRAIGHT), ("right", "down", RIGHT), ("right", "up", LEFT),
            ("down", "down", STRAIGHT), ("down", "left", RIGHT), ("down", "right", LEFT),
            ("left", "left", STRAIGHT), ("left", "up", RIGHT), ("left", "down", LEFT),
        ]
        for facing, pressed, expected in cases:
            with self.subTest(facing=facing, pressed=pressed):
                receipt, seen = self.teach(facing, pressed)
                self.assertEqual(receipt["status"], "applied")
                self.assertEqual(seen["turn"], expected)

    def test_reversing_is_refused_for_every_heading(self):
        for facing, backwards in (("up", "down"), ("right", "left"), ("down", "up"), ("left", "right")):
            with self.subTest(facing=facing):
                before = self.learner.weight.clone()
                receipt, _ = self.teach(facing, backwards)
                self.assertEqual(receipt["status"], "rejected")
                self.assertIn("cannot turn back", receipt["reason"])
                torch.testing.assert_close(before, self.learner.weight)

    def test_teaching_the_chosen_action_agrees_with_a_positive_reward(self):
        """teach() toward what the fly did should move weights like a reward for it."""
        _, _, action = self.learner.last
        taught, rewarded = OnlineLearner(2), OnlineLearner(2)
        for clone in (taught, rewarded):
            clone.weight, clone.bias = self.learner.weight.clone(), self.learner.bias.clone()
        taught.teach(action, torch.ones(1), experience=self.learner.last)
        rewarded.learn(torch.ones(1), experience=self.learner.last, count_moves=False)
        torch.testing.assert_close(taught.weight, rewarded.weight)

    def test_teaching_does_not_count_as_a_game_move(self):
        feedback = HumanFeedback()
        feedback.remember(1, self.learner, [True], [HEADING_NAMES["up"]])
        before = self.learner.moves
        feedback.teach({"direction": "left", "fly": 0, "move": 1}, self.learner, 0.5)
        self.assertEqual(self.learner.moves, before)

    def test_unknown_direction_missing_learner_and_stale_move_are_refused(self):
        feedback = HumanFeedback()
        feedback.remember(1, self.learner, [True], [HEADING_NAMES["up"]])
        for message, learner, weight in (
            ({"direction": "sideways", "fly": 0, "move": 1}, self.learner, 0.5),
            ({"direction": None, "fly": 0, "move": 1}, self.learner, 0.5),
            ({"direction": "left", "fly": 0, "move": 1}, None, 0.5),
            ({"direction": "left", "fly": 0, "move": 999}, self.learner, 0.5),
            ({"direction": "left", "fly": 7, "move": 1}, self.learner, 0.5),
            ({"direction": "left", "fly": 0, "move": 1}, self.learner, 0.0),
            ({"direction": "left", "fly": 0, "move": 1}, self.learner, float("nan")),
        ):
            with self.subTest(message=message, weight=weight):
                before = self.learner.weight.clone()
                self.assertEqual(feedback.teach(message, learner, weight)["status"], "rejected")
                torch.testing.assert_close(before, self.learner.weight)

    def test_a_move_without_headings_is_refused_rather_than_guessed(self):
        feedback = HumanFeedback()
        feedback.remember(1, self.learner, [True])  # older decision, no headings recorded
        before = self.learner.weight.clone()
        receipt = feedback.teach({"direction": "left", "fly": 0, "move": 1}, self.learner, 0.5)
        self.assertEqual(receipt["status"], "rejected")
        torch.testing.assert_close(before, self.learner.weight)


class CrowdWeightTests(unittest.TestCase):
    """Each vote is worth base/N, and one move cannot exceed the shared budget."""

    def setUp(self):
        self.hub = AudienceFeedback()

    def test_weight_is_inversely_proportional_to_connected_phones(self):
        for people in (1, 2, 4, 50):
            self.hub.clients = {f"phone-{i}" for i in range(people)}
            self.hub.spent.clear()
            self.hub.voted.clear()
            self.assertAlmostEqual(self.hub.vote_weight(1, "phone-0"), TEACH_BUDGET / people)

    def test_a_full_room_together_spends_the_same_budget_as_one_voter(self):
        self.hub.clients = {f"phone-{i}" for i in range(10)}
        total = 0.0
        for i in range(10):
            weight = self.hub.vote_weight(5, f"phone-{i}")
            total += weight
            self.hub.charge(5, f"phone-{i}", weight)
        self.assertAlmostEqual(total, TEACH_BUDGET)
        # An eleventh latecomer finds the move's influence already spent.
        self.assertEqual(self.hub.vote_weight(5, "latecomer"), 0.0)

    def test_one_vote_per_phone_per_move(self):
        self.hub.clients = {"a", "b"}
        first = self.hub.vote_weight(3, "a")
        self.hub.charge(3, "a", first)
        self.assertGreater(first, 0)
        self.assertEqual(self.hub.vote_weight(3, "a"), 0.0)
        self.assertGreater(self.hub.vote_weight(3, "b"), 0.0)   # someone else still may
        self.assertGreater(self.hub.vote_weight(4, "a"), 0.0)   # and "a" may teach the next move

    def test_an_empty_room_still_divides_by_one(self):
        self.hub.clients = set()
        self.assertEqual(self.hub.participants(), 1)
        self.assertAlmostEqual(self.hub.vote_weight(1, "ghost"), TEACH_BUDGET)

    def test_bookkeeping_does_not_grow_without_bound(self):
        self.hub.clients = {"a"}
        for move in range(400):
            self.hub.charge(move, "a", 0.01)
        self.assertLessEqual(len(self.hub.spent), 300)
        self.assertLessEqual(len(self.hub.voted), 300)


class TeachingThroughTheHubTests(unittest.IsolatedAsyncioTestCase):
    """The queue path phones actually use, against a real Experiment.tick()."""

    async def asyncSetUp(self):
        self.experiment = small_experiment()
        self.frame = self.experiment.tick()
        self.hub = AudienceFeedback()
        self.hub.clients = {"phone-1", "phone-2"}

    def command(self, direction="up", **extra):
        # Snakes spawn facing right, so "up" is a legal quarter turn and "left" would reverse.
        return {"id": "phone-1", "direction": direction, "fly": 0, "move": self.frame["move"], **extra}

    async def apply(self, command):
        result = asyncio.create_task(self.hub.submit(command, self.experiment))
        await asyncio.sleep(0)
        await asyncio.to_thread(self.hub.apply, self.experiment)
        return await result

    async def test_a_direction_press_trains_the_shared_readout(self):
        before = self.experiment.policy().weight.clone()
        receipt = await self.apply(self.command())
        self.assertEqual(receipt["status"], "applied")
        self.assertEqual(receipt["direction"], "up")
        # Two phones connected, so this vote carries half of the move's influence.
        self.assertAlmostEqual(receipt["weight"], TEACH_BUDGET / 2)
        self.assertFalse(torch.equal(before, self.experiment.policy().weight))

    async def test_the_same_phone_cannot_teach_one_move_twice(self):
        self.assertEqual((await self.apply(self.command()))["status"], "applied")
        repeat = await self.apply(self.command())
        self.assertEqual(repeat["status"], "rejected")
        self.assertIn("already taught", repeat["reason"])

    async def test_opposing_votes_cancel_rather_than_compound(self):
        """Disagreement should leave the readout near where it started."""
        start = self.experiment.policy().weight.clone()
        await self.apply(self.command("up"))
        await self.apply(self.command("down", id="phone-2"))
        drift = (self.experiment.policy().weight - start).abs().max()
        one_sided = small_experiment()
        one_sided.tick()
        hub = AudienceFeedback()
        hub.clients = {"phone-1", "phone-2"}
        base = one_sided.policy().weight.clone()
        for voter in ("phone-1", "phone-2"):
            task = asyncio.create_task(hub.submit(
                {"id": voter, "direction": "up", "fly": 0, "move": one_sided.move}, one_sided))
            await asyncio.sleep(0)
            await asyncio.to_thread(hub.apply, one_sided)
            await task
        agreed = (one_sided.policy().weight - base).abs().max()
        self.assertLess(drift, agreed)

    async def test_paused_manual_and_non_learning_modes_refuse_teaching(self):
        self.experiment.paused = True
        self.assertEqual((await self.apply(self.command()))["status"], "rejected")
        self.experiment.paused = False
        self.experiment.override = {"food_L": 1}
        self.assertEqual((await self.apply(self.command()))["status"], "rejected")
        self.experiment.override = None
        self.experiment.policy_name = "trained"
        self.assertEqual((await self.apply(self.command()))["status"], "rejected")

    async def test_reversing_is_refused_through_the_live_hub(self):
        """Snakes spawn facing right, so a press to the left is backwards."""
        before = self.experiment.policy().weight.clone()
        receipt = await self.apply(self.command("left"))
        self.assertEqual(receipt["status"], "rejected")
        self.assertIn("cannot turn back", receipt["reason"])
        torch.testing.assert_close(before, self.experiment.policy().weight)

    async def test_malformed_direction_messages_are_refused(self):
        for command in (self.command(direction="diagonal"), self.command(direction=2),
                        {"id": "x", "direction": "left", "fly": 0},
                        self.command(move=999999)):
            with self.subTest(command=command):
                before = self.experiment.policy().weight.clone()
                self.assertEqual((await self.apply(command))["status"], "rejected")
                torch.testing.assert_close(before, self.experiment.policy().weight)

    async def test_published_heading_is_the_one_votes_are_judged_against(self):
        """The board turns as the fly moves, so phones must be told the decision's facing.

        Publishing the post-move heading would grey out the wrong D-pad arrow and let a
        genuine reversal through.
        """
        for _ in range(6):
            frame = self.experiment.tick()
            fly = frame["flies"][0]
            decision = self.experiment.feedback.decisions.get(frame["move"])
            self.assertIsNotNone(decision)
            self.assertEqual(fly["heading"], decision.headings[0])
            # Refusing the published heading's opposite must agree with the brain's own check.
            backwards = (fly["heading"] + 2) % 4
            name = next(key for key, value in HEADING_NAMES.items() if value == backwards)
            receipt = self.experiment.feedback.teach(
                {"direction": name, "fly": 0, "move": frame["move"]}, self.experiment.policy(), 0.5)
            self.assertEqual(receipt["status"], "rejected", f"{name} should be backwards here")

    async def test_counters_and_published_crowd_state_follow_the_votes(self):
        """What the phones and dashboard display must match what was actually applied."""
        self.hub.clients = {"phone-1", "phone-2", "phone-3", "phone-4"}
        applied = 0
        for voter in ("phone-1", "phone-2", "phone-3"):
            receipt = await self.apply(self.command(id=voter))
            applied += receipt["status"] == "applied"
        self.assertEqual(applied, 3)
        state = self.experiment.feedback.state()
        self.assertEqual(state["taught"], 3)
        self.assertEqual(state["directions"], {"up": 3})
        # The frame the phones receive reports the same thing, with each vote's share.
        await self.hub.publish(self.experiment.tick())
        crowd = self.hub.latest["audience"]
        self.assertEqual(crowd["participants"], 4)
        self.assertEqual(crowd["taught"], 3)
        self.assertEqual(crowd["directions"], {"up": 3})
        self.assertAlmostEqual(crowd["share"], TEACH_BUDGET / 4)

    async def test_the_automatic_reward_signal_still_runs_alongside_teaching(self):
        """Teaching biases the readout; the game's own learning is untouched."""
        moves_before = self.experiment.policy().moves
        await self.apply(self.command())
        self.assertEqual(self.experiment.policy().moves, moves_before)  # a vote is not a move
        self.experiment.tick()
        self.assertGreater(self.experiment.policy().moves, moves_before)  # the game keeps learning


if __name__ == "__main__":
    unittest.main()

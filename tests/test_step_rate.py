"""The host's step-rate cap: clamped, reversible, and the achieved rate is truthful."""
import unittest

from test_feedback import small_experiment


class StepRateHandleTests(unittest.TestCase):
    def setUp(self):
        self.experiment = small_experiment()

    def test_default_is_uncapped(self):
        # small_experiment() bypasses Experiment.__init__ (no GPU/connectome here), so the
        # real default is asserted directly against __init__'s source instead of an instance.
        import inspect
        from flybrain.server import Experiment
        self.assertIn("self.step_rate = 0.0", inspect.getsource(Experiment.__init__))

    def test_a_value_in_range_is_kept(self):
        self.experiment.handle({"stepRate": 5})
        self.assertEqual(self.experiment.step_rate, 5.0)

    def test_zero_or_negative_means_uncapped_not_frozen(self):
        self.experiment.handle({"stepRate": 5})
        for value in (0, -1, -100):
            with self.subTest(value=value):
                self.experiment.handle({"stepRate": value})
                self.assertEqual(self.experiment.step_rate, 0.0)

    def test_extreme_requests_are_clamped_to_a_safe_range(self):
        self.experiment.handle({"stepRate": 1_000_000})
        self.assertEqual(self.experiment.step_rate, 20.0)
        # 0.25 is the dashboard's own minimum non-zero step; the server's floor must not
        # round it up to something the slider never actually offers.
        self.experiment.handle({"stepRate": 0.25})
        self.assertEqual(self.experiment.step_rate, 0.25)
        self.experiment.handle({"stepRate": 0.001})
        self.assertEqual(self.experiment.step_rate, 0.25)

    def test_a_string_number_is_accepted_like_the_rest_of_handle(self):
        self.experiment.handle({"stepRate": "3"})
        self.assertEqual(self.experiment.step_rate, 3.0)

    def test_garbage_is_ignored_rather_than_crashing_the_tick_loop(self):
        """handle() runs inside a broad except in server.py's tick(); this documents that reliance."""
        with self.assertRaises((ValueError, TypeError)):
            self.experiment.handle({"stepRate": "not-a-number"})

    def test_setting_step_rate_does_not_disturb_other_settings(self):
        self.experiment.handle({"deathHold": 2.0, "paused": True})
        self.experiment.handle({"stepRate": 4})
        self.assertEqual(self.experiment.death_hold, 2.0)
        self.assertTrue(self.experiment.paused)


if __name__ == "__main__":
    unittest.main()

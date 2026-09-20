from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from flybrain.readout import HardwiredPolicy
from flybrain.synaptic import SynapticSites
from test_feedback import small_experiment
from test_synaptic import anatomy


class SynapticServerTests(unittest.TestCase):
    def experiment(self):
        experiment = small_experiment()
        experiment.connectome = anatomy()
        experiment.channels.steer_sign = torch.tensor([0., 0., 1., -1., 1., 0.])
        return experiment

    def test_switch_uses_brain_weights_and_fixed_policy_and_restores_original(self):
        experiment = self.experiment()
        sites = SynapticSites.from_connectome(experiment.connectome)
        original = experiment.brains["real"]
        experiment.lesions = [["DNa01"]]
        experiment.override, experiment.sensor, experiment.paused = {"food_L": 1}, {"danger_L": 1}, True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "brain.npz"
            sites.save(path, np.full(len(sites.labels), .2), generation=8)
            with patch("flybrain.server.SYNAPTIC_MODEL", path):
                experiment.handle({"synaptic": True, "policy": "trained", "wiring": "shuffled"})
                self.assertEqual(experiment.policy_name, "hardwired")
                self.assertEqual(experiment.wiring, "real")
                self.assertIsInstance(experiment.policy(), HardwiredPolicy)
                self.assertIsNot(experiment.brain(), original)
                self.assertIs(experiment.brains["real"], original)
                self.assertEqual(experiment.synaptic_metadata["generation"], 8)
                self.assertEqual(experiment.lesions, [["DNa01"]])
                self.assertIsNone(experiment.override)
                self.assertFalse(experiment.sensor)
                self.assertFalse(experiment.paused)
                experiment.handle({"policy": "trained"})
                self.assertIsNone(experiment.synaptic_parameters)
                self.assertIs(experiment.brain(), original)

    def test_invalid_model_does_not_change_active_experiment(self):
        experiment = self.experiment()
        previous = experiment.policy_name
        with patch("flybrain.server.SYNAPTIC_MODEL", Path("nonexistent-brain-checkpoint.npz")):
            experiment.handle({"synaptic": True, "policy": "hardwired"})
        self.assertEqual(experiment.policy_name, previous)
        self.assertIsNone(experiment.synaptic_parameters)
        self.assertIsNotNone(experiment.synaptic_error)

    def test_readout_learning_and_shuffle_exit_synaptic_mode(self):
        for message in ({"learning": "reset"}, {"learning": "pretrained"}, {"wiring": "shuffled"}, {"policy": "instinct"}):
            experiment = self.experiment()
            experiment.synaptic_parameters = np.array([.2])
            experiment.synaptic_metadata = {"generation": 1}
            experiment.handle(message)
            self.assertIsNone(experiment.synaptic_parameters)

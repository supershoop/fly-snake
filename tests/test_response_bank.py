from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from flybrain.response_bank import response_bank, BANK_VERSION
from flybrain.channels import Channels


class ResponseBankTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "bank.npz"
        self.channels = SimpleNamespace(readout_body_ids=np.array([111, 222]), readout_index=torch.tensor([1, 2]))
        self.saved = dict(version=BANK_VERSION, window=100., dt=.5, seed=42, shuffled=False, weight_scale=.4,
                          wiring_seed=0, readout_body_ids=np.array([111, 222]), counts=np.ones((24, 8, 2), dtype=np.float32))

    def test_matching_bank_does_not_run_simulator(self):
        np.savez(self.path, **self.saved)
        with patch("flybrain.response_bank.Brain", side_effect=AssertionError("Unexpected simulation")):
            counts = response_bank(None, self.channels, self.path, trials=8)
        self.assertEqual(tuple(counts.shape), (24, 8, 2))

    def test_mismatched_bank_is_never_reused(self):
        for key, value in (("window", 50.), ("seed", 43), ("shuffled", True),
                           ("readout_body_ids", [222, 111]), ("counts", np.ones((24, 16, 2)))):
            with self.subTest(key=key):
                np.savez(self.path, **{**self.saved, key: value})
                with patch("flybrain.response_bank.Brain", side_effect=RuntimeError("rebuilding")):
                    with self.assertRaisesRegex(RuntimeError, "rebuilding"):
                        response_bank(None, self.channels, self.path, trials=8, progress=lambda _: None)

    def test_legacy_bank_requires_rebuild(self):
        np.savez(self.path, counts=self.saved["counts"])
        with patch("flybrain.response_bank.Brain", side_effect=RuntimeError("rebuilding")):
            with self.assertRaisesRegex(RuntimeError, "rebuilding"):
                response_bank(None, self.channels, self.path, trials=8, progress=lambda _: None)

    def test_generation_keeps_non_multiple_of_eight_trials(self):
        # A tiny synthetic graph exercises the simulator/cache plumbing only;
        # these responses are never used to train or evaluate the shipped model.
        connectome = SimpleNamespace(n=2, pre=np.array([0]), post=np.array([1]), weight=np.array([1.], dtype=np.float32))
        channels = Channels(torch.tensor([0]), torch.tensor([[1., 0., 0., 0., 0.]]), torch.tensor([0, 1]),
                            np.array([111, 222]), torch.zeros(2))
        counts = response_bank(connectome, channels, self.path, trials=9, progress=lambda _: None)
        self.assertEqual(tuple(counts.shape), (24, 9, 2))
        self.assertTrue(torch.isfinite(counts).all())
        self.assertGreater(float(counts.sum()), 0.)
        cached = response_bank(connectome, channels, self.path, trials=9)
        self.assertTrue(torch.equal(counts, cached))


if __name__ == "__main__":
    unittest.main()

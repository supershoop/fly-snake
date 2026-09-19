from types import SimpleNamespace
import unittest

import numpy as np
import torch

from flybrain.brain import Brain
from flybrain.cpu_synapses import CPUSynapses


class CPUSynapseTests(unittest.TestCase):
    def test_sparse_and_dense_activity_match_full_multiply(self):
        rng = np.random.default_rng(9)
        n = 256
        index = torch.tensor(rng.integers(n, size=(2, 2000)))
        weights = torch.sparse_coo_tensor(index, torch.tensor(rng.normal(size=2000), dtype=torch.float32), (n, n)).coalesce().to_sparse_csr()
        propagate = CPUSynapses(weights)
        for batch in (1, 3, 16):
            for activity in (0., .02, .5, 1.):
                with self.subTest(batch=batch, activity=activity):
                    spikes = torch.tensor(rng.random((n, batch)) < activity, dtype=torch.float32)
                    torch.testing.assert_close(propagate(spikes), torch.sparse.mm(weights, spikes), rtol=1e-5, atol=1e-5)

    def test_continuous_simulation_lesions_and_resize_match_reference(self):
        rng = np.random.default_rng(12)
        connectome = SimpleNamespace(n=64, pre=rng.integers(64, size=500), post=rng.integers(64, size=500),
                                     weight=rng.uniform(-15, 25, size=500).astype(np.float32))
        reference = Brain(connectome, batch=1, device="cpu", seed=21, cpu_sparse=False)
        faster = Brain(connectome, batch=1, device="cpu", seed=21)
        stim = torch.arange(10)
        levels = torch.tensor(rng.uniform(size=(10, 1)), dtype=torch.float32)
        lesion = torch.zeros(64, 1, dtype=torch.bool)
        lesion[:3, 0] = True
        for brain in (reference, faster):
            brain.set_lesion(lesion)
        for _ in range(4):
            self.assertTrue(torch.equal(reference.run(25, stim, levels), faster.run(25, stim, levels)))
            torch.testing.assert_close(faster.v, reference.v, rtol=1e-5, atol=1e-5)
        for brain in (reference, faster):
            brain.resize(3)
        self.assertTrue(torch.equal(reference.run(25, stim, levels.repeat(1, 3), stim),
                                    faster.run(25, stim, levels.repeat(1, 3), stim)))


if __name__ == "__main__":
    unittest.main()

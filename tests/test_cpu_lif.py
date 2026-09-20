from types import SimpleNamespace
import unittest

import numpy as np
import torch

try:
    import numba
except ImportError:
    raise unittest.SkipTest("Optional numba training dependency not installed")

from flybrain.brain import Brain


class CompiledLIFTests(unittest.TestCase):
    def test_continuous_full_state_matches_torch_with_lesions_and_delays(self):
        rng = np.random.default_rng(18)
        c = SimpleNamespace(n=100, pre=rng.integers(100, size=1500), post=rng.integers(100, size=1500),
                            weight=rng.uniform(-30, 40, size=1500).astype(np.float32))
        for dt in (.5, 1.):
            original = Brain(c, device="cpu", seed=101, dt=dt)
            compiled = Brain(c, device="cpu", seed=101, dt=dt, compiled=True)
            stimulus = torch.arange(25)
            for turn in range(12):
                lesion = torch.arange(100) < 5 if turn > 3 else None
                levels = torch.tensor(rng.uniform(size=(25, 1)), dtype=torch.float32)
                for brain in (original, compiled):
                    brain.set_lesion(lesion)
                if turn == 8:
                    original.reset()
                    compiled.reset()
                a = original.run(50, stimulus, levels)
                b = compiled.run(50, stimulus, levels)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                for name in ("v", "g", "refractory", "pending"):
                    torch.testing.assert_close(getattr(original, name), getattr(compiled, name), rtol=1e-5, atol=1e-5)
                self.assertEqual(original.cursor, compiled.cursor)
            torch.testing.assert_close(original.run(50), compiled.run(50), rtol=0, atol=0)

    def test_rejects_unsupported_batch(self):
        c = SimpleNamespace(n=1, pre=np.array([0]), post=np.array([0]), weight=np.array([1.], dtype=np.float32))
        with self.assertRaises(ValueError):
            Brain(c, batch=2, device="cpu", compiled=True)
        with self.assertRaises(ValueError):
            Brain(c, compiled=True, device="cpu").resize(2)

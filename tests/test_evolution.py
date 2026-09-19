import unittest

import numpy as np

try:
    import numba  # noqa: F401
except ImportError:
    raise unittest.SkipTest("Install requirements-training.txt for evolution tests")

from flybrain.evolution import Evaluator, actions_for, breed, feature_basis, features, weights_for


class EvolutionTests(unittest.TestCase):
    def test_exported_linear_readout_matches_feature_basis(self):
        rng = np.random.default_rng(77)
        counts = rng.poisson(4, (24, 16, 48)).astype(np.float32)
        basis = feature_basis(counts[:, :8])
        parameters = rng.normal(size=(3, 24))
        weight, bias = weights_for(parameters, basis)
        projected = features(counts[:, 8:], basis)
        expected = projected @ parameters.T
        actual = np.log1p(counts[:, 8:]) @ weight.T + bias
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(actual.argmax(-1), actions_for(parameters, projected))

    def test_mutation_preserves_elites_and_is_reproducible(self):
        elite = np.ones((3, 3, 24))
        first = breed(elite, 16, np.random.default_rng(99))
        second = breed(elite, 16, np.random.default_rng(99))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[:3], elite)
        np.testing.assert_array_equal(elite, np.ones((3, 3, 24)))
        self.assertTrue(np.isfinite(first).all())
        self.assertGreater(np.std(first[3:]), 0)

    def test_cache_deduplicates_but_invalidates_when_seeds_change(self):
        evaluator = Evaluator(2)
        self.addCleanup(evaluator.close)
        actions = np.ones((24, 4), dtype=int)
        evaluator.evaluate([actions], [100], 30, 99)  # compile before threaded duplicates
        results = evaluator.evaluate([actions, actions.copy()], [100], 30, 99)
        self.assertEqual(results[0], results[1])
        self.assertEqual(evaluator.evaluations, 1)
        self.assertEqual(evaluator.cache_hits, 2)
        evaluator.evaluate([actions], [101], 30, 99)
        self.assertEqual(evaluator.evaluations, 2)
        self.assertEqual(evaluator.games, 2)


if __name__ == "__main__":
    unittest.main()

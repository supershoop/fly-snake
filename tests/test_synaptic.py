import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from flybrain.brain import Brain
from flybrain.connectome import Connectome
from flybrain.cpu_synapses import CPUSynapses
from flybrain.synaptic import LIMIT, SynapticAdapter, SynapticSites, perturb


def anatomy():
    neurons = pd.DataFrame({"bodyId": np.arange(10, 16), "type": ["AOTU015", "PVLP141", "DNa02", "DNa02", "DNa01", "other"],
                            "side": ["L", "R", "L", "R", "L", "L"]})
    return Connectome(neurons, np.array([0, 0, 1, 1, 5, 3]), np.array([2, 3, 2, 4, 2, 5]),
                      np.array([40., 20., -30., 50., 10., 12.], dtype=np.float32))


class SynapticTests(unittest.TestCase):
    def test_only_existing_sites_change_and_both_cpu_caches_follow(self):
        c = anatomy()
        sites = SynapticSites.from_connectome(c)
        brain = Brain(c, device="cpu")
        adapter = SynapticAdapter(brain, sites)
        original = brain.weights.to_dense().clone()
        brain.cpu_synapses = CPUSynapses(brain.weights)
        for factor in (LIMIT, -LIMIT, 0.):
            adapter.apply(np.full(len(sites.labels), factor))
            dense = brain.weights.to_dense()
            expected = original.clone()
            expected[sites.post, sites.pre] *= np.exp(factor)
            torch.testing.assert_close(dense, expected)
            torch.testing.assert_close(torch.sign(dense), torch.sign(original))
            spikes = torch.ones(6, 1)
            torch.testing.assert_close(brain.cpu_synapses(spikes), dense @ spikes)
            np.testing.assert_allclose(brain.cpu_synapses.csc.toarray(), dense.numpy())
        torch.testing.assert_close(brain.weights.to_dense(), original, rtol=0, atol=0)

    def test_cache_created_after_first_update_and_new_cache_are_updated(self):
        c = anatomy()
        sites = SynapticSites.from_connectome(c)
        brain = Brain(c, device="cpu")
        adapter = SynapticAdapter(brain, sites)
        for value in (.5, -.5):
            adapter.apply(np.full(len(sites.labels), value))
            brain.cpu_synapses = CPUSynapses(brain.weights)
            adapter.apply(np.zeros(len(sites.labels)))
            np.testing.assert_allclose(brain.cpu_synapses.csc.toarray(), brain.weights.to_dense().numpy())

    def test_checkpoint_checks_anatomy_sites_and_decoder(self):
        sites = SynapticSites.from_connectome(anatomy())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "brain.npz"
            parameters = np.linspace(-1, 1, len(sites.labels))
            sites.save(path, parameters, generation=3)
            loaded, metadata = sites.load(path)
            np.testing.assert_array_equal(loaded, parameters)
            self.assertEqual(metadata["generation"], 3)
            with np.load(path) as source:
                changed = {k: source[k] for k in source.files}
            metadata["decoder"]["threshold"] = 0.
            changed["metadata"] = np.array(json.dumps(metadata))
            np.savez(path, **changed)
            with self.assertRaises(ValueError):
                sites.load(path)

    def test_invalid_weights_never_mutate_brain(self):
        c = anatomy()
        sites = SynapticSites.from_connectome(c)
        brain = Brain(c, device="cpu")
        adapter = SynapticAdapter(brain, sites)
        original = brain.weights.values().clone()
        for values in ([0.], np.full(len(sites.labels), np.nan), np.full(len(sites.labels), LIMIT + .1)):
            with self.assertRaises(ValueError):
                adapter.apply(values)
            torch.testing.assert_close(brain.weights.values(), original, rtol=0, atol=0)

    def test_proposals_reproducible_bounded_and_do_not_change_parent(self):
        parameters = np.ones(12)
        a = perturb(parameters, np.random.default_rng(9))
        b = perturb(parameters, np.random.default_rng(9))
        np.testing.assert_array_equal(a, b)
        self.assertTrue((np.abs(a) <= LIMIT).all())
        np.testing.assert_array_equal(parameters, np.ones(12))

    def test_rejects_a_different_simulation_or_shuffled_wiring(self):
        c = anatomy()
        sites = SynapticSites.from_connectome(c)
        for options in ({"dt": 1.}, {"shuffled": True}, {"weight_scale": .5}):
            with self.assertRaises(ValueError):
                SynapticAdapter(Brain(c, device="cpu", **options), sites)

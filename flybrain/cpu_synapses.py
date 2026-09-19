"""CPU propagation over outgoing connections of neurons that actually spiked.

This computes the same W @ spikes as the full sparse multiply. Inactive
presynaptic columns contribute zero, so they can be omitted without pruning
connections, changing weights, or approximating the brain's time steps.
"""
import numpy as np
from scipy.sparse import csr_matrix
import torch


class CPUSynapses:
    def __init__(self, weights):
        self.csr = csr_matrix((weights.values().numpy(), weights.col_indices().numpy(),
                               weights.crow_indices().numpy()), shape=tuple(weights.shape))
        self.csc = self.csr.tocsc()

    def __call__(self, spikes):
        values = spikes.numpy()
        active = np.flatnonzero(values[:, 0] if values.shape[1] == 1 else values.any(axis=1))
        if not len(active):
            return torch.zeros_like(spikes)
        # Dense firing can make column selection more expensive than the full
        # multiply. Both paths include every nonzero contribution.
        if len(active) > values.shape[0] // 8:
            result = self.csr @ values
        else:
            result = self.csc[:, active] @ values[active]
        return torch.from_numpy(result)

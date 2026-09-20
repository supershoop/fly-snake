"""Optional compiled single-brain CPU kernel, with the same LIF time steps.

Input events still come from Brain's torch generator. No neurons, synapses,
delays or time steps are omitted. Numba is an optional training dependency.
"""
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def advance(v, g, refractory, pending, cursor, indptr, targets, weights,
            stimulus, events, silenced, dt, refractory_steps):
    n = len(v)
    counts = np.zeros(n, dtype=np.float32)
    fired = np.empty(n, dtype=np.int64)
    membrane = np.float32(dt / 20.)
    decay = np.float32(dt / 5.)
    rest, threshold, strength = np.float32(-52.), np.float32(-45.), np.float32(.275 * 250)
    for step in range(len(events)):
        for neuron in range(n):
            g[neuron] += pending[cursor, neuron]
            pending[cursor, neuron] = 0.
        for index in range(len(stimulus)):
            if events[step, index]:
                g[stimulus[index]] += strength
        number = 0
        for neuron in range(n):
            active = refractory[neuron] == 0
            if active:
                v[neuron] += (g[neuron] - (v[neuron] - rest)) * membrane
            g[neuron] -= g[neuron] * decay
            spike = active and v[neuron] > threshold and not silenced[neuron]
            if spike:
                v[neuron] = rest
                refractory[neuron] = refractory_steps
                counts[neuron] += 1.
                fired[number] = neuron
                number += 1
            elif refractory[neuron] > 0:
                refractory[neuron] -= 1
        for index in range(number):
            pre = fired[index]
            for edge in range(indptr[pre], indptr[pre + 1]):
                pending[cursor, targets[edge]] += weights[edge]
        cursor = (cursor + 1) % len(pending)
    return counts, cursor


def run(brain, steps, stim_index, stim_level, record_index):
    import torch
    from .cpu_synapses import CPUSynapses
    if brain.cpu_synapses is None:
        brain.cpu_synapses = CPUSynapses(brain.weights)
    matrix = brain.cpu_synapses.csc
    if stim_index is None:
        stimulus = np.empty(0, dtype=np.int64)
        events = np.empty((steps, 0), dtype=np.bool_)
    else:
        stimulus = stim_index.cpu().numpy()
        probability = stim_level * (150. * brain.dt / 1000.)
        events = (torch.rand((steps, len(stimulus), 1), generator=brain.rng) < probability).numpy()[:, :, 0]
    silenced = (np.zeros(brain.n, dtype=np.bool_) if brain.silenced is None
                else brain.silenced.numpy()[:, 0])
    counts, brain.cursor = advance(
        brain.v.numpy()[:, 0], brain.g.numpy()[:, 0], brain.refractory.numpy()[:, 0],
        brain.pending.numpy()[:, :, 0], brain.cursor, matrix.indptr, matrix.indices, matrix.data,
        stimulus, events, silenced, brain.dt, brain.refractory_steps)
    result = torch.from_numpy(counts[:, None])
    return result if record_index is None else result[record_index]

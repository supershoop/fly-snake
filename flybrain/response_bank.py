"""Reproducible DN response banks that include activity carried between moves."""
import time

import numpy as np
import torch

from .brain import Brain, MALECNS_WEIGHT_SCALE
from .snake import ALL_STATES, encode_state

BANK_VERSION = 2


def response_bank(connectome, channels, path, *, trials=48, window=100., shuffled=False,
                  seed=42, device="cpu", rebuild=False, progress=print):
    if trials < 8:
        raise ValueError("At least 8 trials are required for separate train/held-out responses")
    metadata = dict(version=BANK_VERSION, window=window, dt=.5, seed=seed,
                    shuffled=shuffled, weight_scale=MALECNS_WEIGHT_SCALE, wiring_seed=0)
    if path.exists() and not rebuild:
        with np.load(path, allow_pickle=False) as saved:
            compatible = (all(key in saved and saved[key].item() == value for key, value in metadata.items())
                          and "readout_body_ids" in saved
                          and np.array_equal(saved["readout_body_ids"], channels.readout_body_ids)
                          and "counts" in saved and saved["counts"].shape == (len(ALL_STATES), trials, len(channels.readout_index)))
            if compatible:
                return torch.as_tensor(saved["counts"], device=device)
        progress("Response-bank settings changed; rebuilding instead of reusing incompatible counts.")
    brain = Brain(connectome, batch=len(ALL_STATES), shuffled=shuffled, seed=0, device=str(device))
    brain.rng.manual_seed(seed)  # Poisson noise changes; shuffled wiring matches the server's seed 0.
    levels = torch.as_tensor(np.stack([encode_state(state) for state in ALL_STATES]), device=device).T
    stim, readout = channels.stim_index.to(device), channels.readout_index.to(device)
    rng, runs, start = np.random.default_rng(seed), [], time.monotonic()
    for trial in range(trials):
        if trial % 4 == 0:
            brain.reset()
        # Each state has both fresh and carried-over responses, with unrelated
        # previous inputs. No synaptic weights are changed during these runs.
        order = rng.permutation(len(ALL_STATES))
        counts = brain.run(window, stim, channels.levels(levels[:, order]), readout)
        runs.append(counts.T[np.argsort(order)].cpu().numpy())
        progress(f"brain responses {trial+1}/{trials}: {time.monotonic()-start:.0f}s")
    counts = np.stack(runs, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, counts=counts, readout_body_ids=channels.readout_body_ids, **metadata)
    return torch.as_tensor(counts, device=device)

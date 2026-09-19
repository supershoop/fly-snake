"""Named neuron groups: what we stimulate (game -> senses) and what we read (brain -> steering).

Chosen with scripts/probe_channels.py. Left-side stimulation of:
  LC10  (small-object / pursuit visual neurons) -> ipsilateral DNa02 (steering DN) ~250 Hz vs 0 Hz: turn TOWARD
  LC4   (looming detectors) -> giant fiber DNp01 ~390 Hz and contralateral DNa01: escape, turn AWAY
  LPLC2 (looming detectors) -> giant fiber, both sides: used for "threat straight ahead"
"""
from dataclasses import dataclass

import numpy as np
import torch

from .connectome import Connectome

CHANNEL_NAMES = ["food_L", "food_R", "danger_L", "danger_R", "danger_ahead"]
CHANNEL_SOURCES = {  # channel -> (regex on annotation `type`, side or None for both)
    "food_L": (r"^LC10", "L"), "food_R": (r"^LC10", "R"),
    "danger_L": (r"^LC4$", "L"), "danger_R": (r"^LC4$", "R"),
    "danger_ahead": (r"^LPLC2$", None),
}
STEER_TYPES = ["DNa02", "DNa01"]  # hardwired (no-learning) control: turn toward the side that fires more


@dataclass
class Channels:
    stim_index: torch.Tensor     # int64 [S] simulator indices of every stimulated neuron
    stim_matrix: torch.Tensor    # float [S, C] membership of each stimulated neuron in each channel
    readout_index: torch.Tensor  # int64 [R] all descending neurons = policy features
    readout_body_ids: np.ndarray
    steer_sign: torch.Tensor     # float [R] +1 left steering DN, -1 right steering DN, else 0

    def levels(self, channel_levels: torch.Tensor) -> torch.Tensor:
        """[C, B] channel drive in 0..1 -> [S, B] per-neuron drive."""
        return self.stim_matrix.to(channel_levels.device) @ channel_levels


def build_channels(connectome: Connectome) -> Channels:
    neurons = connectome.neurons
    kind, side = neurons["type"].fillna(""), neurons["side"]
    masks = []
    for name in CHANNEL_NAMES:
        pattern, wanted_side = CHANNEL_SOURCES[name]
        mask = kind.str.contains(pattern, regex=True)
        masks.append((mask & side.eq(wanted_side) if wanted_side else mask).to_numpy())
    union = np.any(masks, axis=0)
    stim_index = np.flatnonzero(union)
    stim_matrix = np.stack([m[stim_index] for m in masks], axis=1).astype(np.float32)
    readout = neurons[neurons["superclass"].eq("descending_neuron")]
    steering = readout["type"].isin(STEER_TYPES)
    steer_sign = np.where(steering & readout["side"].eq("L"), 1.0, np.where(steering & readout["side"].eq("R"), -1.0, 0.0))
    return Channels(torch.as_tensor(stim_index), torch.as_tensor(stim_matrix), torch.as_tensor(readout.index.to_numpy().copy()),
                    readout["bodyId"].to_numpy(), torch.as_tensor(steer_sign, dtype=torch.float32))

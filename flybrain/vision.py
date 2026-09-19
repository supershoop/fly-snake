"""Retinotopic vision: show the board to the fly through the viewing directions of its own visual detector cells.

Each LC10 (small object), LC4 and LPLC2 (looming) cell watches one patch of the visual field. We recover that patch from
the connectome: the synapse-weighted mean eye-map position (assignedOlHex1/2) of the columnar cells that feed it.
Eye-map axes, established from lamina soma positions: hex1 - hex2 large = FRONT of the eye, hex1 + hex2 large = dorsal.

The game is flat, so a snake sees a 1-D horizon: food and obstacles have an azimuth and an apparent width that grows as
they get closer. Food drives the LC10 cells looking that way; obstacles drive LC4 + LPLC2 cells looking their way, harder
when nearer. Unlike flybrain.channels (5 on/off channels, 24 situations) the input is continuous and cell-specific.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import scipy.sparse as sp
import torch

from .connectome import DATA, WEIGHTS, Connectome
from .snake import HEADINGS, Arena

CACHE = DATA / "retinotopy-cache.npz"
FOOD_TYPES, THREAT_TYPES = r"LC10.*", r"LC4|LPLC2"
FIELD_FRONT_DEG, FIELD_BACK_DEG = -15.0, 150.0  # one eye sees from slightly across the midline to well behind
HORIZON_BAND = 0.35      # use cells whose elevation is within this fraction of the eye's height around the equator
THREAT_RANGE = 4.0       # obstacles farther than this many cells are not shown
MIN_HALF_WIDTH_DEG = 6.0


@dataclass
class Retina:
    index: torch.Tensor      # int64 [S] simulator indices of every mapped detector cell on the horizon band
    azimuth: np.ndarray      # float [S] degrees, 0 = straight ahead, negative = left, positive = right
    is_food: np.ndarray      # bool [S] LC10; the rest are looming detectors

    def paint(self, drive: np.ndarray, forward: float, rightward: float, cells: np.ndarray, strength: float):
        """Light up `cells` looking toward an object at (forward, rightward) in snake-relative grid units."""
        azimuth = np.degrees(np.arctan2(rightward, forward))
        half_width = max(MIN_HALF_WIDTH_DEG, np.degrees(np.arctan2(0.5, np.hypot(forward, rightward))))
        seen = cells & (np.abs((self.azimuth - azimuth + 180) % 360 - 180) <= half_width)
        drive[seen] = np.maximum(drive[seen], strength)

    def render(self, arena: Arena, snake_index: int = 0) -> np.ndarray:
        """Drive in 0..1 for every retina cell, for one snake's view of its arena."""
        drive = np.zeros(len(self.index), dtype=np.float32)
        snake = arena.snakes[snake_index]
        head, (hx, hy) = snake.cells[0], HEADINGS[snake.heading]
        relative = lambda cell: ((cell[0] - head[0]) * hx + (cell[1] - head[1]) * hy, (cell[0] - head[0]) * -hy + (cell[1] - head[1]) * hx)
        for food in arena.foods:
            self.paint(drive, *relative(food), self.is_food, 1.0)
        reach = int(np.ceil(THREAT_RANGE))
        obstacles = {cell for other in arena.snakes if other.alive for cell in (other.cells[1:] if other is snake else other.cells)}
        for x in range(head[0] - reach, head[0] + reach + 1):
            for y in range(head[1] - reach, head[1] + reach + 1):
                distance = np.hypot(x - head[0], y - head[1])
                outside = not (0 <= x < arena.size and 0 <= y < arena.size)
                if 0 < distance <= THREAT_RANGE and (outside or (x, y) in obstacles):
                    self.paint(drive, *relative((x, y)), ~self.is_food, min(1.0, 1.0 / distance))
        return drive


def receptive_fields(connectome: Connectome) -> pd.DataFrame:
    """Per neuron: eye-map centre from ALL synapses (not just >= 5) made onto it by same-side eye-mapped columnar cells."""
    neurons = connectome.neurons
    if CACHE.exists():
        cached = np.load(CACHE)
        if len(cached["u"]) == connectome.n:
            return neurons.assign(u=cached["u"], v=cached["v"], eye_synapses=cached["synapses"])
    mapped = neurons["assignedOlHex1"].notna().to_numpy()
    edges = feather.read_table(WEIGHTS, columns=["body_pre", "body_post", "weight"]).to_pandas()
    row = pd.Series(np.arange(connectome.n), index=neurons["bodyId"].to_numpy())
    pre, post = row.reindex(edges["body_pre"]).to_numpy(), row.reindex(edges["body_post"]).to_numpy()
    keep = ~(np.isnan(pre) | np.isnan(post))
    pre, post, weight = pre[keep].astype(int), post[keep].astype(int), edges["weight"].to_numpy()[keep].astype(np.float64)
    side = neurons["side"].to_numpy()
    keep = mapped[pre] & (side[pre] == side[post])
    inputs = sp.csr_matrix((weight[keep], (post[keep], pre[keep])), shape=(connectome.n, connectome.n))
    h1, h2 = neurons["assignedOlHex1"].fillna(0).to_numpy(), neurons["assignedOlHex2"].fillna(0).to_numpy()
    synapses = np.asarray(inputs.sum(axis=1)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        u, v = inputs @ (h1 - h2) / synapses, inputs @ (h1 + h2) / synapses
    np.savez(CACHE, u=u, v=v, synapses=synapses)
    return neurons.assign(u=u, v=v, eye_synapses=synapses)


def build_retina(connectome: Connectome, min_synapses: int = 10) -> Retina:
    fields = receptive_fields(connectome)
    kind = fields["type"].fillna("")
    food, threat = kind.str.fullmatch(FOOD_TYPES), kind.str.fullmatch(THREAT_TYPES)
    columns = fields[fields["assignedOlHex1"].notna()]
    u_all, v_all = columns["assignedOlHex1"] - columns["assignedOlHex2"], columns["assignedOlHex1"] + columns["assignedOlHex2"]
    u_front, u_back, v_mid, v_span = u_all.max(), u_all.min(), v_all.median(), v_all.max() - v_all.min()
    cells = fields[(food | threat) & (fields["eye_synapses"] >= min_synapses) & fields["side"].isin(["L", "R"])
                   & ((fields["v"] - v_mid).abs() <= HORIZON_BAND * v_span)]
    eccentricity = FIELD_FRONT_DEG + (u_front - cells["u"]) / (u_front - u_back) * (FIELD_BACK_DEG - FIELD_FRONT_DEG)
    azimuth = np.where(cells["side"].eq("L"), -eccentricity, eccentricity)
    return Retina(torch.as_tensor(cells.index.to_numpy().copy()), azimuth.astype(np.float32), food[cells.index].to_numpy())

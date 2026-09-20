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


# --- what the web page shows ----------------------------------------------------------------------------------------
PATHWAY_NODES = [  # (name, regex on annotation type, role) - one node per side. Found by path search, see scripts/lesion_scores.py
    ("LC10", r"LC10.*", "object detectors"), ("AOTU", r"AOTU025|AOTU012|AOTU015", "relay cells"), ("DNa02", r"DNa02", "steering"),
    ("LC4", r"LC4", "looming detectors"), ("LPLC2", r"LPLC2", "looming detectors"), ("DNp01", r"DNp01", "giant fiber · escape"),
    ("PVLP", r"PVLP141|PVLP137", "relay cells"), ("DNa01", r"DNa01", "turn away"),
]
PATHWAY_EDGES = [("LC10", "AOTU", "same"), ("AOTU", "DNa02", "same"), ("LC4", "DNp01", "same"), ("LPLC2", "DNp01", "same"),
                 ("LC4", "PVLP", "same"), ("PVLP", "DNa01", "opposite")]


class VisionDisplay:
    """Static maps sent to the page once, plus the per-move values that animate them."""

    def __init__(self, connectome: Connectome, retina: Retina):
        self.retina, neurons = retina, connectome.neurons
        kind, side = neurons["type"].fillna(""), neurons["side"]
        self.nodes = {f"{name}_{s}": np.flatnonzero((kind.str.fullmatch(pattern) & side.eq(s)).to_numpy())
                      for name, pattern, _ in PATHWAY_NODES for s in "LR"}
        roles = {name: role for name, _, role in PATHWAY_NODES}
        other = {"L": "R", "R": "L"}
        cells = receptive_fields(connectome).loc[retina.index.numpy()]
        columns = neurons[neurons["assignedOlHex1"].notna() & neurons["type"].eq("Mi1")]  # one Mi1 per eye column
        self.static = {
            "pathway": {
                "nodes": [{"id": node, "label": node.split("_")[0], "side": node[-1], "role": roles[node.split("_")[0]],
                           "bodyIds": neurons["bodyId"].to_numpy()[index].tolist()} for node, index in self.nodes.items()],
                "edges": [[f"{a}_{s}", f"{b}_{s if relation == 'same' else other[s]}"] for a, b, relation in PATHWAY_EDGES for s in "LR"],
            },
            # u = hex1 - hex2 (large = front of the eye), v = hex1 + hex2 (large = dorsal)
            "eye": {"columns": [[int(r.assignedOlHex1 - r.assignedOlHex2), int(r.assignedOlHex1 + r.assignedOlHex2), r.side] for r in columns.itertuples()]},
            "retina": {"cells": [[round(float(az), 1), round(float(r.u), 2), round(float(r.v), 2), r.side, bool(food)]
                                 for az, r, food in zip(retina.azimuth, cells.itertuples(), retina.is_food)],
                       "fieldDeg": [FIELD_FRONT_DEG, FIELD_BACK_DEG], "threatRange": THREAT_RANGE},
        }

    def live(self, counts: torch.Tensor, seconds: float, drive: np.ndarray) -> dict:
        """counts [N] spikes of the selected fly this move; drive [S] what its retina cells were shown."""
        return {"pathway": {node: round(float(counts[torch.as_tensor(index, device=counts.device)].mean()) / seconds, 1) if len(index) else 0.0
                            for node, index in self.nodes.items()},
                "view": [[int(i), round(float(drive[i]), 2)] for i in np.flatnonzero(drive)]}


class VisionUntrained:
    """Nothing trained: turn toward the side whose steering neuron DNa02 fires more, and away from the side whose giant
    fiber DNp01 fires more. The second rule is ours: in a real fly the giant fiber triggers a jump, not a turn."""

    def __init__(self, readout_neurons: pd.DataFrame, threshold: float = 2.0):
        def sign(kind):
            return np.where(readout_neurons["type"].eq(kind) & readout_neurons["side"].eq("L"), 1.0,
                            np.where(readout_neurons["type"].eq(kind) & readout_neurons["side"].eq("R"), -1.0, 0.0))
        self.weights, self.threshold = torch.as_tensor(sign("DNa02") - 0.5 * sign("DNp01"), dtype=torch.float32), threshold

    def act(self, dn_counts: torch.Tensor):
        drive = self.weights.to(dn_counts.device) @ dn_counts  # [B], positive = turn left
        action = torch.where(drive > self.threshold, 0, torch.where(drive < -self.threshold, 2, 1))
        return action, torch.nn.functional.one_hot(action, 3).float()

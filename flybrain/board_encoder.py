"""Experimental direct spatial input to the existing fly connectome.

Every square is represented relative to the head, rotated with its heading.
No collision check, path search, teacher, or chosen action is an input.
Neuron receptive-field positions are ENGINEERED, not measured retinotopy.
"""
from dataclasses import dataclass
import hashlib
import json

import numpy as np
from scipy.ndimage import gaussian_filter

from .snake import HEADINGS
from .synaptic import SynapticSites, connectome_hash


@dataclass
class BoardEncoder:
    size: int
    indices: np.ndarray  # [3, width, width], unique real visual neurons
    body_ids: np.ndarray
    food_sigma: float = 3.0
    obstacle_sigma: float = 0.8
    gain: float = 1.0

    @property
    def width(self):
        return 2 * self.size - 1

    @property
    def stimulus(self):
        return self.indices.reshape(-1)

    @classmethod
    def from_connectome(cls, connectome, size=12, **kwargs):
        if size < 6:
            raise ValueError("Board must be at least 6 squares wide")
        neurons = connectome.neurons
        kinds = neurons.type.fillna("")
        pools = [kinds.str.startswith("LC10"), kinds.isin(["LC4", "LPLC2", "LC12"]), kinds.eq("Tm3")]
        width, center = 2 * size - 1, size - 1
        indices = np.empty((3, width, width), dtype=np.int64)
        # Nearby occupancy goes to LC4, then LPLC2, then LC12. This is a fixed
        # interface design, not an inferred biological receptive-field map.
        priority = {"LC4": 0, "LPLC2": 1, "LC12": 2}
        for plane, pool in enumerate(pools):
            for side in ("L", "R"):
                cells = [(y, x) for y in range(width) for x in range(width)
                         if ("L" if x < center or (x == center and y % 2 == 0) else "R") == side]
                cells.sort(key=lambda p: ((p[0] - center) ** 2 + (p[1] - center) ** 2, p))
                available = list(np.flatnonzero(pool & neurons.side.eq(side)))
                available.sort(key=lambda i: (priority.get(kinds.iloc[i], 0), int(neurons.bodyId.iloc[i])))
                if len(available) < len(cells):
                    raise ValueError(f"Not enough {side} visual neurons for plane {plane}: {len(available)} < {len(cells)}")
                for (y, x), neuron in zip(cells, available):
                    indices[plane, y, x] = neuron
        if len(np.unique(indices)) != indices.size:
            raise ValueError("Input neuron assignments must be disjoint")
        encoder = cls(size, indices, neurons.bodyId.to_numpy()[indices], **kwargs)
        if (not np.isfinite([encoder.food_sigma, encoder.obstacle_sigma, encoder.gain]).all()
                or encoder.food_sigma < 0 or encoder.obstacle_sigma < 0 or not 0 < encoder.gain <= 1):
            raise ValueError("Invalid sensory field parameters")
        return encoder

    def planes(self, arena):
        """Food, occupied/outside, ordered body. Head always at center, facing up.

        Body values are (segment index + 1) / size**2. Thus head, tail,
        length and complete segment order can be recovered from the raw planes.
        All out-of-board coordinates are marked occupied with zero body order.
        Rotation discards absolute compass direction, which is irrelevant to
        relative actions on this square solo board. Idle is encoded separately.
        """
        if arena.size != self.size or len(arena.snakes) != 1:
            raise ValueError("Direct-board experiment currently requires a matching solo board")
        snake = arena.snakes[0]
        center = self.size - 1
        head_x, head_y = snake.cells[0]
        hx, hy = HEADINGS[snake.heading]

        def location(cell):
            dx, dy = cell[0] - head_x, cell[1] - head_y
            return center - (dx * hx + dy * hy), center + dx * -hy + dy * hx

        result = np.zeros((3, self.width, self.width), dtype=np.float32)
        result[1] = 1.
        for y in range(self.size):
            for x in range(self.size):
                row, col = location((x, y))
                result[1, row, col] = 0.
        for cell in arena.foods:
            row, col = location(cell)
            result[0, row, col] = 1.
        for order, cell in enumerate(snake.cells):
            row, col = location(cell)
            result[1, row, col] = 1.
            result[2, row, col] = (order + 1) / (self.size ** 2)
        return result

    def encode(self, arena):
        planes = self.planes(arena)
        # Gaussian receptive fields make single-cell objects recruit a local
        # population. Peak-normalized convolution plus the raw signal retains
        # cell detail, but finite noisy spike trains are not lossless messages.
        for plane, sigma in ((0, self.food_sigma), (1, self.obstacle_sigma)):
            if sigma > 0:
                impulse = np.zeros((self.width, self.width), dtype=np.float32)
                impulse[self.size - 1, self.size - 1] = 1.
                peak = gaussian_filter(impulse, sigma, mode="constant")[self.size - 1, self.size - 1]
                smooth = gaussian_filter(planes[plane], sigma, mode="constant") / peak
                planes[plane] = np.maximum(planes[plane], np.minimum(smooth, 1.))
        # Order remains separately represented; an offset gives short snakes
        # detectable input without conflating empty cells with their segments.
        planes[2] = np.where(planes[2] > 0, .25 + .75 * planes[2], 0.)
        # Head position in this plane is otherwise a known constant. Use it
        # for the starvation clock (not a recommendation or danger signal).
        planes[2, self.size - 1, self.size - 1] = .25 + .75 * min(arena.snakes[0].idle / arena.max_idle, 1.)
        return (planes.reshape(-1) * self.gain).astype(np.float32)

    def metadata(self):
        result = {"kind": "direct-board-v1", "size": self.size, "width": self.width,
                  "planes": ["food", "occupied_or_outside", "body_order_and_idle"],
                  "food_sigma": self.food_sigma, "obstacle_sigma": self.obstacle_sigma,
                  "gain": self.gain, "assignment": "engineered spatial positions; not biological retinotopy",
                  "input_neurons": int(self.indices.size)}
        digest = hashlib.sha256(json.dumps(result, sort_keys=True).encode())
        digest.update(self.body_ids.astype(np.int64).tobytes())
        result["sha256"] = digest.hexdigest()
        return result


def board_sites(connectome, encoder):
    """Train existing sensory->steering-relay and relay->steering connections.

    Sensory connections share log gains by plane, spatial region, target side,
    and sign; final steering connections retain the original 22 groups.
    """
    original = SynapticSites.from_connectome(connectome)
    input_set = np.zeros(connectome.n, dtype=bool)
    input_set[encoder.stimulus] = True
    relay_set = np.zeros(connectome.n, dtype=bool)
    relay_set[original.pre] = True
    selected = np.flatnonzero(np.isin(connectome.post, original.post)
                             | (input_set[connectome.pre] & relay_set[connectome.post]))
    names_by_edge = {(int(pre), int(post)): original.labels[group]
                     for pre, post, group in zip(original.pre, original.post, original.group)}
    coordinates = {int(neuron): (plane, y - encoder.size + 1, x - encoder.size + 1)
                   for (plane, y, x), neuron in np.ndenumerate(encoder.indices)}
    names = []
    n = connectome.neurons
    for edge in selected:
        pre, post = int(connectome.pre[edge]), int(connectome.post[edge])
        if (pre, post) in names_by_edge:
            names.append("motor:" + names_by_edge[(pre, post)])
        else:
            plane, y, x = coordinates[pre]
            horizontal = "left" if x < -1 else "right" if x > 1 else "center"
            vertical = "ahead" if y < -1 else "behind" if y > 1 else "near"
            sign = "+" if connectome.weight[edge] > 0 else "-"
            names.append(f"sensory:{plane}:{horizontal}:{vertical}:{sign}->{n.side.iloc[post]}")
    labels = sorted(set(names))
    group = np.array([labels.index(name) for name in names], dtype=np.int64)
    return SynapticSites(connectome.pre[selected], connectome.post[selected], connectome.weight[selected],
                         group, labels, n.bodyId.to_numpy(), connectome_hash(connectome))

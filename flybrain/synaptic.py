"""Reward-search parameters INSIDE the connectome; the motor decoder is fixed.

Only existing inputs to DNa01/DNa02 can change. Anatomical source families
share a bounded gain per target and transmitter sign. No edge is added,
removed or sign-flipped. This is engineered plasticity, not a claim about
the learning rule used by a biological fly.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .brain import W_SYN, MALECNS_WEIGHT_SCALE

LIMIT = float(np.log(4.))
DECODER = {"kind": "hardwired", "types": ["DNa02", "DNa01"], "threshold": 2.0}


def connectome_hash(connectome):
    digest = hashlib.sha256()
    for array in (connectome.neurons.bodyId.to_numpy(dtype=np.int64), connectome.pre,
                  connectome.post, connectome.weight):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


@dataclass
class SynapticSites:
    pre: np.ndarray
    post: np.ndarray
    base: np.ndarray
    group: np.ndarray
    labels: list[str]
    body_ids: np.ndarray
    fingerprint: str

    @classmethod
    def from_connectome(cls, connectome):
        kinds = connectome.neurons.type.fillna("").to_numpy()
        sides = connectome.neurons.side.fillna("?").to_numpy()
        selected = np.flatnonzero(np.isin(kinds[connectome.post], DECODER["types"]))
        pre, post, base = connectome.pre[selected], connectome.post[selected], connectome.weight[selected]
        if len(np.unique(pre * np.int64(connectome.n) + post)) != len(pre):
            raise ValueError("Plastic sites must be unique existing connections")
        if not len(pre):
            raise ValueError("No existing steering synapses found")
        names = []
        for source, target, weight in zip(pre, post, base):
            family = next((prefix for prefix in ("AOTU", "PVLP") if kinds[source].startswith(prefix)), "other")
            names.append(f"{family}:{'excitatory' if weight > 0 else 'inhibitory'}->{kinds[target]}:{sides[target]}")
        labels = sorted(set(names))
        group = np.array([labels.index(name) for name in names], dtype=np.int64)
        return cls(pre, post, base, group, labels, connectome.neurons.bodyId.to_numpy(), connectome_hash(connectome))

    def validate(self, parameters):
        parameters = np.asarray(parameters, dtype=np.float64)
        if parameters.shape != (len(self.labels),) or not np.isfinite(parameters).all() or (np.abs(parameters) > LIMIT + 1e-7).any():
            raise ValueError("Synaptic log-gains must be finite, correctly sized, and within log(1/4)..log(4)")
        return parameters

    def save(self, path, parameters, **metadata):
        parameters = self.validate(parameters)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        description = {**metadata, "version": 1, "connectome_sha256": self.fingerprint,
                       "decoder": DECODER, "dt_ms": .5, "window_ms": 100., "weight_scale": MALECNS_WEIGHT_SCALE,
                       "method": "reward-driven search of existing synaptic strengths; fixed hardwired decoder"}
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, parameters=parameters, labels=np.array(self.labels), group=self.group,
                            pre_body_ids=self.body_ids[self.pre], post_body_ids=self.body_ids[self.post],
                            base=self.base, metadata=np.array(json.dumps(description)))
        temporary.replace(path)

    def load(self, path):
        with np.load(path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata"]))
            if (metadata.get("version") != 1 or metadata.get("connectome_sha256") != self.fingerprint
                    or metadata.get("decoder") != DECODER or metadata.get("dt_ms") != .5
                    or metadata.get("window_ms") != 100. or metadata.get("weight_scale") != MALECNS_WEIGHT_SCALE):
                raise ValueError("Synaptic checkpoint does not match this connectome, simulation or fixed decoder")
            expected = {"labels": np.array(self.labels), "group": self.group, "pre_body_ids": self.body_ids[self.pre],
                        "post_body_ids": self.body_ids[self.post], "base": self.base}
            if any(not np.array_equal(stored[key], value) for key, value in expected.items()):
                raise ValueError("Synaptic checkpoint site identity mismatch")
            return self.validate(stored["parameters"]).copy(), metadata


class SynapticAdapter:
    """Update torch CSR AND its optional CPU CSC cache; never train a readout."""

    def __init__(self, brain, sites):
        if brain.shuffled or brain.dt != .5:
            raise ValueError("Synaptic checkpoints require real wiring and 0.5 ms time steps")
        self.brain, self.sites = brain, sites
        rows = brain.weights.crow_indices().cpu().numpy()
        cols = brain.weights.col_indices().cpu().numpy()
        offsets = []
        for pre, post in zip(sites.pre, sites.post):
            offset = rows[post] + np.searchsorted(cols[rows[post]:rows[post + 1]], pre)
            if offset >= rows[post + 1] or cols[offset] != pre:
                raise ValueError("Brain does not contain the specified anatomical connection")
            offsets.append(offset)
        self.offsets = np.asarray(offsets)
        self.device_offsets = torch.as_tensor(self.offsets, device=brain.device)
        self.base = sites.base * np.float32(W_SYN * MALECNS_WEIGHT_SCALE)
        actual = brain.weights.values()[self.device_offsets].cpu().numpy()
        if not np.allclose(actual, self.base, rtol=1e-6, atol=1e-7):
            raise ValueError("Adapter requires an unchanged brain with the standard weight scale")
        self.cache, self.csc_offsets = None, None

    def apply(self, parameters):
        parameters = self.sites.validate(parameters)
        values = (self.base * np.exp(parameters[self.sites.group])).astype(np.float32)
        self.brain.weights.values()[self.device_offsets] = torch.as_tensor(values, device=self.brain.device)
        cache = self.brain.cpu_synapses
        if cache is not None:
            if cache is not self.cache:
                matrix = cache.csc
                offsets = []
                for pre, post in zip(self.sites.pre, self.sites.post):
                    offset = matrix.indptr[pre] + np.searchsorted(matrix.indices[matrix.indptr[pre]:matrix.indptr[pre + 1]], post)
                    if offset >= matrix.indptr[pre + 1] or matrix.indices[offset] != post:
                        raise ValueError("CPU synapse cache does not match brain")
                    offsets.append(offset)
                self.cache, self.csc_offsets = cache, np.asarray(offsets)
            cache.csr.data[self.offsets] = values
            cache.csc.data[self.csc_offsets] = values


def perturb(parameters, rng, sigma=.45):
    """Antithetic proposals, bounded to preserve every connection's sign."""
    noise = rng.normal(size=len(parameters))
    if rng.random() < .5:
        noise *= rng.random(len(parameters)) < .3
    return [np.clip(parameters + direction * sigma * noise, -LIMIT, LIMIT) for direction in (1, -1)]

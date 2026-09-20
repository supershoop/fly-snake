"""Load the MaleCNS v1.0 flat connectome into a signed sparse weight matrix.

Neurons: every `Traced` body in the annotation table (brain + nerve cord).
Edges:   traced-only connection weights, thresholded at MIN_SYNAPSES like Shiu et al. 2024.
Sign:    from the consensus neurotransmitter prediction (ACh excitatory; GABA, glutamate and
         histamine inhibitory; everything else excitatory, as in Shiu et al.).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

DATA = Path(__file__).resolve().parents[1] / "data"
ANNOTATIONS = DATA / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
TRANSMITTERS = DATA / "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = DATA / "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"
CACHE = DATA / "connectome-cache.npz"

MIN_SYNAPSES = 5
INHIBITORY = {"gaba", "glutamate", "histamine"}
ANNOTATION_COLUMNS = ["bodyId", "type", "instance", "superclass", "class", "subclass", "somaSide",
                      "rootSide", "flywireType", "entryNerve", "exitNerve", "receptorType", "assignedOlHex1", "assignedOlHex2", "status"]


@dataclass
class Connectome:
    neurons: pd.DataFrame  # one row per neuron, row index == simulator index
    pre: np.ndarray        # int64 [E] simulator index of presynaptic neuron
    post: np.ndarray       # int64 [E]
    weight: np.ndarray     # float32 [E] signed synapse count

    @property
    def n(self) -> int:
        return len(self.neurons)

    def index_of(self, body_ids) -> np.ndarray:
        """Simulator indices for MaleCNS body IDs."""
        lookup = pd.Series(np.arange(self.n), index=self.neurons["bodyId"].to_numpy())
        return lookup.loc[list(body_ids)].to_numpy()

    def select(self, **equals) -> pd.DataFrame:
        """Rows whose annotation columns equal the given values, e.g. select(type='MN9', side='L')."""
        rows = self.neurons
        for column, value in equals.items():
            rows = rows[rows[column].isin(value) if isinstance(value, (list, set, tuple)) else rows[column] == value]
        return rows


def load_neurons() -> pd.DataFrame:
    neurons = feather.read_table(ANNOTATIONS, columns=ANNOTATION_COLUMNS).to_pandas()
    neurons = neurons[neurons["status"] == "Traced"].drop(columns="status")
    transmitters = feather.read_table(TRANSMITTERS, columns=["body", "consensus_nt"]).to_pandas()
    neurons = neurons.merge(transmitters, how="left", left_on="bodyId", right_on="body").drop(columns="body")
    neurons["side"] = neurons["somaSide"].fillna(neurons["rootSide"])
    neurons["sign"] = np.where(neurons["consensus_nt"].isin(INHIBITORY), -1, 1).astype(np.int8)
    return neurons.sort_values("bodyId").reset_index(drop=True)


def load_connectome(rebuild: bool = False) -> Connectome:
    neurons = load_neurons()
    if CACHE.exists() and not rebuild:
        cached = np.load(CACHE)
        if cached["n"] == len(neurons):
            return Connectome(neurons, cached["pre"], cached["post"], cached["weight"])
    edges = feather.read_table(WEIGHTS, columns=["body_pre", "body_post", "weight"]).to_pandas()
    edges = edges[edges["weight"] >= MIN_SYNAPSES]
    index = pd.Series(np.arange(len(neurons)), index=neurons["bodyId"].to_numpy())
    pre, post = index.reindex(edges["body_pre"]).to_numpy(), index.reindex(edges["body_post"]).to_numpy()
    keep = ~(np.isnan(pre) | np.isnan(post))
    pre, post = pre[keep].astype(np.int64), post[keep].astype(np.int64)
    weight = (edges["weight"].to_numpy()[keep] * neurons["sign"].to_numpy()[pre]).astype(np.float32)
    np.savez(CACHE, n=len(neurons), pre=pre, post=post, weight=weight)
    return Connectome(neurons, pre, post, weight)

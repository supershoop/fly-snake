"""Batched leaky integrate-and-fire simulation of the whole connectome (after Shiu et al. 2024).

    dv/dt = (g - (v - V_REST)) / TAU_M        dg/dt = -g / TAU_SYN
    spike when v > V_THRESHOLD -> v = V_RESET, refractory T_REFRACTORY,
    and after T_DELAY every target gets g += W_SYN * signed synapse count.

State is [N, B]: B independent copies of the brain run in one sparse matmul per step.
"""
import numpy as np
import torch

from .connectome import Connectome

V_REST = V_RESET = -52.0   # mV
V_THRESHOLD = -45.0        # mV
TAU_M = 20.0               # ms
TAU_SYN = 5.0              # ms
T_REFRACTORY = 2.2         # ms
T_DELAY = 1.8              # ms
W_SYN = 0.275              # mV per synapse
STIM_RATE_HZ = 150.0       # Poisson drive of a fully stimulated neuron
STIM_WEIGHT = W_SYN * 250  # mV per Poisson event (Shiu: one event ~ one forced spike)
# MaleCNS detects ~2x more synapses per connection than FlyWire, which W_SYN was tuned for. At 1.0 any input
# recruits ~20k neurons (runaway); 0.35-0.45 is sparse, stimulus-specific and reproduces sugar GRN -> MN9.
MALECNS_WEIGHT_SCALE = 0.4
SHUFFLE_SEED = 0


def default_device() -> torch.device:
    """Return CUDA only when this PyTorch build can actually execute on it.

    ``torch.cuda.is_available()`` only confirms that a driver is visible.  It
    can still be true when the installed CUDA wheel has no kernels for the
    installed GPU, which otherwise lets the web server start and then kills
    its first simulation tick.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        # Creating a filled tensor and reading it back forces a real kernel
        # launch, rather than merely checking that CUDA can allocate memory.
        torch.zeros(1, device="cuda").sum().item()
    except (torch.AcceleratorError, RuntimeError):
        return torch.device("cpu")
    return torch.device("cuda")


class Brain:
    def __init__(self, connectome: Connectome, batch: int = 1, dt: float = 0.5, shuffled: bool = False,
                 weight_scale: float = MALECNS_WEIGHT_SCALE, device: str | None = None, seed: int = 0,
                 shuffle_seed: int = SHUFFLE_SEED, cpu_sparse: bool = True):
        self.device = torch.device(device) if device is not None else default_device()
        self.n, self.batch, self.dt = connectome.n, batch, dt
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        post = connectome.post
        if shuffled:  # control: same neurons, same out-degrees and weights, random targets.
            # Fixed seed on purpose: `seed` only drives the input noise, so every shuffled Brain is the SAME scrambled network.
            post = np.random.default_rng(shuffle_seed).permutation(post)
        weights = torch.sparse_coo_tensor(np.stack([post, connectome.pre]), connectome.weight * (W_SYN * weight_scale),
                                          (self.n, self.n)).coalesce()
        self.weights = weights.to_sparse_csr().to(self.device)
        self.cpu_sparse = self.device.type == "cpu" and cpu_sparse
        self.cpu_synapses = None
        self.delay_steps = max(1, round(T_DELAY / dt))
        self.refractory_steps = max(1, round(T_REFRACTORY / dt))
        self.silenced = None
        self.reset()

    def resize(self, batch: int):
        """Change how many independent brains run in parallel (clears state and lesions)."""
        self.batch, self.silenced = batch, None
        self.reset()

    def set_lesion(self, silenced: torch.Tensor | None):
        """bool [N] (every brain) or [N, B] (per brain): these neurons can never spike. None = intact."""
        if silenced is not None:
            silenced = silenced.to(self.device)
            silenced = silenced[:, None].expand(-1, self.batch) if silenced.dim() == 1 else silenced
        self.silenced = silenced if silenced is not None and bool(silenced.any()) else None

    def reset(self):
        shape = (self.n, self.batch)
        self.v = torch.full(shape, V_REST, device=self.device)
        self.g = torch.zeros(shape, device=self.device)
        self.refractory = torch.zeros(shape, dtype=torch.int16, device=self.device)
        self.pending = torch.zeros((self.delay_steps, *shape), device=self.device)  # ring buffer of delayed input
        self.cursor = 0

    def reset_brains(self, which: torch.Tensor):
        """bool [B]: put these brains back to rest (a new life starts with a quiet brain). Others keep their state."""
        which = which.to(self.device)
        if not bool(which.any()):
            return
        self.v[:, which] = V_REST
        self.g[:, which] = 0.0
        self.refractory[:, which] = 0
        self.pending[:, :, which] = 0.0

    @torch.no_grad()
    def run(self, ms: float, stim_index: torch.Tensor | None = None, stim_level: torch.Tensor | None = None,
            record_index: torch.Tensor | None = None) -> torch.Tensor:
        """Advance `ms` of brain time.

        stim_index  int64 [S]    neurons receiving Poisson drive
        stim_level  float [S, B] drive in 0..1 (fraction of STIM_RATE_HZ)
        returns spike counts, float [N, B], or [R, B] if record_index is given
        """
        steps = round(ms / self.dt)
        # Single-brain CPU play benefits from skipping inactive columns. Keep
        # the original batched/GPU kernel for training and swarm layouts.
        if self.cpu_sparse and self.batch == 1 and self.cpu_synapses is None:
            from .cpu_synapses import CPUSynapses
            self.cpu_synapses = CPUSynapses(self.weights)
        propagate = self.cpu_synapses if self.batch == 1 else None
        counts = torch.zeros((self.n if record_index is None else len(record_index), self.batch), device=self.device)
        if stim_index is not None:
            stim_probability = stim_level.to(self.device) * (STIM_RATE_HZ * self.dt / 1000.0)
        for _ in range(steps):
            self.g += self.pending[self.cursor]
            if stim_index is not None:
                events = torch.rand(stim_probability.shape, device=self.device, generator=self.rng) < stim_probability
                self.g[stim_index] += events * STIM_WEIGHT
            active = self.refractory == 0
            self.v += torch.where(active, (self.g - (self.v - V_REST)) * (self.dt / TAU_M), 0.0)
            self.g -= self.g * (self.dt / TAU_SYN)
            spikes = (self.v > V_THRESHOLD) & active
            if self.silenced is not None:
                spikes &= ~self.silenced
            self.v[spikes] = V_RESET
            self.refractory = torch.where(spikes, self.refractory_steps, (self.refractory - 1).clamp(min=0)).to(torch.int16)
            spikes = spikes.float()
            self.pending[self.cursor] = (propagate(spikes) if propagate is not None
                                         else torch.sparse.mm(self.weights, spikes))
            self.cursor = (self.cursor + 1) % self.delay_steps
            counts += spikes if record_index is None else spikes[record_index]
        return counts

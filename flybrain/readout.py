"""The only trained part: a linear map from descending-neuron spike counts to left / straight / right."""
from pathlib import Path

import numpy as np
import torch

MODELS = Path(__file__).resolve().parents[1] / "models"


class Policy:
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        self.weight, self.bias = weight, bias  # [3, R], [3]

    def logits(self, dn_counts: torch.Tensor) -> torch.Tensor:
        """dn_counts [R, B] -> [B, 3]"""
        return torch.log1p(dn_counts.T) @ self.weight.T + self.bias

    def act(self, dn_counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = self.logits(dn_counts).softmax(dim=1)
        return probabilities.argmax(dim=1), probabilities

    def save(self, name: str):
        MODELS.mkdir(exist_ok=True)
        np.savez(MODELS / f"{name}.npz", weight=self.weight.cpu().numpy(), bias=self.bias.cpu().numpy())

    @classmethod
    def load(cls, name: str, device="cpu") -> "Policy":
        stored = np.load(MODELS / f"{name}.npz")
        return cls(torch.as_tensor(stored["weight"], device=device), torch.as_tensor(stored["bias"], device=device))


class HardwiredPolicy:
    """No learning: turn toward the side whose steering descending neurons (DNa02, DNa01) fire more."""

    def __init__(self, steer_sign: torch.Tensor, threshold: float = 2.0):
        self.steer_sign, self.threshold = steer_sign, threshold

    def act(self, dn_counts: torch.Tensor):
        drive = self.steer_sign.to(dn_counts.device) @ dn_counts  # [B], positive = left
        action = torch.where(drive > self.threshold, 0, torch.where(drive < -self.threshold, 2, 1))
        return action, torch.nn.functional.one_hot(action, 3).float()


class InstinctPolicy:
    """Nothing trained. Pursuit steering with two escape behaviours, read from named cells only:
    steer toward the side whose steering neurons (DNa02 + DNa01) fire more, except that the giant fiber (DNp01) overrides it -
    veto:  never turn toward the side whose giant fiber fires much harder;
    dodge: when both giant fibers fire (threat ahead) and nothing pulls sideways, turn away from the louder one.
    The three thresholds (Hz) are hand-set, not fitted. See scripts/instinct_analysis.py.
    """

    def __init__(self, groups: dict[str, torch.Tensor], window_ms: float, steer: float = 20.0, veto: float = 100.0, alarm: float = 150.0):
        self.groups, self.seconds = groups, window_ms / 1000  # groups: "DNa02_L" -> row indices into dn_counts
        self.steer, self.veto, self.alarm = steer, veto, alarm

    def rate(self, dn_counts: torch.Tensor, name: str) -> torch.Tensor:
        index = self.groups[name]
        return dn_counts[index].sum(dim=0) / max(1, len(index)) / self.seconds  # Hz, [B]

    def act(self, dn_counts: torch.Tensor):
        rate = lambda name: self.rate(dn_counts, name)
        drive = (rate("DNa02_L") + rate("DNa01_L")) - (rate("DNa02_R") + rate("DNa01_R"))  # positive = pulled left
        fiber_left, fiber_right = rate("DNp01_L"), rate("DNp01_R")
        threat = fiber_left - fiber_right                                                   # positive = more threat on the left
        action = torch.where(drive > self.steer, 0, torch.where(drive < -self.steer, 2, 1))
        vetoed = ((action == 0) & (threat > self.veto)) | ((action == 2) & (threat < -self.veto))
        action = torch.where(vetoed, 1, action)
        dodge = (action == 1) & (torch.minimum(fiber_left, fiber_right) > self.alarm)
        action = torch.where(dodge, torch.where(threat > 0, 2, 0), action)
        return action, torch.nn.functional.one_hot(action, 3).float()


class OnlineLearner(Policy):
    """Learns while playing from reward (policy-gradient on the same linear readout).

    Starts blank by default; from_policy() continues from a copy of an existing readout.
    After each move: weight += rate * reward * (chosen - probabilities) x descending-neuron activity.
    All flies in the batch share one readout, so N flies gather experience N times faster. The brain never changes.
    """

    def __init__(self, features: int, device="cpu", rate: float = 0.005, seed: int = 0, entropy: float = 0.0):
        super().__init__(torch.zeros(3, features, device=device), torch.zeros(3, device=device))
        self.rate, self.moves = rate, 0
        self.entropy = entropy  # > 0 keeps the policy exploring, which cures runs that stall at ~5 points (scripts/live_learning_test.py)
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.last = None

    def act(self, dn_counts: torch.Tensor):
        probabilities = self.logits(dn_counts).softmax(dim=1)
        action = torch.multinomial(probabilities, 1, generator=self.rng)[:, 0]
        self.last = (torch.log1p(dn_counts.T), probabilities, action)
        return action, probabilities

    @classmethod
    def from_policy(cls, policy: Policy) -> "OnlineLearner":
        """Continue training a copy, leaving the saved/offline readout untouched."""
        learner = cls(policy.weight.shape[1], device=policy.weight.device)
        learner.weight = policy.weight.detach().clone()
        learner.bias = policy.bias.detach().clone()
        return learner

    def learn(self, rewards: torch.Tensor, *, experience=None, count_moves: bool = True):
        """Update from the last act(), or a saved decision for delayed human feedback.

        rewards [B]; zero excludes a fly. Human feedback does not count as another game move.
        """
        experience = self.last if experience is None else experience
        if experience is None:
            return
        inputs, probabilities, action = experience
        error = (torch.nn.functional.one_hot(action, 3).float() - probabilities) * rewards.to(inputs.device)[:, None]
        if self.entropy:  # gradient of the policy's entropy with respect to its logits
            log_p = torch.log(probabilities.clamp_min(1e-8))
            error = error - self.entropy * probabilities * (log_p - (probabilities * log_p).sum(dim=1, keepdim=True))
        self.weight += self.rate * error.T @ inputs
        self.bias += self.rate * error.sum(dim=0)
        if count_moves:
            self.moves += len(action)

    def teach(self, desired: torch.Tensor, weights: torch.Tensor, *, experience=None):
        """Nudge a saved decision toward the action a human says it should have taken.

        learn() reinforces the action the fly actually chose; this instead pushes toward
        `desired` [B] with strength `weights` [B] (zero excludes a fly), the cross-entropy
        gradient on the same linear readout. It is a bias on top of the game's own rewards,
        never a replacement: the brain and the automatic reward signal are untouched, and
        teaching does not count as another game move.
        """
        experience = self.last if experience is None else experience
        if experience is None:
            return
        inputs, probabilities, _ = experience
        target = torch.nn.functional.one_hot(desired.to(inputs.device), 3).float()
        error = (target - probabilities) * weights.to(inputs.device)[:, None]
        self.weight += self.rate * error.T @ inputs
        self.bias += self.rate * error.sum(dim=0)


def fit(features: torch.Tensor, labels: torch.Tensor, epochs: int = 300, weight_decay: float = 1e-3) -> Policy:
    """features [M, R] spike counts, labels [M] actions."""
    inputs = torch.log1p(features)
    weight = torch.zeros(3, inputs.shape[1], device=inputs.device, requires_grad=True)
    bias = torch.zeros(3, device=inputs.device, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=0.05, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(inputs @ weight.T + bias, labels).backward()
        optimizer.step()
    return Policy(weight.detach(), bias.detach())

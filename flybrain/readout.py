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

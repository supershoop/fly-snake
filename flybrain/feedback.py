"""Credit human feedback to the displayed decision, even when simulation has moved on."""
from collections import OrderedDict
import math

import torch

from .readout import OnlineLearner


class HumanFeedback:
    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self.decisions = OrderedDict()
        self.positive = self.negative = 0
        self.last = None

    def clear(self):
        """Invalidate decisions when the layout, wiring or policy changes."""
        self.decisions.clear()

    def remember(self, move: int, learner: OnlineLearner, eligible: list[bool]):
        if learner.last is not None:
            self.decisions[move] = (learner, learner.last, list(eligible))
            while len(self.decisions) > self.capacity:
                self.decisions.popitem(last=False)

    def apply(self, message: dict, learner: OnlineLearner | None):
        def reject(reason):
            self.last = {"status": "rejected", "reason": reason}

        if learner is None:
            reject("Choose Learn live or Retrain existing readout first.")
            return
        value = message["feedback"]
        if type(value) not in (int, float) or not 0 < abs(value) <= 1 or not math.isfinite(value):
            reject("Stimulus strength must be between -1 and +1, excluding zero.")
            return
        move = message.get("move", next(reversed(self.decisions), None))
        if type(move) is not int or move not in self.decisions:
            reject("That move has expired or the experiment changed. Try the current move.")
            return
        owner, experience, eligible = self.decisions[move]
        if owner is not learner:
            reject("The readout changed. Wait for a new move before sending feedback.")
            return
        fly = message.get("fly")
        if fly is not None and (type(fly) is not int or not 0 <= fly < len(eligible)):
            reject("That fly is not available.")
            return
        targets = [i for i, moved in enumerate(eligible) if moved and (fly is None or i == fly)]
        if not targets:
            reject("No game move to train yet. Wait for the fly to move.")
            return
        rewards = torch.zeros(len(eligible), device=experience[0].device)
        rewards[targets] = float(value)
        learner.learn(rewards, experience=experience, count_moves=False)
        if value > 0:
            self.positive += 1
        else:
            self.negative += 1
        self.last = {"status": "applied", "value": float(value), "fly": fly, "move": move, "targets": targets}

    def state(self):
        return {"positive": self.positive, "negative": self.negative, "last": self.last}

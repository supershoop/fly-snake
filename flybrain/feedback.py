"""Credit human feedback to the displayed decision, even when simulation has moved on."""
from collections import OrderedDict
from dataclasses import dataclass, field
import math

import torch

from .readout import OnlineLearner
from .snake import HEADING_NAMES, LEFT, RIGHT, STRAIGHT

# A snake turns at most one quarter turn per move, so an absolute direction is only
# reachable if it is the current heading or one step either side of it. Reversing is not
# a legal move, which is why "backwards" is refused rather than quietly turned into a turn.
TURN_FOR_DELTA = {0: STRAIGHT, 1: RIGHT, 3: LEFT}


@dataclass
class Decision:
    """One remembered move: enough to train it later, plus where each fly was pointing."""
    learner: OnlineLearner
    experience: tuple
    eligible: list
    headings: list = field(default_factory=list)


class HumanFeedback:
    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self.decisions = OrderedDict()
        self.positive = self.negative = 0
        self.taught = 0
        self.directions: dict[str, int] = {}
        self.last = None

    def clear(self):
        """Invalidate decisions when the layout, wiring or policy changes."""
        self.decisions.clear()

    def remember(self, move: int, learner: OnlineLearner, eligible: list[bool], headings: list[int] | None = None):
        """headings are the pre-move facings, captured before the arena steps."""
        if learner.last is not None:
            self.decisions[move] = Decision(learner, learner.last, list(eligible), list(headings or []))
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
        decision = self.decisions[move]
        experience, eligible = decision.experience, decision.eligible
        if decision.learner is not learner:
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

    def teach(self, message: dict, learner: OnlineLearner | None, weight: float):
        """Train a remembered move toward an absolute compass direction from a phone D-pad.

        `weight` is the caller's per-vote strength, already divided by the crowd size.
        The direction is absolute, so it is converted with the fly's heading at that move.
        """
        def reject(reason):
            self.last = {"status": "rejected", "reason": reason}
            return self.last

        if learner is None:
            return reject("Choose Learn live or Retrain existing readout first.")
        direction = message.get("direction")
        if direction not in HEADING_NAMES:
            return reject("Pick one of the four directions.")
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
            return reject("That vote carried no weight. Try the next move.")
        move = message.get("move", next(reversed(self.decisions), None))
        if type(move) is not int or move not in self.decisions:
            return reject("That move has expired or the experiment changed. Try the current move.")
        decision = self.decisions[move]
        if decision.learner is not learner:
            return reject("The readout changed. Wait for a new move before teaching.")
        fly = message.get("fly")
        if fly is not None and (type(fly) is not int or not 0 <= fly < len(decision.eligible)):
            return reject("That fly is not available.")
        targets = [i for i, moved in enumerate(decision.eligible) if moved and (fly is None or i == fly)]
        if not targets:
            return reject("No game move to teach yet. Wait for the fly to move.")
        if any(i >= len(decision.headings) for i in targets):
            return reject("That move predates direction teaching. Try the current move.")

        wanted = HEADING_NAMES[direction]
        experience = decision.experience
        desired = torch.zeros(len(decision.eligible), dtype=torch.long, device=experience[0].device)
        weights = torch.zeros(len(decision.eligible), device=experience[0].device)
        taught, reversed_flies = [], []
        for index in targets:
            turn = TURN_FOR_DELTA.get((wanted - decision.headings[index]) % 4)
            if turn is None:  # a quarter turn per move only; backwards is not a move
                reversed_flies.append(index)
                continue
            desired[index], weights[index] = turn, weight
            taught.append(index)
        if not taught:
            facing = next((name for name, value in HEADING_NAMES.items()
                           if reversed_flies and value == decision.headings[reversed_flies[0]]), None)
            return reject(f"A fly cannot turn back on itself{f' while heading {facing}' if facing else ''}."
                          " Pick another direction.")

        learner.teach(desired, weights, experience=experience)
        self.directions[direction] = self.directions.get(direction, 0) + 1
        self.taught += 1
        self.last = {"status": "applied", "direction": direction, "fly": fly, "move": move,
                     "targets": taught, "weight": round(float(weight), 6)}
        return self.last

    def state(self):
        return {"positive": self.positive, "negative": self.negative, "last": self.last,
                "taught": self.taught, "directions": dict(self.directions)}

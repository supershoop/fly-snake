"""Snake with snake-relative actions and an egocentric encoder into the brain's sensory channels."""
import numpy as np

from .channels import CHANNEL_NAMES

LEFT, STRAIGHT, RIGHT = 0, 1, 2
HEADINGS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left as (dx, dy); y grows downward


class Snake:
    def __init__(self, size: int = 12, seed: int | None = None, max_idle: int = 150):
        self.size, self.max_idle, self.rng = size, max_idle, np.random.default_rng(seed)
        self.reset()

    def reset(self):
        middle = self.size // 2
        self.body = [(middle, middle), (middle - 1, middle), (middle - 2, middle)]  # head first
        self.heading, self.score, self.idle, self.alive = 1, 0, 0, True
        self._place_food()

    def _place_food(self):
        free = [(x, y) for x in range(self.size) for y in range(self.size) if (x, y) not in self.body]
        self.food = free[self.rng.integers(len(free))]

    def _cell(self, turn: int):
        dx, dy = HEADINGS[(self.heading + turn - 1) % 4]
        return self.body[0][0] + dx, self.body[0][1] + dy

    def blocked(self, turn: int) -> bool:
        x, y = self._cell(turn)
        return not (0 <= x < self.size and 0 <= y < self.size) or (x, y) in self.body[:-1]

    def food_side(self) -> int:
        """LEFT / STRAIGHT / RIGHT: where the food lies relative to the heading (behind counts as a side)."""
        fx, fy = self.food[0] - self.body[0][0], self.food[1] - self.body[0][1]
        hx, hy = HEADINGS[self.heading]
        forward, rightward = fx * hx + fy * hy, fx * -hy + fy * hx
        if rightward == 0:
            return STRAIGHT if forward > 0 else LEFT
        if forward > abs(rightward):
            return STRAIGHT
        return RIGHT if rightward > 0 else LEFT

    def state(self) -> tuple[int, bool, bool, bool]:
        return self.food_side(), self.blocked(LEFT), self.blocked(STRAIGHT), self.blocked(RIGHT)

    def encode(self) -> np.ndarray:
        """Channel drive in 0..1, ordered like CHANNEL_NAMES."""
        return encode_state(self.state())

    def step(self, action: int):
        if not self.alive:
            return
        if self.blocked(action):
            self.alive = False
            return
        head = self._cell(action)
        self.heading = (self.heading + action - 1) % 4
        self.body.insert(0, head)
        if head == self.food:
            self.score, self.idle = self.score + 1, 0
            self._place_food()
        else:
            self.body.pop()
            self.idle += 1
            self.alive = self.idle < self.max_idle

    def render_state(self) -> dict:
        return {"size": self.size, "body": self.body, "food": self.food, "score": self.score, "alive": self.alive}


ALL_STATES = [(food, l, s, r) for food in (LEFT, STRAIGHT, RIGHT) for l in (False, True) for s in (False, True) for r in (False, True)]


def encode_state(state) -> np.ndarray:
    food, left, ahead, right = state
    levels = dict.fromkeys(CHANNEL_NAMES, 0.0)
    levels["food_L"], levels["food_R"] = float(food in (LEFT, STRAIGHT)), float(food in (RIGHT, STRAIGHT))
    levels["danger_L"], levels["danger_R"], levels["danger_ahead"] = float(left), float(right), float(ahead)
    return np.array([levels[name] for name in CHANNEL_NAMES], dtype=np.float32)


def teacher(state) -> int:
    """Go toward the food unless that cell is blocked; otherwise any free direction."""
    food, *blocked = state
    for action in (food, STRAIGHT, LEFT, RIGHT):
        if not blocked[action]:
            return action
    return STRAIGHT

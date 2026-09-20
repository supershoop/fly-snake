"""Snake arena: any number of snakes (fly- or human-driven) on one board, snake-relative actions, and an
egocentric encoder from each snake's situation into the brain's sensory channels."""
from dataclasses import dataclass, field

import numpy as np

from .channels import CHANNEL_NAMES
from .navigation import dangers

LEFT, STRAIGHT, RIGHT = 0, 1, 2
HEADINGS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left as (dx, dy); y grows downward
HEADING_NAMES = {"up": 0, "right": 1, "down": 2, "left": 3}
REWARD_EAT, REWARD_DIE, REWARD_CLOSER = 1.0, -1.0, 0.1


@dataclass
class Body:
    kind: str = "fly"  # "fly" | "human"
    cells: list = field(default_factory=list)  # head first
    heading: int = 1
    alive: bool = True
    score: int = 0
    idle: int = 0
    games: int = 0
    last_score: int = 0
    high_score: int = 0
    wanted_heading: int | None = None  # human input, absolute
    end_reason: str | None = None  # collision | starvation | filled; evaluation only


class Arena:
    def __init__(self, size: int = 12, kinds=("fly",), foods: int = 1, seed: int | None = None,
                 max_idle: int = 150, respawn: bool = True):
        self.size, self.max_idle, self.respawn = size, max_idle, respawn
        self.rng = np.random.default_rng(seed)
        self.snakes = [Body(kind) for kind in kinds]
        self.foods: list[tuple[int, int]] = []
        self.food_count = foods
        self.reset()

    # --- layout -------------------------------------------------------------------------------------------------
    def reset(self):
        self.foods = []
        for snake in self.snakes:
            snake.cells, snake.score, snake.games, snake.high_score = [], 0, 0, 0
        for index in range(len(self.snakes)):
            self._spawn(index)
        while len(self.foods) < self.food_count:
            if not self._place_food():
                break

    def occupied(self) -> set:
        return {cell for snake in self.snakes if snake.alive for cell in snake.cells}

    def _spawn(self, index: int):
        snake, taken = self.snakes[index], self.occupied() | set(self.foods)
        for _ in range(200):
            x, y = int(self.rng.integers(2, self.size - 3)), int(self.rng.integers(1, self.size - 1))
            cells = [(x, y), (x - 1, y), (x - 2, y)]
            if not taken.intersection(cells + [(x + 1, y), (x + 2, y)]):
                break
        snake.cells, snake.heading, snake.alive, snake.score, snake.idle, snake.wanted_heading = cells, 1, True, 0, 0, None
        snake.end_reason = None

    def _place_food(self):
        taken = self.occupied() | set(self.foods)
        free = [(x, y) for x in range(self.size) for y in range(self.size) if (x, y) not in taken]
        if free:
            self.foods.append(free[self.rng.integers(len(free))])
        return bool(free)

    # --- one snake's view ------------------------------------------------------------------------------------------
    def _cell(self, snake: Body, turn: int):
        dx, dy = HEADINGS[(snake.heading + turn - 1) % 4]
        return snake.cells[0][0] + dx, snake.cells[0][1] + dy

    def blocked(self, index: int, turn: int) -> bool:
        snake = self.snakes[index]
        x, y = self._cell(snake, turn)
        if not (0 <= x < self.size and 0 <= y < self.size):
            return True
        return any((x, y) in (other.cells[:-1] if other is snake else other.cells) for other in self.snakes if other.alive)

    def _nearest_food(self, snake: Body):
        head = snake.cells[0]
        return min(self.foods, key=lambda f: abs(f[0] - head[0]) + abs(f[1] - head[1]), default=head)

    def _food_distance(self, snake: Body) -> int:
        food, head = self._nearest_food(snake), snake.cells[0]
        return abs(food[0] - head[0]) + abs(food[1] - head[1])

    def food_side(self, index: int) -> int:
        """LEFT / STRAIGHT / RIGHT: where the nearest food lies relative to the heading (behind counts as a side)."""
        snake = self.snakes[index]
        food = self._nearest_food(snake)
        fx, fy = food[0] - snake.cells[0][0], food[1] - snake.cells[0][1]
        hx, hy = HEADINGS[snake.heading]
        forward, rightward = fx * hx + fy * hy, fx * -hy + fy * hx
        if rightward == 0:
            return STRAIGHT if forward > 0 else LEFT
        if forward > abs(rightward):
            return STRAIGHT
        return RIGHT if rightward > 0 else LEFT

    def state(self, index: int = 0, *, lookahead: bool = True):
        threats = dangers(self, index) if lookahead else tuple(self.blocked(index, action) for action in range(3))
        return self.food_side(index), *threats

    def encode(self, index: int = 0, *, lookahead: bool = True) -> np.ndarray:
        """Channel drive in 0..1, ordered like CHANNEL_NAMES."""
        return encode_state(self.state(index, lookahead=lookahead))

    # --- dynamics ------------------------------------------------------------------------------------------------
    def steer_human(self, index: int, heading_name: str):
        self.snakes[index].wanted_heading = HEADING_NAMES.get(heading_name)

    def step(self, actions: dict[int, int], hold: frozenset[int] | set[int] = frozenset()) -> dict[int, float]:
        """actions: snake index -> LEFT/STRAIGHT/RIGHT for fly snakes (humans use steer_human). Returns rewards.
        Snakes in `hold` stay where they are this move (a fly pausing to feed) and get no reward entry."""
        rewards = {}
        for index, snake in enumerate(self.snakes):
            if index in hold and snake.alive:
                continue
            if not snake.alive:
                if self.respawn:
                    self._spawn(index)
                    while len(self.foods) < self.food_count:
                        if not self._place_food():
                            break
                continue
            if snake.kind == "human":
                turn = {0: STRAIGHT, 1: RIGHT, 3: LEFT}.get(((snake.wanted_heading if snake.wanted_heading is not None else snake.heading) - snake.heading) % 4, STRAIGHT)
            else:
                turn = actions.get(index, STRAIGHT)
            if self.blocked(index, turn):
                rewards[index] = self._kill(snake)
                continue
            before = self._food_distance(snake)
            head = self._cell(snake, turn)
            snake.heading = (snake.heading + turn - 1) % 4
            snake.cells.insert(0, head)
            if head in self.foods:
                self.foods.remove(head)
                self._place_food()
                snake.score, snake.idle = snake.score + 1, 0
                rewards[index] = REWARD_EAT
                if not self.foods:  # A full board is a completed game, not a crash in the next observation.
                    self._kill(snake, "filled")
            else:
                snake.cells.pop()
                snake.idle += 1
                rewards[index] = REWARD_CLOSER if self._food_distance(snake) < before else -REWARD_CLOSER
                if snake.idle >= self.max_idle:
                    rewards[index] = self._kill(snake, "starvation")
        return rewards

    def _kill(self, snake: Body, reason: str = "collision") -> float:
        snake.alive, snake.games, snake.last_score = False, snake.games + 1, snake.score
        snake.high_score = max(snake.high_score, snake.score)
        snake.end_reason = reason
        return REWARD_DIE

    def render_state(self) -> dict:
        return {"size": self.size, "foods": self.foods,
                "snakes": [{"kind": s.kind, "body": s.cells, "heading": s.heading, "alive": s.alive, "score": s.score,
                            "games": s.games, "lastScore": s.last_score,
                            "highScore": max(s.high_score, s.score)} for s in self.snakes]}


class Snake(Arena):
    """One fly-driven snake that stays dead when it dies (used by the offline training scripts)."""

    def __init__(self, size: int = 12, seed: int | None = None, max_idle: int = 150):
        super().__init__(size, ("fly",), seed=seed, max_idle=max_idle, respawn=False)

    alive = property(lambda self: self.snakes[0].alive)
    score = property(lambda self: self.snakes[0].score)

    def step(self, action: int) -> float:
        return super().step({0: action}).get(0, 0.0) if self.alive else 0.0


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

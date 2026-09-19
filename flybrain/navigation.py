"""Spatial threat sensing for Snake; this supplies observations, never motor actions.

A free neighbouring cell can still close a loop around the snake. Inspect the
space *after* that move (including whether food prevents the tail moving), and
check whether the head can still reach the tail. Other snakes remain obstacles.
"""


def escape_space(size, head, body, obstacles=()):
    """Return (reachable cells, tail reachable) for a hypothetical head-first body.

    The tail is a possible exit because it moves on the following non-food step.
    Interior segments, including the old head, cannot be walked through.
    """
    blocked = set(obstacles) | set(body[1:-1])
    seen = {head}
    pending = [head]
    while pending:
        x, y = pending.pop()
        for cell in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if (0 <= cell[0] < size and 0 <= cell[1] < size
                    and cell not in blocked and cell not in seen):
                seen.add(cell)
                pending.append(cell)
    return len(seen), body[-1] in seen


def dangers(arena, index=0):
    """Three binary threats in LEFT / STRAIGHT / RIGHT order.

    Keep the existing five sensory channels and their 24 possible patterns.
    A threat now includes losing access to the moving tail, not just collision
    on the next step. The readout must learn how to respond to these signals.
    """
    snake = arena.snakes[index]
    obstacles = {cell for other in arena.snakes if other is not snake and other.alive for cell in other.cells}
    result = []
    for action in range(3):
        if arena.blocked(index, action):
            result.append(True)
            continue
        head = arena._cell(snake, action)
        body = [head] + (snake.cells if head in arena.foods else snake.cells[:-1])
        _, tail_reachable = escape_space(arena.size, head, body, obstacles)
        result.append(not tail_reachable)
    return tuple(result)

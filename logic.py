"""Heuristic move-selection logic for the Battlesnake.

Everything flows through :func:`choose_move`, which takes the raw game state and
returns one of ``"up" | "down" | "left" | "right"``.

Board coordinates: ``(0, 0)`` is the bottom-left corner.
  up    -> y + 1
  down  -> y - 1
  left  -> x - 1
  right -> x + 1

Design goals for the tournament version:
  * Never crash. A crash means no move, which means death. Every path is
    wrapped so we always return a legal-looking move.
  * Model tails honestly: a tail cell frees up next turn unless the snake just
    ate (its last two body segments overlap), so we don't refuse safe moves.
  * Score the *quality* of the space we move into with a full flood fill, not
    just "does my body fit". Trapping ourselves in a small pocket loses games.
  * Win head-to-heads against shorter snakes (bonus) and avoid the ones we'd
    lose or tie (heavy penalty).
  * Grow early. Length wins head-to-heads, so eat aggressively while short and
    ease off once we're safely the longest.

Game-state schema reference: https://docs.battlesnake.com/api
"""

from collections import deque
from typing import Dict, List, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

# --- Scoring weights (tuned by feel; safe to adjust) ------------------------
LOSE_H2H_PENALTY = 100_000     # moving where a longer/equal enemy head can also go
WIN_H2H_BONUS = 5_000          # moving where we'd eat a strictly shorter enemy head
SPACE_WEIGHT = 200             # per reachable cell (dominant term)
TRAP_PENALTY = 40_000          # not enough room for our body -> likely death
FOOD_WEIGHT = 12               # pull toward food (scaled by hunger)
WALL_PENALTY = 8               # mild aversion to hugging the edge
CENTER_WEIGHT = 3              # mild pull toward the center for board control

HUNGRY_THRESHOLD = 60          # below this health, food gets much more urgent


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from ``GET /``."""
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#00c896",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "1.0.0",
    }


def choose_move(game_state: Dict) -> str:
    """Return the next move. Never raises."""
    try:
        return _choose_move(game_state)
    except Exception:  # pragma: no cover - last-resort safety net
        try:
            return _safe_fallback(game_state)
        except Exception:
            return "up"


def _choose_move(game_state: Dict) -> str:
    board = game_state["board"]
    you = game_state["you"]
    width: int = board["width"]
    height: int = board["height"]

    head: Point = (you["head"]["x"], you["head"]["y"])
    my_length: int = you["length"]
    my_id: str = you["id"]
    health: int = you["health"]
    snakes: List[Dict] = board["snakes"]

    # Cells that will still be solid *next* turn (bodies minus tails that move).
    blocked = _blocked_next_turn(snakes)
    foods = [(f["x"], f["y"]) for f in board["food"]]

    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    am_i_longest = _am_i_longest(snakes, my_id, my_length)

    best_move = None
    best_score = float("-inf")

    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)

        if not _in_bounds(nxt, width, height):
            continue
        if nxt in blocked:
            continue

        score = 0.0

        # --- Reachable space (dominant term). Trapping ourselves loses. ------
        space = _flood_fill(nxt, blocked, width, height, cap=width * height)
        score += space * SPACE_WEIGHT
        if space <= my_length:
            score -= TRAP_PENALTY

        # --- Head-to-head resolution against enemy snakes -------------------
        h2h = _h2h_outcome(nxt, snakes, my_id, my_length)
        if h2h == "lose":
            score -= LOSE_H2H_PENALTY
        elif h2h == "win":
            score += WIN_H2H_BONUS

        # --- Food: grow early, and always eat when hungry -------------------
        if foods:
            nearest = min(_manhattan(nxt, f) for f in foods)
            urgency = FOOD_WEIGHT
            if health < HUNGRY_THRESHOLD:
                urgency = FOOD_WEIGHT * 4
            elif am_i_longest:
                urgency = FOOD_WEIGHT // 3  # already longest & healthy: relax
            score += (width + height - nearest) * urgency

        # --- Board geometry: avoid walls, gently prefer the center ----------
        if nxt[0] == 0 or nxt[0] == width - 1:
            score -= WALL_PENALTY
        if nxt[1] == 0 or nxt[1] == height - 1:
            score -= WALL_PENALTY
        dist_center = abs(nxt[0] - center[0]) + abs(nxt[1] - center[1])
        score -= dist_center * CENTER_WEIGHT

        if score > best_score:
            best_score = score
            best_move = move

    return best_move or _safe_fallback(game_state)


def _blocked_next_turn(snakes: List[Dict]) -> Set[Point]:
    """Cells solid on the next turn.

    Every body segment is solid, except each snake's tail — which vacates next
    turn UNLESS the snake just ate (then its tail stays put, shown by the last
    two body segments overlapping, e.g. right after eating or at spawn).
    """
    blocked: Set[Point] = set()
    for snake in snakes:
        body = snake["body"]
        for seg in body:
            blocked.add((seg["x"], seg["y"]))
        # Free the tail if it will move (no stacked segment at the tail).
        if len(body) >= 2:
            tail = (body[-1]["x"], body[-1]["y"])
            before_tail = (body[-2]["x"], body[-2]["y"])
            if tail != before_tail:
                blocked.discard(tail)
    return blocked


def _h2h_outcome(cell: Point, snakes: List[Dict], my_id: str, my_length: int) -> str:
    """Would moving to ``cell`` cause a head-to-head, and would we win?

    Returns ``"lose"`` (an equal/longer enemy could move here too),
    ``"win"`` (only strictly shorter enemies could), or ``""`` (no contest).
    """
    result = ""
    for snake in snakes:
        if snake["id"] == my_id:
            continue
        ehead = (snake["head"]["x"], snake["head"]["y"])
        if _manhattan(cell, ehead) != 1:
            continue  # enemy head can't reach this cell in one step
        if snake["length"] >= my_length:
            return "lose"  # any losable contest dominates
        result = "win"
    return result


def _flood_fill(start: Point, blocked: Set[Point], width: int, height: int, cap: int) -> int:
    """Count open cells reachable from ``start`` (breadth-first, capped)."""
    seen: Set[Point] = {start}
    queue: deque = deque([start])
    count = 0
    while queue:
        x, y = queue.popleft()
        count += 1
        if count >= cap:
            break
        for dx, dy in DIRECTIONS.values():
            nbr = (x + dx, y + dy)
            if nbr in seen:
                continue
            if not _in_bounds(nbr, width, height):
                continue
            if nbr in blocked:
                continue
            seen.add(nbr)
            queue.append(nbr)
    return count


def _am_i_longest(snakes: List[Dict], my_id: str, my_length: int) -> bool:
    for snake in snakes:
        if snake["id"] == my_id:
            continue
        if snake["length"] >= my_length:
            return False
    return True


def _safe_fallback(game_state: Dict) -> str:
    """Pick the least-bad move when nothing scored well: prefer staying in
    bounds and off snake bodies; otherwise anything in bounds; else ``up``."""
    board = game_state["board"]
    you = game_state["you"]
    width, height = board["width"], board["height"]
    head = (you["head"]["x"], you["head"]["y"])
    blocked = _blocked_next_turn(board["snakes"])

    in_bounds_moves = []
    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)
        if not _in_bounds(nxt, width, height):
            continue
        in_bounds_moves.append(move)
        if nxt not in blocked:
            return move
    return in_bounds_moves[0] if in_bounds_moves else "up"


def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

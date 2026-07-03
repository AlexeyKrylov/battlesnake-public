"""Move-selection logic for the Battlesnake (search version).

``choose_move`` runs a time-bounded minimax (paranoid: we assume the nearest
enemy moves to hurt us most) with alpha-beta pruning and iterative deepening.
A lightweight forward simulator applies Battlesnake rules each ply, and a
hand-written heuristic scores the leaves. Everything is wrapped so we always
return a legal-looking move within the time budget — a crash or a timeout means
no move, which means death.

Board coordinates: ``(0, 0)`` is the bottom-left corner.
  up -> y+1, down -> y-1, left -> x-1, right -> x+1
Schema: https://docs.battlesnake.com/api
"""

from collections import deque
from time import perf_counter
from typing import Dict, List, Optional, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}
_DIR_ITEMS = list(DIRECTIONS.items())

# --- Time budget ------------------------------------------------------------
# The engine allows 500 ms/turn. Leave headroom for network + JSON so the
# search never makes us miss a turn.
TIME_BUDGET = 0.20
MAX_DEPTH = 8  # iterative deepening rarely reaches this; the clock stops us

# --- Leaf-evaluation weights ------------------------------------------------
ALIVE_BONUS = 1_000_000
DEAD_PENALTY = -1_000_000
SPACE_WEIGHT = 120
TRAP_PENALTY = 30_000
LENGTH_WEIGHT = 400
LEAD_WEIGHT = 300           # reward being longer than the biggest enemy
ENEMY_ALIVE_PENALTY = 600   # fewer live enemies is better
FOOD_WEIGHT = 10
HUNGRY_THRESHOLD = 45


class _Timeout(Exception):
    pass


def get_info() -> Dict[str, str]:
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#00c896",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "2.0.0",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def choose_move(game_state: Dict) -> str:
    try:
        state = _parse(game_state)
        my_id = game_state["you"]["id"]
        move = _search_root(state, my_id)
        if move is not None:
            return move
    except Exception:
        pass
    # Fallbacks: greedy one-ply, then any safe move.
    try:
        return _greedy_move(_parse(game_state), game_state["you"]["id"])
    except Exception:
        try:
            return _safe_fallback(game_state)
        except Exception:
            return "up"


# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------
# A state is a plain dict:
#   {"W": int, "H": int,
#    "snakes": [ {"id": str, "health": int, "body": [ (x,y), ... ]}, ... ],
#    "food": set[(x,y)] }
# body[0] is the head. length == len(body).

def _parse(game_state: Dict) -> Dict:
    board = game_state["board"]
    snakes = []
    for s in board["snakes"]:
        snakes.append({
            "id": s["id"],
            "health": s["health"],
            "body": [(p["x"], p["y"]) for p in s["body"]],
        })
    return {
        "W": board["width"],
        "H": board["height"],
        "snakes": snakes,
        "food": {(f["x"], f["y"]) for f in board["food"]},
    }


def _find(state: Dict, sid: str) -> Optional[Dict]:
    for s in state["snakes"]:
        if s["id"] == sid:
            return s
    return None


# ---------------------------------------------------------------------------
# Forward simulation of one joint turn
# ---------------------------------------------------------------------------

def _step(state: Dict, moves: Dict[str, Point]) -> Dict:
    """Apply one simultaneous turn. ``moves`` maps snake id -> (dx, dy).

    Snakes without an entry keep going straight (or pick any direction).
    Returns a fresh state with eliminated snakes removed.
    """
    W, H = state["W"], state["H"]
    food = state["food"]
    moved: List[Dict] = []

    for s in state["snakes"]:
        dx, dy = moves.get(s["id"]) or _default_dir(s)
        head = (s["body"][0][0] + dx, s["body"][0][1] + dy)
        body = [head] + s["body"]
        health = s["health"] - 1
        if head in food:
            health = 100  # ate: tail stays (growth)
        else:
            body.pop()     # tail moves
        moved.append({"id": s["id"], "health": health, "body": body,
                      "ate": head in food})

    # Consume food eaten this turn.
    eaten = {m["body"][0] for m in moved if m["ate"]}
    new_food = food - eaten if eaten else food

    # --- Eliminations --------------------------------------------------------
    dead: Set[str] = set()

    # Starvation / walls.
    for m in moved:
        if m["health"] <= 0:
            dead.add(m["id"])
            continue
        hx, hy = m["body"][0]
        if not (0 <= hx < W and 0 <= hy < H):
            dead.add(m["id"])

    # Body collisions (head onto any snake body segment past the head).
    for m in moved:
        if m["id"] in dead:
            continue
        head = m["body"][0]
        for o in moved:
            if head in _iter_body_no_head(o["body"]):
                dead.add(m["id"])
                break

    # Head-to-head: equal or longer opponent on the same cell wins.
    for m in moved:
        if m["id"] in dead:
            continue
        head = m["body"][0]
        my_len = len(m["body"])
        for o in moved:
            if o["id"] == m["id"] or o["id"] in dead:
                continue
            if o["body"][0] == head and len(o["body"]) >= my_len:
                dead.add(m["id"])
                break

    survivors = [
        {"id": m["id"], "health": m["health"], "body": m["body"]}
        for m in moved if m["id"] not in dead
    ]
    return {"W": W, "H": H, "snakes": survivors, "food": new_food}


def _iter_body_no_head(body: List[Point]):
    # Skip the head (index 0). Duplicated tail segments are naturally solid.
    return body[1:]


def _default_dir(snake: Dict) -> Point:
    """A non-branching snake keeps its heading if sane, else turns."""
    body = snake["body"]
    if len(body) >= 2 and body[0] != body[1]:
        d = (body[0][0] - body[1][0], body[0][1] - body[1][1])
        if d in DIRECTIONS.values():
            return d
    return (0, 1)


# ---------------------------------------------------------------------------
# Legal / candidate moves
# ---------------------------------------------------------------------------

def _blocked(state: Dict) -> Set[Point]:
    """Cells solid next turn: bodies minus tails that will move."""
    blocked: Set[Point] = set()
    for s in state["snakes"]:
        body = s["body"]
        for seg in body:
            blocked.add(seg)
        if len(body) >= 2 and body[-1] != body[-2]:
            blocked.discard(body[-1])
    return blocked


def _candidate_moves(state: Dict, sid: str, blocked: Set[Point]) -> List[Tuple[str, Point]]:
    snake = _find(state, sid)
    if snake is None:
        return []
    head = snake["body"][0]
    W, H = state["W"], state["H"]
    safe: List[Tuple[str, Point]] = []
    inbounds: List[Tuple[str, Point]] = []
    for move, (dx, dy) in _DIR_ITEMS:
        nxt = (head[0] + dx, head[1] + dy)
        if not (0 <= nxt[0] < W and 0 <= nxt[1] < H):
            continue
        inbounds.append((move, (dx, dy)))
        if nxt not in blocked:
            safe.append((move, (dx, dy)))
    if safe:
        return safe
    return inbounds  # forced: everything is bad, keep options in-bounds


# ---------------------------------------------------------------------------
# Search (paranoid minimax vs the nearest enemy, alpha-beta, iter. deepening)
# ---------------------------------------------------------------------------

def _search_root(state: Dict, my_id: str) -> Optional[str]:
    deadline = perf_counter() + TIME_BUDGET
    me = _find(state, my_id)
    if me is None:
        return None

    blocked = _blocked(state)
    my_moves = _candidate_moves(state, my_id, blocked)
    if not my_moves:
        return None
    if len(my_moves) == 1:
        return my_moves[0][0]

    enemy_id = _nearest_enemy(state, my_id)

    # Order moves by a quick greedy score so alpha-beta prunes well.
    my_moves.sort(key=lambda mv: -_greedy_score(state, my_id, mv[1], blocked))

    best_move = my_moves[0][0]
    for depth in range(1, MAX_DEPTH + 1):
        try:
            best_this_depth = None
            alpha = float("-inf")
            for move, vec in my_moves:
                val = _min_node(state, my_id, enemy_id, move, vec,
                                depth, alpha, float("inf"), deadline)
                if best_this_depth is None or val > best_this_depth[0]:
                    best_this_depth = (val, move)
                alpha = max(alpha, val)
            if best_this_depth is not None:
                best_move = best_this_depth[1]
        except _Timeout:
            break
    return best_move


def _min_node(state: Dict, my_id: str, enemy_id: Optional[str],
              my_move: str, my_vec: Point,
              depth: int, alpha: float, beta: float, deadline: float) -> float:
    """Enemy responds to our committed move, minimizing our evaluation."""
    if perf_counter() > deadline:
        raise _Timeout

    if enemy_id is None or _find(state, enemy_id) is None:
        ns = _step(state, {my_id: my_vec})
        return _max_node(ns, my_id, enemy_id, depth - 1, alpha, beta, deadline)

    blocked = _blocked(state)
    enemy_moves = _candidate_moves(state, enemy_id, blocked)
    value = float("inf")
    for _, evec in enemy_moves:
        ns = _step(state, {my_id: my_vec, enemy_id: evec})
        val = _max_node(ns, my_id, enemy_id, depth - 1, alpha, beta, deadline)
        value = min(value, val)
        beta = min(beta, value)
        if beta <= alpha:
            break  # prune
    return value


def _max_node(state: Dict, my_id: str, enemy_id: Optional[str],
              depth: int, alpha: float, beta: float, deadline: float) -> float:
    if perf_counter() > deadline:
        raise _Timeout

    me = _find(state, my_id)
    if me is None:
        return DEAD_PENALTY - depth  # dying sooner is worse
    enemies = [s for s in state["snakes"] if s["id"] != my_id]
    if not enemies:
        return ALIVE_BONUS + depth  # winning sooner is better
    if depth <= 0:
        return _evaluate(state, my_id)

    blocked = _blocked(state)
    my_moves = _candidate_moves(state, my_id, blocked)
    if not my_moves:
        return _evaluate(state, my_id)

    value = float("-inf")
    for move, vec in my_moves:
        val = _min_node(state, my_id, enemy_id, move, vec,
                        depth, alpha, beta, deadline)
        value = max(value, val)
        alpha = max(alpha, value)
        if alpha >= beta:
            break  # prune
    return value


def _nearest_enemy(state: Dict, my_id: str) -> Optional[str]:
    me = _find(state, my_id)
    head = me["body"][0]
    best, best_d = None, 1e9
    for s in state["snakes"]:
        if s["id"] == my_id:
            continue
        d = _manhattan(head, s["body"][0])
        if d < best_d:
            best, best_d = s["id"], d
    return best


# ---------------------------------------------------------------------------
# Leaf evaluation
# ---------------------------------------------------------------------------

def _evaluate(state: Dict, my_id: str) -> float:
    me = _find(state, my_id)
    if me is None:
        return DEAD_PENALTY
    W, H = state["W"], state["H"]
    head = me["body"][0]
    my_len = len(me["body"])
    enemies = [s for s in state["snakes"] if s["id"] != my_id]

    blocked = _blocked(state)
    score = float(ALIVE_BONUS)

    # Reachable space (survival). Trapping ourselves is close to death.
    space = _flood_fill(head, blocked, W, H, cap=W * H)
    score += space * SPACE_WEIGHT
    if space <= my_len:
        score -= TRAP_PENALTY

    # Length and lead over the biggest enemy.
    score += my_len * LENGTH_WEIGHT
    if enemies:
        biggest = max(len(s["body"]) for s in enemies)
        score += (my_len - biggest) * LEAD_WEIGHT
    score -= len(enemies) * ENEMY_ALIVE_PENALTY

    # Food: pull toward it, urgently when hungry.
    if state["food"]:
        nearest = min(_manhattan(head, f) for f in state["food"])
        urgency = FOOD_WEIGHT * 4 if me["health"] < HUNGRY_THRESHOLD else FOOD_WEIGHT
        score += (W + H - nearest) * urgency

    return score


# ---------------------------------------------------------------------------
# Greedy one-ply (leaf ordering + fallback)
# ---------------------------------------------------------------------------

def _greedy_move(state: Dict, my_id: str) -> str:
    blocked = _blocked(state)
    moves = _candidate_moves(state, my_id, blocked)
    if not moves:
        return "up"
    best_move, best = moves[0][0], float("-inf")
    for move, vec in moves:
        sc = _greedy_score(state, my_id, vec, blocked)
        if sc > best:
            best, best_move = sc, move
    return best_move


def _greedy_score(state: Dict, my_id: str, vec: Point, blocked: Set[Point]) -> float:
    me = _find(state, my_id)
    head = me["body"][0]
    W, H = state["W"], state["H"]
    my_len = len(me["body"])
    nxt = (head[0] + vec[0], head[1] + vec[1])
    if not (0 <= nxt[0] < W and 0 <= nxt[1] < H) or nxt in blocked:
        return float("-inf")
    space = _flood_fill(nxt, blocked, W, H, cap=W * H)
    score = space * SPACE_WEIGHT
    if space <= my_len:
        score -= TRAP_PENALTY
    # Head-to-head shaping.
    for s in state["snakes"]:
        if s["id"] == my_id:
            continue
        ehead = s["body"][0]
        if _manhattan(nxt, ehead) == 1:
            if len(s["body"]) >= my_len:
                score -= 100_000
            else:
                score += 5_000
    if state["food"]:
        nearest = min(_manhattan(nxt, f) for f in state["food"])
        urgency = FOOD_WEIGHT * 4 if me["health"] < HUNGRY_THRESHOLD else FOOD_WEIGHT
        score += (W + H - nearest) * urgency
    return score


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _flood_fill(start: Point, blocked: Set[Point], W: int, H: int, cap: int) -> int:
    if start in blocked:
        return 0
    seen = {start}
    queue = deque([start])
    count = 0
    while queue:
        x, y = queue.popleft()
        count += 1
        if count >= cap:
            break
        for dx, dy in DIRECTIONS.values():
            nbr = (x + dx, y + dy)
            if nbr in seen or not (0 <= nbr[0] < W and 0 <= nbr[1] < H) or nbr in blocked:
                continue
            seen.add(nbr)
            queue.append(nbr)
    return count


def _safe_fallback(game_state: Dict) -> str:
    board = game_state["board"]
    you = game_state["you"]
    W, H = board["width"], board["height"]
    head = (you["head"]["x"], you["head"]["y"])
    blocked = set()
    for s in board["snakes"]:
        for seg in s["body"]:
            blocked.add((seg["x"], seg["y"]))
    inbounds = []
    for move, (dx, dy) in _DIR_ITEMS:
        nxt = (head[0] + dx, head[1] + dy)
        if not (0 <= nxt[0] < W and 0 <= nxt[1] < H):
            continue
        inbounds.append(move)
        if nxt not in blocked:
            return move
    return inbounds[0] if inbounds else "up"


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

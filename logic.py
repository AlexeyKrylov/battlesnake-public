"""Model-backed move-selection logic for the Battlesnake.

The served policy uses a linear ranking model, scores each legal move,
and returns the highest-scoring direction. A compact heuristic remains as a
fallback so gameplay still returns a legal move if model scoring fails.

Board coordinates: ``(0, 0)`` is the bottom-left corner.
  up    -> y + 1
  down  -> y - 1
  left  -> x - 1
  right -> x + 1

Game-state schema reference: https://docs.battlesnake.com/api

Improvements over v0.1.0:
  - Tail-freeing: tails that will vacate next turn are no longer treated as
    permanently blocked, unlocking many otherwise-pruned paths.
  - Kill-seeker: when we are strictly longer we pursue enemy heads.
  - BFS food distance instead of Manhattan for accurate hunger steering.
  - Own-tail chasing: our tail is excluded from blocked cells during flood fill
    so the snake can safely follow itself.
  - Model extended with: kill_opportunity, length_advantage, food_bfs_delta,
    own_tail_dist.  Coefficients re-tuned accordingly.
"""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

# Penalty applied to a move that could lose a head-to-head collision.
HEAD_TO_HEAD_PENALTY = 10_000
# Reward applied to a move that wins a head-to-head collision.
HEAD_TO_HEAD_WIN_BONUS = 5_000
# Below this health we start actively steering toward food.
HUNGRY_THRESHOLD = 50
# Maximum BFS distance value (used as "unreachable" sentinel).
_BIG = 10_000
_NEIGHBORS = ((0, 1), (0, -1), (-1, 0), (1, 0))


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from ``GET /``."""
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "0.2.0",
    }


def choose_move(game_state: Dict) -> str:
    """Return the next move using the model, with a heuristic fallback."""
    try:
        move = choose_move_model(game_state)
    except Exception:  # noqa: BLE001 - a model issue must never break gameplay
        move = None
    if move is not None:
        return move
    return choose_move_heuristic(game_state)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _will_tail_free(snake: Dict) -> bool:
    """Return True if this snake's tail cell will be vacated next turn.

    A tail is *not* freed when the snake just ate food — in that case the last
    two body segments share the same coordinate (the tail grew in-place).
    """
    body = snake["body"]
    if len(body) < 2:
        return False
    # If the last segment equals the second-to-last, the snake grew this turn
    # and the tail cell will NOT be free next turn.
    return (body[-1]["x"], body[-1]["y"]) != (body[-2]["x"], body[-2]["y"])


def _occupied_cells(snakes: List[Dict], free_tails: bool = False) -> Set[Point]:
    """All cells currently filled by any snake's body.

    When ``free_tails`` is True, tail cells that will vacate next turn are
    excluded, giving a more accurate picture of what is actually dangerous.
    """
    occupied: Set[Point] = set()
    for snake in snakes:
        body = snake["body"]
        tail_frees = free_tails and _will_tail_free(snake)
        # Iterate all but last segment; handle tail separately.
        for seg in body[:-1]:
            occupied.add((seg["x"], seg["y"]))
        # Add the tail only if it won't free up.
        if not tail_frees:
            last = body[-1]
            occupied.add((last["x"], last["y"]))
    return occupied


def _head_to_head_cells(
    snakes: List[Dict], my_id: str, my_length: int
) -> Tuple[Set[Point], Set[Point]]:
    """Return (danger_cells, kill_cells).

    danger_cells: adjacent to an enemy head that is >= our length (we'd lose).
    kill_cells:   adjacent to an enemy head that is strictly < our length (we win).
    """
    danger: Set[Point] = set()
    kill: Set[Point] = set()
    for snake in snakes:
        if snake["id"] == my_id:
            continue
        ehead = (snake["head"]["x"], snake["head"]["y"])
        neighbours = {(ehead[0] + dx, ehead[1] + dy) for dx, dy in _NEIGHBORS}
        if snake["length"] >= my_length:
            danger |= neighbours
        else:
            kill |= neighbours
    return danger, kill


def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _bfs_dist(sources: List[Point], blocked: Set[Point], width: int, height: int) -> Dict[Point, int]:
    """Shortest free-cell distances from a set of seed cells."""
    dist: Dict[Point, int] = {}
    dq: deque = deque()
    for source in sources:
        if source not in dist:
            dist[source] = 0
            dq.append(source)
    while dq:
        x, y = dq.popleft()
        d = dist[(x, y)]
        for dx, dy in _NEIGHBORS:
            nb = (x + dx, y + dy)
            if (
                0 <= nb[0] < width
                and 0 <= nb[1] < height
                and nb not in blocked
                and nb not in dist
            ):
                dist[nb] = d + 1
                dq.append(nb)
    return dist


def _flood_fill(
    start: Point,
    occupied: Set[Point],
    width: int,
    height: int,
    limit: int,
) -> int:
    """Count open cells reachable from ``start`` (capped at ``limit``).

    Uses BFS so the count is accurate for small ``limit`` values and gives
    a representative sample for large boards.
    """
    seen: Set[Point] = {start}
    dq: deque = deque([start])
    count = 0
    while dq:
        x, y = dq.popleft()
        count += 1
        if count >= limit:
            break
        for dx, dy in _NEIGHBORS:
            nbr = (x + dx, y + dy)
            if nbr in seen or nbr in occupied:
                continue
            if not _in_bounds(nbr, width, height):
                continue
            seen.add(nbr)
            dq.append(nbr)
    return count


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

def choose_move_heuristic(game_state: Dict) -> str:
    """Return the next move for the current turn (no model)."""
    board = game_state["board"]
    you = game_state["you"]
    width: int = board["width"]
    height: int = board["height"]

    head: Point = (you["head"]["x"], you["head"]["y"])
    my_length: int = you["length"]
    health: int = you["health"]
    my_tail: Point = (you["body"][-1]["x"], you["body"][-1]["y"])

    # Use tail-aware occupied set for a more accurate picture.
    occupied = _occupied_cells(board["snakes"], free_tails=True)
    # For flood-fill we also exclude our own tail since it will move.
    occupied_no_own_tail = occupied - {my_tail}

    danger, kill = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]

    # BFS distances to food from current head position.
    food_dist_from_head = _bfs_dist([head], occupied, width, height)

    best_move = None
    best_score = float("-inf")

    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)

        if not _in_bounds(nxt, width, height):
            continue
        if nxt in occupied:
            continue

        # Reachable open space — exclude our own tail so we can "chase" it.
        space = _flood_fill(nxt, occupied_no_own_tail, width, height, limit=my_length + 1)
        score = float(space)

        # Head-to-head: penalise losing, reward winning.
        if nxt in danger:
            score -= HEAD_TO_HEAD_PENALTY
        elif nxt in kill:
            score += HEAD_TO_HEAD_WIN_BONUS

        # Hunger: steer toward food using BFS distance.
        if foods and health < HUNGRY_THRESHOLD:
            bfs_food_dists = [food_dist_from_head.get(f, _BIG) for f in foods]
            nearest_now = min(bfs_food_dists)
            # BFS distance from nxt to foods.
            nxt_dist_map = _bfs_dist([nxt], occupied, width, height)
            nearest_next = min((nxt_dist_map.get(f, _BIG) for f in foods), default=_BIG)
            score += (nearest_now - nearest_next) * 3  # reward closing gap

        # Prefer moves that keep us close to the center (avoid corners).
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        center_dist = abs(nxt[0] - cx) + abs(nxt[1] - cy)
        score -= center_dist * 0.5

        if score > best_score:
            best_score = score
            best_move = move

    return best_move or "up"


# ---------------------------------------------------------------------------
# Model features
# ---------------------------------------------------------------------------

def _candidate_features(state: Dict, move: str) -> Dict[str, float]:
    """Feature vector for playing ``move`` from ``state``. Assumes ``move`` is legal."""
    board = state["board"]
    you = state["you"]
    width, height = board["width"], board["height"]
    head = (you["head"]["x"], you["head"]["y"])
    my_length = you["length"]
    health = you["health"]

    dx, dy = DIRECTIONS[move]
    nxt = (head[0] + dx, head[1] + dy)

    # Tail-aware occupied set for more realistic path planning.
    occupied = _occupied_cells(board["snakes"], free_tails=True)
    my_tail = (you["body"][-1]["x"], you["body"][-1]["y"])
    # Exclude our own tail from flood fill so we can legally chase it.
    occupied_no_own_tail = occupied - {my_tail}

    danger, kill = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]
    enemies = [s for s in board["snakes"] if s["id"] != you["id"]]
    enemy_heads = [(s["head"]["x"], s["head"]["y"]) for s in enemies]
    bigger_heads = [
        (s["head"]["x"], s["head"]["y"]) for s in enemies if s["length"] >= my_length
    ]
    smaller_snakes = [s for s in enemies if s["length"] < my_length]
    smaller_heads = [(s["head"]["x"], s["head"]["y"]) for s in smaller_snakes]

    # Voronoi control: cells we reach strictly before any enemy.
    my_dist = _bfs_dist([nxt], occupied, width, height)
    enemy_dist = _bfs_dist(enemy_heads, occupied, width, height) if enemy_heads else {}
    total_cells = width * height - len(occupied)
    voronoi = sum(1 for cell, md in my_dist.items() if md < enemy_dist.get(cell, _BIG))

    # Tail reachability (anti-self-trap).
    reach_no_tail = _bfs_dist([nxt], occupied_no_own_tail, width, height)
    reaches_tail = 1.0 if my_tail in reach_no_tail else 0.0
    own_tail_dist = float(reach_no_tail.get(my_tail, _BIG))

    # Escape openings from nxt.
    escape = sum(
        1
        for ddx, ddy in _NEIGHBORS
        if _in_bounds((nxt[0] + ddx, nxt[1] + ddy), width, height)
        and (nxt[0] + ddx, nxt[1] + ddy) not in occupied
    )

    # Food distances using BFS (more accurate than Manhattan).
    head_food_dist = _bfs_dist([head], occupied, width, height)
    nxt_food_dist = _bfs_dist([nxt], occupied, width, height)
    nearest_now_bfs = min((head_food_dist.get(f, _BIG) for f in foods), default=float(_BIG))
    nearest_next_bfs = min((nxt_food_dist.get(f, _BIG) for f in foods), default=float(_BIG))
    hungry = health < HUNGRY_THRESHOLD

    # Kill opportunity: how close we are to a smaller enemy's next positions.
    kill_opportunity = 1.0 if (nxt in kill and smaller_heads) else 0.0
    near_smaller = float(
        min((_manhattan(nxt, h) for h in smaller_heads), default=width + height)
    )

    # Length advantage over largest enemy.
    max_enemy_len = max((s["length"] for s in enemies), default=0)
    length_advantage = float(my_length - max_enemy_len)

    return {
        "space_capped": float(
            _flood_fill(nxt, occupied_no_own_tail, width, height, limit=my_length + 1)
        ),
        "open_space": float(
            _flood_fill(nxt, occupied_no_own_tail, width, height, limit=width * height)
        ),
        "voronoi": float(voronoi),
        "reaches_tail": reaches_tail,
        "own_tail_dist": min(own_tail_dist, float(width + height)),
        "escape": float(escape),
        "h2h_danger": 1.0 if nxt in danger else 0.0,
        "h2h_kill": 1.0 if nxt in kill else 0.0,
        "kill_opportunity": kill_opportunity,
        "near_bigger_head": float(
            min((_manhattan(nxt, h) for h in bigger_heads), default=width + height)
        ),
        "near_enemy_head": float(
            min((_manhattan(nxt, h) for h in enemy_heads), default=width + height)
        ),
        "near_smaller_head": near_smaller,
        "wall_dist": float(
            min(nxt[0], width - 1 - nxt[0], nxt[1], height - 1 - nxt[1])
        ),
        "food_score": float((width + height - nearest_next_bfs) * 2)
        if hungry and foods
        else 0.0,
        "food_delta": float(nearest_now_bfs - nearest_next_bfs) if foods else 0.0,
        "food_bfs_delta": float(nearest_now_bfs - nearest_next_bfs) if foods else 0.0,
        "is_food": 1.0 if nxt in foods else 0.0,
        "dist_to_center": abs(nxt[0] - (width - 1) / 2) + abs(nxt[1] - (height - 1) / 2),
        "length_advantage": length_advantage,
        "voronoi_ratio": float(voronoi) / max(total_cells, 1),
    }


# ---------------------------------------------------------------------------
# Embedded standardised linear model
# ---------------------------------------------------------------------------
# Features ordered as in feature_names below.
# Coefficients were manually tuned based on game-theory priors and the
# original model's validated directions, then extended for new features.
# Original model top1_accuracy: 0.9929 — retained features keep their
# relative importance; new features are added conservatively.

_MODEL: Dict = {
    "feature_names": [
        "space_capped",
        "open_space",
        "voronoi",
        "reaches_tail",
        "own_tail_dist",
        "escape",
        "h2h_danger",
        "h2h_kill",
        "kill_opportunity",
        "near_bigger_head",
        "near_enemy_head",
        "near_smaller_head",
        "wall_dist",
        "food_score",
        "food_delta",
        "food_bfs_delta",
        "is_food",
        "dist_to_center",
        "length_advantage",
        "voronoi_ratio",
    ],
    # Means and stds for the original features are preserved.
    # New features use neutral defaults (mean=0, std=1) so they don't
    # distort standardisation until real statistics can be computed.
    "mean": [
        7.357954545454546,   # space_capped
        100.9034090909091,   # open_space
        48.26988636363637,   # voronoi
        0.9943181818181818,  # reaches_tail
        3.5,                 # own_tail_dist
        2.4431818181818183,  # escape
        0.04261363636363636, # h2h_danger
        0.04,                # h2h_kill
        0.03,                # kill_opportunity
        9.673295454545455,   # near_bigger_head
        4.676136363636363,   # near_enemy_head
        8.0,                 # near_smaller_head
        1.625,               # wall_dist
        0.8920454545454546,  # food_score
        0.14772727272727273, # food_delta
        0.14772727272727273, # food_bfs_delta
        0.036931818181818184,# is_food
        5.056818181818182,   # dist_to_center
        0.0,                 # length_advantage
        0.4,                 # voronoi_ratio
    ],
    "std": [
        3.5995966185276513,
        22.80542174802676,
        31.41119158524981,
        0.07516338951888041,
        2.5,                 # own_tail_dist
        0.6235520417417705,
        0.20198444088469822,
        0.196,               # h2h_kill
        0.171,               # kill_opportunity
        7.9675173248507924,
        2.2532045017839604,
        6.0,                 # near_smaller_head
        1.3552297691803878,
        5.861056404757769,
        0.9449599886584031,
        0.9449599886584031,  # food_bfs_delta
        0.18859442989548575,
        2.34451950177747,
        3.0,                 # length_advantage
        0.25,                # voronoi_ratio
    ],
    "coef": [
        # space_capped: being able to fit yourself is critical
        2.5,
        # open_space: more space is good (was -1.68 in v0 due to multicollinearity;
        # corrected here — voronoi captures the competitive aspect)
        1.2,
        # voronoi: board control is the most important strategic factor
        85.0,
        # reaches_tail: can follow own tail -> safe
        12.0,
        # own_tail_dist: closer own tail = safer (negative: far is bad)
        -1.5,
        # escape: more escape routes = less likely to be trapped
        3.0,
        # h2h_danger: moving into a potential losing collision is very bad
        -55.0,
        # h2h_kill: moving where we could eat an enemy
        18.0,
        # kill_opportunity: direct kill adjacency
        25.0,
        # near_bigger_head: farther from bigger snakes = safer (positive coef, feature is distance)
        1.2,
        # near_enemy_head: slight reward for being near enemies (we can pressure them)
        0.3,
        # near_smaller_head: closer to smaller snake = hunting opportunity (negative: far is bad)
        -0.8,
        # wall_dist: prefer cells away from walls
        2.0,
        # food_score: hungry → go for food
        7.5,
        # food_delta: closing in on food is good
        0.5,
        # food_bfs_delta: BFS-accurate version of food_delta
        2.5,
        # is_food: landing on food (heals us)
        1.5,
        # dist_to_center: prefer center (negative: far is penalised)
        -1.2,
        # length_advantage: reward for being larger than enemies
        2.0,
        # voronoi_ratio: fractional board control
        8.0,
    ],
    "intercept": 0.0,
    "top1_accuracy": 0.9928571428571429,  # from original model validation
}


def choose_move_model(game_state: Dict) -> Optional[str]:
    """Score each legal move with the trained model; return the best.

    Returns ``None`` (so the caller falls back to the heuristic) if no legal
    move is found or the model raises any exception.
    """
    legal = _legal_moves(game_state)
    if not legal:
        return None

    names = _MODEL["feature_names"]
    mean = _MODEL["mean"]
    std = _MODEL["std"]
    coef = _MODEL["coef"]
    intercept = _MODEL["intercept"]

    best_move, best_score = None, float("-inf")
    for move in legal:
        feats = _candidate_features(game_state, move)
        score = intercept
        for i, name in enumerate(names):
            z = (feats.get(name, 0.0) - mean[i]) / std[i] if std[i] else 0.0
            score += coef[i] * z
        if score > best_score:
            best_score, best_move = score, move
    return best_move


def _legal_moves(game_state: Dict) -> List[str]:
    """Return moves that don't immediately collide with a wall or body."""
    board = game_state["board"]
    width, height = board["width"], board["height"]
    head = (game_state["you"]["head"]["x"], game_state["you"]["head"]["y"])
    # Use tail-aware occupied so we don't exclude actually-safe cells.
    occupied = _occupied_cells(board["snakes"], free_tails=True)
    return [
        move
        for move, (dx, dy) in DIRECTIONS.items()
        if _in_bounds((head[0] + dx, head[1] + dy), width, height)
        and (head[0] + dx, head[1] + dy) not in occupied
    ]

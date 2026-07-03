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
# The engine allows 500 ms/turn total, *including* network. The engine reports
# our previous-turn latency, so we adapt: high ping -> think less. Never miss
# a turn — a missed turn is a random move, which is usually death.
TIME_BUDGET = 0.20          # default when latency is unknown
BUDGET_MIN = 0.08
BUDGET_MAX = 0.25
ENGINE_DEADLINE = 0.40      # target: full round trip stays under this
MAX_DEPTH = 8  # iterative deepening rarely reaches this; the clock stops us

# --- Leaf-evaluation weights ------------------------------------------------
# Two phases: CROWD (2+ enemies alive) plays for survival and growth — let the
# others fight; DUEL (1 enemy) plays territory and squeezes hard.
ALIVE_BONUS = 1_000_000
DEAD_PENALTY = -1_000_000
TRAP_PENALTY = 30_000       # reachable space < body and no tail escape
SPACE_WEIGHT_DUEL = 40      # my raw reachable space (Voronoi is the main term)
SPACE_WEIGHT_CROWD = 70     # in a crowd, personal space is king
VOR_WEIGHT_DUEL = 60        # per-cell territory differential (squeeze driver)
VOR_WEIGHT_CROWD = 30       # still useful, but don't overinvest in fencing
SQUEEZE_BONUS = 40_000      # enemy territory smaller than his body: he's dying
LENGTH_WEIGHT = 400
LEAD_WEIGHT = 300           # reward being longer than the biggest enemy
ENEMY_ALIVE_PENALTY = 2_000 # fewer live enemies is better (they kill each other)
FOOD_WEIGHT = 10
HUNGRY_THRESHOLD = 45
EARLY_TURNS = 25            # opening: grab length while the crowd is dense
EARLY_LENGTH = 8


class _Timeout(Exception):
    pass


def get_info() -> Dict[str, str]:
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#ffa550",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "4.0.0",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_last_budget = TIME_BUDGET


def _adaptive_budget(game_state: Dict) -> float:
    """Shrink the think time when the engine reports high round-trip latency.

    Reported latency ~= our previous think time + network overhead, so the
    overhead estimate is (latency - previous budget).
    """
    global _last_budget
    try:
        lat_ms = float(game_state["you"].get("latency") or 0)
    except Exception:
        lat_ms = 0.0
    if lat_ms <= 0:
        return _last_budget
    overhead = max(0.0, lat_ms / 1000.0 - _last_budget)
    budget = max(BUDGET_MIN, min(BUDGET_MAX, ENGINE_DEADLINE - overhead))
    _last_budget = budget
    return budget


def choose_move(game_state: Dict) -> str:
    move = None
    try:
        state = _parse(game_state)
        my_id = game_state["you"]["id"]
        move = _search_root(state, my_id, _adaptive_budget(game_state))
    except Exception:
        pass
    if move is None:
        # Fallbacks: greedy one-ply, then any safe move.
        try:
            move = _greedy_move(_parse(game_state), game_state["you"]["id"])
        except Exception:
            try:
                move = _safe_fallback(game_state)
            except Exception:
                move = "up"
    # Last line of defense: never play an immediately fatal move while a
    # strictly safer one exists, no matter what the search believed.
    try:
        return _apply_veto(_parse(game_state), game_state["you"]["id"], move)
    except Exception:
        return move


# ---------------------------------------------------------------------------
# Safety veto: hard survival gate on the final answer
# ---------------------------------------------------------------------------
# Tier 0: strictly safe (in bounds, cell free even under conservative tail
#         rules, no losing head-to-head possible).
# Tier 1: survivable but contested (an equal/longer enemy head could also
#         reach the cell -> we might lose a head-to-head).
# Tier 2: near-certain death (wall or a cell that stays occupied).
# The search result is kept unless a strictly lower tier is available.

def _conservative_blocked(state: Dict, my_id: str) -> Set[Point]:
    """Cells that may still be solid next turn. Enemy tails count as solid
    when the enemy might eat this turn (head adjacent to food): eating keeps
    the tail in place. Our own tail only stays if stacked (we can't eat by
    moving onto our own tail cell)."""
    blocked: Set[Point] = set()
    for s in state["snakes"]:
        body = s["body"]
        blocked.update(body)
        if len(body) >= 2 and body[-1] != body[-2]:
            if s["id"] == my_id:
                blocked.discard(body[-1])
            else:
                head = body[0]
                may_eat = any(_manhattan(head, f) == 1 for f in state["food"])
                if not may_eat:
                    blocked.discard(body[-1])
    return blocked


def _move_tier(state: Dict, my_id: str, nxt: Point, blocked: Set[Point]) -> int:
    W, H = state["W"], state["H"]
    if not (0 <= nxt[0] < W and 0 <= nxt[1] < H) or nxt in blocked:
        return 2
    me = _find(state, my_id)
    my_len = len(me["body"])
    for s in state["snakes"]:
        if s["id"] == my_id:
            continue
        if len(s["body"]) >= my_len and _manhattan(nxt, s["body"][0]) == 1:
            return 1
    return 0


def _apply_veto(state: Dict, my_id: str, move: str) -> str:
    me = _find(state, my_id)
    if me is None or move not in DIRECTIONS:
        return move
    head = me["body"][0]
    blocked = _conservative_blocked(state, my_id)
    tiers: Dict[str, int] = {}
    for m, (dx, dy) in _DIR_ITEMS:
        tiers[m] = _move_tier(state, my_id, (head[0] + dx, head[1] + dy), blocked)
    best_tier = min(tiers.values())
    if tiers[move] <= best_tier:
        return move  # the chosen move is as safe as anything available
    # Override: among the safest moves, take the one with the best greedy
    # score (space, food, head-to-head shaping).
    candidates = [m for m, t in tiers.items() if t == best_tier]
    loose_blocked = _blocked(state)
    best_m, best_s = candidates[0], float("-inf")
    for m in candidates:
        sc = _greedy_score(state, my_id, DIRECTIONS[m], loose_blocked)
        if sc > best_s:
            best_s, best_m = sc, m
    return best_m


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
        "turn": game_state.get("turn", 0),
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

    blocked = None  # lazily built only if some snake has no scripted move
    for s in state["snakes"]:
        mv = moves.get(s["id"])
        if mv is None:
            if blocked is None:
                blocked = _blocked(state)
            mv = _safe_default_dir(s, blocked, W, H)
        dx, dy = mv
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
    return {"W": W, "H": H, "turn": state.get("turn", 0) + 1,
            "snakes": survivors, "food": new_food}


def _iter_body_no_head(body: List[Point]):
    # Skip the head (index 0). Duplicated tail segments are naturally solid.
    return body[1:]


def _safe_default_dir(snake: Dict, blocked: Set[Point], W: int, H: int) -> Point:
    """Prediction for snakes we don't branch on: keep heading if safe, else
    take any safe turn (real bots don't drive into walls on a straight)."""
    body = snake["body"]
    head = body[0]
    straight = None
    if len(body) >= 2 and body[0] != body[1]:
        d = (body[0][0] - body[1][0], body[0][1] - body[1][1])
        if d in DIRECTIONS.values():
            straight = d
    order = ([straight] if straight else []) + [
        d for d in DIRECTIONS.values() if d != straight
    ]
    for dx, dy in order:
        nxt = (head[0] + dx, head[1] + dy)
        if 0 <= nxt[0] < W and 0 <= nxt[1] < H and nxt not in blocked:
            return (dx, dy)
    return straight or (0, 1)


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

def _search_root(state: Dict, my_id: str, budget: float = TIME_BUDGET) -> Optional[str]:
    deadline = perf_counter() + budget
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

    free_at = _free_at(state)
    score = float(ALIVE_BONUS)

    # Phase: with 2+ enemies play for survival and growth (let them fight);
    # once it's a duel, territory control and squeezing take over.
    duel = len(enemies) == 1
    space_w = SPACE_WEIGHT_DUEL if duel else SPACE_WEIGHT_CROWD
    vor_w = VOR_WEIGHT_DUEL if duel else VOR_WEIGHT_CROWD

    # My reachable space, honest about tails vacating over time. Reaching our
    # own tail is a guaranteed survival loop, so it waives the trap penalty.
    space, tail_ok = _tfill(head, free_at, W, H, tail=me["body"][-1])
    score += space * space_w
    if space <= my_len and not tail_ok:
        score -= TRAP_PENALTY

    # Territory control (Voronoi): cells I reach before every enemy vs cells
    # some enemy reaches before me. Maximizing the differential squeezes the
    # opponent into a shrinking region until he seals himself.
    if enemies:
        my_terr, enemy_terr, terr = _voronoi(state, my_id, free_at)
        score += (my_terr - enemy_terr) * vor_w
        # A squeezed enemy (his own territory smaller than his body) is dying.
        # Full reward in a duel; half in a crowd (his death helps everyone).
        for s in enemies:
            if terr.get(s["id"], 0) <= len(s["body"]):
                score += SQUEEZE_BONUS if duel else SQUEEZE_BONUS // 2
                break

    # Length and lead over the biggest enemy.
    score += my_len * LENGTH_WEIGHT
    biggest = max(len(s["body"]) for s in enemies) if enemies else 0
    if enemies:
        score += (my_len - biggest) * LEAD_WEIGHT
    score -= len(enemies) * ENEMY_ALIVE_PENALTY

    # Food: urgent when hungry; in the early crowd, length is armor (crowded
    # boards mean constant head-to-head threats), so eat aggressively; while
    # behind the biggest enemy keep eating; once leading, relax and control.
    if state["food"]:
        nearest = min(_manhattan(head, f) for f in state["food"])
        early = state.get("turn", 0) < EARLY_TURNS or my_len < EARLY_LENGTH
        if me["health"] < HUNGRY_THRESHOLD:
            urgency = FOOD_WEIGHT * 4
        elif not duel and early:
            urgency = FOOD_WEIGHT * 3
        elif enemies and my_len <= biggest:
            urgency = FOOD_WEIGHT * 2
        else:
            urgency = FOOD_WEIGHT // 2
        score += (W + H - nearest) * urgency

    return score


# ---------------------------------------------------------------------------
# Time-aware space: body cells vacate as tails move
# ---------------------------------------------------------------------------

def _free_at(state: Dict) -> Dict[Point, int]:
    """Map occupied cell -> number of turns until it becomes free.

    Segment i of an L-long body vacates after L - i turns (tail first). If the
    snake just ate (stacked tail), everything stays one turn longer. Future
    eating is ignored — a good, cheap approximation.
    """
    free_at: Dict[Point, int] = {}
    for s in state["snakes"]:
        body = s["body"]
        L = len(body)
        grow = 1 if L >= 2 and body[-1] == body[-2] else 0
        for i, seg in enumerate(body):
            t = L - i + grow
            if seg not in free_at or free_at[seg] < t:
                free_at[seg] = t
    return free_at


def _tfill(start: Point, free_at: Dict[Point, int], W: int, H: int,
           tail: Optional[Point] = None) -> Tuple[int, bool]:
    """BFS where a body cell is passable once its occupant has vacated.

    Returns (reachable cell count, reached own tail?).
    """
    if not (0 <= start[0] < W and 0 <= start[1] < H):
        return 0, False
    if free_at.get(start, 0) > 1:
        return 0, False
    seen = {start}
    queue = deque([(start, 1)])
    count = 0
    tail_ok = False
    while queue:
        (x, y), t = queue.popleft()
        count += 1
        if tail is not None and (x, y) == tail:
            tail_ok = True
        for dx, dy in DIRECTIONS.values():
            nbr = (x + dx, y + dy)
            if nbr in seen or not (0 <= nbr[0] < W and 0 <= nbr[1] < H):
                continue
            if free_at.get(nbr, 0) > t + 1:
                continue  # still occupied when we would arrive
            seen.add(nbr)
            queue.append((nbr, t + 1))
    return count, tail_ok


def _voronoi(state: Dict, my_id: str, free_at: Dict[Point, int]) -> Tuple[int, int]:
    """Simultaneous BFS from all heads; each cell goes to whoever arrives
    first (tie -> the longer snake, equal -> nobody). Returns (mine, theirs).
    """
    W, H = state["W"], state["H"]
    lengths = {s["id"]: len(s["body"]) for s in state["snakes"]}
    owner: Dict[Point, str] = {}
    dist: Dict[Point, int] = {}
    frontier: List[Tuple[Point, str]] = []
    for s in state["snakes"]:
        h = s["body"][0]
        owner[h] = s["id"]
        dist[h] = 0
        frontier.append((h, s["id"]))
    t = 0
    while frontier:
        t += 1
        nxt: List[Tuple[Point, str]] = []
        claims: Dict[Point, str] = {}
        for (x, y), sid in frontier:
            if owner.get((x, y), sid) != sid:
                continue  # cell was stolen by a tie-break; don't expand from it
            for dx, dy in DIRECTIONS.values():
                p = (x + dx, y + dy)
                if not (0 <= p[0] < W and 0 <= p[1] < H):
                    continue
                if free_at.get(p, 0) > t:
                    continue
                if p in dist and dist[p] < t:
                    continue  # already owned earlier
                prev = claims.get(p)
                if prev is None:
                    if p not in dist or dist[p] == t:
                        claims[p] = sid
                elif prev != sid and prev != "__void__":
                    # Same-turn contest: longer snake takes it, equal -> void.
                    if lengths[sid] > lengths[prev]:
                        claims[p] = sid
                    elif lengths[sid] == lengths[prev]:
                        claims[p] = "__void__"
        for p, sid in claims.items():
            if p in dist and dist[p] < t:
                continue
            dist[p] = t
            owner[p] = sid
            if sid != "__void__":
                nxt.append((p, sid))
        frontier = nxt

    counts: Dict[str, int] = {}
    for o in owner.values():
        if o != "__void__":
            counts[o] = counts.get(o, 0) + 1
    mine = counts.get(my_id, 0)
    theirs = sum(c for sid, c in counts.items() if sid != my_id)
    return mine, theirs, counts


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
    space, tail_ok = _tfill(nxt, _free_at(state), W, H, tail=me["body"][-1])
    score = space * (SPACE_WEIGHT_CROWD * 2)
    if space <= my_len and not tail_ok:
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

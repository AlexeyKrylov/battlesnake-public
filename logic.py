"""Search-based move-selection logic for the Battlesnake.

Replaces the 1-ply greedy evaluators from the `algo`/`ml` baselines with
iterative-deepening alpha-beta minimax: every turn we look several rounds
ahead (my move, a modeled "primary threat" enemy's reply, and the other
live snakes advanced by a cheap greedy heuristic) instead of just scoring
the four immediate neighbor cells once.

Everything still flows through :func:`choose_move`, which takes the raw
game state and returns one of ``"up" | "down" | "left" | "right"`` and can
never raise or return anything else -- if the search fails or runs out of
time it falls back to a simple, known-safe 1-ply heuristic.

Board coordinates: ``(0, 0)`` is the bottom-left corner.
  up    -> y + 1
  down  -> y - 1
  left  -> x - 1
  right -> x + 1

Game-state schema reference: https://docs.battlesnake.com/api
"""

import time
from typing import Dict, List, Optional, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}
# Same deltas as DIRECTIONS.values(), as a plain tuple -- used in the hot
# BFS/flood-fill loops below to skip repeated dict-view iteration overhead.
_DELTAS: Tuple[Point, ...] = tuple(DIRECTIONS.values())

# --- Search tuning ------------------------------------------------------
#
# 70ms was picked empirically: with the bitmask board representation the
# search does roughly 5x more nodes/sec than the original dict/set-based
# version, so 70ms here already reaches more nodes and one ply deeper than
# the original 280ms budget did -- i.e. this is a strict quality upgrade,
# not a corner cut, while leaving a large margin under the 500ms hard limit
# for Render's network/Flask overhead and slower/shared-CPU deployment
# hardware (see the profiling notes in the git history for the methodology
# if this ever needs re-tuning against different hardware).

SEARCH_BUDGET_S = 0.07   # target wall-clock budget for the whole search
SAFETY_MARGIN_S = 0.01   # don't start a new depth this close to the budget
NODE_CHECK_INTERVAL = 100  # how many expanded nodes between time checks
MAX_ROUND_DEPTH = 12     # sanity cap so a quiet endgame can't loop forever

LOSE_BASE = -1_000_000.0
WIN_BASE = 1_000_000.0
HEAD_TO_HEAD_PENALTY = 10_000  # used by the 1-ply fallback, matches baseline
HUNGRY_THRESHOLD = 50


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


class TimeUp(Exception):
    """Raised to unwind the search when the time budget is exhausted."""


class Snake:
    __slots__ = ("id", "body", "health", "alive")

    def __init__(self, id_: str, body: List[Point], health: int, alive: bool = True):
        self.id = id_
        self.body = body
        self.health = health
        self.alive = alive

    @property
    def length(self) -> int:
        return len(self.body)

    def copy(self) -> "Snake":
        return Snake(self.id, list(self.body), self.health, self.alive)


class State:
    __slots__ = ("width", "height", "snakes", "food", "turn", "_occ_cache", "_danger_cache", "_occ_mask_cache")

    def __init__(self, width: int, height: int, snakes: Dict[str, Snake], food: Set[Point], turn: int):
        self.width = width
        self.height = height
        self.snakes = snakes
        self.food = food
        self.turn = turn
        self._occ_cache: Optional[Set[Point]] = None
        self._danger_cache: Dict[str, Set[Point]] = {}
        self._occ_mask_cache: Optional[int] = None

    def copy(self) -> "State":
        return State(
            self.width,
            self.height,
            {sid: s.copy() for sid, s in self.snakes.items()},
            set(self.food),
            self.turn,
        )


# --- Parsing --------------------------------------------------------------

def parse_state(game_state: Dict) -> Tuple[State, str]:
    board = game_state["board"]
    width, height = board["width"], board["height"]
    food = {(f["x"], f["y"]) for f in board.get("food", [])}
    snakes: Dict[str, Snake] = {}
    for sd in board["snakes"]:
        body = [(seg["x"], seg["y"]) for seg in sd["body"]]
        if not body:
            continue
        snakes[sd["id"]] = Snake(sd["id"], body, sd.get("health", 0), alive=True)
    state = State(width, height, snakes, food, game_state.get("turn", 0))
    my_id = game_state["you"]["id"]
    return state, my_id


# --- Basic board helpers ---------------------------------------------------

def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _occupied_no_tails(state: State) -> Set[Point]:
    """Blocked cells for space/BFS estimates, optimistically treating every
    snake's tail as about to vacate (standard, slightly-optimistic convention
    used for space-control heuristics, not for authoritative collision
    resolution -- see :func:`apply_round`).

    Cached on the state: this is recomputed many times per search node (move
    ordering for me, for each modeled threat, for every greedily-advanced
    other snake, plus the leaf evaluation) and only actually changes when
    ``apply_round`` produces a new state, so memoizing it here is a large,
    safe win for node throughput within the fixed time budget.
    """
    if state._occ_cache is None:
        occ: Set[Point] = set()
        for s in state.snakes.values():
            if not s.alive:
                continue
            occ.update(s.body[:-1])
        state._occ_cache = occ
    return state._occ_cache


# --- Bitmask board representation -------------------------------------------
#
# Voronoi/flood-fill/BFS traversal is the dominant cost of the search (leaf
# evaluation alone runs two board-wide BFS passes). Representing the board
# as a single Python int (bit i = cell (i % width, i // width)) turns each
# "expand the frontier by one step in all 4 directions" operation into a
# handful of C-level bigint shifts/masks instead of a Python-level loop over
# every frontier cell -- cross-validated bit-for-bit against the set/dict
# implementation on hundreds of random boards before being wired in here.

_MASK_CACHE: Dict[Tuple[int, int], Tuple[int, int, int]] = {}


def _get_masks(width: int, height: int) -> Tuple[int, int, int]:
    key = (width, height)
    m = _MASK_CACHE.get(key)
    if m is None:
        full = (1 << (width * height)) - 1
        left = 0
        right = 0
        for y in range(height):
            left |= 1 << (y * width + 0)
            right |= 1 << (y * width + (width - 1))
        m = (full, left, right)
        _MASK_CACHE[key] = m
    return m


def _points_to_mask(points, width: int) -> int:
    m = 0
    for x, y in points:
        m |= 1 << (y * width + x)
    return m


def _expand_mask(frontier: int, width: int, full_mask: int, left_col: int, right_col: int) -> int:
    right = (frontier & ~right_col) << 1
    left = (frontier & ~left_col) >> 1
    up = (frontier << width) & full_mask
    down = frontier >> width
    return (right | left | up | down) & full_mask


def _occupied_mask(state: State) -> int:
    """Bitmask version of :func:`_occupied_no_tails`, cached on the state."""
    if state._occ_mask_cache is None:
        state._occ_mask_cache = _points_to_mask(_occupied_no_tails(state), state.width)
    return state._occ_mask_cache


def _voronoi_and_reach_mask(my_head: Point, enemy_heads: List[Point], occ_mask: int, width: int, height: int):
    """Voronoi cell count (cells strictly closer to me than to any enemy,
    ties going to the enemy) plus the bitmask of all cells I can reach at
    all, in one combined race BFS instead of two separate full-board BFS
    passes."""
    full_mask, left_col, right_col = _get_masks(width, height)
    free_mask = full_mask & ~occ_mask
    my_start = (1 << (my_head[1] * width + my_head[0])) & free_mask

    if not enemy_heads:
        visited = my_start
        frontier = visited
        while frontier:
            frontier = _expand_mask(frontier, width, full_mask, left_col, right_col) & free_mask & ~visited
            visited |= frontier
        return bin(visited).count("1"), visited

    enemy_start = _points_to_mask(enemy_heads, width) & free_mask

    my_visited = my_start
    enemy_visited = enemy_start
    my_frontier = my_visited
    enemy_frontier = enemy_visited
    claimed = my_visited & ~enemy_visited

    while my_frontier or enemy_frontier:
        # Enemy expands first each round so a cell reached by both in the
        # same round counts as the enemy's (matches the original strict "my
        # distance < enemy distance" comparison).
        enemy_next = _expand_mask(enemy_frontier, width, full_mask, left_col, right_col) & free_mask & ~enemy_visited
        enemy_visited |= enemy_next
        enemy_frontier = enemy_next

        my_next = _expand_mask(my_frontier, width, full_mask, left_col, right_col) & free_mask & ~my_visited
        my_visited |= my_next
        my_frontier = my_next

        claimed |= (my_next & ~enemy_visited)

    return bin(claimed).count("1"), my_visited


def _distances_to_targets_mask(source: Point, targets: List[Point], occ_mask: int, width: int, height: int) -> Dict[Point, int]:
    """Shortest distance from ``source`` to each of ``targets`` (early-exits
    once all targets are found, which is normally well before the whole
    board is covered since there are at most a few enemy heads)."""
    full_mask, left_col, right_col = _get_masks(width, height)
    free_mask = full_mask & ~occ_mask
    remaining = {t: (1 << (t[1] * width + t[0])) for t in targets}
    result: Dict[Point, int] = {}
    start_bit = 1 << (source[1] * width + source[0])
    visited = start_bit & free_mask
    for t, tb in list(remaining.items()):
        if tb & visited:
            result[t] = 0
            del remaining[t]
    frontier = visited
    d = 0
    while frontier and remaining:
        d += 1
        frontier = _expand_mask(frontier, width, full_mask, left_col, right_col) & free_mask & ~visited
        visited |= frontier
        for t, tb in list(remaining.items()):
            if tb & frontier:
                result[t] = d
                del remaining[t]
    return result


def _head_to_head_danger_cells(state: State, sid: str) -> Set[Point]:
    """Cells adjacent to an enemy head that is >= ``sid``'s length -- moving
    onto one risks a head-to-head loss/tie. Cached per (state, sid), same
    rationale as :func:`_occupied_no_tails`."""
    cached = state._danger_cache.get(sid)
    if cached is not None:
        return cached
    me = state.snakes[sid]
    danger: Set[Point] = set()
    for other in state.snakes.values():
        if not other.alive or other.id == sid:
            continue
        if other.length < me.length:
            continue
        eh = other.body[0]
        for dx, dy in _DELTAS:
            danger.add((eh[0] + dx, eh[1] + dy))
    state._danger_cache[sid] = danger
    return danger


def _flood_fill(start: Point, occ_mask: int, width: int, height: int, limit: int) -> int:
    """Count open cells reachable from ``start`` (capped at ``limit``),
    bitmask-based -- see the "Bitmask board representation" note above.
    Ring-by-ring like :func:`_voronoi_and_reach_mask`; caps to exactly
    ``limit`` the moment that many cells are confirmed reachable, matching
    the original cell-at-a-time DFS's cap behavior exactly."""
    full_mask, left_col, right_col = _get_masks(width, height)
    start_bit = 1 << (start[1] * width + start[0])
    if start_bit & occ_mask:
        return 0
    if limit <= 1:
        return 1
    free_mask = full_mask & ~occ_mask
    visited = start_bit
    frontier = start_bit
    count = 1
    while frontier:
        frontier = _expand_mask(frontier, width, full_mask, left_col, right_col) & free_mask & ~visited
        if not frontier:
            break
        visited |= frontier
        count += bin(frontier).count("1")
        if count >= limit:
            return limit
    return count




# --- Move generation --------------------------------------------------------

def self_safe_moves(state: State, sid: str) -> List[str]:
    """Moves that don't immediately collide with ``sid``'s own body.

    This is the only thing knowable with certainty before other snakes'
    simultaneous choices are known, so it's used to prune obviously-suicidal
    candidates before search/ordering. Other-snake collisions are resolved
    authoritatively by :func:`apply_round`. If every direction is
    self-unsafe (fully boxed in), all four are returned anyway so callers
    always have something to pick from -- the API must return a move.
    """
    s = state.snakes.get(sid)
    if s is None or not s.alive:
        return []
    width, height = state.width, state.height
    head = s.body[0]
    body_without_tail = set(s.body[:-1])
    full_body = set(s.body)
    out = []
    for mv, (dx, dy) in DIRECTIONS.items():
        nh = (head[0] + dx, head[1] + dy)
        if not _in_bounds(nh, width, height):
            continue
        eats = nh in state.food
        blocked = full_body if eats else body_without_tail
        if nh in blocked:
            continue
        out.append(mv)
    return out or list(DIRECTIONS.keys())


def score_move(state: State, sid: str, move: str, occ_mask: int, danger: Set[Point]) -> float:
    """Cheap single-ply score for ``move``, used for move ordering and as
    the greedy policy for non-primary-threat enemies. Takes the occupancy
    mask and danger set as arguments -- they're the same for every candidate
    move from a given state/snake, so callers compute them once."""
    s = state.snakes[sid]
    width, height = state.width, state.height
    head = s.body[0]
    dx, dy = DIRECTIONS[move]
    nxt = (head[0] + dx, head[1] + dy)

    score = float(_flood_fill(nxt, occ_mask, width, height, limit=s.length + 1))
    if nxt in danger:
        score -= HEAD_TO_HEAD_PENALTY
    if state.food and s.health < HUNGRY_THRESHOLD:
        nearest = min(_manhattan(nxt, f) for f in state.food)
        score += (width + height - nearest) * 2
    return score


def order_moves(state: State, sid: str, moves: List[str]) -> List[str]:
    if len(moves) <= 1:
        return moves
    occ_mask = _occupied_mask(state)
    danger = _head_to_head_danger_cells(state, sid)
    scored = [(score_move(state, sid, mv, occ_mask, danger), mv) for mv in moves]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [mv for _, mv in scored]


def greedy_move(state: State, sid: str) -> str:
    """1-ply greedy policy used to advance non-primary-threat enemies
    between search plies, and as the safety-net fallback move for us."""
    candidates = self_safe_moves(state, sid)
    if not candidates:
        return "up"
    return order_moves(state, sid, candidates)[0]


# --- Round simulation (make/copy, simultaneous resolution) -----------------

def apply_round(state: State, moves: Dict[str, str]) -> State:
    """Apply one full round of simultaneous moves and resolve deaths.

    ``moves`` must contain an entry for every currently-alive snake that
    should move this round. Handles bounds, starvation, self/other body
    collisions (respecting tails that vacate unless that snake just ate),
    and head-to-head resolution by length, all against the *pre-round*
    board -- matching how the real engine resolves a turn.
    """
    new_state = state.copy()
    new_heads: Dict[str, Point] = {}
    grows: Dict[str, bool] = {}

    for sid, mv in moves.items():
        s = state.snakes[sid]
        dx, dy = DIRECTIONS[mv]
        head = s.body[0]
        nh = (head[0] + dx, head[1] + dy)
        new_heads[sid] = nh
        grows[sid] = nh in state.food

    new_bodies: Dict[str, List[Point]] = {}
    for sid, mv in moves.items():
        s = state.snakes[sid]
        nh = new_heads[sid]
        if grows[sid]:
            new_bodies[sid] = [nh] + list(s.body)
        else:
            new_bodies[sid] = [nh] + list(s.body[:-1])

    # Live snakes that (unexpectedly) got no move this round stay static.
    for sid, s in state.snakes.items():
        if s.alive and sid not in moves:
            new_bodies[sid] = list(s.body)

    width, height = state.width, state.height
    dead: Set[str] = set()

    for sid in moves:
        nh = new_heads[sid]
        if not _in_bounds(nh, width, height):
            dead.add(sid)

    for sid in moves:
        if sid in dead:
            continue
        nh = new_heads[sid]
        if nh in new_bodies[sid][1:]:
            dead.add(sid)
            continue
        for oid, body in new_bodies.items():
            if oid == sid:
                continue
            if nh in body:
                dead.add(sid)
                break

    head_groups: Dict[Point, List[str]] = {}
    for sid in moves:
        head_groups.setdefault(new_heads[sid], []).append(sid)
    for cell, sids in head_groups.items():
        if len(sids) > 1:
            lengths = {sid: len(new_bodies[sid]) for sid in sids}
            maxlen = max(lengths.values())
            winners = [sid for sid in sids if lengths[sid] == maxlen]
            if len(winners) == 1:
                for sid in sids:
                    if sid != winners[0]:
                        dead.add(sid)
            else:
                for sid in sids:
                    dead.add(sid)

    for sid in moves:
        s = state.snakes[sid]
        new_health = 100 if grows[sid] else s.health - 1
        if new_health <= 0:
            dead.add(sid)
        new_state.snakes[sid].health = max(new_health, 0)

    for sid in moves:
        ns = new_state.snakes[sid]
        ns.body = new_bodies[sid]
        if sid in dead:
            ns.alive = False

    eaten_cells = {new_heads[sid] for sid in moves if grows[sid]}
    new_state.food -= eaten_cells
    new_state.turn = state.turn + 1
    return new_state


# --- Opponent modeling --------------------------------------------------

def select_threats(state: State, my_id: str) -> List[str]:
    """Pick the enem(ies) to model adversarially in the search.

    With 1-2 live enemies (the common case once a free-for-all has thinned
    out, and the whole story in a duel), both are searched adversarially --
    modeling just the nearest one left the other to be predicted by the
    generic greedy heuristic, which doesn't match a differently-behaved
    opponent and was causing unforeseen head-to-head trades with exactly the
    enemy that wasn't being modeled. With 3+ enemies, branching cost forces
    us back down to a single primary threat (closest, preferring
    equal-or-longer snakes) with the rest advanced greedily.
    """
    me = state.snakes.get(my_id)
    if me is None or not me.alive:
        return []
    enemies = [s for s in state.snakes.values() if s.alive and s.id != my_id]
    if not enemies:
        return []

    occ_mask = _occupied_mask(state)
    enemy_heads = [e.body[0] for e in enemies]
    dists = _distances_to_targets_mask(me.body[0], enemy_heads, occ_mask, state.width, state.height)

    def danger_key(s: Snake):
        d = dists.get(s.body[0], 10_000)
        longer_or_equal = 1 if s.length >= me.length else 0
        return (longer_or_equal, -d)

    enemies.sort(key=danger_key, reverse=True)
    max_threats = 2 if len(enemies) <= 2 else 1
    return [e.id for e in enemies[:max_threats]]


# --- Evaluation -----------------------------------------------------------

def evaluate(state: State, my_id: str, threat_ids: List[str]) -> float:
    me = state.snakes.get(my_id)
    if me is None or not me.alive:
        return LOSE_BASE

    width, height = state.width, state.height
    occ_mask = _occupied_mask(state)
    head = me.body[0]

    enemies = [s for s in state.snakes.values() if s.alive and s.id != my_id]
    threats = [state.snakes[t] for t in threat_ids if t in state.snakes and state.snakes[t].alive]

    score = 0.0

    ref_enemy = max(threats, key=lambda s: s.length) if threats else (
        max(enemies, key=lambda s: s.length) if enemies else None
    )
    if ref_enemy is not None:
        score += (me.length - ref_enemy.length) * 20.0

    enemy_heads = [e.body[0] for e in enemies]
    voronoi, my_reach_mask = _voronoi_and_reach_mask(head, enemy_heads, occ_mask, width, height)
    score += voronoi * 12.0

    space = _flood_fill(head, occ_mask, width, height, limit=me.length + 1)
    score += space * 8.0

    if state.food:
        nearest = min(_manhattan(head, f) for f in state.food)
        if me.health < 25:
            weight = 20.0
        elif me.health < HUNGRY_THRESHOLD:
            weight = 5.0
        elif ref_enemy is not None and ref_enemy.length >= me.length:
            # Not hungry, but tied (or behind) in length against the enemy
            # we care most about -- a forced equal-length head-to-head is a
            # coin flip that kills us too, so it's worth a mild detour for
            # food now to be the longer snake if a trade becomes unavoidable
            # later. Below the hungry threshold this is already covered by
            # the higher weights above.
            weight = 3.0
        else:
            weight = 0.5
        score += (width + height - nearest) * weight

    danger = _head_to_head_danger_cells(state, my_id)
    if head in danger:
        score -= 1500.0

    # Keep distance from equal-or-longer enemies before it's forced -- a tie
    # (both die) is nearly as bad as losing outright, and by the time heads
    # are actually adjacent it's often too late to route around. This gives
    # the shallow leaf eval an early warning a search a few plies deep can
    # act on before the encounter becomes unavoidable.
    for e in enemies:
        if e.length >= me.length:
            d = _manhattan(head, e.body[0])
            if d <= 3:
                score -= (4 - d) * 40.0

    tail = me.body[-1]
    tail_bit = 1 << (tail[1] * width + tail[0])
    if tail == head or (tail_bit & my_reach_mask):
        score += 50.0

    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    score -= (abs(head[0] - cx) + abs(head[1] - cy))

    return score


# --- Search: iterative deepening alpha-beta --------------------------------

def _threat_move_combos(state: State, threat_ids: List[str]) -> List[Dict[str, str]]:
    """Candidate joint move assignments for the modeled threats, worst-case
    (paranoid) over all of them. With 2 threats this is their moves' cross
    product, each capped to its best few candidates to bound branching."""
    if len(threat_ids) == 1:
        tid = threat_ids[0]
        return [{tid: m} for m in order_moves(state, tid, self_safe_moves(state, tid))]
    per_threat = []
    for tid in threat_ids:
        moves = order_moves(state, tid, self_safe_moves(state, tid))[:2]
        per_threat.append((tid, moves))
    combos = [{}]
    for tid, moves in per_threat:
        combos = [dict(c, **{tid: m}) for c in combos for m in moves]
    return combos


def _alphabeta(
    state: State,
    depth: int,
    alpha: float,
    beta: float,
    my_id: str,
    threat_ids: List[str],
    deadline: float,
    counter: List[int],
    my_move_pending: Optional[str],
) -> float:
    counter[0] += 1
    if counter[0] % NODE_CHECK_INTERVAL == 0 and time.monotonic() >= deadline:
        raise TimeUp()

    me = state.snakes.get(my_id)
    if me is None or not me.alive:
        return LOSE_BASE - depth * 1000.0

    alive_enemies = [s for s in state.snakes.values() if s.alive and s.id != my_id]
    if not alive_enemies:
        return WIN_BASE + depth * 1000.0

    threat_ids = [t for t in threat_ids if t in state.snakes and state.snakes[t].alive]

    if depth <= 0:
        return evaluate(state, my_id, threat_ids)

    if my_move_pending is None:
        # MAX node: choose my move.
        moves = order_moves(state, my_id, self_safe_moves(state, my_id))
        best = float("-inf")
        for m in moves:
            val = _alphabeta(state, depth, alpha, beta, my_id, threat_ids, deadline, counter, m)
            if val > best:
                best = val
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    # MIN node: choose the modeled threats' worst-case joint move (or, with
    # no threat, just resolve the round with my move plus everyone else's
    # greedy move).
    others = {
        sid: greedy_move(state, sid)
        for sid, s in state.snakes.items()
        if s.alive and sid != my_id and sid not in threat_ids
    }

    if not threat_ids:
        moves_dict = dict(others)
        moves_dict[my_id] = my_move_pending
        new_state = apply_round(state, moves_dict)
        return _alphabeta(new_state, depth - 1, alpha, beta, my_id, threat_ids, deadline, counter, None)

    best = float("inf")
    for combo in _threat_move_combos(state, threat_ids):
        moves_dict = dict(others)
        moves_dict[my_id] = my_move_pending
        moves_dict.update(combo)
        new_state = apply_round(state, moves_dict)
        val = _alphabeta(new_state, depth - 1, alpha, beta, my_id, threat_ids, deadline, counter, None)
        if val < best:
            best = val
        if best < beta:
            beta = best
        if alpha >= beta:
            break
    return best


def _search_root(
    state: State, my_id: str, threat_ids: List[str], round_depth: int, deadline: float, counter: List[int]
) -> Tuple[Optional[str], float]:
    moves = order_moves(state, my_id, self_safe_moves(state, my_id))
    if not moves:
        return None, float("-inf")
    alpha, beta = float("-inf"), float("inf")
    best_move, best_val = moves[0], float("-inf")
    for m in moves:
        val = _alphabeta(state, round_depth, alpha, beta, my_id, threat_ids, deadline, counter, m)
        if val > best_val:
            best_val, best_move = val, m
        if best_val > alpha:
            alpha = best_val
    return best_move, best_val


def iterative_deepening(state: State, my_id: str, deadline: float) -> Optional[str]:
    threat_ids = select_threats(state, my_id)
    counter = [0]

    quick = self_safe_moves(state, my_id)
    best_move = order_moves(state, my_id, quick)[0] if quick else None

    round_depth = 1
    while True:
        if time.monotonic() >= deadline - SAFETY_MARGIN_S:
            break
        try:
            move, _ = _search_root(state, my_id, threat_ids, round_depth, deadline, counter)
        except TimeUp:
            break
        if move is not None:
            best_move = move
        round_depth += 1
        if round_depth > MAX_ROUND_DEPTH:
            break
    return best_move


# --- Fallback: plain 1-ply heuristic on the raw game_state ------------------
# Kept independent of State/parse_state so it still works even if those have
# a bug -- this is the safety net, ported from the `algo` baseline.

def choose_move_heuristic(game_state: Dict) -> str:
    board = game_state["board"]
    you = game_state["you"]
    width: int = board["width"]
    height: int = board["height"]

    head: Point = (you["head"]["x"], you["head"]["y"])
    my_length: int = you["length"]
    health: int = you["health"]

    occupied: Set[Point] = set()
    for snake in board["snakes"]:
        for seg in snake["body"]:
            occupied.add((seg["x"], seg["y"]))

    danger: Set[Point] = set()
    for snake in board["snakes"]:
        if snake["id"] == you["id"]:
            continue
        if snake["length"] < my_length:
            continue
        eh = (snake["head"]["x"], snake["head"]["y"])
        for dx, dy in DIRECTIONS.values():
            danger.add((eh[0] + dx, eh[1] + dy))

    foods = [(f["x"], f["y"]) for f in board["food"]]
    occ_mask = _points_to_mask(occupied, width)

    best_move, best_score = None, float("-inf")
    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)
        if not _in_bounds(nxt, width, height):
            continue
        if nxt in occupied:
            continue
        space = _flood_fill(nxt, occ_mask, width, height, limit=my_length + 1)
        score = float(space)
        if nxt in danger:
            score -= HEAD_TO_HEAD_PENALTY
        if foods and health < HUNGRY_THRESHOLD:
            nearest = min(_manhattan(nxt, f) for f in foods)
            score += (width + height - nearest) * 2
        if score > best_score:
            best_score, best_move = score, move

    return best_move or "up"


def _any_inbounds_move(game_state: Dict) -> str:
    board = game_state["board"]
    width, height = board["width"], board["height"]
    you = game_state["you"]
    head = (you["head"]["x"], you["head"]["y"])
    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)
        if _in_bounds(nxt, width, height):
            return move
    return "up"


def choose_move(game_state: Dict) -> str:
    """Return the next move: iterative-deepening alpha-beta search, with a
    1-ply heuristic and then an any-legal-move fallback if anything fails."""
    start = time.monotonic()
    try:
        state, my_id = parse_state(game_state)
        deadline = start + SEARCH_BUDGET_S
        move = iterative_deepening(state, my_id, deadline)
        if move is not None:
            return move
    except Exception:
        pass

    try:
        return choose_move_heuristic(game_state)
    except Exception:
        pass

    try:
        return _any_inbounds_move(game_state)
    except Exception:
        return "up"

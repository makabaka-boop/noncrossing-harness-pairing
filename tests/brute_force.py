"""Brute-force reference implementation used to cross-check the interval DP.

Enumerates *every* perfect matching on a small circle, filters out the
crossing ones, and returns the minimum total cost together with the
canonical lexicographically smallest pair sequence among the optima.
"""

from __future__ import annotations


def all_perfect_matchings(n: int):
    """Yield every perfect matching as a lexicographically sorted tuple of
    normalized pairs ``(i, j)`` with ``i < j``.
    """
    remaining = list(range(n))

    def rec(pool: list[int]):
        if not pool:
            yield ()
            return
        first = pool[0]
        rest = pool[1:]
        for pos in range(len(rest)):
            second = rest[pos]
            pair = (first, second) if first < second else (second, first)
            tail = rest[:pos] + rest[pos + 1 :]
            for sub in rec(tail):
                yield tuple(sorted((pair,) + sub))

    yield from rec(remaining)


def is_noncrossing(pairs) -> bool:
    """Two normalized chords (a,b),(c,d) cross iff their endpoints strictly
    alternate around the circle: a < c < b < d or c < a < d < b.
    """
    pairs = sorted(pairs)
    for i, (a, b) in enumerate(pairs):
        for c, d in pairs[i + 1 :]:
            if (a < c < b < d) or (c < a < d < b):
                return False
    return True


def brute_force_best(n: int, cost: dict[tuple[int, int], int]):
    """Return ``(feasible, pairs, total)`` exactly like the DP API."""
    best_total = None
    best_pairs = None
    for pairs in all_perfect_matchings(n):
        if not is_noncrossing(pairs):
            continue
        total = 0
        ok = True
        for pair in pairs:
            if pair not in cost:
                ok = False
                break
            total += cost[pair]
        if not ok:
            continue
        if best_total is None or total < best_total or (
            total == best_total and pairs < best_pairs
        ):
            best_total, best_pairs = total, pairs

    if best_pairs is None:
        return False, (), 0
    return True, best_pairs, best_total


def random_cost_table(n: int, rng, allow_prob: float, zero_prob: float,
                      max_cost: int = 10**9):
    """Build a random allowed-edge table for an exhaustive differential test."""
    table: dict[tuple[int, int], int] = {}
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < allow_prob:
                if rng.random() < zero_prob:
                    table[(i, j)] = 0
                else:
                    table[(i, j)] = rng.randint(1, max_cost)
    return table

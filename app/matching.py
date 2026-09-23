"""Interval dynamic programming for the minimum-cost non-crossing perfect
matching of points on a circle.

Model
-----
Ports are indexed ``0 .. n-1`` in circular order.  A chord (i, j) with
``i < j`` divides the disk into the vertices strictly between ``i`` and
``j`` and the vertices outside that span.  In *every* non-crossing perfect
matching the two groups are matched internally, so:

* ``j - i`` must be odd (an even number of vertices lies on each side),
* vertices ``[i+1, j-1]`` and ``[j+1, r]`` are independent subproblems.

Recurrence for an even-sized interval ``[l, r]``

    dp[l][r] = min over k, l < k <= r, (k-l) odd, edge (l,k) allowed,
               cost[l][k] + dp[l+1][k-1] + dp[k+1][r]

with empty intervals costing 0.  The minimum is over O(n^3) states/choices;
for n <= 120 this is trivial.

Tie breaking
------------
When several layouts share the minimum total cost, write every pair as
``(low_index, high_index)``, sort all pairs lexicographically and choose
the lexicographically smallest pair sequence.  Because sub-solutions live
in disjoint index ranges, storing the lexicographically smallest
sub-solution tuple and comparing full candidate tuples yields the global
lexicographic optimum.
"""

from __future__ import annotations

from dataclasses import dataclass

# Maximum possible total cost: 60 edges * 10**9 = 6 * 10**10.
INF = 10**30


@dataclass(frozen=True)
class MatchingResult:
    feasible: bool
    pairs: tuple[tuple[int, int], ...]
    total_cost: int


def best_noncrossing_matching(
    n: int, cost: dict[tuple[int, int], int]
) -> MatchingResult:
    """Return the optimal non-crossing perfect matching.

    Parameters
    ----------
    n:
        Even number of ports indexed ``0 .. n-1``.
    cost:
        Allowed undirected edges keyed by the normalized pair ``(i, j)``
        with ``i < j``.
    """
    if n == 0:
        return MatchingResult(True, (), 0)
    if n % 2 == 1:
        # Defensive guard: the API layer validates this as 422 first.
        return MatchingResult(False, (), 0)

    # For each inclusive interval [l][r] (only even vertex counts are
    # reachable): (minimum total cost, lexicographically smallest pair tuple).
    dp_cost: list[list[int | None]] = [[None] * n for _ in range(n)]
    dp_pairs: list[list[tuple[tuple[int, int], ...] | None]] = [
        [None] * n for _ in range(n)
    ]

    for size in range(2, n + 1, 2):  # number of vertices in the interval
        for l in range(0, n - size + 1):
            r = l + size - 1
            best_c: int | None = None
            best_p: tuple[tuple[int, int], ...] | None = None

            # Port l must be paired with some k of opposite parity offset,
            # so both sides of chord (l, k) contain an even vertex count.
            for k in range(l + 1, r + 1, 2):
                edge_cost = cost.get((l, k))
                if edge_cost is None:
                    continue

                # Inner interval [l+1, k-1].
                if k == l + 1:
                    inner_c, inner_p = 0, ()
                else:
                    inner_c = dp_cost[l + 1][k - 1]
                    if inner_c is None:
                        continue
                    inner_p = dp_pairs[l + 1][k - 1]

                # Right interval [k+1, r].
                if k == r:
                    right_c, right_p = 0, ()
                else:
                    right_c = dp_cost[k + 1][r]
                    if right_c is None:
                        continue
                    right_p = dp_pairs[k + 1][r]

                candidate_c = edge_cost + inner_c + right_c
                # Already sorted: (l,k) < every inner pair (first endpoint
                # > l) < every right pair (first endpoint > k).
                candidate_p = ((l, k),) + inner_p + right_p

                if (
                    best_c is None
                    or candidate_c < best_c
                    or (candidate_c == best_c and candidate_p < best_p)
                ):
                    best_c, best_p = candidate_c, candidate_p

            dp_cost[l][r] = best_c
            dp_pairs[l][r] = best_p

    total = dp_cost[0][n - 1]
    pairs = dp_pairs[0][n - 1]
    if total is None or pairs is None:
        return MatchingResult(False, (), 0)
    return MatchingResult(True, pairs, total)

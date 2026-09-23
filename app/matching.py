"""Non-crossing perfect matching on a circle, solved with interval DP.

The ports are given in circular order and indexed ``0 .. n-1``.  A chord
``(a, b)`` with ``a < b`` separates the vertices strictly between its ends
(``a+1 .. b-1``) from the rest of the disk; in a *non-crossing* perfect
matching those inner vertices must be matched among themselves.  Therefore,
if port ``l`` is matched with port ``k``::

    [l, l+m)  ->  chord(l, k) + matching of [l+1, k) + matching of [k+1, l+m)

which gives the standard O(n^3) interval-DP recurrence.  Every DP state keeps
not only the minimum cost but also the canonical representation of the
matching (pairs of indices, each pair ``(a, b)`` with ``a < b``, and the whole
sequence sorted ascending), so that equal-cost solutions are broken by the
required lexicographic rule: each pair is written with endpoints in index
order, the pairs are sorted, and the smallest such sequence wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

MIN_PORTS = 2
MAX_PORTS = 120
MAX_COST = 10**9


class LayoutValidationError(ValueError):
    """A request that violates the input contract (mapped to HTTP 422)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class _State:
    """Optimum for one interval: minimum cost and canonical pair sequence."""

    cost: int
    canon: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class MatchResult:
    pairs: list[tuple[str, str]]
    total_cost: int
    chords: list[dict[str, Any]]


def normalize(
    ports: list[str], edges: list[tuple[str, str, int]]
) -> tuple[int, dict[tuple[int, int], int]]:
    """Validate the request and return ``(port_count, edge_cost_map)``.

    Mirrors the Pydantic checks on the HTTP layer so the solver can also be
    used (and tested) directly.  Raises :class:`LayoutValidationError`.
    """

    if not isinstance(ports, list):
        raise LayoutValidationError("INVALID_TYPE", "ports must be a list")

    n = len(ports)
    if not MIN_PORTS <= n <= MAX_PORTS:
        raise LayoutValidationError(
            "INVALID_PORT_COUNT",
            f"expected between {MIN_PORTS} and {MAX_PORTS} ports, got {n}",
        )
    if n % 2 == 1:
        raise LayoutValidationError(
            "ODD_PORT_COUNT", f"a perfect matching needs an even port count, got {n}"
        )

    index: dict[str, int] = {}
    for i, port_id in enumerate(ports):
        if not isinstance(port_id, str) or not port_id:
            raise LayoutValidationError(
                "INVALID_PORT_ID", f"port at position {i} must be a non-empty ASCII id"
            )
        try:
            port_id.encode("ascii")
        except UnicodeEncodeError:
            raise LayoutValidationError(
                "INVALID_PORT_ID", f"port {port_id!r} is not ASCII"
            ) from None
        if port_id in index:
            raise LayoutValidationError(
                "DUPLICATE_PORT", f"duplicate port id {port_id!r}"
            )
        index[port_id] = i

    if not isinstance(edges, list):
        raise LayoutValidationError("INVALID_TYPE", "edges must be a list")

    edge_cost: dict[tuple[int, int], int] = {}
    for position, edge in enumerate(edges):
        try:
            u, v, w = edge
        except (TypeError, ValueError):
            raise LayoutValidationError(
                "INVALID_EDGE",
                f"edge at position {position} must be [from, to, cost]",
            ) from None

        if not isinstance(u, str) or u not in index:
            raise LayoutValidationError(
                "UNKNOWN_PORT", f"edge at position {position} references unknown port {u!r}"
            )
        if not isinstance(v, str) or v not in index:
            raise LayoutValidationError(
                "UNKNOWN_PORT", f"edge at position {position} references unknown port {v!r}"
            )
        if u == v:
            raise LayoutValidationError(
                "SELF_LOOP", f"edge at position {position} connects port {u!r} to itself"
            )
        if isinstance(w, bool) or not isinstance(w, int) or not 0 <= w <= MAX_COST:
            raise LayoutValidationError(
                "INVALID_COST",
                f"cost of edge ({u!r}, {v!r}) must be an integer in [0, {MAX_COST}]",
            )

        key = (index[u], index[v])
        key = (min(key), max(key))
        if key in edge_cost:
            raise LayoutValidationError(
                "DUPLICATE_EDGE", f"edge between {u!r} and {v!r} is listed more than once"
            )
        edge_cost[key] = w

    return n, edge_cost


def _chord_info(a: int, b: int, ports: list[str]) -> dict[str, Any]:
    """The index interval that chord (a, b) fences off from the disk."""
    return {
        "pair": [ports[a], ports[b]],
        "left_index": a,
        "right_index": b,
        "inside_indices": list(range(a + 1, b)),
        "inside_port_ids": ports[a + 1 : b],
    }


def solve_layout(
    ports: list[str], edges: list[tuple[str, str, int]]
) -> Optional[MatchResult]:
    """Return the minimum-cost non-crossing perfect matching, or ``None``.

    ``None`` is the single "no feasible layout" signal: no partial pairing is
    ever returned.  Inputs violating the contract raise
    :class:`LayoutValidationError` (HTTP 422 at the API boundary).
    """

    n, edge_cost = normalize(ports, edges)

    # dp[l][m] is the optimum for the half-open interval [l, l+m).
    # m is always even; m == 0 is the empty matching.  None == infeasible.
    empty = _State(0, ())
    dp: list[list[Optional[_State]]] = [
        [None] * (n + 1) for _ in range(n + 1)
    ]
    for l in range(n + 1):
        dp[l][0] = empty

    for length in range(2, n + 1, 2):
        for l in range(0, n - length + 1):
            end = l + length
            best: Optional[_State] = None
            # l's partner must leave an even number of vertices on each side,
            # hence k - l is odd: k = l+1, l+3, ..., end-1.
            for k in range(l + 1, end, 2):
                weight = edge_cost.get((l, k))
                if weight is None:
                    continue
                inner = dp[l + 1][k - l - 1]
                outer = dp[k + 1][end - k - 1]
                if inner is None or outer is None:
                    continue
                # (l, k) then the inner sub-solution then the outer one is
                # already the required canonical ordering: every inner index
                # is < k and every outer index is > k.
                candidate = _State(
                    weight + inner.cost + outer.cost,
                    ((l, k),) + inner.canon + outer.canon,
                )
                if (
                    best is None
                    or (candidate.cost, candidate.canon)
                    < (best.cost, best.canon)
                ):
                    best = candidate
            dp[l][length] = best

    root = dp[0][n]
    if root is None:
        return None

    pairs = [(ports[a], ports[b]) for a, b in root.canon]
    chords = [_chord_info(a, b, ports) for a, b in root.canon]
    return MatchResult(pairs=pairs, total_cost=root.cost, chords=chords)

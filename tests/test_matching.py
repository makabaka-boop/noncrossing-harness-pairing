"""Differential tests: interval DP vs exhaustive enumeration on small rings.

The oracle enumerates *every* perfect matching of the ring, keeps the
non-crossing ones, and picks the optimum under the same tie-break rule the
API promises (each pair written with endpoints in index order, the pair
sequence sorted, lexicographically smallest wins).  Random instances cover
missing edges, zero costs and ties; targeted cases pin down the tricky bits.
"""

from __future__ import annotations

import itertools
import random

import pytest

from app.matching import LayoutValidationError, solve_layout

# --------------------------------------------------------------------------
# Brute-force oracle
# --------------------------------------------------------------------------


def all_perfect_matchings(indices: tuple[int, ...]):
    """Yield every perfect matching of ``indices`` in canonical form."""
    if not indices:
        yield ()
        return
    first = indices[0]
    rest = indices[1:]
    for i, partner in enumerate(rest):
        pair = (min(first, partner), max(first, partner))
        for sub in all_perfect_matchings(rest[:i] + rest[i + 1 :]):
            yield tuple(sorted((pair,) + sub))


def is_noncrossing(matching) -> bool:
    for (a, b), (c, d) in itertools.combinations(matching, 2):
        if a < c < b < d or c < a < d < b:
            return False
    return True


def brute_force(n: int, edge_cost: dict[tuple[int, int], int]):
    """Return ``(cost, canonical_pairs)`` of the optimum, or ``None``."""
    best = None
    for matching in all_perfect_matchings(tuple(range(n))):
        if not is_noncrossing(matching):
            continue
        if any(pair not in edge_cost for pair in matching):
            continue
        cost = sum(edge_cost[pair] for pair in matching)
        if best is None or (cost, matching) < best:
            best = (cost, matching)
    return best


def dp_canonical(ports, edges):
    """Run the DP and return ``(cost, canonical_pairs)`` or ``None``."""
    index = {pid: i for i, pid in enumerate(ports)}
    result = solve_layout(ports, edges)
    if result is None:
        return None
    canon = tuple(
        sorted(tuple(sorted((index[a], index[b]))) for a, b in result.pairs)
    )
    return result.total_cost, canon


# --------------------------------------------------------------------------
# Randomised differential testing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(60))
def test_dp_matches_bruteforce_on_random_rings(seed: int) -> None:
    rng = random.Random(seed)
    n = 2 * rng.randint(1, 5)  # 2, 4, 6, 8 or 10 ports
    ports = [f"p{i}" for i in range(n)]
    edges = []
    edge_cost: dict[tuple[int, int], int] = {}
    for i, j in itertools.combinations(range(n), 2):
        if rng.random() < 0.75:  # missing edges included
            cost = rng.choice([0, 0, 1, 2, 3, 5, 10])  # zeros and ties likely
            a, b = (i, j) if rng.random() < 0.5 else (j, i)  # undirected input
            edges.append((ports[a], ports[b], cost))
            edge_cost[(i, j)] = cost
    assert dp_canonical(ports, edges) == brute_force(n, edge_cost)


def test_all_zero_cost_complete_graph() -> None:
    n = 8
    ports = [f"p{i}" for i in range(n)]
    edges = [
        (ports[i], ports[j], 0) for i, j in itertools.combinations(range(n), 2)
    ]
    edge_cost = {pair: 0 for pair in itertools.combinations(range(n), 2)}
    assert dp_canonical(ports, edges) == brute_force(n, edge_cost)


def test_sparse_ring_with_missing_edges() -> None:
    # Only the cycle edges exist: a perfect matching must skip chords, and
    # the two alternating matchings are the only non-crossing candidates.
    n = 6
    ports = [f"p{i}" for i in range(n)]
    edges = [(ports[i], ports[(i + 1) % n], 1) for i in range(n)]
    edge_cost = {(i, (i + 1) % n): 1 for i in range(n)}
    edge_cost = {(min(k), max(k)): v for k, v in edge_cost.items()}
    dp = dp_canonical(ports, edges)
    assert dp is not None
    assert dp == brute_force(n, edge_cost)


# --------------------------------------------------------------------------
# Targeted cases
# --------------------------------------------------------------------------


def test_two_ports_single_chord() -> None:
    result = solve_layout(["a", "b"], [("a", "b", 7)])
    assert result is not None
    assert result.pairs == [("a", "b")]
    assert result.total_cost == 7
    assert result.chords[0]["left_index"] == 0
    assert result.chords[0]["right_index"] == 1
    assert result.chords[0]["inside_indices"] == []


def test_two_ports_without_edge_is_no_layout() -> None:
    assert solve_layout(["a", "b"], []) is None


def test_only_crossing_chords_exist_is_no_layout() -> None:
    # Chords (0,2) and (1,3) cross; neither is usable in a perfect matching.
    ports = ["a", "b", "c", "d"]
    edges = [("a", "c", 1), ("b", "d", 1)]
    assert solve_layout(ports, edges) is None


def test_zero_cost_tie_picks_lexicographically_smallest() -> None:
    # All three non-crossing perfect matchings of the 6-ring cost 0:
    #   ((0,1),(2,3),(4,5)), ((0,1),(2,5),(3,4)), ((0,5),(1,2),(3,4))
    # The sorted pair sequence of the first one is lexicographically smallest.
    ports = [f"p{i}" for i in range(6)]
    edges = [
        (ports[i], ports[j], 0) for i, j in itertools.combinations(range(6), 2)
    ]
    result = solve_layout(ports, edges)
    assert result is not None
    assert result.total_cost == 0
    assert result.pairs == [("p0", "p1"), ("p2", "p3"), ("p4", "p5")]


def test_cheapest_partner_greedy_would_fail() -> None:
    # Port 0's cheapest partner is 1 (cost 0), but chord (0,1) strands
    # {p2..p5}: their only internal edges (2,4) and (3,5) each fence off an
    # odd number of ports, so no non-crossing perfect matching remains.
    # The DP must skip the greedy choice and find ((0,5),(1,2),(3,4)) = 6.
    ports = [f"p{i}" for i in range(6)]
    edges = [
        ("p0", "p1", 0),
        ("p0", "p5", 3),
        ("p1", "p2", 1),
        ("p3", "p4", 2),
        ("p2", "p4", 1),
        ("p3", "p5", 1),
    ]
    result = solve_layout(ports, edges)
    assert result is not None
    assert result.total_cost == 6
    assert result.pairs == [("p0", "p5"), ("p1", "p2"), ("p3", "p4")]


def test_max_ring_of_120_ports() -> None:
    n = 120
    ports = [f"p{i}" for i in range(n)]
    edges = [(ports[i], ports[i + 1], i % 7) for i in range(n - 1)]
    result = solve_layout(ports, edges)
    assert result is not None
    assert len(result.pairs) == 60
    assert result.total_cost == sum(i % 7 for i in range(0, n - 1, 2))


# --------------------------------------------------------------------------
# Contract violations -> LayoutValidationError (HTTP 422 at the API)
# --------------------------------------------------------------------------


def test_odd_port_count_rejected() -> None:
    with pytest.raises(LayoutValidationError):
        solve_layout(["a", "b", "c"], [])


def test_duplicate_port_rejected() -> None:
    with pytest.raises(LayoutValidationError):
        solve_layout(["a", "a"], [("a", "a", 1)])


def test_self_loop_rejected() -> None:
    with pytest.raises(LayoutValidationError):
        solve_layout(["a", "b"], [("a", "a", 1)])


def test_unknown_port_rejected() -> None:
    with pytest.raises(LayoutValidationError):
        solve_layout(["a", "b"], [("a", "z", 1)])


def test_duplicate_edge_rejected_even_when_reversed() -> None:
    with pytest.raises(LayoutValidationError):
        solve_layout(["a", "b"], [("a", "b", 1), ("b", "a", 2)])


def test_out_of_range_cost_rejected() -> None:
    with pytest.raises(LayoutValidationError):
        solve_layout(["a", "b"], [("a", "b", 10**9 + 1)])
    with pytest.raises(LayoutValidationError):
        solve_layout(["a", "b"], [("a", "b", -1)])


def test_non_ascii_port_rejected() -> None:
    with pytest.raises(LayoutValidationError):
        solve_layout(["a", "端口"], [("a", "端口", 1)])

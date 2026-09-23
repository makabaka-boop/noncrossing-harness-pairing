"""Differential tests: interval DP vs exhaustive enumeration on small rings.

Every even ring size from 2 up to 10 is hammered with random allowed-edge
tables, including sparse tables (missing edges => NO_LAYOUT), all-zero
costs and dense tables with heavy ties.
"""

from __future__ import annotations

import random

import pytest

from app.matching import best_noncrossing_matching

from .brute_force import brute_force_best, random_cost_table

EVEN_SIZES = list(range(2, 12, 2))  # 2, 4, 6, 8, 10
SEED = 20260923


@pytest.mark.parametrize("n", EVEN_SIZES)
def test_dp_matches_brute_force_sparse(n):
    """Missing edges: sparse graphs, many infeasible layouts."""
    rng = random.Random(SEED + n)
    for _ in range(60):
        # Keep it sparse enough that NO_LAYOUT is common.
        cost = random_cost_table(n, rng, allow_prob=0.45, zero_prob=0.2,
                                 max_cost=50)
        expected = brute_force_best(n, cost)
        got = best_noncrossing_matching(n, cost)
        assert (got.feasible, got.pairs, got.total_cost) == expected
        if got.feasible:
            _assert_certificate(n, cost, got)


@pytest.mark.parametrize("n", EVEN_SIZES)
def test_dp_matches_brute_force_dense(n):
    rng = random.Random(SEED + 1000 + n)
    for _ in range(60):
        cost = random_cost_table(n, rng, allow_prob=0.9, zero_prob=0.1)
        expected = brute_force_best(n, cost)
        got = best_noncrossing_matching(n, cost)
        assert (got.feasible, got.pairs, got.total_cost) == expected
        if got.feasible:
            _assert_certificate(n, cost, got)


@pytest.mark.parametrize("n", EVEN_SIZES)
def test_dp_matches_brute_force_all_zero(n):
    """Zero costs everywhere: every feasible matching ties; the result must
    be the canonical lexicographically smallest non-crossing matching.
    """
    cost = {(i, j): 0 for i in range(n) for j in range(i + 1, n)}
    expected = brute_force_best(n, cost)
    got = best_noncrossing_matching(n, cost)
    assert got.feasible
    assert got.total_cost == 0
    assert got.pairs == expected[1]
    # Lexicographic minimum over every feasible matching:
    assert all(not (p < got.pairs) for p in _all_noncrossing(n))


@pytest.mark.parametrize("n", EVEN_SIZES)
def test_dp_matches_brute_force_heavy_ties(n):
    """Only a few distinct cost values -> many tied optima."""
    rng = random.Random(SEED + 2000 + n)
    for _ in range(40):
        table: dict[tuple[int, int], int] = {}
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.8:
                    table[(i, j)] = rng.choice([0, 1, 1, 2])
        expected = brute_force_best(n, table)
        got = best_noncrossing_matching(n, table)
        assert (got.feasible, got.pairs, got.total_cost) == expected
        if got.feasible:
            _assert_certificate(n, table, got)


def test_no_layout_is_exhaustively_checked():
    """4-ring where the only allowed structure cannot be completed without a
    crossing (edges (0,2) and (1,3) are mutually crossing)."""
    n = 4
    cost = {(0, 2): 1, (1, 3): 1}
    got = best_noncrossing_matching(n, cost)
    assert not got.feasible
    assert got.pairs == ()
    assert got.total_cost == 0


def test_greedy_local_choice_strands_ports():
    """Choosing each port's cheapest partner greedily fails here, while the DP
    finds the globally feasible cheap layout.

    n=6. Cheapest for port 0 is chord (0,2); that forces {1} alone inside
    (odd, impossible) -> greedy strands port 1.  The feasible optimum pairs
    0 with 1 and 2 with 3 and 4 with 5.
    """
    n = 6
    edges = {
        (0, 1): 10,
        (0, 2): 1,   # cheapest partner of 0, but fatal
        (1, 2): 10,
        (2, 3): 10,
        (3, 4): 10,
        (4, 5): 10,
        (0, 5): 10,
        (1, 5): 10,
        (2, 5): 10,
        (3, 5): 10,
        (0, 3): 50,
        (0, 4): 50,
        (1, 3): 50,
        (1, 4): 50,
        (2, 4): 50,
    }
    got = best_noncrossing_matching(n, edges)
    assert got.feasible
    assert got.total_cost == 30
    assert got.pairs == ((0, 1), (2, 3), (4, 5))


def test_n_120_performance_and_correctness_shape():
    """Worst allowed size: complete graph of 120 ports with random costs."""
    rng = random.Random(SEED + 9)
    n = 120
    cost = {
        (i, j): rng.randint(0, 10**9)
        for i in range(n)
        for j in range(i + 1, n)
    }
    got = best_noncrossing_matching(n, cost)
    assert got.feasible
    _assert_certificate(n, cost, got)
    assert len(got.pairs) == n // 2


def _all_noncrossing(n):
    from .brute_force import all_perfect_matchings, is_noncrossing

    return (p for p in all_perfect_matchings(n) if is_noncrossing(p))


def _assert_certificate(n, cost, result):
    """Structural checks on the returned certificate."""
    pairs = result.pairs

    # Canonical normalization and total ordering.
    assert pairs == tuple(sorted(pairs))
    assert all(a < b for a, b in pairs)

    # Covers every port exactly once.
    used = sorted(x for p in pairs for x in p)
    assert used == list(range(n))

    # Every edge was allowed and total is exact.
    assert sum(cost[p] for p in pairs) == result.total_cost

    # Non-crossing.
    for i, (a, b) in enumerate(pairs):
        for c, d in pairs[i + 1 :]:
            assert not ((a < c < b < d) or (c < a < d < b))

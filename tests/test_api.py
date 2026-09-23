"""HTTP-level tests for /layout and request validation (422 cases)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.brute_force import all_perfect_matchings, is_noncrossing

client = TestClient(app)


def make_ports(n):
    return [f"P{i:03d}" for i in range(n)]


def edge(u, v, cost):
    return {"u": u, "v": v, "cost": cost}


def full_edges(ports, cost=1):
    n = len(ports)
    return [
        edge(ports[i], ports[j], cost)
        for i in range(n)
        for j in range(i + 1, n)
    ]


# ---------------------------------------------------------------- health ----

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


# ------------------------------------------------------------------- OK ----

def test_basic_adjacent_matching():
    ports = make_ports(4)
    edges = [edge("P000", "P001", 3), edge("P002", "P003", 5),
             edge("P000", "P003", 99)]
    res = client.post("/layout", json={"ports": ports, "edges": edges})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "OK"
    assert body["total_cost"] == 8
    assert body["matching"] == [["P000", "P001"], ["P002", "P003"]]
    assert body["index_base"] == 0
    chords = body["chords"]
    # Adjacent chords separate no inner index.
    assert chords[0]["indices"] == [0, 1]
    assert chords[0]["cost"] == 3
    assert chords[0]["separated_interval"] == {"lo": None, "hi": None}
    assert chords[1]["indices"] == [2, 3]
    assert chords[1]["separated_interval"] == {"lo": None, "hi": None}


def test_nested_chords_intervals():
    """Chords (0,3) and (1,2): outer chord separates indices [1,2]."""
    ports = make_ports(4)
    edges = [edge("P000", "P003", 2), edge("P001", "P002", 7),
             edge("P000", "P001", 100)]
    res = client.post("/layout", json={"ports": ports, "edges": edges})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "OK"
    assert body["total_cost"] == 9
    assert body["matching"] == [["P000", "P003"], ["P001", "P002"]]
    outer, inner = body["chords"]
    assert outer["indices"] == [0, 3]
    assert outer["separated_interval"] == {"lo": 1, "hi": 2}
    assert inner["indices"] == [1, 2]
    assert inner["separated_interval"] == {"lo": None, "hi": None}


def test_tie_break_lexicographic_n4():
    """All costs zero: choose canonical smallest pair sequence."""
    ports = make_ports(4)
    res = client.post(
        "/layout", json={"ports": ports, "edges": full_edges(ports, 0)}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "OK"
    assert body["total_cost"] == 0
    # (0,1)(2,3) < (0,3)(1,2)
    assert body["matching"] == [["P000", "P001"], ["P002", "P003"]]


def test_tie_break_lexicographic_n6():
    ports = make_ports(6)
    res = client.post(
        "/layout", json={"ports": ports, "edges": full_edges(ports, 0)}
    )
    body = res.json()
    assert body["status"] == "OK"
    assert body["matching"] == [
        ["P000", "P001"], ["P002", "P003"], ["P004", "P005"]
    ]
    pairs = [tuple(ch["indices"]) for ch in body["chords"]]
    # Certificate sanity: sorted, covers all, non-crossing.
    assert pairs == sorted(pairs)
    assert sorted(v for p in pairs for v in p) == list(range(6))
    for (a, b), (c, d) in zip(pairs, pairs[1:]):
        assert not (a < c < b < d)
        assert not (c < a < d < b)


def test_ascii_port_ids_used_as_pair_identifiers():
    ports = ["A", "Z", "port-1", "x.y"]
    edges = [edge("A", "Z", 1), edge("port-1", "x.y", 1),
             edge("A", "x.y", 1)]
    res = client.post("/layout", json={"ports": ports, "edges": edges})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "OK"
    assert body["matching"] == [["A", "Z"], ["port-1", "x.y"]]
    # Pair endpoints are written in index order, not alphabetical:
    # "Z" < "port-1" alphabetically but order stays circular (A first).
    assert body["chords"][0]["pair"] == ["A", "Z"]


def test_two_ports_single_edge():
    res = client.post(
        "/layout",
        json={"ports": ["a", "b"], "edges": [edge("a", "b", 42)]},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "OK"
    assert body["total_cost"] == 42
    assert body["matching"] == [["a", "b"]]
    assert body["chords"][0]["separated_interval"] == {"lo": None, "hi": None}


def test_zero_cost_accepted():
    res = client.post(
        "/layout",
        json={"ports": ["a", "b"], "edges": [edge("a", "b", 0)]},
    )
    assert res.status_code == 200
    assert res.json()["total_cost"] == 0


def test_cost_boundaries_accepted():
    for cost in (0, 10**9):
        res = client.post(
            "/layout",
            json={"ports": ["a", "b"], "edges": [edge("a", "b", cost)]},
        )
        assert res.status_code == 200
        assert res.json()["total_cost"] == cost


# ------------------------------------------------------------ NO_LAYOUT ----

def test_no_layout_crossing_only():
    """Only mutually crossing chords available => no feasible layout."""
    ports = make_ports(4)
    edges = [edge("P000", "P002", 1), edge("P001", "P003", 1)]
    res = client.post("/layout", json={"ports": ports, "edges": edges})
    assert res.status_code == 200
    assert res.json() == {"status": "NO_LAYOUT"}


def test_no_layout_empty_edges():
    ports = make_ports(4)
    res = client.post("/layout", json={"ports": ports, "edges": []})
    assert res.status_code == 200
    assert res.json() == {"status": "NO_LAYOUT"}


def test_no_layout_missing_strand():
    """Every port has at least one allowed edge, yet no non-crossing perfect
    matching exists: on the 4-ring the only edges are the crossing pair
    (0,2)/(1,3), and (4,5) merely takes the remaining two ports."""
    ports = make_ports(6)
    edges = [
        edge("P000", "P002", 1),
        edge("P001", "P003", 1),
        edge("P004", "P005", 1),
    ]
    # Exhaustively confirm no feasible non-crossing matching.
    allowed = {(0, 2), (1, 3), (4, 5)}
    brute_feasible = any(
        all(p in allowed for p in m)
        for m in all_perfect_matchings(6)
        if is_noncrossing(m)
    )
    assert not brute_feasible
    res = client.post("/layout", json={"ports": ports, "edges": edges})
    assert res.status_code == 200
    assert res.json() == {"status": "NO_LAYOUT"}


# ----------------------------------------------------------------- 422 -----

@pytest.mark.parametrize(
    "payload",
    [
        # odd number of ports
        {"ports": make_ports(3), "edges": []},
        {"ports": make_ports(5), "edges": []},
        # fewer than 2 ports
        {"ports": ["only"], "edges": []},
        {"ports": [], "edges": []},
        # more than 120
        {"ports": make_ports(122), "edges": []},
        # duplicate port ids
        {"ports": ["a", "b", "c", "a"], "edges": []},
        # non-ascii port id
        {"ports": ["a", "端口", "c", "d"], "edges": []},
        # empty port id
        {"ports": ["a", ""], "edges": []},
    ],
)
def test_422_bad_ports(payload):
    res = client.post("/layout", json=payload)
    assert res.status_code == 422


@pytest.mark.parametrize(
    "edges",
    [
        # self connection
        [edge("a", "a", 1), edge("b", "c", 1)],
        # unknown port
        [edge("a", "zzz", 1)],
        # duplicate edge (same direction)
        [edge("a", "b", 1), edge("a", "b", 2)],
        # duplicate edge (reversed direction: undirected)
        [edge("a", "b", 1), edge("b", "a", 2)],
    ],
)
def test_422_bad_edges(edges):
    ports = ["a", "b", "c", "d"]
    res = client.post("/layout", json={"ports": ports, "edges": edges})
    assert res.status_code == 422


@pytest.mark.parametrize("bad_cost", [-1, 10**9 + 1, 1.5, -0.1])
def test_422_bad_cost(bad_cost):
    payload = {
        "ports": ["a", "b"],
        "edges": [{"u": "a", "v": "b", "cost": bad_cost}],
    }
    res = client.post("/layout", json=payload)
    assert res.status_code == 422


def test_422_cost_string_rejected():
    payload = {
        "ports": ["a", "b"],
        "edges": [{"u": "a", "v": "b", "cost": "7"}],
    }
    res = client.post("/layout", json=payload)
    assert res.status_code == 422


def test_422_missing_fields():
    assert client.post("/layout", json={}).status_code == 422
    assert client.post(
        "/layout", json={"ports": ["a", "b"]}
    ).status_code == 422
    assert client.post(
        "/layout",
        json={"ports": ["a", "b"], "edges": [{"u": "a", "cost": 1}]},
    ).status_code == 422


def test_422_extra_fields_rejected():
    payload = {
        "ports": ["a", "b"],
        "edges": [{"u": "a", "v": "b", "cost": 1, "note": "x"}],
    }
    assert client.post("/layout", json=payload).status_code == 422
    payload = {
        "ports": ["a", "b"],
        "edges": [],
        "maximize": True,
    }
    assert client.post("/layout", json=payload).status_code == 422


def test_422_wrong_types():
    assert client.post(
        "/layout", json={"ports": "ab", "edges": []}
    ).status_code == 422
    assert client.post(
        "/layout",
        json={"ports": ["a", "b"], "edges": {"u": "a", "v": "b", "cost": 1}},
    ).status_code == 422


# ------------------------------------------------------- full-ring 120 ------

def test_api_120_ports_complete_graph():
    n = 120
    ports = make_ports(n)
    edges = full_edges(ports, 7)
    res = client.post("/layout", json={"ports": ports, "edges": edges})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "OK"
    assert body["total_cost"] == 7 * (n // 2)
    assert len(body["matching"]) == n // 2
    used = sorted(v for ch in body["chords"] for v in ch["indices"])
    assert used == list(range(n))
    # Every separated interval matches [lo+1, hi-1] of its chord.
    for ch in body["chords"]:
        a, b = ch["indices"]
        iv = ch["separated_interval"]
        if b - a >= 2:
            assert iv == {"lo": a + 1, "hi": b - 1}
        else:
            assert iv == {"lo": None, "hi": None}

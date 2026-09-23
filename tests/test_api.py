"""HTTP-level tests for the /match endpoint."""

from __future__ import annotations

import itertools

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def post(payload: dict):
    return client.post("/match", json=payload)


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_healthz() -> None:
    assert client.get("/healthz").status_code == 200


def test_simple_layout_ok() -> None:
    response = post(
        {
            "ports": ["a", "b", "c", "d"],
            "edges": [
                {"from": "a", "to": "b", "cost": 1},
                {"from": "c", "to": "d", "cost": 2},
                {"from": "b", "to": "c", "cost": 3},
                {"from": "a", "to": "d", "cost": 9},
            ],
        }
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "OK"
    assert body["pairs"] == [["a", "b"], ["c", "d"]]
    assert body["total_cost"] == 3
    assert len(body["chords"]) == 2
    assert body["chords"][0] == {
        "pair": ["a", "b"],
        "left_index": 0,
        "right_index": 1,
        "inside_indices": [],
        "inside_port_ids": [],
    }
    assert body["chords"][1]["left_index"] == 2
    assert body["chords"][1]["right_index"] == 3


def test_response_is_a_valid_noncrossing_certificate() -> None:
    ports = [f"p{i}" for i in range(8)]
    edges = [
        {"from": ports[i], "to": ports[j], "cost": (i * 3 + j) % 11}
        for i, j in itertools.combinations(range(8), 2)
    ]
    body = post({"ports": ports, "edges": edges}).json()
    assert body["status"] == "OK"

    index = {pid: i for i, pid in enumerate(ports)}
    chords = sorted(
        sorted((index[a], index[b])) for a, b in body["pairs"]
    )
    # perfect matching: every port used exactly once
    assert sorted(i for pair in chords for i in pair) == list(range(8))
    # non-crossing
    for (a, b), (c, d) in itertools.combinations(chords, 2):
        assert not (a < c < b < d or c < a < d < b)
    # chord intervals are consistent with the pairs
    by_pair = {tuple(chord["pair"]): chord for chord in body["chords"]}
    for a, b in chords:
        chord = by_pair[(ports[a], ports[b])]
        assert chord["left_index"] == a
        assert chord["right_index"] == b
        assert chord["inside_indices"] == list(range(a + 1, b))
        assert chord["inside_port_ids"] == ports[a + 1 : b]
    # reported cost matches the chosen edges
    cost = {(i, j): (i * 3 + j) % 11 for i, j in itertools.combinations(range(8), 2)}
    assert body["total_cost"] == sum(cost[(a, b)] for a, b in chords)


def test_tie_break_is_deterministic_across_calls() -> None:
    payload = {
        "ports": ["p0", "p1", "p2", "p3", "p4", "p5"],
        "edges": [
            {"from": f"p{i}", "to": f"p{j}", "cost": 0}
            for i, j in itertools.combinations(range(6), 2)
        ],
    }
    first = post(payload).json()
    for _ in range(5):
        assert post(payload).json() == first
    assert first["pairs"] == [["p0", "p1"], ["p2", "p3"], ["p4", "p5"]]


def test_zero_cost_and_max_cost_bounds_accepted() -> None:
    body = post(
        {
            "ports": ["a", "b", "c", "d"],
            "edges": [
                {"from": "a", "to": "b", "cost": 0},
                {"from": "c", "to": "d", "cost": 10**9},
            ],
        }
    ).json()
    assert body["status"] == "OK"
    assert body["total_cost"] == 10**9


def test_undirected_edges_accepted_in_either_direction() -> None:
    payload = {
        "ports": ["a", "b"],
        "edges": [{"from": "b", "to": "a", "cost": 4}],
    }
    body = post(payload).json()
    assert body["status"] == "OK"
    assert body["pairs"] == [["a", "b"]]
    assert body["total_cost"] == 4


# --------------------------------------------------------------------------
# NO_LAYOUT
# --------------------------------------------------------------------------


def test_no_layout_when_edges_missing() -> None:
    body = post(
        {
            "ports": ["a", "b", "c", "d"],
            "edges": [{"from": "a", "to": "c", "cost": 1}],
        }
    ).json()
    assert body["status"] == "NO_LAYOUT"
    assert "pairs" not in body  # never a partial pairing


def test_no_layout_with_empty_edge_list() -> None:
    body = post({"ports": ["a", "b"], "edges": []}).json()
    assert body["status"] == "NO_LAYOUT"


# --------------------------------------------------------------------------
# 422 contract violations
# --------------------------------------------------------------------------


def test_odd_port_count_is_422() -> None:
    assert post({"ports": ["a", "b", "c"], "edges": []}).status_code == 422


def test_duplicate_port_is_422() -> None:
    response = post({"ports": ["a", "a"], "edges": []})
    assert response.status_code == 422


def test_too_few_and_too_many_ports_are_422() -> None:
    assert post({"ports": ["a"], "edges": []}).status_code == 422
    assert post({"ports": [f"p{i}" for i in range(122)], "edges": []}).status_code == 422


def test_self_loop_is_422() -> None:
    response = post(
        {"ports": ["a", "b"], "edges": [{"from": "a", "to": "a", "cost": 1}]}
    )
    assert response.status_code == 422


def test_unknown_port_is_422() -> None:
    response = post(
        {"ports": ["a", "b"], "edges": [{"from": "a", "to": "z", "cost": 1}]}
    )
    assert response.status_code == 422


def test_duplicate_edge_is_422_including_reversed() -> None:
    edges = [
        {"from": "a", "to": "b", "cost": 1},
        {"from": "b", "to": "a", "cost": 2},
    ]
    assert post({"ports": ["a", "b"], "edges": edges}).status_code == 422


def test_out_of_range_cost_is_422() -> None:
    edge = {"from": "a", "to": "b", "cost": 10**9 + 1}
    assert post({"ports": ["a", "b"], "edges": [edge]}).status_code == 422
    edge = {"from": "a", "to": "b", "cost": -1}
    assert post({"ports": ["a", "b"], "edges": [edge]}).status_code == 422


def test_non_integer_cost_is_422() -> None:
    edge = {"from": "a", "to": "b", "cost": 1.5}
    assert post({"ports": ["a", "b"], "edges": [edge]}).status_code == 422
    edge = {"from": "a", "to": "b", "cost": True}
    assert post({"ports": ["a", "b"], "edges": [edge]}).status_code == 422


def test_non_ascii_port_is_422() -> None:
    assert post({"ports": ["a", "端口"], "edges": []}).status_code == 422


def test_malformed_body_is_422() -> None:
    assert client.post("/match", json={"ports": "ab"}).status_code == 422
    assert client.post("/match", json={}).status_code == 422

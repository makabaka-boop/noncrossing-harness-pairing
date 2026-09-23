"""FastAPI application: deterministic non-crossing wiring certificates."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .matching import best_noncrossing_matching
from .schemas import (
    Chord,
    IndexInterval,
    LayoutNoLayout,
    LayoutOk,
    MatchRequest,
)

app = FastAPI(
    title="Circular Fixture Non-Crossing Matching API",
    version="1.0.0",
    description=(
        "Computes the minimum-cost non-crossing perfect matching of ports "
        "on a circular test fixture using interval dynamic programming."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/layout",
    response_model=LayoutOk | LayoutNoLayout,
    summary="Compute a minimum-cost non-crossing perfect matching",
)
def layout(payload: MatchRequest) -> LayoutOk | LayoutNoLayout:
    ports = payload.ports
    n = len(ports)
    index = {pid: i for i, pid in enumerate(ports)}

    # Normalize each allowed undirected edge to (low_index, high_index).
    cost: dict[tuple[int, int], int] = {}
    for edge in payload.edges:
        a, b = index[edge.u], index[edge.v]
        key = (a, b) if a < b else (b, a)
        cost[key] = edge.cost

    result = best_noncrossing_matching(n, cost)
    if not result.feasible:
        # Never emit a partial pairing.
        return LayoutNoLayout()

    chords: list[Chord] = []
    for a, b in result.pairs:  # result.pairs is already lexicographically sorted
        chord_cost = cost[(a, b)]
        # Indices strictly separated inside the chord: [a+1, b-1].
        if b - a >= 2:
            separated = IndexInterval(lo=a + 1, hi=b - 1)
        else:
            separated = IndexInterval(lo=None, hi=None)
        chords.append(
            Chord(
                pair=(ports[a], ports[b]),
                indices=(a, b),
                cost=chord_cost,
                separated_interval=separated,
            )
        )

    return LayoutOk(
        total_cost=result.total_cost,
        matching=[(ports[a], ports[b]) for a, b in result.pairs],
        chords=chords,
    )

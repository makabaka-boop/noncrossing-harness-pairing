"""Pydantic request/response models.

Everything that makes a request structurally invalid is rejected here, so
FastAPI emits HTTP 422 uniformly: wrong port count, duplicate port ids,
non-ASCII ids, self-loops, unknown endpoints, duplicate edges, odd number
of ports, and costs outside [0, 10**9].
"""

from __future__ import annotations

from typing_extensions import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    model_validator,
)

MIN_PORTS = 2
MAX_PORTS = 120
MAX_COST = 10**9

PortId = Annotated[str, Field(strict=True, min_length=1)]
Cost = Annotated[int, Field(strict=True, ge=0, le=MAX_COST)]


class EdgeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    u: PortId
    v: PortId
    cost: Cost


class MatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ports: list[PortId] = Field(min_length=MIN_PORTS, max_length=MAX_PORTS)
    edges: list[EdgeIn]

    @model_validator(mode="after")
    def _validate(self) -> "MatchRequest":
        ports = self.ports

        # Every port id must be non-empty ASCII.
        for pid in ports:
            if not pid.isascii():
                raise ValueError("port ids must be ASCII strings")

        # Port ids must be unique (their circular position matters).
        if len(set(ports)) != len(ports):
            raise ValueError("port ids must be unique")

        # A perfect matching needs an even number of ports.
        if len(ports) % 2 == 1:
            raise ValueError("number of ports must be even for a perfect matching")

        index = {pid: i for i, pid in enumerate(ports)}
        seen: set[tuple[int, int]] = set()

        for edge in self.edges:
            if edge.u == edge.v:
                raise ValueError(f"self-connection is not allowed: {edge.u!r}")
            if edge.u not in index or edge.v not in index:
                raise ValueError(
                    f"edge references unknown port: {edge.u!r} -- {edge.v!r}"
                )
            a, b = index[edge.u], index[edge.v]
            key = (a, b) if a < b else (b, a)
            if key in seen:
                raise ValueError(
                    "duplicate edge is not allowed: "
                    f"{min(edge.u, edge.v)!r} -- {max(edge.u, edge.v)!r}"
                )
            seen.add(key)

        return self


class IndexInterval(BaseModel):
    """Inclusive index range of ports strictly inside the chord.

    Both bounds are null for an adjacent pair (no port lies between them
    along the circle in the chosen indexing direction).
    """

    lo: int | None = Field(
        description="smaller inner index; null when nothing lies inside"
    )
    hi: int | None = Field(
        description="larger inner index; null when nothing lies inside"
    )


class Chord(BaseModel):
    pair: tuple[StrictStr, StrictStr]
    indices: tuple[int, int]
    cost: Cost
    separated_interval: IndexInterval


class LayoutOk(BaseModel):
    status: Literal["OK"] = "OK"
    index_base: Literal[0] = 0
    total_cost: int
    matching: list[tuple[StrictStr, StrictStr]]
    chords: list[Chord]


class LayoutNoLayout(BaseModel):
    status: Literal["NO_LAYOUT"] = "NO_LAYOUT"

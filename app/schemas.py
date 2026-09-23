"""Request/response schemas for the wiring API."""

from __future__ import annotations

from typing import List, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .matching import MAX_COST, MAX_PORTS, MIN_PORTS


class EdgeIn(BaseModel):
    """One allowed undirected connection between two ports."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: str = Field(min_length=1, alias="from")
    to: str = Field(min_length=1)
    cost: int = Field(ge=0, le=MAX_COST)

    @field_validator("cost", mode="before")
    @classmethod
    def _cost_must_be_a_plain_integer(cls, value: object) -> object:
        # bool is an int subclass; floats/strings must not be silently coerced.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("cost must be an integer")
        return value


class LayoutRequest(BaseModel):
    """Ports in circular order plus the allowed (undirected) chords."""

    model_config = ConfigDict(extra="forbid")

    ports: List[str] = Field(min_length=MIN_PORTS, max_length=MAX_PORTS)
    edges: List[EdgeIn] = Field(default_factory=list)


class ChordOut(BaseModel):
    """One chosen chord and the index interval it fences off."""

    pair: List[str]
    left_index: int
    right_index: int
    inside_indices: List[int]
    inside_port_ids: List[str]


class LayoutOk(BaseModel):
    status: str = "OK"
    pairs: List[List[str]]
    total_cost: int
    chords: List[ChordOut]


class LayoutNone(BaseModel):
    status: str = "NO_LAYOUT"
    detail: str = "no non-crossing perfect matching exists for the given ports and edges"


LayoutResponse = Union[LayoutOk, LayoutNone]

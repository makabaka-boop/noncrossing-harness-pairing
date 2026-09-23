"""FastAPI entrypoint for the circular wiring service."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .matching import LayoutValidationError, solve_layout
from .schemas import ChordOut, LayoutNone, LayoutOk, LayoutRequest, LayoutResponse

app = FastAPI(
    title="Circular Wiring API",
    version="1.0.0",
    description=(
        "Picks a minimum-cost, non-crossing perfect matching for ports placed "
        "around a circular test fixture, and certifies it with the index "
        "interval each chord separates."
    ),
)


@app.exception_handler(LayoutValidationError)
async def layout_validation_handler(
    _request: Request, exc: LayoutValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": [{"code": exc.code, "msg": exc.message}]},
    )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/match", response_model=LayoutResponse)
async def match_layout(request: LayoutRequest) -> LayoutResponse:
    edges = [(edge.from_, edge.to, edge.cost) for edge in request.edges]
    result = solve_layout(request.ports, edges)
    if result is None:
        return LayoutNone()
    return LayoutOk(
        pairs=[[a, b] for a, b in result.pairs],
        total_cost=result.total_cost,
        chords=[ChordOut(**chord) for chord in result.chords],
    )

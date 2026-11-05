"""Shared schemas: the intermediate representation every parser converges on,
plus the query vocabulary shared by the API and voice layers.

StructuredQuery lives here (not in app/api/query.py) so that app.voice can
translate text without importing the HTTP layer — that circular import was
flagged in WORKPLAN.md Phase 0.
"""
from typing import Literal

from pydantic import BaseModel


class Point(BaseModel):
    x: float
    y: float


class Wall(BaseModel):
    start: Point
    end: Point


class Room(BaseModel):
    id: str
    label: str | None = None
    polygon: list[Point]
    area_m2: float
    inferred: bool = False  # True if derived from walls, not read from a rooms layer


class NormalizedFloorplan(BaseModel):
    source_format: str
    units: str = "m"  # all coordinates normalised to metres
    bounds: tuple[float, float, float, float]  # minx, miny, maxx, maxy
    walls: list[Wall]
    rooms: list[Room]
    warnings: list[str] = []


class StructuredQuery(BaseModel):
    """The fixed vocabulary of graph operations the translator may emit.

    The translator (rules or LLM) only picks an intent and fills slots —
    it never computes areas, distances, or paths.
    """

    intent: Literal["find_path", "compute_area", "check_egress", "unknown"]
    room_from: str | None = None
    room_to: str | None = None
    room: str | None = None

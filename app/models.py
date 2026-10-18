"""Shared intermediate representation. All input formats converge here."""
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

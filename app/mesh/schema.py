"""
Shared intermediate geometry schema.

Every input format (DXF/CAD, vector PDF, raster image via CV) converges to
this representation before 3D extrusion. This is the contract that keeps
the mesh-building code identical regardless of source format — only the
parser that produces a NormalizedFloorplan differs per input type.
"""

from dataclasses import dataclass, field


@dataclass
class WallSegment:
    """A single wall as a 2D line, in real-world meters (already scale-corrected)."""
    start: tuple[float, float]
    end: tuple[float, float]
    thickness_m: float = 0.15
    height_m: float = 2.7


@dataclass
class RoomPolygon:
    """A room boundary as an ordered list of (x, y) points in meters."""
    room_id: str
    points: list[tuple[float, float]]
    label: str | None = None


@dataclass
class NormalizedFloorplan:
    """The common output of every parser (dxf_parser, pdf_vector_parser, cv_pipeline)."""
    walls: list[WallSegment] = field(default_factory=list)
    rooms: list[RoomPolygon] = field(default_factory=list)
    source_format: str = "unknown"  # "dxf" | "pdf_vector" | "raster_cv"

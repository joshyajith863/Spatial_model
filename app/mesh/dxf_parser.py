"""
DXF/CAD parser.

Easiest input path: DXF files already contain exact vector coordinates on
named layers (e.g. "WALLS", "A-WALL"), so no CV/scale-calibration is needed.
This is why DXF is the recommended first input format to get a working
demo fastest.

Requires: pip install ezdxf
"""

import ezdxf

from app.mesh.schema import NormalizedFloorplan, WallSegment, RoomPolygon


def parse_dxf(file_path: str, wall_layers: list[str] | None = None) -> NormalizedFloorplan:
    """
    Reads LINE/LWPOLYLINE entities from wall-related layers and converts them
    into WallSegments. Layer names vary a lot between CAD tools/firms, so
    wall_layers lets you pass the actual layer name(s) used in your file —
    inspect the DXF first (see list_layers below) if you're not sure.
    """
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    wall_layers = wall_layers or ["WALLS", "A-WALL", "0"]
    walls: list[WallSegment] = []

    for entity in msp.query("LINE"):
        if entity.dxf.layer in wall_layers:
            start = (entity.dxf.start.x, entity.dxf.start.y)
            end = (entity.dxf.end.x, entity.dxf.end.y)
            walls.append(WallSegment(start=start, end=end))

    for entity in msp.query("LWPOLYLINE"):
        if entity.dxf.layer in wall_layers:
            points = [(p[0], p[1]) for p in entity.get_points()]
            # A polyline of N points becomes N-1 (or N, if closed) wall segments.
            closed = entity.closed
            segment_count = len(points) if closed else len(points) - 1
            for i in range(segment_count):
                start = points[i]
                end = points[(i + 1) % len(points)]
                walls.append(WallSegment(start=start, end=end))

    # TODO: room polygons — either read from a dedicated "ROOMS"/"AREA" layer
    # if the CAD file has one, or derive room boundaries from closed wall
    # loops (polygon union/face detection — can reuse Shapely for this,
    # same as the CV-pipeline path does for raster input).
    rooms: list[RoomPolygon] = []

    return NormalizedFloorplan(walls=walls, rooms=rooms, source_format="dxf")


def list_layers(file_path: str) -> list[str]:
    """Utility: inspect a DXF file's layer names before parsing, since wall
    layer naming conventions vary between CAD tools and firms."""
    doc = ezdxf.readfile(file_path)
    return [layer.dxf.name for layer in doc.layers]

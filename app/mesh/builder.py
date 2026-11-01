"""
3D mesh construction: NormalizedFloorplan -> extruded mesh -> GLB bytes.

Coordinate handling (the one place the y-flip lives):
    Floorplan is 2D in metres, plan-Y pointing "up the page".
    We map plan (x, y) -> world (x, -y) on the ground plane, and extrude
    height along world +Y. A -90 deg rotation about X does this cleanly
    (proper rotation, so normals/winding stay correct for GLB).

This module is pure geometry — no HTTP, no LLM. It only reads a
NormalizedFloorplan and emits a trimesh.Scene / GLB bytes plus metadata.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import trimesh
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union

from app.models import NormalizedFloorplan

# --- Sensible defaults (metres). Exposed so the API can override per request. ---
DEFAULT_WALL_HEIGHT_M = 2.7
DEFAULT_WALL_THICKNESS_M = 0.15
DEFAULT_SLAB_M = 0.06  # floor slab thickness

# -90 deg about X: (x, y, z) -> (x, z, -y). Height (extrude axis) -> world +Y.
_TO_WORLD = trimesh.transformations.rotation_matrix(-np.pi / 2.0, [1, 0, 0])


@dataclass
class MeshOptions:
    wall_height_m: float = DEFAULT_WALL_HEIGHT_M
    wall_thickness_m: float = DEFAULT_WALL_THICKNESS_M
    slab_m: float = DEFAULT_SLAB_M
    include_floors: bool = True
    include_ceiling: bool = False  # optional; ceilings make interiors feel closed-in


@dataclass
class MeshResult:
    glb: bytes
    triangle_count: int
    bounds: tuple  # (minx, miny, minz, maxx, maxy, maxz) in world metres
    wall_area_m2: float
    floor_area_m2: float
    warnings: list[str] = field(default_factory=list)


def _wall_polygon(wall, thickness: float) -> Polygon:
    """Buffer a wall centreline into a rectangular slab (flat caps, mitre joins)."""
    line = LineString([(wall.start.x, wall.start.y), (wall.end.x, wall.end.y)])
    if line.length == 0:
        # Degenerate zero-length wall; buffer of a point still yields a square.
        return line.buffer(thickness / 2.0)
    return line.buffer(thickness / 2.0, cap_style=2, join_style=2)


def _plan_geometry(floorplan: NormalizedFloorplan, opts: MeshOptions):
    """Return (wall_polygon, floor_polygon) as shapely geoms in plan coords."""
    wall_parts = [_wall_polygon(w, opts.wall_thickness_m) for w in floorplan.walls]
    wall_geom = unary_union(wall_parts) if wall_parts else None

    floor_geom = None
    if opts.include_floors and floorplan.rooms:
        floor_parts = []
        for room in floorplan.rooms:
            pts = [(p.x, p.y) for p in room.polygon]
            if len(pts) >= 3:
                poly = Polygon(pts)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty:
                    floor_parts.append(poly)
        floor_geom = unary_union(floor_parts) if floor_parts else None

    return wall_geom, floor_geom


def _extrude_to_world(geom, height: float) -> list[trimesh.Trimesh]:
    """Extrude a (Multi)Polygon to `height` and rotate into world orientation."""
    if geom is None or geom.is_empty:
        return []
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    out = []
    for poly in polys:
        if not isinstance(poly, Polygon) or poly.is_empty or poly.area <= 0:
            continue
        m = trimesh.creation.extrude_polygon(poly, height)
        m.apply_transform(_TO_WORLD)
        out.append(m)
    return out


def _paint(mesh: trimesh.Trimesh, rgba) -> trimesh.Trimesh:
    r, g, b, a = rgba
    mesh.visual.material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[r, g, b, a], metallicFactor=0.0, roughnessFactor=0.9
    )
    return mesh


def _combine(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh | None:
    meshes = [m for m in meshes if m is not None and len(m.faces) > 0]
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def build_scene(floorplan: NormalizedFloorplan, opts: MeshOptions | None = None):
    """Build a named trimesh.Scene ('walls', 'floors', optional 'ceiling')."""
    opts = opts or MeshOptions()
    warnings = list(floorplan.warnings)

    wall_geom, floor_geom = _plan_geometry(floorplan, opts)

    walls = _combine(_extrude_to_world(wall_geom, opts.wall_height_m))
    floors = None
    if opts.include_floors:
        floors = _combine(_extrude_to_world(floor_geom, opts.slab_m))

    if walls is None and floors is None:
        raise ValueError("Nothing to extrude: floorplan has no walls and no rooms.")

    if walls is None:
        warnings.append("No wall geometry; extruded rooms only.")
    if floor_geom is None and opts.include_floors:
        warnings.append("No room polygons; no floor slab generated.")

    # Centre the model on the origin and drop the floor to y=0 so the VR
    # camera/teleport plane lines up with the building floor.
    scene = trimesh.Scene()
    shift = _centre_shift([m for m in (walls, floors) if m is not None])
    if walls is not None:
        walls.apply_translation(shift)
        _paint(walls, [0.82, 0.82, 0.85, 1.0])
    if floors is not None:
        floors.apply_translation(shift)
        _paint(floors, [0.62, 0.52, 0.42, 1.0])

    if walls is not None:
        scene.add_geometry(walls, node_name="walls", geom_name="walls")
    if floors is not None:
        scene.add_geometry(floors, node_name="floors", geom_name="floors")

    if opts.include_ceiling:
        ceiling = _combine(_extrude_to_world(floor_geom, opts.slab_m))
        if ceiling is not None:
            # Lift a copy of the floor slab up to wall height as a ceiling plane.
            ceiling.apply_translation([0, opts.wall_height_m, 0])
            ceiling.apply_translation(shift)
            _paint(ceiling, [0.9, 0.9, 0.9, 1.0])
            scene.add_geometry(ceiling, node_name="ceiling", geom_name="ceiling")

    return scene, warnings


def _centre_shift(meshes: list[trimesh.Trimesh]) -> list[float]:
    """Translation that centres the footprint at x=z=0 and puts the floor at y=0."""
    if not meshes:
        return [0.0, 0.0, 0.0]
    allb = np.array([m.bounds for m in meshes])  # (n, 2, 3)
    minv = allb[:, 0, :].min(axis=0)
    maxv = allb[:, 1, :].max(axis=0)
    centre_x = (minv[0] + maxv[0]) / 2.0
    centre_z = (minv[2] + maxv[2]) / 2.0
    return [-centre_x, -minv[1], -centre_z]


def build_glb(floorplan: NormalizedFloorplan, opts: MeshOptions | None = None) -> MeshResult:
    """Build the scene and serialise to GLB bytes, returning geometry metadata."""
    scene, warnings = build_scene(floorplan, opts)

    meshes = [g for g in scene.geometry.values()]
    tri_count = int(sum(len(m.faces) for m in meshes))
    wall_area = float(sum(m.area for name, m in scene.geometry.items() if name == "walls"))
    floor_area = float(sum(m.area for name, m in scene.geometry.items() if name == "floors"))

    # World-space bounds of the whole scene.
    try:
        bounds = scene.bounds  # (min, max)
        bounds_t = (float(bounds[0][0]), float(bounds[0][1]), float(bounds[0][2]),
                    float(bounds[1][0]), float(bounds[1][1]), float(bounds[1][2]))
    except Exception:
        bounds_t = (0, 0, 0, 0, 0, 0)

    glb = scene.export(file_type="glb")
    return MeshResult(
        glb=bytes(glb), triangle_count=tri_count, bounds=bounds_t,
        wall_area_m2=round(wall_area, 3), floor_area_m2=round(floor_area, 3),
        warnings=warnings,
    )


# --- A tiny hand-authored sample so the VR viewer is demoable with no upload. ---
def sample_floorplan() -> NormalizedFloorplan:
    """A 3-room rectangle: two interior walls splitting a 8x6 m footprint."""
    from app.models import Point, Room, Wall

    def rect(x0, y0, x1, y1):
        return [Point(x=x0, y=y0), Point(x=x1, y=y0), Point(x=x1, y=y1), Point(x=x0, y=y1)]

    walls = [
        # outer shell
        Wall(start=Point(x=0, y=0), end=Point(x=8, y=0)),
        Wall(start=Point(x=8, y=0), end=Point(x=8, y=6)),
        Wall(start=Point(x=8, y=6), end=Point(x=0, y=6)),
        Wall(start=Point(x=0, y=6), end=Point(x=0, y=0)),
        # interior partitions
        Wall(start=Point(x=3, y=0), end=Point(x=3, y=4)),
        Wall(start=Point(x=3, y=4), end=Point(x=8, y=4)),
        Wall(start=Point(x=5, y=4), end=Point(x=5, y=6)),
    ]
    rooms = [
        Room(id="room_0", label="Living", area_m2=3 * 6, polygon=rect(0, 0, 3, 6)),
        Room(id="room_1", label="Kitchen", area_m2=5 * 4, polygon=rect(3, 0, 8, 4)),
        Room(id="room_2", label="Bedroom", area_m2=4.0, polygon=rect(3, 4, 5, 6)),
        Room(id="room_3", label="Bath", area_m2=3 * 2, polygon=rect(5, 4, 8, 6)),
    ]
    return NormalizedFloorplan(
        source_format="sample", units="m", bounds=(0, 0, 8, 6),
        walls=walls, rooms=rooms, warnings=["Sample floorplan (synthetic)."],
    )

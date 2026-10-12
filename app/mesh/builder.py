"""
3D mesh builder — the shared stage every input format (DXF, vector PDF,
raster CV) feeds into. Takes a NormalizedFloorplan and extrudes it into a
watertight 3D scene, exported as GLB for browser viewing (model-viewer /
Three.js / WebXR).

Requires: pip install trimesh manifold3d
"""

import trimesh
import numpy as np

from app.mesh.schema import NormalizedFloorplan, WallSegment


def _wall_to_box(wall: WallSegment) -> trimesh.Trimesh:
    """Turn a wall centerline into a 3D box: thickness x length x height."""
    start = np.array(wall.start)
    end = np.array(wall.end)
    direction = end - start
    length = np.linalg.norm(direction)

    if length < 1e-6:
        raise ValueError(f"Degenerate wall segment: {wall}")

    box = trimesh.creation.box(extents=[length, wall.thickness_m, wall.height_m])

    # box is created centered at origin, lying along X — rotate to match
    # the wall's actual direction, then translate to its midpoint.
    angle = np.arctan2(direction[1], direction[0])
    rotation = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
    box.apply_transform(rotation)

    midpoint = (start + end) / 2
    box.apply_translation([midpoint[0], midpoint[1], wall.height_m / 2])

    return box


def _room_floor(room_points: list[tuple[float, float]], thickness_m: float = 0.1) -> trimesh.Trimesh:
    """Extrude a room's 2D polygon downward slightly to form a floor slab."""
    polygon_2d = trimesh.path.polygons.Polygon(room_points)
    mesh = trimesh.creation.extrude_polygon(polygon_2d, height=thickness_m)
    mesh.apply_translation([0, 0, -thickness_m])
    return mesh


def build_mesh(floorplan: NormalizedFloorplan) -> trimesh.Scene:
    """
    Combine all wall boxes (+ room floors, if present) into a single scene.
    Each wall/floor is kept as a separate mesh in the scene rather than
    boolean-unioned, which is faster and good enough for viewing — true
    watertight union (via manifold3d) is only worth the extra cost if you
    need the mesh for physical simulation, not just display.
    """
    if not floorplan.walls:
        raise ValueError("No walls to build — check your parser output.")

    scene = trimesh.Scene()

    for i, wall in enumerate(floorplan.walls):
        try:
            box = _wall_to_box(wall)
            scene.add_geometry(box, node_name=f"wall_{i}")
        except ValueError:
            continue  # skip degenerate/zero-length segments rather than failing the whole build

    for room in floorplan.rooms:
        if len(room.points) >= 3:
            floor = _room_floor(room.points)
            scene.add_geometry(floor, node_name=f"floor_{room.room_id}")

    return scene


def export_glb(scene: trimesh.Scene, output_path: str) -> str:
    """Export the scene as a .glb file ready to serve from app/static/ and
    view via <model-viewer> or Three.js in the browser."""
    scene.export(output_path)
    return output_path


def floorplan_to_glb(floorplan: NormalizedFloorplan, output_path: str) -> str:
    """Convenience: full pipeline from NormalizedFloorplan to a saved .glb file."""
    scene = build_mesh(floorplan)
    return export_glb(scene, output_path)

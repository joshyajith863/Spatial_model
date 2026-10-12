"""DXF -> NormalizedFloorplan (vector input only)."""
import ezdxf
from shapely.geometry import LineString, Point as SPoint, Polygon
from shapely.ops import polygonize, unary_union

from app.models import NormalizedFloorplan, Point, Room, Wall

# ezdxf $INSUNITS code -> metres per unit
UNIT_TO_M = {0: None, 1: 0.0254, 2: 0.3048, 4: 0.001, 5: 0.01, 6: 1.0}
WALL_KEYS = ("wall",)
ROOM_KEYS = ("room", "space", "area")


def _is(layer: str, keys) -> bool:
    return any(k in layer.lower() for k in keys)


def list_layers(path: str) -> list[str]:
    return [l.dxf.name for l in ezdxf.readfile(path).layers]


def parse_dxf(path: str, wall_layers=None, room_layers=None) -> NormalizedFloorplan:
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    warnings: list[str] = []

    scale = UNIT_TO_M.get(doc.header.get("$INSUNITS", 0))
    if scale is None:
        scale = 0.001
        warnings.append("DXF has no unit info; assumed millimetres. Verify scale.")

    def is_wall(l): return l in wall_layers if wall_layers else _is(l, WALL_KEYS)
    def is_room(l): return l in room_layers if room_layers else _is(l, ROOM_KEYS)

    segs: list[tuple] = []
    room_polys: list[Polygon] = []
    texts: list[tuple[str, float, float]] = []

    for e in msp:
        layer, t = e.dxf.layer, e.dxftype()
        if t == "LINE" and is_wall(layer):
            a, b = e.dxf.start, e.dxf.end
            segs.append(((a.x * scale, a.y * scale), (b.x * scale, b.y * scale)))
        elif t in ("LWPOLYLINE", "POLYLINE"):
            pts = ([(p[0], p[1]) for p in e.get_points()] if t == "LWPOLYLINE"
                   else [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices])
            pts = [(x * scale, y * scale) for x, y in pts]
            if len(pts) < 2:
                continue
            closed = e.closed
            if is_room(layer) and closed and len(pts) >= 3:
                room_polys.append(Polygon(pts))
            elif is_wall(layer):
                ring = pts + ([pts[0]] if closed else [])
                segs += list(zip(ring[:-1], ring[1:]))
        elif t in ("TEXT", "MTEXT"):
            txt = e.dxf.text if t == "TEXT" else e.plain_text()
            ins = e.dxf.insert
            if txt.strip():
                texts.append((txt.strip(), ins.x * scale, ins.y * scale))

    if not segs and not room_polys:
        raise ValueError(f"No walls or rooms found. Layers in file: {list_layers(path)}")

    inferred = False
    if not room_polys and segs:
        lines = unary_union([LineString(s) for s in segs])
        room_polys = [p for p in polygonize(lines) if p.area > 1.0]
        inferred = True
        warnings.append("No rooms layer; rooms inferred from wall loops (may be inaccurate).")

    rooms = []
    for i, poly in enumerate(room_polys):
        label = next((t for t, x, y in texts if poly.contains(SPoint(x, y))), None)
        rooms.append(Room(
            id=f"room_{i}", label=label, area_m2=round(poly.area, 3), inferred=inferred,
            polygon=[Point(x=x, y=y) for x, y in poly.exterior.coords[:-1]],
        ))

    xs = [p[0] for s in segs for p in s] + [x for r in rooms for pt in r.polygon for x in (pt.x,)]
    ys = [p[1] for s in segs for p in s] + [y for r in rooms for pt in r.polygon for y in (pt.y,)]
    return NormalizedFloorplan(
        source_format="dxf", bounds=(min(xs), min(ys), max(xs), max(ys)),
        walls=[Wall(start=Point(x=a[0], y=a[1]), end=Point(x=b[0], y=b[1])) for a, b in segs],
        rooms=rooms, warnings=warnings,
    )

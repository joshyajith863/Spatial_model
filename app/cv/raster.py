"""
Raster blueprint ingestion: PNG/JPG/PDF -> NormalizedFloorplan (WORKPLAN Phase 1).

Pure OpenCV/Shapely geometry, no ML. Pipeline:

    load_raster()       image (or PDF page @ ~200 DPI) -> RGB array
    preprocess()        gray -> OTSU ink mask (+ dark-border crop, opt-in deskew)
    isolate_walls()     distance transform -> split thick (walls) from thin
                        (furniture, dimension lines, text, hatching)
    mask_to_segments()  morphological skeleton -> polyline walk -> centrelines
    extract_rooms()     enclosed background regions (white not touching the border)
    parse_raster()      assembles the above into a NormalizedFloorplan

Scale
-----
Raster plans carry no units. `pixels_per_meter` may be supplied by the caller
(Phase 2 adds two-point calibration in the viewer); otherwise it is estimated
from the median wall thickness assuming a typical wall of `assumed_wall_m`
metres, and the estimate is flagged in `floorplan.warnings`.

Coordinates
----------
Images are y-down; plan space is y-up in metres (same convention as the DXF
parser). Every exported coordinate is flipped: y_plan = (height - y_px) / ppm.
To draw floorplan geometry back onto the source image (verification overlay),
invert that mapping.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from app.models import NormalizedFloorplan, Point, Room, Wall

# Canonical working resolution: thresholds (wall thickness, speckle size) are
# expressed in pixels, so images are normalised to this long-side length first.
CANONICAL_LONG_SIDE = 2000
MIN_SHORT_SIDE = 800          # upscale tiny renders so thresholds stay comparable
DEFAULT_ASSUMED_WALL_M = 0.20  # median interior/exterior wall thickness assumption
SKEW_LIMIT_DEG = 15.0          # never "correct" more than this; likely not skew

ImageMeta = tuple[int, int, float]  # (orig_h, orig_w, canonical_factor)


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------
def load_raster(path: str | Path, dpi: int = 200) -> np.ndarray:
    """Read a raster blueprint (or the first page of a PDF) as an RGB array."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path, dpi=dpi)

    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    rgb = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if rgb is None:
        # Fallback for formats OpenCV's decoder doesn't cover (e.g. some webp).
        from PIL import Image

        with Image.open(path) as im:
            rgb = np.array(im.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def _load_pdf(path: Path, dpi: int = 200) -> np.ndarray:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PDF support needs PyMuPDF: pip install pymupdf"
        ) from exc

    with fitz.open(path) as doc:
        if doc.page_count == 0:
            raise ValueError("PDF has no pages")
        page = doc.load_page(0)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return np.ascontiguousarray(img)


# ---------------------------------------------------------------------------
# 2. Preprocess
# ---------------------------------------------------------------------------
def preprocess(rgb: np.ndarray, deskew: bool = False) -> tuple[np.ndarray, ImageMeta]:
    """RGB -> binary ink mask (ink=255) at canonical scale.

    Returns (ink, (orig_h, orig_w, factor)) where canonical = orig * factor.

    `deskew` is opt-in: evaluation on CubiCasa5K showed automatic rotation
    *lowered* wall IoU (0.467 -> 0.381) because ground-truth annotations follow
    the drawing's own orientation, and Hough skew estimates get fooled by
    diagonal furniture/text. It also keeps pixels aligned with the source image,
    which the verification overlay needs.
    """
    orig_h, orig_w = rgb.shape[:2]

    long_side = max(orig_h, orig_w)
    if long_side > CANONICAL_LONG_SIDE:
        factor = CANONICAL_LONG_SIDE / long_side
    elif long_side < MIN_SHORT_SIDE:
        factor = MIN_SHORT_SIDE / long_side
    else:
        factor = 1.0

    if factor != 1.0:
        rgb = cv2.resize(
            rgb,
            (max(1, round(orig_w * factor)), max(1, round(orig_h * factor))),
            interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Global OTSU keeps thin strokes intact (adaptive threshold erodes them,
    # which cost ~0.14 IoU in evaluation) — but only when the split looks sane.
    _, ink = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ratio = float((ink > 0).mean())
    if not (0.02 <= ratio <= 0.30):
        # Uneven scan lighting fooled OTSU; fall back to adaptive thresholding.
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        block = max(15, min(gray.shape) // 60)
        block = block + 1 if block % 2 == 0 else block
        ink = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 10
        )

    ink = _drop_speckle(ink)
    ink = _crop_dark_border(ink)
    if deskew:
        angle = estimate_skew(ink)
        if abs(angle) > 0.2:
            ink = _rotate(ink, -angle)
    return ink, (orig_h, orig_w, factor)


def _crop_dark_border(ink: np.ndarray) -> np.ndarray:
    """Images photographed/scanned against a dark background (or with a black
    frame) binarize to "frame = ink", which reads as one giant wall. When the
    border ring is mostly ink, keep only the largest white region (the paper)."""
    h, w = ink.shape
    r = max(2, int(0.02 * min(h, w)))
    ring = np.concatenate([
        ink[:r].ravel(), ink[-r:].ravel(), ink[:, :r].ravel(), ink[:, -r:].ravel()
    ])
    if float(ring.mean()) / 255.0 < 0.5:
        return ink

    background = (ink == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=8)
    if n <= 1:
        return ink
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = int(stats[biggest, cv2.CC_STAT_LEFT])
    y = int(stats[biggest, cv2.CC_STAT_TOP])
    bw = int(stats[biggest, cv2.CC_STAT_WIDTH])
    bh = int(stats[biggest, cv2.CC_STAT_HEIGHT])

    out = ink.copy()
    out[:y] = 0
    out[y + bh:] = 0
    out[:, :x] = 0
    out[:, x + bw:] = 0
    return out


def _drop_speckle(ink: np.ndarray, min_area: int | None = None) -> np.ndarray:
    """Delete tiny components (scan noise, dust) that aren't strokes."""
    h, w = ink.shape
    if min_area is None:
        min_area = max(4, int(0.00001 * h * w))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def estimate_skew(ink: np.ndarray, max_lines: int = 400) -> float:
    """Median stroke angle (degrees). Plans are near-axis-aligned, so a small
    consistent offset is skew; large values are ignored as non-structural lines."""
    small = ink
    h, w = ink.shape
    if max(h, w) > 900:
        f = 900.0 / max(h, w)
        small = cv2.resize(ink, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)

    lines = cv2.HoughLinesP(small, 1, np.pi / 360, threshold=60,
                            minLineLength=max(30, max(small.shape) // 20), maxLineGap=8)
    if lines is None:
        return 0.0

    lines = np.asarray(lines).reshape(-1, 4)
    angles = []
    for x1, y1, x2, y2 in lines[:max_lines]:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angle = (angle + 90.0) % 90.0 - 45.0   # fold to [-45, 45)
        if abs(angle) <= 30.0:
            angles.append(angle)
    if len(angles) < 8:
        return 0.0

    skew = float(np.median(angles))
    if abs(skew) > SKEW_LIMIT_DEG:
        return 0.0
    return skew


def _rotate(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = mask.shape
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


# ---------------------------------------------------------------------------
# 3. Wall isolation (the Phase 1 core)
# ---------------------------------------------------------------------------
def isolate_walls(ink: np.ndarray) -> tuple[np.ndarray, float]:
    """Split thick strokes (walls) from thin ones (furniture, text, hatching).

    Returns (wall_mask, median_wall_thickness_px) at canonical scale.
    """
    h, w = ink.shape
    binary = (ink > 0).astype(np.uint8)
    dt = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Ridge pixels: local maxima of the distance transform, i.e. one value per
    # stroke whose value is that stroke's half-thickness.
    ridge_mask = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt > 0.25)
    ridge = dt[ridge_mask]
    if ridge.size < 32:
        return np.zeros_like(ink), 0.0

    half_threshold = _wall_half_threshold(ridge)

    thick = dt >= half_threshold
    # Grow the thick ridge back to the full stroke width.
    radius = max(1, int(math.ceil(half_threshold)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    wall = (binary > 0) & cv2.dilate(thick.astype(np.uint8), kernel)

    # Bridge hairline gaps (anti-aliasing, scan breaks) without closing doorways.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    wall = cv2.morphologyEx(wall.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel)

    # Drop leftover blobs too small to be a wall run (text, fixtures).
    wall = _drop_speckle(wall.astype(np.uint8) * 255, min_area=max(500, int(0.00004 * h * w)))
    thickness = _median_thickness(wall, dt, half_threshold)
    return wall, thickness


def _wall_half_threshold(ridge: np.ndarray) -> float:
    """Half-thickness cutoff separating wall strokes from thin line work.

    Tuned against CubiCasa5K wall masks (scripts/eval_raster.py): half of the
    85th-percentile ridge thickness, floored at 1.5 px. A plain Otsu split on
    the same histogram was far worse (it keys off the dominant thin-stroke mode
    and drops entire interior walls), so this stays a fixed rule on purpose.
    """
    return float(max(1.5, 0.50 * np.percentile(ridge, 85)))


def _median_thickness(wall: np.ndarray, dt: np.ndarray, fallback: float) -> float:
    """Median full wall thickness (2 x half-thickness) sampled on the wall mask."""
    samples = dt[(wall > 0) & (dt > 0)]
    if samples.size == 0:
        return fallback * 2.0
    return float(2.0 * np.median(samples))


# ---------------------------------------------------------------------------
# 4. Skeleton -> wall centreline segments
# ---------------------------------------------------------------------------
def mask_to_segments(
    wall: np.ndarray, min_len_px: float = 12.0, eps_px: float = 1.2
) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], np.ndarray]:
    """Vectorise a wall mask into straight centreline segments.

    Returns (segments, skeleton) in canonical pixel coordinates.
    """
    skeleton = _thin(wall)
    segments = _walk_skeleton(skeleton, min_len_px=min_len_px, eps_px=eps_px)
    return segments, skeleton


def _thin(mask: np.ndarray) -> np.ndarray:
    """Morphological skeleton (iterative erosion + subtraction) — no skimage.

    NOTE: subtraction must saturate (cv2.subtract), not use raw uint8 numpy
    arithmetic, or underflow wraps around and the "skeleton" stays full-width.
    """
    img = (mask > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while img.any():
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(img, opened))
        img = cv2.erode(img, element)
    return skeleton


def _walk_skeleton(
    skeleton: np.ndarray, min_len_px: float, eps_px: float
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Turn skeleton pixels into polylines by walking 8-connected paths."""
    pixels = set(map(tuple, np.argwhere(skeleton > 0)))  # (row, col)
    if not pixels:
        return []

    def neighbours(p):
        y, x = p
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                q = (y + dy, x + dx)
                if q in pixels:
                    yield q

    degrees = {p: sum(1 for _ in neighbours(p)) for p in pixels}
    endpoints = [p for p, d in degrees.items() if d == 1]
    # Deterministic traversal: sorted endpoints first, then any leftovers (loops).
    queue = sorted(endpoints) + sorted(pixels - set(endpoints))
    visited: set[tuple[int, int]] = set()
    polylines: list[list[tuple[int, int]]] = []

    for start in queue:
        if start in visited:
            continue
        path = [start]
        visited.add(start)
        current = start
        while True:
            nxt = [q for q in neighbours(current) if q not in visited]
            if not nxt:
                break
            # Prefer the straightest continuation so polylines don't zigzag.
            if len(path) >= 2:
                py, px_ = path[-2]
                cy, cx = current
                dy, dx = cy - py, cx - px_
                nxt.sort(key=lambda q: -((q[0] - cy) * dy + (q[1] - cx) * dx))
            current = nxt[0]
            visited.add(current)
            path.append(current)
            if degrees.get(current, 0) >= 3:
                break  # stop at junctions; the other branches start on their own

        if len(path) < 3:
            continue
        pts = np.array([(x, y) for y, x in path], dtype=np.float32)  # (x, y)
        length = float(np.sum(np.hypot(*(np.diff(pts, axis=0).T))))
        if length < min_len_px:
            continue
        simplified = cv2.approxPolyDP(pts.reshape(-1, 1, 2), eps_px, closed=False)
        ring = simplified.reshape(-1, 2)
        for a, b in zip(ring[:-1], ring[1:]):
            if np.hypot(*(b - a)) < 1e-6:
                continue
            polylines.append([(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))])

    return polylines


# ---------------------------------------------------------------------------
# 5. Rooms (enclosed background regions)
# ---------------------------------------------------------------------------
def extract_rooms(ink: np.ndarray, min_area_px: int | None = None) -> list[np.ndarray]:
    """Room candidates = background components not touching the image border.

    Fills the role of the Shapely `polygonize` step for Phase 1: doorways that
    are open merge adjacent rooms, and furniture outlines inside a room come
    back as holes (dropped here). Phase 3 refines this with OCR labels and
    explicit door-gap detection.
    """
    h, w = ink.shape
    if min_area_px is None:
        min_area_px = max(400, int(0.0002 * h * w))

    background = (ink == 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(background, connectivity=4)

    border_labels = np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
    ]))
    enclosed = np.isin(labels, border_labels) == False  # noqa: E712
    enclosed &= background > 0

    contours, hierarchy = cv2.findContours(
        enclosed.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return []

    rooms = []
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] != -1:       # child contour = hole (furniture, etc.)
            continue
        if cv2.contourArea(contour) < min_area_px:
            continue
        ring = cv2.approxPolyDP(contour, 2.0, closed=True).reshape(-1, 2)
        if len(ring) >= 3:
            rooms.append(ring.astype(np.float32))
    return rooms


# ---------------------------------------------------------------------------
# 6. Assembly
# ---------------------------------------------------------------------------
def parse_raster(
    path: str | Path,
    pixels_per_meter: float | None = None,
    assumed_wall_m: float = DEFAULT_ASSUMED_WALL_M,
    dpi: int = 200,
    deskew: bool = False,
    extract_room_geometry: bool = True,
    min_room_area_m2: float = 1.0,
) -> NormalizedFloorplan:
    """Raster blueprint -> NormalizedFloorplan in metres (y-up)."""
    rgb = load_raster(path, dpi=dpi)
    ink, (orig_h, orig_w, factor) = preprocess(rgb, deskew=deskew)

    wall_mask, thickness_canon = isolate_walls(ink)
    if not wall_mask.any():
        raise ValueError("No walls detected — is this a floor plan?")

    segments, skeleton = mask_to_segments(wall_mask)
    if not segments:
        raise ValueError("Walls detected but no centrelines could be extracted.")

    warnings: list[str] = []
    scale_factor = 1.0 / factor  # canonical px -> original px

    # --- scale ------------------------------------------------------------
    ppm = pixels_per_meter
    if ppm is None:
        thickness_orig = thickness_canon * scale_factor
        ppm = thickness_orig / assumed_wall_m if thickness_orig > 0 else 50.0
        ppm = float(min(max(ppm, 5.0), 1000.0))
        warnings.append(
            f"Scale uncalibrated: estimated {ppm:.1f} px/m from median wall thickness "
            f"(assumed {assumed_wall_m:.2f} m). Provide pixels_per_meter to override."
        )
    elif ppm <= 0:
        raise ValueError("pixels_per_meter must be positive")

    def to_plan(x_can: float, y_can: float) -> tuple[float, float]:
        x_px = x_can * scale_factor
        y_px = y_can * scale_factor
        return (x_px / ppm, (orig_h - y_px) / ppm)

    walls = []
    for a, b in segments:
        ax, ay = to_plan(*a)
        bx, by = to_plan(*b)
        walls.append(Wall(start=Point(x=ax, y=ay), end=Point(x=bx, y=by)))

    rooms: list[Room] = []
    if extract_room_geometry:
        dropped = 0
        for i, ring in enumerate(extract_rooms(ink)):
            pts = []
            for x, y in ring:
                px, py = to_plan(float(x), float(y))
                pts.append(Point(x=px, y=py))
            area_m2 = _polygon_area_m2(pts)
            if area_m2 <= 0:
                continue
            # Text counters, furniture loops and hatching cells enclose a few
            # m² at most; real rooms are much larger. Phase 3 refines this.
            if area_m2 < min_room_area_m2:
                dropped += 1
                continue
            rooms.append(Room(
                id=f"room_{i}", label=None, polygon=pts,
                area_m2=round(area_m2, 3), inferred=True,
            ))
        if dropped:
            warnings.append(
                f"Dropped {dropped} enclosed regions smaller than "
                f"{min_room_area_m2:.1f} m² (text counters, furniture, hatching)."
            )
        if not rooms:
            warnings.append("No enclosed room regions found (open plan or doorways everywhere).")

    xs = [p.x for w_ in walls for p in (w_.start, w_.end)]
    ys = [p.y for w_ in walls for p in (w_.start, w_.end)]
    for room in rooms:
        xs += [p.x for p in room.polygon]
        ys += [p.y for p in room.polygon]
    if not xs:
        raise ValueError("Floor plan produced no geometry.")

    warnings.insert(0, "Raster plan: rooms inferred from enclosed regions (Phase 3 refines).")
    return NormalizedFloorplan(
        source_format="raster",
        units="m",
        bounds=(min(xs), min(ys), max(xs), max(ys)),
        walls=walls,
        rooms=rooms,
        warnings=warnings,
    )


def _polygon_area_m2(pts: list[Point]) -> float:
    """Shoelace area (y-up plan space)."""
    if len(pts) < 3:
        return 0.0
    total = 0.0
    for a, b in zip(pts, pts[1:] + pts[:1]):
        total += a.x * b.y - b.x * a.y
    return abs(total) / 2.0

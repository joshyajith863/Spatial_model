"""
CV perception pipeline: raster blueprint -> vector geometry (draft SKG input).

Planned stages (not yet implemented):
    1. preprocess(image)      -> deskew, normalize, upscale if low-res (OpenCV)
    2. detect_segment(image)  -> YOLOv10-seg: wall/door/window/room polygons
    3. read_text(image)       -> PaddleOCR (lite model): room labels, dimension strings
    4. calibrate_scale(...)   -> scale annotation -> dimension-line inference -> manual fallback
    5. build_draft_graph(...) -> assemble into app.skg.engine graph format,
                                  tagging low-confidence detections for human review
"""


def run(image_path: str) -> dict:
    """Entry point called from app.api.blueprints after upload. Returns a draft
    graph dict (nodes = rooms/doors/windows, edges = adjacency) plus a list of
    low-confidence detections for the human-verification overlay."""
    raise NotImplementedError("CV pipeline not yet implemented.")

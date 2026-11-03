"""
CV perception pipeline: raster blueprint -> vector geometry (draft SKG input).

Stages:
    1. preprocess(image)      -> deskew, normalize, upscale if low-res (OpenCV) [planned]
    2. detect_segment(image)  -> segmenter: wall/door/window/floor polygons       [planned]
    3. read_text(image)       -> OCR: room labels, dimension strings              [implemented]
    4. calibrate_scale(...)   -> scale annotation -> dimension-line -> manual      [planned]
    5. build_draft_graph(...) -> assemble into app.skg.engine graph format         [planned]
"""

from app.cv.ocr import get_ocr


def read_text(image_path: str) -> dict:
    """Stage 3: OCR a raster blueprint.

    Returns {"detections": [...], "image_size": (w, h)} where each detection is
    {"text", "confidence", "box", "center"}. Boxes/centres are in original image
    pixel coordinates, so they overlay the source image directly.
    """
    result = get_ocr().read(image_path)
    return {
        "image_size": result.image_size,
        "detections": [
            {
                "text": d.text,
                "confidence": round(d.confidence, 4),
                "box": [list(p) for p in d.box],
                "center": list(d.center),
            }
            for d in result.detections
        ],
    }


def run(image_path: str) -> dict:
    """Entry point called from app.api.blueprints after upload. Returns a draft
    graph dict (nodes = rooms/doors/windows, edges = adjacency) plus a list of
    low-confidence detections for the human-verification overlay."""
    raise NotImplementedError("Full CV pipeline not yet implemented (OCR stage is: read_text).")

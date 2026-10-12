"""
Vector PDF parser.

Only works for PDFs that were exported directly from CAD/drawing software
(i.e. the lines are real vector paths, not a scanned raster image embedded
in a PDF wrapper). If extraction returns ~0 lines, the PDF is almost
certainly a raster scan — route it to the CV pipeline (app/cv/pipeline.py)
instead.

Requires: pip install pymupdf
"""

import fitz  # PyMuPDF

from app.mesh.schema import NormalizedFloorplan, WallSegment


def is_vector_pdf(file_path: str, min_lines: int = 10) -> bool:
    """Quick check: does this PDF have enough vector drawing commands to be
    worth parsing as CAD data, or should it be treated as a raster scan?"""
    doc = fitz.open(file_path)
    line_count = 0
    for page in doc:
        drawings = page.get_drawings()
        for d in drawings:
            line_count += len(d.get("items", []))
    return line_count >= min_lines


def parse_pdf_vector(file_path: str, page_number: int = 0) -> NormalizedFloorplan:
    """
    Extracts line segments from a vector PDF's drawing commands.

    Note: PDF coordinates are in points (1/72 inch), not meters — this
    returns raw coordinates. Run the same scale-calibration step used for
    raster input (manual two-point calibration, see WORKPLAN.md) before
    treating these as real-world distances.
    """
    doc = fitz.open(file_path)
    page = doc[page_number]
    drawings = page.get_drawings()

    walls: list[WallSegment] = []
    for d in drawings:
        for item in d.get("items", []):
            if item[0] == "l":  # line: ("l", Point(start), Point(end))
                p1, p2 = item[1], item[2]
                walls.append(WallSegment(start=(p1.x, p1.y), end=(p2.x, p2.y)))
            elif item[0] == "re":  # rect: ("re", Rect) — common for wall outlines
                r = item[1]
                corners = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
                for i in range(4):
                    walls.append(WallSegment(start=corners[i], end=corners[(i + 1) % 4]))

    return NormalizedFloorplan(walls=walls, rooms=[], source_format="pdf_vector")

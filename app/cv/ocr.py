"""
Blueprint OCR — room labels and dimension strings from raster plans.

This is pipeline stage 3 (`read_text`) named in app/cv/pipeline.py, and the
`app.cv.ocr` module that app/cv/segmenter.py defers to: the segmenter is
*semantic* segmentation (walls/doors/windows/floor) with no notion of rooms, so
room names and dimension strings have to come from text recognition instead.

Engine: PaddleOCR (lite / mobile model), CPU-friendly. The model is loaded once
in __init__ and reused across images, mirroring FloorplanSegmenter.

Room-label assignment (OCR box -> containing room polygon) lives here too,
because it is the join between text and geometry and both inputs are needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Load-order critical on Windows: importing paddle/paddleocr BEFORE torch makes
# torch's shm.dll fail with WinError 127 (paddle ships a conflicting runtime).
# torch must win the race, so it is imported eagerly here, before anything from
# paddle is touched. Do not move or remove this import.
import torch  # noqa: F401
from shapely.geometry import Point, Polygon


# --- Result types ----------------------------------------------------------

@dataclass
class TextDetection:
    """One recognised text run. Box is 4 (x, y) corners in original image px."""

    text: str
    confidence: float
    box: list[tuple[float, float]]  # [[x,y], ...] length 4, TL/TR/BR/BL order

    @property
    def center(self) -> tuple[float, float]:
        xs = [p[0] for p in self.box]
        ys = [p[1] for p in self.box]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.box]
        ys = [p[1] for p in self.box]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class OCRResult:
    """All text found on one image, in original image pixel coordinates."""

    image_size: tuple[int, int]  # (width, height)
    detections: list[TextDetection] = field(default_factory=list)

    @property
    def texts(self) -> list[str]:
        return [d.text for d in self.detections]

    def min_confidence(self) -> float:
        return min((d.confidence for d in self.detections), default=1.0)


# --- Engine ----------------------------------------------------------------

# PaddleOCR model load costs several seconds, so the engine is created once and
# reused. Without this, every request would re-load the det/rec models.
_OCR: BlueprintOCR | None = None


def get_ocr(lang: str = "en") -> BlueprintOCR:
    """Return a process-wide cached BlueprintOCR (loads models on first use)."""
    global _OCR
    if _OCR is None or _OCR.lang != lang:
        _OCR = BlueprintOCR(lang=lang)
    return _OCR


class BlueprintOCR:
    """Load PaddleOCR once, then read many images. CPU-friendly.

    Written against PaddleOCR 3.7.0, whose result object exposes parallel
    `rec_texts` / `rec_scores` / `rec_polys` lists (the pre-3.x nested
    `[[box, (text, score)], ...]` layout is gone).
    """

    def __init__(self, lang: str = "en") -> None:
        from paddleocr import PaddleOCR

        # enable_mkldnn=False is REQUIRED on Windows CPU. With oneDNN on
        # (the default), inference dies in paddle's onednn_instruction.cc with
        # `ConvertPirAttribute2RuntimeAttribute not support
        # [pir::ArrayAttribute<pir::DoubleAttribute>]`. Forcing run_mode="paddle"
        # routes around it; see paddleocr/_common_args.py. Do not turn it back on.
        self._ocr = PaddleOCR(lang=lang, enable_mkldnn=False)
        self.lang = lang

    def read(self, image_path: str) -> OCRResult:
        """Run OCR on an image file and return detections in original pixels."""
        from PIL import Image

        with Image.open(image_path) as im:
            width, height = im.size

        results = self._ocr.predict(image_path)

        detections: list[TextDetection] = []
        for result in results or []:
            # OCRResult behaves like a dict of parallel lists.
            texts = result.get("rec_texts") or []
            scores = result.get("rec_scores") or []
            polys = result.get("rec_polys") or []
            for text, score, poly in zip(texts, scores, polys):
                text = str(text).strip()
                if not text:
                    continue
                box = [(float(p[0]), float(p[1])) for p in poly]
                if len(box) != 4:
                    continue
                detections.append(
                    TextDetection(text=text, confidence=float(score), box=box)
                )

        return OCRResult(image_size=(width, height), detections=detections)


# --- Room-label assignment -------------------------------------------------

def label_rooms(
    ocr: OCRResult,
    room_polygons: dict[str, Polygon],
    min_confidence: float = 0.5,
) -> tuple[dict[str, str], list[TextDetection]]:
    """Assign OCR text to the room polygon that contains it.

    Returns (room_id -> label, unassigned detections). A detection is a room
    label candidate only if its centre falls inside a room polygon and its
    confidence clears min_confidence. When several detections land in the same
    room the highest-confidence one wins; ties go to the shortest text (a room
    name beats a dimension string).
    """
    assigned: dict[str, str] = {}
    best_score: dict[str, tuple[float, int]] = {}
    unassigned: list[TextDetection] = []

    for det in ocr.detections:
        if det.confidence < min_confidence:
            unassigned.append(det)
            continue
        cx, cy = det.center
        point = Point(cx, cy)
        owner = next((rid for rid, poly in room_polygons.items() if poly is not None and poly.covers(point)), None)
        if owner is None:
            unassigned.append(det)
            continue

        # Rank: confidence first, then prefer shorter (label-like) text.
        rank = (det.confidence, -len(det.text))
        if owner not in best_score or rank > best_score[owner]:
            best_score[owner] = rank
            assigned[owner] = det.text

    return assigned, unassigned

"""
Pretrained floorplan segmenter.

Loads the MIT-licensed ResNet34-UNet from HuggingFace
(Yytsi/floorplan-to-3d-walls), trained on CubiCasa5K, and runs inference on a
raster blueprint to produce per-class polygons:

    floor (0) / wall (1) / door (2) / window (3)

This replaces the need to fine-tune YOLOv10-seg from the WORKPLAN: the model
already segments walls/doors/windows out of the box on CPU. Rooms and their
text labels come from OCR (app.cv.ocr), not from this model — it is semantic
segmentation, not instance/room detection.

Weights are expected at weights/best.safetensors with weights/config.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

# --- Class taxonomy (must match the training setup) -------------------------
CLASS_NAMES: tuple[str, ...] = ("floor", "wall", "door", "window")
CLASS_TO_ID: dict[str, int] = {n: i for i, n in enumerate(CLASS_NAMES)}
NUM_CLASSES: int = len(CLASS_NAMES)
FLOOR_ID: int = CLASS_TO_ID["floor"]

# Classes we turn into vector geometry. `floor` is the implicit negative space.
EXTRACT_CLASSES: tuple[str, ...] = ("wall", "door", "window")

# Visualization colors (not used in training).
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "floor": (240, 240, 235),
    "wall": (40, 40, 45),
    "door": (230, 120, 50),
    "window": (60, 150, 220),
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Extraction tuning (mirrors upstream extract_polygons.py) ---------------
CLOSING_KERNEL_PX: int = 3      # bridge hairline wall breaks
APPROX_EPSILON_PX: float = 1.5  # collapse pixel staircases on axis-aligned walls
MIN_POLYGON_AREA_PX: float = 30.0  # drop speckle below this area


@dataclass
class Polygon:
    outer: list[list[float]]                  # [[x, y], ...]
    holes: list[list[list[float]]] = field(default_factory=list)


@dataclass
class SegmentationResult:
    """Polygons are in ORIGINAL image pixel coordinates."""

    original_size: tuple[int, int]            # (width, height)
    polygons: dict[str, list[Polygon]]        # class -> polygons
    mask: np.ndarray                          # (H, W) uint8, model input space
    label_map: dict[int, str]                 # class id -> name

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.polygons.items()}


class FloorplanSegmenter:
    """Load the model once, then segment many images. CPU-friendly."""

    def __init__(
        self,
        weights_dir: str | Path = "weights",
        device: str = "auto",
        ckpt_name: str = "best.safetensors",
    ) -> None:
        weights_dir = Path(weights_dir)
        cfg_path = weights_dir / "config.yaml"
        ckpt_path = weights_dir / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Missing weights: {ckpt_path}. Download from "
                "https://huggingface.co/Yytsi/floorplan-to-3d-walls"
            )
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f)

        self.device = self._resolve_device(device)
        self.image_size: tuple[int, int] = tuple(cfg["data"]["image_size"])  # (H, W)
        self.letterbox: bool = bool(cfg["data"].get("letterbox", False))
        self.normalize: bool = bool(cfg["data"]["normalize"])

        # Build the UNet to match training, then load safetensors weights.
        import segmentation_models_pytorch as smp

        self.model = smp.Unet(
            encoder_name=cfg["model"]["encoder_name"],
            encoder_weights=None,       # weights come from the checkpoint
            in_channels=3,
            classes=NUM_CLASSES,
        ).to(self.device)

        from safetensors.torch import load_file

        state = load_file(str(ckpt_path), device="cpu" if self.device.type != "cuda" else "cuda")
        self.model.load_state_dict(state)
        self.model.eval()

    @staticmethod
    def _resolve_device(name: str) -> torch.device:
        if name != "auto":
            return torch.device(name)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _preprocess(self, image: Image.Image) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """Letterbox + normalize. Returns (tensor(3,H,W), (left, top, inner_w, inner_h))."""
        H, W = self.image_size
        orig_w, orig_h = image.size

        if self.letterbox:
            scale = min(W / orig_w, H / orig_h)
            inner_w = max(1, int(round(orig_w * scale)))
            inner_h = max(1, int(round(orig_h * scale)))
        else:
            inner_w, inner_h = W, H

        resized = image.convert("RGB").resize((inner_w, inner_h), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        if self.normalize:
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            tensor = (tensor - mean) / std

        if self.letterbox and (inner_h, inner_w) != (H, W):
            top = (H - inner_h) // 2
            left = (W - inner_w) // 2
            # Padding is zeros in normalized space (matches upstream).
            canvas = torch.zeros(3, H, W)
            canvas[:, top:top + inner_h, left:left + inner_w] = tensor
            return canvas, (left, top, inner_w, inner_h)

        return tensor, (0, 0, W, H)

    @torch.no_grad()
    def segment(self, image_path: str | Path) -> SegmentationResult:
        """Run inference and return vector polygons in original image coords."""
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

        tensor, (left, top, inner_w, inner_h) = self._preprocess(image)
        logits = self.model(tensor.unsqueeze(0).to(self.device))
        mask = logits.argmax(dim=1).squeeze(0).to("cpu", torch.uint8).numpy()

        # Zero out letterbox padding so we never extract from the gray border.
        if (inner_h, inner_w) != mask.shape:
            cleaned = np.full_like(mask, FLOOR_ID)
            cleaned[top:top + inner_h, left:left + inner_w] = mask[top:top + inner_h, left:left + inner_w]
            mask = cleaned

        raw_polygons = self._mask_to_polygons(mask)

        # Map model-space coords back to original image pixel coords.
        scale_x = inner_w / orig_w
        scale_y = inner_h / orig_h

        def _map_ring(ring: list[list[float]]) -> list[list[float]]:
            return [[(x - left) / scale_x, (y - top) / scale_y] for x, y in ring]

        mapped: dict[str, list[Polygon]] = {}
        for cls, polys in raw_polygons.items():
            mapped[cls] = [
                Polygon(outer=_map_ring(p.outer), holes=[_map_ring(h) for h in p.holes])
                for p in polys
            ]

        return SegmentationResult(
            original_size=(orig_w, orig_h),
            polygons=mapped,
            mask=mask,
            label_map=dict(CLASS_TO_ID),
        )

    # --- polygon extraction (ported from upstream) --------------------------

    @staticmethod
    def _approx(contour: np.ndarray) -> list[list[float]]:
        simplified = cv2.approxPolyDP(contour, APPROX_EPSILON_PX, closed=True)
        return simplified.reshape(-1, 2).astype(float).tolist()

    @classmethod
    def _polygons_for_class(cls, mask: np.ndarray, class_id: int) -> list[Polygon]:
        binary = (mask == class_id).astype(np.uint8)
        if binary.sum() == 0:
            return []

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSING_KERNEL_PX, CLOSING_KERNEL_PX))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            return []
        hierarchy = hierarchy[0]

        holes_by_parent: dict[int, list[int]] = {}
        for i, (_, _, _, parent) in enumerate(hierarchy):
            if parent != -1:
                holes_by_parent.setdefault(parent, []).append(i)

        polygons: list[Polygon] = []
        for i, (_, _, _, parent) in enumerate(hierarchy):
            if parent != -1:
                continue
            if cv2.contourArea(contours[i]) < MIN_POLYGON_AREA_PX:
                continue
            outer = cls._approx(contours[i])
            if len(outer) < 3:
                continue
            holes: list[list[list[float]]] = []
            for j in holes_by_parent.get(i, []):
                if cv2.contourArea(contours[j]) < MIN_POLYGON_AREA_PX:
                    continue
                ring = cls._approx(contours[j])
                if len(ring) >= 3:
                    holes.append(ring)
            polygons.append(Polygon(outer=outer, holes=holes))
        return polygons

    @classmethod
    def _mask_to_polygons(cls, mask: np.ndarray) -> dict[str, list[Polygon]]:
        return {c: cls._polygons_for_class(mask, CLASS_TO_ID[c]) for c in EXTRACT_CLASSES}

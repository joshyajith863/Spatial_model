#!/usr/bin/env python3
"""
Spatial Copilot - main loader / control panel.

One entry point for the three things you do with this repo:

    1. Train   fine-tune the YOLO11n-seg model on CubiCasa5K (live progress bars)
    2. Load    instantiate the pretrained UNet segmenter or a YOLO checkpoint
    3. Host    serve the FastAPI app + WebXR viewer over the LAN

Interactive menu (no args):

    python main.py

Or script it directly:

    python main.py train --epochs 5 --batch 8
    python main.py train --dry-run
    python main.py load segmenter --image path/to/plan.png
    python main.py load yolo --weights runs/segment/cubicasa/weights/best.pt
    python main.py serve --port 8000 --open
    python main.py download
    python main.py check

Run with the project venv, e.g.:
    .\\venv\\Scripts\\python.exe main.py
"""

from __future__ import annotations

import argparse
import importlib.util
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Project defaults (kept in sync with scripts/train_yolo.py) -------------
DEFAULT_DATA = ROOT / "data" / "yolo" / "data.yaml"
DEFAULT_BASE_MODEL = "yolo11n-seg.pt"
DEFAULT_PROJECT = ROOT / "runs" / "segment"
DEFAULT_NAME = "cubicasa"
WEIGHTS_DIR = ROOT / "weights"

# --- Tiny terminal UI ------------------------------------------------------
try:
    "█░".encode(sys.stdout.encoding or "ascii")
    FULL, EMPTY = "█", "░"
except (UnicodeEncodeError, LookupError):
    FULL, EMPTY = "#", "-"


def bar(fraction: float, width: int = 26) -> str:
    """Render a fixed-width progress bar for a 0..1 fraction."""
    try:
        f = float(fraction)
    except (TypeError, ValueError):
        f = 0.0
    f = min(1.0, max(0.0, f))
    filled = int(round(f * width))
    return f"[{FULL * filled}{EMPTY * (width - filled)}]"


def pct(fraction: float) -> str:
    return f"{max(0.0, min(1.0, float(fraction))) * 100:5.1f}%"


def human_time(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def banner(title: str) -> None:
    line = "=" * 62
    print(f"\n{line}\n  {title}\n{line}", flush=True)


def pause() -> None:
    try:
        input("\nPress Enter to return to the menu...")
    except (EOFError, KeyboardInterrupt):
        pass


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _resolve_device(device: str) -> str:
    """'auto' -> '0' when CUDA is present, else 'cpu' (Ultralytics spelling)."""
    if str(device).lower() != "auto":
        return str(device)
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


# ==========================================================================
# 1. TRAIN
# ==========================================================================
class TrainDisplay:
    """Ultralytics callbacks that print the run as live progress bars."""

    def __init__(self) -> None:
        self.started = time.time()
        self.last_epoch_at = self.started
        self.history: list[dict] = []

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _metrics(trainer) -> dict:
        return dict(getattr(trainer, "metrics", {}) or {})

    @classmethod
    def _metric(cls, trainer, needle: str) -> float | None:
        """Look up a metric like 'precision' or 'mAP50-95', preferring mask (M)."""
        metrics = cls._metrics(trainer)
        for suffix in ("(M)", "(B)"):
            key = f"metrics/{needle}{suffix}"
            if key in metrics:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    return None
        for key, value in metrics.items():
            if key.startswith("metrics/") and needle in key:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _loss(trainer) -> float | None:
        tloss = getattr(trainer, "tloss", None)
        if tloss is None:
            return None
        try:
            if isinstance(tloss, dict):
                return float(sum(float(v) for v in tloss.values()))
            values = tloss.tolist() if hasattr(tloss, "tolist") else tloss
            if isinstance(values, (list, tuple)):
                return float(sum(values))
            return float(values)
        except (TypeError, ValueError):
            return None

    # -- callbacks ---------------------------------------------------------
    def on_train_start(self, trainer) -> None:
        self.started = self.last_epoch_at = time.time()
        total = int(getattr(trainer, "epochs", 0) or 0)
        save_dir = getattr(trainer, "save_dir", None)
        print(flush=True)
        print(f"  training {total} epoch(s) - Ctrl+C to stop (resume from last.pt)")
        if save_dir:
            print(f"  run dir  {save_dir}")
        print(flush=True)

    def on_fit_epoch_end(self, trainer) -> None:
        now = time.time()
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        total = max(1, int(getattr(trainer, "epochs", 0) or 1))
        # Ultralytics runs a final validation after the last epoch, which fires
        # this callback once more with epoch == total + 1. Fold those final
        # numbers into the last epoch instead of printing a bogus extra row.
        if epoch > total:
            if self.history:
                last = self.history[-1]
                last["precision"] = self._metric(trainer, "precision")
                last["recall"] = self._metric(trainer, "recall")
                last["mAP50"] = self._metric(trainer, "mAP50")
                last["mAP50-95"] = self._metric(trainer, "mAP50-95")
            return

        entry = {
            "epoch": epoch,
            "loss": self._loss(trainer),
            "precision": self._metric(trainer, "precision"),
            "recall": self._metric(trainer, "recall"),
            "mAP50": self._metric(trainer, "mAP50"),
            "mAP50-95": self._metric(trainer, "mAP50-95"),
        }
        self.history.append(entry)

        elapsed = now - self.started
        avg = elapsed / max(1, len(self.history))
        eta = avg * (total - epoch)
        loss = entry["loss"]
        loss_txt = f"loss {loss:.4f}" if loss is not None else "loss   n/a"

        print(
            f"Epoch {epoch:3d}/{total:<3d} {bar(epoch / total)} {pct(epoch / total)}"
            f"  {loss_txt}  elapsed {human_time(elapsed)}  eta {human_time(eta)}",
            flush=True,
        )

        parts = []
        for label, key in (
            ("P", "precision"),
            ("R", "recall"),
            ("mAP50", "mAP50"),
            ("mAP50-95", "mAP50-95"),
        ):
            value = entry.get(key)
            if value is None:
                parts.append(f"{label} {bar(0, 12)} n/a")
            else:
                parts.append(f"{label} {bar(value, 12)} {value * 100:5.1f}%")
        print("           " + "   ".join(parts), flush=True)

        self.last_epoch_at = now

    def summary(self) -> None:
        if not self.history:
            print("  no epochs completed.", flush=True)
            return

        ranked = [h for h in self.history if h.get("mAP50-95") is not None]
        if ranked:
            best = max(ranked, key=lambda h: h["mAP50-95"])
            mode = "mAP50-95"
        else:
            ranked = [h for h in self.history if h.get("mAP50") is not None]
            best = max(ranked, key=lambda h: h["mAP50"]) if ranked else self.history[-1]
            mode = "mAP50"

        banner(f"Training complete - {len(self.history)} epoch(s), {human_time(time.time() - self.started)}")
        print(f"  best epoch : {best.get('epoch')} (by {mode})", flush=True)
        for label, key in (
            ("loss", "loss"),
            ("precision", "precision"),
            ("recall", "recall"),
            ("mAP50", "mAP50"),
            ("mAP50-95", "mAP50-95"),
        ):
            value = best.get(key)
            if value is None:
                continue
            if key == "loss":
                print(f"  {label:<10}: {value:.4f}", flush=True)
            else:
                print(f"  {label:<10}: {bar(value)} {value * 100:5.1f}%", flush=True)


def train_model(
    *,
    data: str | Path = DEFAULT_DATA,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = 10,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "auto",
    project: str | Path = DEFAULT_PROJECT,
    name: str = DEFAULT_NAME,
    workers: int = 2,
    patience: int = 20,
    seed: int = 0,
    exist_ok: bool = False,
    amp: bool = True,
    resume: str | Path | None = None,
    dry_run: bool = False,
) -> int:
    if not _has_module("ultralytics"):
        print("  ! ultralytics is not installed for this interpreter.")
        print("    pip install ultralytics   (or use the venv: .\\venv\\Scripts\\python.exe main.py)")
        return 1

    data_path = Path(data)
    if not resume and not data_path.exists():
        print(f"  ! dataset config not found: {data_path}")
        print("    build it first:  python scripts/convert_cubicasa_yolo.py")
        return 1

    device = _resolve_device(device)
    project_path = Path(project)
    if not project_path.is_absolute():
        project_path = ROOT / project_path

    banner("Train - YOLO11 segmentation")
    print(f"  model   : {resume or base_model}")
    print(f"  data    : {data_path if not resume else '(from checkpoint)'}")
    print(f"  device  : {device}   epochs={epochs}  imgsz={imgsz}  batch={batch}")
    if device == "cpu":
        print("  ! CPU-only: expect ~1h+/epoch on the full set. Use --epochs/--limit to start small.")

    if dry_run:
        print("\n  [dry-run] config OK, not training. Would call:")
        print(f"    YOLO({base_model!r}).train(data={str(data_path)!r}, epochs={epochs}, ...)")
        return 0

    from ultralytics import YOLO

    model = YOLO(str(resume) if resume else base_model)
    display = TrainDisplay()
    model.add_callback("on_train_start", display.on_train_start)
    model.add_callback("on_fit_epoch_end", display.on_fit_epoch_end)

    kwargs = dict(
        data=str(data_path),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        device=device,
        workers=int(workers),
        project=str(project_path),
        name=name,
        exist_ok=bool(exist_ok),
        patience=int(patience),
        cache=False,
        amp=bool(amp),
        verbose=False,   # our progress display replaces the default table
        plots=True,
        seed=int(seed),
    )

    try:
        if resume:
            model.train(resume=True)
        else:
            model.train(**kwargs)
    except KeyboardInterrupt:
        print("\n\n  interrupted - check runs/ for last.pt, then resume with:")
        print(f"    python main.py train --resume runs/segment/{name}/weights/last.pt")
        return 130

    display.summary()

    # model.train() returns metrics; the run directory lives on the trainer.
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir:
        best = Path(save_dir) / "weights" / "best.pt"
        if best.exists():
            print(f"\n  best weights: {best}", flush=True)
        print(f"  results csv : {Path(save_dir) / 'results.csv'}", flush=True)
    return 0


def interactive_train() -> int:
    print("\n  Leave blank to accept the default.\n")
    epochs = ask_int("epochs", 10, minimum=1)
    batch = ask_int("batch size", 8, minimum=1)
    imgsz = ask_int("image size", 640, minimum=64)
    device = ask("device ('auto', 'cpu' or '0')", "auto")
    data = ask("dataset yaml", str(DEFAULT_DATA))
    return train_model(
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
    )


# ==========================================================================
# 2. LOAD MODELS
# ==========================================================================
def find_yolo_checkpoints(limit: int = 10) -> list[Path]:
    runs = ROOT / "runs"
    if not runs.exists():
        return []
    found = sorted(
        (p for p in runs.rglob("*.pt") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    # Prefer "best.pt" files, then most recently modified.
    found.sort(key=lambda p: (p.name != "best.pt", -p.stat().st_mtime))
    return found[:limit]


def load_segmenter(weights_dir: str | Path = WEIGHTS_DIR, device: str = "auto",
                   image: str | Path | None = None) -> int:
    weights_dir = Path(weights_dir)
    missing = [n for n in ("best.safetensors", "config.yaml") if not (weights_dir / n).exists()]
    if missing:
        print(f"  ! missing weights in {weights_dir}: {', '.join(missing)}")
        print("    run:  python main.py download")
        return 1

    banner("Load - floorplan segmenter (ResNet34-UNet)")
    print(f"  weights : {weights_dir / 'best.safetensors'}")
    try:
        from app.cv.segmenter import CLASS_NAMES, FloorplanSegmenter

        segmenter = FloorplanSegmenter(weights_dir=weights_dir, device=device)
    except Exception as exc:
        print(f"  ! could not load segmenter: {type(exc).__name__}: {exc}")
        return 1

    print(f"  device  : {segmenter.device}")
    print(f"  input   : {segmenter.image_size} (H, W)  letterbox={segmenter.letterbox}")
    print(f"  classes : {', '.join(CLASS_NAMES)}")
    print("  status  : loaded OK", flush=True)

    if image:
        path = Path(image)
        if not path.exists():
            print(f"  ! image not found: {path}")
            return 1
        print(f"\n  segmenting {path} ...", flush=True)
        t0 = time.time()
        result = segmenter.segment(path)
        print(f"  done in {time.time() - t0:.2f}s - {result.original_size[0]}x{result.original_size[1]}")
        for cls, count in result.counts().items():
            print(f"    {cls:<7}: {count} polygon(s)")
    return 0


def load_yolo(weights: str | Path | None = None, image: str | Path | None = None,
              device: str = "auto", conf: float = 0.25) -> int:
    if not _has_module("ultralytics"):
        print("  ! ultralytics is not installed for this interpreter.")
        return 1

    if weights is None:
        candidates = find_yolo_checkpoints()
        if candidates:
            print("  found checkpoints:")
            for i, path in enumerate(candidates, 1):
                print(f"    {i}) {path.relative_to(ROOT)}")
            weights = ask("checkpoint path or number", str(candidates[0]))
            if str(weights).isdigit():
                idx = int(weights) - 1
                weights = candidates[idx] if 0 <= idx < len(candidates) else candidates[0]
        else:
            weights = ask("checkpoint path", DEFAULT_BASE_MODEL)

    weights = str(weights)
    banner("Load - YOLO segmentation checkpoint")
    print(f"  weights : {weights}")
    try:
        from ultralytics import YOLO

        model = YOLO(weights)
    except Exception as exc:
        print(f"  ! could not load YOLO model: {type(exc).__name__}: {exc}")
        return 1

    names = getattr(model, "names", {}) or {}
    print(f"  task    : {getattr(model, 'task', 'unknown')}")
    print(f"  classes : {', '.join(str(v) for v in names.values()) if isinstance(names, dict) else names}")
    print("  status  : loaded OK", flush=True)

    if image:
        path = Path(image)
        if not path.exists():
            print(f"  ! image not found: {path}")
            return 1
        print(f"\n  predicting on {path} ...", flush=True)
        device = _resolve_device(device)
        t0 = time.time()
        results = model.predict(str(path), conf=conf, device=device, verbose=False)
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.2f}s")
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                print("    no detections")
                continue
            counts: dict[str, int] = {}
            for cls_id in boxes.cls.tolist():
                label = names.get(int(cls_id), str(int(cls_id))) if isinstance(names, dict) else str(int(cls_id))
                counts[label] = counts.get(label, 0) + 1
            for label, count in sorted(counts.items()):
                print(f"    {label:<10}: {count} detection(s)  {bar(min(1.0, count / 20), 12)}")
    return 0


def interactive_load() -> int:
    print("\n  1) Floorplan segmenter (weights/best.safetensors)")
    print("  2) YOLO checkpoint (runs/**/best.pt or a .pt file)")
    choice = ask("choose", "1")
    image = ask("image to run inference on (blank = just load)", "")
    image = image or None
    if choice.strip() == "2":
        return load_yolo(image=image)
    return load_segmenter(image=image)


# ==========================================================================
# 3. HOST
# ==========================================================================
def lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def serve(*, host: str = "0.0.0.0", port: int = 8000, reload: bool = False,
          open_browser: bool = False) -> int:
    static_dir = ROOT / "app" / "static"
    if not static_dir.exists():
        print(f"  ! app/static not found at {static_dir}")
        return 1
    if not _has_module("uvicorn"):
        print("  ! uvicorn is not installed for this interpreter.")
        return 1

    banner("Host - Spatial Copilot API + WebXR viewer")
    local = "127.0.0.1"
    ips = [ip for ip in (local, lan_ip()) if ip]
    for ip in dict.fromkeys(ips):
        print(f"  viewer : http://{ip}:{port}/")
        print(f"  docs   : http://{ip}:{port}/docs")
        print(f"  health : http://{ip}:{port}/health")
    print(f"\n  host={host} port={port} reload={reload}")
    print("  Ctrl+C to stop.\n", flush=True)

    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{local}:{port}/")).start()

    import uvicorn

    try:
        uvicorn.run("app.main:app", host=host, port=int(port), reload=bool(reload))
    except KeyboardInterrupt:
        print("\n  server stopped.")
    except OSError as exc:
        print(f"\n  ! could not start server on {host}:{port}: {exc}")
        print("    Is another process using that port? Try --port 8001.")
        return 1
    return 0


def interactive_serve() -> int:
    host = ask("host", "0.0.0.0")
    port = ask_int("port", 8000, minimum=1)
    reload = ask_yes_no("auto-reload on file changes", False)
    open_browser = ask_yes_no("open the viewer in a browser", True)
    return serve(host=host, port=port, reload=reload, open_browser=open_browser)


# ==========================================================================
# 4. DOWNLOAD WEIGHTS
# ==========================================================================
def download_weights() -> int:
    script = ROOT / "scripts" / "download_weights.py"
    if not script.exists():
        print(f"  ! {script} not found")
        return 1
    banner("Download - pretrained segmenter weights")
    return subprocess.call([sys.executable, str(script)], cwd=str(ROOT))


# ==========================================================================
# 5. SYSTEM CHECK
# ==========================================================================
def system_check() -> int:
    banner("System check")
    print(f"  python      : {sys.version.split()[0]}  ({sys.executable})")
    print(f"  project     : {ROOT}")

    packages = (
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("torch", "torch"),
        ("ultralytics", "ultralytics"),
        ("segmentation_models_pytorch", "segmentation-models-pytorch"),
        ("safetensors", "safetensors"),
        ("cv2", "opencv-python-headless"),
        ("trimesh", "trimesh"),
    )
    print("\n  packages:")
    missing: list[str] = []
    for module, pip_name in packages:
        ok = _has_module(module)
        print(f"    [{'x' if ok else ' '}] {module:<30} {'ok' if ok else f'pip install {pip_name}'}")
        if not ok:
            missing.append(pip_name)

    if _has_module("torch"):
        try:
            import torch

            cuda = torch.cuda.is_available()
            gpu = torch.cuda.get_device_name(0) if cuda else "none"
            print(f"\n  torch       : {torch.__version__}  cuda={cuda}  gpu={gpu}")
        except Exception as exc:
            print(f"\n  torch       : import failed ({exc})")

    print("\n  data:")
    train_imgs = list((ROOT / "data" / "yolo" / "images" / "train").glob("*")) if (ROOT / "data" / "yolo" / "images" / "train").exists() else []
    val_imgs = list((ROOT / "data" / "yolo" / "images" / "valid").glob("*")) if (ROOT / "data" / "yolo" / "images" / "valid").exists() else []
    print(f"    dataset yaml : {'ok' if DEFAULT_DATA.exists() else 'MISSING (run scripts/convert_cubicasa_yolo.py)'}")
    print(f"    train images : {len(train_imgs)}")
    print(f"    val images   : {len(val_imgs)}")

    ckpt = WEIGHTS_DIR / "best.safetensors"
    print(f"    UNet weights : {'ok' if ckpt.exists() else 'MISSING (python main.py download)'}")

    yolo_ckpts = find_yolo_checkpoints(limit=3)
    print(f"    YOLO ckpts   : {len(yolo_ckpts)} found" + (f" (newest: {yolo_ckpts[0].relative_to(ROOT)})" if yolo_ckpts else ""))
    print(f"    viewer       : {'ok' if (ROOT / 'app' / 'static' / 'viewer.html').exists() else 'MISSING'}")

    if missing:
        print("\n  install missing packages with:")
        print(f"    pip install {' '.join(dict.fromkeys(missing))}")
    return 0


# ==========================================================================
# Interactive menu
# ==========================================================================
def ask(prompt: str, default: str = "") -> str:
    try:
        raw = input(f"  {prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(130)
    return raw or default


def ask_int(prompt: str, default: int, minimum: int | None = None) -> int:
    while True:
        raw = ask(prompt, str(default))
        try:
            value = int(raw)
        except ValueError:
            print("    ! enter a whole number")
            continue
        if minimum is not None and value < minimum:
            print(f"    ! must be >= {minimum}")
            continue
        return value


def ask_yes_no(prompt: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        raw = ask(f"{prompt} ({suffix})", "").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("    ! answer y or n")


MENU = """
==============================================================
  Spatial Copilot - main loader
==============================================================
  1) Train YOLO segmentation model
  2) Load models (segmenter / YOLO)
  3) Host API + WebXR viewer
  4) Download pretrained segmenter weights
  5) System check
  0) Exit
=============================================================="""


def interactive_menu() -> int:
    print(MENU)
    while True:
        try:
            choice = ask("select", "0").strip()
        except SystemExit:
            print()
            return 0

        try:
            if choice in {"0", "q", "quit", "exit"}:
                print("  bye.")
                return 0
            if choice == "1":
                interactive_train()
            elif choice == "2":
                interactive_load()
            elif choice == "3":
                interactive_serve()
            elif choice == "4":
                download_weights()
            elif choice == "5":
                system_check()
            else:
                print("  ! unknown option")
        except KeyboardInterrupt:
            print("\n  cancelled.")
        except Exception as exc:  # keep the menu alive on unexpected errors
            print(f"\n  ! {type(exc).__name__}: {exc}")

        pause()
        print(MENU)


# ==========================================================================
# CLI
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Spatial Copilot loader: train, load models, or host the app.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py                              interactive menu\n"
            "  python main.py train --epochs 5 --batch 8\n"
            "  python main.py load segmenter --image plan.png\n"
            "  python main.py serve --port 8000 --open\n"
        ),
    )
    sub = parser.add_subparsers(dest="command")

    train = sub.add_parser("train", help="fine-tune the YOLO segmentation model")
    train.add_argument("--data", default=str(DEFAULT_DATA))
    train.add_argument("--model", default=DEFAULT_BASE_MODEL, dest="base_model")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--device", default="auto", help="'auto', 'cpu', '0' or '0,1'")
    train.add_argument("--project", default=str(DEFAULT_PROJECT))
    train.add_argument("--name", default=DEFAULT_NAME)
    train.add_argument("--workers", type=int, default=2)
    train.add_argument("--patience", type=int, default=20)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--exist-ok", action="store_true")
    train.add_argument("--no-amp", action="store_true")
    train.add_argument("--resume", default=None, help="path to last.pt to resume")
    train.add_argument("--dry-run", action="store_true")

    load = sub.add_parser("load", help="load and optionally run a model")
    load.add_argument("model", nargs="?", choices=("segmenter", "yolo"), default=None)
    load.add_argument("--weights", default=None)
    load.add_argument("--image", default=None, help="optional image for a test inference")
    load.add_argument("--device", default="auto")
    load.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")

    serve_p = sub.add_parser("serve", help="host the FastAPI app + viewer")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--reload", action="store_true")
    serve_p.add_argument("--open", action="store_true", help="open the viewer in a browser")

    sub.add_parser("download", help="download the pretrained UNet weights")
    sub.add_parser("check", help="show environment / dataset status")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command is None:
        return interactive_menu()
    if args.command == "train":
        return train_model(
            data=args.data,
            base_model=args.base_model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            workers=args.workers,
            patience=args.patience,
            seed=args.seed,
            exist_ok=args.exist_ok,
            amp=not args.no_amp,
            resume=args.resume,
            dry_run=args.dry_run,
        )
    if args.command == "load":
        if args.model is None:
            return interactive_load()
        if args.model == "yolo":
            return load_yolo(args.weights, args.image, args.device, args.conf)
        return load_segmenter(device=args.device, image=args.image)
    if args.command == "serve":
        return serve(host=args.host, port=args.port, reload=args.reload, open_browser=args.open)
    if args.command == "download":
        return download_weights()
    if args.command == "check":
        return system_check()
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n  interrupted.")
        raise SystemExit(130)

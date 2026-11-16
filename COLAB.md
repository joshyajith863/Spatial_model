# Training on Google Colab (GPU)

The repo is self-contained: **code is in git, the dataset is not** (`data/` is
gitignored — 4k images don't belong in git). `scripts/convert_cubicasa_yolo.py`
rebuilds the dataset by downloading CubiCasa5K (COCO/Parquet) from HuggingFace,
so a fresh Colab session only needs the small code clone.

Open `notebooks/colab_train.ipynb` in Colab for the click-through version of
these same steps.

> Training on Kaggle instead? Use **`kaggle.ipynb`** at the repo root — same pipeline,
> with the dataset in `/kaggle/temp` and weights in `/kaggle/working`.

---

## 1. Clone + install (2 minutes)

```bash
!git clone https://github.com/joshyajith863/Spatial_model.git
%cd Spatial_model

# Training deps only. Do NOT `pip install -r requirements.txt` on Colab:
# it pins CPU torch and will replace Colab's CUDA build.
!pip install -q ultralytics pandas pyarrow huggingface_hub pyyaml
```

Check the GPU:

```bash
!nvidia-smi
```

## 2. Build the dataset (5–10 minutes, downloads ~1.5 GB)

```bash
!python scripts/convert_cubicasa_yolo.py --split both
```

Produces `data/yolo/{images,labels}/{train,valid}` plus `data/yolo/data.yaml`
with the **local** absolute root.

> The converter writes an absolute `path:` into `data.yaml`. If you instead copy
> a dataset from another machine (e.g. a Windows checkout), the training scripts
> now rewrite that root automatically to the yaml's own folder — you don't need
> to edit the file by hand.

## 3. Train

```bash
!python scripts/train_yolo.py --device 0 --epochs 50 --batch 16 --imgsz 640
```

or the loader, which prints a live progress bar per epoch:

```bash
!python main.py train --device 0 --epochs 50 --batch 16
```

Useful flags:

| flag | why |
|---|---|
| `--device 0` | first GPU (`cpu` is the default in the script) |
| `--batch 16` | T4/free tier fits this at 640; drop to 8 on OOM |
| `--workers 4` | Linux dataloader workers (Windows default of 2 is conservative) |
| `--limit 200` | 200 images per split — smoke-test the pipeline in minutes |
| `--resume runs/segment/cubicasa/weights/last.pt` | continue after a disconnect (loader only) |
| `--no-amp` | disable mixed precision if you hit NaNs |

## 4. Checkpoints & resume (what actually matters)

Ultralytics already checkpoints **every epoch** into the run directory:

| file | role |
|---|---|
| `weights/last.pt` | exact resume point (weights + optimizer + EMA + scaler + epoch + args) |
| `weights/best.pt` | best validation fitness so far — this is what you deploy |
| `results.csv` | per-epoch metrics, so a resumed run continues the same curve |
| `weights/epochN.pt` | only with `--save-period N` |

So the problem is **not** saving state — it's *where*. `runs/` is gitignored and Colab's VM
is disposable, so point the run directory at Drive:

```python
from google.colab import drive; drive.mount('/content/drive')
```

```bash
!python main.py train --device 0 --epochs 50 --batch 16 --save-period 5 \
    --project /content/drive/MyDrive/spatial_runs/segment --name cubicasa
```

Weights land in `.../spatial_runs/segment/cubicasa/weights/`. Nothing to copy afterwards.

**Resume after a disconnect / 12 h timeout:**

```bash
!python main.py train --resume /content/drive/MyDrive/spatial_runs/segment/cubicasa/weights/last.pt --device 0
```

It continues in the *checkpoint's own* directory and restores epoch count, LR schedule,
optimizer and augmentation state — no arguments from the old command needed. `main.py`
prints the progress it found (`12/50 epochs done -> <dir>`) before starting.

Two details worth knowing:

- Only an **unfinished** run can be resumed. A finished run's checkpoint has `epoch = -1`;
  `main.py` warns and Ultralytics starts a fresh run instead.
- Don't hand-roll the generic `torch.save({'model_state_dict': ...})` recipe you find in blog
  posts. Ultralytics checkpoints already carry more state than that, and replacing them
  desynchronises the trainer (LR schedule, warmup, mosaic-disable-at-`close_mosaic`, AMP
  scaler). Use `--resume`; it's the supported path.

Drive writes are slower than local disk, but a checkpoint is ~6 MB per epoch — negligible.

## 5. Evaluate

```bash
!python scripts/eval_yolo.py --weights /content/drive/MyDrive/spatial_runs/cubicasa/weights/best.pt
```

Per-class IoU table (floor / wall / door / window). Copy `best.pt` into
`weights/` on your local machine for the app to use:

```bash
python main.py download          # segmenter weights (CPU)
python main.py load yolo --weights path/to/best.pt
```

## Gotchas

- **`ModuleNotFoundError: ultralytics`** → step 1 was skipped, or you pip-installed
  into a different interpreter.
- **`dataset not found: data/yolo/data.yaml`** → run step 2 first.
- **`CUDA out of memory`** → `--batch 8` (or `--imgsz 512`).
- **Training on CPU by accident** → pass `--device 0`; the scripts default to CPU.
- **`Invalid CUDA 'device=0' requested` / `torch 2.x.y+cpu`** → a CPU-only torch wheel
  replaced Colab's CUDA build (nearly always `pip install -r requirements.txt`). Repair, then
  **restart the runtime** — the new wheel only loads in a fresh kernel:

  ```bash
  !pip uninstall -y torch torchvision
  !pip install --force-reinstall torch torchvision
  import torch; print(torch.cuda.is_available())   # must be True
  ```

  `main.py train` and `scripts/train_yolo.py` now print this diagnosis instead of
  Ultralytics' three-line traceback, and exit 1 without starting a run.
- **Disconnected runtime** → rerun steps 1–2, then resume from Drive
  (`--resume .../last.pt`). The HF dataset download is cached within a session only.
- **Raster Phase 1 eval needs no GPU**: `python scripts/eval_raster.py --limit 100`
  after `pip install -q opencv-python-headless pymupdf pyyaml`.

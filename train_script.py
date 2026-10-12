print("YOLO11 DETECTION TRAINING STARTED")

from pathlib import Path
from ultralytics import YOLO
import torch

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_YAML = BASE_DIR / "archery_yolo_dataset" / "data.yaml"

RUNS_DIR = BASE_DIR / "runs"

MODEL_WEIGHTS = "yolo11s.pt"      # Options: yolo11s.pt, yolo11m.pt

# ============================================================
# TRAIN CONFIG
# ============================================================

EPOCHS = 200
IMAGE_SIZE = 1280
BATCH_SIZE = 2       # Use 2 if you get CUDA OOM
DEVICE = 0
WORKERS = 0
PATIENCE = 50

RUN_NAME = "archery_box_model"

# ============================================================
# CHECK GPU
# ============================================================

print("=" * 60)

print("Torch :", torch.__version__)
print("CUDA Available :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA Version :", torch.version.cuda)
    print("GPU :", torch.cuda.get_device_name(0))

print("=" * 60)

# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(MODEL_WEIGHTS)

# ============================================================
# TRAIN
# ============================================================

results = model.train(

    data=str(DATA_YAML),

    epochs=EPOCHS,

    imgsz=IMAGE_SIZE,

    batch=BATCH_SIZE,

    device=DEVICE,

    workers=WORKERS,

    patience=PATIENCE,

    project=str(RUNS_DIR),

    name=RUN_NAME,

    pretrained=True,

    optimizer="auto",

    lr0=0.001,

    # cache=True,
    cache=False,

    amp=True,

    cos_lr=True,

    mosaic=0.5,

    close_mosaic=20,

    degrees=0,

    translate=0.05,

    scale=0.30,

    shear=0,

    perspective=0,

    fliplr=0.5,

    flipud=0,

    hsv_h=0.015,

    hsv_s=0.7,

    hsv_v=0.4,

    save=True,

    save_period=10,

    plots=True,

    verbose=True
)

print("\nTraining Finished!")

print("Best model:")
print(RUNS_DIR / RUN_NAME / "weights" / "best.pt")
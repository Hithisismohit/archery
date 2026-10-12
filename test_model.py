print("YOLO11 MODEL TEST STARTED")

from pathlib import Path
from ultralytics import YOLO

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "runs" / "archery_box_model" / "weights" / "best.pt"
TEST_IMAGES = BASE_DIR / "Test_data"

# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(str(MODEL_PATH))

# ============================================================
# RUN INFERENCE
# ============================================================

model.predict(
    source=str(TEST_IMAGES),
    imgsz=1280,
    conf=0.25,
    iou=0.50,
    device=0,
    save=True,
    save_txt=True,
    save_conf=True,
    project=str(BASE_DIR),      # Save in main folder
    name="test_results",        # Folder name
    exist_ok=True,
    verbose=True
)

print("\n===================================")
print("TEST COMPLETED SUCCESSFULLY")
print("Results Saved To:")
print(BASE_DIR / "test_results")
print("===================================")
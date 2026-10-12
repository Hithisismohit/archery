from pathlib import Path
import json
import random
import shutil
from collections import defaultdict
from tqdm import tqdm

# ======================================================
# CONFIG
# ======================================================

SOURCE = Path(r"P:\archery\archery_box_model_3\archery_data_sets_box")
OUTPUT = Path(r"P:\archery\archery_box_model_3\archery_yolo_dataset")

REPORT = OUTPUT / "validation_report"

TRAIN_RATIO = 0.80
SEED = 42

CLASS_MAP = {
    "target": 0,
    "tip": 1
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

# ======================================================
# CREATE OUTPUT FOLDERS
# ======================================================

for split in ["train", "val"]:

    (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)

REPORT.mkdir(parents=True, exist_ok=True)

# ======================================================
# REPORT STORAGE
# ======================================================

reports = defaultdict(list)

summary = {
    "images": 0,
    "json": 0,
    "valid": 0,
    "converted": 0,
    "missing_json": 0,
    "missing_image": 0,
    "corrupted_json": 0,
    "invalid_json": 0,
    "empty_json": 0,
    "empty_annotation": 0,
    "invalid_label": 0,
    "invalid_coordinate": 0,
    "duplicate": 0,
    "skipped": 0,
}

# ======================================================
# SCAN DATASET
# ======================================================

print("=" * 60)
print("Scanning Dataset...")
print("=" * 60)

image_files = {}
json_files = {}

duplicate_images = []
duplicate_json = []

# -------------------------------
# Scan Images
# -------------------------------

for ext in IMAGE_EXTENSIONS:

    for img in SOURCE.glob(f"*{ext}"):

        stem = img.stem.lower()

        if stem in image_files:
            duplicate_images.append(img.name)
        else:
            image_files[stem] = img

# -------------------------------
# Scan JSON
# -------------------------------

for js in SOURCE.glob("*.json"):

    stem = js.stem.lower()

    if stem in json_files:
        duplicate_json.append(js.name)
    else:
        json_files[stem] = js

summary["images"] = len(image_files)
summary["json"] = len(json_files)

# ======================================================
# SAVE DUPLICATES
# ======================================================

if duplicate_images:

    summary["duplicate"] += len(duplicate_images)

    reports["duplicate_files"].extend(
        ["IMAGE : " + x for x in duplicate_images]
    )

if duplicate_json:

    summary["duplicate"] += len(duplicate_json)

    reports["duplicate_files"].extend(
        ["JSON : " + x for x in duplicate_json]
    )

# ======================================================
# FIND MATCHING FILES
# ======================================================

valid_pairs = []

print("Checking Image / JSON pairs...")

all_names = sorted(
    set(image_files.keys()) | set(json_files.keys())
)

for name in tqdm(all_names):

    img = image_files.get(name)
    js = json_files.get(name)

    if img is None:

        summary["missing_image"] += 1
        summary["skipped"] += 1

        reports["missing_images"].append(js.name)

        continue

    if js is None:

        summary["missing_json"] += 1
        summary["skipped"] += 1

        reports["missing_json"].append(img.name)

        continue

    valid_pairs.append((img, js))

print()
print("=" * 60)
print("Scan Completed")
print("=" * 60)

print(f"Images Found : {summary['images']}")
print(f"JSON Found   : {summary['json']}")
print(f"Matched Pair : {len(valid_pairs)}")
print()
# ======================================================
# VALIDATION FUNCTIONS
# ======================================================

def validate_json(img_path: Path, json_path: Path):
    """
    Validate one Image + JSON pair.

    Returns
    -------
    (True, data)
        if JSON is valid

    (False, reason)
        if validation failed
    """

    # --------------------------------------------------
    # Load JSON
    # --------------------------------------------------

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    except json.JSONDecodeError:

        summary["corrupted_json"] += 1
        summary["skipped"] += 1
        reports["corrupted_json"].append(json_path.name)

        reports["skipped_files"].append(
    f"{json_path.name} -> Corrupted JSON"
)

        return False, "Corrupted JSON"

    except Exception as e:

        summary["invalid_json"] += 1
        summary["skipped"] += 1

        reports["invalid_json"].append(
            f"{json_path.name} : {e}"
        )

        return False, "Invalid JSON"

    # --------------------------------------------------
    # Required Fields
    # --------------------------------------------------

    if "imageWidth" not in data:

        summary["invalid_json"] += 1
        summary["skipped"] += 1

        reports["invalid_json"].append(
            f"{json_path.name} : Missing imageWidth"
        )

        return False, "Missing imageWidth"

    if "imageHeight" not in data:

        summary["invalid_json"] += 1
        summary["skipped"] += 1

        reports["invalid_json"].append(
            f"{json_path.name} : Missing imageHeight"
        )

        return False, "Missing imageHeight"

    w = data["imageWidth"]
    h = data["imageHeight"]

    if w <= 0 or h <= 0:

        summary["invalid_json"] += 1
        summary["skipped"] += 1

        reports["invalid_json"].append(
            f"{json_path.name} : Invalid image size"
        )

        return False, "Invalid image size"

    # --------------------------------------------------
    # Shapes
    # --------------------------------------------------

    shapes = data.get("shapes", [])

    if len(shapes) == 0:

        summary["empty_annotation"] += 1
        summary["skipped"] += 1

        reports["empty_annotations"].append(
            json_path.name
        )

        return False, "No annotations"

    valid_shape_found = False

    # --------------------------------------------------
    # Validate every shape
    # --------------------------------------------------

    for shape in shapes:

        label = shape.get("label")

        if label not in CLASS_MAP:

            summary["invalid_label"] += 1

            reports["invalid_labels"].append(
                f"{json_path.name} -> {label}"
            )

            continue

        points = shape.get("points", [])

        if len(points) != 2:

            summary["invalid_coordinate"] += 1

            reports["invalid_coordinates"].append(
                f"{json_path.name} -> Invalid point count"
            )

            continue

        try:

            x1, y1 = points[0]
            x2, y2 = points[1]

        except Exception:

            summary["invalid_coordinate"] += 1

            reports["invalid_coordinates"].append(
                f"{json_path.name} -> Cannot read points"
            )

            continue

        # ------------------------------------------
        # Coordinate Check
        # ------------------------------------------

        if not (
            0 <= x1 <= w and
            0 <= x2 <= w and
            0 <= y1 <= h and
            0 <= y2 <= h
        ):

            summary["invalid_coordinate"] += 1

            reports["invalid_coordinates"].append(
                f"{json_path.name} -> Outside image"
            )

            continue

        xmin = min(x1, x2)
        ymin = min(y1, y2)

        xmax = max(x1, x2)
        ymax = max(y1, y2)

        bw = xmax - xmin
        bh = ymax - ymin

        if bw <= 0 or bh <= 0:

            summary["invalid_coordinate"] += 1

            reports["invalid_coordinates"].append(
                f"{json_path.name} -> Zero-size box"
            )

            continue

        valid_shape_found = True

    # --------------------------------------------------
    # Final Decision
    # --------------------------------------------------

    if not valid_shape_found:

        summary["empty_annotation"] += 1
        summary["skipped"] += 1

        reports["empty_annotations"].append(
            json_path.name
        )

        return False, "No valid annotation"

    summary["valid"] += 1

    return True, data


# ======================================================
# VALIDATE COMPLETE DATASET
# ======================================================

print("=" * 60)
print("Validating Dataset...")
print("=" * 60)

validated_pairs = []

for img_path, json_path in tqdm(valid_pairs):

    ok, result = validate_json(img_path, json_path)

    if ok:
        validated_pairs.append(
            (
                img_path,
                json_path,
                result
            )
        )

print()
print("=" * 60)
print("Validation Finished")
print("=" * 60)
print(f"Valid Files : {len(validated_pairs)}")
print()
# ======================================================
# TRAIN / VAL SPLIT
# ======================================================

random.seed(SEED)
random.shuffle(validated_pairs)

split_index = int(len(validated_pairs) * TRAIN_RATIO)

train_pairs = validated_pairs[:split_index]
val_pairs = validated_pairs[split_index:]

print("=" * 60)
print("Dataset Split")
print("=" * 60)

print(f"Train : {len(train_pairs)}")
print(f"Val   : {len(val_pairs)}")
print()

# ======================================================
# CONVERT TO YOLO
# ======================================================

def convert(dataset, split):

    print(f"\nConverting {split}...")

    for img_path, json_path, data in tqdm(dataset):

        w = data["imageWidth"]
        h = data["imageHeight"]

        labels = []

        for shape in data["shapes"]:

            label = shape.get("label")

            if label not in CLASS_MAP:
                continue

            points = shape.get("points", [])

            if len(points) != 2:
                continue

            x1, y1 = points[0]
            x2, y2 = points[1]

            xmin = min(x1, x2)
            ymin = min(y1, y2)

            xmax = max(x1, x2)
            ymax = max(y1, y2)

            # ------------------------------
            # Normalize
            # ------------------------------

            xc = ((xmin + xmax) / 2) / w
            yc = ((ymin + ymax) / 2) / h

            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h

            labels.append(
                f"{CLASS_MAP[label]} "
                f"{xc:.6f} "
                f"{yc:.6f} "
                f"{bw:.6f} "
                f"{bh:.6f}"
            )

        # ----------------------------------
        # Copy Image
        # ----------------------------------

        shutil.copy2(
            img_path,
            OUTPUT / "images" / split / img_path.name
        )

        # ----------------------------------
        # Save Label
        # ----------------------------------

        txt_path = (
            OUTPUT /
            "labels" /
            split /
            f"{img_path.stem}.txt"
        )

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(labels))

        summary["converted"] += 1

# ======================================================
# RUN CONVERSION
# ======================================================

print("=" * 60)
print("Creating YOLO Dataset")
print("=" * 60)

convert(train_pairs, "train")
convert(val_pairs, "val")

print()
print("YOLO Conversion Finished.")
print()
# ======================================================
# SAVE VALIDATION REPORTS
# ======================================================

print("=" * 60)
print("Saving Validation Reports...")
print("=" * 60)


def save_report(filename, lines):

    file_path = REPORT / filename

    with open(file_path, "w", encoding="utf-8") as f:

        if len(lines) == 0:
            f.write("No issues found.\n")
        else:
            for line in lines:
                f.write(str(line) + "\n")


# ------------------------------------------------------
# Save individual reports
# ------------------------------------------------------

save_report("missing_json.txt",
            reports["missing_json"])

save_report("missing_images.txt",
            reports["missing_images"])

save_report("corrupted_json.txt",
            reports["corrupted_json"])

save_report("invalid_json.txt",
            reports["invalid_json"])

save_report("empty_annotations.txt",
            reports["empty_annotations"])

save_report("invalid_labels.txt",
            reports["invalid_labels"])

save_report("invalid_coordinates.txt",
            reports["invalid_coordinates"])

save_report("duplicate_files.txt",
            reports["duplicate_files"])

save_report("skipped_files.txt",
            reports["skipped_files"])


# ======================================================
# SUMMARY REPORT
# ======================================================

summary_file = REPORT / "summary.txt"

with open(summary_file, "w", encoding="utf-8") as f:

    f.write("=" * 50 + "\n")
    f.write("DATASET VALIDATION SUMMARY\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Images Found           : {summary['images']}\n")
    f.write(f"JSON Found             : {summary['json']}\n")
    f.write(f"Valid Files            : {summary['valid']}\n")
    f.write(f"Converted Files        : {summary['converted']}\n")
    f.write("\n")

    f.write(f"Missing JSON           : {summary['missing_json']}\n")
    f.write(f"Missing Images         : {summary['missing_image']}\n")
    f.write(f"Corrupted JSON         : {summary['corrupted_json']}\n")
    f.write(f"Invalid JSON           : {summary['invalid_json']}\n")
    f.write(f"Empty Annotation       : {summary['empty_annotation']}\n")
    f.write(f"Invalid Labels         : {summary['invalid_label']}\n")
    f.write(f"Invalid Coordinates    : {summary['invalid_coordinate']}\n")
    f.write(f"Duplicate Files        : {summary['duplicate']}\n")
    f.write(f"Skipped Files          : {summary['skipped']}\n")


# ======================================================
# CREATE data.yaml
# ======================================================

yaml = f"""
path: {OUTPUT.as_posix()}

train: images/train
val: images/val

names:
"""

for name, idx in sorted(CLASS_MAP.items(), key=lambda x: x[1]):
    yaml += f"\n  {idx}: {name}"

with open(OUTPUT / "data.yaml", "w", encoding="utf-8") as f:
    f.write(yaml.strip())


# ======================================================
# FINAL TERMINAL SUMMARY
# ======================================================

print()
print("=" * 60)
print("DATASET VALIDATION SUMMARY")
print("=" * 60)

print(f"Images Found         : {summary['images']}")
print(f"JSON Found           : {summary['json']}")
print(f"Valid Files          : {summary['valid']}")
print(f"Converted Files      : {summary['converted']}")
print()

print(f"Missing JSON         : {summary['missing_json']}")
print(f"Missing Images       : {summary['missing_image']}")
print(f"Corrupted JSON       : {summary['corrupted_json']}")
print(f"Invalid JSON         : {summary['invalid_json']}")
print(f"Empty Annotation     : {summary['empty_annotation']}")
print(f"Invalid Labels       : {summary['invalid_label']}")
print(f"Invalid Coordinates  : {summary['invalid_coordinate']}")
print(f"Duplicate Files      : {summary['duplicate']}")
print(f"Skipped Files        : {summary['skipped']}")

print()
print("=" * 60)
print("YOLO DATASET CREATED SUCCESSFULLY")
print("=" * 60)

print(f"Dataset Location : {OUTPUT}")
print(f"Reports Location : {REPORT}")

print("=" * 60)
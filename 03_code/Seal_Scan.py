# ==============================================================================
# 1. Import packages
# ==============================================================================
from pathlib import Path
import os
import random
import shutil
import YOLO
from PIL import (Image)
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import pandas as pd
import yaml



# ==============================================================================
# 2. Set project root
# ==============================================================================
project_root = Path(__file__).resolve().parent.parent
print("Projektordner:", project_root)
print("Arbeitsverzeichnis:", os.getcwd())


# ==============================================================================
# 3. Split Dataset
# ==============================================================================
# Input directories
image_dir = project_root / "01_data_raw" / "01_Seal_images" / "images"
label_dir = project_root / "01_data_raw" / "01_Seal_images" / "labels"

# Output directory
output_dir = project_root / "02_data" / "01_Training_Data"

# Create directories
for split in ["train", "val", "test"]:
    (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

# Collect images
images = list(image_dir.glob("*.jpg"))
random.shuffle(images)
n = len(images)
train_end = int(0.7 * n)
val_end = int(0.9 * n)

splits = {
    "train": images[:train_end],
    "val": images[train_end:val_end],
    "test": images[val_end:]}

for split, files in splits.items():
    for img in files:
        label = label_dir / f"{img.stem}.txt"

        shutil.copy(img, output_dir / "images" / split / img.name)

        if label.exists():
            shutil.copy(
                label,
                output_dir / "labels" / split / label.name)

print("Allocation complete.")


# ==============================================================================
# 4. Tiling
# ==============================================================================
# Define splits
SPLITS = ["train", "val", "test"]

# Set Directories
input_root  = project_root / "02_data" / "01_Training_Data"
output_root = project_root / "02_data" / "02_Training_Data_Til"

# Settings
TILE_SIZE        = 1024
OVERLAP_RATIO    = 0.25
MIN_VISIBILITY   = 0.2
SAVE_EMPTY_TILES = False

# Process
## HELPERS
def load_yolo_labels(path):
    labels = []
    if not path.exists():
        return labels

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            c, xc, yc, w, h = map(float, parts)
            labels.append((int(c), xc, yc, w, h))
    return labels


def yolo_to_xyxy(c, xc, yc, w, h, img_w, img_h):
    xc *= img_w
    yc *= img_h
    w *= img_w
    h *= img_h

    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2

    return c, (x1, y1, x2, y2)


def clip_box(box, tile):
    x1, y1, x2, y2 = box
    tx1, ty1, tx2, ty2 = tile

    nx1 = max(x1, tx1)
    ny1 = max(y1, ty1)
    nx2 = min(x2, tx2)
    ny2 = min(y2, ty2)

    if nx1 >= nx2 or ny1 >= ny2:
        return None

    return nx1, ny1, nx2, ny2


def area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def xyxy_to_yolo(box, tile_size):
    x1, y1, x2, y2 = box

    w = x2 - x1
    h = y2 - y1
    xc = x1 + w / 2
    yc = y1 + h / 2

    return xc / tile_size, yc / tile_size, w / tile_size, h / tile_size


def tile_split(image_dir, label_dir, out_image_dir, out_label_dir):
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(image_dir.glob("*.jpg"))
    stride = int(TILE_SIZE * (1 - OVERLAP_RATIO))

    print(f"Images found: {len(images)}")

    n_tiles = 0
    for img_path in images:

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        labels = load_yolo_labels(label_dir / f"{img_path.stem}.txt")

        # YOLO -> absolute boxes
        abs_boxes = [yolo_to_xyxy(c, xc, yc, w, h, img_w, img_h)
                     for c, xc, yc, w, h in labels]

        for y in range(0, img_h, stride):
            for x in range(0, img_w, stride):

                x2 = min(x + TILE_SIZE, img_w)
                y2 = min(y + TILE_SIZE, img_h)
                tile_box = (x, y, x2, y2)

                ### 1. LABEL PROCESSING
                new_labels = []
                for c, box in abs_boxes:

                    box_area = area(box)
                    if box_area <= 0:
                        continue

                    clipped = clip_box(box, tile_box)
                    if clipped is None:
                        continue

                    if area(clipped) / box_area < MIN_VISIBILITY:
                        continue

                    # shift into tile coordinates
                    tx1, ty1, tx2, ty2 = clipped
                    yolo_box = xyxy_to_yolo(
                        (tx1 - x, ty1 - y, tx2 - x, ty2 - y), TILE_SIZE
                    )
                    new_labels.append((c, *yolo_box))

                if not new_labels and not SAVE_EMPTY_TILES:
                    continue

                ### 2. RAW TILE (mit Padding am Rand)
                raw_tile = img.crop(tile_box)
                if raw_tile.size != (TILE_SIZE, TILE_SIZE):
                    padded = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
                    padded.paste(raw_tile, (0, 0))
                    raw_tile = padded

                ### 3. SAVE TRAINING DATA
                name = f"{img_path.stem}_x{x}_y{y}"
                raw_tile.save(out_image_dir / f"{name}.jpg")

                with open(out_label_dir / f"{name}.txt", "w") as f:
                    for l in new_labels:
                        f.write(f"{l[0]} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")

                n_tiles += 1

    print(f"Tiles saved: {n_tiles}")


## Main process (Loop über Splits)
for split in SPLITS:
    image_dir     = input_root  / "images" / split
    label_dir     = input_root  / "labels" / split
    out_image_dir = output_root / "images" / split
    out_label_dir = output_root / "labels" / split

    print("\n" + "=" * 60)
    print(f"TILING | SPLIT: {split}")
    print("=" * 60)

    if not image_dir.exists():
        print(f"[SKIP] {image_dir} not existing.")
        continue

    tile_split(image_dir, label_dir, out_image_dir, out_label_dir)

print("\nClean tiling finished.")


# ==============================================================================
# 5. Delete fragments
# ==============================================================================
# Define splits
SPLITS = ["train", "val", "test"]

# Set Directories
input_root  = project_root / "02_data" / "02_Training_Data_Til"
backup_root = output_root / "label_fragment_backup"

# Settings
MIN_WIDTH  = 0.01    # 1% Tile-width
MIN_HEIGHT = 0.01    # 1% Tile-height
MAX_ASPECT = 10      # aspect (width/height or height/width)
DRY_RUN    = False

# Process
## HELPERS
def is_bad_box(parts):
    _, xc, yc, w, h = parts
    w, h = float(w), float(h)

    if w <= 0 or h <= 0:
        return True

    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return True

    aspect = w / h

    if aspect > MAX_ASPECT or aspect < (1 / MAX_ASPECT):
        return True

    return False


def clean_fragments(label_dir, backup_dir):
    # Backup erstellen
    if not backup_dir.exists():
        shutil.copytree(label_dir, backup_dir)
        print(f"Backup erstellt: {backup_dir}")
    else:
        print(f"Backup existiert bereits: {backup_dir}")

    removed_total = 0
    files_fixed = 0

    for label_file in label_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()

        new_lines = []
        removed = 0

        for line in lines:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            if is_bad_box(parts):
                removed += 1
            else:
                new_lines.append(line)

        if removed > 0:
            files_fixed += 1
            removed_total += removed

            print(f"[FIX] {label_file.name}: removed {removed} boxes")

            if not DRY_RUN:
                with open(label_file, "w") as f:
                    f.writelines(new_lines)

    print(f"Files modified: {files_fixed} | Boxes removed: {removed_total}")
    return files_fixed, removed_total


## Main process (Loop for splits)
total_files, total_boxes = 0, 0

for split in SPLITS:
    label_dir  = input_root / "labels" / split
    backup_dir = backup_root / split

    print("\n" + "=" * 60)
    print(f"FRAGMENT CLEANUP | SPLIT: {split}")
    print("=" * 60)

    if not label_dir.exists():
        print(f"[SKIP] {label_dir} not existing.")
        continue

    files, boxes = clean_fragments(label_dir, backup_dir)
    total_files += files
    total_boxes += boxes

print("\n====================")
print(f"Files modified (gesamt): {total_files}")
print(f"Boxes removed (gesamt):  {total_boxes}")
print("====================")


# ==============================================================================
# 6. Create Yaml
# ==============================================================================
# Set directory
dataset_dir = project_root / "02_data" / "02_Training_Data_Til"

# Create Yaml
data = {
    "path": str(dataset_dir.resolve()),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": 3,
    "names": ["Kegelrobbe_adult",
              "Kegelrobbe_juvenil",
              "Seehund"]}


yaml_path = dataset_dir / "SealScan.yaml"

with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(data, f, sort_keys=False, allow_unicode=True)

print(f"YAML-Datei erstellt: {yaml_path}")


# ==============================================================================
# 7. Train & test Yolo model
# ==============================================================================
# Training
def train_model(project_root):
    yaml_path = project_root / "02_data" / "02_Training_Data_Til" / "SealScan.yaml"
    output_dir = project_root / "04_results" / "SealScan_Model"

    print("YAML:", yaml_path)

    model = YOLO("yolov8m.pt")

    results = model.train(
        data = str(yaml_path),
        epochs = 150,
        imgsz = 1024,
        patience = 50,
        batch = 20,
        degrees = 5,
        translate = 0.05,
        scale = 0.1,
        fliplr = 0.5,
        flipud = 0.5,
        hsv_h = 0.01,
        hsv_s = 0.2,
        hsv_v = 0.2,
        mosaic = 1.0,
        mixup = 0.0,
        copy_paste = 0.0,
        shear = 0.0,
        perspective = 0.0,
        cos_lr = True,
        device = 0,
        project = str(output_dir),
        name = "SealScan_Yolo",)

# Test
def test_model(project_root, model_path):
    model = YOLO(str(model_path))

    metrics = model.val(
        data = str(project_root / "02_data" / "02_Training_Data_Til" / "SealScan.yaml"),
        split = "test",)
    print(metrics)

# Process training and test
def main():
    # Pfad zum trainierten Modell
    model_path = (
        project_root / "04_results" / "SealScan_Model" / "SealScan_Yolo" / "weights" / "best.pt")

    # Training
    train_model(project_root)

    # test if best.pt is existing
    print("ModelPath:", model_path)
    print("Exists:", model_path.exists())
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Test model
    test_model(project_root, model_path)

if __name__ == "__main__":
    main()


# ==============================================================================
# 8. Prediction with SAHI
# ==============================================================================
# Set directories
results_dir = project_root / "04_results" / "Sahi"
results_dir.mkdir(exist_ok=True)
pred_img_dir = results_dir / "Sahi_img"
pred_img_dir.mkdir(parents=True, exist_ok=True)
image_dir = project_root / "02_Training_Data_Til" / "Images" / "test"

# Model
model = AutoDetectionModel.from_pretrained(
    model_type = "ultralytics",
    model_path = project_root / "04_results" / "SealScan_Model" / "SealScan_Yolo" / "weights" / "best.pt",
    confidence_threshold = 0.6,
    device = "0")

# List for all detections
detections = []

# Process
for image in image_dir.glob("*.jpg"):

    result = get_sliced_prediction(
        str(image),
        model,
        slice_height = 1024,
        slice_width = 1024,
        overlap_height_ratio = 0.15,
        overlap_width_ratio = 0.15,
        postprocess_type = "NMS",
        postprocess_match_metric = "IOS",
        postprocess_match_threshold = 0.4,
        postprocess_class_agnostic = True
    )

    # All detected objects
    for pred in result.object_prediction_list:

        bbox = pred.bbox

        detections.append({
            "image": image.name,
            "class_name": pred.category.name,
            "confidence": pred.score.value,
            "xmin": bbox.minx,
            "ymin": bbox.miny,
            "xmax": bbox.maxx,
            "ymax": bbox.maxy
        })

    # Save images

    result.export_visuals(
        export_dir = str(pred_img_dir),
        file_name = image.stem,
        hide_conf = True,
        hide_labels = False,
        rect_th = 2)

    # DataFrame with all detections
    df = pd.DataFrame(detections)

# Save csv
df.to_csv(
    results_dir / "Seal_detections.csv",
    index = False
)

# Statistik
print(f"Number of detected Seals: {len(df)}")
print(df["class_name"].value_counts())
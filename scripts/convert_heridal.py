"""
Converts HERIDAL's Roboflow-exported annotations (TensorFlow Object
Detection CSV format: filename,width,height,class,xmin,ymin,xmax,ymax)
to YOLO .txt format.

This specific mirror (Kaggle: imadeddinelassakeur/heridal, originally
from Roboflow "Heridal - v2 yolo_non_412") ships ALL 1,546 real HERIDAL
images in a single "train" folder with one shared _annotations.csv --
despite the folder's name, it's TF Object Detection CSV format, not
per-image XML or native YOLO .txt.

Since this mirror has no pre-made valid/test split, this script creates
one -- holding out the last N images (sorted) as validation, matching
roughly the proportion HERIDAL's own official split uses.

Usage:
    python scripts/convert_heridal.py \
        --input data/raw/heridal_kaggle \
        --output data/heridal
"""
import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

CLASS_MAP = {"human": 0}  # Note: this mirror uses "human", not "person"


def convert_box(img_w, img_h, xmin, ymin, xmax, ymax):
    """Convert pixel (xmin, ymin, xmax, ymax) to YOLO (x_center, y_center, w, h), normalized."""
    dw, dh = 1.0 / img_w, 1.0 / img_h
    x_center = (xmin + xmax) / 2.0 * dw
    y_center = (ymin + ymax) / 2.0 * dh
    w = (xmax - xmin) * dw
    h = (ymax - ymin) * dh
    return x_center, y_center, w, h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/heridal_kaggle")
    parser.add_argument("--output", type=str, default="data/heridal")
    parser.add_argument("--val-ratio", type=float, default=0.065,
                         help="Fraction held out for validation (~0.065 = ~101/1546, "
                              "matching HERIDAL's own official test-split size)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    img_dir = input_dir / "train"
    csv_path = img_dir / "_annotations.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected {csv_path} to exist. Check the actual folder structure with:\n"
            f"  find {input_dir} -maxdepth 2"
        )

    for split in ["train", "val"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Group all box rows by filename first (a file can have multiple boxes)
    boxes_by_file = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls_name = row["class"].strip().lower()
            if cls_name not in CLASS_MAP:
                continue
            boxes_by_file[row["filename"]].append({
                "width": int(row["width"]),
                "height": int(row["height"]),
                "xmin": float(row["xmin"]),
                "ymin": float(row["ymin"]),
                "xmax": float(row["xmax"]),
                "ymax": float(row["ymax"]),
                "class_id": CLASS_MAP[cls_name],
            })

    filenames_with_boxes = sorted(boxes_by_file.keys())
    if not filenames_with_boxes:
        raise ValueError(f"No annotated rows found in {csv_path} -- check the CSV contents.")

    # Include ALL images in the folder, not just ones with annotations --
    # images absent from the CSV are background/negative (no person present)
    # and get an empty label file. These are valuable: tile_dataset.py's
    # --keep-empty-ratio already controls the final background ratio at
    # tiling time, so we don't filter them out here.
    all_image_files = sorted(p.name for p in img_dir.glob("*.jpg"))
    background_files = [f for f in all_image_files if f not in boxes_by_file]

    filenames = all_image_files
    n_val = max(1, int(len(filenames) * args.val_ratio))
    val_files = set(filenames[-n_val:])

    print(f"Found {len(filenames_with_boxes)} images with person annotations, "
          f"{len(background_files)} background (no-person) images -- "
          f"including all {len(filenames)} in the converted dataset.")

    converted, missing_images = 0, 0
    for filename in tqdm(filenames, desc="Converting HERIDAL (CSV)"):
        img_path = img_dir / filename
        if not img_path.exists():
            missing_images += 1
            continue

        split = "val" if filename in val_files else "train"
        out_img = output_dir / split / "images" / filename
        out_label = output_dir / split / "labels" / (Path(filename).stem + ".txt")

        shutil.copy(img_path, out_img)

        lines = []
        for box in boxes_by_file[filename]:
            x, y, w, h = convert_box(
                box["width"], box["height"],
                box["xmin"], box["ymin"], box["xmax"], box["ymax"],
            )
            lines.append(f"{box['class_id']} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        out_label.write_text("\n".join(lines))
        converted += 1

    print(f"\nConverted: {converted} images. Missing image files: {missing_images}")
    print(f"Train: {converted - len(val_files)} | Val: {len(val_files)}")
    print(f"Output written to: {output_dir}")
    print("\nNote: this mirror has no separate held-out test set from HERIDAL's own "
          "official split -- this val split was created by holding out the last "
          f"{n_val} images (sorted alphabetically), not HERIDAL's original 101-image "
          "test set. Fine for training/validation purposes, just don't cite this as "
          "matching published HERIDAL benchmark numbers exactly.")


if __name__ == "__main__":
    main()

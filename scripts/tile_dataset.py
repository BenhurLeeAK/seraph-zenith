"""
Slices training images into overlapping tiles (SAHI-style "slicing-aided
fine-tuning"), so the model trains on the same scale it will see at
tiled inference time on the drone.

Research basis: SAHI's slicing-aided fine-tuning produced cumulative
AP gains of +12.7/+13.4/+14.5 (vs +5-7 for inference-only slicing) on
VisDrone/xView (Akyon et al., ICIP 2022). This matters most for your
setup because HERIDAL's average person bbox is only ~64x64px (sigma~23) --
naive full-frame downscaling to 640x640 shrinks distant/small targets
below detectable size.

Uses the `sahi` library's slicing utilities directly on YOLO-format data.

Usage:
    python scripts/tile_dataset.py \
        --input data/processed/merged \
        --output data/processed/tiled \
        --tile-size 640 \
        --overlap 0.2
"""
import argparse
from pathlib import Path

import cv2
from sahi.slicing import slice_image
from tqdm import tqdm


def load_yolo_labels(label_path, img_w, img_h):
    """Load YOLO-format labels and convert to absolute pixel (xmin, ymin, xmax, ymax, class_id)."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id, xc, yc, w, h = parts
        cls_id = int(cls_id)
        xc, yc, w, h = float(xc) * img_w, float(yc) * img_h, float(w) * img_w, float(h) * img_h
        xmin, ymin = xc - w / 2, yc - h / 2
        xmax, ymax = xc + w / 2, yc + h / 2
        boxes.append((xmin, ymin, xmax, ymax, cls_id))
    return boxes


def box_in_tile(box, tile_xmin, tile_ymin, tile_xmax, tile_ymax, min_visibility=0.3):
    """Check if a box overlaps a tile enough to keep it, and clip it to tile bounds."""
    xmin, ymin, xmax, ymax, cls_id = box
    ix_min = max(xmin, tile_xmin)
    iy_min = max(ymin, tile_ymin)
    ix_max = min(xmax, tile_xmax)
    iy_max = min(ymax, tile_ymax)

    if ix_min >= ix_max or iy_min >= iy_max:
        return None

    orig_area = (xmax - xmin) * (ymax - ymin)
    inter_area = (ix_max - ix_min) * (iy_max - iy_min)
    if orig_area <= 0 or (inter_area / orig_area) < min_visibility:
        return None

    # Convert to tile-relative coords
    rel_xmin = ix_min - tile_xmin
    rel_ymin = iy_min - tile_ymin
    rel_xmax = ix_max - tile_xmin
    rel_ymax = iy_max - tile_ymin
    return rel_xmin, rel_ymin, rel_xmax, rel_ymax, cls_id


def process_split(input_dir, output_dir, tile_size, overlap, keep_empty_ratio):
    img_dir = input_dir / "images"
    label_dir = input_dir / "labels"
    out_img_dir = output_dir / "images"
    out_label_dir = output_dir / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    empty_tiles_kept = 0
    positive_tiles = 0

    image_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

    for img_path in tqdm(image_files, desc=f"Tiling {input_dir.name}"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        img_h, img_w = image.shape[:2]
        label_path = label_dir / (img_path.stem + ".txt")
        boxes = load_yolo_labels(label_path, img_w, img_h)

        slice_result = slice_image(
            image=image,
            slice_height=tile_size,
            slice_width=tile_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
        )

        for i, sliced_image in enumerate(slice_result.sliced_image_list):
            tile_xmin, tile_ymin = sliced_image.starting_pixel
            tile_xmax, tile_ymax = tile_xmin + tile_size, tile_ymin + tile_size

            tile_boxes = []
            for box in boxes:
                clipped = box_in_tile(box, tile_xmin, tile_ymin, tile_xmax, tile_ymax)
                if clipped:
                    tile_boxes.append(clipped)

            # Keep only a fraction of empty (no-person) tiles to control the
            # background ratio (research suggests ~10-20% background images
            # helps false-positive rate without drowning out positives)
            if not tile_boxes:
                if empty_tiles_kept >= positive_tiles * keep_empty_ratio:
                    continue
                empty_tiles_kept += 1
            else:
                positive_tiles += 1

            tile_name = f"{img_path.stem}_tile{i}"
            tile_img = sliced_image.image
            actual_h, actual_w = tile_img.shape[:2]

            cv2.imwrite(str(out_img_dir / f"{tile_name}.jpg"), tile_img)

            lines = []
            for xmin, ymin, xmax, ymax, cls_id in tile_boxes:
                xc = ((xmin + xmax) / 2) / actual_w
                yc = ((ymin + ymax) / 2) / actual_h
                w = (xmax - xmin) / actual_w
                h = (ymax - ymin) / actual_h
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            (out_label_dir / f"{tile_name}.txt").write_text("\n".join(lines))

    print(f"{input_dir.name}: {positive_tiles} positive tiles, {empty_tiles_kept} background tiles")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/merged")
    parser.add_argument("--output", type=str, default="data/processed/tiled")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--keep-empty-ratio", type=float, default=0.15,
                         help="Background tiles kept, as a ratio of positive tiles (~0.15 = 15%%)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    for split in ["train", "val"]:
        process_split(
            input_dir / split, output_dir / split,
            args.tile_size, args.overlap, args.keep_empty_ratio,
        )

    print(f"\nTiled dataset written to: {output_dir}")


if __name__ == "__main__":
    main()

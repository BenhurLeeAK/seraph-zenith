#!/usr/bin/env python3
"""
validate_by_distance.py

Per-distance (3m/5m/7m) validation for the SAR human-detection close-range
fine-tune. Runs tiled inference against the 45 labeled supplement images
with two checkpoints (baseline vs. fine-tuned candidate) and reports
detection rate, confidence, and false positives per distance bucket, plus
a delta table so a regression at one distance can't hide behind an
aggregate win.

Tiling logic (grid + padding) is copied directly from infer_tiled.py to
match your actual deployed inference path: fixed-step tiles, padding up
to tile_size if the image is smaller, and -- importantly -- NO cross-tile
merging or NMS. infer_tiled.py doesn't reconcile duplicate detections in
tile overlap regions on-device, so this script doesn't either; duplicate
detections there will show up as extra FPs here, which is an honest
reflection of current deployed behavior, not a bug in the eval.

Caveat: this runs the .pt checkpoint via ultralytics on your laptop GPU,
not the compiled .hef on actual Hailo hardware. Quantization (.pt -> .onnx
-> .hef) can shift confidence scores somewhat, so treat this as a strong
proxy for go/no-go on the fine-tune, not a substitute for a final on-device
check with infer_tiled.py once/if you compile the candidate checkpoint.

Assumes filenames like: supplement_3m_20260921_094310_001_jpg.rf.<hash>.jpg
matched against a same-stem .txt label file in YOLO format (class cx cy w h,
normalized). Single class ("human"), so the class id is ignored.

Usage:
    python validate_by_distance.py \
        --baseline runs/detect/models/sar_stage2-2/weights/best.pt \
        --finetuned runs/detect/models/finetune_gentle-X/weights/best.pt \
        --images supplement_labeled/train/images \
        --labels supplement_labeled/train/labels \
        --tile-size 640 --overlap 0.2 --conf 0.3 --iou-thresh 0.5
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

DISTANCE_RE = re.compile(r"supplement_(\d+m)_")


def get_distance_bucket(filename: str) -> str:
    m = DISTANCE_RE.search(filename)
    return m.group(1) if m else "unknown"


def get_tiles(img_w, img_h, tile_size, overlap):
    """Identical to infer_tiled.py's tiling grid -- keep these in sync."""
    step = int(tile_size * (1 - overlap))
    xs = list(range(0, max(img_w - tile_size, 0) + 1, step))
    ys = list(range(0, max(img_h - tile_size, 0) + 1, step))
    if not xs or xs[-1] + tile_size < img_w:
        xs.append(max(img_w - tile_size, 0))
    if not ys or ys[-1] + tile_size < img_h:
        ys.append(max(img_h - tile_size, 0))
    return [(x, y) for y in ys for x in xs]


def load_yolo_labels(label_path: Path, img_w: int, img_h: int):
    """Ground-truth boxes as (x1, y1, x2, y2) in pixel coords, on the
    ORIGINAL (unpadded) image size."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        _, cx, cy, w, h = parts[:5]
        cx, cy, w, h = float(cx) * img_w, float(cy) * img_h, float(w) * img_w, float(h) * img_h
        boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate_checkpoint(weights_path, images_dir, labels_dir, tile_size, overlap, conf, iou_thresh, dump_rows=None):
    from ultralytics import YOLO
    import cv2

    model = YOLO(str(weights_path))

    per_bucket = defaultdict(lambda: {
        "images": 0, "gt_boxes": 0, "tp": 0, "fn": 0, "fp": 0,
        "tp_confidences": [], "fp_confidences": [],
    })

    for img_path in sorted(Path(images_dir).glob("*.jpg")):
        bucket = get_distance_bucket(img_path.name)
        label_path = Path(labels_dir) / (img_path.stem + ".txt")

        full_img = cv2.imread(str(img_path))
        img_h, img_w = full_img.shape[:2]

        # GT boxes computed on the original, unpadded image size
        gt_boxes = load_yolo_labels(label_path, img_w, img_h)

        # Pad exactly like infer_tiled.py before tiling
        pad_w = max(tile_size - img_w, 0)
        pad_h = max(tile_size - img_h, 0)
        if pad_w or pad_h:
            full_img = cv2.copyMakeBorder(full_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            img_h, img_w = full_img.shape[:2]

        tiles = get_tiles(img_w, img_h, tile_size, overlap)

        # NO cross-tile merge -- matches infer_tiled.py's on-device behavior
        preds = []
        for tx, ty in tiles:
            tile = full_img[ty:ty + tile_size, tx:tx + tile_size]
            result = model.predict(tile, conf=conf, verbose=False)[0]
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                preds.append(((tx + x1, ty + y1, tx + x2, ty + y2), score))

        matched_gt = set()
        bucket_stats = per_bucket[bucket]
        bucket_stats["images"] += 1
        bucket_stats["gt_boxes"] += len(gt_boxes)

        for pred_box, score in sorted(preds, key=lambda p: -p[1]):
            best_iou, best_idx = 0.0, -1
            for idx, gt_box in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue
                cur_iou = iou(pred_box, gt_box)
                if cur_iou > best_iou:
                    best_iou, best_idx = cur_iou, idx
            is_tp = best_iou >= iou_thresh
            if is_tp:
                matched_gt.add(best_idx)
                bucket_stats["tp"] += 1
                bucket_stats["tp_confidences"].append(score)
            else:
                bucket_stats["fp"] += 1
                bucket_stats["fp_confidences"].append(score)

            if dump_rows is not None:
                dump_rows.append((img_path.name, bucket, round(score, 3), round(best_iou, 3), "TP" if is_tp else "FP"))

        n_missed = len(gt_boxes) - len(matched_gt)
        bucket_stats["fn"] += n_missed
        if dump_rows is not None and n_missed > 0:
            for idx in range(len(gt_boxes)):
                if idx not in matched_gt:
                    dump_rows.append((img_path.name, bucket, None, None, "FN"))

    return per_bucket


def summarize(label, per_bucket):
    print(f"\n=== {label} ===")
    for bucket in sorted(per_bucket.keys()):
        s = per_bucket[bucket]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        avg_tp_conf = sum(s["tp_confidences"]) / len(s["tp_confidences"]) if s["tp_confidences"] else 0.0
        print(
            f"  {bucket:>4}: images={s['images']:>2}  gt={s['gt_boxes']:>2}  "
            f"TP={tp:>2} FP={fp:>2} FN={fn:>2}  "
            f"precision={precision:.3f}  recall={recall:.3f}  avg_tp_conf={avg_tp_conf:.3f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Path to baseline checkpoint (best.pt, mAP50=0.854)")
    ap.add_argument("--finetuned", required=True, help="Path to the new fine-tuned checkpoint to compare")
    ap.add_argument("--images", default="supplement_labeled/train/images")
    ap.add_argument("--labels", default="supplement_labeled/train/labels")
    ap.add_argument("--tile-size", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.3, help="Matches infer_tiled.py's CONF_THRESHOLD default")
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--dump-csv", type=str, default=None,
                     help="Optional path to write a per-detection CSV (filename, bucket, conf, iou, TP/FP/FN)")
    args = ap.parse_args()

    baseline_rows = [] if args.dump_csv else None
    finetuned_rows = [] if args.dump_csv else None

    print(f"Evaluating baseline: {args.baseline}")
    baseline_stats = evaluate_checkpoint(
        args.baseline, args.images, args.labels, args.tile_size, args.overlap, args.conf, args.iou_thresh,
        dump_rows=baseline_rows,
    )
    summarize("BASELINE (sar_stage2-2)", baseline_stats)

    print(f"\nEvaluating fine-tuned: {args.finetuned}")
    finetuned_stats = evaluate_checkpoint(
        args.finetuned, args.images, args.labels, args.tile_size, args.overlap, args.conf, args.iou_thresh,
        dump_rows=finetuned_rows,
    )
    summarize("FINE-TUNED (candidate)", finetuned_stats)

    print("\n=== DELTA (fine-tuned - baseline) ===")
    for bucket in sorted(set(baseline_stats) | set(finetuned_stats)):
        b = baseline_stats.get(bucket, {"tp": 0, "fp": 0, "fn": 0, "tp_confidences": []})
        f = finetuned_stats.get(bucket, {"tp": 0, "fp": 0, "fn": 0, "tp_confidences": []})
        b_recall = b["tp"] / (b["tp"] + b["fn"]) if (b["tp"] + b["fn"]) else 0.0
        f_recall = f["tp"] / (f["tp"] + f["fn"]) if (f["tp"] + f["fn"]) else 0.0
        b_conf = sum(b["tp_confidences"]) / len(b["tp_confidences"]) if b["tp_confidences"] else 0.0
        f_conf = sum(f["tp_confidences"]) / len(f["tp_confidences"]) if f["tp_confidences"] else 0.0
        print(
            f"  {bucket:>4}: recall {b_recall:.3f} -> {f_recall:.3f} "
            f"({f_recall - b_recall:+.3f})   "
            f"avg_conf {b_conf:.3f} -> {f_conf:.3f} ({f_conf - b_conf:+.3f})   "
            f"FP {b['fp']} -> {f['fp']}"
        )

    if args.dump_csv:
        import csv
        with open(args.dump_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["checkpoint", "filename", "bucket", "confidence", "iou", "result"])
            for row in baseline_rows:
                writer.writerow(["baseline", *row])
            for row in finetuned_rows:
                writer.writerow(["finetuned", *row])
        print(f"\nPer-detection dump written to: {args.dump_csv}")


if __name__ == "__main__":
    main()

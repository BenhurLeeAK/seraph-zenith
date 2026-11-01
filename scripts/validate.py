"""
Evaluates a trained model against the held-out validation split and
reports mAP50, mAP50-95, precision, recall.

Target from research: mAP@50 >= 0.80 on held-out HERIDAL/SARD split
matches the best published civil-SAR YOLOv8 result (Ciccone & Ceruti,
Drones 2025). Use this as your benchmark before optimizing for speed.

Usage:
    python scripts/validate.py --weights models/sar_stage2/weights/best.pt
"""
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, default="configs/data_sar.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device, plots=True)

    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print("=" * 50)

    target_map50 = 0.80
    if metrics.box.map50 >= target_map50:
        print(f"\n✓ Meets research benchmark target (mAP50 >= {target_map50})")
    else:
        gap = target_map50 - metrics.box.map50
        print(f"\n✗ Below research benchmark target by {gap:.4f}")
        print("  Consider: more epochs, Stage 1 VisDrone pretraining if skipped,")
        print("  checking tile overlap/background ratio, or reviewing failure cases")
        print("  in the confusion matrix (saved alongside this run's plots).")


if __name__ == "__main__":
    main()

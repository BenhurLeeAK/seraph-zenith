"""
Stage 1: Pretrain YOLOv8s on VisDrone for general aerial small-object
detection capability, before specializing on SAR human detection in Stage 2.

Research basis: two-stage transfer (VisDrone -> HERIDAL) produced 0.802
mAP@50 on HERIDAL (Ciccone & Ceruti, Drones 2025), vs 0.781/0.788 for
direct fine-tuning without the VisDrone pretraining step.

This step is OPTIONAL if you're short on time -- direct fine-tuning on
HERIDAL+SARD alone still works, just with a smaller accuracy gain. VisDrone
auto-downloads via Ultralytics on first run (no manual dataset prep needed).

Usage:
    python scripts/train_stage1_visdrone.py --epochs 50
"""
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                         help="Starting checkpoint (COCO-pretrained YOLOv8s)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0",
                         help="GPU device index, or 'cpu'")
    parser.add_argument("--name", type=str, default="stage1_visdrone")
    args = parser.parse_args()

    model = YOLO(args.model)

    # 'VisDrone.yaml' is a built-in Ultralytics dataset config that
    # auto-downloads VisDrone2019-DET on first run.
    results = model.train(
        data="VisDrone.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project="models",
        patience=15,
        save=True,
        pretrained=True,
        optimizer="auto",
        # Standard augmentation is fine here -- this stage is about general
        # aerial small-object capability, not SAR-specific tuning yet.
    )

    print(f"\nStage 1 complete. Best weights: models/{args.name}/weights/best.pt")
    print("Use this checkpoint as the starting point for Stage 2:")
    print(f"  python scripts/train_stage2_sar.py --model models/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()

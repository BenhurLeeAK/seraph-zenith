"""
Stage 2: Fine-tune on the tiled HERIDAL + SARD dataset, using the
P2-head architecture and SAR-specific augmentation.

If you ran Stage 1 (VisDrone pretraining), pass its best.pt as --model.
If skipping Stage 1, this defaults to standard COCO-pretrained yolov8s.pt.

Usage:
    # With Stage 1 pretraining:
    python scripts/train_stage2_sar.py --model models/stage1_visdrone/weights/best.pt

    # Without Stage 1 (direct fine-tune):
    python scripts/train_stage2_sar.py --model yolov8s.pt
"""
import argparse

import yaml
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                         help="Starting checkpoint -- Stage 1 output, or yolov8s.pt for direct fine-tune")
    parser.add_argument("--use-p2-head", action="store_true",
                         help="Use the custom P2-head architecture (configs/yolov8s-p2.yaml). "
                              "Note: requires re-initializing detection head, so Stage 1 "
                              "transfer is at the backbone level only when this is set.")
    parser.add_argument("--data", type=str, default="configs/data_sar.yaml")
    parser.add_argument("--augment-cfg", type=str, default="configs/augment_sar.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--name", type=str, default="sar_stage2")
    args = parser.parse_args()

    if args.use_p2_head:
        # Build P2-head architecture, then load Stage 1 weights into matching layers
        model = YOLO("configs/yolov8s-p2.yaml")
        if args.model != "yolov8s.pt":
            model.load(args.model)  # transfers matching backbone weights
    else:
        model = YOLO(args.model)

    with open(args.augment_cfg) as f:
        aug_hyp = yaml.safe_load(f)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project="models",
        patience=20,
        save=True,
        optimizer="auto",
        cos_lr=True,
        **aug_hyp,
    )

    print(f"\nStage 2 complete. Best weights: models/{args.name}/weights/best.pt")
    print("Next: python scripts/validate.py --weights models/{}/weights/best.pt".format(args.name))


if __name__ == "__main__":
    main()

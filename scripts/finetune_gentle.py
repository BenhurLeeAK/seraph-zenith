"""
Gentle fine-tuning of an already-converged YOLOv8 checkpoint on a small
supplementary dataset, without regressing the original benchmark.

Implements the corrected recipe (see research summary):
- Explicit optimizer=SGD with a low lr0 (~1/10 of the original training's
  0.01), since Ultralytics' default optimizer="auto" silently ignores a
  manually-set lr0 for short runs.
- close_mosaic set proportionally to the (short) epoch count, instead of
  the default 10, so mosaic augmentation isn't disabled prematurely.
- Short warmup (1 epoch instead of the default 3) appropriate for a
  fine-tune rather than a from-scratch run.
- Trains on the MERGED dataset (original + new tiles together), which is
  the primary safeguard against regressing the original distribution --
  not just a low learning rate.

Usage:
    python scripts/finetune_gentle.py \
        --model runs/detect/models/sar_stage2-2/weights/best.pt \
        --epochs 35 \
        --lr0 0.001
"""
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                         help="Starting checkpoint -- the converged model to gently fine-tune")
    parser.add_argument("--data", type=str, default="configs/data_sar.yaml",
                         help="Dataset config -- should point at the MERGED tiled dataset "
                              "(original + new supplement tiles together)")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr0", type=float, default=0.001,
                         help="~1/10 of the original training's lr0 (0.01) -- avoids "
                              "perturbing the converged checkpoint too aggressively")
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--close-mosaic", type=int, default=4,
                         help="Set proportionally to --epochs so mosaic stays active for "
                              "~85-90%% of the run, matching the original training's proportion "
                              "(NOT the default 10, which on a short run disables mosaic too early)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--name", type=str, default="ft_gentle")
    parser.add_argument("--freeze", type=int, default=None,
                         help="Optionally freeze the first N layers (e.g. 10 for backbone). "
                              "Not required since merged data already guards against forgetting; "
                              "leave unset unless VRAM-constrained or overfitting is observed.")
    args = parser.parse_args()

    model = YOLO(args.model)

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project="models",
        save=True,
        # Explicit optimizer is required: default optimizer="auto" ignores
        # a manually-set lr0 for short runs, silently reverting to its own
        # schedule -- exactly what caused the earlier regression.
        optimizer="SGD",
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        close_mosaic=args.close_mosaic,
        cos_lr=True,
    )
    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze

    results = model.train(**train_kwargs)

    print(f"\nGentle fine-tune complete.")
    print(f"IMPORTANT: this is NOT automatically the new best model.")
    print(f"Run scripts/validate.py against the ORIGINAL benchmark (mAP50 >= 0.854) before adopting it:")
    print(f"  python scripts/validate.py --weights models/{args.name}/weights/best.pt")
    print(f"(check the actual saved folder name -- Ultralytics may append -2/-3 if it already exists)")


if __name__ == "__main__":
    main()

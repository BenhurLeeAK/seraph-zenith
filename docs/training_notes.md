# SIH26177 — Stranded Human Detection Pipeline

Trains and prepares a YOLOv8-based detector for finding stranded/trapped humans from
the Seraph Zenith Mark-2 drone (Arducam B0445 / IMX296, 1456×1088, no thermal/depth),
for deployment on Raspberry Pi 5 + Hailo-8 (AI HAT+).

This pipeline implements the recommended methodology from our research:
1. **Two-stage transfer learning**: pretrain on VisDrone (general aerial small-object),
   then fine-tune on HERIDAL + SARD (SAR-specific human detection).
2. **Slicing-aided training** (SAHI-style): train on overlapping tiles matching
   deployment-time tiled inference, not naive full-frame downscaling.
3. **YOLOv8s + P2 head**: small model for edge deployment headroom, with an added
   small-object detection layer for distant/tiny targets.
4. **SAR-specific augmentation**: heavy mosaic/copy-paste, pose diversity (prone,
   seated, occluded), no motion-blur augmentation (global shutter camera doesn't need it).
5. **INT8 quantization prep** for Hailo-8 HEF compilation.

## Project structure

```
sar-human-detection/
├── data/
│   ├── raw/              # Downloaded datasets land here first
│   ├── heridal/           # HERIDAL after extraction
│   ├── sard/               # SARD after extraction
│   └── processed/         # Final tiled, YOLO-format training data
├── scripts/
│   ├── download_datasets.sh    # Fetch HERIDAL + SARD
│   ├── convert_heridal.py      # HERIDAL (VOC XML) -> YOLO format
│   ├── tile_dataset.py         # SAHI slicing-aided tiling for training
│   ├── merge_datasets.py       # Combine HERIDAL + SARD into one YOLO dataset
│   ├── train_stage1_visdrone.py  # Pretrain on VisDrone
│   ├── train_stage2_sar.py       # Fine-tune on tiled HERIDAL+SARD
│   ├── validate.py             # mAP/precision/recall evaluation
│   └── export_hailo_prep.py    # Export to ONNX, ready for Hailo Dataflow Compiler
├── configs/
│   ├── data_sar.yaml           # YOLO dataset config (stage 2)
│   ├── data_visdrone.yaml      # YOLO dataset config (stage 1)
│   └── augment_sar.yaml        # SAR-specific augmentation hyperparameters
└── models/                     # Trained weights land here
```

## Quickstart

```bash
# 1. Set up environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Download datasets (see scripts/download_datasets.sh for manual steps too)
bash scripts/download_datasets.sh

# 3. Convert HERIDAL VOC XML labels to YOLO format
python scripts/convert_heridal.py

# 4. Merge HERIDAL + SARD into one unified dataset
python scripts/merge_datasets.py

# 5. Tile the merged dataset (SAHI slicing-aided training)
python scripts/tile_dataset.py --input data/processed/merged --output data/processed/tiled --tile-size 640 --overlap 0.2

# 6. Stage 1: pretrain on VisDrone (optional but recommended — skip if short on time)
python scripts/train_stage1_visdrone.py

# 7. Stage 2: fine-tune on tiled SAR data
python scripts/train_stage2_sar.py

# 8. Validate
python scripts/validate.py --weights models/sar_stage2/weights/best.pt

# 9. Export for Hailo compilation (run the actual HEF compile step on an x86 Ubuntu host separately)
python scripts/export_hailo_prep.py --weights models/sar_stage2/weights/best.pt
```

## Key numbers to keep in mind (from research)

- Target GSD: 1–2 cm/px (SARUAV-optimal) → fly at **~20–30m AGL** with this camera/lens
- HERIDAL average person bbox: ~64×64 px (σ≈23) — your tiles should keep targets near/above this
- Benchmark to beat: **mAP@50 ≥ 0.80** on held-out HERIDAL/SARD split (matches best published civil-SAR YOLOv8 result)
- Expect some mAP loss after INT8 quantization — validate post-quantization, budget for it

## Notes

- This scaffold assumes a **local GPU machine** for training (per your setup). Hailo HEF
  compilation itself requires a separate **x86 Ubuntu host** with the Hailo Dataflow
  Compiler installed — that's a separate step from training, covered in
  `export_hailo_prep.py`'s instructions.
- Dataset licenses: HERIDAL is CC BY 3.0 (free use with attribution). SARD via Kaggle —
  check the specific Kaggle listing's license before any commercial/competition use.

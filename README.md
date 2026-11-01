# Seraph Zenith: autonomous search-and-rescue drone, on-board human detection

Team Orbital Bytes, SIH26177. Technical lead: Benhur Lee A K.

Rescuers lose hours scanning aerial footage for survivors, and disaster sites often have no network.
Seraph Zenith detects people on the drone itself: global-shutter camera, Raspberry Pi 5, Hailo-8 (26 TOPS).

## Pipeline
1. A 1456x1088 frame is cut (not resized) into 640x640 tiles with 20% overlap (SAHI): 3x2 = 6 tiles. See `docs/sahi_tiling.png`.
2. YOLOv8s (pretrained on VisDrone, fine-tuned on HERIDAL + SARD) runs on each tile, INT8-compiled to a Hailo HEF.
3. Tile boxes are mapped back to frame coordinates and merged with NMS.
Navigation: RTAB-Map SLAM into ArduPilot EKF3 via MAVLink odometry (non-GPS flight). A 6-class hazard model is in progress.

## Results (human detection, tile-level validation split: 2,424 tiles, 3,445 people)
| | mAP@50 | mAP@50-95 |
|---|---|---|
| Float, overall | 0.854 | 0.501 |
| Float, HERIDAL | 0.762 | 0.415 |
| Float, SARD | 0.949 | 0.622 |
| INT8 vs float (same scorer) | 0.781 vs 0.849 | 0.365 vs 0.547 |

INT8 HEF runs at 185 FPS (model only, one tile). INT8 adds about 78% more false positives at confidence 0.30.

## Limits
Not flight-validated. 3-7 m ground tests detect 0%, 7% and 20% (outside the operating range). 10-15 m not yet measured.
End-to-end tiled FPS not measured. Calibration used about 300 validation images. Possible near-duplicate SARD frames.

## Contents
- `eval/score_predictions.py`: scores COCO-style predictions against YOLO labels (mAP@50, mAP@50-95, best-F1, per-source)
- `docs/`: tiling diagram and its script
- `deck/`: talk slide sources
- TODO: add training scripts, Hailo compile steps and the Pi inference script.

Paper: "Edge-Deployed Human Detection for Drone-Based Search and Rescue", ICOIICS 2026.

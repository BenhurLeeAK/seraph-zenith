"""
Exports the trained YOLOv8 model to ONNX format, ready for Hailo Dataflow
Compiler ingestion. The actual HEF compilation step (ONNX -> .hef) must
run on a separate x86 Ubuntu host with the Hailo Dataflow Compiler
installed -- it does NOT run on ARM/Raspberry Pi, and doesn't run here
either (Hailo's compiler isn't available via pip/public package indexes).

This script handles the part that CAN run on your local GPU machine.

Usage:
    python scripts/export_hailo_prep.py --weights models/sar_stage2/weights/best.pt
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(args.weights)

    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=11,       # Hailo Dataflow Compiler compatibility
        simplify=True,  # onnx-simplifier -- reduces graph complexity for the compiler
        dynamic=False,  # Hailo compilation requires a fixed input shape
    )

    print(f"\nONNX model exported to: {onnx_path}")
    print("\n" + "=" * 60)
    print("NEXT STEPS (on a separate x86 Ubuntu host with Hailo Dataflow Compiler):")
    print("=" * 60)
    print("""
1. Install the Hailo Dataflow Compiler + Hailo Model Zoo on an x86
   Ubuntu machine (this cannot run on the Pi 5 or ARM):
     https://hailo.ai/developer-zone/ (requires free Hailo Developer account)

2. Prepare a calibration dataset -- a few hundred representative images
   (use held-out validation images from your tiled SAR dataset) for
   INT8 quantization calibration.

3. Run the Hailo Model Zoo compilation pipeline, e.g.:
     hailomz compile yolov8s \\
         --ckpt {onnx_path} \\
         --calib-path <path_to_calibration_images> \\
         --classes 1

4. This produces a .hef file -- copy it to the Raspberry Pi 5.

5. VALIDATE POST-QUANTIZATION ACCURACY before trusting it in the field:
   run the compiled .hef against your validation set (via the Hailo
   Python API / hailortcli) and compare mAP against the pre-quantization
   ONNX result from this script. Research shows INT8 quantization
   typically costs a few points of mAP -- if the drop is large, the
   calibration set may need to be larger/more representative, or you
   may need to adjust quantization settings.

6. Test real end-to-end FPS on the actual Pi 5 + Hailo-8 with your
   camera pipeline (Picamera2 dual-stream + hailo_platform inference).
   Do not trust Hailo Model Zoo's synthetic benchmark numbers --
   measure it yourself with hailortcli run and your own tiling pipeline.
""".format(onnx_path=onnx_path))


if __name__ == "__main__":
    main()

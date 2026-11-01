"""
Runs Hailo YOLOv8s inference using TILED preprocessing, matching how the
model was actually trained (SAHI-style 640x640 slices from the native
1456x1088 camera frame), instead of a naive full-frame resize.
"""
import argparse

import cv2
import numpy as np
from hailo_platform import (
    HEF,
    ConfigureParams,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)

CONF_THRESHOLD = 0.3
TILE_SIZE = 640
OVERLAP = 0.2


def get_tiles(img_w, img_h, tile_size, overlap):
    step = int(tile_size * (1 - overlap))
    xs = list(range(0, max(img_w - tile_size, 0) + 1, step))
    ys = list(range(0, max(img_h - tile_size, 0) + 1, step))
    if not xs or xs[-1] + tile_size < img_w:
        xs.append(max(img_w - tile_size, 0))
    if not ys or ys[-1] + tile_size < img_h:
        ys.append(max(img_h - tile_size, 0))
    return [(x, y) for y in ys for x in xs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--hef", type=str, required=True)
    parser.add_argument("--output", type=str, default="tiled_annotated.jpg")
    args = parser.parse_args()

    hef = HEF(args.hef)
    target = VDevice()
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    input_vstream_info = hef.get_input_vstream_infos()[0]
    output_name = hef.get_output_vstream_infos()[0].name

    input_vstream_params = InputVStreamParams.make(network_group)
    output_vstream_params = OutputVStreamParams.make(network_group)

    full_img = cv2.imread(args.image)
    img_h, img_w = full_img.shape[:2]

    pad_w = max(TILE_SIZE - img_w, 0)
    pad_h = max(TILE_SIZE - img_h, 0)
    if pad_w or pad_h:
        full_img = cv2.copyMakeBorder(full_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        img_h, img_w = full_img.shape[:2]

    tiles = get_tiles(img_w, img_h, TILE_SIZE, OVERLAP)
    print(f"Image size: {img_w}x{img_h}, generated {len(tiles)} tiles")

    annotated = full_img.copy()
    total_detections = 0
    all_scores = []

    with InferVStreams(network_group, input_vstream_params, output_vstream_params) as infer_pipeline:
        with network_group.activate(network_group_params):
            for tx, ty in tiles:
                tile = full_img[ty:ty + TILE_SIZE, tx:tx + TILE_SIZE]
                tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                input_data = {input_vstream_info.name: np.expand_dims(tile_rgb, axis=0)}

                results = infer_pipeline.infer(input_data)
                detections = results[output_name][0]

                for class_id, class_detections in enumerate(detections):
                    for det in class_detections:
                        score = float(det[4])
                        all_scores.append(score)
                        if score < CONF_THRESHOLD:
                            continue
                        ymin, xmin, ymax, xmax = det[0], det[1], det[2], det[3]
                        x1 = int(tx + xmin * TILE_SIZE)
                        y1 = int(ty + ymin * TILE_SIZE)
                        x2 = int(tx + xmax * TILE_SIZE)
                        y2 = int(ty + ymax * TILE_SIZE)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(annotated, f"person {score:.2f}", (x1, max(y1 - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        total_detections += 1

    all_scores.sort(reverse=True)
    print(f"\nDIAGNOSTIC - top 10 raw scores across all tiles: {all_scores[:10]}")
    print(f"DIAGNOSTIC - total raw candidates across all tiles: {len(all_scores)}")
    print(f"\nDetections found (above {CONF_THRESHOLD}): {total_detections}")

    cv2.imwrite(args.output, annotated)
    print(f"Annotated image saved to: {args.output}")


if __name__ == "__main__":
    main()

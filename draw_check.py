#!/usr/bin/env python3
"""Draw GT (green) and HEF detections >= conf (red) on a few supplement images.
  python draw_check.py --dets out/dets_hef_supp45.json --images supplement_labeled/train/images \
      --labels supplement_labeled/train/labels --out out/draw --conf 0.3
Writes one JPG per distance bucket (first image of 3m, 5m, 7m) plus the 2nd of each."""
import argparse, json, re
from pathlib import Path
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--dets", required=True); ap.add_argument("--images", required=True)
ap.add_argument("--labels", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--conf", type=float, default=0.3)
a = ap.parse_args()
dets = json.loads(Path(a.dets).read_text())["dets"]
Path(a.out).mkdir(parents=True, exist_ok=True)
seen = {}
for p in sorted(Path(a.images).iterdir()):
    m = re.search(r"supplement_(\d+)m_", p.name)
    if not m or p.stem not in dets:
        continue
    k = m.group(1)
    if seen.get(k, 0) >= 2:
        continue
    seen[k] = seen.get(k, 0) + 1
    img = cv2.imread(str(p)); h, w = img.shape[:2]
    lf = Path(a.labels) / (p.stem + ".txt")
    if lf.exists():
        for line in lf.read_text().split("\n"):
            t = line.split()
            if len(t) >= 5:
                cx, cy, bw, bh = [float(x) for x in t[1:5]]
                cv2.rectangle(img, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)),
                              (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), (0, 255, 0), 3)
    for x1, y1, x2, y2, s in dets[p.stem]:
        if s >= a.conf:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(img, f"{s:.2f}", (int(x1), max(int(y1) - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    o = Path(a.out) / f"{k}m_{seen[k]}.jpg"
    cv2.imwrite(str(o), img); print("wrote", o)

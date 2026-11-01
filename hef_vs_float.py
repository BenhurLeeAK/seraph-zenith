#!/usr/bin/env python3
"""
hef_vs_float.py -- WHERE does the compiled model (HEF) lose people compared with the float model?

For every ground-truth person, asks: did the float model find it? did the HEF find it? (any detection
with IoU >= 0.5 at conf >= --conf), split by person size in pixels. Also sanity-checks the HEF's own
detections (sizes, scores, whether they sit on people at all, and whether swapping x/y would line them up).

Save next to paper_eval.py, then:
  python hef_vs_float.py --float out/dets_valjson.json --hef out/dets_hef_val.json \
      --images data/processed/tiled/val/images --labels data/processed/tiled/val/labels
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_eval as pe  # noqa: E402

BINS = [(0, 32), (32, 48), (48, 64), (64, 96), (96, 128), (128, 1e9)]


def found(g, d, thr):
    if len(g) == 0:
        return np.zeros(0, bool)
    if len(d) == 0:
        return np.zeros(len(g), bool)
    return pe.iou_matrix(g, d[:, :4]).max(1) >= thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--float", dest="fl", required=True)
    ap.add_argument("--hef", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()

    gt, _ = pe.load_gt(a.labels, a.images)
    fd, _ = pe.load_dets(a.fl)
    hd, hmeta = pe.load_dets(a.hef)

    sizes, ff, hf = [], [], []
    h_sizes, h_scores, h_on_person, h_swap_hits, n_h = [], [], 0, 0, 0
    for s, g in gt.items():
        f = pe.prep(fd.get(s, np.zeros((0, 5))), a.conf)
        h = pe.prep(hd.get(s, np.zeros((0, 5))), a.conf)
        if len(g):
            sizes += np.sqrt((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])).tolist()
            ff += found(g, f, a.iou).tolist()
            hf += found(g, h, a.iou).tolist()
        if len(h):
            n_h += len(h)
            h_sizes += np.sqrt((h[:, 2] - h[:, 0]) * (h[:, 3] - h[:, 1])).tolist()
            h_scores += h[:, 4].tolist()
            if len(g):
                h_on_person += int((pe.iou_matrix(g, h[:, :4]).max(0) >= 0.1).sum())
                sw = h[:, [1, 0, 3, 2]]
                h_swap_hits += int((pe.iou_matrix(g, sw).max(0) >= a.iou).sum())
    sizes, ff, hf = np.array(sizes), np.array(ff), np.array(hf)

    print(f"\nconf >= {a.conf}, a person counts as found if any detection has IoU >= {a.iou}")
    print(f"HEF file: {hmeta.get('hef', '?')}  sha256 {hmeta.get('sha256', '?')[:16]}...  min_score_seen {hmeta.get('min_score_seen', '?')}\n")
    print("| person size (px, sqrt of area) | n people | float found | HEF found |")
    print("|---|---|---|---|")
    for lo, hi in BINS:
        m = (sizes >= lo) & (sizes < hi)
        if m.sum():
            lab = f"{lo:.0f}-{hi:.0f}" if hi < 1e8 else f"{lo:.0f}+"
            print(f"| {lab} | {int(m.sum())} | {100 * ff[m].mean():.0f}% | {100 * hf[m].mean():.0f}% |")
    print(f"| ALL | {len(sizes)} | {100 * ff.mean():.0f}% | {100 * hf.mean():.0f}% |")

    print(f"\nHEF detections at conf >= {a.conf}: {n_h}")
    if n_h:
        q = lambda x: np.percentile(x, [5, 50, 95]).round(2).tolist()
        print(f"  score 5/50/95 pct: {q(h_scores)}   box size (px) 5/50/95 pct: {q(h_sizes)}")
        print(f"  overlapping any real person (IoU >= 0.1): {h_on_person} ({100 * h_on_person / n_h:.0f}%)")
        print(f"  would match a person at IoU >= {a.iou} if x and y were swapped: {h_swap_hits}")
    print(f"float detections at conf >= {a.conf} for comparison: "
          f"{sum(len(pe.prep(fd.get(s, np.zeros((0, 5))), a.conf)) for s in gt)}")


if __name__ == "__main__":
    main()

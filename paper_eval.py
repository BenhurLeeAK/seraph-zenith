#!/usr/bin/env python3
"""
paper_eval.py -- one evaluator for every number in the ICOIICS_368 revision.

The same matching/AP code scores the float model and the INT8 HEF, so any
float-vs-INT8 difference is not an artifact of two different evaluators.
Matching and AP follow Ultralytics' DetectionValidator (IoU-descending greedy,
101-point interpolated AP, smoothed F1 curve) so the paper's headline numbers
can be reproduced as a self-check before anything new is reported.

Detections are exchanged as JSON:
  {"meta": {...}, "dets": {"<image stem>": [[x1,y1,x2,y2,conf], ...]}}
Float model -> `predict-val`/`distance` here. INT8 HEF -> hailo_predict.py on the Pi.

Subcommands
  predict-val   float model on val tiles -> dets.json            (needs ultralytics)
  metrics       dets.json + labels -> operating points, mAP, per-source split,
                confusion-matrix PNG, markdown report              (numpy/matplotlib)
  distance      per-distance table with Wilson + cluster-bootstrap CIs
  calib         build an INT8 calibration set from TRAIN tiles

Baseline only: weights paths that look like the fine-tune are refused.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

IOUV = np.linspace(0.5, 0.95, 10)
IMG_EXT = {".jpg", ".jpeg", ".png"}
FINETUNE_PAT = re.compile(r"ft_gentle|finetune|fine_tune|ft-|supp_ft", re.I)


# ----------------------------------------------------------------- basics
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_images(d):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in IMG_EXT)


def read_size(p):
    from PIL import Image
    with Image.open(p) as im:
        return im.size  # (w, h)


def load_gt(labels_dir, images_dir):
    """stem -> (n,4) xyxy pixel boxes; stem -> (w,h)."""
    gt, sizes = {}, {}
    for ip in list_images(images_dir):
        w, h = read_size(ip)
        lp = Path(labels_dir) / (ip.stem + ".txt")
        boxes = []
        if lp.exists():
            for line in lp.read_text().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                cx, cy, bw, bh = map(float, p[1:5])
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                              (cx + bw / 2) * w, (cy + bh / 2) * h])
        gt[ip.stem] = np.array(boxes, dtype=float).reshape(-1, 4)
        sizes[ip.stem] = (w, h)
    return gt, sizes


def load_dets(path):
    j = json.loads(Path(path).read_text())
    dets = {k: np.array(v, dtype=float).reshape(-1, 5) for k, v in j["dets"].items()}
    return dets, j.get("meta", {})


def save_dets(path, dets, meta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(
        {"meta": meta, "dets": {k: np.round(v, 3).tolist() for k, v in dets.items()}}))


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-16)


def match(iou, thr):
    """Ultralytics-style matching. iou: (n_gt, n_det), dets already sorted by conf desc.
    Returns tp flags (n_det,), gt-matched flags (n_gt,), pairs (k,2) of (gt, det)."""
    tp = np.zeros(iou.shape[1], bool)
    gm = np.zeros(iou.shape[0], bool)
    if iou.size == 0:
        return tp, gm, np.zeros((0, 2), int)
    g, d = np.nonzero(iou >= thr)
    if len(g) == 0:
        return tp, gm, np.zeros((0, 2), int)
    o = np.argsort(-iou[g, d], kind="stable")
    g, d = g[o], d[o]
    _, i = np.unique(d, return_index=True)       # each det keeps its best-IoU gt
    g, d = g[i], d[i]
    _, i = np.unique(g, return_index=True)       # each gt keeps highest-conf det
    g, d = g[i], d[i]
    tp[d] = True
    gm[g] = True
    return tp, gm, np.stack([g, d], 1)


def prep(dets_img, min_conf=0.0):
    d = dets_img[dets_img[:, 4] >= min_conf] if len(dets_img) else np.zeros((0, 5))
    return d[np.argsort(-d[:, 4], kind="stable")]


# ------------------------------------------------------------ AP / curves
def collect(gt, dets, min_conf=0.0):
    confs, tps, n_gt = [], [], 0
    for stem, g in gt.items():
        d = prep(dets.get(stem, np.zeros((0, 5))), min_conf)
        n_gt += len(g)
        iou = iou_matrix(g, d[:, :4])
        tp = np.zeros((len(d), len(IOUV)), bool)
        for k, t in enumerate(IOUV):
            tp[:, k] = match(iou, t)[0]
        confs.append(d[:, 4])
        tps.append(tp)
    if not confs:
        return np.zeros(0), np.zeros((0, len(IOUV)), bool), 0
    return np.concatenate(confs), np.concatenate(tps), n_gt


def _trapz(y, x):
    return (getattr(np, "trapezoid", None) or np.trapz)(y, x)


def ap_from(conf, tp, n_gt):
    if n_gt == 0 or len(conf) == 0:
        return 0.0
    i = np.argsort(-conf, kind="stable")
    tp = tp[i]
    tpc, fpc = np.cumsum(tp), np.cumsum(~tp)
    rec = tpc / (n_gt + 1e-16)
    prec = tpc / (tpc + fpc + 1e-16)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(_trapz(np.interp(x, mrec, mpre), x))


def curves(conf, tp0, n_gt, n=1000):
    px = np.linspace(0, 1, n)
    if n_gt == 0 or len(conf) == 0:
        z = np.zeros(n)
        return px, z, z, z
    i = np.argsort(-conf, kind="stable")
    c, t = conf[i], tp0[i]
    tpc, fpc = np.cumsum(t), np.cumsum(~t)
    rec = tpc / (n_gt + 1e-16)
    prec = tpc / (tpc + fpc + 1e-16)
    r = np.interp(-px, -c, rec, left=0)
    p = np.interp(-px, -c, prec, left=1)
    return px, p, r, 2 * p * r / (p + r + 1e-16)


def smooth(y, f=0.1):
    nf = round(len(y) * f * 2) // 2 + 1
    pad = np.ones(nf // 2)
    yp = np.concatenate((pad * y[0], y, pad * y[-1]))
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def counts_at(gt, dets, conf_thr, iou_thr):
    TP = FP = FN = 0
    for stem, g in gt.items():
        d = prep(dets.get(stem, np.zeros((0, 5))), conf_thr)
        tp, gm, _ = match(iou_matrix(g, d[:, :4]), iou_thr)
        TP += int(tp.sum())
        FP += int((~tp).sum())
        FN += int((~gm).sum())
    return TP, FP, FN


def prf(TP, FP, FN):
    p = TP / (TP + FP) if TP + FP else 0.0
    r = TP / (TP + FN) if TP + FN else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


# ------------------------------------------------------------ statistics
def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def cluster_bootstrap(tp_i, gt_i, B=5000, seed=0):
    """CI for pooled detection rate, resampling IMAGES (frames of the same people are not independent)."""
    tp_i, gt_i = np.asarray(tp_i, float), np.asarray(gt_i, float)
    n = len(tp_i)
    if n < 2 or gt_i.sum() == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (B, n))
    den = gt_i[idx].sum(1)
    ok = den > 0
    rates = tp_i[idx].sum(1)[ok] / den[ok]
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


# ------------------------------------------------------------ evaluation
def evaluate_subset(gt, dets, min_conf, op_iou, extra_confs):
    conf, tp, n_gt = collect(gt, dets, min_conf)
    aps = [ap_from(conf, tp[:, k], n_gt) for k in range(len(IOUV))]
    px, p_c, r_c, f1_c = curves(conf, tp[:, 0] if len(tp) else tp, n_gt)
    i_u = int(smooth(f1_c, 0.1).argmax())
    i_r = int(f1_c.argmax())
    out = {"n_tiles": len(gt), "n_gt": n_gt, "mAP50": aps[0], "mAP50_95": float(np.mean(aps)),
           "ultra_conf": float(px[i_u]), "ultra_P": float(p_c[i_u]), "ultra_R": float(r_c[i_u])}
    points = [(f"conf {c:.2f}", c) for c in extra_confs]
    points.append(("F1-opt (raw sweep)", float(px[i_r])))
    points.append(("F1-opt (Ultralytics-smoothed)", float(px[i_u])))
    rows = []
    for name, c in points:
        TP, FP, FN = counts_at(gt, dets, c, op_iou)
        p, r, f1 = prf(TP, FP, FN)
        rows.append({"point": name, "conf": round(c, 3), "iou": op_iou, "TP": TP, "FP": FP,
                     "FN": FN, "precision": round(p, 4), "recall": round(r, 4), "F1": round(f1, 4)})
    return out, rows


def plot_cm(TP, FP, FN, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = np.array([[TP, FP], [FN, 0]], float)
    fig, ax = plt.subplots(figsize=(4.6, 4.3), dpi=300)
    ax.imshow(M, cmap="Blues", vmin=0, vmax=max(M.max(), 1))
    labels = [[f"TP\n{TP:,}", f"FP\n{FP:,}"], [f"FN\n{FN:,}", "n/a"]]
    for r in range(2):
        for c in range(2):
            ax.text(c, r, labels[r][c], ha="center", va="center", fontsize=12,
                    color="white" if M[r, c] > 0.6 * M.max() else "black")
    ax.set_xticks([0, 1], ["human", "background"])
    ax.set_yticks([0, 1], ["human", "background"])
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def md_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------ subcommands
def cmd_metrics(a):
    gt, _ = load_gt(a.labels, a.images)
    dets, meta = load_dets(a.dets)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    miss = [s for s in gt if s not in dets]
    if miss:
        print(f"WARNING: {len(miss)} images have no entry in the dets file (treated as 'no detections'): {miss[:3]}...")
    subsets = {"all": gt}
    for pre in [p for p in a.prefixes.split(",") if p]:
        sub = {s: g for s, g in gt.items() if s.startswith(pre)}
        if sub:
            subsets[pre.rstrip("_")] = sub
    other = [s for s in gt if not any(s.startswith(p) for p in a.prefixes.split(",") if p)]
    if other:
        print(f"NOTE: {len(other)} images match no prefix (e.g. {other[:3]}); they count only in 'all'.")
    if any(s.startswith("supp_") for s in gt):
        print("WARNING: supp_ images are present in this evaluation set -- they are training data for the fine-tune.")

    summ, ops = [], []
    for name, sub in subsets.items():
        s, rows = evaluate_subset(sub, dets, a.min_conf, a.iou, [0.25, 0.30])
        s["subset"] = name
        summ.append(s)
        for r in rows:
            r["subset"] = name
            ops.append(r)
    TP, FP, FN = counts_at(gt, dets, a.cm_conf, a.cm_iou)
    p, r, f1 = prf(TP, FP, FN)
    cm_title = (f"{a.label}: conf >= {a.cm_conf}, IoU >= {a.cm_iou}\n"
                f"P={p:.3f}  R={r:.3f}  F1={f1:.3f}  (n={sum(len(g) for g in gt.values()):,} instances)")
    png = out / f"confusion_matrix_{a.label}.png"
    plot_cm(TP, FP, FN, cm_title, png)
    print(f"Confusion matrix (conf {a.cm_conf}, IoU {a.cm_iou}): TP={TP} FP={FP} FN={FN} -> {png}")
    if a.expect_cm:
        eTP, eFP, eFN = map(int, a.expect_cm.split(","))
        ok = (TP, FP, FN) == (eTP, eFP, eFN)
        print(f"  expected TP/FP/FN = {eTP}/{eFP}/{eFN}: {'MATCH' if ok else 'MISMATCH -- see note in report'}")

    scols = ["subset", "n_tiles", "n_gt", "mAP50", "mAP50_95", "ultra_conf", "ultra_P", "ultra_R"]
    ocols = ["subset", "point", "conf", "iou", "TP", "FP", "FN", "precision", "recall", "F1"]
    for s in summ:
        for k in ("mAP50", "mAP50_95", "ultra_conf", "ultra_P", "ultra_R"):
            s[k] = round(s[k], 4)
    write_csv(out / f"summary_{a.label}.csv", summ, scols)
    write_csv(out / f"operating_points_{a.label}.csv", ops, ocols)
    rep = [f"# {a.label}: validation metrics", "",
           f"- dets: `{a.dets}`  (weights/hef sha256: `{meta.get('sha256', 'n/a')}`)",
           f"- detections below conf {a.min_conf} discarded before scoring; operating-point IoU = {a.iou}",
           f"- confusion matrix: conf {a.cm_conf}, IoU {a.cm_iou}: TP {TP}, FP {FP}, FN {FN}", "",
           "## mAP (Ultralytics-style, 101-pt) and Ultralytics-smoothed max-F1 point", md_table(summ, scols), "",
           "## Operating points", md_table(ops, ocols)]
    (out / f"report_{a.label}.md").write_text("\n".join(rep))
    print(md_table(summ, scols))
    print(md_table(ops, ocols))


def tiled_detect(predict_fn, img, tile=640, overlap=0.2):
    """Mirror of infer_tiled.py: same grid, same bottom/right zero padding, NO cross-tile merge."""
    h0, w0 = img.shape[:2]
    pad_w, pad_h = max(tile - w0, 0), max(tile - h0, 0)
    if pad_w or pad_h:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)))
    h, w = img.shape[:2]
    step = int(tile * (1 - overlap))
    xs = list(range(0, max(w - tile, 0) + 1, step))
    ys = list(range(0, max(h - tile, 0) + 1, step))
    if not xs or xs[-1] + tile < w:
        xs.append(max(w - tile, 0))
    if not ys or ys[-1] + tile < h:
        ys.append(max(h - tile, 0))
    out = []
    for ty in ys:
        for tx in xs:
            d = predict_fn(np.ascontiguousarray(img[ty:ty + tile, tx:tx + tile]))
            if len(d):
                d = np.array(d, float)
                d[:, [0, 2]] += tx
                d[:, [1, 3]] += ty
                out.append(d)
    return np.concatenate(out) if out else np.zeros((0, 5))


def _guard(weights, allow, expect_sha):
    if FINETUNE_PAT.search(str(weights)) and not allow:
        sys.exit(f"REFUSING {weights}: looks like a fine-tuned checkpoint. Paper numbers use the baseline only "
                 f"(--allow-nonbaseline to override).")
    s = sha256(weights)
    print(f"weights sha256: {s}")
    if expect_sha and not s.startswith(expect_sha):
        sys.exit("sha256 mismatch vs --expect-sha256")
    return s


def cmd_predict_val(a):
    from ultralytics import YOLO
    import ultralytics
    sha = _guard(a.weights, a.allow_nonbaseline, a.expect_sha256)
    model = YOLO(a.weights)
    paths = list_images(a.images)
    dets, t0 = {}, time.time()
    for i in range(0, len(paths), 64):
        chunk = paths[i:i + 64]
        res = model.predict([str(p) for p in chunk], imgsz=a.imgsz, conf=a.floor, iou=a.nms_iou,
                            max_det=300, half=a.half, device=a.device, verbose=False)
        for p, r in zip(chunk, res):
            b = r.boxes
            dets[p.stem] = np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]], 1)
    meta = {"kind": "float-val-tiles", "weights": str(a.weights), "sha256": sha, "floor": a.floor,
            "nms_iou": a.nms_iou, "imgsz": a.imgsz, "half": a.half, "n_images": len(paths),
            "ultralytics": ultralytics.__version__, "seconds": round(time.time() - t0, 1)}
    save_dets(a.out, dets, meta)
    print(f"wrote {a.out} ({len(paths)} images)")


def coco_json_to_dets(path, stems):
    """Ultralytics predictions.json (COCO xywh, top-left) -> {stem: (n,5) xyxy+conf}."""
    dets = {s: [] for s in stems}
    for e in json.loads(Path(path).read_text()):
        s = str(e["image_id"])
        x, y, w, h = e["bbox"]
        dets.setdefault(s, []).append([x, y, x + w, y + h, e["score"]])
    return {k: np.array(v, dtype=float).reshape(-1, 5) for k, v in dets.items()}


def cmd_val_json(a):
    """Detections from Ultralytics' OWN validator (same preprocessing + NMS as scripts/validate.py),
    converted to the dets format. Used to check my scorer against validate.py's numbers."""
    from ultralytics import YOLO
    import ultralytics
    sha = _guard(a.weights, a.allow_nonbaseline, a.expect_sha256)
    model = YOLO(a.weights)
    res = model.val(data=a.data, split="val", imgsz=a.imgsz, batch=a.batch, conf=a.floor, iou=a.nms_iou,
                    max_det=300, save_json=True, plots=False, device=a.device,
                    project=a.project, name=a.name, exist_ok=True)
    print(f"ULTRALYTICS VALIDATOR SAYS: mAP50={res.box.map50:.4f}  mAP50-95={res.box.map:.4f}  "
          f"P={res.box.mp:.4f}  R={res.box.mr:.4f}")
    pj = Path(res.save_dir) / "predictions.json"
    dets = coco_json_to_dets(pj, [p.stem for p in list_images(a.images)])
    meta = {"kind": "float-val-tiles-ultralytics-validator", "weights": str(a.weights), "sha256": sha,
            "floor": a.floor, "nms_iou": a.nms_iou, "imgsz": a.imgsz, "n_images": len(dets),
            "ultralytics": ultralytics.__version__, "source": str(pj)}
    save_dets(a.out, dets, meta)
    print(f"wrote {a.out} ({len(dets)} images) from {pj}")


def distance_row(bucket, stems, gt, dets, conf_thr, iou_thr, loose):
    TP = G = FPo = FPb = 0
    tp_i, gt_i, confs, ious, best, loose_hits, found = [], [], [], [], [], 0, 0
    for s in stems:
        g = gt[s]
        d = prep(dets.get(s, np.zeros((0, 5))), conf_thr)
        iou = iou_matrix(g, d[:, :4])
        tp, gm, pairs = match(iou, iou_thr)
        tp_i.append(int(tp.sum()))
        gt_i.append(len(g))
        TP += int(tp.sum())
        G += len(g)
        found += int(tp.sum() > 0)
        if len(pairs):
            confs += d[pairs[:, 1], 4].tolist()
            ious += iou[pairs[:, 0], pairs[:, 1]].tolist()
        if len(g):
            b = iou.max(1) if iou.shape[1] else np.zeros(len(g))
            best += b.tolist()
            loose_hits += int((b >= loose).sum())
        if (~tp).any():
            # classify each FP against ALL gt boxes, including ones already matched
            ov = iou.max(0) if len(g) else np.zeros(len(d))
            FPo += int(((~tp) & (ov >= loose)).sum())
            FPb += int(((~tp) & (ov < loose)).sum())
    lo, hi = wilson(TP, G)
    blo, bhi = cluster_bootstrap(tp_i, gt_i)
    f = lambda x: round(float(x), 3)
    return {"bucket": f"{bucket} m", "n_images": len(stems), "n_instances": G, "TP": TP,
            "det_rate": f(TP / G) if G else "nan", "wilson95": f"[{lo:.3f}, {hi:.3f}]",
            "cluster_boot95": f"[{blo:.3f}, {bhi:.3f}]",
            "loose_rate": f(loose_hits / G) if G else "nan",
            "image_found": f"{found}/{len(stems)}",
            "conf_TP_mean_sd": f"{np.mean(confs):.3f} +/- {np.std(confs, ddof=1) if len(confs) > 1 else float('nan'):.3f} (n={len(confs)})" if confs else "n/a (no TP)",
            "IoU_TP_mean": f(np.mean(ious)) if ious else "n/a",
            "bestIoU_per_GT_mean": f(np.mean(best)) if best else "n/a",
            "FP_total": FPo + FPb, "FP_per_image": f((FPo + FPb) / len(stems)),
            "FP_on_person": FPo, "FP_background": FPb}


def cmd_distance(a):
    gt, _ = load_gt(a.labels, a.images)
    rx = re.compile(a.bucket_regex)
    if a.dets:
        dets, meta = load_dets(a.dets)
    else:
        from ultralytics import YOLO
        from PIL import Image
        sha = _guard(a.weights, a.allow_nonbaseline, a.expect_sha256)
        model = YOLO(a.weights)

        def pf(tile_bgr):
            r = model.predict(tile_bgr, imgsz=a.tile, conf=a.floor, iou=a.nms_iou, verbose=False)[0]
            b = r.boxes
            return np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]], 1)

        dets = {}
        for p in list_images(a.images):
            rgb = np.asarray(Image.open(p).convert("RGB"))
            dets[p.stem] = tiled_detect(pf, np.ascontiguousarray(rgb[:, :, ::-1]), a.tile, a.overlap)
        meta = {"kind": "float-tiled-fullframe", "weights": str(a.weights), "sha256": sha}
    buckets = {}
    for s in gt:
        m = rx.search(s)
        buckets.setdefault(m.group(1) if m else "unparsed", []).append(s)
    if "unparsed" in buckets:
        print(f"WARNING: {len(buckets['unparsed'])} files did not match --bucket-regex")
    rows = [distance_row(b, buckets[b], gt, dets, a.conf, a.iou, a.loose_iou)
            for b in sorted(buckets, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0))]
    cols = list(rows[0].keys())
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / f"distance_{a.label}.csv", rows, cols)
    (out / f"distance_{a.label}.md").write_text(
        f"# Per-distance ({a.label}); conf >= {a.conf}, strict IoU >= {a.iou}, loose IoU >= {a.loose_iou}\n\n"
        + md_table(rows, cols) + f"\n\nsource: sha256 {meta.get('sha256', 'n/a')}\n")
    print(md_table(rows, cols))


def cmd_calib(a):
    import shutil
    from PIL import Image
    rng = np.random.default_rng(a.seed)
    img_dir, lab_dir = Path(a.train_images), Path(a.train_labels)
    pos, neg = [], []
    for p in list_images(img_dir):
        if p.stem.startswith("supp_") and not a.include_supp:
            continue
        lp = lab_dir / (p.stem + ".txt")
        (pos if lp.exists() and lp.read_text().strip() else neg).append(p)
    n_neg = min(len(neg), int(round(a.n * (1 - a.pos_frac))))
    n_pos = min(len(pos), a.n - n_neg)
    pick = [pos[i] for i in rng.choice(len(pos), n_pos, replace=False)] + \
           [neg[i] for i in rng.choice(len(neg), n_neg, replace=False)]
    out = Path(a.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(out / f"calib_{len(pick)}_640x640x3_uint8_rgb.npy", mode="w+",
                                    dtype=np.uint8, shape=(len(pick), 640, 640, 3))
    for i, p in enumerate(pick):
        shutil.copy(p, out / "images" / p.name)
        im = np.asarray(Image.open(p).convert("RGB"))
        assert im.shape == (640, 640, 3), f"{p.name} has shape {im.shape}"
        arr[i] = im
    arr.flush()
    (out / "files.txt").write_text("\n".join(p.name for p in pick))
    print(f"calibration set: {n_pos} positive + {n_neg} background tiles from TRAIN (supp_ excluded: {not a.include_supp})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common_w(p):
        p.add_argument("--weights")
        p.add_argument("--expect-sha256", default=None, help="prefix of the baseline best.pt sha256")
        p.add_argument("--allow-nonbaseline", action="store_true")

    p = sp.add_parser("predict-val")
    common_w(p)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--floor", type=float, default=0.001)
    p.add_argument("--nms-iou", type=float, default=0.7)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true")
    p.add_argument("--device", default="0")
    p.set_defaults(fn=cmd_predict_val)

    p = sp.add_parser("val-json")
    common_w(p)
    p.add_argument("--data", default="configs/data_sar.yaml")
    p.add_argument("--images", required=True, help="val images dir (used only to list every tile, incl. empty ones)")
    p.add_argument("--out", required=True)
    p.add_argument("--floor", type=float, default=0.001)
    p.add_argument("--nms-iou", type=float, default=0.7)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="runs/detect/paper_val")
    p.add_argument("--name", default="float_val")
    p.set_defaults(fn=cmd_val_json)

    p = sp.add_parser("metrics")
    p.add_argument("--dets", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--label", default="float")
    p.add_argument("--min-conf", type=float, default=0.0,
                   help="discard dets below this before scoring (use the HEF's NMS floor to compare like with like)")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--cm-conf", type=float, default=0.25)
    p.add_argument("--cm-iou", type=float, default=0.45)
    p.add_argument("--prefixes", default="heridal_,sard_")
    p.add_argument("--expect-cm", default=None, help="TP,FP,FN to check against, e.g. 2755,603,690")
    p.set_defaults(fn=cmd_metrics)

    p = sp.add_parser("distance")
    common_w(p)
    p.add_argument("--dets", default=None, help="precomputed dets (e.g. from the Pi); otherwise --weights is run here")
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--label", default="float")
    p.add_argument("--conf", type=float, default=0.3, help="0.3 = infer_tiled.py's deployed threshold")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--loose-iou", type=float, default=0.1)
    p.add_argument("--tile", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--floor", type=float, default=0.05)
    p.add_argument("--nms-iou", type=float, default=0.7)
    p.add_argument("--bucket-regex", default=r"(?:^|[_\-])(\d+)m(?=[_\-.]|$)")
    p.set_defaults(fn=cmd_distance)

    p = sp.add_parser("calib")
    p.add_argument("--train-images", required=True)
    p.add_argument("--train-labels", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--pos-frac", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include-supp", action="store_true")
    p.set_defaults(fn=cmd_calib)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
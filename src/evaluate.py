"""
evaluate.py — compute mAP50 for a Detector over the YOLO val split, torch-free.

Standard detection mAP: per class, sort predictions by confidence, greedily match to
unmatched ground-truth boxes at IoU >= 0.5 (TP else FP), integrate the precision-recall
curve for AP, average over classes. Used to (a) verify the local ONNX pipeline reproduces
the training-time mAP (parity), and (b) measure the INT8 accuracy drop.
"""
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from detector import Detector, YCB_CLASSES

IMG_W, IMG_H = 640, 480


def _load_gt(label_path: Path):
    """YOLO label -> list of (cls, x1, y1, x2, y2) in pixels."""
    out = []
    if not label_path.exists():
        return out
    for ln in label_path.read_text().splitlines():
        if not ln.strip():
            continue
        c, xc, yc, w, h = ln.split()
        c = int(c); xc, yc, w, h = (float(v) for v in (xc, yc, w, h))
        out.append((c, (xc - w / 2) * IMG_W, (yc - h / 2) * IMG_H,
                    (xc + w / 2) * IMG_W, (yc + h / 2) * IMG_H))
    return out


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _ap(rec, prec):
    """All-point (VOC2010) AP: area under the monotonic-envelope PR curve."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_map50(detector: Detector, image_dir: Path, label_dir: Path,
                  iou_thr: float = 0.5, limit: int | None = None):
    imgs = sorted(image_dir.glob("*.png"))
    if limit:
        imgs = imgs[:limit]
    nc = len(YCB_CLASSES)
    preds = [[] for _ in range(nc)]     # per class: (conf, img_idx, box)
    n_gt = [0] * nc
    gts = []                            # per image: list of (cls, box, matched-flag holder)

    for i, ip in enumerate(tqdm(imgs, desc="eval", leave=False)):
        gt = _load_gt(label_dir / f"{ip.stem}.txt")
        gts.append(gt)
        for c, *_ in gt:
            n_gt[c] += 1
        for d in detector.detect(cv2.imread(str(ip))):
            preds[d.cls].append((d.conf, i, (d.x1, d.y1, d.x2, d.y2)))

    aps = []
    for c in range(nc):
        if n_gt[c] == 0:
            continue
        dets = sorted(preds[c], key=lambda t: -t[0])
        used = {i: np.zeros(sum(1 for g in gts[i] if g[0] == c), bool) for i in range(len(gts))}
        tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
        for k, (conf, i, box) in enumerate(dets):
            gt_boxes = [g[1:] for g in gts[i] if g[0] == c]
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(gt_boxes):
                v = _iou(box, gb)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= iou_thr and not used[i][best_j]:
                tp[k] = 1; used[i][best_j] = True
            else:
                fp[k] = 1
        tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
        rec = tp_c / n_gt[c]
        prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
        aps.append(_ap(rec, prec))
    return float(np.mean(aps)), aps


if __name__ == "__main__":
    import sys
    repo = Path(__file__).resolve().parents[1]
    onnx = sys.argv[1] if len(sys.argv) > 1 else str(repo / "runs/ycb_artifacts/best.onnx")
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    # mAP convention (matches Ultralytics val): keep all preds (conf~0), NMS iou=0.7.
    det = Detector(onnx, conf=0.001, iou=0.7)
    m, _ = compute_map50(det, repo / "data/ycb_yolo/images/val",
                         repo / "data/ycb_yolo/labels/val", limit=limit)
    print(f"{Path(onnx).name}: mAP50 = {m:.4f}  (over {limit or 'all'} val images)")

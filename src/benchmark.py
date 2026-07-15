"""
benchmark.py — the optimization sweep: model size, CPU latency (p50/p95/p99), and
accuracy (mAP50), for the fp32 ONNX vs the INT8 model. This is the Day-3 deliverable:
one table quantifying the size/latency/accuracy trade-off of quantization.
"""
from __future__ import annotations
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from detector import Detector, letterbox
from evaluate import compute_map50

REPO = Path(__file__).resolve().parents[1]
VAL_IMG = REPO / "data/ycb_yolo/images/val"
VAL_LBL = REPO / "data/ycb_yolo/labels/val"


def _fixed_input(imgsz=640):
    img = cv2.imread(str(next(VAL_IMG.glob("*.png"))))
    lb, *_ = letterbox(img, imgsz)
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def bench_latency(onnx_path: Path, n=300, warmup=30, intra_threads=None):
    so = ort.SessionOptions()
    if intra_threads:
        so.intra_op_num_threads = intra_threads
    sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = _fixed_input()
    for _ in range(warmup):
        sess.run(None, {name: x})
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        sess.run(None, {name: x})
        ts.append((time.perf_counter() - t) * 1000.0)
    ts = np.array(ts)
    return {"p50": np.percentile(ts, 50), "p95": np.percentile(ts, 95),
            "p99": np.percentile(ts, 99), "mean": ts.mean()}


def main():
    art = REPO / "runs/ycb_artifacts"
    rows = []
    for label, fname in [("fp32 ONNX", "best.onnx"), ("INT8", "best_int8.onnx")]:
        p = art / fname
        lat = bench_latency(p)
        size = p.stat().st_size / 1e6
        m, _ = compute_map50(Detector(str(p), conf=0.001, iou=0.7), VAL_IMG, VAL_LBL)
        rows.append((label, size, lat["p50"], lat["p95"], lat["p99"], m))

    print(f"\n{'model':12s} {'size MB':>8s} {'p50 ms':>8s} {'p95 ms':>8s} {'p99 ms':>8s} {'mAP50':>7s}")
    print("-" * 56)
    for r in rows:
        print(f"{r[0]:12s} {r[1]:8.1f} {r[2]:8.2f} {r[3]:8.2f} {r[4]:8.2f} {r[5]:7.3f}")
    fp, iq = rows[0], rows[1]
    print(f"\nINT8 vs fp32: {fp[1] / iq[1]:.1f}x smaller, "
          f"{fp[2] / iq[2]:.2f}x faster p50 ({fp[2]:.1f}->{iq[2]:.1f} ms), "
          f"{(iq[5] - fp[5]):+.3f} mAP50")


if __name__ == "__main__":
    main()

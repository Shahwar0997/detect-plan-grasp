"""
detector.py — torch-free YOLOv8 inference via ONNX Runtime.

Loads an exported YOLOv8 ONNX (output [1, 4+nc, 8400]: boxes already decoded to xywh in the
640x640 letterboxed frame, no NMS baked in) and runs the full detect pipeline:
letterbox preprocess -> ORT -> confidence filter -> class-aware NMS -> boxes in original
image coordinates. This same path is benchmarked fp32-vs-INT8 (Day 3) and drives the sim
loop (Days 5-6). No PyTorch — deployment-shaped.
"""
from __future__ import annotations
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import onnxruntime as ort

YCB_CLASSES = [
    "002_master_chef_can", "003_cracker_box", "004_sugar_box", "005_tomato_soup_can",
    "006_mustard_bottle", "007_tuna_fish_can", "008_pudding_box", "009_gelatin_box",
    "010_potted_meat_can", "011_banana", "019_pitcher_base", "021_bleach_cleanser",
    "024_bowl", "025_mug", "035_power_drill", "036_wood_block", "037_scissors",
    "040_large_marker", "051_large_clamp", "052_extra_large_clamp", "061_foam_brick",
]


class Detection(NamedTuple):
    x1: float; y1: float; x2: float; y2: float
    cls: int
    conf: float


def letterbox(img: np.ndarray, size: int = 640, color=(114, 114, 114)):
    """Resize preserving aspect ratio and pad to a square. Returns (padded, scale, dx, dy)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), color, np.uint8)
    out[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, dx, dy


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Plain greedy NMS on xyxy boxes; returns kept indices."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.clip(xx2 - xx1, 0, None); h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][ovr <= iou_thr]
    return keep


class Detector:
    def __init__(self, onnx_path, conf: float = 0.25, iou: float = 0.45,
                 imgsz: int = 640, intra_threads: int | None = None):
        so = ort.SessionOptions()
        if intra_threads:
            so.intra_op_num_threads = intra_threads
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.conf, self.iou, self.imgsz = conf, iou, imgsz

    def preprocess(self, img: np.ndarray):
        lb, r, dx, dy = letterbox(img, self.imgsz)
        x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), r, dx, dy

    def infer_raw(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.inp: x})[0]           # [1, 4+nc, 8400]

    def postprocess(self, out: np.ndarray, r, dx, dy) -> list[Detection]:
        p = out[0].T                                            # [8400, 4+nc]
        boxes, scores = p[:, :4], p[:, 4:]
        cls = scores.argmax(1)
        conf = scores.max(1)
        keep = conf > self.conf
        boxes, cls, conf = boxes[keep], cls[keep], conf[keep]
        if not len(boxes):
            return []
        # xywh (letterboxed px) -> xyxy, then undo the letterbox back to original coords
        xy = np.empty_like(boxes)
        xy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        xy[:, [0, 2]] = (xy[:, [0, 2]] - dx) / r
        xy[:, [1, 3]] = (xy[:, [1, 3]] - dy) / r
        dets: list[Detection] = []
        for c in np.unique(cls):                                # class-aware NMS
            m = cls == c
            for i in _nms(xy[m], conf[m], self.iou):
                b = xy[m][i]
                dets.append(Detection(*b.tolist(), int(c), float(conf[m][i])))
        return dets

    def detect(self, img: np.ndarray) -> list[Detection]:
        x, r, dx, dy = self.preprocess(img)
        return self.postprocess(self.infer_raw(x), r, dx, dy)


if __name__ == "__main__":
    import sys
    repo = Path(__file__).resolve().parents[1]
    det = Detector(repo / "runs/ycb_artifacts/best.onnx")
    img_path = sys.argv[1] if len(sys.argv) > 1 else next(
        (repo / "data/ycb_yolo/images/val").glob("*.png"))
    img = cv2.imread(str(img_path))
    for d in det.detect(img):
        print(f"{YCB_CLASSES[d.cls]:24s} {d.conf:.2f}  "
              f"[{d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f}]")

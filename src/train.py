"""
train.py — train the YCB 21-class YOLOv8-nano detector.

Meant to run on a GPU (Colab); also runs locally on CPU/MPS for a smoke test. Reads the
YOLO manifest produced by prepare_ycb.py and writes weights + curves to runs/. Exports ONNX
at the end (opset 13, per-channel INT8-ready) so Day 3's optimization has its artifact.

Usage (Colab GPU):
    python src/train.py --data data/ycb_yolo/data.yaml --epochs 50 --batch 64 --device 0
"""
from __future__ import annotations
import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ycb_yolo/data.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=64)      # 64 fits a T4/A100; drop for local
    ap.add_argument("--device", default="0")              # "0" GPU · "cpu" · "mps"
    ap.add_argument("--name", default="ycb_detector")
    ap.add_argument("--model", default="yolov8n.pt")      # nano: small, quantizes cleanly
    args = ap.parse_args()

    model = YOLO(args.model)                                # pretrained COCO weights -> transfer
    model.train(
        data=str((repo / args.data).resolve()),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=str(repo / "runs"), name=args.name,
        patience=15,                                       # early-stop if val plateaus
    )
    # Best weights are saved by Ultralytics at runs/<name>/weights/best.pt
    best = repo / "runs" / args.name / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=str((repo / args.data).resolve()), device=args.device)
    print(f"val mAP50 = {metrics.box.map50:.4f} · mAP50-95 = {metrics.box.map:.4f}")

    # Export ONNX for Day 3 (opset 13 enables per-channel INT8 quantization later).
    YOLO(str(best)).export(format="onnx", opset=13, imgsz=args.imgsz)
    print(f"exported {best.with_suffix('.onnx')}")


if __name__ == "__main__":
    main()

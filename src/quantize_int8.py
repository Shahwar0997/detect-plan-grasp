"""
quantize_int8.py — static INT8 post-training quantization of the YCB detector ONNX.

Measures per-layer activation ranges on real calibration images, then quantizes weights +
activations to INT8 — EXCEPT the detection head (`model.22`), which is precision-sensitive:
quantizing it collapses the class scores (the mAP=0 failure mode seen on the PPE detector).
Per-channel, QDQ format.
"""
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from detector import letterbox


class YcbCalib(CalibrationDataReader):
    """Feeds preprocessed real frames so activation ranges are measured, not guessed."""
    def __init__(self, image_dir: Path, input_name: str, n: int = 200, imgsz: int = 640):
        self.input_name = input_name
        imgs = sorted(Path(image_dir).glob("*.png"))[:n]
        self._it = iter(self._gen(imgs, imgsz))

    def _gen(self, imgs, imgsz):
        for ip in imgs:
            lb, *_ = letterbox(cv2.imread(str(ip)), imgsz)
            x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            yield {self.input_name: np.ascontiguousarray(x.transpose(2, 0, 1)[None])}

    def get_next(self):
        return next(self._it, None)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    src = repo / "runs/ycb_artifacts/best.onnx"
    prepped = repo / "runs/ycb_artifacts/best_prep.onnx"
    dst = repo / "runs/ycb_artifacts/best_int8.onnx"

    quant_pre_process(str(src), str(prepped))            # shape inference + cleanup
    graph = onnx.load(str(prepped)).graph
    input_name = graph.input[0].name

    # Exclude the detection head — quantizing model.22 destroys class-score precision.
    exclude = [n.name for n in graph.node if "model.22" in n.name]
    print(f"excluding {len(exclude)} detection-head (model.22) nodes from quantization")

    quantize_static(
        str(prepped), str(dst),
        calibration_data_reader=YcbCalib(repo / "data/ycb_yolo/images/val", input_name),
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        per_channel=True,
        nodes_to_exclude=exclude,
    )
    fp32_mb = src.stat().st_size / 1e6
    int8_mb = dst.stat().st_size / 1e6
    print(f"fp32 {fp32_mb:.1f} MB -> INT8 {int8_mb:.1f} MB  ({fp32_mb / int8_mb:.1f}x smaller)")


if __name__ == "__main__":
    main()

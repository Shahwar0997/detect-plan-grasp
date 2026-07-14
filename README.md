# Detect · Plan · Grasp

**A robotic manipulation pipeline: a fast, optimized object detector given a body.**
From an RGB-D view of a cluttered scene, the system detects objects, lifts each 2D
detection into 3D, plans a grasp, and executes it on a simulated arm — then measures
**grasp success rate**, not just detection accuracy.

> 🚧 **Work in progress.** Built step by step; this README grows as each phase lands.

## The idea

Two cleanly separable halves:

- **Perception (learned, optimized).** A YOLOv8-nano detector on the 21 YCB-Video
  objects, run through a full inference-optimization pipeline — ONNX → INT8
  quantization → C++ ONNX Runtime → benchmarking (p50/p99). *(The systems half.)*
- **Action (deterministic geometry).** Back-project the detected region's depth into a
  3D point cloud, estimate the object's pose, plan a grasp analytically, and execute it
  via inverse kinematics in a PyBullet physics sim. *(The robotics half.)*

Detection feeds planning; planning feeds motion; the loop verifies itself
(**perceive → decide → act → verify**).

## Pipeline

| Stage | What | Status |
|---|---|---|
| Data | YCB-Video (BOP) → YOLO labels | ⬜ |
| Train | YOLOv8-n, 21 classes | ⬜ |
| Optimize | ONNX → INT8 → C++ → benchmark | ⬜ |
| Lift | depth back-projection → 3D pose | ⬜ |
| Plan | analytic grasp planner | ⬜ |
| Execute | PyBullet arm + IK, closed loop | ⬜ |
| Evaluate | grasp success vs. baseline + cycle time | ⬜ |

See [`BUILD_PLAN.md`](BUILD_PLAN.md) for the full step-by-step plan and scope.

## Stack

Python · Ultralytics YOLOv8 · ONNX Runtime (Python + C++) · INT8 quantization ·
Open3D · PyBullet · OpenCV · NumPy. Training on Colab GPU; inference & sim run locally.

## Dataset

[YCB-Video](https://bop.felk.cvut.cz/datasets/) (via the BOP benchmark) — 21 household
objects across 92 RGB-D videos with 6D pose, masks, and depth. Only a subset is used;
data is not committed (see `.gitignore`).

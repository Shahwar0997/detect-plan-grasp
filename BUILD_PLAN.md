# Build Plan — Detect · Plan · Grasp (Robotic Manipulation)

> Standalone ML-systems + robotics project. A **new** YOLOv8 detector on YCB-Video,
> optimized the same way as the PPE engine (ONNX → INT8 → C++ → benchmark), then given
> a body: lift 2D detections to 3D, plan a grasp, execute in PyBullet, measure grasp
> success. North-star reference: `~/Sh-Context/project_deep_docs/manipulation-build-guide.html`.

## Thesis
Two cleanly separable halves:
- **Perception (learned, optimized):** YOLOv8-n on 21 YCB objects → ONNX → INT8 → C++ → benchmark.
- **Action (deterministic geometry):** depth back-projection → point cloud → pose → analytic grasp planner → PyBullet arm + IK → closed loop.

Headline metric = **grasp success rate vs. a naive baseline** + perception **cycle time**.

## MVP (stop here = a complete, defensible project)
| # | Step | File(s) | Deliverable |
|---|------|---------|-------------|
| 1 | Env + **small YCB subset** (BOP modular: base + models + subset of train_real) | — | data ready |
| 2 | Stage 1: masks→YOLO labels, train, export ONNX, INT8, benchmark | `prepare_ycb.py` `train.py` `export_onnx.py` `quantize_int8.py` `benchmark.py` | fast 21-class detector + p50/p99 |
| 3 | Stage 2: depth crop → point cloud → **centroid position** | `lift_to_3d.py` | 2D box → 3D position |
| 4 | PyBullet: table + Panda arm + camera + IK `reach()` | `sim.py` | arm reaches a typed pose |
| 5 | Analytic grasp planner (top-down) + closed loop + height-threshold success | `grasp_planner.py` `run.py` | working pick |
| 6 | **Measure:** success rate vs. naive baseline + cycle time | `run.py` | results table ← the point |

## Stretch (résumé bonuses, only after MVP)
- PCA / ICP orientation → full 6D pose (`refine_with_mesh`)
- C++ ONNX Runtime inference path (mirrors PPE `cpp/`)
- TensorRT / GPU numbers (Colab)
- Lean Docker serving image + recorded demo clip

## Scope guardrails (prep-first, lean)
- Position-only closed loop **first**. Do not start with ICP.
- YCB **subset** of videos, not the full 265 GB.
- Use ground-truth masks to validate lift→plan→grasp **before** wiring the real detector in
  (isolates any drop to the detector).
- This is *robotics-flavored ML systems*: no ROS, no hardware, no controls theory.

## Structure
- Code: `~/projects/manipulation-grasp/` (this repo → GitHub `Shahwar0997/manipulation-grasp`)
- Deep-dive docs: `~/Sh-Context/grasp-docs/` (per-phase, figures — mirrors `ppe-docs/`)
- Build log: `~/Sh-Context/grasp-build-log.md`

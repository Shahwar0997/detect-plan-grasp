"""
run_detector.py — the closed loop with the REAL detector driving the grasp (not the oracle).

Per trial: randomize the YCB can on the table -> render RGB-D -> INT8 YOLO detects it ->
back-project its box with the sim depth -> transform camera-frame to world -> plan grasp ->
execute -> verify. This is the full "perception drives action" loop: the arm grasps an object
it was *not* told the location of; the detector found it.

Reports detection rate, grasp success rate, and the perceive+plan cycle time (which now
includes real detector inference).
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sim import Sim                                        # noqa: E402
from detector import Detector, YCB_CLASSES                 # noqa: E402
from lift_to_3d import back_project, estimate_position     # noqa: E402
from grasp_planner import plan_grasp, execute_grasp        # noqa: E402

SCENE = str(Path(__file__).resolve().parents[1] / "sim/franka/dpg_scene_ycb.xml")
MODEL = str(Path(__file__).resolve().parents[1] / "runs/ycb_artifacts/best_int8.onnx")


def perceive(sim: Sim, det: Detector):
    """Render RGB-D, detect the object, and return its 3D world position (or None)."""
    rgb, depth = sim.render(), sim.render_depth()
    dets = det.detect(rgb[:, :, ::-1])                     # detector expects BGR
    if not dets:
        return None, None
    d0 = max(dets, key=lambda t: t.conf)
    pts = back_project(depth, (d0.x1, d0.y1, d0.x2, d0.y2), sim.intrinsics())
    c = estimate_position(pts)                             # camera-frame (CV: x right, y down, z fwd)
    if c is None:
        return None, d0
    c_mj = np.array([c[0], -c[1], -c[2]])                  # CV -> MuJoCo camera frame
    world = sim.d.cam_xpos[sim.cam] + sim.d.cam_xmat[sim.cam].reshape(3, 3) @ c_mj
    return world, d0


def run(n: int, seed: int = 0, verbose: bool = False):
    det = Detector(MODEL, conf=0.25)
    rng = np.random.default_rng(seed)
    sim = Sim(scene=SCENE)
    detected = grasped = 0
    cyc = []
    for t in range(n):
        x, y = 0.5 + rng.uniform(-0.1, 0.1), rng.uniform(-0.15, 0.15)
        sim.reset(obj_pos=(x, y, 0.36))
        for _ in range(200):
            mujoco.mj_step(sim.m, sim.d)

        t0 = time.perf_counter()
        pos, d0 = perceive(sim, det)                       # detect + lift + transform
        cyc.append((time.perf_counter() - t0) * 1000)
        if pos is None:
            if verbose:
                print(f"  trial {t+1:2d}: not detected")
            continue
        detected += 1
        pos[2] -= 0.02          # grip lower: the visible-surface centroid sits high on the can
        g = plan_grasp(pos, sim)
        ok = bool(g is not None and execute_grasp(sim, g))
        grasped += ok
        if verbose:
            print(f"  trial {t+1:2d}: {YCB_CLASSES[d0.cls]} {d0.conf:.2f} -> "
                  f"{'GRASP' if ok else 'miss '}")
    return detected / n, grasped / n, float(np.median(cyc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    print(f"detector-driven closed loop (n={args.trials}, YCB can, randomized)\n")
    det_rate, grasp_rate, cyc = run(args.trials, args.seed, args.verbose)
    print(f"\ndetection rate:     {det_rate*100:5.1f}%")
    print(f"grasp success rate: {grasp_rate*100:5.1f}%  (perception-driven, end-to-end)")
    print(f"perceive+plan cycle: {cyc:.1f} ms  (incl. INT8 detector inference)")


if __name__ == "__main__":
    main()

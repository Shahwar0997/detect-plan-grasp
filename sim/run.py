"""
run.py — the closed loop: perceive -> plan -> execute -> verify, over many trials.

Headline experiment: randomize the object's position on the table, then compare the analytic
grasp planner against a naive baseline (grasp the table center regardless of where the object
is). Success = the object is lifted clear of the table. Reports grasp success rate and the
per-trial perception+planning cycle time.

Perception here uses the sim's ground-truth object position (oracle) to isolate grasp
planning/execution — the detect->lift pipeline was validated on real data in Days 3-4. The
detector-in-the-loop variant lives in run_detector.py.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim import Sim                                                       # noqa: E402
from grasp_planner import plan_grasp, plan_grasp_baseline, execute_grasp  # noqa: E402

PLANNERS = {"analytic": plan_grasp, "baseline": plan_grasp_baseline}


def run_trials(n: int, planner: str, seed: int = 0, verbose: bool = False):
    fn = PLANNERS[planner]
    rng = np.random.default_rng(seed)
    sim = Sim()
    successes, cycle_ms = 0, []
    for t in range(n):
        x = 0.5 + rng.uniform(-0.12, 0.12)          # randomize object on the table
        y = rng.uniform(-0.18, 0.18)
        sim.reset(obj_pos=(x, y, 0.335))
        for _ in range(150):
            mujoco.mj_step(sim.m, sim.d)            # settle
        obj_pos = sim.d.xpos[sim.obj].copy()        # oracle perception

        t0 = time.perf_counter()
        grasp = fn(obj_pos, sim)                     # plan (perception+planning timed)
        cycle_ms.append((time.perf_counter() - t0) * 1000)

        ok = bool(grasp is not None and execute_grasp(sim, grasp))
        successes += ok
        if verbose:
            print(f"  trial {t+1:2d}: obj=({x:.2f},{y:+.2f})  {'GRASP' if ok else 'miss '}")
    return successes / n, float(np.median(cycle_ms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"closed-loop grasp trials (n={args.trials}, randomized object position)\n")
    rows = []
    for planner in ("baseline", "analytic"):
        rate, cyc = run_trials(args.trials, planner, args.seed, args.verbose)
        rows.append((planner, rate, cyc))
        print(f"{planner:9s}: success {rate*100:5.1f}%   cycle {cyc:.2f} ms")
    print(f"\nanalytic planner: {rows[1][1]*100:.0f}% vs baseline {rows[0][1]*100:.0f}% "
          f"= +{(rows[1][1]-rows[0][1])*100:.0f} pts")


if __name__ == "__main__":
    main()

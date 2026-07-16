"""
plan_and_move.py — pick an object from bin 1, PLAN a collision-free path over the divider,
and place it in bin 2.

Ties the whole stack together: detect -> grasp -> lift out of the bin -> RRT-Connect transport
around the obstacle (planner.py) -> lower into bin 2 -> release -> verify the object arrived.
The transport is the planned part; a naive straight-line joint move would hit the divider.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rearrange import MultiSim, perceive_all, plan_grasp, NAME2CLASS, Detector, MODEL  # noqa: E402
from planner import Planner                                                            # noqa: E402

import mujoco

BINS = str(Path(__file__).resolve().parents[1] / "sim/franka/dpg_scene_bins.xml")
OVER_Z = 0.50                    # "just above a bin": clear of bin walls, below divider top (0.55)


def _rel_pose(sim, body_id):
    """Pose of a body in the hand frame (recorded at grasp, to keep it attached in transit)."""
    hR = sim.d.xmat[sim.hand].reshape(3, 3)
    return hR.T @ (sim.d.xpos[body_id] - sim.d.xpos[sim.hand]), hR.T @ sim.d.xmat[body_id].reshape(3, 3)


def _carry(sim, mesh, rel_p, rel_R):
    """Rigidly pin the grasped object to the hand (force-closure stand-in during planned motion)."""
    b = sim.m.body(f"obj_{mesh}")
    qadr = sim.m.jnt_qposadr[b.jntadr[0]]
    hR = sim.d.xmat[sim.hand].reshape(3, 3)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (hR @ rel_R).flatten())
    sim.d.qpos[qadr:qadr + 3] = sim.d.xpos[sim.hand] + hR @ rel_p
    sim.d.qpos[qadr + 3:qadr + 7] = quat
    sim.d.qvel[sim.m.jnt_dofadr[b.jntadr[0]]:sim.m.jnt_dofadr[b.jntadr[0]] + 6] = 0


def move_object_planned(sim: MultiSim, det: Detector, target_substr: str, dest_xy) -> dict:
    """Grasp target from bin1, RRT-plan a path over the divider, place at dest_xy in bin2.
    Returns a dict with each stage's result (grasp/plan/placed) — all verified."""
    percepts, _ = perceive_all(sim, det)
    hits = [w for c, w, _ in percepts if target_substr in c]
    if not hits:
        return {"detected": False}
    p = hits[0].copy()
    mesh = next((n for n, *_ in sim.objects if target_substr in NAME2CLASS.get(n, "")), None)
    z0 = sim.body_z(mesh)
    g = plan_grasp(p, sim)

    # 1. grasp in bin1, 2. lift out to the "above bin1" config
    sim.set_gripper(open_=True, steps=100)
    sim.reach([p[0], p[1], p[2] + 0.13], R_des=g.R, steps=600)
    sim.reach(p, R_des=g.R, steps=600)
    sim.set_gripper(open_=False, steps=500)
    q_over1 = sim.solve_ik([p[0], p[1], OVER_Z], g.R)
    sim.move_to(q_over1, 700)
    grasped = (sim.body_z(mesh) - z0) > 0.05
    if not grasped:
        return {"detected": True, "grasped": False}

    # 3. RRT-Connect transport over the divider to the "above bin2" config
    q_over2 = sim.solve_ik([dest_xy[0], dest_xy[1], OVER_Z], g.R)
    pl = Planner(sim, held_obj=mesh)
    naive_collides = pl.path_collides(q_over1, q_over2)
    path = pl.plan(q_over1, q_over2)
    if path is None:
        return {"detected": True, "grasped": True, "planned": False}
    rel = _rel_pose(sim, sim.m.body(f"obj_{mesh}").id)           # object pose in the hand frame
    sim.frame_hook = lambda: _carry(sim, mesh, *rel)            # keep it attached through transport
    for q in path:                                     # 4. execute the planned path
        sim.move_to(q, 300)
    sim.reach([dest_xy[0], dest_xy[1], p[2] + 0.03], R_des=g.R, steps=500)   # 5. lower into bin2

    # 6. release + settle + verify arrival
    sim.frame_hook = None
    sim.set_gripper(open_=True, steps=250)
    for _ in range(200):
        mujoco.mj_step(sim.m, sim.d)
    sim.home_arm()
    obj = sim.d.xpos[sim.m.body(f"obj_{mesh}").id]
    placed = bool(abs(obj[0] - dest_xy[0]) < 0.18 and abs(obj[1] - dest_xy[1]) < 0.18
                  and obj[2] > 0.30)
    return {"detected": True, "grasped": True, "naive_collides": naive_collides,
            "plan_waypoints": len(path), "placed": placed}


def run_bins_command(sim: MultiSim, det: Detector, command: str) -> dict:
    """Natural-language -> planned pick-place: parse the object, move it to the other bin."""
    from language import parse, _short
    from make_scene_bins import BIN2
    percepts, _ = perceive_all(sim, det)
    plan = parse(command, [_short(c) for c, _, _ in percepts])
    if not plan or not plan.get("object"):
        return {"ok": False, "reason": "could not parse object"}
    res = move_object_planned(sim, det, plan["object"], (BIN2[0], BIN2[1]))
    res["object"] = plan["object"]
    return res


if __name__ == "__main__":
    from make_scene_bins import OBJECTS
    command = " ".join(sys.argv[1:]) or "move the soup can to the other bin"
    sim = MultiSim(OBJECTS, scene=BINS)
    sim.settle()
    det = Detector(MODEL, conf=0.30)
    print(f'command: "{command}"')
    print(run_bins_command(sim, det, command))

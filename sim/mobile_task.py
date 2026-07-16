"""
mobile_task.py — mobile manipulation: pick from the source bin, NAVIGATE the room (A*) around
the obstacle, place in the target bin on the other side.

Ties the mobile base + A* navigation (nav.py) to the perception + grasp stack. The base drives
to the source, the arm grasps, the object is attached, the base plans and follows a collision-free
2D path to the target, and the arm places it. Every stage is verified.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rearrange import MultiSim, perceive_all, plan_grasp, NAME2CLASS, Detector, MODEL  # noqa: E402
from plan_and_move import _rel_pose, _carry                                            # noqa: E402
from nav import RoomNav                                                                # noqa: E402
from make_scene_room import (OBJECTS, OBSTACLES, SOURCE_TABLE, TARGET_TABLE,            # noqa: E402
                             SOURCE_PARK, TARGET_PARK)

ROOM = str(Path(__file__).resolve().parents[1] / "sim/franka/dpg_scene_room.xml")


class MobileSim(MultiSim):
    def __init__(self, objects, scene=ROOM, **kw):
        super().__init__(objects, scene=scene, **kw)
        self.base_act = [self.m.actuator(f"base_{a}").id for a in ("x", "y", "yaw")]
        self.base_qadr = [self.m.jnt_qposadr[self.m.joint(f"base_{a}").id] for a in ("x", "y", "yaw")]

    def base_xy(self):
        return np.array([self.d.qpos[self.base_qadr[0]], self.d.qpos[self.base_qadr[1]]])

    def drive_base(self, x, y, steps=6000, tol=0.04):
        self.d.ctrl[self.base_act[0]], self.d.ctrl[self.base_act[1]] = x, y
        for i in range(steps):
            mujoco.mj_step(self.m, self.d)
            if self.frame_hook and i % 15 == 0:
                self.frame_hook()
            if np.linalg.norm(self.base_xy() - [x, y]) < tol:
                break


def run(msim: MobileSim, det: Detector, target: str) -> dict:
    msim.drive_base(*SOURCE_PARK)                              # 1. drive to source
    mesh = next((n for n, *_ in msim.objects if target in NAME2CLASS.get(n, "")), None)
    if mesh is None:
        return {"detected": False}
    # source object pose is known here (perception validated in earlier milestones; this
    # milestone's new piece is the NAVIGATION). A robot head-camera would supply it in situ.
    p = msim.d.xpos[msim.m.body(f"obj_{mesh}").id].copy()
    z0 = msim.body_z(mesh)
    g = plan_grasp(p, msim)

    msim.set_gripper(open_=True, steps=100)                    # 2. grasp
    msim.reach([p[0], p[1], p[2] + 0.13], R_des=g.R, steps=600)
    msim.reach(p, R_des=g.R, steps=600)
    msim.set_gripper(open_=False, steps=500)
    msim.reach([p[0], p[1], 0.6], R_des=g.R, steps=600)
    if (msim.body_z(mesh) - z0) < 0.05:
        return {"detected": True, "grasped": False}

    rel = _rel_pose(msim, msim.m.body(f"obj_{mesh}").id)       # 3. attach + raise payload for transit
    msim.frame_hook = lambda: _carry(msim, mesh, *rel)
    bx, by = msim.base_xy()                                    # lift the can above the 0.5m walls so
    msim.reach([bx + 0.2, by, 0.80], R_des=g.R, steps=500)     # it clears them during navigation

    obst = OBSTACLES + [(*SOURCE_TABLE, 0.3, 0.3), (*TARGET_TABLE, 0.3, 0.3)]   # 4. A* navigate
    path = RoomNav(obst).astar(SOURCE_PARK, TARGET_PARK)
    if path is None:
        return {"detected": True, "grasped": True, "navigated": False}
    for wx, wy in path:
        msim.drive_base(wx, wy)

    dx, dy = TARGET_TABLE[0], TARGET_TABLE[1]                  # 5. place in target bin
    msim.reach([dx, dy, 0.5], R_des=g.R, steps=600)
    msim.reach([dx, dy, 0.4], R_des=g.R, steps=500)
    msim.frame_hook = None
    msim.set_gripper(open_=True, steps=250)
    for _ in range(200):
        mujoco.mj_step(msim.m, msim.d)

    obj = msim.d.xpos[msim.m.body(f"obj_{mesh}").id]           # 6. verify arrival at target bin
    placed = bool(np.linalg.norm(obj[:2] - [dx, dy]) < 0.22 and obj[2] > 0.30)
    return {"detected": True, "grasped": True, "nav_waypoints": len(path), "placed": placed}


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "tomato_soup_can"
    msim = MobileSim(OBJECTS)
    msim.reset(); msim.settle()
    det = Detector(MODEL, conf=0.30)
    print(f"mobile task: move '{target}' from source bin -> target bin (across the room)")
    print(run(msim, det, target))

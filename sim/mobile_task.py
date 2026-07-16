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
from rearrange import MultiSim, NAME2CLASS, Detector, MODEL                            # noqa: E402
from grasp_planner import grasp_candidates                                             # noqa: E402
from detector import YCB_CLASSES                                                       # noqa: E402
from lift_to_3d import back_project, estimate_position                                 # noqa: E402
from plan_and_move import _rel_pose, _carry                                            # noqa: E402
from nav import RoomNav                                                                # noqa: E402
from make_scene_room import (OBJECTS, OBSTACLES, SOURCE_TABLE, TARGET_TABLE,            # noqa: E402
                             SOURCE_PARK, TARGET_PARK)

ROOM = str(Path(__file__).resolve().parents[1] / "sim/franka/dpg_scene_room.xml")


SRC_CLASSES = set(NAME2CLASS.values())      # only accept real catalog objects, not stray detections


class MobileSim(MultiSim):
    def __init__(self, objects, scene=ROOM, **kw):
        super().__init__(objects, scene=scene, **kw)
        self.base_act = [self.m.actuator(f"base_{a}").id for a in ("x", "y", "yaw")]
        self.base_qadr = [self.m.jnt_qposadr[self.m.joint(f"base_{a}").id] for a in ("x", "y", "yaw")]
        self.src_cam = self.m.camera("srccam").id
        for fb in ("left_finger", "right_finger"):            # finger pads: condim=6 so the grip
            bid = self.m.body(fb).id                          # resists rotation and a grasped object
            for gi in range(self.m.ngeom):                    # is carried upright, not tilted out
                if self.m.geom_bodyid[gi] == bid:
                    self.m.geom_condim[gi] = 6

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


def perceive_source(msim: MobileSim, det: Detector):
    """Detect the objects in the SOURCE bin via its overhead-oblique shelf camera. For each, back-
    project the detection to a 3D centroid AND a world-frame surface point cloud (used to align the
    grasp to the object's narrow axis). Keeps only catalog objects inside the bin. -> [(cls,w,d,pts)].
    """
    saved_cam, msim.cam = msim.cam, msim.src_cam
    rgb, depth, K = msim.render(), msim.render_depth(), msim.intrinsics()
    cam_xpos = msim.d.cam_xpos[msim.cam].copy()
    cam_xmat = msim.d.cam_xmat[msim.cam].reshape(3, 3).copy()
    msim.cam = saved_cam
    sx, sy = SOURCE_TABLE
    out = []
    for d0 in det.detect(rgb[:, :, ::-1]):
        pts = back_project(depth, (d0.x1, d0.y1, d0.x2, d0.y2), K)     # camera-frame (CV) metres
        c = estimate_position(pts)
        if c is None:
            continue
        w = cam_xpos + cam_xmat @ np.array([c[0], -c[1], -c[2]])       # world centroid
        cls = YCB_CLASSES[d0.cls]                                     # keep detections on the shelf
        if cls.split("_", 1)[1] not in SRC_CLASSES or abs(w[0] - sx) > 0.32 or abs(w[1] - sy) > 0.32:
            continue
        pts_world = cam_xpos + (pts * [1, -1, -1]) @ cam_xmat.T        # world-frame surface points
        out.append((cls, w, d0, pts_world))
    return out


def grasp_verified(msim: MobileSim, p, pts, mesh) -> bool:
    """Try ranked grasp candidates (narrow-axis first); keep the FIRST that actually lifts the
    object > 5 cm, probing each on a saved sim state so a failed attempt costs nothing. Gripper
    dynamics eject the object at some yaws even when reachable, so this falls through to the next.
    Leaves the object grasped and raised to z=0.6 on success. -> True if grasped."""
    for g in grasp_candidates(p, msim, pts_world=pts):
        save = (msim.d.qpos.copy(), msim.d.qvel.copy(), msim.d.ctrl.copy())
        z0 = msim.body_z(mesh)
        msim.set_gripper(open_=True, steps=100)
        msim.reach([p[0], p[1], p[2] + 0.13], R_des=g.R, steps=500)
        msim.reach(p, R_des=g.R, steps=500)
        msim.set_gripper(open_=False, steps=500)
        msim.reach([p[0], p[1], 0.6], R_des=g.R, steps=500)
        if msim.body_z(mesh) - z0 > 0.05:
            msim.grasp_R = g.R
            return True
        msim.d.qpos[:], msim.d.qvel[:], msim.d.ctrl[:] = save   # restore and try the next yaw
        mujoco.mj_forward(msim.m, msim.d)
    return False


def run(msim: MobileSim, det: Detector, target: str) -> dict:
    percepts = perceive_source(msim, det)                     # 1. DETECT the target in the source bin
    hits = [(w, d.conf, pts) for cls, w, d, pts in percepts if target in cls]
    if not hits:
        return {"detected": False, "seen": [c for c, _, _, _ in percepts]}
    p, conf, pts = hits[0]
    p = p.copy()                                              # detector-localized grasp pose
    mesh = next((n for n, *_ in msim.objects if target in NAME2CLASS.get(n, "")), None)

    msim.drive_base(*SOURCE_PARK)                             # 2. drive to source, grasp what we detected
    if not grasp_verified(msim, p, pts, mesh):               # try candidate grasps until one lifts
        return {"detected": True, "conf": round(conf, 2), "grasped": False}
    g = type("G", (), {"R": msim.grasp_R})                   # the grasp orientation that worked

    rel = _rel_pose(msim, msim.m.body(f"obj_{mesh}").id)       # 3. attach + raise payload for transit
    msim.frame_hook = lambda: _carry(msim, mesh, *rel)
    bx, by = msim.base_xy()                                    # lift the can above the 0.5m walls so
    msim.reach([bx + 0.2, by, 0.80], R_des=g.R, steps=500)     # it clears them during navigation

    obst = OBSTACLES + [(*SOURCE_TABLE, 0.25, 0.25), (*TARGET_TABLE, 0.25, 0.25)]   # 4. A* navigate
    path = RoomNav(obst, inflate=0.20).astar(SOURCE_PARK, TARGET_PARK)
    if path is None:
        return {"detected": True, "grasped": True, "navigated": False}
    for wx, wy in path:
        msim.drive_base(wx, wy)

    dx, dy = TARGET_TABLE[0], TARGET_TABLE[1]                  # 5. place in target bin
    msim.reach([dx, dy, 0.55], R_des=g.R, steps=600)
    msim.reach([dx, dy, 0.46], R_des=g.R, steps=500)           # lower to just above the bin floor
    msim.frame_hook = None
    msim.set_gripper(open_=True, steps=250)                    # release; it settles upright
    for _ in range(250):
        mujoco.mj_step(msim.m, msim.d)

    obj = msim.d.xpos[msim.m.body(f"obj_{mesh}").id]           # 6. verify arrival at target bin
    placed = bool(np.linalg.norm(obj[:2] - [dx, dy]) < 0.22 and obj[2] > 0.30)
    return {"detected": True, "conf": round(conf, 2), "grasped": True,
            "nav_waypoints": len(path), "placed": placed}


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "tomato_soup_can"
    msim = MobileSim(OBJECTS)
    msim.reset(); msim.settle()
    det = Detector(MODEL, conf=0.30)
    print(f"mobile task: move '{target}' from source bin -> target bin (across the room)")
    print(run(msim, det, target))

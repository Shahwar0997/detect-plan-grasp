"""
store_task.py — Step 3: the end-to-end, prompt-driven store run.

One spoken command drives the whole system end to end:
    parse (store_language, LLM-grounded) -> drive to the source shelf -> detect the commanded object
    on that shelf -> verified narrow-axis grasp -> A* navigate around the shelves to the destination
    -> place it upright on the destination shelf.

This is where milestones 01-07 run as one system. Almost everything is reused: the INT8 detector,
back-projection, the mobile base + A* nav, the narrow-axis verified grasp, the condim=6 upright-carry
fix, and the LLM parser. The new glue is StoreSim (per-shelf cameras, object placement, and facing
the base toward whichever shelf it works at).

    python sim/store_task.py "take the mustard from shelf C to shelf B"
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mobile_task import MobileSim, grasp_verified                                       # noqa: E402
from rearrange import Detector, MODEL                                                   # noqa: E402
from detector import YCB_CLASSES                                                        # noqa: E402
from lift_to_3d import back_project, estimate_position                                  # noqa: E402
from plan_and_move import _rel_pose, _carry                                             # noqa: E402
from nav import RoomNav                                                                 # noqa: E402
import store_language as SL                                                             # noqa: E402
from make_scene_store import SHELVES, OBJECTS, OBJ_Z, SHELF_HX, SHELF_HY, SHELF_TOP, shelf_park

STORE = str(Path(__file__).resolve().parents[1] / "sim/franka/dpg_scene_store.xml")
STORE_CLASSES = set(SL.MESH2CLASS.values())
CLASS2MESH = {c: m for m, c in SL.MESH2CLASS.items()}


class StoreSim(MobileSim):
    def __init__(self, scene=STORE, **kw):
        # (body_suffix, mesh, shelf, x, y, z, yaw); the body is obj_<suffix>
        self.placements = []
        for i, (mesh, shelf, dx) in enumerate(OBJECTS):
            cx, cy, side = SHELVES[shelf]
            yaw = 0.0 if side < 0 else np.pi                       # face the shelf's camera
            self.placements.append((f"{shelf}_{mesh}_{i}", mesh, shelf,
                                    cx + dx, cy + side * 0.04, OBJ_Z, yaw))
        super().__init__(objects=[], scene=scene, **kw)
        self.shelf_cam = {s: self.m.camera(f"cam_{s}").id for s in SHELVES}

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("home").id)
        for suffix, mesh, shelf, x, y, z, yaw in self.placements:
            adr = self.m.jnt_qposadr[self.m.body(f"obj_{suffix}").jntadr[0]]
            self.d.qpos[adr:adr + 7] = [x, y, z, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.d.ctrl[:] = self.m.key("home").ctrl
        mujoco.mj_forward(self.m, self.d)
        # arm pose for travelling: raised & compact (hand ~0.25 fwd, 0.82 up in base frame) so it
        # clears the shelf objects instead of the folded home arm plowing through them on approach
        self.q_travel = self.solve_ik(np.array([0.25, 0.0, 0.82]), self.DOWN)

    def grasp_park(self, shelf, x):
        _, cy, side = SHELVES[shelf]
        return (x, cy + side * 0.45)                           # park directly in front of a column

    def clear_spot(self, shelf, extra=()):
        """An x on `shelf` clear of the objects already there (so a placed object doesn't knock one).

        `extra` = offsets the robot has already used on this shelf. Without it, a multi-object task
        ("put all the cans on D") drops every object on the same spot: the layout below is the
        *spawn* layout, so it cannot see what the robot itself put there.
        """
        cx, cy, side = SHELVES[shelf]
        # Placements go in the BACK row. The spawn row is full — 3 objects across 0.76 m — so the
        # best free front-row slot left only ~0.11 m of clearance, and a wide flat can clipped its
        # neighbour and tipped over (reproducibly). The back row is empty and still in reach.
        taken = list(extra)
        dx = max([-0.20, 0.0, 0.20], key=lambda c: min([abs(c - t) for t in taken], default=9.0))
        return (cx + dx, cy - side * 0.10)

    def drive_to(self, x, y, yaw=None, steps=6000, tol=0.05):
        self.d.ctrl[self.base_act[0]], self.d.ctrl[self.base_act[1]] = x, y
        if yaw is not None:
            self.d.ctrl[self.base_act[2]] = yaw
        for i in range(steps):
            mujoco.mj_step(self.m, self.d)
            if self.frame_hook and i % 15 == 0:
                self.frame_hook()
            if np.linalg.norm(self.base_xy() - [x, y]) < tol:
                break

    def face_shelf_yaw(self, shelf):
        _, _, side = SHELVES[shelf]
        return -side * np.pi / 2                                   # base +x points at the shelf

    def perceive_shelf(self, det, shelf):
        """Detect catalog objects on `shelf` from its fixed camera; back-project each to a world
        centroid + point cloud, keeping only detections on that shelf. -> [(cls, w, det, pts)]."""
        cx, cy, _ = SHELVES[shelf]
        saved, self.cam = self.cam, self.shelf_cam[shelf]
        rgb, depth, K = self.render(), self.render_depth(), self.intrinsics()
        cam_xpos = self.d.cam_xpos[self.cam].copy()
        cam_xmat = self.d.cam_xmat[self.cam].reshape(3, 3).copy()
        self.cam = saved
        out = []
        for d0 in det.detect(rgb[:, :, ::-1]):
            pts = back_project(depth, (d0.x1, d0.y1, d0.x2, d0.y2), K)
            c = estimate_position(pts)
            if c is None:
                continue
            w = cam_xpos + cam_xmat @ np.array([c[0], -c[1], -c[2]])
            cls = YCB_CLASSES[d0.cls]
            if cls.split("_", 1)[1] not in STORE_CLASSES or abs(w[0]-cx) > 0.45 or abs(w[1]-cy) > 0.40:
                continue
            pts_world = cam_xpos + (pts * [1, -1, -1]) @ cam_xmat.T
            out.append((cls, w, d0, pts_world))
        return out


def _obstacles():
    return [(cx, cy, SHELF_HX, SHELF_HY) for cx, cy, _ in SHELVES.values()]


def run(msim: StoreSim, det: Detector, command: str) -> dict:
    task = SL.parse(command)                                       # 0. understand the command
    if task is None:
        return {"command": command, "parsed": None}
    if not task.get("valid"):
        return {"command": command, "task": task}                 # grounded but not doable
    res = SL.resolve(task)
    tgt_class, source, dest = res["target_class"], task["source"], task["dest"]

    percepts = msim.perceive_shelf(det, source)                   # 1. detect the object on the source
    hits = [(w, d.conf, pts) for cls, w, d, pts in percepts if tgt_class in cls]   #    shelf (from afar,
    if not hits:                                                  #    before the base can occlude it)
        return {"task": task, "detected": False, "seen": [c for c, *_ in percepts]}
    p, conf, pts = hits[0]
    p = p.copy()
    mesh = CLASS2MESH[tgt_class]
    suffix = next(s for s, m, sh, *_ in msim.placements if sh == source and m == mesh)

    src_park = msim.grasp_park(source, p[0])                      # 2. tuck arm, drive up aligned+facing
    msim.move_to(msim.q_travel, steps=300)                       #    (tucked so it clears the shelf)
    msim.drive_to(*src_park, yaw=msim.face_shelf_yaw(source))
    if not grasp_verified(msim, p, pts, suffix):                  # 3. grasp what we detected
        return {"task": task, "detected": True, "conf": round(conf, 2), "grasped": False}
    gR = msim.grasp_R

    rel = _rel_pose(msim, msim.m.body(f"obj_{suffix}").id)         # 4. attach + tuck payload up high
    msim.frame_hook = lambda: _carry(msim, suffix, *rel)
    bx, by = msim.base_xy()
    msim.reach([bx, by, 0.78], R_des=gR, steps=500)               # over the base, clear of shelves

    px, py = msim.clear_spot(dest)                               # 5. A* navigate to a clear spot on
    dcx, dcy, dside = SHELVES[dest]                              #    the destination shelf
    dst_park = (px, dcy + dside * 0.45)
    path = RoomNav(_obstacles(), xlim=(-2.0, 2.0), ylim=(-2.0, 2.0),
                   inflate=0.22).astar(src_park, dst_park)
    if path is None:
        return {"task": task, "grasped": True, "navigated": False}
    for wx, wy in path:
        msim.drive_to(wx, wy)

    msim.drive_to(*dst_park, yaw=msim.face_shelf_yaw(dest))       # 6. face dest, place upright at spot
    msim.reach([px, py, 0.72], R_des=msim.DOWN, steps=500)       #    fresh top-down orientation (gR was
    msim.reach([px, py, SHELF_TOP + 0.13], R_des=msim.DOWN, steps=450)   #  captured facing the source)
    msim.frame_hook = None
    msim.set_gripper(open_=True, steps=250)
    for _ in range(250):
        mujoco.mj_step(msim.m, msim.d)

    o = msim.d.xpos[msim.m.body(f"obj_{suffix}").id]              # 7. verify: on the dest shelf, upright
    R = msim.d.xmat[msim.m.body(f"obj_{suffix}").id].reshape(3, 3)
    tilt = float(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))
    placed = bool(abs(o[0]-px) < 0.20 and abs(o[1]-py) < 0.18 and o[2] > SHELF_TOP - 0.02)
    return {"task": {"object": task["object"], "source": source, "dest": dest},
            "conf": round(conf, 2), "grasped": True, "nav_waypoints": len(path),
            "placed": placed, "tilt_deg": round(tilt)}


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "take the mustard from shelf C to shelf B"
    msim = StoreSim()
    msim.reset()
    msim.settle()
    det = Detector(MODEL, conf=0.30)
    print(f'command: "{command}"')
    print(run(msim, det, command))

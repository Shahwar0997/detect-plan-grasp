"""
rearrange.py — multi-object perception + grasping the object you *name*.

MultiSim loads a cluttered YCB scene and places the objects. `perceive_all` renders RGB-D,
runs the detector, and lifts every detection to a 3D world position. `grasp_named` picks the
object of a given class and grasps it — with a lift-high traversal so the arm doesn't sweep
through its neighbours. This is the bridge from "detect a table" to selective manipulation,
and the substrate for language-driven rearrangement.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sim import Sim                                        # noqa: E402
from detector import Detector, YCB_CLASSES                 # noqa: E402
from lift_to_3d import back_project, estimate_position     # noqa: E402
from grasp_planner import plan_grasp                       # noqa: E402

MULTI_SCENE = str(Path(__file__).resolve().parents[1] / "sim/franka/dpg_scene_multi.xml")
MODEL = str(Path(__file__).resolve().parents[1] / "runs/ycb_artifacts/best_int8.onnx")

# mesh name (body obj_<name>) -> detector class, so we can verify the right object was lifted
NAME2CLASS = {"soup": "tomato_soup_can", "mustard": "mustard_bottle", "mug": "mug",
              "spam": "potted_meat_can", "sugar": "sugar_box", "gelatin": "gelatin_box",
              "chef": "master_chef_can", "cracker": "cracker_box", "bleach": "bleach_cleanser"}


class MultiSim(Sim):
    """A Sim over a scene with several free-jointed objects named obj_<name>."""
    def __init__(self, objects, scene=MULTI_SCENE, **kw):
        self.objects = objects                             # [(name, x, y, z), ...]
        super().__init__(scene=scene, **kw)

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("home").id)
        for name, x, y, z in self.objects:
            adr = self.m.jnt_qposadr[self.m.body(f"obj_{name}").jntadr[0]]
            self.d.qpos[adr:adr + 7] = [x, y, z, 1, 0, 0, 0]
        self.d.ctrl[:] = self.m.key("home").ctrl
        mujoco.mj_forward(self.m, self.d)

    def settle(self, steps=250):
        for _ in range(steps):
            mujoco.mj_step(self.m, self.d)

    def body_z(self, name) -> float:
        return float(self.d.xpos[self.m.body(f"obj_{name}").id][2])


def perceive_all(sim: MultiSim, det: Detector):
    """Detect every object and lift each to a 3D world position. -> [(cls_name, world, det)]."""
    rgb, depth = sim.render(), sim.render_depth()
    out = []
    for d0 in det.detect(rgb[:, :, ::-1]):
        c = estimate_position(back_project(depth, (d0.x1, d0.y1, d0.x2, d0.y2), sim.intrinsics()))
        if c is None:
            continue
        out.append((YCB_CLASSES[d0.cls], sim.world_from_cam(c), d0))
    return out, rgb


def grasp_named(sim: MultiSim, det: Detector, target_substr: str,
                place=None, lift_z=0.52) -> bool:
    """Detect, find the object whose class contains `target_substr`, grasp it (optionally place
    at `place`=(x,y)), and return whether it was ACTUALLY lifted (verified against the object's
    body height, not just 'the motion ran'). High traversal clears the other objects."""
    percepts, _ = perceive_all(sim, det)
    hits = [(cls, w) for cls, w, _ in percepts if target_substr in cls]
    if not hits:
        return False
    p = hits[0][1].copy()
    mesh = next((n for n, *_ in sim.objects if target_substr in NAME2CLASS.get(n, "")), None)
    z0 = sim.body_z(mesh) if mesh else None
    g = plan_grasp(p, sim)
    if g is None:
        return False

    sim.set_gripper(open_=True, steps=100)
    sim.reach([p[0], p[1], p[2] + 0.13], R_des=g.R, steps=600)        # descend over target
    sim.reach(p, R_des=g.R, steps=600)                               # settle at grasp pose
    sim.set_gripper(open_=False, steps=500)                          # close
    sim.reach([p[0], p[1], lift_z], R_des=g.R, steps=600)            # lift high (clear others)

    lifted = mesh is not None and (sim.body_z(mesh) - z0) > 0.05     # VERIFY it rose
    if place is not None and lifted:
        sim.reach([place[0], place[1], lift_z], R_des=g.R, steps=700)         # traverse high
        sim.reach([place[0], place[1], p[2] + 0.01], R_des=g.R, steps=500)    # lower
        sim.set_gripper(open_=True, steps=250)                                # release
        sim.reach([place[0], place[1], lift_z], R_des=g.R, steps=400)
    sim.home_arm()                    # clear the camera view for the next perception
    return lifted


if __name__ == "__main__":
    import sys as _sys
    from make_multi_scene import OBJECTS
    target = _sys.argv[1] if len(_sys.argv) > 1 else "tomato_soup_can"
    sim = MultiSim(OBJECTS)
    sim.settle()
    det = Detector(MODEL, conf=0.30)
    percepts, _ = perceive_all(sim, det)
    print("on the table:", [f"{c.split('_',1)[1]} @({w[0]:.2f},{w[1]:.2f})" for c, w, _ in percepts])
    print(f"grasping the one containing '{target}'...")
    z0 = {c: w for c, w, _ in percepts}
    ok = grasp_named(sim, det, target)
    print("grasp success:", ok)

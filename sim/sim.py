"""
sim.py — MuJoCo grasp environment: Franka Panda + table + object + RGB-D camera.

Provides the interface the grasp loop needs: reset, render RGB-D (+ intrinsics),
solve inverse kinematics (damped least squares over the 7 arm joints), drive the arm,
open/close the gripper, and read the object height (the grasp-success signal).

MuJoCo has no one-call IK (unlike PyBullet), so IK is implemented here with the analytic
end-effector Jacobian — a cleaner thing to be able to explain than a black-box solver.
"""
from __future__ import annotations
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCENE = REPO / "sim" / "franka" / "dpg_scene.xml"
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
GRASP_OFFSET = np.array([0.0, 0.0, 0.103])   # hand frame -> point between the fingertips


class Sim:
    def __init__(self, scene: Path = SCENE, render_hw=(480, 640)):
        self.m = mujoco.MjModel.from_xml_path(str(scene))
        self.d = mujoco.MjData(self.m)
        self.hand = self.m.body("hand").id
        # single-object scenes have a body named "object"; multi-object scenes don't
        try:
            self.obj = self.m.body("object").id
            self.obj_qadr = self.m.jnt_qposadr[self.m.body("object").jntadr[0]]
        except KeyError:
            self.obj = self.obj_qadr = None
        self.arm_qadr = np.array([self.m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.arm_dof = np.array([self.m.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.arm_act = np.array([self.m.actuator(f"actuator{i}").id for i in range(1, 8)])
        self.grip_act = self.m.actuator("actuator8").id
        self.cam = self.m.camera("cam").id
        self.renderer = mujoco.Renderer(self.m, *render_hw)
        self.reset()

    # ---- state --------------------------------------------------------------
    def reset(self, obj_pos=(0.5, 0.0, 0.34)):
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("home").id)
        self.d.qpos[self.obj_qadr:self.obj_qadr + 7] = [*obj_pos, 1, 0, 0, 0]
        self.d.ctrl[:] = self.m.key("home").ctrl
        mujoco.mj_forward(self.m, self.d)

    def grasp_point(self) -> np.ndarray:
        """World position of the point between the fingertips."""
        return self.d.xpos[self.hand] + self.d.xmat[self.hand].reshape(3, 3) @ GRASP_OFFSET

    def object_z(self) -> float:
        return float(self.d.xpos[self.obj][2])

    # ---- inverse kinematics (6-DOF damped least squares) --------------------
    # Default orientation = gripper pointing straight down (hand z -> world -z), so the
    # fingers descend onto the object and close horizontally around it (top-down grasp).
    DOWN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)

    def solve_ik(self, target: np.ndarray, R_des=None, iters: int = 300,
                 tol: float = 1e-3) -> np.ndarray:
        """7 arm-joint angles putting the grasp point at `target` with orientation R_des.
        Solves position + orientation via the stacked end-effector Jacobian. Non-destructive."""
        if R_des is None:
            R_des = self.DOWN
        saved = self.d.qpos.copy()
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            mujoco.mj_forward(self.m, self.d)
            gp = self.grasp_point()
            R = self.d.xmat[self.hand].reshape(3, 3)
            p_err = target - gp
            r_err = 0.5 * sum(np.cross(R[:, i], R_des[:, i]) for i in range(3))
            if np.linalg.norm(p_err) < tol and np.linalg.norm(r_err) < 0.02:
                break
            mujoco.mj_jac(self.m, self.d, jacp, jacr, gp, self.hand)
            J = np.vstack([jacp[:, self.arm_dof], jacr[:, self.arm_dof]])   # 6x7
            err = np.concatenate([p_err, r_err])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            self.d.qpos[self.arm_qadr] += np.clip(dq, -0.3, 0.3)
        q = self.d.qpos[self.arm_qadr].copy()
        self.d.qpos[:] = saved
        mujoco.mj_forward(self.m, self.d)
        return q

    # ---- actuation ----------------------------------------------------------
    frame_hook = None          # optional callable() invoked during motion (for rendering demos)

    def move_to(self, q: np.ndarray, steps: int = 800):
        self.d.ctrl[self.arm_act] = q
        for i in range(steps):
            mujoco.mj_step(self.m, self.d)
            if self.frame_hook and i % 15 == 0:
                self.frame_hook()

    def set_gripper(self, open_: bool, steps: int = 300):
        self.d.ctrl[self.grip_act] = 255 if open_ else 0
        for i in range(steps):
            mujoco.mj_step(self.m, self.d)
            if self.frame_hook and i % 15 == 0:
                self.frame_hook()

    def reach(self, target: np.ndarray, R_des=None, steps: int = 800) -> float:
        """Solve IK to target (with optional orientation) and drive the arm there."""
        self.move_to(self.solve_ik(np.asarray(target), R_des), steps)
        return float(np.linalg.norm(np.asarray(target) - self.grasp_point()))

    def home_arm(self, steps: int = 500):
        """Return the arm to the 'home' joint pose (folded up, out of the camera's view)."""
        self.move_to(self.m.key("home").qpos[self.arm_qadr], steps)

    # ---- perception (for the closed loop, Day 6) ----------------------------
    def intrinsics(self):
        h, w = self.renderer.height, self.renderer.width
        fovy = np.deg2rad(self.m.cam_fovy[self.cam])
        fy = (h / 2) / np.tan(fovy / 2)
        return (fy, fy, w / 2, h / 2)       # fx, fy, cx, cy (square pixels)

    def world_from_cam(self, c_cv: np.ndarray) -> np.ndarray:
        """Map a CV camera-frame point (x right, y down, z forward) to world coordinates.
        MuJoCo's camera frame is (x right, y up, z back), hence the [X, -Y, -Z] flip."""
        c_mj = np.array([c_cv[0], -c_cv[1], -c_cv[2]])
        return self.d.cam_xpos[self.cam] + self.d.cam_xmat[self.cam].reshape(3, 3) @ c_mj

    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.d, camera=self.cam)
        return self.renderer.render()

    def render_depth(self) -> np.ndarray:
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.d, camera=self.cam)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return depth

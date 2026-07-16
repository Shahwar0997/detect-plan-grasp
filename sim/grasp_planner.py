"""
grasp_planner.py — analytic top-down grasp planning + execution.

Given an object's 3D position (from the perception half: detect -> lift), propose candidate
grasps, score them by reachability and centering, pick the best, and execute the
approach -> descend -> close -> lift sequence. Deliberately analytic and inspectable — a
planner you can explain and measure, and a clean baseline to compare against.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

DOWN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)   # gripper pointing straight down


def _yaw(theta: float) -> np.ndarray:
    """Top-down orientation rotated by `theta` about the vertical (which way fingers close)."""
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Rz @ DOWN


@dataclass
class Grasp:
    pos: np.ndarray                 # grasp point = object centroid (X, Y, Z)
    R: np.ndarray                   # gripper approach orientation
    score: float = 0.0
    pre_height: float = 0.12        # pre-grasp standoff above the object


GRIPPER_MAX = 0.075     # Panda parallel-jaw max opening (m); object must fit across the fingers


def grasp_candidates(object_pos, sim, pts_world=None, n_yaw: int = 12) -> list[Grasp]:
    """Top-down grasp candidates over a range of yaws, ranked best-first by IK reachability and (if
    the object's point cloud is given) how well it fits between the fingers.

    With `pts_world` (world-frame surface points, from perception) the score aligns the fingers
    ACROSS the object's NARROW axis and rejects yaws where the object is wider than the gripper —
    which is what lets it grasp boxes and bottles, not just cans. Returning a ranked *list* (not
    just the best) lets the caller verify grasps in sim and fall through to the next candidate,
    since gripper dynamics make some yaws eject the object even when reachable.
    """
    p = np.asarray(object_pos, float)
    sl = None
    if pts_world is not None and len(pts_world) >= 10:
        pts_world = np.asarray(pts_world)
        sl = pts_world[np.abs(pts_world[:, 2] - p[2]) < 0.03][:, :2]   # cross-section at grasp height
        if len(sl) < 10:
            sl = pts_world[:, :2]
    cands = []
    for theta in np.linspace(0, np.pi, n_yaw, endpoint=False):     # 0..180deg finger azimuths
        R = _yaw(theta)
        q = sim.solve_ik(p, R)                            # non-destructive IK probe
        resid = _ik_residual(sim, q, p, R)
        if resid > 0.02:                                  # unreachable at this yaw
            continue
        fit = 0.0
        if sl is not None:
            close_dir = R[:2, 1] / (np.linalg.norm(R[:2, 1]) + 1e-9)   # world finger-closing axis
            width = float((sl @ close_dir).ptp())                     # object extent across fingers
            if width > 0.11:                              # physically can't grip this axis at all
                continue
            fit = 1.0 - width / 0.11                      # prefer the NARROW axis (min width)
        cands.append(Grasp(pos=p, R=R, score=(1.0 - resid) + fit))
    return sorted(cands, key=lambda g: g.score, reverse=True)


def plan_grasp(object_pos, sim, pts_world=None, n_yaw: int = 12) -> Grasp | None:
    """Best single top-down grasp (see grasp_candidates). None if nothing is reachable / fits."""
    cands = grasp_candidates(object_pos, sim, pts_world, n_yaw)
    return cands[0] if cands else None


def _ik_residual(sim, q, target, R) -> float:
    """Forward-kinematics residual of an IK solution (probe on a saved state)."""
    saved = sim.d.qpos.copy()
    sim.d.qpos[sim.arm_qadr] = q
    import mujoco
    mujoco.mj_forward(sim.m, sim.d)
    gp = sim.grasp_point()
    Rc = sim.d.xmat[sim.hand].reshape(3, 3)
    p_err = np.linalg.norm(target - gp)
    r_err = np.linalg.norm(0.5 * sum(np.cross(Rc[:, i], R[:, i]) for i in range(3)))
    sim.d.qpos[:] = saved
    mujoco.mj_forward(sim.m, sim.d)
    return p_err + 0.15 * r_err          # weight orientation: reject tilted (non-top-down) grasps


def execute_grasp(sim, grasp: Grasp, lift_h: float = 0.2) -> bool:
    """Run the grasp; return True if the object rose clear of the table."""
    p = grasp.pos
    sim.set_gripper(open_=True, steps=120)
    sim.reach([p[0], p[1], p[2] + grasp.pre_height], steps=600)   # pre-grasp
    sim.reach(p, R_des=grasp.R, steps=600)                        # descend to centroid
    sim.set_gripper(open_=False, steps=500)                       # close on object
    z0 = sim.object_z()
    for dz in np.linspace(0.06, lift_h, 4):                       # gentle staged lift
        sim.reach([p[0], p[1], p[2] + dz], R_des=grasp.R, steps=400)
    return sim.object_z() - z0 > 0.08                            # rose > 8 cm = success


# Baseline for Day 6's comparison: ignore the perceived 3D, grasp at a fixed table-center
# guess. Succeeds only when the object happens to sit there.
def plan_grasp_baseline(object_pos, sim, table_center=(0.5, 0.0, 0.335)) -> Grasp:
    return Grasp(pos=np.asarray(table_center, float), R=DOWN, score=1.0)


# `sim.reach` needs to accept R_des; ensure that in sim.py (it does via solve_ik(R_des=...)).

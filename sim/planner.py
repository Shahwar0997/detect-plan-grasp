"""
planner.py — RRT-Connect collision-free motion planning for the 7-DOF arm.

Plans a joint-space path from a start configuration to a goal configuration that avoids the
scene obstacles (the divider, bins, table, other objects), checked with MuJoCo's own collision
detection. This is the "path finding" piece: a naive straight-line joint interpolation drives
the arm through the divider; RRT-Connect routes around it. (Collision is checked for the arm +
gripper; the small held object rides with the gripper.)
"""
from __future__ import annotations

import numpy as np
import mujoco

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]


class Planner:
    def __init__(self, sim, held_obj: str | None = None, pad: float = 0.0):
        self.m, self.d = sim.m, sim.d
        self.arm_qadr = sim.arm_qadr
        rng = np.array([self.m.joint(j).range for j in ARM_JOINTS])
        self.lo, self.hi = rng[:, 0], rng[:, 1]
        robot_bodies = {i for i in range(self.m.nbody)
                        if any(k in self.m.body(i).name for k in ("link", "hand", "finger"))}
        self.robot_geoms = {g for g in range(self.m.ngeom)
                            if self.m.geom_bodyid[g] in robot_bodies}
        self.held_geom = self.m.geom(f"geom_{held_obj}").id if held_obj else -1

    # ---- collision -----------------------------------------------------------
    def collision_free(self, q: np.ndarray) -> bool:
        saved = self.d.qpos.copy()
        self.d.qpos[self.arm_qadr] = q
        mujoco.mj_forward(self.m, self.d)
        ok = True
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            r1, r2 = c.geom1 in self.robot_geoms, c.geom2 in self.robot_geoms
            if r1 ^ r2:                                   # robot vs environment
                other = c.geom2 if r1 else c.geom1
                if other != self.held_geom and c.dist < -0.002:
                    ok = False
                    break
        self.d.qpos[:] = saved
        mujoco.mj_forward(self.m, self.d)
        return ok

    def path_collides(self, q0, q1, step=0.05) -> bool:
        """Does the straight-line joint interpolation q0->q1 hit anything? (naive baseline)"""
        n = int(np.linalg.norm(q1 - q0) / step) + 1
        return any(not self.collision_free(q0 + (q1 - q0) * k / n) for k in range(n + 1))

    # ---- RRT-Connect ---------------------------------------------------------
    def plan(self, q0, q1, max_iter=4000, step=0.12, seed=0):
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        if not (self.collision_free(q0) and self.collision_free(q1)):
            return None
        rng = np.random.default_rng(seed)
        A, Ap = [q0], [-1]                                # tree from start
        B, Bp = [q1], [-1]                                # tree from goal

        def nearest(T, q):
            return int(np.argmin([np.sum((np.asarray(t) - q) ** 2) for t in T]))

        def connect(T, Tp, q):                            # greedily extend T toward q
            reached = False
            while True:
                i = nearest(T, q)
                d = q - T[i]
                dist = np.linalg.norm(d)
                qn = q if dist < step else T[i] + d / dist * step
                if not self.collision_free(qn):
                    return None
                T.append(qn); Tp.append(i)
                if dist < step:
                    reached = True
                    return len(T) - 1
                # keep going toward q (CONNECT)

        for _ in range(max_iter):
            qr = rng.uniform(self.lo, self.hi)
            i = nearest(A, qr)
            d = qr - A[i]; dist = np.linalg.norm(d)
            qn = A[i] + d / max(dist, 1e-9) * min(step, dist)
            if not self.collision_free(qn):
                A, Ap, B, Bp = B, Bp, A, Ap
                continue
            A.append(qn); Ap.append(i)
            j = connect(B, Bp, qn)                        # try to link the other tree
            if j is not None:
                pa, ia = [], len(A) - 1
                while ia != -1:
                    pa.append(A[ia]); ia = Ap[ia]
                pb, ib = [], j
                while ib != -1:
                    pb.append(B[ib]); ib = Bp[ib]
                path = pa[::-1] + pb
                # orient start->goal (A may be the goal tree after swaps)
                if np.linalg.norm(path[0] - q0) > np.linalg.norm(path[-1] - q0):
                    path = path[::-1]
                return self._shortcut(path, rng)
            A, Ap, B, Bp = B, Bp, A, Ap                   # swap trees
        return None

    def _shortcut(self, path, rng, iters=100):
        path = [np.asarray(p) for p in path]
        for _ in range(iters):
            if len(path) < 3:
                break
            i, j = sorted(rng.integers(0, len(path), 2))
            if j - i < 2:
                continue
            if not self.path_collides(path[i], path[j]):
                path = path[:i + 1] + path[j:]
        return path

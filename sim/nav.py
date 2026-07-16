"""
nav.py — 2D occupancy-grid A* navigation for the mobile base (the masters-project method).

Discretizes the room floor into a grid, marks cells inside the obstacles (inflated by the robot
radius) as occupied, and runs A* (8-connected) from the base's start pose to a goal pose. Returns
a list of (x, y) waypoints the base drives through — routing around obstacles, not through them.
"""
from __future__ import annotations
import heapq

import numpy as np


class RoomNav:
    def __init__(self, obstacles, xlim=(-1.7, 2.1), ylim=(-1.3, 2.1),
                 res=0.08, inflate=0.30):
        self.res, (self.x0, self.x1), (self.y0, self.y1) = res, xlim, ylim
        self.nx = int((self.x1 - self.x0) / res) + 1
        self.ny = int((self.y1 - self.y0) / res) + 1
        self.grid = np.zeros((self.nx, self.ny), bool)
        for cx, cy, hx, hy in obstacles:                       # inflate obstacles by robot radius
            for i in range(self.nx):
                x = self.x0 + i * res
                if abs(x - cx) > hx + inflate:
                    continue
                for j in range(self.ny):
                    y = self.y0 + j * res
                    if abs(y - cy) <= hy + inflate:
                        self.grid[i, j] = True

    def _cell(self, x, y):
        return (int(round((x - self.x0) / self.res)), int(round((y - self.y0) / self.res)))

    def _xy(self, i, j):
        return (self.x0 + i * self.res, self.y0 + j * self.res)

    def _free(self, i, j):
        return 0 <= i < self.nx and 0 <= j < self.ny and not self.grid[i, j]

    def astar(self, start_xy, goal_xy):
        s, g = self._cell(*start_xy), self._cell(*goal_xy)
        if not self._free(*g):                                 # nudge goal to nearest free cell
            g = min(((i, j) for i in range(self.nx) for j in range(self.ny) if self._free(i, j)),
                    key=lambda c: (c[0] - g[0]) ** 2 + (c[1] - g[1]) ** 2)
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        def h(c):
            return np.hypot(c[0] - g[0], c[1] - g[1])
        openq = [(h(s), 0.0, s)]
        came, cost = {s: None}, {s: 0.0}
        while openq:
            _, c, u = heapq.heappop(openq)
            if u == g:
                break
            for dx, dy in nbrs:
                v = (u[0] + dx, u[1] + dy)
                if not self._free(*v):
                    continue
                nc = c + np.hypot(dx, dy)
                if v not in cost or nc < cost[v]:
                    cost[v] = nc
                    came[v] = u
                    heapq.heappush(openq, (nc + h(v), nc, v))
        if g not in came:
            return None
        path, u = [], g                                        # reconstruct + simplify
        while u is not None:
            path.append(self._xy(*u))
            u = came[u]
        path = path[::-1]
        return self._simplify(path)

    def _simplify(self, path):
        if len(path) < 3:
            return path
        out = [path[0]]
        for k in range(1, len(path) - 1):
            a, b, c = np.array(out[-1]), np.array(path[k]), np.array(path[k + 1])
            if abs(np.cross(b - a, c - a)) > 1e-6:              # keep only turning points
                out.append(tuple(b))
        out.append(path[-1])
        return out

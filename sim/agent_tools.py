"""
agent_tools.py — Step 1 of the agentic upgrade: the robot's ACTION SPACE.

`store_task.run()` is a hardcoded sequence: parse -> perceive -> grasp -> navigate -> place. This
module breaks that sequence into a small set of *tools* an agent can call in any order and reason
about. Each tool returns a STRUCTURED OBSERVATION (a plain dict) so a planner can read the result
and decide the next action.

    perceive(shelf) -> what objects are on a shelf (class + confidence), via that shelf's camera
    go_to(shelf)    -> drive + face the base to a shelf (A* around obstacles; tuck arm to travel)
    grasp(object)   -> pick a perceived object off the shelf the robot is at (verified lift)
    place(shelf)    -> put the held object upright at a clear spot on a shelf (verified)
    world_state()   -> where the robot is, what it holds, what it last saw

No LLM yet — Step 2 wraps a ReAct loop around these. Run this file to exercise each tool by hand:
    python sim/agent_tools.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from store_task import StoreSim, _obstacles                       # noqa: E402
from mobile_task import grasp_verified                            # noqa: E402
from rearrange import Detector, MODEL                             # noqa: E402
from plan_and_move import _rel_pose, _carry                       # noqa: E402
from nav import RoomNav                                           # noqa: E402
from make_scene_store import SHELVES, SHELF_TOP                   # noqa: E402
import store_language as SL                                       # noqa: E402

CLASS2MESH = {c: m for m, c in SL.MESH2CLASS.items()}


def _short(cls: str) -> str:
    """'005_tomato_soup_can' -> 'tomato_soup_can'."""
    head = cls.split("_", 1)
    return head[1] if len(head) == 2 and head[0].isdigit() else cls


class RobotTools:
    """Stateful wrapper turning the store robot's capabilities into agent-callable tools.

    State the tools carry between calls:
      _seen    shelf -> list of percepts (object, conf, world pos, point cloud, body suffix)
      _held    the currently-grasped object (or None)
      _at      the shelf the base is currently parked at (or None)
    """

    def __init__(self, sim: StoreSim, det: Detector):
        self.sim, self.det = sim, det
        self._seen: dict[str, list[dict]] = {}
        self._held: dict | None = None
        self._grasp_R = None
        self._at: str | None = None
        self._placed: dict[str, str] = {}
        self._slots: dict[str, list[float]] = {}   # shelf -> x-offsets this robot has already used     # object -> shelf it was successfully placed on

    # ---------- perception ----------
    def perceive(self, shelf: str) -> dict:
        """Detect the objects on `shelf` from its fixed camera; back-project each to a world
        position. Caches the result so a later grasp() can target the physical object."""
        shelf = str(shelf).strip().upper()
        if shelf not in SHELVES:
            return {"tool": "perceive", "ok": False, "error": f"no shelf {shelf!r}; have {list(SHELVES)}"}
        percepts = self.sim.perceive_shelf(self.det, shelf)
        self._seen[shelf] = []
        objs = []
        for cls, w, d0, pts in percepts:
            name = _short(cls)
            mesh = CLASS2MESH.get(name)
            suffix = next((s for s, m, sh, *_ in self.sim.placements if sh == shelf and m == mesh), None)
            self._seen[shelf].append({"object": name, "conf": float(d0.conf),
                                      "w": w.copy(), "pts": pts, "suffix": suffix})
            objs.append({"object": name, "confidence": round(float(d0.conf), 2)})
        return {"tool": "perceive", "ok": True, "shelf": shelf, "objects": objs}

    # ---------- navigation ----------
    def _ensure_at(self, shelf: str) -> tuple[bool, int, str | None]:
        """Navigate to `shelf` and face it, unless already there. -> (ok, waypoints, error).
        Used internally so grasp()/place() don't require the agent to hand-coordinate motion."""
        if self._at == shelf:
            return True, 0, None
        cx, cy, side = SHELVES[shelf]
        park = (cx, cy + side * 0.45)
        if self._held is None:
            self.sim.move_to(self.sim.q_travel, steps=300)          # tuck arm to travel
        start = tuple(self.sim.base_xy())
        path = RoomNav(_obstacles(), xlim=(-2.0, 2.0), ylim=(-2.0, 2.0), inflate=0.22).astar(start, park)
        if path is None:
            return False, 0, f"no path to shelf {shelf}"
        for wx, wy in path:
            self.sim.drive_to(wx, wy)
        self.sim.drive_to(*park, yaw=self.sim.face_shelf_yaw(shelf))
        self._at = shelf
        return True, len(path), None

    def go_to(self, shelf: str) -> dict:
        """A*-navigate the base to `shelf` and face it (optional — grasp/place navigate on their own)."""
        shelf = str(shelf).strip().upper()
        if shelf not in SHELVES:
            return {"tool": "go_to", "ok": False, "error": f"no shelf {shelf!r}"}
        ok, wp, err = self._ensure_at(shelf)
        return {"tool": "go_to", "ok": ok, "at": self._at, "waypoints": wp} if ok \
            else {"tool": "go_to", "ok": False, "error": err}

    # ---------- manipulation ----------
    def grasp(self, object: str) -> dict:
        """Grasp a perceived object (verified — must actually lift). Navigates to the object's
        shelf on its own; only requires that the object was perceived somewhere."""
        if self._held is not None:
            return {"tool": "grasp", "ok": False, "error": f"already holding {self._held['object']}"}
        obj = str(object).strip().lower()
        hit = None
        for sh, items in self._seen.items():
            e = next((x for x in items if obj in x["object"] or x["object"] in obj), None)
            if e is not None:
                hit = (sh, e); break
        if hit is None:
            return {"tool": "grasp", "ok": False,
                    "error": f"{object!r} not perceived anywhere; perceive its shelf first",
                    "perceived": {sh: [x["object"] for x in items] for sh, items in self._seen.items()}}
        shelf, e = hit
        if e["suffix"] is None:
            return {"tool": "grasp", "ok": False, "error": f"cannot resolve body for {object!r}"}
        ok, _, err = self._ensure_at(shelf)
        if not ok:
            return {"tool": "grasp", "ok": False, "error": err}
        p = e["w"].copy()
        self.sim.drive_to(*self.sim.grasp_park(shelf, p[0]), yaw=self.sim.face_shelf_yaw(shelf))
        if not grasp_verified(self.sim, p, e["pts"], e["suffix"]):
            return {"tool": "grasp", "ok": True, "object": e["object"], "grasped": False}
        self._grasp_R = self.sim.grasp_R
        rel = _rel_pose(self.sim, self.sim.m.body(f"obj_{e['suffix']}").id)
        self.sim.frame_hook = lambda: _carry(self.sim, e["suffix"], *rel)
        bx, by = self.sim.base_xy()
        self.sim.reach([bx, by, 0.78], R_des=self._grasp_R, steps=500)   # tuck payload high to travel
        # Perception is a snapshot: once lifted, the object is no longer on that shelf. Leaving it
        # in `_seen` made a later grasp drive to the OLD shelf and fail — stale state, not bad planning.
        self._seen[shelf] = [x for x in self._seen[shelf] if x["object"] != e["object"]]
        self._held = {"object": e["object"], "suffix": e["suffix"]}
        return {"tool": "grasp", "ok": True, "object": e["object"], "grasped": True, "holding": e["object"]}

    def place(self, shelf: str) -> dict:
        """Place the held object upright at a clear spot on `shelf` (verified: on the shelf, tilt≈0).
        Navigates to the shelf on its own."""
        shelf = str(shelf).strip().upper()
        if self._held is None:
            return {"tool": "place", "ok": False, "error": "not holding anything"}
        if shelf not in SHELVES:
            return {"tool": "place", "ok": False, "error": f"no shelf {shelf!r}"}
        ok, _, err = self._ensure_at(shelf)
        if not ok:
            return {"tool": "place", "ok": False, "error": err}
        suffix = self._held["suffix"]
        px, py = self.sim.clear_spot(shelf, self._slots.get(shelf, []))
        _, dcy, dside = SHELVES[shelf]
        self.sim.drive_to(px, dcy + dside * 0.45, yaw=self.sim.face_shelf_yaw(shelf))
        self.sim.reach([px, py, 0.72], R_des=self.sim.DOWN, steps=500)
        # Release height matters: at +0.13 a short can (potted_meat) fell ~10 cm, bounced and slid
        # off the shelf edge — reproducibly, while a taller can survived. Set the object down instead
        # of dropping it: descend until it is nearly touching, then open.
        self.sim.reach([px, py, SHELF_TOP + 0.09], R_des=self.sim.DOWN, steps=450)
        self.sim.frame_hook = None
        self.sim.set_gripper(open_=True, steps=250)
        for _ in range(250):
            mujoco.mj_step(self.sim.m, self.sim.d)
        o = self.sim.d.xpos[self.sim.m.body(f"obj_{suffix}").id]
        R = self.sim.d.xmat[self.sim.m.body(f"obj_{suffix}").id].reshape(3, 3)
        tilt = float(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))
        # Verify the goal ("is it on the shelf?"), not the trajectory ("did it land on my exact
        # pixel?"). The old check compared against the drop point, so a can that settled a few cm
        # away was reported as failed — it then vanished from the agent's state and sent the
        # planner hunting for an object that was sitting safely on the target shelf.
        scx, scy, _ = SHELVES[shelf]
        placed = bool(abs(o[0] - scx) < 0.45 and abs(o[1] - scy) < 0.30 and o[2] > SHELF_TOP - 0.02)
        drift = round(float(np.hypot(o[0] - px, o[1] - py)), 3)
        obj = self._held["object"]
        self._held = None
        self._slots.setdefault(shelf, []).append(px - SHELVES[shelf][0])   # this spot is now taken
        if placed:
            self._placed[obj] = shelf
            return {"tool": "place", "ok": True, "object": obj, "shelf": shelf,
                    "placed": True, "tilt_deg": round(tilt), "drift_m": drift}
        return {"tool": "place", "ok": True, "object": obj, "shelf": shelf, "placed": False,
                "tilt_deg": round(tilt),
                "note": f"{obj} did not settle on {shelf}; perceive {shelf} to re-locate it"}

    # ---------- state ----------
    def world_state(self) -> dict:
        """A snapshot the planner can read: where the base is, what's held, what's been seen."""
        bx, by = self.sim.base_xy()
        return {"tool": "world_state", "at": self._at, "base_xy": [round(bx, 2), round(by, 2)],
                "holding": self._held["object"] if self._held else None,
                "placed": dict(self._placed),
                "seen": {sh: [e["object"] for e in es] for sh, es in self._seen.items()}}


if __name__ == "__main__":
    sim = StoreSim(); sim.reset(); sim.settle()
    det = Detector(MODEL, conf=0.30)
    bot = RobotTools(sim, det)
    # exercise each tool by hand: a full pick-and-place, one tool at a time
    plan = [("perceive", "A"), ("go_to", "A"), ("grasp", "soup"),
            ("go_to", "D"), ("place", "D"), ("world_state", None)]
    for name, arg in plan:
        obs = getattr(bot, name)() if arg is None else getattr(bot, name)(arg)
        call = f"{name}()" if arg is None else f"{name}({arg!r})"
        print(f">> {call:22} -> {obs}")

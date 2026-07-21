"""record_agent.py — render an agent run to MP4.

The agent itself never renders: it perceives through the shelf cameras and acts. To *show* a run,
this wraps `mujoco.mj_step` so every Nth physics step grabs a frame from an overview camera, and
wraps each tool so the frame is captioned with the action currently executing. The result is a
video of the LLM's plan being carried out, with the tool calls visible as they happen.

    python sim/record_agent.py "put all the cans from shelf A onto shelf D" docs/agent-demo.mp4
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_tools import RobotTools, StoreSim, Detector, MODEL   # noqa: E402
from agent import run_agent                                     # noqa: E402

EVERY = 60          # capture one frame per N physics steps
MAX_FRAMES = 1200   # safety cap
CAPTION = {"text": "", "step": 0}


def _overview() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.45]
    cam.distance, cam.azimuth, cam.elevation = 5.2, 90.0, -34.0
    return cam


def _wrap_tools(bot: RobotTools) -> None:
    """Caption each frame with the tool call in flight."""
    for name in ("perceive", "go_to", "grasp", "place"):
        orig = getattr(bot, name)

        def make(n, f):
            def call(arg=None, *a, **kw):
                CAPTION["step"] += 1
                CAPTION["text"] = f'{CAPTION["step"]}. {n}({arg if arg is not None else ""})'
                return f(arg, *a, **kw) if arg is not None else f(*a, **kw)
            return call
        setattr(bot, name, make(name, orig))


def record(goal: str, out: Path) -> dict:
    sim = StoreSim(); sim.reset(); sim.settle()
    bot = RobotTools(sim, Detector(MODEL, conf=0.30))
    _wrap_tools(bot)
    cam, frames, n = _overview(), [], {"i": 0}
    orig_step = mujoco.mj_step

    def step(m, d, nstep=1):
        orig_step(m, d, nstep)
        n["i"] += 1
        if n["i"] % EVERY == 0 and len(frames) < MAX_FRAMES:
            sim.renderer.update_scene(d, camera=cam)
            img = np.ascontiguousarray(sim.renderer.render())
            cv2.rectangle(img, (0, 0), (img.shape[1], 26), (18, 18, 18), -1)
            cv2.putText(img, f'GOAL: {goal}', (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(img, CAPTION["text"], (8, img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 255, 120), 2, cv2.LINE_AA)
            frames.append(img)

    mujoco.mj_step = step
    try:
        result = run_agent(goal, bot)
    finally:
        mujoco.mj_step = orig_step

    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=30, quality=7, macro_block_size=1)
    print(f"\nwrote {out}  ({len(frames)} frames, {len(frames)/30:.1f}s)")
    return result


if __name__ == "__main__":
    goal = sys.argv[1] if len(sys.argv) > 1 else "put all the cans from shelf A onto shelf D"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("docs/agent-demo.mp4")
    r = record(goal, out)
    print(f"=== final state: {r['final_state']}  ({r['steps']} steps)")

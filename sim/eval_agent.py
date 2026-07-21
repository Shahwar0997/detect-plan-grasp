"""eval_agent.py — Step 7: measure the agent instead of demoing it.

Two systems run the same goal suite:

  agent    — the ReAct loop (perceive -> act -> observe -> re-plan, `agent.run_agent`)
  baseline — the same LLM asked ONCE for a complete plan, executed blindly with no feedback

The comparison isolates the value of the loop itself: same model, same tools, same goals; the only
difference is whether the planner gets to see what happened. Success is checked against the
simulator's own state, not the agent's report, so an agent that *claims* success still fails.

    python sim/eval_agent.py            # both systems, full suite
    python sim/eval_agent.py agent      # one system only
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_tools import RobotTools, StoreSim, Detector, MODEL      # noqa: E402
from agent import run_agent, LLM, TOOLS_DOC                        # noqa: E402

# (label, goal, required {object: shelf}) — "kind" groups the report by task type.
SUITE = [
    ("single",  "take the soup from shelf A to shelf D",   {"tomato_soup_can": "D"}),
    ("single",  "take the mug from shelf B to shelf D",    {"mug": "D"}),
    ("recovery", "take the mug from shelf A to shelf C",   {"mug": "C"}),   # the mug is on B
    ("compositional", "put all the cans from shelf A onto shelf D",
     {"tomato_soup_can": "D", "potted_meat_can": "D"}),
    # A also holds a mustard bottle the gripper cannot hold: success = move what is graspable,
    # skip the rest, and still terminate.
    ("compositional", "clear shelf A onto shelf C",
     {"tomato_soup_can": "C", "potted_meat_can": "C"}),
]

BASELINE_SYS = f"""You control a warehouse robot. Shelves: A, B, C, D. It holds AT MOST one object.

Tools:
{TOOLS_DOC}

Output the COMPLETE plan to achieve the goal, as JSON: {{"plan": [{{"tool": ..., "arg": ...}}, ...]}}
`grasp` and `place` drive to the right shelf themselves. You get NO feedback — plan everything now."""


def _score(bot: RobotTools, need: dict) -> bool:
    placed = bot.world_state()["placed"]
    return all(placed.get(o) == s for o, s in need.items())


def run_baseline(goal: str, bot: RobotTools, max_steps: int = 12) -> dict:
    """One shot: ask for the whole plan, execute it blindly."""
    r = ollama.chat(model=LLM, format="json", options={"temperature": 0},
                    messages=[{"role": "system", "content": BASELINE_SYS},
                              {"role": "user", "content": f"GOAL: {goal}"}])
    try:
        plan = json.loads(r["message"]["content"]).get("plan", [])
    except Exception:
        plan = []
    for a in plan[:max_steps]:
        tool = str(a.get("tool", "")).split("(")[0].strip()
        arg = a.get("arg") or a.get("shelf") or a.get("object")
        if tool in {"perceive", "go_to", "grasp", "place"}:
            try:
                getattr(bot, tool)(arg) if arg is not None else None
            except Exception:
                pass
    return {"steps": len(plan[:max_steps])}


def main(which: str = "both") -> None:
    systems = ["agent", "baseline"] if which == "both" else [which]
    rows = []
    for system in systems:
        for kind, goal, need in SUITE:
            sim = StoreSim(); sim.reset(); sim.settle()
            bot = RobotTools(sim, Detector(MODEL, conf=0.30))
            t0 = time.time()
            res = (run_agent(goal, bot, verbose=False) if system == "agent"
                   else run_baseline(goal, bot))
            ok = _score(bot, need)
            rows.append({"system": system, "kind": kind, "goal": goal, "ok": ok,
                         "steps": res["steps"], "sec": round(time.time() - t0, 1)})
            print(f"{system:9s} {kind:14s} {'PASS' if ok else 'FAIL'}  "
                  f"{res['steps']:2d} steps  {rows[-1]['sec']:5.1f}s  {goal}", flush=True)

    print("\n=== summary ===")
    for system in systems:
        sub = [r for r in rows if r["system"] == system]
        done = [r for r in sub if r["ok"]]
        print(f"{system:9s} success {len(done)}/{len(sub)} "
              f"({100*len(done)/len(sub):.0f}%)  avg steps (successful) "
              f"{sum(r['steps'] for r in done)/max(1,len(done)):.1f}")
        for kind in ("single", "recovery", "compositional"):
            k = [r for r in sub if r["kind"] == kind]
            if k:
                print(f"    {kind:14s} {sum(r['ok'] for r in k)}/{len(k)}")
    Path("docs/eval-agent.json").write_text(json.dumps(rows, indent=2))
    print("\nwrote docs/eval-agent.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "both")

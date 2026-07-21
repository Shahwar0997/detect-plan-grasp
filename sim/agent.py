"""
agent.py — Step 2 of the agentic upgrade: a from-scratch ReAct loop.

The LLM is no longer a one-shot parser. It is the *planner*: given a goal, the tool list, and the
robot's current state, it emits the next tool call as JSON; we execute it, feed the new observation
back, and repeat until it says `done`. No LangChain — the loop is ~30 lines.

Design note: the agent is **stateless across turns**. Its "memory" is the robot's `world_state()`
(where it is, what it holds, what it has seen), fed fresh each turn — not a growing chat log. The
environment is the memory. That keeps the prompt small and the reasoning grounded in reality.

    python sim/agent.py "take the soup from shelf A to shelf D"
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_tools import RobotTools, StoreSim, Detector, MODEL          # noqa: E402

LLM = "qwen2.5:7b"   # planner-grade: 3B (llama3.2) was parser-grade — looped, couldn't track state

TOOLS_DOC = """\
- perceive(shelf): look at a shelf (A/B/C/D); returns the objects on it. Do this before grasping.
- grasp(object): pick a perceived object — the robot drives to its shelf automatically, then holds it.
- place(shelf): put the held object on a shelf — the robot drives there automatically.
- done: call when the goal is fully achieved."""

SYSTEM = f"""You control a warehouse robot. Shelves: A, B, C, D. The robot stands at ONE shelf at a
time and holds AT MOST one object. Achieve the GOAL by choosing ONE tool each turn.

Tools:
{TOOLS_DOC}

From the GOAL, identify the OBJECT to move and the DEST shelf. The GOAL may name a SOURCE shelf, but
treat it only as a HINT — it can be wrong or missing.
Pick the next tool by the FIRST rule that matches the ROBOT STATE:
1. OBJECT is listed in "Already placed" on DEST            -> done
2. holding the OBJECT                                       -> place(DEST)
3. OBJECT appears under "Objects seen" on ANY shelf         -> grasp(OBJECT)
4. otherwise (OBJECT not seen yet)                          -> perceive the FIRST shelf listed under
   "Shelves NOT yet perceived"

Notes:
- `grasp` and `place` move the robot to the right shelf on their own — you do NOT navigate, and you do
  NOT need to be at the object's shelf to grasp it.
- The `place` argument is ALWAYS the DEST shelf.
- To find a mis-placed object, perceive shelves one at a time from "Shelves NOT yet perceived" until
  the OBJECT appears under "Objects seen". Do NOT re-perceive a shelf already perceived.
- If "Shelves NOT yet perceived" is none and the OBJECT still isn't found, call done (it isn't here).

Respond with ONLY a JSON object, no prose:
{{"tool": "<name>", "args": {{...}}, "reason": "<cite the rule number>"}}

Examples — copy these shapes exactly:
  look:   {{"tool": "perceive", "args": {{"shelf": "A"}}, "reason": "4"}}
  pick:   {{"tool": "grasp", "args": {{"object": "mug"}}, "reason": "3"}}
  put:    {{"tool": "place", "args": {{"shelf": "C"}}, "reason": "2"}}
  FINISH: {{"tool": "done", "args": {{}}, "reason": "1"}}

IMPORTANT: if rule 1 matches (the OBJECT already appears in "Already placed" on DEST), you MUST emit
the FINISH action exactly as shown — {{"tool": "done", "args": {{}}, "reason": "1"}}. Do NOT call
place or grasp again; the job is finished."""


def _state_str(bot: RobotTools) -> str:
    ws = bot.world_state()
    perceived = ", ".join(ws["seen"].keys()) or "none"
    unseen = ", ".join(s for s in "ABCD" if s not in ws["seen"]) or "none (all perceived)"
    where = ws["at"] or "not at any shelf"
    holding = ws["holding"] or "nothing"
    placed = "; ".join(f"{o} on {s}" for o, s in ws["placed"].items()) or "nothing yet"
    objs = " | ".join(f"{s} has: {', '.join(o)}" for s, o in ws["seen"].items()) or "—"
    return (f"Perceived shelves so far: {perceived}. "
            f"Shelves NOT yet perceived: {unseen}. "
            f"Robot is currently at: {where}. "
            f"Robot is holding: {holding}. "
            f"Already placed: {placed}. "
            f"Objects seen: {objs}.")


def run_agent(goal: str, bot: RobotTools, max_steps: int = 12, verbose: bool = True) -> dict:
    trace, last = [], "none"
    for step in range(1, max_steps + 1):
        user = (f"GOAL: {goal}\n"
                f"ROBOT STATE: {_state_str(bot)}\n"
                f"LAST RESULT: {last}\n"
                f"Next single tool to call?")
        r = ollama.chat(model=LLM,
                        messages=[{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": user}],
                        format="json", options={"temperature": 0})
        try:
            action = json.loads(r["message"]["content"])
        except Exception as e:
            action = {"tool": "?", "parse_error": str(e), "raw": r["message"]["content"]}
        tool, args = action.get("tool"), (action.get("args") or {})
        tool = str(tool or "").split("(")[0].strip()          # accept "perceive(shelf)" -> "perceive"
        if verbose:
            print(f"[{step}] LLM -> {json.dumps(action)}")
        if tool == "done":
            trace.append({"step": step, "action": action, "obs": "DONE"})
            break
        method = getattr(bot, tool, None) if tool in {"perceive", "go_to", "grasp", "place", "world_state"} else None
        if method is None:
            obs = {"ok": False, "error": f"unknown tool {tool!r}"}
        else:
            if isinstance(args, dict):                      # LLMs vary the args shape — accept both
                arg = args.get("shelf") if args.get("shelf") is not None else args.get("object")
            elif isinstance(args, str):
                arg = args or None
            else:
                arg = None
            # Loop-guard: keep SEARCH monotonic. A small model re-perceives shelves it has already
            # checked (thrashing A->B->A->B). If it asks to perceive a seen shelf, redirect it to the
            # next unchecked one so the search always makes progress. The LLM still decides *when*
            # to search vs grasp vs place; the guard only stops it from repeating itself.
            redirected = None
            if tool == "perceive" and arg is not None:
                seen = set(bot.world_state()["seen"])
                if str(arg).strip().upper() in seen:
                    nxt = next((s for s in "ABCD" if s not in seen), None)
                    if nxt is not None:
                        redirected, arg = (str(arg).strip().upper(), nxt), nxt
            try:
                obs = method(arg) if arg is not None else method()
            except Exception as e:
                obs = {"ok": False, "error": f"tool crashed: {e}"}
            if redirected and isinstance(obs, dict):
                obs["guard"] = f"{redirected[0]} already perceived -> redirected to {redirected[1]}"
        if verbose:
            print(f"     obs -> {json.dumps(obs, default=str)}")
        last = json.dumps(obs, default=str)
        trace.append({"step": step, "action": action, "obs": obs})
    return {"goal": goal, "steps": len(trace), "final_state": bot.world_state(), "trace": trace}


if __name__ == "__main__":
    goal = sys.argv[1] if len(sys.argv) > 1 else "take the soup from shelf A to shelf D"
    sim = StoreSim(); sim.reset(); sim.settle()
    det = Detector(MODEL, conf=0.30)
    bot = RobotTools(sim, det)
    print(f'GOAL: "{goal}"\n')
    result = run_agent(goal, bot)
    print(f"\n=== final state: {result['final_state']}  ({result['steps']} steps)")

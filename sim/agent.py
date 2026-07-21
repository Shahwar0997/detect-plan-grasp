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

From the GOAL identify the DEST shelf and the TARGET. The TARGET is either ONE named object
("take the mug to C") or a GROUP ("put all the cans from A on D", "clear shelf A" = every object on A).
A SOURCE shelf in the GOAL is only a HINT — it can be wrong.
Pick the next tool by the FIRST rule that matches the ROBOT STATE:
1. every TARGET object is listed in "Already placed" on DEST -> done
2. holding something                                        -> place(DEST)
3. a shelf named in the GOAL is not yet perceived           -> perceive(that shelf)
4. any TARGET object appears under "Still to move"          -> grasp(the FIRST such object)
5. TARGET object still not found anywhere                   -> perceive the FIRST shelf listed under
   "Shelves NOT yet perceived"
6. otherwise                                                 -> done

Notes:
- `grasp` and `place` move the robot to the right shelf on their own — you do NOT navigate, and you do
  NOT need to be at the object's shelf to grasp it.
- The `place` argument is ALWAYS the DEST shelf.
- To find a mis-placed object, perceive shelves one at a time from "Shelves NOT yet perceived" until
  the OBJECT appears under "Objects seen". Do NOT re-perceive a shelf already perceived.
- If "Shelves NOT yet perceived" is none and the OBJECT still isn't found, call done (it isn't here).
- GROUP goals: rule 5 does NOT apply. If the GOAL names a SOURCE shelf and that shelf is already
  perceived, NEVER perceive another shelf — every TARGET object is on SOURCE. The moment "Still to
  move" on SOURCE holds no TARGET objects (only non-matching ones, e.g. a bottle when the goal says
  "cans"), the job is FINISHED: emit done immediately.

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


def _state_str(bot: RobotTools, skip: set[str] | None = None) -> str:
    """Render the world as text.

    Step 6 addition — "Still to move". A compositional goal ("put all the cans on D") requires
    tracking a *set*: everything on the source shelf, minus what is already moved, minus what
    can't be grasped. A 7B model reliably fails that set arithmetic across turns. So the harness
    computes the remaining set and states it outright; the model only has to pick the first item.
    Same principle as the search guard: the LLM decides *what to do*, not *what is left*.
    """
    ws = bot.world_state()
    placed, skip = ws["placed"], (skip or set())
    perceived = ", ".join(ws["seen"].keys()) or "none"
    unseen = ", ".join(s for s in "ABCD" if s not in ws["seen"]) or "none (all perceived)"
    where = ws["at"] or "not at any shelf"
    holding = ws["holding"] or "nothing"
    placed_s = "; ".join(f"{o} on {s}" for o, s in placed.items()) or "nothing yet"
    objs = " | ".join(f"{s} has: {', '.join(o)}" for s, o in ws["seen"].items()) or "—"
    remain = " | ".join(
        f"{s}: {', '.join([o for o in objl if o not in placed and o not in skip]) or '(nothing left)'}"
        for s, objl in ws["seen"].items()) or "—"
    return (f"Perceived shelves so far: {perceived}. "
            f"Shelves NOT yet perceived: {unseen}. "
            f"Robot is currently at: {where}. "
            f"Robot is holding: {holding}. "
            f"Already placed: {placed_s}. "
            f"Objects seen: {objs}. "
            f"Still to move (per shelf): {remain}. "
            f"Could NOT be grasped, skip these: {', '.join(sorted(skip)) or 'none'}.")


def run_agent(goal: str, bot: RobotTools, max_steps: int = 12, verbose: bool = True) -> dict:
    trace, last = [], "none"
    failed: dict[str, int] = {}          # object -> consecutive grasp failures (2 = give up on it)
    # Goal type decides the search policy: a GROUP goal is bounded by its source shelf, a single
    # -object goal may need to search the whole store. The harness can tell these apart; the 7B
    # planner reliably cannot.
    group = any(w in goal.lower() for w in (" all ", "every", "clear ", "all the"))
    for step in range(1, max_steps + 1):
        skip = {o for o, n in failed.items() if n >= 2}
        user = (f"GOAL: {goal}\n"
                f"ROBOT STATE: {_state_str(bot, skip)}\n"
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
            redirected, blocked = None, None
            if tool == "perceive" and arg is not None:
                seen = set(bot.world_state()["seen"])
                if str(arg).strip().upper() in seen:
                    if group:
                        # A GROUP goal ("all the cans from A") is bounded by its source shelf. The
                        # redirect below is right for a SEARCH but wrong here — it sends the agent
                        # exploring for more cans after the job is done. Block instead, and say why.
                        blocked = {"tool": "perceive", "ok": False, "shelf": str(arg).upper(),
                                   "error": f"{str(arg).upper()} is already perceived and this goal "
                                            "is limited to its source shelf. Do NOT perceive again. "
                                            "If no TARGET objects remain under 'Still to move', "
                                            "the goal is complete — call done."}
                    else:
                        nxt = next((s for s in "ABCD" if s not in seen), None)
                        if nxt is not None:
                            redirected, arg = (str(arg).strip().upper(), nxt), nxt
            if blocked is not None:
                obs = blocked
            else:
                try:
                    obs = method(arg) if arg is not None else method()
                except Exception as e:
                    obs = {"ok": False, "error": f"tool crashed: {e}"}
            if redirected and isinstance(obs, dict):
                obs["guard"] = f"{redirected[0]} already perceived -> redirected to {redirected[1]}"
            # Give-up guard: an object the gripper physically cannot hold (a smooth tapered bottle —
            # see doc 05) would otherwise be retried forever. Two failures = drop it from "Still to
            # move" and tell the planner to skip it, so the rest of the task can still finish.
            if tool == "grasp" and isinstance(obs, dict) and not obs.get("grasped", obs.get("ok")):
                o = str(arg or obs.get("object") or "")
                if o:
                    failed[o] = failed.get(o, 0) + 1
                    if failed[o] >= 2:
                        obs["guard"] = f"{o} failed to grasp twice -> skipping it"
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

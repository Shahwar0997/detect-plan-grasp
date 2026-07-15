"""
language.py — natural-language rearrangement.

Turns a command like "move the soup can to the left" into a structured grasp plan
{object, destination}, grounds it to the objects the detector actually sees, resolves a 3D
place location, and executes it via rearrange.grasp_named. Uses a **local LLM** (Ollama) if
available for robust free-form phrasing, and falls back to a rule-based parser otherwise —
so it runs with or without the model.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rearrange import MultiSim, perceive_all, grasp_named, Detector, MODEL   # noqa: E402

LLM = "llama3.2:3b"                 # small local model; good enough for structured parsing
TABLE_X = (0.34, 0.66)             # reachable placement bounds on the table
TABLE_Y = (-0.28, 0.28)
OFFSET = 0.16

# short aliases -> YCB class names (for the rule fallback + grounding)
ALIASES = {"soup": "tomato_soup_can", "can": "tomato_soup_can", "tomato": "tomato_soup_can",
           "mustard": "mustard_bottle", "bottle": "mustard_bottle",
           "mug": "mug", "cup": "mug",
           "spam": "potted_meat_can", "meat": "potted_meat_can"}


# ---------------------------------------------------------------- parsing
def _llm_parse(command: str, names: list[str]) -> dict | None:
    try:
        import ollama
        prompt = (
            "You control a robot arm over a table. Objects present: "
            f"{names}.\nUser command: \"{command}\"\n"
            "Reply with ONLY JSON: {\"object\": <exactly one name from the list>, "
            "\"direction\": <one of \"left\",\"right\",\"front\",\"back\",\"beside\">, "
            "\"beside\": <a name from the list, or null>}.")
        r = ollama.chat(model=LLM, messages=[{"role": "user", "content": prompt}],
                        format="json", options={"temperature": 0})
        p = json.loads(r["message"]["content"])
        return p if p.get("object") in names else None
    except Exception:
        return None


def _find_obj(text: str, names: list[str]) -> str | None:
    return (next((n for n in names if n.replace("_", " ") in text), None)
            or next((c for a, c in ALIASES.items() if a in text and c in names), None))


def _rule_parse(command: str, names: list[str]) -> dict | None:
    cmd = command.lower()
    if "beside" in cmd or "next to" in cmd:                 # "move X beside Y"
        sep = "beside" if "beside" in cmd else "next to"
        before, after = cmd.split(sep, 1)
        obj = _find_obj(before, names)
        return {"object": obj, "direction": "beside", "beside": _find_obj(after, names)} \
            if obj else None
    obj = _find_obj(cmd, names)
    direction = next((d for d in ("left", "right", "front", "back") if d in cmd), None)
    return {"object": obj, "direction": direction, "beside": None} if obj else None


def parse(command: str, names: list[str]) -> dict | None:
    return _llm_parse(command, names) or _rule_parse(command, names)


# ---------------------------------------------------------------- grounding
def resolve_xy(plan: dict, percepts) -> np.ndarray:
    pos = {c: w for c, w, _ in percepts}
    cur = pos[plan["object"]][:2]
    d = plan.get("direction")
    if d == "beside" and plan.get("beside") in pos:
        near = pos[plan["beside"]][:2]
        xy = near + np.array([OFFSET * 0.9, 0.0])            # to the right of the neighbour
    else:
        step = {"left": (-OFFSET, 0), "right": (OFFSET, 0),
                "front": (0, -OFFSET), "back": (0, OFFSET)}.get(d, (0, 0))
        xy = cur + np.array(step, float)
    return np.array([np.clip(xy[0], *TABLE_X), np.clip(xy[1], *TABLE_Y)])


# ---------------------------------------------------------------- execution
def _short(n: str) -> str:
    return n.split("_", 1)[1] if n[:3].isdigit() else n     # 005_tomato_soup_can -> tomato_soup_can


def run_command(sim: MultiSim, det: Detector, command: str) -> dict:
    percepts, _ = perceive_all(sim, det)
    percepts = [(_short(c), w, d) for c, w, d in percepts]   # work in short names
    names = [c for c, _, _ in percepts]
    plan = parse(command, names)
    if plan is None:
        return {"ok": False, "reason": "could not parse", "seen": names}
    place = resolve_xy(plan, percepts)
    ok = grasp_named(sim, det, plan["object"], place=place)  # substring-matches full class
    return {"ok": ok, "plan": plan, "place": place.round(2).tolist(), "seen": names}


if __name__ == "__main__":
    from make_multi_scene import OBJECTS
    command = " ".join(sys.argv[1:]) or "move the soup can to the front"
    sim = MultiSim(OBJECTS)
    sim.settle()
    det = Detector(MODEL, conf=0.30)
    print(f'command: "{command}"')
    print("result:", run_command(sim, det, command))

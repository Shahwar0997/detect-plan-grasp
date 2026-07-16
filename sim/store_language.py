"""
store_language.py — turn a natural-language store command into a structured, validated shelf task.

    "take the mustard from shelf C to shelf B"
        -> {"object": "mustard", "source": "C", "dest": "B", "valid": True}

Uses a local LLM (Ollama, llama3.2:3b) if it's available, with a deterministic rule-based fallback
so the pipeline runs with no server. `resolve()` then maps the task to the detector class and the
source/dest park poses via the SHELVES registry (make_scene_store) — grounding free language to the
store's known map. Step 2 of the prompt-driven store demo (parsing only; execution is Step 3).
"""
from __future__ import annotations
import json
import re

from make_scene_store import SHELVES, OBJECTS, shelf_park
from rearrange import NAME2CLASS

LLM = "llama3.2:3b"
CATALOG = sorted({m for m, *_ in OBJECTS})                 # meshes actually stocked in the store
MESH2CLASS = {**NAME2CLASS, "banana": "banana"}            # short mesh name -> YCB class (unprefixed)

# colloquial phrasing -> catalog mesh name (checked longest-first so "soup can" beats "soup")
ALIASES = {
    "tomato soup": "soup", "soup can": "soup", "can of soup": "soup", "soup": "soup",
    "mustard bottle": "mustard", "mustard": "mustard",
    "potted meat": "spam", "meat can": "spam", "canned meat": "spam", "spam": "spam", "meat": "spam",
    "coffee cup": "mug", "mug": "mug", "cup": "mug",
    "bleach cleanser": "bleach", "cleanser": "bleach", "cleaner": "bleach", "bleach": "bleach",
    "cheez-it": "cracker", "cheezit": "cracker", "cracker box": "cracker", "crackers": "cracker",
    "cracker": "cracker",
    "jell-o": "gelatin", "jello": "gelatin", "gelatin": "gelatin",
    "master chef": "chef", "coffee can": "chef", "chef": "chef",
    "banana": "banana",
}


def _find_object(text: str) -> str | None:
    for phrase in sorted(ALIASES, key=len, reverse=True):     # longest alias first
        if phrase in text:
            return ALIASES[phrase]
    return None


def _shelf_after(text: str, preps: list[str]) -> str | None:
    for p in preps:
        m = re.search(rf"\b{p}\b\s+(?:the\s+)?(?:shelf\s+)?([a-d])\b", text)
        if m:
            return m.group(1).upper()
    return None


def _rule_parse(command: str) -> dict:
    t = command.lower()
    return {"object": _find_object(t),
            "source": _shelf_after(t, ["from", "on", "at"]),
            "dest": _shelf_after(t, ["to", "onto", "into", "on"])}


def _llm_parse(command: str) -> dict | None:
    try:
        import ollama
        prompt = (
            f"You control a store robot. Shelves: {list(SHELVES)}. Objects: {CATALOG}.\n"
            f'Command: "{command}"\n'
            'Reply with ONLY JSON: {"object": <one object from the list>, '
            '"source": <one shelf letter>, "dest": <one shelf letter>}.')
        r = ollama.chat(model=LLM, messages=[{"role": "user", "content": prompt}],
                        format="json", options={"temperature": 0})
        return json.loads(r["message"]["content"])
    except Exception:
        return None


def parse(command: str) -> dict | None:
    """Parse a command into {object, source, dest, valid[, reason]}. None if it can't be grounded."""
    raw = _llm_parse(command) or _rule_parse(command)
    obj = ALIASES.get(str(raw.get("object", "")).lower(), raw.get("object"))
    src = str(raw.get("source") or "").upper()
    dst = str(raw.get("dest") or "").upper()
    if obj not in CATALOG or src not in SHELVES or dst not in SHELVES:
        return None                                          # couldn't ground object/shelves
    on_source = {m for m, sh, _ in OBJECTS if sh == src}
    if obj not in on_source:                                 # grounded, but the object isn't there
        return {"object": obj, "source": src, "dest": dst, "valid": False,
                "reason": f"no {obj} on shelf {src}"}
    return {"object": obj, "source": src, "dest": dst, "valid": True}


def resolve(task: dict) -> dict:
    """Map a valid task to the detector class + source/dest park poses (from the shelf registry)."""
    return {"target_class": MESH2CLASS[task["object"]],
            "source": task["source"], "dest": task["dest"],
            "source_park": shelf_park(task["source"]),
            "dest_park": shelf_park(task["dest"])}


if __name__ == "__main__":
    import sys
    cmds = sys.argv[1:] or [
        "take the mustard from shelf C to shelf B",
        "grab the soup can on shelf A and put it on shelf D",
        "move the cheez-it crackers from C to A",
        "bring the coffee cup from shelf B to shelf C",
        "put the banana from shelf D onto shelf A",
        "take the bleach from shelf A to shelf B",          # invalid: no bleach on A
    ]
    for c in cmds:
        task = parse(c)
        line = f'"{c}"\n   -> {task}'
        if task and task.get("valid"):
            line += f"\n   resolved: {resolve(task)}"
        print(line + "\n")

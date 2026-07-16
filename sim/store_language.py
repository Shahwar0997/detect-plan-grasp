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
import sys

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


# short descriptions so the LLM can map colloquial phrasing to catalog names
DESC = {"soup": "tomato soup can", "mustard": "mustard bottle", "spam": "potted meat / spam can",
        "mug": "coffee mug / cup", "bleach": "bleach cleanser", "cracker": "cheez-it cracker box",
        "gelatin": "jell-o gelatin box", "chef": "master chef coffee can", "banana": "banana"}


def _shelf_contents() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {s: [] for s in SHELVES}
    for m, sh, _ in OBJECTS:
        if m not in out[sh]:
            out[sh].append(m)
    return out


def _llm_parse(command: str) -> dict | None:
    """Parse via the local LLM. Returns None (and warns) if the server is unreachable or the reply
    is unusable, so a real outage is visible rather than silently masked by the rule fallback."""
    try:
        import ollama
    except ImportError:
        return None
    contents = _shelf_contents()
    shelves = "\n".join(f"  {s}: " + ", ".join(f"{m} ({DESC.get(m, m)})" for m in contents[s])
                        for s in sorted(contents))
    prompt = (
        "You parse commands for a store robot. The shelves currently hold:\n"
        f"{shelves}\n"
        f'Command: "{command}"\n'
        "Choose the object (exactly one item name shown above), the source shelf it is on now, and "
        "the destination shelf. Reply with ONLY JSON: "
        '{"object": <item name>, "source": <A|B|C|D>, "dest": <A|B|C|D>}.')
    try:
        r = ollama.chat(model=LLM, messages=[{"role": "user", "content": prompt}],
                        format="json", options={"temperature": 0})
    except Exception as e:
        print(f"[store_language] LLM server unreachable ({type(e).__name__}); using rule fallback",
              file=sys.stderr)
        return None
    try:
        return json.loads(r["message"]["content"])
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"[store_language] LLM reply not usable ({e}); using rule fallback", file=sys.stderr)
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

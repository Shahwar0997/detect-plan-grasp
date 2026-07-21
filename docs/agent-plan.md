# DPG → Agentic Upgrade — Incremental Plan

**The idea (one line):** turn DPG's *single-shot* LLM grounding into a real **LLM agent** that plans multi-step, calls the robot's tools in a perceive → decide → act loop, observes results, and **recovers from failure**.

**Why:** closes the #1 recurring gap in the job search (AI agents / agentic workflows) and differentiates — "an LLM agent that orchestrates a robot's perception/planning/action tools with recovery" beats yet another chatbot/RAG agent.

**Approach:** a **ReAct-style loop built from scratch** (no LangChain/heavy framework — same from-scratch ethos as the IK solver), using the existing **local Llama 3.2 via Ollama** with JSON tool-calls. Stay in the existing MuJoCo store sim. **One step per sitting; commit/push + a short deep-dive doc each; don't race ahead.**

**Today vs. target:**
- *Today:* LLM parses one command → `{object, source, dest}`; classical planners (A*/RRT/IK) do the rest. The LLM is a **parser**, not an agent.
- *Target:* LLM is the **orchestrator** — it reasons, picks the next tool, sees the result, and re-plans, in a loop, until the goal is verified done.

---

## Steps (each = one sitting · commit + deep-dive doc)

**Step 1 — Tools layer (the agent's action space).**
Wrap existing DPG capabilities as a clean, documented tool API the LLM can call: `perceive(shelf)`, `detect()`, `grasp(object)`, `navigate(location)`, `place(location)`, `world_state()`. Each returns a *structured observation*.
✅ Done when: a script calls each tool and prints its observation. No LLM yet.

**Step 2 — Agent loop skeleton (ReAct).**
Prompt = goal + tool list + latest observation → LLM emits the next `{tool, args}` (JSON-mode, reuse existing grounding). Execute → observe → feed back → repeat until the LLM emits `done`.
✅ Done when: the agent completes the *current* single pick-and-place **through the loop** (parity with today, but agent-driven).

**Step 3 — Multi-step planning.**
The LLM sequences the steps itself (navigate → perceive → grasp → navigate → place) instead of a hardcoded pipeline.
✅ Done when: agent plans + executes the full pick-and-place as a tool sequence *it chose*.

**Step 4 — Observe & adapt.**
Ground each decision in real observations (detector output, grasp-lifted check, base position). Agent perceives before acting and confirms results.
✅ Done when: the agent's next action visibly depends on what it actually observed.

**Step 5 — Failure recovery (the money step).**
Inject failures (grasp slips, object missing, path blocked). Agent observes the failure → re-plans (retry a different grasp / re-perceive / pick an alternative).
✅ Done when: the agent autonomously recovers from a failed grasp *and* a missing object. (This is your verify-and-retry habit lifted to the **reasoning** level.)

**Step 6 — Compositional goals.**
Handle goals that need decomposition: *"clear shelf A"*, *"put all the cans on shelf D"*, *"tidy up"*. Agent decomposes + loops over objects.
✅ Done when: agent completes a multi-object goal end-to-end.

**Step 7 — Eval + artifacts.**
Eval harness (goal set + success criteria): task-success rate, avg steps, recovery rate; **agent vs. old single-shot pipeline**. Then README/demo GIF, a field-guide section, and the resume bullet.
✅ Done when: numbers + public artifacts exist.

---

## Resume payoff (after Step 7)
> Built an **LLM agent** that orchestrates a robot's perception, planning, and action tools via a **from-scratch ReAct loop** (local Llama 3.2, JSON tool-calling) — multi-step planning with **autonomous failure recovery**; +X% task success and Y% recovery vs. the single-shot baseline.

## Job-search note
While building this, keep applying to **edge-fit roles** (optimization / CV / ML-systems / robotics — Quadric-type). Resume applications to **agent / LLM-product roles** once Step 7 lands.

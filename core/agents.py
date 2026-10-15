"""
Multi-agent layer: the main JARVIS conversation routes work to specialist agents.

Two paths, chosen by the main model:

  1. A single, simple command ("open Chrome", "what's the weather") is still a
     direct tool call — instant, no extra model round-trip. The call is only
     *attributed* to the agent that owns the tool, so status/timeline views
     can show who did what.

  2. A multi-step task ("research the best laptops under 60k and save a
     summary", "build me a React todo app and open it") goes through the
     `delegate_task` tool. The chosen agent runs its own plan → act → observe
     loop in a background thread, restricted to ITS tools, and reports back
     when finished — the voice conversation stays free meanwhile.

The sub-agent loop speaks a tiny JSON protocol instead of native function
calling, so it works on every rung of core.gemini's model ladder (including
the Live rung, which cannot do REST-style tool calls).
"""
from __future__ import annotations

import itertools
import json
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from core import gemini


# ── Agent definitions ────────────────────────────────────────────────────────
# `tools` lists the tool names an agent may use. Names that are not installed
# are ignored, so an agent never sees a tool that does not exist. Shared tools
# are attributed to the first owner in _OWNER_ORDER (open_app → System Agent).

@dataclass(frozen=True)
class AgentSpec:
    key: str
    title: str
    icon: str
    purpose: str
    tools: tuple[str, ...]
    guidance: str = ""


AGENTS: dict[str, AgentSpec] = {
    "dev": AgentSpec(
        key="dev", title="Dev Agent", icon="💻",
        purpose=("Software work: building, running, debugging and explaining code "
                 "and projects (Python, React/Vite, Node/Express, etc.)."),
        tools=("dev_agent", "code_helper", "terminal", "file_controller", "file_processor",
               "open_app", "browser_control"),
        guidance=("To create a whole project call dev_agent ONCE with action=build and the full "
                  "description — it plans, writes, installs, verifies, fixes and starts it by "
                  "itself. Use dev_agent run/debug/stop/list for existing projects, terminal for "
                  "single commands (git, npm, pip), code_helper for explaining or editing code."),
    ),
    "research": AgentSpec(
        key="research", title="Research Agent", icon="🔎",
        purpose=("Finding and comparing information on the web: searches, reading "
                 "pages, prices, news, weather, flights, videos."),
        tools=("web_search", "world_monitor", "browser_control", "weather_report", "flight_finder",
               "youtube_video", "file_processor"),
        guidance=("Start with web_search — mode 'research' for a topic, 'compare' with items for "
                  "comparisons, 'price' for prices, 'news' for latest events. ONE web_search call "
                  "returns a synthesised answer, so 1-3 searches are normally enough. Use "
                  "browser_control only to read ONE specific page (go_to, then get_text) — never "
                  "scroll or click around to find information. Finish as soon as you can answer."),
    ),
    "system": AgentSpec(
        key="system", title="System Agent", icon="🖥️",
        purpose=("Controlling this computer: apps, windows, files and folders, "
                 "settings, terminal commands, processes, games, CPU/RAM status."),
        tools=("open_app", "computer_control", "computer_settings", "desktop_control",
               "file_controller", "terminal", "system_status", "game_updater", "video_player"),
        guidance=("Process/CPU/RAM questions: terminal action=processes or system_status. Shell "
                  "work: terminal action=run. Files: file_controller. Closing an app: "
                  "computer_settings action=close_app. Destructive actions wait for the user's "
                  "on-screen confirmation — report that, never claim they are done."),
    ),
    "task": AgentSpec(
        key="task", title="Task Agent", icon="⚙️",
        purpose=("Automating everyday chores that combine several steps: reminders, "
                 "messages, filling web forms, routines across apps."),
        tools=("reminder", "send_message", "browser_control", "open_app",
               "computer_control", "file_controller", "web_search", "weather_report"),
    ),
}

# Tools not owned by any agent are attributed to the main assistant.
MAIN_AGENT = "jarvis"

MAX_STEPS        = 10
RESULT_CHARS     = 1_500     # how much of each tool result the agent sees
STEP_TIMEOUT_MS  = 60_000


_OWNER_ORDER = ("system", "research", "dev", "task")


def agent_for_tool(tool_name: str) -> str:
    for key in _OWNER_ORDER:
        if tool_name in AGENTS[key].tools:
            return key
    return MAIN_AGENT


# ── Status hub ───────────────────────────────────────────────────────────────

@dataclass
class AgentJob:
    id: int
    agent: str
    task: str
    status: str = "running"            # running | done | failed | cancelled
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    steps: list[dict] = field(default_factory=list)
    summary: str = ""
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)


class AgentHub:
    """
    Owns agent state (what each agent is doing right now), the background jobs,
    and a short activity timeline. Thread-safe. UI / dashboard subscribe with
    `on_event(callback)`; callbacks get plain dicts and must not block.
    """

    def __init__(self,
                 tool_runner: Callable[[str, dict], str],
                 tool_decls: Callable[[], dict[str, dict]],
                 assistant_name: Callable[[], str] = lambda: "JARVIS"):
        self._run_tool       = tool_runner
        self._tool_decls     = tool_decls
        self._assistant_name = assistant_name
        self._lock           = threading.Lock()
        self._ids            = itertools.count(1)
        self._jobs: dict[int, AgentJob] = {}
        self._current: dict[str, str]   = {}          # agent -> activity text
        self._timeline: deque[dict]     = deque(maxlen=100)
        self._listeners: list[Callable[[dict], None]] = []

    # -- events --------------------------------------------------------------
    def on_event(self, callback: Callable[[dict], None]) -> None:
        self._listeners.append(callback)

    def _emit(self, event: dict) -> None:
        event.setdefault("ts", time.time())
        with self._lock:
            self._timeline.append(event)
        for cb in list(self._listeners):
            try:
                cb(event)
            except Exception as e:
                print(f"[Agents] listener failed: {e}")

    # -- direct tool calls (attribution only) --------------------------------
    def tool_started(self, tool: str, args: dict | None = None) -> str:
        agent = agent_for_tool(tool)
        with self._lock:
            self._current[agent] = tool
        self._emit({"type": "tool_start", "agent": agent, "tool": tool,
                    "args": _short(args)})
        return agent

    def tool_finished(self, tool: str, result: str) -> None:
        agent = agent_for_tool(tool)
        with self._lock:
            if self._current.get(agent) == tool:
                self._current.pop(agent, None)
        self._emit({"type": "tool_end", "agent": agent, "tool": tool,
                    "result": str(result)[:200]})

    # -- delegated jobs -------------------------------------------------------
    def delegate(self, agent: str, task: str,
                 on_done: Callable[[AgentJob], None] | None = None) -> AgentJob:
        spec = AGENTS.get(agent)
        if spec is None:
            raise ValueError(f"unknown agent '{agent}' (choose: {', '.join(AGENTS)})")
        job = AgentJob(id=next(self._ids), agent=agent, task=task.strip())
        with self._lock:
            self._jobs[job.id] = job
            self._current[agent] = f"task #{job.id}"
        self._emit({"type": "job_start", "agent": agent, "job": job.id, "task": job.task})

        def _worker():
            try:
                self._run_job(spec, job)
            except Exception as e:
                traceback.print_exc()
                job.status, job.summary = "failed", f"{spec.title} crashed: {e}"
            job.finished = time.time()
            with self._lock:
                if self._current.get(agent, "").startswith(f"task #{job.id}"):
                    self._current.pop(agent, None)
            self._emit({"type": "job_end", "agent": agent, "job": job.id,
                        "status": job.status, "summary": job.summary})
            if on_done:
                try:
                    on_done(job)
                except Exception as e:
                    print(f"[Agents] on_done failed: {e}")

        threading.Thread(target=_worker, daemon=True,
                         name=f"Agent-{agent}-{job.id}").start()
        return job

    def cancel(self, job_id: int | None = None) -> str:
        with self._lock:
            running = [j for j in self._jobs.values() if j.status == "running"]
        if job_id is not None:
            running = [j for j in running if j.id == job_id]
        if not running:
            return "No running agent task to cancel."
        for j in running:
            j.cancel.set()
        return "Cancelling: " + ", ".join(f"#{j.id} ({AGENTS[j.agent].title})" for j in running)

    def status_text(self) -> str:
        with self._lock:
            jobs    = sorted(self._jobs.values(), key=lambda j: j.id, reverse=True)[:6]
            current = dict(self._current)
        lines = []
        for spec in AGENTS.values():
            doing = current.get(spec.key)
            lines.append(f"{spec.icon} {spec.title}: {'working — ' + doing if doing else 'idle'}")
        if jobs:
            lines.append("Recent tasks:")
            for j in jobs:
                tail = f" — {j.summary[:160]}" if j.summary else f" — step {len(j.steps)}"
                lines.append(f"  #{j.id} [{j.status}] {AGENTS[j.agent].title}: {j.task[:80]}{tail}")
        return "\n".join(lines)

    def snapshot(self) -> dict:
        """Plain-data view for a UI panel / dashboard."""
        with self._lock:
            return {
                "agents": [{"key": s.key, "title": s.title, "icon": s.icon,
                            "busy": s.key in self._current,
                            "activity": self._current.get(s.key, "")}
                           for s in AGENTS.values()],
                "jobs": [{"id": j.id, "agent": j.agent, "task": j.task,
                          "status": j.status, "steps": len(j.steps),
                          "summary": j.summary}
                         for j in sorted(self._jobs.values(), key=lambda j: -j.id)[:20]],
                "timeline": list(self._timeline)[-40:],
            }

    # -- the agent loop -------------------------------------------------------
    def _agent_tools(self, spec: AgentSpec) -> dict[str, dict]:
        decls = self._tool_decls()
        return {n: decls[n] for n in spec.tools if n in decls}

    def _run_job(self, spec: AgentSpec, job: AgentJob) -> None:
        tools = self._agent_tools(spec)
        if not tools:
            job.status, job.summary = "failed", f"{spec.title} has no tools installed."
            return

        tool_doc = json.dumps([
            {"name": n, "description": " ".join(str(d.get("description", "")).split())[:600],
             "parameters": d.get("parameters", {}).get("properties", {}),
             "required": d.get("parameters", {}).get("required", [])}
            for n, d in tools.items()
        ], ensure_ascii=False)

        seen_calls: set[str] = set()
        for step_no in range(1, MAX_STEPS + 1):
            if job.cancel.is_set():
                job.status, job.summary = "cancelled", "Cancelled by the user."
                return

            decision = gemini.as_json(
                self._prompt(spec, job, tool_doc), tier=gemini.SMART,
                timeout_ms=STEP_TIMEOUT_MS, default=None,
            )
            if not isinstance(decision, dict):
                job.status = "failed"
                job.summary = "The planning model did not answer — try again shortly."
                return

            if decision.get("done"):
                job.status  = "done"
                job.summary = str(decision.get("summary") or "Task finished.").strip()
                return

            tool = str(decision.get("tool") or "").strip()
            args = decision.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            thought = str(decision.get("thought") or "")[:200]

            if tool not in tools:
                result = f"ERROR: '{tool}' is not one of your tools: {', '.join(tools)}."
            else:
                sig = tool + json.dumps(args, sort_keys=True, ensure_ascii=False)
                if sig in seen_calls:
                    result = ("ERROR: you already made this exact call. Use its earlier "
                              "result, try something different, or finish.")
                else:
                    seen_calls.add(sig)
                    with self._lock:
                        self._current[spec.key] = f"task #{job.id}: {tool}"
                    self._emit({"type": "job_step", "agent": spec.key, "job": job.id,
                                "step": step_no, "tool": tool, "args": _short(args),
                                "thought": thought})
                    try:
                        result = str(self._run_tool(tool, args) or "Done.")
                    except Exception as e:
                        result = f"ERROR: {tool} failed: {e}"

            print(f"[Agents] #{job.id} {spec.title} step {step_no}: {tool} → {result[:100]}")
            job.steps.append({"tool": tool, "args": args, "thought": thought,
                              "result": result[:RESULT_CHARS]})

        # Out of steps: still give the user an answer from what was gathered.
        final = gemini.as_json(
            self._prompt(spec, job, tool_doc) + "\n\nYou have NO steps left. Reply now with "
            '{"done": true, "summary": "..."} built only from the results above; say what is '
            "still missing, if anything.",
            tier=gemini.SMART, timeout_ms=STEP_TIMEOUT_MS, default=None,
        )
        job.status  = "done"
        job.summary = (str(final.get("summary")).strip()
                       if isinstance(final, dict) and final.get("summary")
                       else "Ran out of steps before finishing: "
                            + "; ".join(f"{s['tool']} → {s['result'][:80]}" for s in job.steps[-3:]))

    def _prompt(self, spec: AgentSpec, job: AgentJob, tool_doc: str) -> str:
        history = "\n".join(
            f"STEP {i}: tool={s['tool']} args={json.dumps(s['args'], ensure_ascii=False)}\n"
            f"RESULT: {s['result']}"
            for i, s in enumerate(job.steps, 1)
        ) or "(nothing yet)"
        return (
            f"You are the {spec.title} of {self._assistant_name()}, a desktop AI "
            f"assistant on the user's own computer. Your speciality: {spec.purpose}\n"
            + (f"HOW TO WORK: {spec.guidance}\n" if spec.guidance else "") +
            f"Current date and time: {time.strftime('%A %Y-%m-%d %H:%M')}.\n\n"
            f"TASK FROM THE USER:\n{job.task}\n\n"
            f"YOUR TOOLS (exactly one call per step):\n{tool_doc}\n\n"
            f"WHAT YOU HAVE DONE SO FAR:\n{history}\n\n"
            "Decide the single next step. Reply with ONLY one JSON object, either\n"
            '  {"thought": "<why>", "tool": "<tool name>", "args": {<parameters>}}\n'
            "or, when the task is complete or cannot be completed,\n"
            '  {"thought": "<why>", "done": true, "summary": "<result for the user>"}\n\n'
            "Rules:\n"
            "- Use only the tools listed, with their documented parameters.\n"
            "- Base every claim on tool RESULTS; never invent data, prices or outcomes.\n"
            "- If a step failed, adapt (different parameters or tool) instead of repeating it.\n"
            "- The summary is spoken to the user: at most 3 short sentences with the "
            "concrete outcome (names, numbers, file paths). Say plainly what failed, if anything.\n"
            f"- You have {MAX_STEPS - len(job.steps)} steps left; finish before running out."
        )


def _short(args) -> dict:
    if not isinstance(args, dict):
        return {}
    return {k: (str(v)[:80] if not isinstance(v, (int, float, bool)) else v)
            for k, v in args.items()}

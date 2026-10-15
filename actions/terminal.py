"""
Terminal commands and process control.

run        — runs a shell command and returns its output. Everyday commands
             (ls, git status, pip list, npm -v, df -h …) run at once. Anything
             destructive or system-wide (rm -r, sudo, kill, shutdown, mkfs, dd,
             chmod -R, curl | sh, package removal …) is parked behind the HUD
             confirmation from core/confirm.py — the model cannot approve it.
processes  — top processes by CPU or memory, optionally filtered by name.
kill       — ends a process by PID or name, after on-screen confirmation.
"""
from __future__ import annotations

import os
import platform
import re
import signal
import subprocess
from pathlib import Path

import psutil

from core import confirm

_OS = platform.system()
_MAX_OUT = 4000

_DANGEROUS = re.compile(
    r"""(
        \brm\s | \bunlink\b |                      # any delete
        \brmdir\b | \bshred\b | \bmkfs | \bdd\s+ | \bfdisk\b | \bparted\b | \bwipefs\b |
        \bsudo\b | \bsu\s | \bpkexec\b | \bdoas\b |
        \bshutdown\b | \breboot\b | \bpoweroff\b | \bhalt\b | \binit\s+[06]\b |
        \bsystemctl\s+(stop|disable|mask|restart|poweroff|reboot) |
        \bkill(all)?\b | \bpkill\b | \btaskkill\b |
        \bchmod\s+-R | \bchown\s+-R | \bchmod\s+777 |
        \b(apt|apt-get|dnf|yum|pacman|snap|brew|winget|choco)\s+(remove|purge|uninstall|autoremove|-R) |
        \b(pip|pip3|npm)\s+uninstall |
        \bgit\s+(push\s+.*--force|push\s+-f|reset\s+--hard|clean\s+-\w*f) |
        (curl|wget)[^|]*\|\s*(ba|z)?sh |
        >\s*/dev/sd | :\(\)\s*\{ |
        \bformat\s+[a-z]: | \bdel\s+/[sq] | \brd\s+/s | Remove-Item.*-Recurse
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def _shell_cmd(command: str) -> list[str] | str:
    if _OS == "Windows":
        return ["powershell", "-NoProfile", "-Command", command]
    return ["/bin/bash", "-lc", command]


def _exec(command: str, cwd: str, timeout: int) -> str:
    try:
        r = subprocess.run(_shell_cmd(command), cwd=cwd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout,
                           stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return f"Command still running after {timeout}s — stopped. (Interactive or long-running commands are not supported here.)"
    except Exception as e:
        return f"Could not run command: {e}"
    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()
    text = out
    if err:
        text += ("\n" if text else "") + "STDERR:\n" + err
    if len(text) > _MAX_OUT:
        text = text[:1200] + "\n…\n" + text[-(_MAX_OUT - 1200):]
    return f"Exit code {r.returncode}.\n{text}" if text else f"Exit code {r.returncode} (no output)."


def _resolve_cwd(cwd: str) -> str:
    if not cwd:
        return str(Path.home())
    shortcuts = {"desktop": Path.home() / "Desktop", "downloads": Path.home() / "Downloads",
                 "documents": Path.home() / "Documents", "home": Path.home(),
                 "projects": Path.home() / "Desktop" / "JarvisProjects"}
    p = shortcuts.get(cwd.lower().strip()) or Path(cwd).expanduser()
    return str(p) if p.is_dir() else str(Path.home())


def _processes(sort: str, count: int, name: str) -> str:
    procs = []
    for p in psutil.process_iter(["pid", "name", "memory_info", "username"]):
        try:
            p.cpu_percent(None)
            procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    psutil.cpu_percent(0.5)          # sampling window
    rows = []
    for p in procs:
        try:
            pname = p.info["name"] or ""
            if name and name.lower() not in pname.lower():
                continue
            rows.append((p.pid, pname, p.cpu_percent(None),
                         (p.info["memory_info"].rss if p.info["memory_info"] else 0) / 1024 ** 2))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    key = (lambda r: r[3]) if sort.startswith("mem") else (lambda r: r[2])
    rows.sort(key=key, reverse=True)
    rows = rows[:max(1, min(count, 40))]
    if not rows:
        return f"No process matching '{name}'." if name else "No processes found."
    lines = [f"{'PID':>7}  {'CPU%':>5}  {'RAM MB':>8}  NAME"]
    lines += [f"{pid:>7}  {cpu:>5.1f}  {mem:>8.0f}  {pname}" for pid, pname, cpu, mem in rows]
    vm = psutil.virtual_memory()
    lines.append(f"Total: CPU {psutil.cpu_percent(None):.0f}%, RAM {vm.percent:.0f}% "
                 f"({vm.used / 1024 ** 3:.1f}/{vm.total / 1024 ** 3:.1f} GB)")
    return "\n".join(lines)


def _kill_targets(pid, name: str) -> list[psutil.Process]:
    me = {os.getpid(), os.getppid()}
    out = []
    if pid not in (None, ""):
        try:
            p = psutil.Process(int(pid))
            if p.pid not in me:
                out.append(p)
        except (psutil.NoSuchProcess, ValueError):
            pass
    elif name:
        for p in psutil.process_iter(["pid", "name"]):
            if p.pid not in me and (p.info["name"] or "").lower() == name.lower().strip():
                out.append(p)
    return out


def _do_kill(procs: list[psutil.Process]) -> str:
    killed = 0
    for p in procs:
        try:
            p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    gone, alive = psutil.wait_procs(procs, timeout=4)
    for p in alive:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    killed = len(gone) + len(alive)
    return f"Ended {killed} process(es)."


def terminal(parameters: dict, player=None, **_) -> str:
    p       = parameters or {}
    action  = (p.get("action") or "run").lower().strip()

    if action in ("processes", "process", "list_processes", "top"):
        return _processes(str(p.get("sort") or "cpu").lower(), int(p.get("count") or 10),
                          str(p.get("name") or ""))

    if action in ("kill", "kill_process", "end"):
        procs = _kill_targets(p.get("pid"), str(p.get("name") or ""))
        if not procs:
            return "No matching process found (or it is JARVIS itself)."
        desc = ", ".join(f"{pr.name()} ({pr.pid})" for pr in procs[:5])
        if confirm.pending_title():
            return "There is already a confirmation waiting on screen. Ask the user to answer that one first."
        return confirm.request(key="kill_process", title=f"End process: {desc}",
                               detail="Unsaved work in it will be lost.",
                               run=lambda: _do_kill(procs))

    command = str(p.get("command") or "").strip()
    if not command:
        return "No command given."
    cwd     = _resolve_cwd(str(p.get("cwd") or ""))
    timeout = max(5, min(int(p.get("timeout") or 60), 600))

    if _DANGEROUS.search(command):
        if confirm.pending_title():
            return "There is already a confirmation waiting on screen. Ask the user to answer that one first."

        def _run_confirmed() -> str:
            out = _exec(command, cwd, timeout)
            if player:
                player.show_content(f"$ {command[:40]}", out)
            return out[:300]

        return confirm.request(key="terminal", title=f"Run: {command[:80]}",
                               detail=f"in {cwd}", run=_run_confirmed)

    out = _exec(command, cwd, timeout)
    if player:
        try:
            player.show_content(f"$ {command[:40]}", out)
        except Exception:
            pass
    return out


TOOL = {
    "name": "terminal",
    "description": (
        "Runs terminal/shell commands on this computer and manages processes. "
        "action=run: execute a command (git, npm, pip, ls, df, ping, python scripts…) and get "
        "its output; destructive or system-wide commands (sudo, rm -r, kill, shutdown, package "
        "removal, force-push) need the user to press CONFIRM on screen — never claim they ran. "
        "action=processes: top processes by cpu or memory (what is using my RAM/CPU). "
        "action=kill: end a process by pid or exact name (asks for on-screen confirmation). "
        "Commands must be non-interactive."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action":  {"type": "STRING", "description": "run (default) | processes | kill"},
            "command": {"type": "STRING", "description": "Shell command for run (bash on Linux/macOS, PowerShell on Windows)"},
            "cwd":     {"type": "STRING", "description": "Working directory or shortcut: home, desktop, downloads, documents, projects"},
            "timeout": {"type": "INTEGER", "description": "Seconds before the command is stopped (default 60)"},
            "sort":    {"type": "STRING", "description": "processes: cpu (default) | memory"},
            "count":   {"type": "INTEGER", "description": "processes: how many to list (default 10)"},
            "name":    {"type": "STRING", "description": "processes: filter by name; kill: exact process name"},
            "pid":     {"type": "INTEGER", "description": "kill: process id"},
        },
        "required": [],
    },
    "handler": terminal,
}

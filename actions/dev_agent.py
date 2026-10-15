"""
Developer agent — builds, runs, debugs and stops real projects.

Stacks
    python      main.py style scripts / apps
    react       Vite + React front-end
    node        Node.js / Express API (optionally MongoDB via Mongoose)
    fullstack   MERN: client/ (Vite React) + server/ (Express + MongoDB)

Actions
    build   (default) plan → write files → install → verify → auto-fix → start
    run     start an existing project (servers stay up in the background)
    debug   verify an existing project and auto-fix the errors found
    stop    stop servers this agent started (one project or all)
    list    show projects and which ones are running

Verification is real, not "it did not crash in 30 s": React clients are checked
with `npm run build`, Express servers are started and their /api/health is
requested, Python entry points are executed. Errors go back to the model with
the failing file and the other files as context, up to MAX_FIX_ATTEMPTS times.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path


def get_base_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR         = get_base_dir()
API_CONFIG_PATH  = BASE_DIR / "config" / "api_keys.json"
PROJECTS_DIR     = Path.home() / "Desktop" / "JarvisProjects"
MAX_FIX_ATTEMPTS = 5
# Model choice, timeout and fallback ladder all live in core/gemini.py.
from core import gemini

MODEL_PLANNER    = gemini.SMART
MODEL_WRITER     = gemini.SMART

_IS_WIN = sys.platform == "win32"
_NPM    = "npm.cmd" if _IS_WIN else "npm"
_NPX    = "npx.cmd" if _IS_WIN else "npx"

_SKIP_DIRS  = {"node_modules", "dist", "build", ".git", "__pycache__", ".venv",
               "venv", ".jarvis", ".next", "coverage"}
_SOURCE_EXT = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".css",
               ".html", ".json", ".env", ".md"}
_SKIP_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}


# ── Model helpers ────────────────────────────────────────────────────────────

class RateLimitError(Exception):
    pass


def _get_model(model_name: str = gemini.SMART):
    """Planning and writing whole files — the reasoning tier, and a long
    deadline because the answer is a source file rather than a sentence."""
    class _W:
        def generate_content(self, contents):
            resp = gemini.call(contents, tier=model_name, timeout_ms=90_000)
            if resp is None:
                raise RuntimeError("every Gemini model on the ladder failed")
            return resp

    return _W()


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\r?\n?", "", text)
    text = re.sub(r"\r?\n?```\s*$", "", text)
    return text.strip()


def _is_rate_limit(error: Exception) -> bool:
    msg = str(error).lower()
    return "429" in msg or "quota" in msg or "resource_exhausted" in msg


def _ask(prompt: str) -> str:
    try:
        return _get_model(MODEL_WRITER).generate_content(prompt).text or ""
    except Exception as e:
        if _is_rate_limit(e):
            raise RateLimitError(str(e))
        raise


# ── Stack detection ──────────────────────────────────────────────────────────

_STACK_ALIASES = {
    "python": "python", "py": "python",
    "react": "react", "reactjs": "react", "react.js": "react", "vite": "react",
    "node": "node", "nodejs": "node", "node.js": "node", "express": "node",
    "javascript": "node", "js": "node", "typescript": "node", "ts": "node",
    "fullstack": "fullstack", "full-stack": "fullstack", "full stack": "fullstack",
    "mern": "fullstack",
}


def _detect_stack(description: str, language: str) -> str:
    d = description.lower()
    if any(k in d for k in ("mern", "full stack", "full-stack", "fullstack")):
        return "fullstack"
    if ("react" in d or "frontend" in d or "front-end" in d) and \
            any(k in d for k in ("express", "backend", "back-end", "api server", "mongo")):
        return "fullstack"
    lang = _STACK_ALIASES.get(language.lower().strip())
    if lang:
        return lang
    if any(k in d for k in ("express", "node.js", "nodejs", "rest api", "backend", "mongo")):
        return "node"
    if any(k in d for k in ("react", "vite")):
        return "react"
    return "python"


def _wants_mongo(description: str, stack: str) -> bool:
    d = description.lower()
    return stack == "fullstack" or "mongo" in d or "database" in d and stack == "node"


# ── Processes, ports, MongoDB ────────────────────────────────────────────────

_running: dict[str, list[dict]] = {}       # project name -> [{proc, name, url, log}]
_run_lock = threading.Lock()


def _free_port(start: int) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _http_get(url: str, timeout: float = 3.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read(2000).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read(2000).decode("utf-8", "replace")
    except Exception as e:
        return 0, str(e)


def ensure_mongodb() -> str:
    """Makes sure a MongoDB is reachable on localhost:27017 (starts Docker if needed)."""
    if _port_open(27017):
        return "MongoDB is running on localhost:27017."
    docker = shutil.which("docker")
    if not docker:
        return ("MongoDB is not running and Docker is not installed — install MongoDB "
                "or set MONGODB_URI in the project's .env (e.g. a MongoDB Atlas URL).")
    try:
        r = subprocess.run([docker, "start", "jarvis-mongo"], capture_output=True,
                           text=True, timeout=30)
        if r.returncode != 0:
            r = subprocess.run(
                [docker, "run", "-d", "--name", "jarvis-mongo", "-p", "27017:27017",
                 "-v", "jarvis-mongo-data:/data/db", "mongo:7"],
                capture_output=True, text=True, timeout=300,
            )
        if r.returncode != 0:
            return f"Could not start MongoDB in Docker: {r.stderr.strip()[-300:]}"
        for _ in range(30):
            if _port_open(27017):
                return "MongoDB started in Docker (container jarvis-mongo) on localhost:27017."
            time.sleep(1)
        return "MongoDB container started but port 27017 is not answering yet."
    except Exception as e:
        return f"Could not start MongoDB: {e}"


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if _IS_WIN:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=10)
        else:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _stop_project(name: str) -> int:
    with _run_lock:
        items = _running.pop(name, [])
    for it in items:
        _kill(it["proc"])
    # Servers started by an earlier JARVIS run are not in _running — find them
    # by their working directory.
    stray = 0
    try:
        import psutil
        root = str(PROJECTS_DIR / name)
        for p in psutil.process_iter(["pid"]):
            try:
                cwd = p.cwd()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            if (cwd == root or cwd.startswith(root + os.sep)) and p.pid != os.getpid():
                p.terminate()
                stray += 1
    except Exception:
        pass
    return len(items) + stray


def running_projects() -> list[dict]:
    """Projects with live processes started in this JARVIS run (for dashboards)."""
    out = []
    with _run_lock:
        for name, items in _running.items():
            live = [i for i in items if i["proc"].poll() is None]
            if live:
                out.append({"name": name, "urls": [i["url"] for i in live if i["url"]],
                            "processes": [i["name"] for i in live]})
    return out


def _start_process(project: str, label: str, cmd: list[str], cwd: Path,
                   env: dict | None = None) -> dict:
    log_dir = cwd / ".jarvis"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{label}.log"
    log_file = open(log_path, "w", encoding="utf-8", errors="replace")
    kw: dict = {"stdout": log_file, "stderr": subprocess.STDOUT, "cwd": str(cwd),
                "env": {**os.environ, **(env or {})}}
    if _IS_WIN:
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kw)
    item = {"proc": proc, "name": label, "url": "", "log": log_path}
    with _run_lock:
        _running.setdefault(project, []).append(item)
    return item


def _log_tail(path: Path, n: int = 3000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-n:]
    except Exception:
        return ""


def _wait_server(item: dict, port: int, health_path: str = "/",
                 timeout: float = 25.0) -> tuple[bool, str]:
    """Waits for a started server: (True, info) once it answers, (False, log) if it dies."""
    deadline = time.time() + timeout
    url = f"http://localhost:{port}{health_path}"
    while time.time() < deadline:
        if item["proc"].poll() is not None:
            return False, (f"Process exited with code {item['proc'].returncode}.\n"
                           + _log_tail(item["log"]))
        if _port_open(port):
            status, body = _http_get(url)
            if status and status < 500:
                item["url"] = f"http://localhost:{port}"
                return True, f"{url} → HTTP {status}: {body[:300]}"
            if status >= 500:
                return False, f"{url} → HTTP {status}: {body[:800]}\n" + _log_tail(item["log"])
        time.sleep(0.7)
    log = _log_tail(item["log"])
    if _ERROR_RE.search(log):
        return False, "Server did not answer and logged errors:\n" + log
    return False, f"Server did not answer on port {port} within {int(timeout)}s.\n" + log


# ── Running commands ─────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: Path, timeout: int = 300, env: dict | None = None) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout,
                           env={**os.environ, **(env or {})})
        return r.returncode, (r.stdout + "\n" + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, f"Timed out after {timeout}s: {' '.join(cmd)}"
    except FileNotFoundError as e:
        return -2, f"Command not found: {e}"


_ERROR_RE = re.compile(
    r"(Traceback \(most recent call last\)|SyntaxError|ReferenceError|TypeError|"
    r"Cannot find module|Cannot find package|ERR_MODULE_NOT_FOUND|"
    r"Failed to resolve import|\[vite\].*error|Error: |error during build|"
    r"ModuleNotFoundError|ImportError|NameError|AttributeError|EADDRINUSE)"
)


def _npm_install(cwd: Path, packages: list[str] | None = None, dev: bool = False) -> str:
    if not (cwd / "package.json").exists():
        return f"No package.json in {cwd.name}; skipped npm install."
    cmd = [_NPM, "install", "--no-audit", "--no-fund"]
    if packages:
        cmd += (["--save-dev"] if dev else []) + packages
    code, out = _run(cmd, cwd, timeout=600)
    what = ", ".join(packages) if packages else "dependencies"
    return (f"npm installed {what} in {cwd.name}/." if code == 0
            else f"npm install failed in {cwd.name}/: {out[-600:]}")


def _pip_install(packages: list[str], cwd: Path) -> str:
    missing = []
    for dep in packages:
        name = re.split(r"[>=<!\[]", dep)[0].strip()
        if name and _run([sys.executable, "-m", "pip", "show", name], cwd, 30)[0] != 0:
            missing.append(dep)
    if not missing:
        return "Python dependencies already installed."
    code, out = _run([sys.executable, "-m", "pip", "install"] + missing, cwd, 600)
    return f"pip installed {', '.join(missing)}." if code == 0 else f"pip install failed: {out[-400:]}"


def _missing_modules(output: str) -> tuple[list[str], list[str]]:
    """(npm packages, python packages) named as missing in an error output."""
    npm = set()
    for pat in (r"Cannot find (?:module|package) '([^'./][^']*)'",
                r'Failed to resolve import "([^"./][^"]*)"',
                r"Could not resolve \"([^\"./][^\"]*)\""):
        for m in re.findall(pat, output):
            parts = m.split("/")
            npm.add("/".join(parts[:2]) if m.startswith("@") else parts[0])
    py = {m.split(".")[0] for m in re.findall(r"No module named ['\"]([\w.\-]+)['\"]", output)}
    builtin_node = {"fs", "path", "http", "https", "url", "os", "crypto", "events",
                    "stream", "util", "child_process", "net", "zlib"}
    return sorted(n for n in npm if n not in builtin_node and not n.startswith("node:")), sorted(py)


# ── Planning ─────────────────────────────────────────────────────────────────

_STACK_RULES = {
    "python": """Stack: Python.
- Entry point main.py, run with "python main.py".
- External packages go in "dependencies" (pip names).""",

    "react": """Stack: React + Vite (already bootstrapped with create-vite, JavaScript).
- Do NOT plan package.json, vite.config.js, index.html, package-lock.json or node_modules.
- Entry point is src/main.jsx; the main component is src/App.jsx. Add components under src/components/.
- Use plain CSS files (src/App.css, src/index.css) unless the user asks for a library.
- UI quality matters: a modern, polished, responsive design — clean layout with a header/navigation, cards, consistent spacing (8px scale), a coherent colour palette defined as CSS variables, readable typography (system-ui font stack), hover/focus states, loading and error states, and empty states. Never leave default unstyled HTML.
- "dependencies": extra npm packages the code imports (e.g. "react-router-dom", "axios"); [] if none.""",

    "node": """Stack: Node.js + Express REST API, ES modules.
- Plan package.json with "type": "module", a "start" script ("node src/server.js") and a
  "dev" script ("node --watch src/server.js"), and every package the code imports in "dependencies"
  (express, cors, dotenv, plus mongoose if MongoDB is used).
- src/server.js reads PORT from process.env (default 5000), enables cors and express.json(),
  and MUST expose GET /api/health returning JSON {"status":"ok","db":"connected"|"disconnected"|"none"}.
- Structure: src/server.js, src/routes/*.js, src/controllers/*.js, src/models/*.js (Mongoose).
- If MongoDB is used: connect with mongoose to process.env.MONGODB_URI (default
  "mongodb://127.0.0.1:27017/<project_name>"). A failed DB connection must be LOGGED, never crash
  the server — the server must still start and /api/health must report db "disconnected".
- Include a .env.example. "dependencies": [] (they live in package.json).""",

    "fullstack": """Stack: MERN full-stack — two folders.
server/ — Node.js + Express + Mongoose REST API, ES modules:
- Plan server/package.json with "type": "module", "start": "node src/server.js",
  "dev": "node --watch src/server.js", dependencies express, cors, dotenv, mongoose (+ any others used).
- server/src/server.js reads PORT from process.env (default 5000), uses cors + express.json(),
  mounts all routes under /api, and MUST expose GET /api/health returning
  {"status":"ok","db":"connected"|"disconnected"}.
- Mongoose connects to process.env.MONGODB_URI (default "mongodb://127.0.0.1:27017/<project_name>").
  A failed connection is LOGGED and must never crash the server.
- Structure: server/src/server.js, server/src/routes/, server/src/controllers/, server/src/models/.
- Include server/.env.example.
client/ — React + Vite, ALREADY bootstrapped with create-vite (JavaScript):
- Do NOT plan client/package.json or client/index.html.
- DO plan client/vite.config.js: React plugin and a dev proxy that forwards "/api" to
  `http://localhost:${process.env.API_PORT || 5000}`.
- Entry client/src/main.jsx, main component client/src/App.jsx, components in client/src/components/.
- The client calls the API with relative URLs ("/api/...") using fetch.
- Plan client/src/index.css and client/src/App.css with the full styling.
- UI quality matters: a modern, polished, responsive design — clean layout with a header/navigation, cards, consistent spacing (8px scale), a coherent colour palette defined as CSS variables, readable typography (system-ui font stack), hover/focus states, loading and error states, and empty states. Never leave default unstyled HTML.
- "dependencies": {"server": [], "client": [extra npm packages the client imports, e.g. "react-router-dom"]}.""",
}

_ENTRY = {"python": "main.py", "react": "src/main.jsx", "node": "src/server.js",
          "fullstack": "server/src/server.js"}


def _plan_project(description: str, stack: str, mongo: bool) -> dict:
    prompt = f"""You are a senior software architect. Plan a complete, minimal, RUNNABLE project.

Project description:
{description}

{_STACK_RULES[stack]}
{"MongoDB IS used by this project." if mongo else ""}

Return ONLY valid JSON (no markdown, no explanations) in this format:
{{
  "project_name": "kebab-or-snake-case-name",
  "summary": "one sentence describing what was built",
  "files": [
    {{"path": "{_ENTRY[stack]}", "description": "what this file contains and exports", "imports": ["other/project/file.js"]}}
  ],
  "dependencies": []
}}

Rules:
- Every file the code needs must be listed (project-relative paths, forward slashes).
- "imports" lists other PROJECT files this file imports (paths as in "files").
- List files in dependency order: files that are imported come before files that import them.
- Focused but complete and fully working — no placeholder features, no missing styling.
JSON:"""
    raw = _strip_fences(_ask(prompt))
    if "{" in raw:
        raw = raw[raw.find("{"): raw.rfind("}") + 1]
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Planner returned invalid JSON: {e}\nRaw: {raw[:400]}")
    files = [f for f in plan.get("files", []) if isinstance(f, dict) and f.get("path")]
    for f in files:
        f["path"] = f["path"].replace("\\", "/").lstrip("./")
    # Never let the model overwrite what create-vite generated.
    if stack == "react":
        files = [f for f in files if f["path"] not in ("package.json", "index.html", "vite.config.js")]
    if stack == "fullstack":
        files = [f for f in files if f["path"] not in ("client/package.json", "client/index.html")]
    plan["files"] = files
    return plan


# ── Writing & fixing files ───────────────────────────────────────────────────

def _lang_rules(path: str) -> str:
    if path.endswith(".py"):
        return ("Python rules: type hints, docstrings for public functions, "
                "an if __name__ == '__main__': guard in the entry point, imports matching "
                "the project structure exactly.")
    if path.endswith((".js", ".jsx", ".mjs", ".ts", ".tsx")):
        return ("JavaScript rules: ES modules (import/export, never require). Relative imports "
                "MUST include the file extension (./routes/todos.js). React components are "
                "function components with hooks. Handle async errors with try/catch. "
                "Every imported npm package must be in package.json / the dependency list.")
    if path.endswith("package.json"):
        return "Output valid JSON only — a complete package.json with name, version, type, scripts, dependencies."
    return ""


def _context_for(file_info: dict, written: dict[str, str], budget: int = 9000) -> str:
    wanted = [i.replace("\\", "/").lstrip("./") for i in file_info.get("imports", [])]
    chunks, used = [], 0
    ordered = sorted(written, key=lambda p: 0 if any(p.startswith(w.rsplit(".", 1)[0]) for w in wanted) else 1)
    for p in ordered:
        code = written[p][:3000]
        if used + len(code) > budget:
            break
        chunks.append(f"--- {p} ---\n{code}")
        used += len(code)
    return "\n\n".join(chunks)


def _write_file(file_info: dict, description: str, plan_files: list[dict], stack: str,
                project_dir: Path, written: dict[str, str]) -> str:
    path = file_info["path"]
    file_list = "\n".join(f"  - {f['path']}: {f.get('description', '')}" for f in plan_files)
    ctx = _context_for(file_info, written)
    prompt = f"""You are a senior developer writing production-quality code for a real project.

Project goal: {description}
{_STACK_RULES[stack]}

Project files:
{file_list}

{"Already written files (match their exports and APIs exactly):" + chr(10) + ctx if ctx else ""}

Write the COMPLETE content of: {path}
Purpose: {file_info.get('description', '')}
{_lang_rules(path)}

Output ONLY the raw file content — no markdown fences, no explanations, no placeholders or TODOs.
Content of {path}:"""
    code = _strip_fences(_ask(prompt))
    full = project_dir / path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(code + ("\n" if not code.endswith("\n") else ""), encoding="utf-8")
    return code


def _source_files(project_dir: Path, limit: int = 60) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(project_dir.rglob("*")):
        if len(out) >= limit:
            break
        rel = p.relative_to(project_dir)
        if any(part in _SKIP_DIRS for part in rel.parts) or not p.is_file():
            continue
        if p.name in _SKIP_FILES or p.suffix not in _SOURCE_EXT or p.stat().st_size > 60_000:
            continue
        try:
            out[rel.as_posix()] = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
    return out


def _files_in_error(output: str, files: dict[str, str]) -> list[str]:
    """Project files mentioned in an error output, most relevant first."""
    hits: list[tuple[int, str]] = []
    norm = output.replace("\\", "/")
    for rel in files:
        idx = norm.rfind(rel)
        if idx < 0:
            idx = norm.rfind("/" + Path(rel).name + ":")
        if idx >= 0:
            hits.append((idx, rel))
    # The LAST mention in a traceback is usually the failing frame.
    return [rel for _, rel in sorted(hits, reverse=True)][:3]


def _fix(error_output: str, description: str, stack: str, project_dir: Path,
         default_files: list[str], log) -> list[str]:
    files = _source_files(project_dir)
    targets = _files_in_error(error_output, files) or [f for f in default_files if f in files][:2]
    if not targets:
        return []
    fixed = []
    for target in targets:
        others = "\n\n".join(f"--- {p} ---\n{c[:1500]}" for p, c in files.items()
                             if p != target and not p.endswith(".md"))[:7000]
        prompt = f"""You are an expert debugger. Fix the file below so the project works.

Project goal: {description}
{_STACK_RULES[stack]}

Other project files (read-only context):
{others}

Error output:
{error_output[-3000:]}

File to fix: {target}
Current content:
{files[target]}

{_lang_rules(target)}
Output ONLY the complete fixed content of {target} — no markdown, no explanations.
Keep everything that works; fix every error the output shows that belongs to this file."""
        try:
            code = _strip_fences(_ask(prompt))
        except RateLimitError:
            raise
        except Exception as e:
            log(f"Could not fix {target}: {e}")
            continue
        if code and code.strip() != files[target].strip():
            (project_dir / target).write_text(code + "\n", encoding="utf-8")
            fixed.append(target)
            log(f"🔧 Fixed {target}")
    return fixed


# ── Checks per stack ─────────────────────────────────────────────────────────

def _project_layout(project_dir: Path) -> str:
    if (project_dir / "client" / "package.json").exists() and (project_dir / "server").is_dir():
        return "fullstack"
    pkg = project_dir / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
        return "react" if ("vite" in deps or "react" in deps) else "node"
    return "python"


def _python_entry(project_dir: Path) -> Path | None:
    for name in ("main.py", "app.py", "run.py"):
        if (project_dir / name).exists():
            return project_dir / name
    pys = sorted(project_dir.glob("*.py"))
    return pys[0] if pys else None


def _check(project: str, stack: str, project_dir: Path, mongo: bool, log) -> tuple[bool, str, Path]:
    """Runs the verification for a stack. Returns (ok, output, dir that failed/was checked)."""
    if stack == "python":
        entry = _python_entry(project_dir)
        if not entry:
            return False, "No Python entry point (main.py) found.", project_dir
        code, out = _run([sys.executable, entry.name], project_dir, timeout=30)
        if code == -1:            # still running after 30s: a server/GUI — fine
            return True, "Program kept running for 30s (server/GUI) without crashing.", project_dir
        ok = code == 0 and "Traceback (most recent call last)" not in out
        return ok, out or f"Exited with code {code}.", project_dir

    if stack == "react":
        code, out = _run([_NPM, "run", "build"], project_dir, timeout=300)
        return code == 0, out, project_dir

    if stack == "node":
        _stop_project(project)
        port = _free_port(5000)
        env = {"PORT": str(port)}
        if mongo:
            env["MONGODB_URI"] = os.environ.get("MONGODB_URI") or f"mongodb://127.0.0.1:27017/{project}"
        item = _start_process(project, "server", [_NPM, "start"], project_dir, env)
        ok, out = _wait_server(item, port, "/api/health")
        _stop_project(project)
        return ok, out, project_dir

    if stack == "fullstack":
        ok, out, d = _check(project, "node", project_dir / "server", mongo, log)
        if not ok:
            return ok, "[server] " + out, d
        code, out = _run([_NPM, "run", "build"], project_dir / "client", timeout=300)
        return code == 0, "[client] " + out, project_dir / "client"

    return False, f"Unknown stack {stack}", project_dir


def _start(project: str, stack: str, project_dir: Path, mongo: bool) -> str:
    """Starts the project for the user; servers keep running in the background."""
    _stop_project(project)
    if stack == "python":
        entry = _python_entry(project_dir)
        if not entry:
            return "No Python entry point found."
        item = _start_process(project, "app", [sys.executable, entry.name], project_dir)
        time.sleep(3)
        if item["proc"].poll() is not None:
            return f"{entry.name} finished:\n{_log_tail(item['log'], 1200)}"
        return f"{entry.name} is running (PID {item['proc'].pid})."

    api_port = None
    lines = []
    if stack in ("node", "fullstack"):
        server_dir = project_dir / "server" if stack == "fullstack" else project_dir
        api_port = _free_port(5000)
        env = {"PORT": str(api_port)}
        if mongo:
            env["MONGODB_URI"] = os.environ.get("MONGODB_URI") or f"mongodb://127.0.0.1:27017/{project}"
        item = _start_process(project, "server", [_NPM, "run", "dev"], server_dir, env)
        ok, info = _wait_server(item, api_port, "/api/health")
        lines.append(f"API: http://localhost:{api_port} ({'healthy' if ok else 'NOT healthy'})")
        if not ok:
            lines.append(info[-600:])

    if stack in ("react", "fullstack"):
        client_dir = project_dir / "client" if stack == "fullstack" else project_dir
        web_port = _free_port(5173)
        env = {"API_PORT": str(api_port or 5000)}
        item = _start_process(project, "client",
                              [_NPM, "run", "dev", "--", "--port", str(web_port), "--strictPort"],
                              client_dir, env)
        ok, info = _wait_server(item, web_port, "/", timeout=40)
        url = f"http://localhost:{web_port}"
        lines.append(f"App: {url} ({'running' if ok else 'NOT running'})")
        if ok:
            _open_in_browser(url)
        else:
            lines.append(info[-600:])
    return "\n".join(lines)


def _open_in_browser(url: str) -> None:
    try:
        from actions.browser_control import browser_control
        browser_control({"action": "go_to", "url": url})
    except Exception as e:
        print(f"[DevAgent] Could not open browser: {e}")


def _open_vscode(project_dir: Path) -> bool:
    code = shutil.which("code") or shutil.which("code.cmd")
    if not code:
        for cand in (Path.home() / "AppData/Local/Programs/Microsoft VS Code/bin/code.cmd",
                     Path("C:/Program Files/Microsoft VS Code/bin/code.cmd")):
            if cand.exists():
                code = str(cand)
                break
    if not code:
        return False
    try:
        subprocess.Popen([code, str(project_dir)], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


# ── Verify + auto-fix loop ───────────────────────────────────────────────────

def _verify_and_fix(project: str, stack: str, project_dir: Path, description: str,
                    mongo: bool, entry_files: list[str], log) -> tuple[bool, str, int]:
    last = ""
    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        log(f"Checking project (attempt {attempt}/{MAX_FIX_ATTEMPTS})…")
        ok, out, failed_dir = _check(project, stack, project_dir, mongo, log)
        last = out
        if ok:
            return True, out, attempt
        log(f"Problem found: {out.strip().splitlines()[-1][:140] if out.strip() else 'unknown'}")
        if attempt == MAX_FIX_ATTEMPTS:
            break

        npm_missing, py_missing = _missing_modules(out)
        if npm_missing:
            log(f"Installing missing npm packages: {', '.join(npm_missing)}")
            log(_npm_install(failed_dir, npm_missing))
            continue
        if py_missing:
            log(f"Installing missing Python packages: {', '.join(py_missing)}")
            log(_pip_install(py_missing, project_dir))
            continue
        if "EADDRINUSE" in out:
            continue          # a fresh free port is picked on the next check

        fixed = _fix(out, description, stack, project_dir, entry_files, log)
        if not fixed:
            log("No file could be fixed automatically.")
            break
    return False, last, MAX_FIX_ATTEMPTS


# ── Build ────────────────────────────────────────────────────────────────────

def _bootstrap_vite(parent: Path, folder: str) -> tuple[bool, str]:
    target = parent / folder
    if (target / "package.json").exists():
        return True, "Vite project already exists."
    if target.exists() and any(target.iterdir()):
        return False, f"{target} already has files but no package.json — refusing to overwrite."
    parent.mkdir(parents=True, exist_ok=True)
    code, out = _run([_NPX, "--yes", "create-vite@latest", folder, "--template", "react",
                      "--no-interactive"], parent, timeout=300)
    if not (target / "package.json").exists():
        # Older create-vite versions do not know --no-interactive.
        code, out = _run([_NPX, "--yes", "create-vite@latest", folder, "--template", "react"],
                         parent, timeout=300)
    if not (target / "package.json").exists():
        return False, f"create-vite failed: {out[-500:]}"
    return True, _npm_install(target)


def _build_project(description: str, language: str, project_name: str, player=None) -> str:
    def log(msg: str):
        print(f"[DevAgent] {msg}")
        if player:
            player.write_log(f"[DevAgent] {msg}")

    stack = _detect_stack(description, language)
    mongo = _wants_mongo(description, stack)
    log(f"Stack: {stack}{' + MongoDB' if mongo else ''}. Planning…")

    try:
        plan = _plan_project(description, stack, mongo)
    except RateLimitError:
        return "Gemini rate limit reached — try again in a minute."
    except Exception as e:
        return f"Planning failed: {e}"

    name = re.sub(r"[^\w\-]", "_", project_name or plan.get("project_name") or "jarvis_project")
    project_dir = PROJECTS_DIR / name
    if project_dir.exists() and any(project_dir.iterdir()):
        name = f"{name}_{time.strftime('%m%d%H%M')}"
        project_dir = PROJECTS_DIR / name
    project_dir.mkdir(parents=True, exist_ok=True)
    files = plan["files"]
    log(f"Project {name}: {len(files)} files → {project_dir}")

    if stack == "react":
        project_dir.rmdir()
        ok, msg = _bootstrap_vite(PROJECTS_DIR, name)
        log(msg)
        if not ok:
            return f"Could not create the React project: {msg}"
    elif stack == "fullstack":
        ok, msg = _bootstrap_vite(project_dir, "client")
        log(f"client: {msg}")
        if not ok:
            return f"Could not create the React client: {msg}"

    written: dict[str, str] = {}
    for fi in files:
        log(f"Writing {fi['path']}…")
        for attempt in range(2):
            try:
                written[fi["path"]] = _write_file(fi, description, files, stack, project_dir, written)
                break
            except RateLimitError:
                if attempt == 0:
                    log("Rate limit — waiting 20s…")
                    time.sleep(20)
                else:
                    log(f"Skipped {fi['path']} (rate limit).")
            except Exception as e:
                log(f"Failed to write {fi['path']}: {e}")
                break
    if not written:
        return "I could not write any project files."

    # Dependencies
    deps = plan.get("dependencies") or []
    if stack == "python":
        if deps:
            log(_pip_install([str(d) for d in deps], project_dir))
    elif stack == "react":
        if deps:
            log(_npm_install(project_dir, [str(d) for d in deps]))
    elif stack == "node":
        log(_npm_install(project_dir))
    elif stack == "fullstack":
        log(_npm_install(project_dir / "server"))
        client_deps = deps.get("client", []) if isinstance(deps, dict) else []
        if client_deps:
            log(_npm_install(project_dir / "client", [str(d) for d in client_deps]))

    if mongo:
        log(ensure_mongodb())

    _open_vscode(project_dir)
    entry_files = [f["path"] for f in files][-2:]
    try:
        ok, out, attempts = _verify_and_fix(name, stack, project_dir, description,
                                            mongo, entry_files, log)
    except RateLimitError:
        return f"Rate limit reached while fixing. Project saved at {project_dir}."

    _save_meta(project_dir, {"stack": stack, "mongo": mongo, "description": description})
    if not ok:
        return (f"Project '{name}' was written to {project_dir} but still fails after "
                f"{attempts} fix attempts. Last error:\n{out[-700:]}")

    started = _start(name, stack, project_dir, mongo)
    return (f"Project '{name}' ({stack}{' + MongoDB' if mongo else ''}) is built and verified "
            f"in {attempts} attempt{'s' if attempts > 1 else ''}. Saved at {project_dir}.\n"
            f"{plan.get('summary', '')}\n{started}")


# ── Existing projects ────────────────────────────────────────────────────────

def _save_meta(project_dir: Path, meta: dict) -> None:
    try:
        d = project_dir / ".jarvis"
        d.mkdir(exist_ok=True)
        (d / "project.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass


def _load_meta(project_dir: Path) -> dict:
    try:
        return json.loads((project_dir / ".jarvis" / "project.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_project(name: str) -> Path | None:
    if not name:
        return None
    p = Path(name).expanduser()
    if p.is_absolute() and p.is_dir():
        return p
    if (PROJECTS_DIR / name).is_dir():
        return PROJECTS_DIR / name
    if PROJECTS_DIR.is_dir():
        low = name.lower().replace(" ", "").replace("-", "").replace("_", "")
        for d in sorted(PROJECTS_DIR.iterdir(), key=lambda x: -x.stat().st_mtime):
            n = d.name.lower().replace("-", "").replace("_", "")
            if d.is_dir() and (low in n or n in low):
                return d
    return None


def _project_info(project_dir: Path) -> tuple[str, bool, str]:
    meta = _load_meta(project_dir)
    stack = meta.get("stack") or _project_layout(project_dir)
    mongo = meta.get("mongo")
    if mongo is None:
        blob = " ".join(_source_files(project_dir, 30).values()).lower()
        mongo = "mongoose" in blob or "mongodb" in blob
    return stack, bool(mongo), meta.get("description", "")


def _list_projects() -> str:
    if not PROJECTS_DIR.is_dir():
        return "No projects yet."
    rows = []
    for d in sorted(PROJECTS_DIR.iterdir(), key=lambda x: -x.stat().st_mtime)[:15]:
        if not d.is_dir():
            continue
        with _run_lock:
            items = [i for i in _running.get(d.name, []) if i["proc"].poll() is None]
        state = ("running " + " ".join(i["url"] or i["name"] for i in items)) if items else "stopped"
        rows.append(f"- {d.name} [{_project_info(d)[0]}] — {state}")
    return "Projects in " + str(PROJECTS_DIR) + ":\n" + "\n".join(rows) if rows else "No projects yet."


def dev_agent(
    parameters: dict,
    response=None,
    player=None,
    session_memory=None,
    speak=None,
) -> str:
    p            = parameters or {}
    action       = (p.get("action") or "build").lower().strip()
    description  = (p.get("description") or "").strip()
    language     = (p.get("language") or "").strip()
    project_name = (p.get("project_name") or "").strip()

    if action == "list":
        return _list_projects()

    if action == "stop":
        if project_name:
            d = _find_project(project_name)
            n = _stop_project(d.name if d else project_name)
            return f"Stopped {n} process(es) of {project_name}." if n else f"{project_name} was not running."
        with _run_lock:
            names = list(_running)
        total = sum(_stop_project(n) for n in names)
        return f"Stopped {total} process(es)." if total else "No projects were running."

    if action in ("run", "debug"):
        d = _find_project(project_name)
        if d is None:
            return f"Project '{project_name}' not found in {PROJECTS_DIR}. " + _list_projects()
        stack, mongo, saved_desc = _project_info(d)
        if mongo:
            print(f"[DevAgent] {ensure_mongodb()}")
        if action == "run":
            return f"{d.name} [{stack}]:\n" + _start(d.name, stack, d, mongo)

        def log(msg: str):
            print(f"[DevAgent] {msg}")
            if player:
                player.write_log(f"[DevAgent] {msg}")
        if stack in ("node", "react"):
            log(_npm_install(d))
        elif stack == "fullstack":
            log(_npm_install(d / "server"))
            log(_npm_install(d / "client"))
        try:
            ok, out, attempts = _verify_and_fix(
                d.name, stack, d, description or saved_desc or f"the {stack} project {d.name}",
                mongo, [], log)
        except RateLimitError:
            return "Rate limit reached while debugging."
        if ok:
            return (f"{d.name} works after {attempts} check(s).\n" + _start(d.name, stack, d, mongo))
        return f"{d.name} still fails after {attempts} attempts. Last error:\n{out[-800:]}"

    if not description:
        return "Please describe the project you want me to build."
    return _build_project(description, language, project_name, player)


# ── Tool declaration (auto-discovered by core/action_loader.py) ──────────────
TOOL = {
    "name": "dev_agent",
    "description": (
        "Builds, runs, debugs and stops real software projects in ~/Desktop/JarvisProjects. "
        "Stacks: Python, React (Vite), Node.js/Express REST API, MongoDB, and full-stack MERN "
        "(React client + Express/MongoDB server). 'build' plans and writes all files, installs "
        "packages, verifies (npm build / API health check / running Python), auto-fixes errors, "
        "then starts it and opens the app in the browser. Building takes minutes — prefer "
        "delegate_task(agent='dev') for builds so the conversation is not blocked."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "description": "build (default) | run | debug | stop | list"
            },
            "description": {
                "type": "STRING",
                "description": "For build: what the project should do, with every feature the user asked for"
            },
            "language": {
                "type": "STRING",
                "description": "python | react | node | fullstack (MERN). Auto-detected from the description when omitted."
            },
            "project_name": {
                "type": "STRING",
                "description": "Project folder name (build: optional; run/debug/stop: which project)"
            },
        },
        "required": []
    },
    "handler": dev_agent,
}

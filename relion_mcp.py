#!/usr/bin/env python3
"""
RELION MCP Server v3.0 for RELION 5.0.1

Usage:
  python relion_mcp.py                                  # stdio (Claude Code)
  python relion_mcp.py --transport http --port 8000     # HTTP  (OpenClaw)

Environment variables:
  RELION_PROJECT_DIR  — RELION project directory (default: cwd)
  RELION_BIN          — Path prefix for relion binaries (default: use PATH)
  RELION_THREADS      — Default thread count (default: 4)
  RELION_MPI          — Default MPI process count (default: 1)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RELION_BIN: str = os.environ.get("RELION_BIN", "")
PROJECT_DIR: str = os.environ.get("RELION_PROJECT_DIR", os.getcwd())
DEFAULT_THREADS: int = int(os.environ.get("RELION_THREADS", "4"))
DEFAULT_MPI: int = int(os.environ.get("RELION_MPI", "1"))
STAR_RE = re.compile(r"^_rln(\w+)\s+#(\d+)")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cmd(program: str) -> str:
    return str(Path(RELION_BIN) / program) if RELION_BIN else program


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else str(Path(PROJECT_DIR) / path)


def _mpi_prefix(program: str, mpi: int) -> List[str]:
    return ["mpirun", "-np", str(mpi), program] if mpi > 1 else [program]


def _next_job(job_type: str) -> str:
    d = Path(PROJECT_DIR) / job_type
    d.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        [x for x in d.iterdir() if x.is_dir() and x.name.startswith("job")]
    )
    n = int(existing[-1].name.replace("job", "")) + 1 if existing else 1
    return str(d / f"job{n:03d}")


def _mark_success(job_dir: str) -> None:
    Path(job_dir, "RELION_JOB_EXIT_SUCCESS").touch()


def _job_status(job_dir: str) -> str:
    p = Path(_resolve(job_dir))
    if (p / "RELION_JOB_EXIT_SUCCESS").exists():
        return "COMPLETED"
    if (p / "RELION_JOB_EXIT_FAILURE").exists():
        return "FAILED"
    if (p / "RELION_JOB_EXIT_ABORTED").exists():
        return "ABORTED"
    if p.exists():
        return "RUNNING_OR_IDLE"
    return "NOT_FOUND"


def _run_sync(cmd: List[str], cwd: Optional[str] = None, timeout: int = 600) -> dict:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=cwd or PROJECT_DIR, timeout=timeout)
        return {
            "returncode": r.returncode,
            "stdout": r.stdout[-4000:] if len(r.stdout) > 4000 else r.stdout,
            "stderr": r.stderr[-2000:] if len(r.stderr) > 2000 else r.stderr,
            "command": " ".join(cmd),
        }
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "stdout": "", "stderr": f"Timeout after {timeout}s",
                "command": " ".join(cmd)}
    except FileNotFoundError:
        return {"returncode": -1, "stdout": "", "stderr": f"Binary not found: {cmd[0]}",
                "command": " ".join(cmd)}


async def _run(cmd: List[str], cwd: Optional[str] = None, timeout: int = 600) -> dict:
    return await asyncio.get_running_loop().run_in_executor(
        None, _run_sync, cmd, cwd, timeout)


def _run_background(cmd: List[str], job_dir: str, cwd: Optional[str] = None) -> dict:
    os.makedirs(job_dir, exist_ok=True)
    stdout_path = os.path.join(job_dir, "run.out")
    stderr_path = os.path.join(job_dir, "run.err")
    cmd_str = " ".join(cmd)
    with open(os.path.join(job_dir, "run.cmd"), "w") as f:
        f.write(cmd_str + "\n")
    wrapper_path = os.path.join(job_dir, "run.sh")
    with open(wrapper_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"{cmd_str} > {stdout_path} 2> {stderr_path}\n")
        f.write("EXIT_CODE=$?\n")
        f.write(f"if [ $EXIT_CODE -eq 0 ]; then\n")
        f.write(f"  touch {os.path.join(job_dir, 'RELION_JOB_EXIT_SUCCESS')}\n")
        f.write(f"else\n")
        f.write(f"  touch {os.path.join(job_dir, 'RELION_JOB_EXIT_FAILURE')}\n")
        f.write(f"fi\nexit $EXIT_CODE\n")
    os.chmod(wrapper_path, 0o755)
    try:
        proc = subprocess.Popen(["bash", wrapper_path], cwd=cwd or PROJECT_DIR,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True)
        return {"pid": proc.pid, "command": cmd_str, "job_dir": job_dir,
                "stdout_log": stdout_path, "stderr_log": stderr_path}
    except FileNotFoundError:
        return {"pid": -1, "command": cmd_str, "job_dir": job_dir,
                "error": f"Binary not found: {cmd[0]}"}


def _format_launched(name: str, info: dict) -> str:
    if info.get("pid", -1) < 0:
        return (f"## {name}\n**Status:** ❌ Failed to launch\n"
                f"**Error:** {info.get('error', 'Unknown')}\n"
                f"**Command:** `{info.get('command', '')}`")
    return (f"## {name}\n**Status:** 🚀 Launched (background)\n"
            f"**Job:** `{info['job_dir']}`\n**PID:** `{info['pid']}`\n"
            f"**Command:** `{info['command']}`\n\n"
            f"📋 Utilisez `relion_job_status(job_dir=\"{info['job_dir']}\")` "
            f"et `relion_job_logs(job_dir=\"{info['job_dir']}\")` pour suivre.")


def _format_result(name: str, job_dir: str, result: dict) -> str:
    ok = result["returncode"] == 0
    status = "✅ Success" if ok else f"❌ Failed ({result['returncode']})"
    lines = [f"## {name}", f"**Status:** {status}",
             f"**Job:** `{job_dir}`", f"**Command:** `{result['command']}`"]
    if not ok and result["stderr"]:
        lines.append(f"\n**Error:**\n```\n{result['stderr'][:1500]}\n```")
    if ok and result["stdout"]:
        lines.append(f"\n**Output:**\n```\n{result['stdout'][-800:]}\n```")
    p = Path(job_dir)
    if p.exists():
        files = [f for f in p.iterdir() if f.suffix in (".star", ".mrc", ".mrcs")]
        if files:
            lines.append("\n**Output files:**")
            for f in sorted(files)[:15]:
                lines.append(f"  - `{f.name}` ({f.stat().st_size / 1e6:.1f} MB)")
    return "\n".join(lines)


def _parse_star(path: str, table: str = "data_") -> dict:
    path = _resolve(path)
    if not os.path.isfile(path):
        return {"error": f"File not found: {path}"}
    headers: Dict[str, int] = {}
    rows: List[Dict[str, str]] = []
    in_table = in_header = False
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith(table) or s == "loop_":
                if s.startswith(table):
                    in_table = True
                if s == "loop_":
                    in_header = True
                continue
            if in_header:
                m = STAR_RE.match(s)
                if m:
                    headers[m.group(1)] = int(m.group(2)) - 1
                    continue
                else:
                    in_header = False
            if in_table and headers and s and not s.startswith("#"):
                if s.startswith("data_") or s.startswith("loop_"):
                    break
                parts = s.split()
                if len(parts) >= len(headers):
                    rows.append({n: parts[i] if i < len(parts) else "" for n, i in headers.items()})
    return {"columns": list(headers.keys()), "num_rows": len(rows), "rows": rows[:200]}


# ---------------------------------------------------------------------------
# Preview / Confirm system
# ---------------------------------------------------------------------------


def _preview_params(params_dict: Dict[str, Any], tutorial_defaults: Dict[str, Any],
                    required_fields: List[str], label: str) -> str:
    lines = [f"## 🔍 Preview — {label}", ""]
    missing: List[str] = []
    for key in sorted(set(list(params_dict.keys()) + list(tutorial_defaults.keys()))):
        if key in ("confirm", "extra_args"):
            continue
        val = params_dict.get(key)
        tut = tutorial_defaults.get(key)
        is_required = key in required_fields
        if val is not None:
            lines.append(f"  ✏️  **{key}** = `{val}` (défini)")
        elif tut is not None:
            lines.append(f"  📋  **{key}** = `{tut}` (tutorial)")
        elif is_required:
            lines.append(f"  ❌  **{key}** — requis, non fourni")
            missing.append(key)
        else:
            lines.append(f"  ⬜  **{key}** = _(non défini, optionnel)_")
    lines.append("")
    if missing:
        lines.append(f"⚠️ **Paramètres requis manquants :** {', '.join(missing)}")
        lines.append("Fournissez-les puis relancez avec `confirm=True`.")
    else:
        lines.append("✅ Tous les paramètres requis sont présents.\n"
                      "→ Relancez avec **`confirm=True`** pour exécuter le job.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("relion_mcp")


class Fmt(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


# =====================================================================
# READ-ONLY TOOL 1 — Project Info
# =====================================================================

class ProjectInfoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    response_format: Fmt = Field(default=Fmt.MARKDOWN)


@mcp.tool(name="relion_project_info",
          annotations={"readOnlyHint": True, "destructiveHint": False,
                       "idempotentHint": True, "openWorldHint": False})
async def relion_project_info(params: ProjectInfoInput) -> str:
    """Get overview of the RELION project: job types, counts, latest status."""
    pdir = Path(PROJECT_DIR)
    types = ["Import", "MotionCorr", "CtfFind", "ManualPick", "AutoPick",
             "Extract", "Select", "Class2D", "Class3D", "InitialModel",
             "Refine3D", "CtfRefine", "Polish", "MaskCreate", "PostProcess",
             "LocalRes", "ModelAngelo", "External", "DynaMight", "Blush"]
    info: Dict[str, Any] = {"project_dir": str(pdir), "jobs": {}}
    for t in types:
        td = pdir / t
        if td.is_dir():
            jobs = sorted([d.name for d in td.iterdir() if d.is_dir()])
            info["jobs"][t] = {"count": len(jobs), "latest": jobs[-1] if jobs else None,
                               "status": _job_status(f"{t}/{jobs[-1]}") if jobs else "EMPTY"}
    if params.response_format == Fmt.JSON:
        return json.dumps(info, indent=2)
    lines = [f"# RELION Project: `{pdir.name}`", f"**Path:** `{pdir}`", ""]
    if not info["jobs"]:
        lines.append("_No jobs — new project._")
    for t, d in info["jobs"].items():
        lines.append(f"- **{t}** — {d['count']} job(s), latest: `{d['latest']}` ({d['status']})")
    return "\n".join(lines)


# =====================================================================
# READ-ONLY TOOL 2 — Read STAR
# =====================================================================

class ReadStarInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    star_file: str = Field(..., description="Path to STAR file")
    table_name: str = Field(default="data_")
    max_rows: int = Field(default=20, ge=1, le=200)
    response_format: Fmt = Field(default=Fmt.MARKDOWN)


@mcp.tool(name="relion_read_star",
          annotations={"readOnlyHint": True, "destructiveHint": False,
                       "idempotentHint": True, "openWorldHint": False})
async def relion_read_star(params: ReadStarInput) -> str:
    """Parse a RELION STAR file and return columns + data rows."""
    p = _parse_star(params.star_file, params.table_name)
    if "error" in p:
        return f"Error: {p['error']}"
    p["rows"] = p["rows"][:params.max_rows]
    if params.response_format == Fmt.JSON:
        return json.dumps(p, indent=2)
    lines = [f"## `{params.star_file}`",
             f"**Columns:** {len(p['columns'])} | **Rows:** {p['num_rows']}",
             "**Fields:** " + ", ".join(f"`{c}`" for c in p["columns"]), ""]
    for i, row in enumerate(p["rows"]):
        lines.append(f"Row {i+1}: " + " | ".join(f"{k}={v}" for k, v in list(row.items())[:8]))
    if p["num_rows"] > params.max_rows:
        lines.append(f"\n_{p['num_rows'] - params.max_rows} more rows…_")
    return "\n".join(lines)


# =====================================================================
# READ-ONLY TOOL 3 — Job Status 
# =====================================================================

def _check_pid_alive(job_dir: str) -> Optional[int]:
    run_sh = Path(_resolve(job_dir)) / "run.sh"
    if not run_sh.exists():
        return None
    try:
        result = subprocess.run(["pgrep", "-f", str(run_sh)],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip().splitlines()[0])
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return None


class JobStatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_dir: str = Field(..., description="Job directory, e.g. 'Class2D/job015'")


@mcp.tool(name="relion_job_status",
          annotations={"readOnlyHint": True, "destructiveHint": False,
                       "idempotentHint": True, "openWorldHint": False})
async def relion_job_status(params: JobStatusInput) -> str:
    """Check execution status of a RELION job with PID detection and stderr tail."""
    jd = Path(_resolve(params.job_dir))
    status = _job_status(params.job_dir)
    pid_info = ""
    if status == "RUNNING_OR_IDLE":
        pid = _check_pid_alive(params.job_dir)
        if pid:
            status = "🔄 RUNNING"
            pid_info = f"\n**PID:** `{pid}`"
        else:
            status = "⚠️ IDLE (no process — may have crashed)"
    files: List[str] = []
    if jd.exists():
        for f in sorted(jd.iterdir()):
            if f.suffix in (".star", ".mrc", ".mrcs", ".log", ".out", ".err"):
                sz = f.stat().st_size
                files.append(f"  - `{f.name}` ({sz/1e6:.1f} MB)" if sz > 1e6
                             else f"  - `{f.name}` ({sz/1e3:.1f} KB)")
    err_tail = ""
    err_file = jd / "run.err"
    if err_file.exists() and err_file.stat().st_size > 0:
        with open(err_file) as ef:
            tail = ef.read()[-1000:]
            if tail.strip():
                err_tail = f"\n**stderr (tail):**\n```\n{tail.strip()}\n```"
    return (f"## `{params.job_dir}`\n**Status:** {status}{pid_info}\n"
            + ("\n".join(files) if files else "_No output files._") + err_tail)


# =====================================================================
# READ-ONLY TOOL 4 — Job Logs
# =====================================================================

class JobLogsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_dir: str = Field(..., description="Job directory, e.g. 'MotionCorr/job001'")
    stream: str = Field(default="both", description="'stdout', 'stderr', or 'both'")
    tail: int = Field(default=100, ge=1, le=500)


@mcp.tool(name="relion_job_logs",
          annotations={"readOnlyHint": True, "destructiveHint": False,
                       "idempotentHint": True, "openWorldHint": False})
async def relion_job_logs(params: JobLogsInput) -> str:
    """Read stdout/stderr logs from a RELION background job."""
    jd = Path(_resolve(params.job_dir))
    if not jd.exists():
        return f"❌ Job directory not found: `{params.job_dir}`"
    lines = [f"## Logs — `{params.job_dir}`", ""]
    for sname, fname in [("stdout", "run.out"), ("stderr", "run.err")]:
        if params.stream not in ("both", sname):
            continue
        lf = jd / fname
        if not lf.exists():
            lines.append(f"### {sname}\n_`{fname}` not found._\n")
            continue
        with open(lf) as f:
            cl = f.read().splitlines()
        t = cl[-params.tail:]
        lines.append(f"### {sname} ({lf.stat().st_size/1e3:.1f} KB, {len(cl)} lines)")
        if len(cl) > params.tail:
            lines.append(f"_(last {params.tail} lines)_")
        lines.append(f"```\n{chr(10).join(t)}\n```\n")
    return "\n".join(lines)


# =====================================================================
# READ-ONLY TOOL 5 — Suggest Next Step
# =====================================================================

class SuggestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


@mcp.tool(name="relion_suggest_next_step",
          annotations={"readOnlyHint": True, "destructiveHint": False,
                       "idempotentHint": True, "openWorldHint": False})
async def relion_suggest_next_step(params: SuggestInput) -> str:
    """Analyze project state and recommend the next pipeline step."""
    pdir = Path(PROJECT_DIR)
    pipeline = [
        ("Import", "relion_import", "Import movies/micrographs"),
        ("MotionCorr", "relion_motioncorr", "Motion correction"),
        ("CtfFind", "relion_ctffind", "CTF estimation"),
        ("AutoPick", "relion_autopick", "Particle picking"),
        ("Extract", "relion_extract", "Particle extraction"),
        ("Class2D", "relion_class2d", "2D classification"),
        ("InitialModel", "relion_initial_model", "Ab-initio 3D model (VDAM)"),
        ("Class3D", "relion_class3d", "3D classification"),
        ("Refine3D", "relion_refine3d", "3D refinement"),
        ("MaskCreate", "relion_mask_create", "Solvent mask creation"),
        ("PostProcess", "relion_postprocess", "Post-processing & resolution"),
        ("CtfRefine", "relion_ctf_refine", "CTF refinement"),
        ("Polish", "relion_bayesian_polishing", "Bayesian polishing"),
        ("LocalRes", "relion_local_resolution", "Local resolution estimation"),
    ]
    completed = set()
    for step, _, _ in pipeline:
        sd = pdir / step
        if sd.is_dir():
            for job in sorted(sd.iterdir()):
                if (job / "RELION_JOB_EXIT_SUCCESS").exists():
                    completed.add(step)
                    break
    lines = ["## Pipeline Progress\n"]
    for step, tool, desc in pipeline:
        icon = "✅" if step in completed else "⬜"
        lines.append(f"{icon} **{step}** — {desc} → `{tool}`")
    lines.append("\n---\n")
    nxt = next(((s, t, d) for s, t, d in pipeline if s not in completed), None)
    if nxt:
        lines.append(f"**Next:** {nxt[0]} — {nxt[2]}\n**Tool:** `{nxt[1]}`")
    else:
        lines.append("All steps done! Consider ModelAngelo or Blush.")
    return "\n".join(lines)


# =====================================================================
# READ-ONLY TOOL 6 — Run Any Command (escape hatch)
# =====================================================================

class RunCommandInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    program: str = Field(..., description="RELION program, e.g. 'relion_mask_create'")
    arguments: List[str] = Field(default_factory=list)
    job_type: str = Field(default="External")
    timeout: int = Field(default=3600, ge=10, le=604800)

    @field_validator("program")
    @classmethod
    def validate_program(cls, v: str) -> str:
        if not v.startswith("relion"):
            raise ValueError("Program must start with 'relion'")
        if any(c in v for c in ";|&`$()"):
            raise ValueError("Invalid characters")
        return v


@mcp.tool(name="relion_run_command",
          annotations={"readOnlyHint": False, "destructiveHint": True,
                       "idempotentHint": False, "openWorldHint": False})
async def relion_run_command(params: RunCommandInput) -> str:
    """Run any relion_* binary with custom arguments. Advanced escape hatch."""
    job_dir = _next_job(params.job_type)
    os.makedirs(job_dir, exist_ok=True)
    cmd = [_cmd(params.program)] + params.arguments
    result = await _run(cmd, timeout=params.timeout)
    if result["returncode"] == 0:
        _mark_success(job_dir)
    return _format_result(params.program, job_dir, result)


# =====================================================================
# READ-ONLY TOOL 7 — RELION Help
# =====================================================================

def _parse_help_output(raw: str) -> Dict[str, Any]:
    sections: List[Dict[str, Any]] = []
    cur_sec = "General"
    cur_opts: List[Dict[str, str]] = []
    sec_re = re.compile(r"^=+\s*(.+?)\s*=+$")
    opt_re = re.compile(r"^\s*(--\S+)\s+(.+?)(?:\s+\[([^\]]*)\]|\s+\(([^)]*)\))?\s*$")
    opt_simple = re.compile(r"^\s*(--\S+)\s+(.+)$")
    for line in raw.splitlines():
        ls = line.rstrip()
        m = sec_re.match(ls)
        if m:
            if cur_opts:
                sections.append({"name": cur_sec, "options": cur_opts})
                cur_opts = []
            cur_sec = m.group(1).strip()
            continue
        if ls.endswith(":") and not ls.startswith("-") and len(ls) < 80 and not ls.startswith(" "):
            if cur_opts:
                sections.append({"name": cur_sec, "options": cur_opts})
                cur_opts = []
            cur_sec = ls.rstrip(":")
            continue
        m2 = opt_re.match(ls)
        if m2:
            dv = m2.group(3) if m2.group(3) is not None else m2.group(4)
            cur_opts.append({"flag": m2.group(1), "description": m2.group(2).strip(),
                             "default": dv if dv else ""})
            continue
        m3 = opt_simple.match(ls)
        if m3:
            cur_opts.append({"flag": m3.group(1), "description": m3.group(2).strip(), "default": ""})
    if cur_opts:
        sections.append({"name": cur_sec, "options": cur_opts})
    return {"sections": sections, "total_flags": sum(len(s["options"]) for s in sections),
            "raw_truncated": raw[-500:] if len(raw) > 500 else raw}


class HelpInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    program: str = Field(..., description="RELION binary name, e.g. 'relion_refine'")
    search: Optional[str] = Field(default=None, description="Filter flags by keyword")
    response_format: Fmt = Field(default=Fmt.MARKDOWN)

    @field_validator("program")
    @classmethod
    def validate_program(cls, v: str) -> str:
        if not v.startswith("relion"):
            raise ValueError("Program must start with 'relion'")
        if any(c in v for c in ";|&`$(){}[]"):
            raise ValueError("Invalid characters")
        return v


@mcp.tool(name="relion_help",
          annotations={"readOnlyHint": True, "destructiveHint": False,
                       "idempotentHint": True, "openWorldHint": False})
async def relion_help(params: HelpInput) -> str:
    """Run `relion_<program> --help` and return parsed flags with descriptions and defaults.
    Use `search` to filter by keyword."""
    cmd = [_cmd(params.program), "--help"]
    result = await _run(cmd, timeout=30)
    raw = (result.get("stdout", "") + "\n" + result.get("stderr", "")).strip()
    if not raw:
        return f"❌ No output from `{params.program} --help`. Binary may not be in PATH."
    parsed = _parse_help_output(raw)
    if params.search:
        needle = params.search.lower()
        parsed["sections"] = [
            {"name": s["name"], "options": [o for o in s["options"]
             if needle in o["flag"].lower() or needle in o["description"].lower()]}
            for s in parsed["sections"]]
        parsed["sections"] = [s for s in parsed["sections"] if s["options"]]
        parsed["total_flags"] = sum(len(s["options"]) for s in parsed["sections"])
    if params.response_format == Fmt.JSON:
        return json.dumps({"program": params.program, **parsed}, indent=2)
    lines = [f"## `{params.program} --help`", f"**Flags:** {parsed['total_flags']}"]
    if params.search:
        lines.append(f"**Filter:** `{params.search}`")
    lines.append("")
    if not parsed["sections"]:
        lines.append("_No flags matched._\n```\n" + parsed["raw_truncated"] + "\n```")
        return "\n".join(lines)
    for sec in parsed["sections"]:
        lines.append(f"### {sec['name']}")
        for o in sec["options"]:
            d = f" `[{o['default']}]`" if o["default"] else ""
            lines.append(f"  `{o['flag']}` — {o['description']}{d}")
        lines.append("")
    return "\n".join(lines)


# =====================================================================
# PIPELINE TOOL 1 — Import 
# =====================================================================

_IMPORT_TUT: Dict[str, Any] = {
    "voltage": 200.0, "pixel_size": 0.885, "cs": 1.4, "q0": 0.1,
    "optics_group_name": "opticsGroup1", "is_movie": True,
    "beamtilt_x": 0.0, "beamtilt_y": 0.0,
}
_IMPORT_REQ = ["input_files", "pixel_size"]


class ImportInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_files: str = Field(..., description="Wildcard, e.g. 'Movies/*.tiff'")
    pixel_size: float = Field(..., description="Pixel size Å (--angpix)", gt=0, le=20)
    voltage: float = Field(default=200.0, description="kV (--kV)", ge=60, le=400)
    cs: float = Field(default=1.4, description="Spherical aberration mm (--Cs)")
    q0: float = Field(default=0.1, description="Amplitude contrast (--Q0)", ge=0, le=1)
    optics_group_name: str = Field(default="opticsGroup1")
    is_movie: bool = Field(default=True, description="True=movies, False=micrographs")
    mtf: Optional[str] = Field(default=None, description="MTF STAR file (--mtf)")
    beamtilt_x: float = Field(default=0.0, description="Beamtilt X mrad (--beamtilt_x)")
    beamtilt_y: float = Field(default=0.0, description="Beamtilt Y mrad (--beamtilt_y)")
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False)


@mcp.tool(name="relion_import",
          annotations={"readOnlyHint": False, "destructiveHint": False,
                       "idempotentHint": False, "openWorldHint": False})
async def relion_import(params: ImportInput) -> str:
    """Import raw movies or micrographs. confirm=False to preview."""
    if not params.confirm:
        return _preview_params(params.model_dump(), _IMPORT_TUT, _IMPORT_REQ, "Import")
    job_dir = _next_job("Import")
    os.makedirs(job_dir, exist_ok=True)
    cmd = [_cmd("relion_import"), "--i", params.input_files,
           "--odir", job_dir + "/", "--ofile",
           "movies.star" if params.is_movie else "micrographs.star",
           "--do_movies" if params.is_movie else "--do_micrographs",
           "--optics_group_name", params.optics_group_name,
           "--angpix", str(params.pixel_size), "--kV", str(params.voltage),
           "--Cs", str(params.cs), "--Q0", str(params.q0)]
    if params.mtf:
        cmd += ["--mtf", _resolve(params.mtf)]
    if params.beamtilt_x != 0:
        cmd += ["--beamtilt_x", str(params.beamtilt_x)]
    if params.beamtilt_y != 0:
        cmd += ["--beamtilt_y", str(params.beamtilt_y)]
    if params.extra_args:
        cmd += params.extra_args
    result = await _run(cmd)
    if result["returncode"] == 0:
        _mark_success(job_dir)
    return _format_result("Import", job_dir, result)


# =====================================================================
# PIPELINE TOOL 2 — MotionCorr 
# =====================================================================

_MOTIONCORR_TUT: Dict[str, Any] = {
    "dose_per_frame": 1.277, "voltage": 200.0, "patch_x": 5, "patch_y": 5,
    "bfactor": 150, "bin_factor": 1, "first_frame": 1, "last_frame": -1,
    "gain_rot": 0, "gain_flip": 0, "float16": True, "save_noDW": False,
    "save_ps": True, "group_frames": 3, "preexposure": 0.0,
    "eer_grouping": 32, "threads": DEFAULT_THREADS, "mpi": DEFAULT_MPI,
}
_MOTIONCORR_REQ = ["input_star", "dose_per_frame"]


class MotionCorrInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_star: str = Field(..., description="STAR file with movies")
    dose_per_frame: float = Field(default=1.277, description="e-/Å²/frame (--dose_per_frame)", gt=0)
    voltage: float = Field(default=200.0, ge=60, le=400)
    pixel_size: float = Field(default=-1, description="Å, -1=from STAR")
    patch_x: int = Field(default=5, ge=1, le=20)
    patch_y: int = Field(default=5, ge=1, le=20)
    bfactor: int = Field(default=150, ge=0, le=1500)
    bin_factor: int = Field(default=1, ge=1, le=8)
    gainref: Optional[str] = Field(default=None, description="Gain reference MRC")
    gain_rot: int = Field(default=0, ge=0, le=3)
    gain_flip: int = Field(default=0, ge=0, le=2)
    defect_file: Optional[str] = Field(default=None, description="Defect file (--defect_file)")
    first_frame: int = Field(default=1, ge=1)
    last_frame: int = Field(default=-1)
    preexposure: float = Field(default=0.0, description="Pre-exposure e-/Å² (--preexposure)", ge=0)
    eer_grouping: int = Field(default=32, description="EER fractionation (--eer_grouping)", ge=1)
    float16: bool = Field(default=True)
    save_noDW: bool = Field(default=False)
    save_ps: bool = Field(default=True)
    group_frames: int = Field(default=3, description="Group N frames for PS (--groupNframes)", ge=1)
    threads: int = Field(default=DEFAULT_THREADS, ge=1, le=64)
    mpi: int = Field(default=DEFAULT_MPI, ge=1, le=128)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False)


@mcp.tool(name="relion_motioncorr",
          annotations={"readOnlyHint": False, "destructiveHint": False,
                       "idempotentHint": False, "openWorldHint": False})
async def relion_motioncorr(params: MotionCorrInput) -> str:
    """Beam-induced motion correction. confirm=False to preview."""
    if not params.confirm:
        return _preview_params(params.model_dump(), _MOTIONCORR_TUT, _MOTIONCORR_REQ, "Motion Correction")
    job_dir = _next_job("MotionCorr")
    os.makedirs(job_dir, exist_ok=True)
    prog = _cmd("relion_run_motioncorr" + ("_mpi" if params.mpi > 1 else ""))
    cmd = _mpi_prefix(prog, params.mpi) + [
        "--i", _resolve(params.input_star), "--o", job_dir + "/",
        "--use_own", "--dose_weighting",
        "--dose_per_frame", str(params.dose_per_frame),
        "--voltage", str(params.voltage),
        "--patch_x", str(params.patch_x), "--patch_y", str(params.patch_y),
        "--bfactor", str(params.bfactor), "--bin_factor", str(params.bin_factor),
        "--first_frame_sum", str(params.first_frame),
        "--last_frame_sum", str(params.last_frame),
        "--groupNframes", str(params.group_frames),
        "--preexposure", str(params.preexposure),
        "--eer_grouping", str(params.eer_grouping),
        "--j", str(params.threads)]
    if params.pixel_size > 0:
        cmd += ["--angpix", str(params.pixel_size)]
    if params.gainref:
        cmd += ["--gainref", _resolve(params.gainref)]
    if params.gain_rot != 0:
        cmd += ["--gain_rot", str(params.gain_rot)]
    if params.gain_flip != 0:
        cmd += ["--gain_flip", str(params.gain_flip)]
    if params.defect_file:
        cmd += ["--defect_file", _resolve(params.defect_file)]
    if params.float16:
        cmd.append("--float16")
    if params.save_noDW:
        cmd.append("--save_noDW")
    if params.save_ps:
        cmd.append("--save_ps")
    if params.extra_args:
        cmd += params.extra_args
    info = _run_background(cmd, job_dir)
    return _format_launched("Motion Correction", info)


# =====================================================================
# PIPELINE TOOL 3 — CTF 
# =====================================================================

_CTF_TUT: Dict[str, Any] = {
    "box": 512, "res_min": 30.0, "res_max": 5.0,
    "df_min": 5000.0, "df_max": 50000.0, "f_step": 500.0, "d_ast": 100.0,
    "ctffind_exe": "ctffind", "is_ctffind4": True,
    "use_given_ps": True, "do_phaseshift": False, "fast_search": True,
    "use_noDW": False, "ctf_win": -1,
    "threads": DEFAULT_THREADS, "mpi": DEFAULT_MPI,
}
_CTF_REQ = ["input_star"]


class CtfFindInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_star: str = Field(..., description="STAR with corrected micrographs")
    box: int = Field(default=512, ge=64, le=2048)
    res_min: float = Field(default=30.0, gt=0)
    res_max: float = Field(default=5.0, gt=0)
    df_min: float = Field(default=5000.0, ge=0)
    df_max: float = Field(default=50000.0, ge=0)
    f_step: float = Field(default=500.0, gt=0)
    d_ast: float = Field(default=100.0, ge=0)
    ctffind_exe: str = Field(default="ctffind")
    is_ctffind4: bool = Field(default=True)
    use_given_ps: bool = Field(default=True, description="Use power spectra from MotionCorr (--use_given_ps)")
    use_noDW: bool = Field(default=False, description="Use non-dose-weighted mics (--use_noDW)")
    do_phaseshift: bool = Field(default=False, description="Estimate phase shift (--do_phaseshift)")
    fast_search: bool = Field(default=True, description="Fast search, not exhaustive (--fast_search)")
    ctf_win: int = Field(default=-1, description="Window size for CTF estimation (--ctfWin), -1=full")
    threads: int = Field(default=DEFAULT_THREADS, ge=1, le=64)
    mpi: int = Field(default=DEFAULT_MPI, ge=1, le=128)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False)


@mcp.tool(name="relion_ctffind",
          annotations={"readOnlyHint": False, "destructiveHint": False,
                       "idempotentHint": False, "openWorldHint": False})
async def relion_ctffind(params: CtfFindInput) -> str:
    """CTF estimation via CTFFIND4. confirm=False to preview."""
    if not params.confirm:
        return _preview_params(params.model_dump(), _CTF_TUT, _CTF_REQ, "CTF Estimation")
    job_dir = _next_job("CtfFind")
    os.makedirs(job_dir, exist_ok=True)
    prog = _cmd("relion_run_ctffind" + ("_mpi" if params.mpi > 1 else ""))
    cmd = _mpi_prefix(prog, params.mpi) + [
        "--i", _resolve(params.input_star), "--o", job_dir + "/",
        "--ctffind_exe", params.ctffind_exe,
        "--Box", str(params.box), "--ResMin", str(params.res_min),
        "--ResMax", str(params.res_max), "--dFMin", str(params.df_min),
        "--dFMax", str(params.df_max), "--FStep", str(params.f_step),
        "--dAst", str(params.d_ast), "--j", str(params.threads)]
    if params.is_ctffind4:
        cmd.append("--is_ctffind4")
    if params.use_given_ps:
        cmd.append("--use_given_ps")
    if params.use_noDW:
        cmd.append("--use_noDW")
    if params.do_phaseshift:
        cmd.append("--do_phaseshift")
    if params.fast_search:
        cmd.append("--fast_search")
    if params.ctf_win > 0:
        cmd += ["--ctfWin", str(params.ctf_win)]
    if params.extra_args:
        cmd += params.extra_args
    info = _run_background(cmd, job_dir)
    return _format_launched("CTF Estimation", info)


# =====================================================================
# PIPELINE TOOL 4 — AutoPick 
# =====================================================================

_AUTOPICK_TUT: Dict[str, Any] = {
    "particle_diameter": 180.0, "angpix": -1.0, "use_log": True,
    "log_diam_min": 150.0, "log_diam_max": 180.0,
    "log_adjust_threshold": 0.0, "log_upper_threshold": 5.0,
    "log_use_ctf": True, "log_invert": False, "log_maxres": 20.0,
    "threads": DEFAULT_THREADS, "mpi": DEFAULT_MPI,
}
_AUTOPICK_REQ = ["input_star", "particle_diameter"]


class AutoPickInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_star: str = Field(..., description="Micrograph STAR with CTF")
    particle_diameter: float = Field(..., gt=0, le=2000)
    angpix: float = Field(default=-1.0, description="Pixel size Å, -1=from STAR")
    use_log: bool = Field(default=True)
    log_diam_min: Optional[float] = Field(default=150.0)
    log_diam_max: Optional[float] = Field(default=180.0)
    log_adjust_threshold: float = Field(default=0.0)
    log_upper_threshold: float = Field(default=5.0, description="Upper threshold (--LoG_upper_threshold)")
    log_use_ctf: bool = Field(default=True)
    log_invert: bool = Field(default=False, description="White particles (--LoG_invert)")
    log_maxres: float = Field(default=20.0, description="Max resolution Å (--LoG_maxres)")
    ref: Optional[str] = Field(default=None, description="Template ref for non-LoG")
    lowpass: float = Field(default=-1)
    threads: int = Field(default=DEFAULT_THREADS, ge=1, le=64)
    mpi: int = Field(default=DEFAULT_MPI, ge=1, le=128)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False)


@mcp.tool(name="relion_autopick",
          annotations={"readOnlyHint": False, "destructiveHint": False,
                       "idempotentHint": False, "openWorldHint": False})
async def relion_autopick(params: AutoPickInput) -> str:
    """Auto-pick particles: LoG or template-based. confirm=False to preview."""
    if not params.confirm:
        return _preview_params(params.model_dump(), _AUTOPICK_TUT, _AUTOPICK_REQ, "AutoPick")
    job_dir = _next_job("AutoPick")
    os.makedirs(job_dir, exist_ok=True)
    prog = _cmd("relion_autopick" + ("_mpi" if params.mpi > 1 else ""))
    cmd = _mpi_prefix(prog, params.mpi) + [
        "--i", _resolve(params.input_star), "--odir", job_dir + "/",
        "--particle_diameter", str(params.particle_diameter)]
    if params.angpix > 0:
        cmd += ["--angpix", str(params.angpix)]
    if params.use_log:
        cmd += ["--LoG", "--LoG_adjust_threshold", str(params.log_adjust_threshold),
                "--LoG_upper_threshold", str(params.log_upper_threshold),
                "--LoG_maxres", str(params.log_maxres)]
        if params.log_diam_min is not None:
            cmd += ["--LoG_diam_min", str(params.log_diam_min)]
        if params.log_diam_max is not None:
            cmd += ["--LoG_diam_max", str(params.log_diam_max)]
        if params.log_use_ctf:
            cmd.append("--LoG_use_ctf")
        if params.log_invert:
            cmd.append("--LoG_invert")
    else:
        if params.ref:
            cmd += ["--ref", _resolve(params.ref)]
        if params.lowpass > 0:
            cmd += ["--lowpass", str(params.lowpass)]
    if params.extra_args:
        cmd += params.extra_args
    info = _run_background(cmd, job_dir)
    return _format_launched("Auto-Picking", info)


# =====================================================================
# PIPELINE TOOL 5 — Extract 
# =====================================================================

_EXTRACT_TUT: Dict[str, Any] = {
    "extract_size": 256, "rescale": 64, "coord_suffix": "_autopick.star",
    "normalize": True, "bg_radius": 200, "invert_contrast": True,
    "white_dust": -1.0, "black_dust": -1.0, "float16": True,
    "use_fom_threshold": False, "minimum_pick_fom": -999.0,
}
_EXTRACT_REQ = ["input_star", "extract_size"]


class ExtractInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_star: str = Field(...)
    coord_suffix: str = Field(default="_autopick.star")
    extract_size: int = Field(..., ge=16, le=2048)
    rescale: Optional[int] = Field(default=64, ge=16, le=2048)
    normalize: bool = Field(default=True)
    bg_radius: int = Field(default=200)
    invert_contrast: bool = Field(default=True)
    white_dust: float = Field(default=-1.0)
    black_dust: float = Field(default=-1.0)
    float16: bool = Field(default=True)
    reextract_data_star: Optional[str] = Field(default=None, description="Re-extract from this _data.star (--reextract_data_star)")
    recenter: bool = Field(default=False, description="Re-center refined coordinates (--recenter)")
    use_fom_threshold: bool = Field(default=False, description="Apply FOM threshold (--use_fom_threshold)")
    minimum_pick_fom: float = Field(default=-999.0, description="Minimum autopick FOM (--minimum_pick_fom)")
    mpi: int = Field(default=DEFAULT_MPI, ge=1, le=128)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False)


@mcp.tool(name="relion_extract",
          annotations={"readOnlyHint": False, "destructiveHint": False,
                       "idempotentHint": False, "openWorldHint": False})
async def relion_extract(params: ExtractInput) -> str:
    """Extract particles from micrographs. confirm=False to preview."""
    if not params.confirm:
        return _preview_params(params.model_dump(), _EXTRACT_TUT, _EXTRACT_REQ, "Particle Extraction")
    job_dir = _next_job("Extract")
    os.makedirs(job_dir, exist_ok=True)
    prog = _cmd("relion_preprocess" + ("_mpi" if params.mpi > 1 else ""))
    cmd = _mpi_prefix(prog, params.mpi) + [
        "--i", _resolve(params.input_star),
        "--part_star", os.path.join(job_dir, "particles.star"),
        "--part_dir", os.path.join(job_dir, "Particles/"),
        "--extract", "--extract_size", str(params.extract_size)]
    if params.reextract_data_star:
        cmd += ["--reextract_data_star", _resolve(params.reextract_data_star)]
        if params.recenter:
            cmd.append("--recenter")
    else:
        cmd += ["--coord_suffix", params.coord_suffix]
    if params.rescale and params.rescale > 0:
        cmd += ["--scale", str(params.rescale)]
    if params.normalize:
        cmd.append("--norm")
        if params.bg_radius > 0:
            cmd += ["--bg_radius", str(params.bg_radius)]
    if params.invert_contrast:
        cmd.append("--invert_contrast")
    if params.white_dust > 0:
        cmd += ["--white_dust", str(params.white_dust)]
    if params.black_dust > 0:
        cmd += ["--black_dust", str(params.black_dust)]
    if params.float16:
        cmd.append("--float16")
    if params.use_fom_threshold:
        cmd += ["--use_fom_threshold", "--minimum_pick_fom", str(params.minimum_pick_fom)]
    if params.extra_args:
        cmd += params.extra_args
    info = _run_background(cmd, job_dir)
    return _format_launched("Particle Extraction", info)


# =====================================================================
# PIPELINE TOOL 6 — Class2D 
# =====================================================================

_CLASS2D_TUT: Dict[str, Any] = {
    "num_classes": 50, "num_iterations": 25, "tau_fudge": 2.0,
    "particle_diameter": 200.0, "do_ctf": True, "ctf_intact_first_peak": False,
    "center_classes": True, "use_vdam": False, "grad_write_iter": 100,
    "strict_highres_exp": -1.0, "pool": 30, "preread_images": False,
    "gpu": "", "threads": DEFAULT_THREADS, "mpi": DEFAULT_MPI,
}
_CLASS2D_REQ = ["input_star", "particle_diameter"]


class Class2DInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_star: str = Field(...)
    num_classes: int = Field(default=50, ge=2, le=500)
    num_iterations: int = Field(default=25, ge=1, le=200)
    particle_diameter: float = Field(..., gt=0)
    tau_fudge: float = Field(default=2.0, gt=0)
    do_ctf: bool = Field(default=True)
    ctf_intact_first_peak: bool = Field(default=False, description="Ignore CTFs until first peak")
    center_classes: bool = Field(default=True)
    use_vdam: bool = Field(default=False, description="Use VDAM gradient algorithm (--grad)")
    grad_write_iter: int = Field(default=100, description="VDAM mini-batches (--grad_write_iter)")
    strict_highres_exp: float = Field(default=-1.0, description="Limit E-step resolution Å")
    pool: int = Field(default=30, description="Pooled particles per thread (--pool)", ge=1)
    preread_images: bool = Field(default=False, description="Pre-read all into RAM (--preread_images)")
    scratch_dir: Optional[str] = Field(default=None, description="Fast scratch disk (--scratch_dir)")
    gpu: str = Field(default="", description="GPU IDs, e.g. '0:1' or '0,1' (--gpu)")
    threads: int = Field(default=DEFAULT_THREADS, ge=1, le=64)
    mpi: int = Field(default=DEFAULT_MPI, ge=1, le=128)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False)

    @field_validator("mpi")
    @classmethod
    def validate_vdam_mpi(cls, v: int, info: Any) -> int:
        # VDAM requires MPI=1, validated at runtime in tool function
        return v


@mcp.tool(name="relion_class2d",
          annotations={"readOnlyHint": False, "destructiveHint": False,
                       "idempotentHint": False, "openWorldHint": False})
async def relion_class2d(params: Class2DInput) -> str:
    """2D classification (EM or VDAM). confirm=False to preview."""
    if params.use_vdam and params.mpi > 1:
        return "❌ VDAM algorithm cannot use MPI > 1. Set mpi=1."
    if not params.confirm:
        return _preview_params(params.model_dump(), _CLASS2D_TUT, _CLASS2D_REQ, "2D Classification")
    job_dir = _next_job("Class2D")
    os.makedirs(job_dir, exist_ok=True)
    prog = _cmd("relion_refine" + ("_mpi" if params.mpi > 1 else ""))
    cmd = _mpi_prefix(prog, params.mpi) + [
        "--i", _resolve(params.input_star), "--o", os.path.join(job_dir, "run"),
        "--K", str(params.num_classes),
        "--particle_diameter", str(params.particle_diameter),
        "--tau2_fudge", str(params.tau_fudge),
        "--flatten_solvent", "--zero_mask",
        "--oversampling", "1", "--psi_step", "12",
        "--offset_range", "5", "--offset_step", "2",
        "--pool", str(params.pool),
        "--dont_combine_weights_via_disc",
        "--j", str(params.threads)]
    if params.use_vdam:
        cmd += ["--grad", "--grad_write_iter", str(params.grad_write_iter)]
    else:
        cmd += ["--iter", str(params.num_iterations)]
    if params.do_ctf:
        cmd.append("--ctf")
    if params.ctf_intact_first_peak:
        cmd.append("--ctf_intact_first_peak")
    if params.center_classes:
        cmd.append("--center_classes")
    if params.strict_highres_exp > 0:
        cmd += ["--strict_highres_exp", str(params.strict_highres_exp)]
    if params.preread_images:
        cmd.append("--preread_images")
    if params.scratch_dir:
        cmd += ["--scratch_dir", params.scratch_dir]
    if params.gpu:
        cmd += ["--gpu", params.gpu]
    if params.extra_args:
        cmd += params.extra_args
    info = _run_background(cmd, job_dir)
    return _format_launched("2D Classification", info)


# =====================================================================
# PIPELINE TOOL 7 — Initial Model 
# =====================================================================

_INIMODEL_TUT: Dict[str, Any] = {
    "num_classes": 1, "particle_diameter": 200.0, "symmetry": "C1",
    "grad_write_iter": 100, "tau_fudge": 4.0, "do_ctf": True,
    "pool": 30, "preread_images": False, "gpu": "",
    "threads": DEFAULT_THREADS, "mpi": 1,
}
_INIMODEL_REQ = ["input_star", "particle_diameter"]


class InitialModelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_star: str = Field(...)
    num_classes: int = Field(default=1, ge=1, le=10)
    particle_diameter: float = Field(..., gt=0)
    symmetry: str = Field(default="C1")
    apply_sym_later: bool = Field(default=True, description="Run in C1 then apply symmetry with relion_align_symmetry")
    grad_write_iter: int = Field(default=100, description="VDAM mini-batches (--grad_write_iter)")
    tau_fudge: float = Field(default=4.0, gt=0)
    do_ctf: bool = Field(default=True)
    flatten_solvent: bool = Field(default=True)
    pool: int = Field(default=30, ge=1)
    preread_images: bool = Field(default=False)
    scratch_dir: Optional[str] = Field(default=None)
    gpu: str = Field(default="", description="GPU IDs (--gpu)")
    threads: int = Field(default=DEFAULT_THREADS, ge=1, le=64)
    mpi: int = Field(default=1, description="Must be 1 for VDAM", ge=1, le=1)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False)


@mcp.tool(name="relion_initial_model",
          annotations={"readOnlyHint": False, "destructiveHint": False,
                       "idempotentHint": False, "openWorldHint": False})
async def relion_initial_model(params: InitialModelInput) -> str:
    """Generate ab-initio 3D model via VDAM. confirm=False to preview."""
    if not params.confirm:
        return _preview_params(params.model_dump(), _INIMODEL_TUT, _INIMODEL_REQ, "Initial Model (VDAM)")
    job_dir = _next_job("InitialModel")
    os.makedirs(job_dir, exist_ok=True)
    run_sym = "C1" if params.apply_sym_later else params.symmetry
    prog = _cmd("relion_refine")
    cmd = [prog, "--i", _resolve(params.input_star),
           "--o", os.path.join(job_dir, "run"), "--denovo_3dref",
           "--K", str(params.num_classes),
           "--grad", "--grad_write_iter", str(params.grad_write_iter),
           "--particle_diameter", str(params.particle_diameter),
           "--sym", run_sym, "--tau2_fudge", str(params.tau_fudge),
           "--zero_mask", "--oversampling", "1",
           "--pool", str(params.pool),
           "--dont_combine_weights_via_disc",
           "--j", str(params.threads)]
    if params.do_ctf:
        cmd.append("--ctf")
    if params.flatten_solvent:
        cmd.append("--flatten_solvent")
    if params.preread_images:
        cmd.append("--preread_images")
    if params.scratch_dir:
        cmd += ["--scratch_dir", params.scratch_dir]
    if params.gpu:
        cmd += ["--gpu", params.gpu]
    if params.extra_args:
        cmd += params.extra_args
    info = _run_background(cmd, job_dir)
    note = ""
    if params.apply_sym_later and params.symmetry != "C1":
        note = (f"\n\n💡 Après la fin du job, lancez `relion_align_symmetry` "
                f"avec --sym {params.symmetry} pour appliquer la symétrie.")
    return _format_launched("Initial Model (VDAM)", info) + note


# =====================================================================
# PIPELINE TOOL 8 — Class3D 
# =====================================================================

_CLASS3D_TUT: Dict[str, Any] = {
    "num_classes": 4, "num_iterations": 25, "particle_diameter": 200.0,
    "symmetry": "C1", "initial_lowpass": 50.0, "tau_fudge": 4.0,
    "do_ctf": True, "ctf_intact_first_peak": False,
    "ref_correct_greyscale": True, "healpix_order": 2,
    "skip_gridding": False, "use_blush": False, "use_fast_subsets": False,
    "strict_highres_exp": -1.0, "pool": 30, "skip_padding": False,
    "preread_images": False, "gpu": "",
    "threads": DEFAULT_THREADS, "mpi": DEFAULT_MPI,
}
_CLASS3D_REQ = ["input_star", "reference_map", "particle_diameter"]


class Class3DInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_star: str = Field(...)
    reference_map: str = Field(...)
    num_classes: int = Field(default=4, ge=2, le=50)
    num_iterations: int = Field(default=25, ge=1, le=200)
    particle_diameter: float = Field(..., gt=0)
    symmetry: str = Field(default="C1")
    initial_lowpass: float = Field(default=50.0, gt=0)
    tau_fudge: float = Field(default=4.0, gt=0)
    do_ctf: bool = Field(default=True)
    ctf_intact_first_peak: bool = Field(default=False)
    ref_correct_greyscale: bool = Field(default=True, description="Ref on absolute greyscale")
    solvent_mask: Optional[str] = Field(default=None, description="Optional reference mask (--solvent_mask)")
    healpix_order: int = Field(default=2, ge=1, le=6)
    skip_gridding: bool = Field(default=False)
    use_blush: bool = Field(default=False, description="Blush regularisation (--blush)")
    use_fast_subsets: bool = Field(default=False, description="Fast subsets for large datasets (--fast_subsets)")
    strict_highres_exp: float = Field(default=-1.0)
    pool: int = Field(default=30, ge=1)
    skip_padding: bool = Field(default=False, description="Skip padding to save RAM (--skip_padding)")
    preread_images: bool = Field(default=False)
    scratch_dir: Optional[str] = Field(default=None)
    gpu: str = Field(default="")
    threads: int = Field(default=DEFAULT_THREADS, ge=1, le=64)
    mpi: int = Field(default=DEFAULT_MPI, ge=1, le=128)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False)


@mcp.tool(name="relion_class3d",
          annotations={"readOnlyHint": False, "destructiveHint": False,
                       "idempotentHint": False, "openWorldHint": False})
async def relion_class3d(params: Class3DInput) -> str:
    """3D classification. confirm=False to preview."""
    if not params.confirm:
        return _preview_params(params.model_dump(), _CLASS3D_TUT, _CLASS3D_REQ, "3D Classification")
    job_dir = _next_job("Class3D")
    os.makedirs(job_dir, exist_ok=True)
    prog = _cmd("relion_refine" + ("_mpi" if params.mpi > 1 else ""))
    cmd = _mpi_prefix(prog, params.mpi) + [
        "--i", _resolve(params.input_star), "--ref", _resolve(params.reference_map),
        "--o", os.path.join(job_dir, "run"),
        "--K", str(params.num_classes), "--iter", str(params.num_iterations),
        "--particle_diameter", str(params.particle_diameter),
        "--sym", params.symmetry, "--ini_high", str(params.initial_lowpass),
        "--tau2_fudge", str(params.tau_fudge),
        "--flatten_solvent", "--zero_mask", "--oversampling", "1",
        "--healpix_order", str(params.healpix_order),
        "--offset_range", "5", "--offset_step", "2",
        "--pool", str(params.pool),
        "--dont_combine_weights_via_disc",
        "--j", str(params.threads)]
    if params.do_ctf:
        cmd.append("--ctf")
    if params.ctf_intact_first_peak:
        cmd.append("--ctf_intact_first_peak")
    if params.ref_correct_greyscale:
        cmd.append("--ref_correct_greyscale")
    if params.solvent_mask:
        cmd += ["--solvent_mask", _resolve(params.solvent_mask)]
    if params.skip_gridding:
        cmd.append("--skip_gridding")
    if params.use_blush:
        cmd.append("--blush")
    if params.use_fast_subsets:
        cmd.append("--fast_subsets")
    if params.strict_highres_exp > 0:
        cmd += ["--strict_highres_exp", str(params.strict_highres_exp)]
    if params.skip_padding:
        cmd.append("--skip_padding")
    if params.preread_images:
        cmd.append("--preread_images")
    if params.scratch_dir:
        cmd += ["--scratch_dir", params.scratch_dir]
    if params.gpu:
        cmd += ["--gpu", params.gpu]
    if params.extra_args:
        cmd += params.extra_args
    info = _run_background(cmd, job_dir)
    return _format_launched("3D Classification", info)


# =====================================================================
# PIPELINE TOOL 9 — Refine3D 
# =====================================================================

_REFINE3D_TUT: Dict[str, Any] = {
    "particle_diameter": 200.0, "symmetry": "D2", "initial_lowpass": 50.0,
    "ref_correct_greyscale": False, "do_ctf": True, "ctf_intact_first_peak": False,
    "use_blush": False, "strict_highres_exp": -1.0, "use_solvent_fsc": False,
    "auto_faster_sampling": True, "pool": 30, "skip_padding": False,
    "preread_images": False, "gpu": "",
    "threads": DEFAULT_THREADS, "mpi": 3,
}
_REFINE3D_REQ = ["input_star", "reference_map", "particle_diameter"]


class Refine3DInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_star: str = Field(...)
    reference_map: str = Field(...)
    particle_diameter: float = Field(..., gt=0)
    symmetry: str = Field(default="D2")
    initial_lowpass: float = Field(default=50.0, gt=0)
    solvent_mask: Optional[str] = Field(default=None)
    ref_correct_greyscale: bool = Field(default=False, description="Ref on absolute greyscale — False if rescaled")
    do_ctf: bool = Field(default=True, description="CTF correction (--ctf)")
    ctf_intact_first_peak: bool = Field(default=False)
    use_blush: bool = Field(default=False, description="Blush regularisation (--blush)")
    strict_highres_exp: float = Field(default=-1.0)
    use_solvent_fsc: bool = Field(default=False, description="Solvent-flattened FSCs (--solvent_correct_fsc)")
    auto_faster_sampling: bool = Field(default=True, description="Finer angular sampling faster (--auto_ignore_angles)")
    pool: int = Field(default=30, ge=1)
    skip_padding: bool = Field(default=False)
    preread_images: bool = Field(default=False)
    scratch_dir: Optional[str] = Field(default=None)
    gpu: str = Field(default="")
    threads: int = Field(default=DEFAULT_THREADS, ge=1, le=64)
    mpi: int = Field(default=3, ge=1, le=128)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False)

    @field_validator("mpi")
    @classmethod
    def validate_mpi_odd_ge3(cls, v: int) -> int:
        if v > 1 and (v < 3 or v % 2 == 0):
            raise ValueError("For 3D auto-refine, MPI must be odd ≥3. Try 3, 5, 7…")
        return v


@mcp.tool(name="relion_refine3d",
          annotations={"readOnlyHint": False, "destructiveHint": False,
                       "idempotentHint": False, "openWorldHint": False})
async def relion_refine3d(params: Refine3DInput) -> str:
    """Gold-standard 3D auto-refinement. MPI must be odd ≥3. confirm=False to preview."""
    if not params.confirm:
        return _preview_params(params.model_dump(), _REFINE3D_TUT, _REFINE3D_REQ, "3D Auto-Refinement")
    job_dir = _next_job("Refine3D")
    os.makedirs(job_dir, exist_ok=True)
    prog = _cmd("relion_refine" + ("_mpi" if params.mpi > 1 else ""))
    cmd = _mpi_prefix(prog, params.mpi) + [
        "--i", _resolve(params.input_star), "--ref", _resolve(params.reference_map),
        "--o", os.path.join(job_dir, "run"),
        "--auto_refine", "--split_random_halves",
        "--particle_diameter", str(params.particle_diameter),
        "--sym", params.symmetry, "--ini_high", str(params.initial_lowpass),
        "--flatten_solvent", "--zero_mask",
        "--oversampling", "1", "--pad", "2",
        "--healpix_order", "2", "--auto_local_healpix_order", "4",
        "--offset_range", "5", "--offset_step", "2",
        "--pool", str(params.pool),
        "--dont_combine_weights_via_disc",
        "--j", str(params.threads)]
    if params.do_ctf:
        cmd.append("--ctf")
    if params.ctf_intact_first_peak:
        cmd.append("--ctf_intact_first_peak")
    if params.ref_correct_greyscale:
        cmd.append("--ref_correct_greyscale")
    if params.solvent_mask:
        cmd += ["--solvent_mask", _resolve(params.solvent_mask)]
    if params.use_blush:
        cmd.append("--blush")
    if params.strict_highres_exp > 0:
        cmd += ["--strict_highres_exp", str(params.strict_highres_exp)]
    if params.use_solvent_fsc:
        cmd.append("--solvent_correct_fsc")
    if params.auto_faster_sampling:
        cmd.append("--auto_ignore_angles")
    if params.skip_padding:
        cmd.append("--skip_padding")
    if params.preread_images:
        cmd.append("--preread_images")
    if params.scratch_dir:
        cmd += ["--scratch_dir", params.scratch_dir]
    if params.gpu:
        cmd += ["--gpu", params.gpu]
    if params.extra_args:
        cmd += params.extra_args
    info = _run_background(cmd, job_dir)
    return _format_launched("3D Refinement", info)


# =====================================================================
# PIPELINE TOOL 10 — Mask Create 
# =====================================================================

_MASK_TUT: Dict[str, Any] = {
    "lowpass_filter": 15.0, "threshold": 0.01, "extend_inimask": 3,
    "width_soft_edge": 8, "threads": DEFAULT_THREADS,
}
_MASK_REQ = ["input_map"]


class MaskCreateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_map: str = Field(...)
    lowpass_filter: float = Field(default=15.0, gt=0)
    threshold: float = Field(default=0.01, description="Initial binarisation threshold", gt=0)
    extend_inimask: int = Field(default=3, description="Extend binary mask N pixels", ge=0)
    width_soft_edge: int = Field(default=8, description="Soft edge width pixels", ge=0)
    angpix: float = Field(default=-1, description="-1=from header")
    threads: int = Field(default=DEFAULT_THREADS, description="Number of threads (--j)", ge=1, le=64)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False)


@mcp.tool(name="relion_mask_create",
          annotations={"readOnlyHint": False, "destructiveHint": False,
                       "idempotentHint": False, "openWorldHint": False})
async def relion_mask_create(params: MaskCreateInput) -> str:
    """Create solvent mask. confirm=False to preview."""
    if not params.confirm:
        return _preview_params(params.model_dump(), _MASK_TUT, _MASK_REQ, "Mask Creation")
    job_dir = _next_job("MaskCreate")
    os.makedirs(job_dir, exist_ok=True)
    cmd = [_cmd("relion_mask_create"),
           "--i", _resolve(params.input_map),
           "--o", os.path.join(job_dir, "mask.mrc"),
           "--lowpass", str(params.lowpass_filter),
           "--ini_threshold", str(params.threshold),
           "--extend_inimask", str(params.extend_inimask),
           "--width_soft_edge", str(params.width_soft_edge),
           "--j", str(params.threads)]
    if params.angpix > 0:
        cmd += ["--angpix", str(params.angpix)]
    if params.extra_args:
        cmd += params.extra_args
    info = _run_background(cmd, job_dir)
    return _format_launched("Mask Creation", info)


# =====================================================================
# PIPELINE TOOL 11 — PostProcess 
# =====================================================================

_POSTPROCESS_TUT: Dict[str, Any] = {
    "auto_bfac": True, "autob_lowres": 10.0, "autob_highres": 0.0,
    "low_pass": 0, "skip_fsc_weighting": False, "mtf_angpix": -1.0,
}
_POSTPROCESS_REQ = ["half1_map", "mask"]


class PostProcessInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    half1_map: str = Field(...)
    mask: str = Field(...)
    angpix: float = Field(default=-1)
    auto_bfac: bool = Field(default=True)
    autob_lowres: float = Field(default=10.0)
    autob_highres: float = Field(default=0.0)
    adhoc_bfac: Optional[float] = Field(default=None)
    mtf: Optional[str] = Field(default=None)
    mtf_angpix: float = Field(default=-1.0)
    low_pass: float = Field(default=0)
    skip_fsc_weighting: bool = Field(default=False)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False)


@mcp.tool(name="relion_postprocess",
          annotations={"readOnlyHint": False, "destructiveHint": False,
                       "idempotentHint": False, "openWorldHint": False})
async def relion_postprocess(params: PostProcessInput) -> str:
    """Post-process: masking, B-factor sharpening, FSC. confirm=False to preview."""
    if not params.confirm:
        return _preview_params(params.model_dump(), _POSTPROCESS_TUT, _POSTPROCESS_REQ, "Post-Processing")
    job_dir = _next_job("PostProcess")
    os.makedirs(job_dir, exist_ok=True)
    cmd = [_cmd("relion_postprocess"),
           "--i", _resolve(params.half1_map), "--mask", _resolve(params.mask),
           "--o", os.path.join(job_dir, "postprocess")]
    if params.angpix > 0:
        cmd += ["--angpix", str(params.angpix)]
    if params.auto_bfac:
        cmd += ["--auto_bfac", "--autob_lowres", str(params.autob_lowres)]
        if params.autob_highres > 0:
            cmd += ["--autob_highres", str(params.autob_highres)]
    elif params.adhoc_bfac is not None:
        cmd += ["--adhoc_bfac", str(params.adhoc_bfac)]
    if params.mtf:
        cmd += ["--mtf", _resolve(params.mtf)]
    if params.mtf_angpix > 0:
        cmd += ["--mtf_angpix", str(params.mtf_angpix)]
    if params.low_pass > 0:
        cmd += ["--low_pass", str(params.low_pass)]
    if params.skip_fsc_weighting:
        cmd.append("--skip_fsc_weighting")
    if params.extra_args:
        cmd += params.extra_args
    info = _run_background(cmd, job_dir)
    return _format_launched("Post-Processing", info) + (
        f"\n\n💡 Résolution finale dans `{job_dir}/postprocess.star`.")


# =====================================================================
# PIPELINE TOOL 12 — CTF Refine 
# =====================================================================

_CTFREFINE_TUT: Dict[str, Any] = {
    "do_aniso_mag": False, "do_defocus": False, "do_astig": False,
    "do_bfactor": False, "do_beamtilt": False, "do_trefoil": False,
    "do_4thorder": False, "fit_phase": False, "minres": 30.0,
    "threads": DEFAULT_THREADS, "mpi": DEFAULT_MPI,
}
_CTFREFINE_REQ = ["input_star", "postprocess_star"]


class CtfRefineInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_star: str = Field(...)
    postprocess_star: str = Field(...)
    do_aniso_mag: bool = Field(default=False, description="Anisotropic magnification")
    do_defocus: bool = Field(default=False, description="Per-particle defocus (--fit_defocus)")
    do_astig: bool = Field(default=False, description="Per-micrograph astigmatism (--fit_astig)")
    do_bfactor: bool = Field(default=False, description="Per-particle B-factor (--fit_bfac)")
    do_beamtilt: bool = Field(default=False, description="Estimate beamtilt (--do_beamtilt)")
    do_trefoil: bool = Field(default=False, description="Trefoil aberrations (--do_trefoil)")
    do_4thorder: bool = Field(default=False, description="4th order aberrations (--do_4thorder)")
    fit_phase: bool = Field(default=False, description="Fit phase shift for phase-plate data (--fit_phase)")
    minres: float = Field(default=30.0, description="Minimum resolution for fits Å (--minres)", gt=0)
    threads: int = Field(default=DEFAULT_THREADS, ge=1, le=64)
    mpi: int = Field(default=DEFAULT_MPI, ge=1, le=128)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False)


@mcp.tool(name="relion_ctf_refine",
          annotations={"readOnlyHint": False, "destructiveHint": False,
                       "idempotentHint": False, "openWorldHint": False})
async def relion_ctf_refine(params: CtfRefineInput) -> str:
    """CTF refinement: defocus, aberrations, magnification. confirm=False to preview.
    Run multiple passes: 1) aberrations, 2) aniso_mag, 3) defocus/astig."""
    if not params.confirm:
        return _preview_params(params.model_dump(), _CTFREFINE_TUT, _CTFREFINE_REQ, "CTF Refinement")
    job_dir = _next_job("CtfRefine")
    os.makedirs(job_dir, exist_ok=True)
    prog = _cmd("relion_ctf_refine" + ("_mpi" if params.mpi > 1 else ""))
    cmd = _mpi_prefix(prog, params.mpi) + [
        "--i", _resolve(params.input_star), "--f", _resolve(params.postprocess_star),
        "--o", os.path.join(job_dir, "run"),
        "--minres", str(params.minres),
        "--j", str(params.threads)]
    if params.do_defocus:
        cmd.append("--fit_defocus")
    if params.do_astig:
        cmd.append("--fit_astig")
    if params.do_bfactor:
        cmd.append("--fit_bfac")
    if params.fit_phase:
        cmd.append("--fit_phase")
    if params.do_aniso_mag:
        cmd.append("--do_aniso_mag")
    if params.do_beamtilt:
        cmd.append("--do_beamtilt")
    if params.do_trefoil:
        cmd.append("--do_trefoil")
    if params.do_4thorder:
        cmd.append("--do_4thorder")
    if params.extra_args:
        cmd += params.extra_args
    info = _run_background(cmd, job_dir)
    return _format_launched("CTF Refinement", info)


# =====================================================================
# PIPELINE TOOL 13 — Bayesian Polishing
# =====================================================================

_POLISHING_TUT: Dict[str, Any] = {
    "first_frame": 1, "last_frame": -1, "do_train": False, "do_polish": True,
    "eval_frac": 0.5, "max_train_particles": 3000,
    "s_vel": 0.45, "s_div": 1290.0, "s_acc": 2.66,
    "minres_bfac": 20.0, "maxres_bfac": -1.0,
    "float16": True, "extract_size": -1, "rescale_size": -1,
    "threads": DEFAULT_THREADS, "mpi": DEFAULT_MPI,
}
_POLISHING_REQ = ["input_star", "postprocess_star"]


class BayesianPolishingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_star: str = Field(..., description="Particles STAR from refinement")
    postprocess_star: str = Field(..., description="PostProcess STAR for FSC")
    first_frame: int = Field(default=1, ge=1)
    last_frame: int = Field(default=-1)
    # Mode selection
    do_train: bool = Field(default=False, description="Train optimal sigma parameters")
    do_polish: bool = Field(default=True, description="Perform particle polishing")
    # Training params
    eval_frac: float = Field(default=0.5, description="Fraction Fourier pixels for testing", gt=0, le=1)
    max_train_particles: int = Field(default=3000, description="Particles for training", ge=100)
    # Polish params
    opt_params_file: Optional[str] = Field(default=None, description="Optimised params file from training")
    use_own_params: bool = Field(default=False, description="Use manual sigma values instead of opt file")
    s_vel: float = Field(default=0.45, description="Sigma for velocity Å/dose")
    s_div: float = Field(default=1290.0, description="Sigma for divergence Å")
    s_acc: float = Field(default=2.66, description="Sigma for acceleration Å/dose")
    minres_bfac: float = Field(default=20.0, description="Min resolution B-factor fit Å")
    maxres_bfac: float = Field(default=-1.0, description="Max resolution B-factor fit Å, -1=auto")
    # Output options
    float16: bool = Field(default=True)
    extract_size: int = Field(default=-1, description="Extraction size unbinned, -1=from input")
    rescale_size: int = Field(default=-1, description="Re-scaled size, -1=from input")
    threads: int = Field(default=DEFAULT_THREADS, ge=1, le=64)
    mpi: int = Field(default=DEFAULT_MPI, ge=1, le=128)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False)


@mcp.tool(name="relion_bayesian_polishing",
          annotations={"readOnlyHint": False, "destructiveHint": False,
                       "idempotentHint": False, "openWorldHint": False})
async def relion_bayesian_polishing(params: BayesianPolishingInput) -> str:
    """Bayesian polishing: train sigma params or polish particles. confirm=False to preview.
    Training requires MPI=1."""
    if params.do_train and params.mpi > 1:
        return "❌ Training mode requires MPI=1. Set mpi=1."
    if not params.confirm:
        return _preview_params(params.model_dump(), _POLISHING_TUT, _POLISHING_REQ, "Bayesian Polishing")
    job_dir = _next_job("Polish")
    os.makedirs(job_dir, exist_ok=True)
    prog = _cmd("relion_motion_refine" + ("_mpi" if params.mpi > 1 and not params.do_train else ""))
    cmd_parts = _mpi_prefix(prog, params.mpi) if not params.do_train else [prog]
    cmd = cmd_parts + [
        "--i", _resolve(params.input_star), "--f", _resolve(params.postprocess_star),
        "--o", os.path.join(job_dir, "run"),
        "--first_frame", str(params.first_frame), "--last_frame", str(params.last_frame),
        "--j", str(params.threads)]
    if params.do_train:
        cmd += ["--train", "--eval_frac", str(params.eval_frac),
                "--max_particles", str(params.max_train_particles)]
    if params.do_polish:
        if params.opt_params_file:
            cmd += ["--opt_params", _resolve(params.opt_params_file)]
        elif params.use_own_params:
            cmd += ["--s_vel", str(params.s_vel), "--s_div", str(params.s_div),
                    "--s_acc", str(params.s_acc)]
        cmd += ["--minres_bfac", str(params.minres_bfac)]
        if params.maxres_bfac > 0:
            cmd += ["--maxres_bfac", str(params.maxres_bfac)]
    if params.float16:
        cmd.append("--float16")
    if params.extract_size > 0:
        cmd += ["--window", str(params.extract_size)]
    if params.rescale_size > 0:
        cmd += ["--scale", str(params.rescale_size)]
    if params.extra_args:
        cmd += params.extra_args
    info = _run_background(cmd, job_dir)
    return _format_launched("Bayesian Polishing", info)


# =====================================================================
# PIPELINE TOOL 14 — Blush AI Denoising
# =====================================================================

class BlushInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    half1_map: str = Field(...)
    half2_map: str = Field(...)
    mask: Optional[str] = Field(default=None)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False)


@mcp.tool(name="relion_blush",
          annotations={"readOnlyHint": False, "destructiveHint": False,
                       "idempotentHint": False, "openWorldHint": False})
async def relion_blush(params: BlushInput) -> str:
    """AI map denoising with Blush. confirm=False to preview."""
    if not params.confirm:
        return _preview_params(params.model_dump(), {}, ["half1_map", "half2_map"], "Blush (AI Denoising)")
    job_dir = _next_job("Blush")
    os.makedirs(job_dir, exist_ok=True)
    cmd = [_cmd("relion_python_blush"),
           "--i1", _resolve(params.half1_map), "--i2", _resolve(params.half2_map),
           "--o", os.path.join(job_dir, "blush")]
    if params.mask:
        cmd += ["--mask", _resolve(params.mask)]
    if params.extra_args:
        cmd += params.extra_args
    info = _run_background(cmd, job_dir)
    return _format_launched("Blush (AI Denoising)", info)


# =====================================================================
# PIPELINE TOOL 15 — Local Resolution (NEW)
# =====================================================================

_LOCALRES_TUT: Dict[str, Any] = {
    "adhoc_bfac": -30.0, "mpi": DEFAULT_MPI,
}
_LOCALRES_REQ = ["half1_map", "mask"]


class LocalResInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    half1_map: str = Field(..., description="One of the two unfiltered half-maps")
    mask: str = Field(..., description="Solvent mask")
    angpix: float = Field(default=-1, description="Calibrated pixel size, -1=header")
    adhoc_bfac: float = Field(default=-30.0, description="B-factor for locally-filtered map")
    mtf: Optional[str] = Field(default=None, description="MTF STAR file")
    mpi: int = Field(default=DEFAULT_MPI, ge=1, le=128)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False)


@mcp.tool(name="relion_local_resolution",
          annotations={"readOnlyHint": False, "destructiveHint": False,
                       "idempotentHint": False, "openWorldHint": False})
async def relion_local_resolution(params: LocalResInput) -> str:
    """Estimate local resolution using RELION's own method. confirm=False to preview."""
    if not params.confirm:
        return _preview_params(params.model_dump(), _LOCALRES_TUT, _LOCALRES_REQ, "Local Resolution")
    job_dir = _next_job("LocalRes")
    os.makedirs(job_dir, exist_ok=True)
    prog = _cmd("relion_postprocess" + ("_mpi" if params.mpi > 1 else ""))
    cmd = _mpi_prefix(prog, params.mpi) + [
        "--i", _resolve(params.half1_map), "--mask", _resolve(params.mask),
        "--o", os.path.join(job_dir, "relion"),
        "--locres", "--adhoc_bfac", str(params.adhoc_bfac)]
    if params.angpix > 0:
        cmd += ["--angpix", str(params.angpix)]
    if params.mtf:
        cmd += ["--mtf", _resolve(params.mtf)]
    if params.extra_args:
        cmd += params.extra_args
    info = _run_background(cmd, job_dir)
    return _format_launched("Local Resolution", info)


# =====================================================================
# PIPELINE TOOL 16 — ModelAngelo (NEW)
# =====================================================================

_MODELANGELO_TUT: Dict[str, Any] = {"gpu": "0"}
_MODELANGELO_REQ = ["input_map"]


class ModelAngeloInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_map: str = Field(..., description="B-factor sharpened map (postprocess_masked.mrc)")
    fasta_protein: Optional[str] = Field(default=None, description="FASTA for proteins")
    fasta_dna: Optional[str] = Field(default=None, description="FASTA for DNA")
    fasta_rna: Optional[str] = Field(default=None, description="FASTA for RNA")
    do_hmmer: bool = Field(default=False, description="Perform HMMer search for unknown proteins")
    hmmer_fasta: Optional[str] = Field(default=None, description="Genome FASTA for HMMer")
    hmmer_alphabet: str = Field(default="amino", description="HMMer alphabet: amino, dna, rna")
    gpu: str = Field(default="0", description="GPU IDs, e.g. '0,1,2,3'")
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False)


@mcp.tool(name="relion_modelangelo",
          annotations={"readOnlyHint": False, "destructiveHint": False,
                       "idempotentHint": False, "openWorldHint": False})
async def relion_modelangelo(params: ModelAngeloInput) -> str:
    """Automated atomic model building with ModelAngelo. confirm=False to preview."""
    if not params.confirm:
        return _preview_params(params.model_dump(), _MODELANGELO_TUT, _MODELANGELO_REQ, "ModelAngelo")
    job_dir = _next_job("ModelAngelo")
    os.makedirs(job_dir, exist_ok=True)
    cmd = [_cmd("relion_python_modelangelo"),
           "--i", _resolve(params.input_map),
           "--o", job_dir,
           "--gpu", params.gpu]
    if params.fasta_protein:
        cmd += ["--fasta", _resolve(params.fasta_protein)]
    if params.fasta_dna:
        cmd += ["--fasta_dna", _resolve(params.fasta_dna)]
    if params.fasta_rna:
        cmd += ["--fasta_rna", _resolve(params.fasta_rna)]
    if params.do_hmmer:
        cmd.append("--do_hmmer")
        if params.hmmer_fasta:
            cmd += ["--hmmer_fasta", _resolve(params.hmmer_fasta)]
        cmd += ["--hmmer_alphabet", params.hmmer_alphabet]
    if params.extra_args:
        cmd += params.extra_args
    info = _run_background(cmd, job_dir)
    return _format_launched("ModelAngelo", info)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="RELION MCP Server v3.0")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=8422)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--project-dir", default=None)
    args = parser.parse_args()
    if args.project_dir:
        global PROJECT_DIR
        PROJECT_DIR = args.project_dir
    if args.transport == "http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()

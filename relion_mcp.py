#!/usr/bin/env python3
"""
RELION MCP Server v2.0 — Complete rewrite for RELION 5.0.1

Features:
  - Preview/Confirm mode: every pipeline tool shows all params before launch
  - Full parameter coverage per RELION 5 tutorial (EMPIAR-10204)
  - 4 new tools: InitialModel, MaskCreate, CTFRefine, BayesianPolishing
  - 19 tools total (5 read-only + 14 pipeline)

Compatible with Claude Code (stdio), Claude Cowork, OpenClaw/NemoClaw (HTTP).

Usage:
  python relion_mcp.py                                  # stdio (Claude Code)
  python relion_mcp.py --transport http --port 8422     # HTTP  (OpenClaw)

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
    """Full path to a RELION binary, or just the name if RELION_BIN unset."""
    return str(Path(RELION_BIN) / program) if RELION_BIN else program


def _resolve(path: str) -> str:
    """Resolve a path relative to the project directory."""
    return path if os.path.isabs(path) else str(Path(PROJECT_DIR) / path)


def _mpi_prefix(program: str, mpi: int) -> List[str]:
    """Build MPI launch prefix if mpi > 1."""
    return ["mpirun", "-np", str(mpi), program] if mpi > 1 else [program]


def _next_job(job_type: str) -> str:
    """Compute next sequential job directory, e.g. Class2D/job003."""
    d = Path(PROJECT_DIR) / job_type
    d.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        [x for x in d.iterdir() if x.is_dir() and x.name.startswith("job")]
    )
    n = int(existing[-1].name.replace("job", "")) + 1 if existing else 1
    return str(d / f"job{n:03d}")


def _mark_success(job_dir: str) -> None:
    """Create RELION_JOB_EXIT_SUCCESS marker file."""
    Path(job_dir, "RELION_JOB_EXIT_SUCCESS").touch()


def _job_status(job_dir: str) -> str:
    """Check status of a RELION job directory."""
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


def _run_sync(
    cmd: List[str], cwd: Optional[str] = None, timeout: int = 600
) -> dict:
    """Run a subprocess synchronously, return structured result."""
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd or PROJECT_DIR,
            timeout=timeout,
        )
        return {
            "returncode": r.returncode,
            "stdout": r.stdout[-4000:] if len(r.stdout) > 4000 else r.stdout,
            "stderr": r.stderr[-2000:] if len(r.stderr) > 2000 else r.stderr,
            "command": " ".join(cmd),
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Timeout after {timeout}s",
            "command": " ".join(cmd),
        }
    except FileNotFoundError:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Binary not found: {cmd[0]}",
            "command": " ".join(cmd),
        }


async def _run(
    cmd: List[str], cwd: Optional[str] = None, timeout: int = 600
) -> dict:
    """Run subprocess in thread pool (non-blocking)."""
    return await asyncio.get_running_loop().run_in_executor(
        None, _run_sync, cmd, cwd, timeout
    )


def _format_result(name: str, job_dir: str, result: dict) -> str:
    """Format job result as markdown."""
    ok = result["returncode"] == 0
    status = "✅ Success" if ok else f"❌ Failed ({result['returncode']})"
    lines = [
        f"## {name}",
        f"**Status:** {status}",
        f"**Job:** `{job_dir}`",
        f"**Command:** `{result['command']}`",
    ]
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
    """Parse a RELION STAR file into columns + rows."""
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
                    rows.append(
                        {n: parts[i] if i < len(parts) else "" for n, i in headers.items()}
                    )
    return {"columns": list(headers.keys()), "num_rows": len(rows), "rows": rows[:200]}


# ---------------------------------------------------------------------------
# Preview / Confirm system
# ---------------------------------------------------------------------------


def _preview_params(
    params_dict: Dict[str, Any],
    tutorial_defaults: Dict[str, Any],
    required_fields: List[str],
    label: str,
) -> str:
    """Build a formatted preview of all parameters for a pipeline tool.

    For each parameter:
      - ✏️ défini      : value was explicitly supplied by the user
      - 📋 tutorial    : using the RELION 5 tutorial default
      - ❌ requis      : required field not yet provided
    """
    lines = [f"## 🔍 Preview — {label}", ""]
    missing_required: List[str] = []
    for key in sorted(set(list(params_dict.keys()) + list(tutorial_defaults.keys()))):
        if key == "confirm":
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
            missing_required.append(key)
        else:
            lines.append(f"  ⬜  **{key}** = _(non défini, optionnel)_")
    lines.append("")
    if missing_required:
        lines.append(
            f"⚠️ **Paramètres requis manquants :** {', '.join(missing_required)}"
        )
        lines.append("Fournissez-les puis relancez avec `confirm=True`.")
    else:
        lines.append(
            "✅ Tous les paramètres requis sont présents.\n"
            "→ Relancez avec **`confirm=True`** pour exécuter le job."
        )
    return "\n".join(lines)


def _merge_defaults(
    params: BaseModel, tutorial_defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """Return a dict with user values overlaid on tutorial defaults."""
    d = params.model_dump()
    for k, v in tutorial_defaults.items():
        if k not in d or d[k] is None:
            d[k] = v
    return d


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("relion_mcp")


class Fmt(str, Enum):
    """Response format option."""
    MARKDOWN = "markdown"
    JSON = "json"


# =====================================================================
# READ-ONLY TOOL 1 — Project Info
# =====================================================================


class ProjectInfoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    response_format: Fmt = Field(default=Fmt.MARKDOWN)


@mcp.tool(
    name="relion_project_info",
    annotations={
        "readOnlyHint": True, "destructiveHint": False,
        "idempotentHint": True, "openWorldHint": False,
    },
)
async def relion_project_info(params: ProjectInfoInput) -> str:
    """Get overview of the RELION project: job types, counts, latest status."""
    pdir = Path(PROJECT_DIR)
    types = [
        "Import", "MotionCorr", "CtfFind", "ManualPick", "AutoPick",
        "Extract", "Select", "Class2D", "Class3D", "InitialModel",
        "Refine3D", "CtfRefine", "Polish", "MaskCreate", "PostProcess",
        "LocalRes", "External", "ModelAngelo", "DynaMight", "Blush",
    ]
    info: Dict[str, Any] = {"project_dir": str(pdir), "jobs": {}}
    for t in types:
        td = pdir / t
        if td.is_dir():
            jobs = sorted([d.name for d in td.iterdir() if d.is_dir()])
            info["jobs"][t] = {
                "count": len(jobs),
                "latest": jobs[-1] if jobs else None,
                "status": _job_status(f"{t}/{jobs[-1]}") if jobs else "EMPTY",
            }
    if params.response_format == Fmt.JSON:
        return json.dumps(info, indent=2)
    lines = [f"# RELION Project: `{pdir.name}`", f"**Path:** `{pdir}`", ""]
    if not info["jobs"]:
        lines.append("_No jobs — new project._")
    for t, d in info["jobs"].items():
        lines.append(
            f"- **{t}** — {d['count']} job(s), latest: `{d['latest']}` ({d['status']})"
        )
    return "\n".join(lines)


# =====================================================================
# READ-ONLY TOOL 2 — Read STAR file
# =====================================================================


class ReadStarInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    star_file: str = Field(..., description="Path to STAR file")
    table_name: str = Field(default="data_", description="STAR table to parse")
    max_rows: int = Field(default=20, ge=1, le=200)
    response_format: Fmt = Field(default=Fmt.MARKDOWN)


@mcp.tool(
    name="relion_read_star",
    annotations={
        "readOnlyHint": True, "destructiveHint": False,
        "idempotentHint": True, "openWorldHint": False,
    },
)
async def relion_read_star(params: ReadStarInput) -> str:
    """Parse a RELION STAR file and return columns + data rows."""
    p = _parse_star(params.star_file, params.table_name)
    if "error" in p:
        return f"Error: {p['error']}"
    p["rows"] = p["rows"][: params.max_rows]
    if params.response_format == Fmt.JSON:
        return json.dumps(p, indent=2)
    lines = [
        f"## `{params.star_file}`",
        f"**Columns:** {len(p['columns'])} | **Rows:** {p['num_rows']}",
        "**Fields:** " + ", ".join(f"`{c}`" for c in p["columns"]),
        "",
    ]
    for i, row in enumerate(p["rows"]):
        lines.append(
            f"Row {i + 1}: "
            + " | ".join(f"{k}={v}" for k, v in list(row.items())[:8])
        )
    if p["num_rows"] > params.max_rows:
        lines.append(f"\n_{p['num_rows'] - params.max_rows} more rows…_")
    return "\n".join(lines)


# =====================================================================
# READ-ONLY TOOL 3 — Job Status
# =====================================================================


class JobStatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_dir: str = Field(..., description="Job directory, e.g. 'Class2D/job015'")


@mcp.tool(
    name="relion_job_status",
    annotations={
        "readOnlyHint": True, "destructiveHint": False,
        "idempotentHint": True, "openWorldHint": False,
    },
)
async def relion_job_status(params: JobStatusInput) -> str:
    """Check execution status of a RELION job."""
    jd = Path(_resolve(params.job_dir))
    status = _job_status(params.job_dir)
    files: List[str] = []
    if jd.exists():
        for f in sorted(jd.iterdir()):
            if f.suffix in (".star", ".mrc", ".mrcs", ".log"):
                files.append(f"  - `{f.name}` ({f.stat().st_size / 1e6:.1f} MB)")
    return (
        f"## `{params.job_dir}`\n**Status:** {status}\n"
        + ("\n".join(files) if files else "_No output files._")
    )


# =====================================================================
# READ-ONLY TOOL 4 — Suggest Next Step
# =====================================================================


class SuggestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


@mcp.tool(
    name="relion_suggest_next_step",
    annotations={
        "readOnlyHint": True, "destructiveHint": False,
        "idempotentHint": True, "openWorldHint": False,
    },
)
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
    ]
    completed = set()
    for step, _, _ in pipeline:
        step_dir = pdir / step
        if step_dir.is_dir():
            for job in sorted(step_dir.iterdir()):
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
        lines.append("All main steps done! Consider Blush (`relion_blush`) for AI denoising.")
    return "\n".join(lines)


# =====================================================================
# READ-ONLY TOOL 5 — Run Any RELION Command (escape hatch)
# =====================================================================


class RunCommandInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    program: str = Field(..., description="RELION program, e.g. 'relion_mask_create'")
    arguments: List[str] = Field(default_factory=list, description="CLI arguments")
    job_type: str = Field(default="External", description="Job type directory")
    timeout: int = Field(default=3600, ge=10, le=604800)

    @field_validator("program")
    @classmethod
    def validate_program(cls, v: str) -> str:
        if not v.startswith("relion"):
            raise ValueError("Program must start with 'relion'")
        if any(c in v for c in ";|&`$()"):
            raise ValueError("Invalid characters in program name")
        return v


@mcp.tool(
    name="relion_run_command",
    annotations={
        "readOnlyHint": False, "destructiveHint": True,
        "idempotentHint": False, "openWorldHint": False,
    },
)
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
# PIPELINE TOOL 1 — Import
# =====================================================================

# Tutorial defaults for Import (EMPIAR-10204 / beta-galactosidase)
_IMPORT_TUTORIAL: Dict[str, Any] = {
    "voltage": 200.0,
    "pixel_size": 0.885,
    "cs": 1.4,
    "q0": 0.1,
    "optics_group_name": "opticsGroup1",
    "is_movie": True,
}
_IMPORT_REQUIRED = ["input_files", "pixel_size"]


class ImportInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_files: str = Field(..., description="Wildcard for input, e.g. 'Movies/*.tiff'")
    pixel_size: float = Field(..., description="Pixel size Å (--angpix)", gt=0, le=20)
    voltage: float = Field(default=200.0, description="kV (--kV)", ge=60, le=400)
    cs: float = Field(default=1.4, description="Spherical aberration mm (--Cs)")
    q0: float = Field(default=0.1, description="Amplitude contrast (--Q0)", ge=0, le=1)
    optics_group_name: str = Field(default="opticsGroup1")
    is_movie: bool = Field(default=True, description="True=movies, False=micrographs")
    extra_args: Optional[List[str]] = Field(default=None, description="Additional CLI args")
    confirm: bool = Field(default=False, description="False=preview, True=run")


@mcp.tool(
    name="relion_import",
    annotations={
        "readOnlyHint": False, "destructiveHint": False,
        "idempotentHint": False, "openWorldHint": False,
    },
)
async def relion_import(params: ImportInput) -> str:
    """Import raw movies or micrographs into the RELION project. First pipeline step.
    Call with confirm=False (default) to preview parameters, confirm=True to execute."""
    if not params.confirm:
        return _preview_params(
            params.model_dump(), _IMPORT_TUTORIAL, _IMPORT_REQUIRED, "Import"
        )
    job_dir = _next_job("Import")
    os.makedirs(job_dir, exist_ok=True)
    node_type = "movies.star" if params.is_movie else "micrographs.star"
    cmd = [
        _cmd("relion_import"),
        "--i", params.input_files,
        "--odir", job_dir + "/",
        "--ofile", node_type,
        "--do_movies" if params.is_movie else "--do_micrographs",
        "--optics_group_name", params.optics_group_name,
        "--angpix", str(params.pixel_size),
        "--kV", str(params.voltage),
        "--Cs", str(params.cs),
        "--Q0", str(params.q0),
    ]
    if params.extra_args:
        cmd += params.extra_args
    result = await _run(cmd)
    if result["returncode"] == 0:
        _mark_success(job_dir)
    return _format_result("Import", job_dir, result)


# =====================================================================
# PIPELINE TOOL 2 — Motion Correction
# =====================================================================

_MOTIONCORR_TUTORIAL: Dict[str, Any] = {
    "dose_per_frame": 1.277,
    "voltage": 200.0,
    "patch_x": 5,
    "patch_y": 5,
    "bfactor": 150,
    "bin_factor": 1,
    "first_frame": 1,
    "last_frame": -1,
    "gain_rot": 0,
    "gain_flip": 0,
    "float16": True,
    "save_noDW": False,
    "save_ps": True,
    "group_frames": 3,
    "threads": DEFAULT_THREADS,
    "mpi": DEFAULT_MPI,
}
_MOTIONCORR_REQUIRED = ["input_star", "dose_per_frame"]


class MotionCorrInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_star: str = Field(..., description="STAR file with movies")
    dose_per_frame: float = Field(default=1.277, description="e-/Å²/frame (--dose_per_frame)", gt=0)
    voltage: float = Field(default=200.0, description="kV (--voltage)", ge=60, le=400)
    pixel_size: float = Field(default=-1, description="Å (--angpix), -1=from STAR")
    patch_x: int = Field(default=5, description="Patches X (--patch_x)", ge=1, le=20)
    patch_y: int = Field(default=5, description="Patches Y (--patch_y)", ge=1, le=20)
    bfactor: int = Field(default=150, description="B-factor for motion (--bfactor)", ge=0, le=1500)
    bin_factor: int = Field(default=1, description="Binning (--bin_factor)", ge=1, le=8)
    gainref: Optional[str] = Field(default=None, description="Gain reference MRC (--gainref)")
    gain_rot: int = Field(default=0, description="Gain rotation 0/1/2/3 (--gain_rot)", ge=0, le=3)
    gain_flip: int = Field(default=0, description="Gain flip 0/1/2 (--gain_flip)", ge=0, le=2)
    first_frame: int = Field(default=1, description="First frame (--first_frame_sum)", ge=1)
    last_frame: int = Field(default=-1, description="Last frame, -1=all (--last_frame_sum)")
    float16: bool = Field(default=True, description="Write float16 MRCs (--float16)")
    save_noDW: bool = Field(default=False, description="Save non-dose-weighted (--save_noDW)")
    save_ps: bool = Field(default=True, description="Save power spectra (--save_ps)")
    group_frames: int = Field(default=3, description="Group N frames for power spectra (--groupNframes)", ge=1)
    threads: int = Field(default=DEFAULT_THREADS, ge=1, le=64)
    mpi: int = Field(default=DEFAULT_MPI, ge=1, le=128)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False, description="False=preview, True=run")


@mcp.tool(
    name="relion_motioncorr",
    annotations={
        "readOnlyHint": False, "destructiveHint": False,
        "idempotentHint": False, "openWorldHint": False,
    },
)
async def relion_motioncorr(params: MotionCorrInput) -> str:
    """Beam-induced motion correction using RELION's own implementation.
    Call with confirm=False to preview, confirm=True to execute."""
    if not params.confirm:
        return _preview_params(
            params.model_dump(), _MOTIONCORR_TUTORIAL,
            _MOTIONCORR_REQUIRED, "Motion Correction",
        )
    job_dir = _next_job("MotionCorr")
    os.makedirs(job_dir, exist_ok=True)
    prog = _cmd("relion_run_motioncorr" + ("_mpi" if params.mpi > 1 else ""))
    cmd = _mpi_prefix(prog, params.mpi) + [
        "--i", _resolve(params.input_star),
        "--o", job_dir + "/",
        "--use_own",
        "--dose_weighting",
        "--dose_per_frame", str(params.dose_per_frame),
        "--voltage", str(params.voltage),
        "--patch_x", str(params.patch_x),
        "--patch_y", str(params.patch_y),
        "--bfactor", str(params.bfactor),
        "--bin_factor", str(params.bin_factor),
        "--first_frame_sum", str(params.first_frame),
        "--last_frame_sum", str(params.last_frame),
        "--groupNframes", str(params.group_frames),
        "--j", str(params.threads),
    ]
    if params.pixel_size > 0:
        cmd += ["--angpix", str(params.pixel_size)]
    if params.gainref:
        cmd += ["--gainref", _resolve(params.gainref)]
    if params.gain_rot != 0:
        cmd += ["--gain_rot", str(params.gain_rot)]
    if params.gain_flip != 0:
        cmd += ["--gain_flip", str(params.gain_flip)]
    if params.float16:
        cmd.append("--float16")
    if params.save_noDW:
        cmd.append("--save_noDW")
    if params.save_ps:
        cmd.append("--save_ps")
    if params.extra_args:
        cmd += params.extra_args
    result = await _run(cmd, timeout=86400)
    if result["returncode"] == 0:
        _mark_success(job_dir)
    return _format_result("Motion Correction", job_dir, result)


# =====================================================================
# PIPELINE TOOL 3 — CTF Estimation
# =====================================================================

_CTF_TUTORIAL: Dict[str, Any] = {
    "box": 512,
    "res_min": 30.0,
    "res_max": 5.0,
    "df_min": 5000.0,
    "df_max": 50000.0,
    "f_step": 500.0,
    "d_ast": 100.0,
    "ctffind_exe": "ctffind",
    "is_ctffind4": True,
    "use_given_ps": False,
    "do_phaseshift": False,
    "fast_search": True,
    "threads": DEFAULT_THREADS,
    "mpi": DEFAULT_MPI,
}
_CTF_REQUIRED = ["input_star"]


class CtfFindInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_star: str = Field(..., description="STAR file with corrected micrographs")
    box: int = Field(default=512, description="FFT box size pixels (--Box)", ge=64, le=2048)
    res_min: float = Field(default=30.0, description="Min resolution Å (--ResMin)", gt=0)
    res_max: float = Field(default=5.0, description="Max resolution Å (--ResMax)", gt=0)
    df_min: float = Field(default=5000.0, description="Min defocus Å (--dFMin)", ge=0)
    df_max: float = Field(default=50000.0, description="Max defocus Å (--dFMax)", ge=0)
    f_step: float = Field(default=500.0, description="Defocus step Å (--FStep)", gt=0)
    d_ast: float = Field(default=100.0, description="Expected astigmatism Å (--dAst)", ge=0)
    ctffind_exe: str = Field(default="ctffind", description="ctffind executable path (--ctffind_exe)")
    is_ctffind4: bool = Field(default=True, description="Using CTFFIND4 (--is_ctffind4)")
    use_given_ps: bool = Field(default=False, description="Use pre-calculated power spectra (--use_given_ps)")
    do_phaseshift: bool = Field(default=False, description="Estimate phase shift / Volta (--do_phaseshift)")
    fast_search: bool = Field(default=True, description="Use fast search (--fast_search)")
    threads: int = Field(default=DEFAULT_THREADS, ge=1, le=64)
    mpi: int = Field(default=DEFAULT_MPI, ge=1, le=128)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False, description="False=preview, True=run")


@mcp.tool(
    name="relion_ctffind",
    annotations={
        "readOnlyHint": False, "destructiveHint": False,
        "idempotentHint": False, "openWorldHint": False,
    },
)
async def relion_ctffind(params: CtfFindInput) -> str:
    """Estimate CTF parameters for each micrograph using CTFFIND4.
    Call with confirm=False to preview, confirm=True to execute."""
    if not params.confirm:
        return _preview_params(
            params.model_dump(), _CTF_TUTORIAL, _CTF_REQUIRED, "CTF Estimation"
        )
    job_dir = _next_job("CtfFind")
    os.makedirs(job_dir, exist_ok=True)
    prog = _cmd("relion_run_ctffind" + ("_mpi" if params.mpi > 1 else ""))
    cmd = _mpi_prefix(prog, params.mpi) + [
        "--i", _resolve(params.input_star),
        "--o", job_dir + "/",
        "--ctffind_exe", params.ctffind_exe,
        "--Box", str(params.box),
        "--ResMin", str(params.res_min),
        "--ResMax", str(params.res_max),
        "--dFMin", str(params.df_min),
        "--dFMax", str(params.df_max),
        "--FStep", str(params.f_step),
        "--dAst", str(params.d_ast),
        "--j", str(params.threads),
    ]
    if params.is_ctffind4:
        cmd.append("--is_ctffind4")
    if params.use_given_ps:
        cmd.append("--use_given_ps")
    if params.do_phaseshift:
        cmd.append("--do_phaseshift")
    if params.fast_search:
        cmd.append("--fast_search")
    if params.extra_args:
        cmd += params.extra_args
    result = await _run(cmd, timeout=86400)
    if result["returncode"] == 0:
        _mark_success(job_dir)
    return _format_result("CTF Estimation", job_dir, result)


# =====================================================================
# PIPELINE TOOL 4 — AutoPick (LoG / template-based)
# =====================================================================

_AUTOPICK_TUTORIAL: Dict[str, Any] = {
    "particle_diameter": 180.0,
    "angpix": 0.885,
    "use_log": True,
    "log_diam_min": 150.0,
    "log_diam_max": 180.0,
    "log_adjust_threshold": 0.0,
    "log_upper_threshold": 999.0,
    "log_use_ctf": True,
    "threads": DEFAULT_THREADS,
    "mpi": DEFAULT_MPI,
}
_AUTOPICK_REQUIRED = ["input_star", "particle_diameter"]


class AutoPickInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_star: str = Field(..., description="Micrograph STAR with CTF info")
    particle_diameter: float = Field(..., description="Particle diameter Å (--particle_diameter)", gt=0, le=2000)
    angpix: float = Field(default=0.885, description="Pixel size Å (--angpix)", gt=0)
    use_log: bool = Field(default=True, description="LoG picker (--LoG)")
    log_diam_min: Optional[float] = Field(default=150.0, description="Min LoG diameter Å (--LoG_diam_min)")
    log_diam_max: Optional[float] = Field(default=180.0, description="Max LoG diameter Å (--LoG_diam_max)")
    log_adjust_threshold: float = Field(default=0.0, description="+less -more (--LoG_adjust_threshold)")
    log_upper_threshold: float = Field(default=999.0, description="Upper threshold LoG (--LoG_upper_threshold)")
    log_use_ctf: bool = Field(default=True, description="Use CTF for LoG (--LoG_use_ctf)")
    ref: Optional[str] = Field(default=None, description="Template ref STAR/MRC (--ref)")
    lowpass: float = Field(default=-1, description="Lowpass for refs Å (--lowpass)")
    threads: int = Field(default=DEFAULT_THREADS, ge=1, le=64)
    mpi: int = Field(default=DEFAULT_MPI, ge=1, le=128)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False, description="False=preview, True=run")


@mcp.tool(
    name="relion_autopick",
    annotations={
        "readOnlyHint": False, "destructiveHint": False,
        "idempotentHint": False, "openWorldHint": False,
    },
)
async def relion_autopick(params: AutoPickInput) -> str:
    """Automatically pick particles: LoG (template-free) or reference-based.
    Call with confirm=False to preview, confirm=True to execute."""
    if not params.confirm:
        return _preview_params(
            params.model_dump(), _AUTOPICK_TUTORIAL,
            _AUTOPICK_REQUIRED, "AutoPick",
        )
    job_dir = _next_job("AutoPick")
    os.makedirs(job_dir, exist_ok=True)
    prog = _cmd("relion_autopick" + ("_mpi" if params.mpi > 1 else ""))
    cmd = _mpi_prefix(prog, params.mpi) + [
        "--i", _resolve(params.input_star),
        "--odir", job_dir + "/",
        "--angpix", str(params.angpix),
        "--particle_diameter", str(params.particle_diameter),
    ]
    if params.use_log:
        cmd += [
            "--LoG",
            "--LoG_adjust_threshold", str(params.log_adjust_threshold),
            "--LoG_upper_threshold", str(params.log_upper_threshold),
        ]
        if params.log_diam_min is not None:
            cmd += ["--LoG_diam_min", str(params.log_diam_min)]
        if params.log_diam_max is not None:
            cmd += ["--LoG_diam_max", str(params.log_diam_max)]
        if params.log_use_ctf:
            cmd.append("--LoG_use_ctf")
    else:
        if params.ref:
            cmd += ["--ref", _resolve(params.ref)]
        if params.lowpass > 0:
            cmd += ["--lowpass", str(params.lowpass)]
    if params.extra_args:
        cmd += params.extra_args
    result = await _run(cmd, timeout=86400)
    if result["returncode"] == 0:
        _mark_success(job_dir)
    return _format_result("Auto-Picking", job_dir, result)


# =====================================================================
# PIPELINE TOOL 5 — Extract
# =====================================================================

_EXTRACT_TUTORIAL: Dict[str, Any] = {
    "extract_size": 256,
    "rescale": 64,
    "coord_suffix": "_autopick.star",
    "normalize": True,
    "bg_radius": 200,
    "invert_contrast": True,
    "white_dust": -1.0,
    "black_dust": -1.0,
    "float16": True,
    "mpi": DEFAULT_MPI,
}
_EXTRACT_REQUIRED = ["input_star", "extract_size"]


class ExtractInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_star: str = Field(..., description="Micrograph STAR file (--i)")
    coord_suffix: str = Field(default="_autopick.star", description="Coordinate suffix (--coord_suffix)")
    extract_size: int = Field(..., description="Box size pixels (--extract_size)", ge=16, le=2048)
    rescale: Optional[int] = Field(default=64, description="Rescale to this box (--scale)", ge=16, le=2048)
    normalize: bool = Field(default=True, description="Normalize particles (--norm)")
    bg_radius: int = Field(default=200, description="Background mask radius px (--bg_radius)")
    invert_contrast: bool = Field(default=True, description="Invert contrast — needed for cryo data (--invert_contrast)")
    white_dust: float = Field(default=-1.0, description="Remove white dust >N sigma, -1=off (--white_dust)")
    black_dust: float = Field(default=-1.0, description="Remove black dust >N sigma, -1=off (--black_dust)")
    float16: bool = Field(default=True, description="Write float16 MRCs (--float16)")
    mpi: int = Field(default=DEFAULT_MPI, ge=1, le=128)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False, description="False=preview, True=run")


@mcp.tool(
    name="relion_extract",
    annotations={
        "readOnlyHint": False, "destructiveHint": False,
        "idempotentHint": False, "openWorldHint": False,
    },
)
async def relion_extract(params: ExtractInput) -> str:
    """Extract particle images from micrographs at picked coordinates.
    Call with confirm=False to preview, confirm=True to execute."""
    if not params.confirm:
        return _preview_params(
            params.model_dump(), _EXTRACT_TUTORIAL,
            _EXTRACT_REQUIRED, "Particle Extraction",
        )
    job_dir = _next_job("Extract")
    os.makedirs(job_dir, exist_ok=True)
    prog = _cmd("relion_preprocess" + ("_mpi" if params.mpi > 1 else ""))
    cmd = _mpi_prefix(prog, params.mpi) + [
        "--i", _resolve(params.input_star),
        "--coord_suffix", params.coord_suffix,
        "--part_star", os.path.join(job_dir, "particles.star"),
        "--part_dir", os.path.join(job_dir, "Particles/"),
        "--extract",
        "--extract_size", str(params.extract_size),
    ]
    if params.rescale is not None and params.rescale > 0:
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
    if params.extra_args:
        cmd += params.extra_args
    result = await _run(cmd, timeout=86400)
    if result["returncode"] == 0:
        _mark_success(job_dir)
    return _format_result("Particle Extraction", job_dir, result)


# =====================================================================
# PIPELINE TOOL 6 — 2D Classification
# =====================================================================

_CLASS2D_TUTORIAL: Dict[str, Any] = {
    "num_classes": 50,
    "num_iterations": 25,
    "tau_fudge": 2.0,
    "particle_diameter": 200.0,
    "do_ctf": True,
    "center_classes": True,
    "threads": DEFAULT_THREADS,
    "mpi": DEFAULT_MPI,
}
_CLASS2D_REQUIRED = ["input_star", "particle_diameter"]


class Class2DInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_star: str = Field(..., description="Particles STAR file (--i)")
    num_classes: int = Field(default=50, description="Number of 2D classes (--K)", ge=2, le=500)
    num_iterations: int = Field(default=25, description="Iterations (--iter)", ge=1, le=200)
    particle_diameter: float = Field(..., description="Mask diameter Å (--particle_diameter)", gt=0)
    tau_fudge: float = Field(default=2.0, description="Regularisation (--tau2_fudge)", gt=0)
    do_ctf: bool = Field(default=True, description="Perform CTF correction (--ctf)")
    center_classes: bool = Field(default=True, description="Center class averages (--center_classes)")
    threads: int = Field(default=DEFAULT_THREADS, ge=1, le=64)
    mpi: int = Field(default=DEFAULT_MPI, ge=1, le=128)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False, description="False=preview, True=run")


@mcp.tool(
    name="relion_class2d",
    annotations={
        "readOnlyHint": False, "destructiveHint": False,
        "idempotentHint": False, "openWorldHint": False,
    },
)
async def relion_class2d(params: Class2DInput) -> str:
    """Reference-free 2D classification for particle quality assessment.
    Call with confirm=False to preview, confirm=True to execute."""
    if not params.confirm:
        return _preview_params(
            params.model_dump(), _CLASS2D_TUTORIAL,
            _CLASS2D_REQUIRED, "2D Classification",
        )
    job_dir = _next_job("Class2D")
    os.makedirs(job_dir, exist_ok=True)
    prog = _cmd("relion_refine" + ("_mpi" if params.mpi > 1 else ""))
    cmd = _mpi_prefix(prog, params.mpi) + [
        "--i", _resolve(params.input_star),
        "--o", os.path.join(job_dir, "run"),
        "--K", str(params.num_classes),
        "--iter", str(params.num_iterations),
        "--particle_diameter", str(params.particle_diameter),
        "--tau2_fudge", str(params.tau_fudge),
        "--flatten_solvent", "--zero_mask",
        "--oversampling", "1",
        "--psi_step", "12",
        "--offset_range", "5",
        "--offset_step", "2",
        "--dont_combine_weights_via_disc",
        "--j", str(params.threads),
    ]
    if params.do_ctf:
        cmd.append("--ctf")
    if params.center_classes:
        cmd.append("--center_classes")
    if params.extra_args:
        cmd += params.extra_args
    result = await _run(cmd, timeout=86400 * 3)
    if result["returncode"] == 0:
        _mark_success(job_dir)
    return _format_result("2D Classification", job_dir, result)


# =====================================================================
# PIPELINE TOOL 7 — Initial Model (VDAM — NEW)
# =====================================================================

_INIMODEL_TUTORIAL: Dict[str, Any] = {
    "num_classes": 1,
    "particle_diameter": 200.0,
    "symmetry": "C1",
    "num_iterations": 200,
    "do_ctf": True,
    "threads": DEFAULT_THREADS,
    "mpi": DEFAULT_MPI,
}
_INIMODEL_REQUIRED = ["input_star", "particle_diameter"]


class InitialModelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_star: str = Field(..., description="Particles STAR file (--i)")
    num_classes: int = Field(default=1, description="Number of initial models (--K)", ge=1, le=10)
    particle_diameter: float = Field(..., description="Mask diameter Å (--particle_diameter)", gt=0)
    symmetry: str = Field(default="C1", description="Point-group symmetry (--sym)")
    num_iterations: int = Field(default=200, description="Max VDAM iterations (--iter)", ge=1, le=999)
    do_ctf: bool = Field(default=True, description="Apply CTF correction (--ctf)")
    flatten_solvent: bool = Field(default=True, description="Flatten solvent (--flatten_solvent)")
    threads: int = Field(default=DEFAULT_THREADS, ge=1, le=64)
    mpi: int = Field(default=DEFAULT_MPI, ge=1, le=128)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False, description="False=preview, True=run")


@mcp.tool(
    name="relion_initial_model",
    annotations={
        "readOnlyHint": False, "destructiveHint": False,
        "idempotentHint": False, "openWorldHint": False,
    },
)
async def relion_initial_model(params: InitialModelInput) -> str:
    """Generate ab-initio 3D initial model using VDAM (relion_refine --denovo_3dref).
    Call with confirm=False to preview, confirm=True to execute."""
    if not params.confirm:
        return _preview_params(
            params.model_dump(), _INIMODEL_TUTORIAL,
            _INIMODEL_REQUIRED, "Initial Model (VDAM)",
        )
    job_dir = _next_job("InitialModel")
    os.makedirs(job_dir, exist_ok=True)
    prog = _cmd("relion_refine" + ("_mpi" if params.mpi > 1 else ""))
    cmd = _mpi_prefix(prog, params.mpi) + [
        "--i", _resolve(params.input_star),
        "--o", os.path.join(job_dir, "run"),
        "--denovo_3dref",
        "--K", str(params.num_classes),
        "--iter", str(params.num_iterations),
        "--particle_diameter", str(params.particle_diameter),
        "--sym", params.symmetry,
        "--zero_mask",
        "--oversampling", "1",
        "--dont_combine_weights_via_disc",
        "--j", str(params.threads),
    ]
    if params.do_ctf:
        cmd.append("--ctf")
    if params.flatten_solvent:
        cmd.append("--flatten_solvent")
    if params.extra_args:
        cmd += params.extra_args
    result = await _run(cmd, timeout=86400 * 3)
    if result["returncode"] == 0:
        _mark_success(job_dir)
    return _format_result("Initial Model (VDAM)", job_dir, result)


# =====================================================================
# PIPELINE TOOL 8 — 3D Classification
# =====================================================================

_CLASS3D_TUTORIAL: Dict[str, Any] = {
    "num_classes": 4,
    "num_iterations": 25,
    "particle_diameter": 200.0,
    "symmetry": "C1",
    "initial_lowpass": 50.0,
    "tau_fudge": 4.0,
    "do_ctf": True,
    "healpix_order": 2,
    "skip_gridding": False,
    "threads": DEFAULT_THREADS,
    "mpi": DEFAULT_MPI,
}
_CLASS3D_REQUIRED = ["input_star", "reference_map", "particle_diameter"]


class Class3DInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_star: str = Field(..., description="Particles STAR (--i)")
    reference_map: str = Field(..., description="Reference .mrc (--ref)")
    num_classes: int = Field(default=4, description="3D classes (--K)", ge=2, le=50)
    num_iterations: int = Field(default=25, ge=1, le=200)
    particle_diameter: float = Field(..., description="Mask diameter Å", gt=0)
    symmetry: str = Field(default="C1", description="Symmetry (--sym)")
    initial_lowpass: float = Field(default=50.0, description="Initial LP Å (--ini_high)", gt=0)
    tau_fudge: float = Field(default=4.0, gt=0)
    do_ctf: bool = Field(default=True, description="Apply CTF correction (--ctf)")
    healpix_order: int = Field(default=2, description="Angular sampling order (--healpix_order)", ge=1, le=6)
    skip_gridding: bool = Field(default=False, description="Skip gridding in reconstruction (--skip_gridding)")
    threads: int = Field(default=DEFAULT_THREADS, ge=1, le=64)
    mpi: int = Field(default=DEFAULT_MPI, ge=1, le=128)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False, description="False=preview, True=run")


@mcp.tool(
    name="relion_class3d",
    annotations={
        "readOnlyHint": False, "destructiveHint": False,
        "idempotentHint": False, "openWorldHint": False,
    },
)
async def relion_class3d(params: Class3DInput) -> str:
    """3D classification to separate conformational states.
    Call with confirm=False to preview, confirm=True to execute."""
    if not params.confirm:
        return _preview_params(
            params.model_dump(), _CLASS3D_TUTORIAL,
            _CLASS3D_REQUIRED, "3D Classification",
        )
    job_dir = _next_job("Class3D")
    os.makedirs(job_dir, exist_ok=True)
    prog = _cmd("relion_refine" + ("_mpi" if params.mpi > 1 else ""))
    cmd = _mpi_prefix(prog, params.mpi) + [
        "--i", _resolve(params.input_star),
        "--ref", _resolve(params.reference_map),
        "--o", os.path.join(job_dir, "run"),
        "--K", str(params.num_classes),
        "--iter", str(params.num_iterations),
        "--particle_diameter", str(params.particle_diameter),
        "--sym", params.symmetry,
        "--ini_high", str(params.initial_lowpass),
        "--tau2_fudge", str(params.tau_fudge),
        "--flatten_solvent", "--zero_mask",
        "--oversampling", "1",
        "--healpix_order", str(params.healpix_order),
        "--offset_range", "5",
        "--offset_step", "2",
        "--dont_combine_weights_via_disc",
        "--j", str(params.threads),
    ]
    if params.do_ctf:
        cmd.append("--ctf")
    if params.skip_gridding:
        cmd.append("--skip_gridding")
    if params.extra_args:
        cmd += params.extra_args
    result = await _run(cmd, timeout=86400 * 7)
    if result["returncode"] == 0:
        _mark_success(job_dir)
    return _format_result("3D Classification", job_dir, result)


# =====================================================================
# PIPELINE TOOL 9 — 3D Refinement
# =====================================================================

_REFINE3D_TUTORIAL: Dict[str, Any] = {
    "particle_diameter": 200.0,
    "symmetry": "D2",
    "initial_lowpass": 50.0,
    "ref_is_correct_greyscale": True,
    "threads": DEFAULT_THREADS,
    "mpi": 3,
}
_REFINE3D_REQUIRED = ["input_star", "reference_map", "particle_diameter"]


class Refine3DInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_star: str = Field(..., description="Particles STAR (--i)")
    reference_map: str = Field(..., description="Reference .mrc (--ref)")
    particle_diameter: float = Field(..., description="Mask diameter Å", gt=0)
    symmetry: str = Field(default="D2", description="Symmetry (--sym)")
    initial_lowpass: float = Field(default=50.0, description="Initial LP Å (--ini_high)", gt=0)
    solvent_mask: Optional[str] = Field(default=None, description="Mask .mrc (--solvent_mask)")
    ref_is_correct_greyscale: bool = Field(
        default=True,
        description="Reference has correct greyscale (--ref_correct_greyscale). "
                    "Set False if from PDB/external.",
    )
    threads: int = Field(default=DEFAULT_THREADS, ge=1, le=64)
    mpi: int = Field(default=3, description="MPI procs (odd ≥3 for gold-standard)", ge=1, le=128)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False, description="False=preview, True=run")

    @field_validator("mpi")
    @classmethod
    def validate_mpi_odd_ge3(cls, v: int) -> int:
        if v > 1 and (v < 3 or v % 2 == 0):
            raise ValueError(
                "For 3D auto-refine, MPI must be odd and ≥3 "
                "(master + 2N workers for gold-standard half-sets). Got: "
                f"{v}. Try 3, 5, 7, …"
            )
        return v


@mcp.tool(
    name="relion_refine3d",
    annotations={
        "readOnlyHint": False, "destructiveHint": False,
        "idempotentHint": False, "openWorldHint": False,
    },
)
async def relion_refine3d(params: Refine3DInput) -> str:
    """Gold-standard 3D auto-refinement for high-resolution structures.
    MPI must be odd ≥3 for gold-standard split. confirm=False to preview."""
    if not params.confirm:
        return _preview_params(
            params.model_dump(), _REFINE3D_TUTORIAL,
            _REFINE3D_REQUIRED, "3D Auto-Refinement",
        )
    job_dir = _next_job("Refine3D")
    os.makedirs(job_dir, exist_ok=True)
    prog = _cmd("relion_refine" + ("_mpi" if params.mpi > 1 else ""))
    cmd = _mpi_prefix(prog, params.mpi) + [
        "--i", _resolve(params.input_star),
        "--ref", _resolve(params.reference_map),
        "--o", os.path.join(job_dir, "run"),
        "--auto_refine", "--split_random_halves",
        "--particle_diameter", str(params.particle_diameter),
        "--sym", params.symmetry,
        "--ini_high", str(params.initial_lowpass),
        "--flatten_solvent", "--zero_mask",
        "--oversampling", "1", "--pad", "2",
        "--healpix_order", "2",
        "--auto_local_healpix_order", "4",
        "--offset_range", "5",
        "--offset_step", "2",
        "--dont_combine_weights_via_disc",
        "--j", str(params.threads),
    ]
    if params.ref_is_correct_greyscale:
        cmd.append("--ref_correct_greyscale")
    if params.solvent_mask:
        cmd += ["--solvent_mask", _resolve(params.solvent_mask)]
    if params.extra_args:
        cmd += params.extra_args
    result = await _run(cmd, timeout=86400 * 7)
    if result["returncode"] == 0:
        _mark_success(job_dir)
    return _format_result("3D Refinement", job_dir, result)


# =====================================================================
# PIPELINE TOOL 10 — Mask Create (NEW)
# =====================================================================

_MASK_TUTORIAL: Dict[str, Any] = {
    "lowpass_filter": 15.0,
    "threshold": 0.005,
    "extend_inimask": 0,
    "width_soft_edge": 6,
}
_MASK_REQUIRED = ["input_map"]


class MaskCreateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_map: str = Field(..., description="Input map .mrc (--i)")
    lowpass_filter: float = Field(default=15.0, description="Initial lowpass Å (--lowpass)", gt=0)
    threshold: float = Field(default=0.005, description="Binarization threshold (--ini_threshold)", gt=0)
    extend_inimask: int = Field(default=0, description="Extend binary mask by N pixels (--extend_inimask)", ge=0)
    width_soft_edge: int = Field(default=6, description="Soft edge width pixels (--width_soft_edge)", ge=0)
    angpix: float = Field(default=-1, description="Pixel size Å, -1=from header (--angpix)")
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False, description="False=preview, True=run")


@mcp.tool(
    name="relion_mask_create",
    annotations={
        "readOnlyHint": False, "destructiveHint": False,
        "idempotentHint": False, "openWorldHint": False,
    },
)
async def relion_mask_create(params: MaskCreateInput) -> str:
    """Create a solvent mask from a reference map (relion_mask_create).
    Call with confirm=False to preview, confirm=True to execute."""
    if not params.confirm:
        return _preview_params(
            params.model_dump(), _MASK_TUTORIAL,
            _MASK_REQUIRED, "Mask Creation",
        )
    job_dir = _next_job("MaskCreate")
    os.makedirs(job_dir, exist_ok=True)
    output_mask = os.path.join(job_dir, "mask.mrc")
    cmd = [
        _cmd("relion_mask_create"),
        "--i", _resolve(params.input_map),
        "--o", output_mask,
        "--lowpass", str(params.lowpass_filter),
        "--ini_threshold", str(params.threshold),
        "--extend_inimask", str(params.extend_inimask),
        "--width_soft_edge", str(params.width_soft_edge),
    ]
    if params.angpix > 0:
        cmd += ["--angpix", str(params.angpix)]
    if params.extra_args:
        cmd += params.extra_args
    result = await _run(cmd, timeout=3600)
    if result["returncode"] == 0:
        _mark_success(job_dir)
    return _format_result("Mask Creation", job_dir, result)


# =====================================================================
# PIPELINE TOOL 11 — PostProcess
# =====================================================================

_POSTPROCESS_TUTORIAL: Dict[str, Any] = {
    "auto_bfac": True,
    "autob_lowres": 10.0,
    "autob_highres": 0.0,
    "low_pass": 0,
    "skip_fsc_weighting": False,
    "mtf_angpix": -1.0,
}
_POSTPROCESS_REQUIRED = ["half1_map", "mask"]


class PostProcessInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    half1_map: str = Field(..., description="Half-map 1 (--i)")
    mask: str = Field(..., description="Mask .mrc (--mask)")
    angpix: float = Field(default=-1, description="Pixel size Å, -1=header (--angpix)")
    auto_bfac: bool = Field(default=True, description="Auto B-factor (--auto_bfac)")
    autob_lowres: float = Field(default=10.0, description="Low-res limit for B-fit Å (--autob_lowres)")
    autob_highres: float = Field(default=0.0, description="High-res limit for B-fit Å, 0=Nyquist (--autob_highres)")
    adhoc_bfac: Optional[float] = Field(default=None, description="Manual B-factor Å² (--adhoc_bfac)")
    mtf: Optional[str] = Field(default=None, description="MTF STAR file (--mtf)")
    mtf_angpix: float = Field(default=-1.0, description="MTF pixel size Å, -1=same as map (--mtf_angpix)")
    low_pass: float = Field(default=0, description="Low-pass Å, 0=disable (--low_pass)")
    skip_fsc_weighting: bool = Field(default=False, description="Skip FSC weighting (--skip_fsc_weighting)")
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False, description="False=preview, True=run")


@mcp.tool(
    name="relion_postprocess",
    annotations={
        "readOnlyHint": False, "destructiveHint": False,
        "idempotentHint": False, "openWorldHint": False,
    },
)
async def relion_postprocess(params: PostProcessInput) -> str:
    """Post-process: masking, B-factor sharpening, FSC resolution estimation.
    Call with confirm=False to preview, confirm=True to execute."""
    if not params.confirm:
        return _preview_params(
            params.model_dump(), _POSTPROCESS_TUTORIAL,
            _POSTPROCESS_REQUIRED, "Post-Processing",
        )
    job_dir = _next_job("PostProcess")
    os.makedirs(job_dir, exist_ok=True)
    cmd = [
        _cmd("relion_postprocess"),
        "--i", _resolve(params.half1_map),
        "--mask", _resolve(params.mask),
        "--o", os.path.join(job_dir, "postprocess"),
    ]
    if params.angpix > 0:
        cmd += ["--angpix", str(params.angpix)]
    if params.auto_bfac:
        cmd.append("--auto_bfac")
        cmd += ["--autob_lowres", str(params.autob_lowres)]
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
    result = await _run(cmd, timeout=3600)
    if result["returncode"] == 0:
        _mark_success(job_dir)
    # Extract resolution from output
    resolution = ""
    fsc_file = os.path.join(job_dir, "postprocess.star")
    if os.path.isfile(fsc_file):
        with open(fsc_file) as f:
            m = re.search(r"_rlnFinalResolution\s+(\d+\.?\d*)", f.read())
            if m:
                resolution = f"\n\n**Final Resolution: {m.group(1)} Å**"
    return _format_result("Post-Processing", job_dir, result) + resolution


# =====================================================================
# PIPELINE TOOL 12 — CTF Refinement (NEW)
# =====================================================================

_CTFREFINE_TUTORIAL: Dict[str, Any] = {
    "do_aniso_mag": False,
    "do_defocus": True,
    "do_astig": True,
    "do_bfactor": False,
    "do_trefoil": False,
    "do_4thorder": False,
    "threads": DEFAULT_THREADS,
    "mpi": DEFAULT_MPI,
}
_CTFREFINE_REQUIRED = ["input_star", "postprocess_star"]


class CtfRefineInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_star: str = Field(..., description="Particles STAR from refinement (--i)")
    postprocess_star: str = Field(..., description="PostProcess STAR for FSC weighting (--f)")
    do_aniso_mag: bool = Field(default=False, description="Anisotropic magnification (--do_aniso_mag)")
    do_defocus: bool = Field(default=True, description="Per-particle defocus (--fit_defocus)")
    do_astig: bool = Field(default=True, description="Per-micrograph astigmatism (--fit_astig)")
    do_bfactor: bool = Field(default=False, description="Per-particle B-factor (--fit_bfac)")
    do_trefoil: bool = Field(default=False, description="Trefoil aberrations (--do_trefoil)")
    do_4thorder: bool = Field(default=False, description="4th order aberrations (--do_4thorder)")
    threads: int = Field(default=DEFAULT_THREADS, ge=1, le=64)
    mpi: int = Field(default=DEFAULT_MPI, ge=1, le=128)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False, description="False=preview, True=run")


@mcp.tool(
    name="relion_ctf_refine",
    annotations={
        "readOnlyHint": False, "destructiveHint": False,
        "idempotentHint": False, "openWorldHint": False,
    },
)
async def relion_ctf_refine(params: CtfRefineInput) -> str:
    """CTF refinement: per-particle defocus, beam tilt, aberrations (relion_ctf_refine).
    Call with confirm=False to preview, confirm=True to execute."""
    if not params.confirm:
        return _preview_params(
            params.model_dump(), _CTFREFINE_TUTORIAL,
            _CTFREFINE_REQUIRED, "CTF Refinement",
        )
    job_dir = _next_job("CtfRefine")
    os.makedirs(job_dir, exist_ok=True)
    prog = _cmd("relion_ctf_refine" + ("_mpi" if params.mpi > 1 else ""))
    cmd = _mpi_prefix(prog, params.mpi) + [
        "--i", _resolve(params.input_star),
        "--f", _resolve(params.postprocess_star),
        "--o", os.path.join(job_dir, "run"),
        "--j", str(params.threads),
    ]
    if params.do_defocus:
        cmd.append("--fit_defocus")
    if params.do_astig:
        cmd.append("--fit_astig")
    if params.do_bfactor:
        cmd.append("--fit_bfac")
    if params.do_aniso_mag:
        cmd.append("--do_aniso_mag")
    if params.do_trefoil:
        cmd.append("--do_trefoil")
    if params.do_4thorder:
        cmd.append("--do_4thorder")
    if params.extra_args:
        cmd += params.extra_args
    result = await _run(cmd, timeout=86400 * 3)
    if result["returncode"] == 0:
        _mark_success(job_dir)
    return _format_result("CTF Refinement", job_dir, result)


# =====================================================================
# PIPELINE TOOL 13 — Bayesian Polishing (NEW)
# =====================================================================

_POLISHING_TUTORIAL: Dict[str, Any] = {
    "first_frame": 1,
    "last_frame": -1,
    "threads": DEFAULT_THREADS,
    "mpi": DEFAULT_MPI,
}
_POLISHING_REQUIRED = ["input_star", "postprocess_star"]


class BayesianPolishingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_star: str = Field(..., description="Particles STAR from refinement (--i)")
    postprocess_star: str = Field(..., description="PostProcess STAR for FSC (--f)")
    first_frame: int = Field(default=1, description="First movie frame (--first_frame)", ge=1)
    last_frame: int = Field(default=-1, description="Last movie frame, -1=all (--last_frame)")
    threads: int = Field(default=DEFAULT_THREADS, ge=1, le=64)
    mpi: int = Field(default=DEFAULT_MPI, ge=1, le=128)
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False, description="False=preview, True=run")


@mcp.tool(
    name="relion_bayesian_polishing",
    annotations={
        "readOnlyHint": False, "destructiveHint": False,
        "idempotentHint": False, "openWorldHint": False,
    },
)
async def relion_bayesian_polishing(params: BayesianPolishingInput) -> str:
    """Per-particle Bayesian polishing (relion_motion_refine).
    Improves resolution by fitting per-particle motion trajectories.
    Call with confirm=False to preview, confirm=True to execute."""
    if not params.confirm:
        return _preview_params(
            params.model_dump(), _POLISHING_TUTORIAL,
            _POLISHING_REQUIRED, "Bayesian Polishing",
        )
    job_dir = _next_job("Polish")
    os.makedirs(job_dir, exist_ok=True)
    prog = _cmd("relion_motion_refine" + ("_mpi" if params.mpi > 1 else ""))
    cmd = _mpi_prefix(prog, params.mpi) + [
        "--i", _resolve(params.input_star),
        "--f", _resolve(params.postprocess_star),
        "--o", os.path.join(job_dir, "run"),
        "--first_frame", str(params.first_frame),
        "--last_frame", str(params.last_frame),
        "--j", str(params.threads),
    ]
    if params.extra_args:
        cmd += params.extra_args
    result = await _run(cmd, timeout=86400 * 3)
    if result["returncode"] == 0:
        _mark_success(job_dir)
    return _format_result("Bayesian Polishing", job_dir, result)


# =====================================================================
# PIPELINE TOOL 14 — Blush AI Denoising
# =====================================================================


class BlushInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    half1_map: str = Field(..., description="Half-map 1 (.mrc)")
    half2_map: str = Field(..., description="Half-map 2 (.mrc)")
    mask: Optional[str] = Field(default=None, description="Solvent mask (.mrc)")
    extra_args: Optional[List[str]] = Field(default=None)
    confirm: bool = Field(default=False, description="False=preview, True=run")


_BLUSH_TUTORIAL: Dict[str, Any] = {}
_BLUSH_REQUIRED = ["half1_map", "half2_map"]


@mcp.tool(
    name="relion_blush",
    annotations={
        "readOnlyHint": False, "destructiveHint": False,
        "idempotentHint": False, "openWorldHint": False,
    },
)
async def relion_blush(params: BlushInput) -> str:
    """AI map denoising with Blush (RELION 5). Requires Python+PyTorch environment.
    Call with confirm=False to preview, confirm=True to execute."""
    if not params.confirm:
        return _preview_params(
            params.model_dump(), _BLUSH_TUTORIAL,
            _BLUSH_REQUIRED, "Blush (AI Denoising)",
        )
    job_dir = _next_job("Blush")
    os.makedirs(job_dir, exist_ok=True)
    cmd = [
        _cmd("relion_python_blush"),
        "--i1", _resolve(params.half1_map),
        "--i2", _resolve(params.half2_map),
        "--o", os.path.join(job_dir, "blush"),
    ]
    if params.mask:
        cmd += ["--mask", _resolve(params.mask)]
    if params.extra_args:
        cmd += params.extra_args
    result = await _run(cmd, timeout=7200)
    if result["returncode"] == 0:
        _mark_success(job_dir)
    return _format_result("Blush (AI Denoising)", job_dir, result)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RELION MCP Server v2.0")
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

# RELION MCP Server v2.1

An MCP (Model Context Protocol) server that lets AI agents drive [RELION 5.x](https://github.com/3dem/relion) — the gold-standard software for cryo-EM structure determination.

> **Tested and verified** against RELION 5.0.1 on Ubuntu 24.04 (WSL2). All CLI flags validated against actual `--help` output.

## What It Does

An AI agent (Claude Code, OpenClaw, NemoClaw, etc.) can process cryo-EM data through natural language:

```
You: "Import the movies from Movies/*.tiff, 200 kV, pixel size 0.885 Å, then run motion correction"
Agent: → relion_import(..., confirm=False)  → shows parameter preview
You: "Looks good, launch it"
Agent: → relion_import(..., confirm=True)   → job runs (instant)
       → relion_motioncorr(..., confirm=False) → preview
You: "Ok go"
Agent: → relion_motioncorr(..., confirm=True) → 🚀 Launched (PID 12345)
       → relion_job_status("MotionCorr/job001") → 🔄 RUNNING
       → relion_job_status("MotionCorr/job001") → ✅ COMPLETED
```

The server exposes **21 tools** covering the complete single-particle analysis pipeline, from import to Bayesian polishing.

## Key Features

### 1. Preview Before Launch

Every pipeline tool uses a **preview/confirm** system to prevent costly mistakes:

- **`confirm=False`** (default) — the tool does **not** launch the job. Instead it returns a full parameter list:
  - ✏️ **défini** — value provided by the user
  - 📋 **tutorial** — RELION 5 tutorial default (EMPIAR-10204, beta-galactosidase)
  - ❌ **requis** — required parameter still missing
  - ⬜ **optionnel** — optional, not set
- **`confirm=True`** — the tool launches the job in background

### 2. Non-Blocking Background Execution

All long-running jobs (MotionCorr, CTF, Class2D, Refine3D, etc.) launch in **background mode**:

- `confirm=True` returns **immediately** with the PID and job directory
- The agent stays responsive and can answer questions, check other jobs, etc.
- A wrapper script automatically creates `RELION_JOB_EXIT_SUCCESS` or `_FAILURE` markers
- Use `relion_job_status` and `relion_job_logs` to monitor progress

Only `relion_import` (instant) and `relion_run_command` (escape hatch) run synchronously.

### 3. Live Flag Discovery

`relion_help` runs any `relion_* --help` in real time and returns parsed flags with descriptions and defaults. Supports keyword filtering (e.g. `search="gpu"`).

## Architecture

```
AI Agent (Claude Code / OpenClaw / NemoClaw)
    │
    │  stdio or HTTP
    ▼
RELION MCP Server v2.1 (Python)
    │
    │  Popen (detached)        subprocess.run (short jobs)
    ▼                          ▼
RELION 5.x binaries        relion_import, relion_help
(background, non-blocking)  (synchronous, fast)
```

## Tools

### Pipeline Tools (14 tools — all with preview/confirm)

| Tool | RELION Binary | Description |
|------|--------------|-------------|
| `relion_import` | `relion_import` | Import movies/micrographs |
| `relion_motioncorr` | `relion_run_motioncorr` | Beam-induced motion correction |
| `relion_ctffind` | `relion_run_ctffind` | CTF estimation (via CTFFIND4) |
| `relion_autopick` | `relion_autopick` | Particle picking (LoG or template) |
| `relion_extract` | `relion_preprocess` | Particle extraction |
| `relion_class2d` | `relion_refine` | 2D classification |
| `relion_initial_model` | `relion_refine --denovo_3dref` | Ab-initio 3D model (VDAM) **NEW** |
| `relion_class3d` | `relion_refine` | 3D classification |
| `relion_refine3d` | `relion_refine` | 3D auto-refinement |
| `relion_mask_create` | `relion_mask_create` | Solvent mask creation **NEW** |
| `relion_postprocess` | `relion_postprocess` | Map sharpening & resolution |
| `relion_ctf_refine` | `relion_ctf_refine` | Per-particle CTF refinement **NEW** |
| `relion_bayesian_polishing` | `relion_motion_refine` | Per-particle Bayesian polishing **NEW** |
| `relion_blush` | `relion_python_blush` | AI map denoising (RELION 5) |

### Read-Only Tools (7 tools — no confirm needed)

| Tool | Description |
|------|-------------|
| `relion_project_info` | Project overview: jobs, counts, status |
| `relion_read_star` | Parse STAR files (RELION's metadata format) |
| `relion_job_status` | Check job status: SUCCESS/FAILURE/RUNNING + PID detection + stderr tail |
| `relion_job_logs` | **NEW** Read stdout/stderr logs from background jobs (with tail) |
| `relion_suggest_next_step` | Recommend next pipeline step (13-step pipeline) |
| `relion_run_command` | Run any `relion_*` binary with custom args (escape hatch, synchronous) |
| `relion_help` | **NEW** Run `relion_* --help` and return parsed flags, descriptions & defaults. Supports keyword filtering. |

## Tutorial Defaults (EMPIAR-10204)

All pipeline tools ship with defaults matching the RELION 5 beta-galactosidase tutorial:

| Step | Key defaults |
|------|-------------|
| Import | 200 kV, 0.885 Å, Cs 1.4, Q0 0.1 |
| MotionCorr | dose 1.277 e⁻/Å²/frame, patches 5×5, bfactor 150, float16 |
| CTF | Box 512, ResMin 30 Å, ResMax 5 Å, dF 5000–50000 Å, dAst 100 |
| AutoPick | LoG, diam 150–180 Å |
| Extract | box 256 → rescale 64, invert contrast, bg_radius 200 |
| Class2D | K=50, iter 25, T=2, mask 200 Å, CTF on |
| Class3D | K=4, iter 25, T=4, C1, ini_high 50 Å |
| Refine3D | D2, ini_high 50 Å, MPI=3 (odd ≥ 3 validated) |
| Mask | lowpass 15 Å, threshold 0.005, soft edge 6 px |
| PostProcess | auto B-factor, autob_lowres 10 Å |

## Prerequisites

- **RELION 5.x** compiled and in `PATH`
- **Python ≥ 3.10**
- **MCP Python SDK**:
  ```bash
  pip install mcp pydantic
  ```
  - mcp >= 1.0.0
  - pydantic >= 2.0.0
  - uvicorn >= 0.30.0 (for HTTP mode)

## Installation

```bash
git clone https://github.com/kdursunnizam-art/relion-mcp-server.git
cd ~/relion-mcp-server
```

## Usage

### With Claude Code (recommended for local use)

From your terminal, run this command:
```bash
claude mcp add-json relion '{"command":"python3","args":["/path/to/relion-mcp-server/relion_mcp.py"],"env":{"RELION_PROJECT_DIR":"/path/to/data/relion_tutorial"}}' --scope user
```
Verify it's registered:
```bash
claude mcp list
```
To remove and reconfigure:
```bash
claude mcp remove relion
```
Note: --scope user makes the server available in all your projects.
The command automatically writes to the correct config file — no need to
find or edit ~/.claude.json manually.

Then in Claude Code:
```
> Use relion_project_info to show the project status
> Import movies from Movies/*.tiff with pixel size 0.885, 200 kV, Cs 1.4
> Run motion correction with dose 1.277 e-/Å²/frame and gain ref Movies/gain.mrc
> Show me the Class2D parameters before running (agent calls with confirm=False)
> Change threads to 8 and launch (agent calls with confirm=True)
```

### With OpenClaw / NemoClaw (remote access)

Start the server in HTTP mode:

```bash
cd /path/to/relion_mcp.py/
source venv/bin/activate
export RELION_PROJECT_DIR=/data/my_project
python relion_mcp.py --transport http --port 8000 --host 0.0.0.0
```

Configure openclaw.json:

```json
"skills": {
  "install": {
    "nodeManager": "npm"
  },
  "entries": {
    "mcp-integration": {
      "enabled": true,
      "config": {
        "servers": [
          {
            "name": "relion-stdio",
            "transport": "stdio",
            "command": "python3",
            "args": ["/chemin/vers/relion-mcp-server/relion_mcp.py"],
            "env": {
              "RELION_PROJECT_DIR": "/chemin/vers/votre/projet_relion"
            }
          },
          {
            "name": "relion-http",
            "transport": "streamable-http",
            "url": "http://127.0.0.1:8000"
          }
        ],
        "toolPrefix": true
      }
    }
  }
}
```

### WSL2 Setup (Windows)

This server works great in WSL2 with RELION compiled in CPU-only mode:

```bash
# In WSL2 Ubuntu
cmake -DCUDA=OFF -DCMAKE_INSTALL_PREFIX=$HOME/relion-install ..
make -j4 && make install

# Start Claude Code
cd /your/relion/project
claude
```

Claude Code on Windows connects to the MCP server in WSL2 seamlessly via stdio.

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `RELION_PROJECT_DIR` | RELION project directory | Current directory |
| `RELION_BIN` | Path prefix for RELION binaries | (uses PATH) |
| `RELION_THREADS` | Default thread count | 4 |
| `RELION_MPI` | Default MPI processes | 1 |

## Security

- Only `relion_*` executables can be run (validated)
- No shell injection: uses `subprocess.run` without `shell=True`
- File paths resolved relative to project directory
- In HTTP mode, binds to `127.0.0.1` by default
- Preview/confirm prevents accidental job launches

## MCP SDK Compatibility

This server is designed for **MCP SDK 1.26+** with the following constraints:
- No `lifespan` (causes crash with MCP SDK 1.26)
- No `ctx.report_progress` (causes crash with MCP SDK 1.26)
- All tool functions are `async` without `Context` parameter
- Passes `python3 -m py_compile` cleanly

## Known Limitations

- **No GPU support in tools**: GPU device selection is not exposed (RELION uses `--gpu` flag). Use `relion_run_command` with `--gpu 0` for GPU jobs, or use `relion_help(program="relion_refine", search="gpu")` to discover GPU flags.

## Changelog

### v2.1 (current)
- **Background execution**: all long-running jobs now launch via `Popen(start_new_session=True)` and return immediately with PID. No more agent blocking.
- **`relion_job_logs`**: new tool to read stdout/stderr from background jobs in real time
- **`relion_job_status` enhanced**: detects if PID is alive, shows stderr tail on failure, distinguishes RUNNING vs IDLE (crashed)
- **`relion_help`**: new tool to run `relion_* --help` and parse all flags live, with keyword filtering
- Wrapper script (`run.sh`) in each job_dir auto-creates SUCCESS/FAILURE markers
- 21 tools total (7 read-only + 14 pipeline)

### v2.0
- **Preview/confirm system** on all 14 pipeline tools
- **5 new tools**: `relion_initial_model`, `relion_mask_create`, `relion_ctf_refine`, `relion_bayesian_polishing`, `relion_help`
- **Parameters added**: bfactor, gain_rot/flip, float16, save_ps, d_ast, phase shift, invert_contrast, white/black dust, --ctf flag, center_classes, healpix_order, skip_gridding, ref_correct_greyscale, MPI validation, autob_lowres/highres, mtf_angpix, skip_fsc_weighting
- **Tutorial defaults** from EMPIAR-10204 (beta-galactosidase) baked in
- 20 tools total (was 15)

### v1.0
- Initial release with 15 tools
- Verified against RELION 5.0.1

## Tested With

- RELION 5.0.1 (commit cad71bf)
- Ubuntu 24.04 LTS (WSL2)
- Python 3.12, MCP SDK 1.26.0
- Claude Code 2.1.89
- Openclaw 2026.4.2 (commit d74a122)
- Tutorial dataset: beta-galactosidase (EMPIAR-10204)

## License

MIT — RELION itself is GPLv2. This server interacts with RELION solely through its CLI.

## References

- Scheres, S.H.W. (2012). RELION: Implementation of a Bayesian approach to cryo-EM structure determination. *J. Struct. Biol.* 180(3), 519–530.
- Kimanius, D. et al. (2021). New tools for automated cryo-EM single-particle analysis in RELION-4.0. *Biochem. J.* 478(24), 4169–4185.
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [RELION Documentation](https://relion.readthedocs.io/)
- Steinberger, P. (2025). *OpenClaw: An open-source autonomous AI agent* (Version 2026.x.x) [Computer software]. GitHub. https://github.com/openclaw/openclaw

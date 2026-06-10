# RELION MCP Server v3

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

The server exposes **23 tools** covering the complete single-particle analysis pipeline.

## Key Features

### 1. Preview Before Launch
Every pipeline tool: `confirm=False` shows all parameters (✏️ user / 📋 tutorial default / ❌ missing / ⬜ optional), `confirm=True` launches the job.

### 2. Non-Blocking Background Execution
All long-running jobs launch via detached `Popen` and return immediately with PID. Monitor with `relion_job_status` and `relion_job_logs`.

### 3. GPU Support
Class2D, InitialModel, Class3D, and Refine3D all expose `--gpu` for GPU acceleration.

### 4. Blush Regularisation
RELION 5's neural-network prior is available on Class3D and Refine3D via `use_blush=True`.

### 5. VDAM Algorithm
Class2D and InitialModel support the VDAM gradient algorithm via `use_vdam=True`, with MPI=1 validation.

### 6. Live Flag Discovery
`relion_help` runs `relion_* --help` in real time with keyword filtering.

## Architecture

```
AI Agent (Claude Code / OpenClaw / NemoClaw)
    │
    │  stdio or HTTP
    ▼
RELION MCP Server v3 (Python)
    │
    │  Popen (detached)        subprocess.run (short jobs)
    ▼                          ▼
RELION 5.x binaries        relion_import, relion_help
(background, non-blocking)  (synchronous, fast)
```

## Tools

### Pipeline Tools (16 tools — all with preview/confirm)

| Tool | Binary |
|------|--------|
| `relion_import` | `relion_import` |
| `relion_motioncorr` | `relion_run_motioncorr` |
| `relion_ctffind` | `relion_run_ctffind` |
| `relion_autopick` | `relion_autopick` |
| `relion_extract` | `relion_preprocess` |
| `relion_class2d` | `relion_refine` |
| `relion_initial_model` | `relion_refine --denovo_3dref` |
| `relion_class3d` | `relion_refine` |
| `relion_refine3d` | `relion_refine` |
| `relion_mask_create` | `relion_mask_create` |
| `relion_postprocess` | `relion_postprocess` |
| `relion_ctf_refine` | `relion_ctf_refine` |
| `relion_bayesian_polishing` | `relion_motion_refine` |
| `relion_blush` | `relion_python_blush` |
| `relion_local_resolution` | `relion_postprocess --locres` |
| `relion_modelangelo` | `relion_python_modelangelo` |

### Read-Only Tools (7 tools)

| Tool | Description |
|------|-------------|
| `relion_project_info` | Project overview |
| `relion_read_star` | Parse STAR files |
| `relion_job_status` | Job status + PID detection + stderr tail |
| `relion_job_logs` | Read stdout/stderr from background jobs |
| `relion_suggest_next_step` | Recommend next step (14-step pipeline) |
| `relion_run_command` | Run any `relion_*` binary (escape hatch) |
| `relion_help` | Parse `--help` output from any RELION binary |

## Tutorial Defaults (EMPIAR-10204)

All defaults match the RELION 5 beta-galactosidase tutorial:

| Step | Key defaults |
|------|-------------|
| Import | 200 kV, 0.885 Å, Cs 1.4, Q0 0.1 |
| MotionCorr | dose 1.277, patches 5×5, bfactor 150, float16, save_ps |
| CTF | Box 512, 30-5 Å, dF 5000-50000, dAst 100, **use_given_ps=True** |
| AutoPick | LoG, 150-180 Å, **upper_threshold=5**, maxres=20 |
| Extract | box 256 → 64, invert, bg_radius 200 |
| Class2D | K=50, T=2, mask 200, CTF, center |
| InitialModel | VDAM 100 mini-batches, T=4, C1 + apply_sym_later |
| Class3D | K=4, T=4, C1, ini_high 50, healpix 2 |
| Refine3D | D2, ini_high 50, MPI=3 (odd≥3), pool 30 |
| Mask | lowpass 15, **threshold 0.01, extend 3, soft_edge 8** |
| PostProcess | auto B-factor, autob_lowres 10 |
| CTF Refine | All flags off by default (multi-pass workflow) |
| Polishing | Train/Polish modes, sigma vel/div/acc, float16 |

## Prerequisites

- **RELION 5.x** compiled and in `PATH`
- **Python ≥ 3.10**
- **MCP Python SDK** and dependencies (see `requirements.txt`):
  ```bash
  pip install -r requirements.txt
  ```
  - mcp >= 1.0.0
  - pydantic >= 2.0.0
  - uvicorn >= 0.30.0 (for HTTP mode)

## Installation

```bash
git clone https://github.com/kdursunnizam-art/relion-mcp-server.git
cd relion-mcp-server
pip install -r requirements.txt
```

Optionally, use a virtual environment (recommended for HTTP mode):

```bash
cd relion-mcp-server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### With Claude Code (recommended for local use)

#### stdio (local)
From your terminal:
```bash
claude mcp add-json relion '{"command":"python3","args":["/path/to/relion-mcp-server/relion_mcp.py"],"env":{"RELION_PROJECT_DIR":"/path/to/data/relion_tutorial"}}' --scope user
```
Verify:
```bash
claude mcp list
```
Remove / reconfigure:
```bash
claude mcp remove relion
```
Note: `--scope user` makes the server available in all your projects.

#### HTTP (remote) — EXPERIMENTAL

1. Start the server manually in a terminal:
```bash
cd /path/to/relion-mcp-server
source venv/bin/activate
export RELION_PROJECT_DIR=/path/to/data/relion_tutorial
python relion_mcp.py --transport http --port 8000 --host 0.0.0.0
```
Keep this terminal open.

2. Register the running server with Claude Code:
```bash
claude mcp add --transport http relion http://YOUR.IP.ADDRESS:8000/mcp --scope user
```

3. Verify:
```bash
claude mcp list
```
It should show `relion` with the HTTP transport and URL `http://YOUR.IP.ADDRESS:8000/mcp`.

Then in Claude Code:
```
> Use relion_project_info to show the project status
> Import movies from Movies/*.tiff with pixel size 0.885, 200 kV, Cs 1.4
> Run motion correction with dose 1.277 e-/Å²/frame and gain ref Movies/gain.mrc
> Show me the Class2D parameters before running (agent calls with confirm=False)
> Change threads to 8 and launch (agent calls with confirm=True)
```

### With Claude Desktop

Claude Desktop only supports stdio servers via manual config. Edit `claude_desktop_config.json`:
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "relion": {
      "command": "python3",
      "args": ["/path/to/relion-mcp-server/relion_mcp.py"],
      "env": {
        "RELION_PROJECT_DIR": "/path/to/data/projet_relion",
        "RELION_THREADS": "4",
        "RELION_MPI": "1"
      }
    }
  }
}
```

On Windows with WSL2, set `"command": "wsl"` and prepend `python3` to `args`:
```json
{
  "mcpServers": {
    "relion": {
      "command": "wsl",
      "args": ["python3", "/home/you/relion-mcp-server/relion_mcp.py"],
      "env": { "RELION_PROJECT_DIR": "/home/you/relion_tutorial" }
    }
  }
}
```
Restart Claude Desktop after editing the config.

### With OpenClaw / NemoClaw

#### stdio (local)
```bash
openclaw mcp add --transport stdio --scope user relion --cmd python3 --args "/path/to/relion-mcp-server/relion_mcp.py" --env RELION_PROJECT_DIR="/path/to/data/relion_tutorial"
```
Verify:
```bash
openclaw mcp list
```

#### HTTP (remote)
Start the server:
```bash
cd /path/to/relion-mcp-server
source venv/bin/activate
export RELION_PROJECT_DIR=/data/my_project
python relion_mcp.py --transport http --port 8000 --host 0.0.0.0
```
Register:
```bash
openclaw mcp add --transport http --scope user relion http://YOUR.IP.ADDRESS:8000/mcp
```

Or configure `openclaw.json` manually (both stdio and HTTP):
```json
{
  "skills": {
    "install": { "nodeManager": "npm" },
    "entries": {
      "mcp-integration": {
        "enabled": true,
        "config": {
          "servers": [
            {
              "name": "relion-stdio",
              "transport": "stdio",
              "command": "python3",
              "args": ["/path/to/relion-mcp-server/relion_mcp.py"],
              "env": { "RELION_PROJECT_DIR": "/path/to/data/projet_relion" }
            },
            {
              "name": "relion-http",
              "transport": "streamable-http",
              "url": "http://YOUR.IP.ADDRESS:8000/mcp"
            }
          ],
          "toolPrefix": true
        }
      }
    }
  }
}
```

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `RELION_PROJECT_DIR` | RELION project directory | Current directory |
| `RELION_BIN` | Path prefix for RELION binaries | (uses PATH) |
| `RELION_THREADS` | Default thread count | 4 |
| `RELION_MPI` | Default MPI processes | 1 |

| CLI Flag | Description | Default |
|----------|-------------|---------|
| `--transport` | `stdio` or `http` | `stdio` |
| `--port` | HTTP port | 8000 |
| `--host` | HTTP host (use `0.0.0.0` for remote access) | 127.0.0.1 |
| `--project-dir` | Override `RELION_PROJECT_DIR` | (env or cwd) |

## Security

- Only `relion_*` executables can be run (validated)
- No shell injection: subprocess calls do not use `shell=True`
- File paths resolved relative to the project directory
- In HTTP mode, the server binds to `127.0.0.1` by default. For remote access, set `--host 0.0.0.0` (or your specific IP). Be aware this exposes the server on your network — use only on trusted networks.
- Preview/confirm prevents accidental job launches

## MCP SDK Compatibility

Designed for **MCP SDK 1.26+** with these constraints:
- No `lifespan` (causes crash with MCP SDK 1.26)
- No `ctx.report_progress` (causes crash with MCP SDK 1.26)
- All tool functions are `async` without a `Context` parameter
- Passes `python3 -m py_compile` cleanly

## Changelog

### v3 (current)
- 68 missing params added, 11 defaults fixed, 3 MPI validations
- **GPU support** (`--gpu`) on Class2D, InitialModel, Class3D, Refine3D
- **Blush** on Class3D, Refine3D
- **VDAM** on Class2D, InitialModel (with MPI=1 validation)
- **Polishing fully rewritten**: train/polish modes, sigma params, opt_params
- **Compute params** factored: `--pool`, `--preread_images`, `--scratch_dir`, `--skip_padding`
- **2 new tools**: `relion_local_resolution`, `relion_modelangelo`
- **CTF Refine** fixed: +beamtilt, +fit_phase, +minres, defaults all False
- **Mask Create** defaults fixed to match tutorial
- **HTTP host/port** now correctly applied from CLI flags
- 23 tools total

### v2.1
- **Background execution**: long-running jobs launch via `Popen(start_new_session=True)` and return immediately with PID. No more agent blocking.
- **`relion_job_logs`**: read stdout/stderr from background jobs in real time
- **`relion_job_status` enhanced**: PID liveness detection, stderr tail on failure, RUNNING vs IDLE
- **`relion_help`**: run `relion_* --help` and parse all flags live, with keyword filtering
- Wrapper script (`run.sh`) in each job_dir auto-creates SUCCESS/FAILURE markers
- 21 tools total

### v2.0
- **Preview/confirm system** on all pipeline tools
- **5 new tools**: `relion_initial_model`, `relion_mask_create`, `relion_ctf_refine`, `relion_bayesian_polishing`, `relion_help`
- **Parameters added**: bfactor, gain_rot/flip, float16, save_ps, d_ast, phase shift, invert_contrast, white/black dust, --ctf flag, center_classes, healpix_order, skip_gridding, ref_correct_greyscale, MPI validation, autob_lowres/highres, mtf_angpix, skip_fsc_weighting
- **Tutorial defaults** from EMPIAR-10204 baked in
- 20 tools total

### v1.0
- Initial release with 15 tools
- Verified against RELION 5.0.1

## Tested With

- RELION 5.0.1 (commit cad71bf)
- Ubuntu 24.04 LTS (WSL2)
- Python 3.12, MCP SDK 1.26.0
- Claude Code 2.1.89
- OpenClaw 2026.4.2 (commit d74a122)
- Tutorial dataset: beta-galactosidase (EMPIAR-10204)

## License

MIT — RELION itself is GPLv2. This server interacts with RELION solely through its CLI.

## References

- Scheres, S.H.W. (2012). RELION: Implementation of a Bayesian approach to cryo-EM structure determination. *J. Struct. Biol.* 180(3), 519–530.
- Kimanius, D. et al. (2021). New tools for automated cryo-EM single-particle analysis in RELION-4.0. *Biochem. J.* 478(24), 4169–4185.
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [RELION Documentation](https://relion.readthedocs.io/)
- Steinberger, P. (2025). *OpenClaw: An open-source autonomous AI agent* (Version 2026.x.x) [Computer software]. GitHub. https://github.com/openclaw/openclaw

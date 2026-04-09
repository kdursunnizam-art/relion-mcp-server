# RELION MCP Server v2.1

An MCP (Model Context Protocol) server that lets AI agents drive [RELION 5.x](https://github.com/3dem/relion) the gold-standard software for cryo-EM structure determination.

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
RELION MCP Server v2.1 (Python)
    │
    │  Popen (detached)        subprocess.run (short jobs)
    ▼                          ▼
RELION 5.x binaries        relion_import, relion_help
(background, non-blocking)  (synchronous, fast)
```

## Tools

### Pipeline Tools (16 tools — all with preview/confirm)

| Tool | Binary | Key audit additions |
|------|--------|-------------------|
| `relion_import` | `relion_import` | +mtf, +beamtilt_x/y |
| `relion_motioncorr` | `relion_run_motioncorr` | +preexposure, +eer_grouping, +defect_file |
| `relion_ctffind` | `relion_run_ctffind` | +use_given_ps=True, +ctfWin, +use_noDW |
| `relion_autopick` | `relion_autopick` | +LoG_invert, +LoG_maxres, fix upper_threshold=5 |
| `relion_extract` | `relion_preprocess` | +reextract_data_star, +fom_threshold |
| `relion_class2d` | `relion_refine` | +gpu, +VDAM, +pool, +preread, +strict_highres |
| `relion_initial_model` | `relion_refine --denovo_3dref` | +gpu, +grad_write_iter, +tau, +apply_sym_later |
| `relion_class3d` | `relion_refine` | +gpu, +blush, +fast_subsets, +skip_padding, +pool |
| `relion_refine3d` | `relion_refine` | +gpu, +blush, +ctf, +solvent_fsc, +skip_padding |
| `relion_mask_create` | `relion_mask_create` | fix threshold=0.01, extend=3, soft_edge=8, +threads |
| `relion_postprocess` | `relion_postprocess` | ✅ Already complete |
| `relion_ctf_refine` | `relion_ctf_refine` | +beamtilt, +fit_phase, +minres, fix defaults |
| `relion_bayesian_polishing` | `relion_motion_refine` | Full rewrite: train/polish modes, sigma params |
| `relion_blush` | `relion_python_blush` | Unchanged |
| `relion_local_resolution` | `relion_postprocess --locres` | **NEW** |
| `relion_modelangelo` | `relion_python_modelangelo` | **NEW** |

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


## Changelog

### v3.0 (current)
- 68 missing params added, 11 defaults fixed, 3 MPI validations
- **GPU support** (`--gpu`) on Class2D, InitialModel, Class3D, Refine3D
- **Blush** on Class3D, Refine3D
- **VDAM** on Class2D, InitialModel (with MPI=1 validation)
- **Polishing fully rewritten**: train/polish modes, sigma params, opt_params
- **Compute params** factored: `--pool`, `--preread_images`, `--scratch_dir`, `--skip_padding`
- **2 new tools**: `relion_local_resolution`, `relion_modelangelo`
- **CTF Refine** fixed: +beamtilt, +fit_phase, +minres, defaults all False
- **Mask Create** defaults fixed to match tutorial
- 23 tools total

### v2.1
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
- 20 tools total

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

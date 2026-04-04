# RELION MCP Server

An MCP (Model Context Protocol) server that lets AI agents drive [RELION 5.x](https://github.com/3dem/relion) — the gold-standard software for cryo-EM structure determination.

> **Tested and verified** against RELION 5.0.1 on Ubuntu 24.04 (WSL2). All CLI flags validated against actual `--help` output.

## What It Does

An AI agent (Claude Code, OpenClaw, NemoClaw, etc.) can process cryo-EM data through natural language:

```
You: "Import the movies from Movies/*.tiff, 200 kV, pixel size 0.885 Å, then run motion correction"
Agent: → relion_import(...) → relion_motioncorr(...) → Done!
```

The server exposes **15 tools** covering the complete single-particle analysis pipeline.

## Architecture

```
AI Agent (Claude Code / OpenClaw / NemoClaw)
    │
    │  stdio or HTTP
    ▼
RELION MCP Server (Python)
    │
    │  subprocess calls
    ▼
RELION 5.x binaries (relion_refine, relion_autopick, etc.)
```

## Tools

### Pipeline (10 tools)
| Tool | RELION Binary | Description |
|------|--------------|-------------|
| `relion_import` | `relion_import` | Import movies/micrographs |
| `relion_motioncorr` | `relion_run_motioncorr` | Beam-induced motion correction |
| `relion_ctffind` | `relion_run_ctffind` | CTF estimation (via CTFFIND4) |
| `relion_autopick` | `relion_autopick` | Particle picking (LoG or template) |
| `relion_extract` | `relion_preprocess` | Particle extraction |
| `relion_class2d` | `relion_refine` | 2D classification |
| `relion_class3d` | `relion_refine` | 3D classification |
| `relion_refine3d` | `relion_refine` | 3D auto-refinement |
| `relion_postprocess` | `relion_postprocess` | Map sharpening & resolution |
| `relion_blush` | `relion_python_blush` | AI map denoising (RELION 5) |

### Monitoring (3 tools)
| Tool | Description |
|------|-------------|
| `relion_project_info` | Project overview: jobs, counts, status |
| `relion_read_star` | Parse STAR files (RELION's metadata format) |
| `relion_job_status` | Check if a job is running/completed/failed |

### Advanced (2 tools)
| Tool | Description |
|------|-------------|
| `relion_run_command` | Run any `relion_*` binary with custom args |
| `relion_suggest_next_step` | Recommend next pipeline step |

## Prerequisites

- **RELION 5.x** compiled and in `PATH`
- **Python ≥ 3.10**
- **MCP Python SDK**: `pip install mcp pydantic`
- mcp>=1.0.0
- pydantic>=2.0.0
- uvicorn>=0.30.0

## Installation

```bash
git clone https://github.com/kdursunnizam-art/relion-mcp-server.git
cd ~/relion-mcp-server
```

## Usage

### With Claude Code (recommended for local use)

Add to `~/.claude.json` under `"mcpServers"`:

```json
{
  "mcpServers": {
    "relion": {
      "command": "python3",
      "args": ["/path/to/relion-mcp-server/relion_mcp.py"],
      "env": {
        "RELION_PROJECT_DIR": "/path/to/your/relion/project"
      }
    }
  }
}
```

Then in Claude Code:
```
> Use relion_project_info to show the project status
> Import movies from Movies/*.tiff with pixel size 0.885, 200 kV, Cs 1.4
> Run motion correction with dose 1.0 e-/A²/frame and gain ref Movies/gain.mrc
> etc...
```

### With OpenClaw / NemoClaw (remote access)

Start the server in HTTP mode:

```bash
cd /path/to/relion_mcp.py/
source venv/bin/activate
export RELION_PROJECT_DIR=/data/my_project
python relion_mcp.py --transport http --port 8000 --host 0.0.0.0
```

Configure openclaw.json

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
            "url": "http:////your.IP.adress:8000" (ex : "http://127.0.0.1:8000")
          }
        ],
        "toolPrefix": true
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

## Known Limitations

- **No GPU support in tools**: GPU device selection is not exposed (RELION uses `--gpu` flag). Use `relion_run_command` with `--gpu 0` for GPU jobs.
- **Long-running jobs**: Classification and refinement can take hours/days on CPU. The server waits for completion (timeout up to 7 days).

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

# akk

A CLI for managing ML training workflows across Kaggle, Google Colab, and local environments. Built for the [Akkadian cuneiform translation competition](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation) but useful for any ML project.

## Features

- **Kaggle Integration**: Upload notebooks, track kernel versions, monitor execution, download outputs
- **Colab/GCS Integration**: Upload notebooks to GCS, monitor training, download artifacts
- **MLflow Tracking**: Local and GCS-backed experiment tracking with full UI support
- **Preflight Validation**: Catch OOM, disk space, and API deprecation issues before deployment
- **Notebook Generation**: Deterministic notebook builds from TOML configuration
- **MCP Server**: LLM agent integration via Model Context Protocol

## Installation

### Homebrew (macOS/Linux)

```bash
brew install manwithacat/tap/akk
```

### From Source

Requires [Bun](https://bun.sh) runtime.

```bash
git clone https://github.com/manwithacat/akkadian.git
cd akkadian/tools/akk-cli
bun install
bun run build
./dist/akk --help
```

## Quick Start

```bash
# Check environment and dependencies
akk doctor

# Upload notebook to Kaggle with auto-versioning
akk kaggle upload-notebook train.py

# Monitor kernel execution
akk kaggle status manwithacat/my-kernel
akk kaggle logs manwithacat/my-kernel

# Validate notebook before deployment
akk preflight check train.py --platform kaggle-p100

# Generate notebook from TOML config
akk notebook build training.toml

# Start MCP server for LLM agents
akk mcp serve
```

## Commands

### Competition Management

```bash
akk competition init              # Initialize competition directory
akk competition status            # Show competition status and scores
```

### Preflight Checks

```bash
akk preflight check <path>        # Check notebook against platform limits
akk preflight validate <path>     # Full validation suite
akk preflight platforms           # List available platform profiles
```

### Notebook Generation

```bash
akk notebook build <config.toml>  # Generate notebook from TOML config
```

### Data Management

```bash
akk data download                 # Download competition data
akk data list                     # List registered datasets
akk data register <path>          # Register dataset with lineage
akk data explore                  # Launch Datasette
```

### Kaggle Integration

```bash
akk kaggle upload-notebook <path> # Upload notebook to Kaggle kernels
akk kaggle upload-model <path>    # Upload model to Kaggle Models
akk kaggle run-kernel <slug>      # Run and monitor a kernel
akk kaggle download-output <slug> # Download kernel outputs
```

### Colab / GCS Integration

```bash
akk colab configure               # Set up GCS bucket and auth
akk colab upload-notebook <path>  # Upload notebook to GCS
akk colab status --run <name>     # Check training status
akk colab download-artifacts      # Download training artifacts
```

### MLFlow Integration

```bash
akk mlflow start                  # Start MLFlow tracking server
akk mlflow log                    # Log experiment metrics
akk mlflow sync                   # Sync runs from GCS
akk mlflow register               # Register model in registry
```

### Local Development

```bash
akk local evaluate                # Run k-fold evaluation
akk local infer                   # Run inference on translation model
akk local analyze                 # Analyze training results
```

### Vertex AI

```bash
akk vertex submit                 # Submit training job
akk vertex status                 # Check job status
akk vertex list                   # List jobs
```

## Configuration

### Project Configuration (akk.toml)

```toml
[project]
name = "my-project"
version = "0.1.0"

[kaggle]
username = "myuser"
competition = "my-competition"

[colab]
gcs_bucket = "my-bucket"
project = "my-gcp-project"

[mlflow]
tracking_uri = "sqlite:///mlflow/mlflow.db"
port = 5001

[paths]
notebooks = "notebooks"
models = "models"
```

### Competition Configuration (competition.toml)

```toml
[competition]
name = "My Competition"
slug = "my-competition"
platform = "kaggle"
metric = "bleu"
metric_direction = "maximize"

[competition.kaggle]
username = "myuser"
data_sources = ["my-competition"]

[active_model]
name = "model-v1"
path = "models/model-v1"
best_score = 23.45

[training]
default_platform = "kaggle-p100"
default_epochs = 10
default_batch_size = 2
```

## Platform Profiles

| Platform | GPU | VRAM | Disk | Max Hours | Notes |
|----------|-----|------|------|-----------|-------|
| kaggle-p100 | Tesla P100 | 16GB | 10GB | 9h | Competition kernels |
| kaggle-t4x2 | Tesla T4 x2 | 30GB | 10GB | 9h | Larger models |
| colab-t4 | Tesla T4 | 15GB | 100GB | 12h | Development |
| colab-a100 | A100-40GB | 40GB | 200GB | 24h | Large-scale training |
| vertex-a100 | A100-40GB | 40GB | 400GB | 168h | Production training |

> Note: Kaggle's 10GB working disk is shared with HF cache. Use `save_total_limit=1` and clear cache after model load.

## Development

```bash
# Run in development mode
bun run src/index.ts <command>

# Build binary
bun run build

# Type check
bun run typecheck
```

## Architecture

```
src/
├── index.ts              # Entry point
├── cli.ts                # Command router
├── types/
│   ├── commands.ts       # Command types
│   ├── competition.ts    # Competition config types
│   ├── platform.ts       # Platform profiles
│   └── template.ts       # Template system types
├── lib/
│   ├── config.ts         # Configuration loading
│   ├── output.ts         # Output formatting
│   ├── kaggle.ts         # Kaggle CLI wrapper
│   ├── gcs.ts            # GCS/gsutil wrapper
│   └── mlflow.ts         # MLFlow client
└── commands/
    ├── competition/      # Competition commands
    ├── preflight/        # Preflight checks
    ├── kaggle/           # Kaggle commands
    ├── colab/            # Colab/GCS commands
    ├── mlflow/           # MLFlow commands
    ├── local/            # Local commands
    ├── vertex/           # Vertex AI commands
    └── workflow/         # Workflow orchestration
```

## MCP Server

The MCP server enables LLM agents (like Claude) to interact with akk:

```json
{
  "mcpServers": {
    "akkadian": {
      "command": "akk",
      "args": ["mcp", "serve"]
    }
  }
}
```

Agents can then use:
- `akk help` - Quick command reference
- `akk help <topic>` - Detailed help on commands, workflows, patterns
- Any CLI command via the `akk` tool

## License

MIT

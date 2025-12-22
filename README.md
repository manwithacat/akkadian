# Akkadian

A CLI tool for managing ML competition workflows across Kaggle, Google Colab, Vertex AI, and RunPod.

## Features

- **Competition Management**: Initialize and manage competition directories with standardized structure
- **Multi-Platform Support**: Deploy notebooks to Kaggle, Colab, Vertex AI, or RunPod
- **Preflight Checks**: Validate notebooks against platform resource limits before deployment
- **MLFlow Integration**: Track experiments and models across training runs
- **Template System**: Generate platform-optimized notebooks from templates (coming soon)
- **MCP Server**: Expose tools and resources for LLM agent integration (coming soon)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/akkadian.git
cd akkadian

# Install dependencies
bun install

# Build the CLI
bun run build

# Add to PATH (optional)
export PATH="$PATH:$(pwd)/dist"
```

## Quick Start

```bash
# Initialize a new competition
akk competition init --slug my-competition --username myuser

# Check competition status
akk competition status

# Pre-flight check a notebook
akk preflight check notebook.py --platform kaggle-p100

# List available platforms
akk preflight platforms
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
akk preflight platforms           # List available platform profiles
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

| Platform | GPU | VRAM | Disk | Max Hours | Pricing |
|----------|-----|------|------|-----------|---------|
| kaggle-p100 | Tesla P100 | 16GB | 20GB | 9h | Quota |
| kaggle-t4x2 | Tesla T4 x2 | 15GB | 20GB | 9h | Quota |
| colab-free | Tesla T4 | 15GB | 50GB | 12h | Free |
| colab-pro | A100-40GB | 40GB | 100GB | 24h | Paid |
| vertex-a100 | A100-40GB | 40GB | 400GB | 168h | Paid |
| runpod-a100 | A100-80GB | 80GB | 400GB | Unlimited | Paid |
| runpod-3090 | RTX 3090 | 24GB | 150GB | Unlimited | Paid |

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

## Roadmap

- [x] Competition management
- [x] Preflight checks
- [x] Platform profiles
- [ ] Template system for notebook generation
- [ ] Platform migration tools
- [ ] MCP server for LLM agent integration
- [ ] Extended competition workflow (submit, history, leaderboard)

## License

MIT

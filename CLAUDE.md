# Akkadian ML CLI (`akk`)

A Bun-based CLI for managing Akkadian ML workflows, integrating Kaggle, Google Colab/GCS, and MLFlow.

## Quick Start

```bash
# Run directly with bun
bun run src/index.ts <command>

# Or build and use the binary
bun run build
./dist/akk <command>
```

## Commands

### System

```bash
akk doctor                 # Check environment and dependencies
akk version               # Show version info
```

### Kaggle

```bash
akk kaggle upload-notebook <path>    # Upload .py/.ipynb to Kaggle kernels
akk kaggle upload-model <path>       # Upload model to Kaggle Models
akk kaggle run-kernel <slug>         # Check/monitor kernel status
akk kaggle download-output <slug>    # Download kernel output files
```

### Colab / GCS

```bash
akk colab configure                   # Set up GCS bucket and auth
akk colab upload-notebook <path>      # Upload notebook to GCS for Colab
akk colab status --run <name>         # Check training status from GCS
akk colab status --run <name> --watch # Watch training progress
akk colab download-model <gs://path>  # Download model from GCS
akk colab download-artifacts --run <name>  # Download training artifacts
akk colab cleanup --run <name>        # Clean up GCS artifacts to save storage
akk colab cleanup --all --older-than 7  # Clean up old runs
```

### Workflow

```bash
akk workflow train --notebook <path>  # End-to-end training workflow
akk workflow prepare --notebook <path> # Prepare notebook for Colab (copy to Downloads)
akk workflow list-runs                 # List all runs in GCS
akk workflow list-runs --status completed  # Filter by status
akk workflow import-run --run <name>   # Import completed run into local MLFlow
akk workflow import-run --run <name> --register  # Import and register model
```

### MLFlow

```bash
akk mlflow start                  # Start MLFlow tracking server
akk mlflow start --stop           # Stop the server
akk mlflow log --metrics '{"bleu": 23.1}'  # Log experiment metrics
akk mlflow sync --list            # List remote experiments in GCS
akk mlflow sync --experiment nllb-akkadian  # Sync experiment from GCS
akk mlflow register --list        # List registered models
akk mlflow register --run nllb-v4 --name nllb-akkadian  # Register model from run
akk mlflow register --run nllb-v5 --alias champion  # Register and set alias
```

### Local Evaluation, Inference & Analysis

```bash
akk local evaluate                # Run k-fold evaluation
akk local evaluate --visualize    # Open visualization after
akk local infer --model facebook/nllb-200-distilled-600M --interactive  # Interactive translation
akk local infer --model ./models/nllb --text "šarrum dannum"  # Single translation
akk local infer --input texts.txt --output translations.txt  # Batch translation
akk local analyze --run nllb-v4   # Analyze training results
akk local analyze --run nllb-v4 --compare nllb-v3  # Compare with baseline
```

## Configuration

Create `akk.toml` in your project root:

```toml
[project]
name = "akkadian"
version = "0.1.0"

[kaggle]
username = "your-username"
competition = "babylonian-engine-efficiency-challenge"

[colab]
gcs_bucket = "your-bucket"
project = "your-gcp-project"

[mlflow]
tracking_uri = "sqlite:///mlflow/mlflow.db"
artifact_location = "./mlflow/artifacts"
port = 5001

[paths]
notebooks = "notebooks"
scripts = "scripts"
datasets = "datasets"
models = "models"
```

## Global Options

All commands support:
- `--help, -h` - Show help
- `--json` - Output as JSON
- `--verbose, -v` - Verbose output
- `--quiet, -q` - Suppress output

## Architecture

```
src/
├── index.ts          # Entry point
├── cli.ts            # Command router
├── types/            # TypeScript types
├── lib/              # Core libraries
│   ├── output.ts     # Output formatting
│   ├── config.ts     # TOML config
│   ├── kaggle.ts     # Kaggle CLI wrapper
│   ├── gcs.ts        # gsutil wrapper
│   └── mlflow.ts     # MLFlow REST client
└── commands/         # Command implementations
    ├── kaggle/
    ├── colab/
    ├── mlflow/
    └── local/
```

## Dependencies

- **Required CLI tools**: kaggle, gcloud, gsutil, mlflow, jupytext, python3
- **Run `akk doctor` to check all dependencies**

## Colab MLFlow Integration

For training in Google Colab with GCS-backed MLFlow tracking:

```python
# In Colab notebook
!pip install mlflow>=2.15.1 google-cloud-storage psutil nvidia-ml-py
!gsutil cp gs://akkadian-models/mlflow/tracking/colab_tracker.py .

from colab_tracker import ColabTracker

# Initialize with system metrics (GPU/CPU/memory logging)
tracker = ColabTracker(
    experiment_name="nllb-akkadian",
    gcs_bucket="akkadian-models",
    run_name="nllb-1.3B-v5",
    enable_system_metrics=True,
)

# Log parameters and model info
tracker.log_params({"model": "nllb-1.3B", "epochs": 3, "batch_size": 4})
tracker.log_model_info("facebook/nllb-200-1.3B", num_parameters=1300000000)

# During training
tracker.log_metrics({"loss": 0.5, "bleu": 15.2}, step=100)

# Log rich evaluation data
tracker.log_evaluation_table(predictions, references, sources)
tracker.log_bleu_breakdown({"bleu": 23.1, "bleu-1": 55.2, "bleu-2": 32.1})
tracker.log_training_curves(train_losses, eval_losses, eval_bleus)

# Log trained model with signature
tracker.log_transformers_model(model, tokenizer, task="translation")

# End run
tracker.end_run(status="FINISHED")
```

### MLFlow UI Data Model Integration

The ColabTracker fully populates all MLFlow UI fields:

```python
# Datasets used (populates "Datasets used" in UI)
tracker.log_dataset(
    df=train_df,
    name="akkadian-train",
    context="training",
    source="gs://bucket/datasets/train.csv"
)

# Tags (populates "Tags" in UI)
tracker.add_run_tags({
    "model_type": "translation",
    "language_pair": "akk-en",
    "framework": "transformers",
})

# Source (populates "Source" in UI)
tracker.set_source(
    source_type="NOTEBOOK",
    source_name="train.ipynb",
    source_url="https://colab.research.google.com/..."
)

# Registered prompts (populates "Registered prompts" in UI)
tracker.register_prompt(
    name="akkadian-translation-prompt",
    template="Translate: {{source}}\nEnglish:",
    commit_message="Initial version",
    tags={"language": "akkadian"}
)

# Logged & registered models (populates "Logged models" and "Registered models")
tracker.register_model(
    model=trained_model,
    tokenizer=tokenizer,
    registered_model_name="akkadian-nllb",
    task="translation"
)
```

### Local Inference with Tracing

Run traced inference against translation models:

```bash
# Start MLFlow server for trace collection
akk mlflow start --port 5001

# Interactive translation with tracing
akk local infer --model facebook/nllb-200-distilled-600M --interactive

# View traces at http://localhost:5001/#/experiments
```

All inference calls are traced with:
- Input/output text logged
- Inference timing
- Token counts
- Model metadata

### Syncing to Local

After Colab training, sync runs to your local MLFlow server:

```bash
# Start local MLFlow server
akk mlflow start --port 5001

# List remote experiments
akk mlflow sync --bucket akkadian-models --list

# Sync specific experiment
akk mlflow sync --experiment nllb-akkadian

# Sync with artifacts (models, checkpoints)
akk mlflow sync --experiment nllb-akkadian --run nllb-1.3B-v5 --artifacts
```

## End-to-End Training Workflow

The recommended workflow for training and iterating on models:

### 1. Prepare and Upload Notebook

```bash
# Edit your training notebook locally (with LLM support)
# Then upload to GCS for Colab execution
akk colab upload-notebook notebooks/colab/nllb_train_v4.ipynb --version v4
```

### 2. Run Training in Colab

```bash
# Use the workflow command for full automation
akk workflow train --notebook notebooks/colab/nllb_train_v4.ipynb --name nllb-v4

# Or manually:
# 1. Open notebook in Colab from GCS
# 2. Monitor progress: akk colab status --run nllb-v4 --watch
```

### 3. Download Artifacts

```bash
# Download trained model, checkpoints, metrics
akk colab download-artifacts --run nllb-v4

# Or download everything
akk colab download-artifacts --run nllb-v4 --all
```

### 4. Analyze Results

```bash
# Start MLFlow for local tracking
akk mlflow start --port 5001

# Sync from GCS
akk mlflow sync --experiment nllb-akkadian

# Analyze with recommendations
akk local analyze --run nllb-v4 --compare nllb-v3

# Test translations interactively
akk local infer --model ./artifacts/nllb-v4/model --interactive
```

### 5. Submit to Kaggle

```bash
# Upload model for competition scoring
akk kaggle upload-model ./artifacts/nllb-v4/model --name nllb-v4

# Create and submit inference kernel
akk kaggle upload-notebook notebooks/colab/nllb_inference_v4.py
akk kaggle run-kernel manwithacat/nllb-inference-v4 --wait --download
```

## GCS Cross-Account Access

For Colab with a different Google account than your GCS bucket, see `docs/gcs-colab-setup.md`.

Quick setup:
```bash
# Grant Colab user access to your bucket
gsutil iam ch user:colab-user@gmail.com:objectAdmin gs://akkadian-models
```

Or use a service account key (recommended for automation).

## Training Notebook Conventions

Training notebooks should follow these conventions for workflow integration:

```python
# Environment variables (set by akk workflow train)
GCS_BUCKET = os.environ.get("GCS_BUCKET", "akkadian-models")
RUN_NAME = os.environ.get("RUN_NAME", "nllb-v4")
EXPERIMENT = os.environ.get("EXPERIMENT", "nllb-akkadian")

# Output paths
OUTPUT_PATH = f"gs://{GCS_BUCKET}/mlflow/runs/{EXPERIMENT}/{RUN_NAME}"

# Status updates (for monitoring)
def update_status(phase, progress=None, metrics=None):
    status = {"phase": phase, "progress": progress, "metrics": metrics}
    # Write to gs://{bucket}/mlflow/runs/{experiment}/{run}/status.json
```

See `notebooks/colab/nllb_train_v4.py` for a complete example.

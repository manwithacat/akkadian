/**
 * Vertex AI Platform Adapter
 *
 * Generates Vertex AI-specific notebook code for training jobs.
 */

import type { PlatformId } from '../../types/platform'
import type { PlatformAdapter, PlatformPaths, TemplateContext } from '../../types/template'
import { getRecommendedBatchSize } from '../../types/template'

export class VertexAdapter implements PlatformAdapter {
  id: PlatformId
  name: string

  constructor(platformId: PlatformId = 'vertex-a100') {
    this.id = platformId
    this.name = platformId === 'vertex-a100' ? 'Vertex AI A100' : 'Vertex AI T4'
  }

  generateSetupCell(ctx: TemplateContext): string {
    const bucket = ctx.gcs?.bucket || 'your-bucket'
    const project = ctx.gcs?.project || 'your-project'
    const gpuType = this.id === 'vertex-a100' ? 'A100' : 'T4'

    return `# Vertex AI Environment Setup
import os
import sys

# Environment detection
IS_VERTEX = os.environ.get('CLOUD_ML_PROJECT_ID') is not None
print(f"Running on Vertex AI: {IS_VERTEX}")

# GPU detection
import subprocess
try:
    gpu_info = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
        capture_output=True, text=True
    )
    print(f"GPU: {gpu_info.stdout.strip()}")
except Exception as e:
    print(f"GPU detection failed: {e}")

# Memory optimization settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# GCS Configuration
GCS_BUCKET = os.environ.get("GCS_BUCKET", "${bucket}")
GCS_PROJECT = os.environ.get("GCP_PROJECT", "${project}")
RUN_NAME = os.environ.get("RUN_NAME", "vertex-run")

# Paths - Vertex AI uses /gcs/ mount for GCS buckets
if os.path.exists("/gcs"):
    # Using GCS FUSE mount
    INPUT_PATH = f"/gcs/{GCS_BUCKET}/datasets"
    OUTPUT_PATH = f"/gcs/{GCS_BUCKET}/runs/{RUN_NAME}/output"
    CHECKPOINT_PATH = f"/gcs/{GCS_BUCKET}/runs/{RUN_NAME}/checkpoints"
else:
    # Fall back to local paths with gsutil sync
    INPUT_PATH = "/tmp/data"
    OUTPUT_PATH = "/tmp/output"
    CHECKPOINT_PATH = "/tmp/checkpoints"
    os.makedirs(INPUT_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

MODEL_CACHE = "/root/.cache/huggingface"

print(f"GCS Bucket: {GCS_BUCKET}")
print(f"Input path: {INPUT_PATH}")
print(f"Output path: {OUTPUT_PATH}")
print(f"Platform: Vertex AI ${gpuType}")
`
      .replace(/\$\{bucket\}/g, bucket)
      .replace(/\$\{project\}/g, project)
      .replace(/\$\{gpuType\}/g, gpuType)
  }

  generateDataLoading(_ctx: TemplateContext): string {
    return `# Data Loading (Vertex AI)
import pandas as pd
import glob

# Check if using GCS FUSE or need to download
if os.path.exists(INPUT_PATH) and os.listdir(INPUT_PATH):
    print(f"Data available at {INPUT_PATH}")
else:
    print("Downloading data from GCS...")
    os.makedirs(INPUT_PATH, exist_ok=True)
    !gsutil -m cp -r gs://{GCS_BUCKET}/datasets/* {INPUT_PATH}/

# List data files
csv_files = glob.glob(f"{INPUT_PATH}/**/*.csv", recursive=True)
print(f"Found {len(csv_files)} CSV files:")
for f in csv_files:
    print(f"  - {f}")

# Load training data
train_files = [f for f in csv_files if 'train' in f.lower()]
if train_files:
    train_df = pd.read_csv(train_files[0])
    print(f"Training data: {len(train_df)} samples")
else:
    raise FileNotFoundError("No training data file found")

# Load validation data if available
val_files = [f for f in csv_files if 'val' in f.lower() or 'dev' in f.lower()]
if val_files:
    val_df = pd.read_csv(val_files[0])
    print(f"Validation data: {len(val_df)} samples")
else:
    val_df = None
`
  }

  generateCheckpointSave(_ctx: TemplateContext): string {
    return `# Checkpoint Saving (Vertex AI)
import json
import shutil

def save_checkpoint(model, tokenizer, step, metrics=None):
    """Save model checkpoint - auto-syncs via GCS FUSE."""
    checkpoint_dir = f"{CHECKPOINT_PATH}/checkpoint-{step}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    if metrics:
        with open(f"{checkpoint_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    print(f"Saved checkpoint to {checkpoint_dir}")

    # If not using FUSE, sync to GCS
    if not os.path.exists("/gcs"):
        gcs_path = f"gs://{GCS_BUCKET}/runs/{RUN_NAME}/checkpoints/checkpoint-{step}"
        !gsutil -m cp -r {checkpoint_dir}/* {gcs_path}/

def load_latest_checkpoint():
    """Load the latest checkpoint."""
    checkpoints = sorted(glob.glob(f"{CHECKPOINT_PATH}/checkpoint-*"))
    return checkpoints[-1] if checkpoints else None

def update_status(phase, progress=None, metrics=None):
    """Update training status."""
    status = {
        "phase": phase,
        "progress": progress,
        "metrics": metrics,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    status_path = f"{OUTPUT_PATH}/status.json"
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    with open(status_path, "w") as f:
        json.dump(logStep, f, indent=2)
`
  }

  generateOutputSave(_ctx: TemplateContext): string {
    return `# Output Saving (Vertex AI)
import json

def save_model_for_submission(model, tokenizer, output_name="model"):
    """Save final model - auto-syncs via GCS FUSE."""
    output_dir = f"{OUTPUT_PATH}/{output_name}"
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model saved to {output_dir}")

    # If not using FUSE, sync to GCS
    if not os.path.exists("/gcs"):
        gcs_path = f"gs://{GCS_BUCKET}/runs/{RUN_NAME}/output/{output_name}"
        !gsutil -m cp -r {output_dir}/* {gcs_path}/
        print(f"Synced to {gcs_path}")

    return output_dir

def save_predictions(predictions, filename="submission.csv"):
    """Save predictions to CSV."""
    output_path = f"{OUTPUT_PATH}/{filename}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if isinstance(predictions, pd.DataFrame):
        predictions.to_csv(output_path, index=False)
    else:
        pd.DataFrame(predictions).to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")
    return output_path

def finalize_run(status="completed"):
    """Mark run as complete."""
    update_status(logStep, progress=100)
    print(f"Vertex AI run finalized with status: {status}")
`
  }

  getPaths(ctx: TemplateContext): PlatformPaths {
    const bucket = ctx.gcs?.bucket || 'your-bucket'
    return {
      input: `/gcs/${bucket}/datasets`,
      output: `/gcs/${bucket}/output`,
      checkpoint: `/gcs/${bucket}/checkpoints`,
      model_cache: '/root/.cache/huggingface',
    }
  }

  getRecommendedBatchSize(modelName: string): number {
    return getRecommendedBatchSize(this.id, modelName)
  }

  getRecommendedGradientAccumulation(modelName: string, targetBatchSize: number): number {
    const actualBatch = this.getRecommendedBatchSize(modelName)
    return Math.max(1, Math.ceil(targetBatchSize / actualBatch))
  }
}

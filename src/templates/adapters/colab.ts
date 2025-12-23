/**
 * Google Colab Platform Adapter
 *
 * Generates Colab-specific notebook code with GCS integration.
 * Supports cross-account GCS access (Colab user != bucket owner).
 */

import type { PlatformId } from '../../types/platform'
import type { PlatformAdapter, PlatformPaths, TemplateContext } from '../../types/template'
import { getRecommendedBatchSize } from '../../types/template'

export class ColabAdapter implements PlatformAdapter {
  id: PlatformId
  name: string

  constructor(platformId: PlatformId = 'colab-free') {
    this.id = platformId
    this.name = platformId === 'colab-free' ? 'Colab Free (T4)' : 'Colab Pro (A100)'
  }

  generateSetupCell(ctx: TemplateContext): string {
    const bucket = ctx.gcs?.bucket || 'your-bucket'
    const project = ctx.gcs?.project || 'your-project'
    const crossAccount = ctx.gcs?.cross_account || false
    const gpuType = this.id === 'colab-pro' ? 'A100' : 'T4'

    let authSection = ''
    if (crossAccount) {
      authSection = `
# Cross-account GCS authentication
# (Colab user is different from GCS bucket owner)
from google.colab import auth
auth.authenticate_user()

# Verify authentication
!gcloud auth list
`
    } else {
      authSection = `
# Standard GCS authentication
from google.colab import auth
auth.authenticate_user()
`
    }

    return `# Google Colab Environment Setup
import os
import sys
import subprocess

# Environment detection
IS_COLAB = 'google.colab' in sys.modules
print(f"Running on Colab: {IS_COLAB}")

# GPU detection
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Memory optimization settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

${authSection}

# GCS Configuration
GCS_BUCKET = "${bucket}"
GCS_PROJECT = "${project}"

# Set default project
!gcloud config set project {GCS_PROJECT}

# Paths
INPUT_PATH = "/content/data"
OUTPUT_PATH = "/content/output"
CHECKPOINT_PATH = "/content/checkpoints"
MODEL_CACHE = "/root/.cache/huggingface"

# GCS paths
GCS_INPUT = f"gs://{GCS_BUCKET}/datasets"
GCS_OUTPUT = f"gs://{GCS_BUCKET}/runs/{RUN_NAME}/output" if 'RUN_NAME' in dir() else f"gs://{GCS_BUCKET}/output"
GCS_CHECKPOINTS = f"gs://{GCS_BUCKET}/runs/{RUN_NAME}/checkpoints" if 'RUN_NAME' in dir() else f"gs://{GCS_BUCKET}/checkpoints"

# Create local directories
os.makedirs(INPUT_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

print(f"GCS Bucket: {GCS_BUCKET}")
print(f"Local input: {INPUT_PATH}")
print(f"Local output: {OUTPUT_PATH}")
print(f"Platform: ${gpuType}")
`
      .replace(/\$\{bucket\}/g, bucket)
      .replace(/\$\{project\}/g, project)
      .replace(/\$\{gpuType\}/g, gpuType)
  }

  generateDataLoading(ctx: TemplateContext): string {
    const bucket = ctx.gcs?.bucket || 'your-bucket'

    return `# Data Loading (Colab with GCS)
import pandas as pd

# Download data from GCS
print("Downloading data from GCS...")
!gsutil -m cp -r {GCS_INPUT}/* {INPUT_PATH}/

# List downloaded files
import glob
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

# Load validation/test data if available
val_files = [f for f in csv_files if 'val' in f.lower() or 'dev' in f.lower()]
test_files = [f for f in csv_files if 'test' in f.lower()]

if val_files:
    val_df = pd.read_csv(val_files[0])
    print(f"Validation data: {len(val_df)} samples")
else:
    val_df = None
    print("No validation file found, will split from training")

if test_files:
    test_df = pd.read_csv(test_files[0])
    print(f"Test data: {len(test_df)} samples")
`
  }

  generateCheckpointSave(ctx: TemplateContext): string {
    return `# Checkpoint Saving (Colab with GCS)
import shutil
import json

def save_checkpoint(model, tokenizer, step, metrics=None, upload_to_gcs=True):
    """Save model checkpoint locally and optionally to GCS."""
    checkpoint_dir = f"{CHECKPOINT_PATH}/checkpoint-{step}"

    # Save model and tokenizer locally
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    # Save metrics if provided
    if metrics:
        with open(f"{checkpoint_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    print(f"Saved checkpoint to {checkpoint_dir}")

    # Upload to GCS
    if upload_to_gcs:
        gcs_checkpoint = f"{GCS_CHECKPOINTS}/checkpoint-{step}"
        !gsutil -m cp -r {checkpoint_dir}/* {gcs_checkpoint}/
        print(f"Uploaded to {gcs_checkpoint}")

    # Clean up old local checkpoints (keep only last 2)
    checkpoints = sorted(glob.glob(f"{CHECKPOINT_PATH}/checkpoint-*"))
    for old_ckpt in checkpoints[:-2]:
        shutil.rmtree(old_ckpt)
        print(f"Removed old checkpoint: {old_ckpt}")

def load_latest_checkpoint(from_gcs=True):
    """Load the latest checkpoint, optionally from GCS."""
    if from_gcs:
        # List GCS checkpoints
        result = subprocess.run(
            ["gsutil", "ls", GCS_CHECKPOINTS],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            gcs_checkpoints = sorted(result.stdout.strip().split('\\n'))
            if gcs_checkpoints:
                latest = gcs_checkpoints[-1].rstrip('/')
                local_path = f"{CHECKPOINT_PATH}/{os.path.basename(latest)}"
                !gsutil -m cp -r {latest}/* {local_path}/
                return local_path

    # Fall back to local checkpoints
    checkpoints = sorted(glob.glob(f"{CHECKPOINT_PATH}/checkpoint-*"))
    return checkpoints[-1] if checkpoints else None

def update_status(phase, progress=None, metrics=None):
    """Update training status in GCS for monitoring."""
    status = {
        "phase": phase,
        "progress": progress,
        "metrics": metrics,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    status_path = f"{OUTPUT_PATH}/status.json"
    with open(status_path, "w") as f:
        json.dump(logStep, f, indent=2)
    !gsutil cp {status_path} {GCS_OUTPUT}/status.json
`
  }

  generateOutputSave(ctx: TemplateContext): string {
    return `# Output Saving (Colab with GCS)
import json

def save_model_for_submission(model, tokenizer, output_name="model", upload_to_gcs=True):
    """Save final model locally and to GCS."""
    output_dir = f"{OUTPUT_PATH}/{output_name}"

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model saved to {output_dir}")
    print("Files saved:")
    for f in os.listdir(output_dir):
        size_mb = os.path.getsize(f"{output_dir}/{f}") / (1024 * 1024)
        print(f"  - {f}: {size_mb:.1f} MB")

    if upload_to_gcs:
        gcs_model = f"{GCS_OUTPUT}/{output_name}"
        !gsutil -m cp -r {output_dir}/* {gcs_model}/
        print(f"Uploaded to {gcs_model}")

    return output_dir

def save_predictions(predictions, filename="submission.csv", upload_to_gcs=True):
    """Save predictions to CSV."""
    output_path = f"{OUTPUT_PATH}/{filename}"

    if isinstance(predictions, pd.DataFrame):
        predictions.to_csv(output_path, index=False)
    else:
        pd.DataFrame(predictions).to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")

    if upload_to_gcs:
        !gsutil cp {output_path} {GCS_OUTPUT}/{filename}
        print(f"Uploaded to {GCS_OUTPUT}/{filename}")

    return output_path

def save_training_log(history, filename="training_log.json", upload_to_gcs=True):
    """Save training history/metrics."""
    output_path = f"{OUTPUT_PATH}/{filename}"

    with open(output_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training log saved to {output_path}")

    if upload_to_gcs:
        !gsutil cp {output_path} {GCS_OUTPUT}/{filename}

def finalize_run(status="completed"):
    """Mark run as complete and upload final status."""
    update_status(logStep, progress=100)
    print(f"Run finalized with status: {status}")
`
  }

  getPaths(ctx: TemplateContext): PlatformPaths {
    return {
      input: '/content/data',
      output: '/content/output',
      checkpoint: '/content/checkpoints',
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

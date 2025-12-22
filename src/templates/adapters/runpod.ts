/**
 * RunPod Platform Adapter
 *
 * Generates RunPod-specific notebook code for serverless GPU training.
 */

import type { PlatformAdapter, TemplateContext, PlatformPaths } from '../../types/template'
import type { PlatformId } from '../../types/platform'
import { getRecommendedBatchSize } from '../../types/template'

export class RunPodAdapter implements PlatformAdapter {
  id: PlatformId
  name: string

  constructor(platformId: PlatformId = 'runpod-a100') {
    this.id = platformId
    this.name = platformId === 'runpod-a100' ? 'RunPod A100' : 'RunPod RTX 3090'
  }

  generateSetupCell(ctx: TemplateContext): string {
    const gpuType = this.id === 'runpod-a100' ? 'A100' : 'RTX 3090'

    return `# RunPod Environment Setup
import os
import sys
import subprocess

# Environment detection
IS_RUNPOD = os.environ.get('RUNPOD_POD_ID') is not None
print(f"Running on RunPod: {IS_RUNPOD}")

# GPU detection
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

# RunPod paths - /workspace is persistent storage
INPUT_PATH = "/workspace/data"
OUTPUT_PATH = "/workspace/output"
CHECKPOINT_PATH = "/workspace/checkpoints"
MODEL_CACHE = "/workspace/.cache/huggingface"

# Set HuggingFace cache
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE

# Create directories
os.makedirs(INPUT_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
os.makedirs(MODEL_CACHE, exist_ok=True)

# Optional: S3/GCS sync configuration
S3_BUCKET = os.environ.get("S3_BUCKET", "")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "")

print(f"Input path: {INPUT_PATH}")
print(f"Output path: {OUTPUT_PATH}")
print(f"Platform: RunPod ${gpuType}")

# Check for rclone (useful for cloud sync)
try:
    subprocess.run(['rclone', 'version'], capture_output=True, check=True)
    print("rclone available for cloud sync")
except:
    print("rclone not available (optional)")
`.replace(/\$\{gpuType\}/g, gpuType)
  }

  generateDataLoading(ctx: TemplateContext): string {
    return `# Data Loading (RunPod)
import pandas as pd
import glob

# Check if data already exists in workspace
csv_files = glob.glob(f"{INPUT_PATH}/**/*.csv", recursive=True)

if not csv_files:
    print("No data found in workspace. Upload your data to /workspace/data/")
    print("You can use:")
    print("  - runpodctl send <file>")
    print("  - rclone copy remote:path /workspace/data/")
    print("  - wget/curl for direct downloads")
    raise FileNotFoundError("Please upload training data to /workspace/data/")

print(f"Found {len(csv_files)} CSV files:")
for f in csv_files:
    print(f"  - {f}")

# Load training data
train_files = [f for f in csv_files if 'train' in f.lower()]
if train_files:
    train_df = pd.read_csv(train_files[0])
    print(f"Training data: {len(train_df)} samples")
else:
    # Try to use first CSV file
    train_df = pd.read_csv(csv_files[0])
    print(f"Using {csv_files[0]}: {len(train_df)} samples")

# Load validation data if available
val_files = [f for f in csv_files if 'val' in f.lower() or 'dev' in f.lower()]
if val_files:
    val_df = pd.read_csv(val_files[0])
    print(f"Validation data: {len(val_df)} samples")
else:
    val_df = None
    print("No validation file found, will split from training")
`
  }

  generateCheckpointSave(ctx: TemplateContext): string {
    return `# Checkpoint Saving (RunPod)
import json
import shutil

def save_checkpoint(model, tokenizer, step, metrics=None, sync_to_cloud=False):
    """Save model checkpoint to workspace."""
    checkpoint_dir = f"{CHECKPOINT_PATH}/checkpoint-{step}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    if metrics:
        with open(f"{checkpoint_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    print(f"Saved checkpoint to {checkpoint_dir}")

    # Optional cloud sync
    if sync_to_cloud:
        if S3_BUCKET:
            !aws s3 sync {checkpoint_dir} s3://{S3_BUCKET}/checkpoints/checkpoint-{step}/
        elif GCS_BUCKET:
            !gsutil -m cp -r {checkpoint_dir}/* gs://{GCS_BUCKET}/checkpoints/checkpoint-{step}/

    # Clean up old checkpoints (keep only last 3)
    checkpoints = sorted(glob.glob(f"{CHECKPOINT_PATH}/checkpoint-*"))
    for old_ckpt in checkpoints[:-3]:
        shutil.rmtree(old_ckpt)
        print(f"Removed old checkpoint: {old_ckpt}")

def load_latest_checkpoint():
    """Load the latest checkpoint from workspace."""
    checkpoints = sorted(glob.glob(f"{CHECKPOINT_PATH}/checkpoint-*"))
    return checkpoints[-1] if checkpoints else None
`
  }

  generateOutputSave(ctx: TemplateContext): string {
    return `# Output Saving (RunPod)
import json

def save_model_for_submission(model, tokenizer, output_name="model", sync_to_cloud=False):
    """Save final model to workspace."""
    output_dir = f"{OUTPUT_PATH}/{output_name}"
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model saved to {output_dir}")
    print("Files saved:")
    for f in os.listdir(output_dir):
        size_mb = os.path.getsize(f"{output_dir}/{f}") / (1024 * 1024)
        print(f"  - {f}: {size_mb:.1f} MB")

    # Optional cloud sync
    if sync_to_cloud:
        if S3_BUCKET:
            !aws s3 sync {output_dir} s3://{S3_BUCKET}/models/{output_name}/
            print(f"Synced to s3://{S3_BUCKET}/models/{output_name}/")
        elif GCS_BUCKET:
            !gsutil -m cp -r {output_dir}/* gs://{GCS_BUCKET}/models/{output_name}/
            print(f"Synced to gs://{GCS_BUCKET}/models/{output_name}/")

    return output_dir

def save_predictions(predictions, filename="submission.csv"):
    """Save predictions to CSV."""
    output_path = f"{OUTPUT_PATH}/{filename}"

    if isinstance(predictions, pd.DataFrame):
        predictions.to_csv(output_path, index=False)
    else:
        pd.DataFrame(predictions).to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")
    return output_path

def save_training_log(history, filename="training_log.json"):
    """Save training history/metrics."""
    output_path = f"{OUTPUT_PATH}/{filename}"

    with open(output_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training log saved to {output_path}")

def download_from_runpod(local_path, remote_name="output"):
    """Instructions for downloading from RunPod."""
    print(f"\\nTo download {local_path} from RunPod:")
    print(f"  runpodctl receive {remote_name}")
    print("\\nOr use rclone/rsync for larger transfers")
`
  }

  getPaths(ctx: TemplateContext): PlatformPaths {
    return {
      input: '/workspace/data',
      output: '/workspace/output',
      checkpoint: '/workspace/checkpoints',
      model_cache: '/workspace/.cache/huggingface',
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

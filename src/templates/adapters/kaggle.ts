/**
 * Kaggle Platform Adapter
 *
 * Generates Kaggle-specific notebook code for training and inference.
 */

import type { MLFramework } from '../../types/commands'
import type { PlatformId } from '../../types/platform'
import type { PlatformAdapter, PlatformPaths, TemplateContext } from '../../types/template'
import { getRecommendedBatchSize } from '../../types/template'

export class KaggleAdapter implements PlatformAdapter {
  id: PlatformId
  name: string

  constructor(platformId: PlatformId = 'kaggle-p100') {
    this.id = platformId
    this.name =
      platformId === 'kaggle-p100' ? 'Kaggle P100' : platformId === 'kaggle-t4x2' ? 'Kaggle T4 x2' : 'Kaggle CPU'
  }

  /**
   * Generate framework-specific environment variables
   */
  private generateFrameworkEnvVars(framework: MLFramework = 'pytorch'): string {
    const common = `
# Suppress Python warnings from common noisy sources
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*MessageFactory.*")  # Protobuf
warnings.filterwarnings("ignore", message=".*SymbolDatabase.*")  # Protobuf

# Suppress transformers and tokenizers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Disable tqdm progress bars on Kaggle (each update creates a log line)
os.environ["TQDM_DISABLE"] = "1"
`

    if (framework === 'pytorch') {
      return `
# ==== PyTorch-only: Prevent TF/Flax backends from loading ====
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

# Suppress TensorFlow/XLA C++ logging (cuFFT, cuDNN, cuBLAS registration messages)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Suppress PyTorch-specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.*")

# PyTorch memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
${common}`
    }

    if (framework === 'tensorflow') {
      return `
# ==== TensorFlow-only: Prevent PyTorch/Flax backends from loading ====
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_FLAX"] = "0"

# TensorFlow logging configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
${common}`
    }

    if (framework === 'jax') {
      return `
# ==== JAX/Flax: Prevent TF/PyTorch backends from loading ====
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

# JAX configuration
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
${common}`
    }

    return common
  }

  generateSetupCell(ctx: TemplateContext): string {
    const competition = ctx.competition?.competition.slug || 'competition-name'
    const gpuType = this.id === 'kaggle-p100' ? 'P100' : this.id === 'kaggle-t4x2' ? 'T4' : 'CPU'
    // Get framework from context or default to pytorch
    const framework: MLFramework = (ctx as { framework?: MLFramework }).framework || 'pytorch'

    return `# Kaggle Environment Setup
import os
import sys
import subprocess
import warnings
${this.generateFrameworkEnvVars(framework)}
# Environment detection
IS_KAGGLE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None
print(f"Running on Kaggle: {IS_KAGGLE}")

# GPU detection
if IS_KAGGLE:
    try:
        gpu_info = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        print(f"GPU: {gpu_info.stdout.strip()}")
    except Exception as e:
        print(f"No GPU detected or nvidia-smi failed: {e}")

# Paths
INPUT_PATH = "/kaggle/input/${competition}"
OUTPUT_PATH = "/kaggle/working"
CHECKPOINT_PATH = "/kaggle/working/checkpoints"
MODEL_CACHE = "/kaggle/working/.cache/huggingface"

# Set HuggingFace cache to working directory (persists across runs)
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE

# Create directories
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
os.makedirs(MODEL_CACHE, exist_ok=True)

print(f"Input path: {INPUT_PATH}")
print(f"Output path: {OUTPUT_PATH}")
print(f"Platform: ${gpuType}")
`
      .replace(/\$\{competition\}/g, competition)
      .replace(/\$\{gpuType\}/g, gpuType)
  }

  generateDataLoading(ctx: TemplateContext): string {
    const _competition = ctx.competition?.competition.slug || 'competition-name'

    return `# Data Loading (Kaggle)
import glob
import pandas as pd

# Find CSV files in competition input
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

  generateCheckpointSave(_ctx: TemplateContext): string {
    return `# Checkpoint Saving (Kaggle)
import shutil

def save_checkpoint(model, tokenizer, step, metrics=None):
    """Save model checkpoint to working directory."""
    checkpoint_dir = f"{CHECKPOINT_PATH}/checkpoint-{step}"

    # Save model and tokenizer
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    # Save metrics if provided
    if metrics:
        import json
        with open(f"{checkpoint_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    print(f"Saved checkpoint to {checkpoint_dir}")

    # Clean up old checkpoints (keep only last 2)
    checkpoints = sorted(glob.glob(f"{CHECKPOINT_PATH}/checkpoint-*"))
    for old_ckpt in checkpoints[:-2]:
        shutil.rmtree(old_ckpt)
        print(f"Removed old checkpoint: {old_ckpt}")

def load_latest_checkpoint():
    """Load the latest checkpoint if available."""
    checkpoints = sorted(glob.glob(f"{CHECKPOINT_PATH}/checkpoint-*"))
    if checkpoints:
        return checkpoints[-1]
    return None
`
  }

  generateOutputSave(_ctx: TemplateContext): string {
    return `# Output Saving (Kaggle)
import json

def save_model_for_submission(model, tokenizer, output_name="model"):
    """Save final model for Kaggle submission."""
    output_dir = f"{OUTPUT_PATH}/{output_name}"

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model saved to {output_dir}")
    print("Files saved:")
    for f in os.listdir(output_dir):
        size_mb = os.path.getsize(f"{output_dir}/{f}") / (1024 * 1024)
        print(f"  - {f}: {size_mb:.1f} MB")

    return output_dir

def save_predictions(predictions, filename="submission.csv"):
    """Save predictions to CSV for submission."""
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
`
  }

  getPaths(ctx: TemplateContext): PlatformPaths {
    const competition = ctx.competition?.competition.slug || 'competition-name'
    return {
      input: `/kaggle/input/${competition}`,
      output: '/kaggle/working',
      checkpoint: '/kaggle/working/checkpoints',
      model_cache: '/kaggle/working/.cache/huggingface',
    }
  }

  getRecommendedBatchSize(modelName: string): number {
    return getRecommendedBatchSize(this.id, modelName)
  }

  getRecommendedGradientAccumulation(modelName: string, targetBatchSize: number): number {
    const actualBatch = this.getRecommendedBatchSize(modelName)
    return Math.max(1, Math.ceil(targetBatchSize / actualBatch))
  }

  generateMetadata(ctx: TemplateContext): Record<string, unknown> {
    const competition = ctx.competition?.competition.slug || 'competition-name'
    const username = ctx.competition?.competition.kaggle?.username || 'unknown'

    return {
      id: `${username}/notebook-${Date.now()}`,
      title: `Training Notebook - ${competition}`,
      code_file: 'notebook.py',
      language: 'python',
      kernel_type: 'script',
      is_private: true,
      enable_gpu: this.id !== 'kaggle-cpu',
      enable_tpu: false,
      enable_internet: true,
      dataset_sources: [],
      competition_sources: [competition],
      kernel_sources: [],
      model_sources: ctx.competition?.competition.kaggle?.model_sources || [],
    }
  }
}

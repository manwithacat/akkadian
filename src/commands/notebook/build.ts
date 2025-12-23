/**
 * Notebook Build Command
 *
 * Generates notebooks deterministically from a TOML configuration file.
 * This ensures reproducible notebook generation with explicit ML decisions.
 */

import TOML from '@iarna/toml'
import { existsSync, readFileSync } from 'fs'
import { basename, dirname, join } from 'path'
import { z } from 'zod'
import { getActiveCompetitionDir, loadActiveCompetitionDirectory } from '../../lib/config'
import { error, success } from '../../lib/output'
import { createTemplateEngine, PLATFORM_DISPLAY_NAMES } from '../../templates'
import type { CommandDefinition } from '../../types/commands'
import type { PlatformId } from '../../types/platform'
import type { TemplateContext } from '../../types/template'

/**
 * Training config schema matching training.toml structure
 */
interface TrainingConfig {
  meta: {
    name: string
    description: string
    version: string
    template: string
  }
  model: {
    name: string
    src_lang: string
    tgt_lang: string
    precision: {
      training: 'fp32' | 'fp16' | 'bf16'
      checkpoint: 'fp32' | 'fp16'
      mixed_precision: boolean
    }
  }
  data: {
    sources: string[]
    source_column: string
    target_column: string
    val_split: number
    seed: number
    preprocessing: {
      max_src_len: number
      max_tgt_len: number
      dynamic_padding: boolean
    }
  }
  training: {
    num_epochs: number
    batch_size: number
    gradient_accumulation_steps: number
    learning_rate: number
    weight_decay: number
    warmup_ratio: number
    optimizer?: {
      name: string
      beta1?: number
      beta2?: number
      epsilon?: number
    }
    scheduler?: {
      name: string
    }
  }
  evaluation: {
    eval_steps: number
    save_steps: number
    logging_steps: number
    metric: string
    greater_is_better: boolean
    predict_with_generate: boolean
    early_stopping?: {
      enabled: boolean
      patience: number
    }
  }
  checkpoints: {
    save_total_limit: number
    save_optimizer: boolean
    load_best_at_end: boolean
  }
  generation: {
    num_beams: number
    max_new_tokens: number
    repetition_penalty: number
    no_repeat_ngram_size: number
  }
  platform: {
    target: string
    clear_hf_cache: boolean
  }
  output: {
    dir: string
    save_model: boolean
    save_tokenizer: boolean
    save_config: boolean
    save_length_config: boolean
  }
}

/**
 * Parse and validate training config from TOML file
 */
function parseTrainingConfig(path: string): TrainingConfig {
  const content = readFileSync(path, 'utf-8')
  const config = TOML.parse(content) as unknown as TrainingConfig
  return config
}

/**
 * Convert dataset source to Kaggle format
 */
function parseDataSource(source: string): { type: string; name: string; path: string } {
  // Handle Kaggle dataset references
  if (source.includes('/') && !source.startsWith('/')) {
    // Kaggle dataset: username/dataset-name
    const parts = source.split('/')
    if (parts.length === 2) {
      return {
        type: 'dataset',
        name: source,
        path: `/kaggle/input/${parts[1]}/*.csv`,
      }
    }
  }
  // Handle competition references (no slash)
  if (!source.includes('/')) {
    return {
      type: 'competition',
      name: source,
      path: `/kaggle/input/${source}/*.csv`,
    }
  }
  // Local path
  return {
    type: 'local',
    name: basename(source),
    path: source,
  }
}

/**
 * Generate Python notebook from training config
 */
function generateNotebook(config: TrainingConfig, configPath: string): string {
  const dataSources = config.data.sources.map(parseDataSource)
  const timestamp = new Date().toISOString()

  // Generate Kaggle/Jupytext header
  const header = `# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   kaggle:
#     accelerator: gpu
#     dataSources:
${dataSources
  .map(
    (s) => `#       - type: ${s.type}
#         name: ${s.name}`
  )
  .join('\n')}
#     docker_image: gcr.io/kaggle-gpu-images/python
#     isGpuEnabled: true
#     isInternetEnabled: true
#     language: python
#     sourceType: script
# ---`

  // Generate the notebook content
  return `${header}

# %% [markdown]
# # ${config.meta.name}
#
# ${config.meta.description}
#
# **Generated from**: \`${basename(configPath)}\`
# **Version**: ${config.meta.version}
# **Generated at**: ${timestamp}

# %%
import os
import sys
import time
import json
import glob
import subprocess
import shutil

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import pandas as pd
import numpy as np
from datetime import datetime

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## Install Dependencies

# %%
packages = ["sentencepiece", "sacrebleu", "evaluate"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps"] + packages, check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.35.0", "accelerate"], check=True)
print("Packages installed")

# %%
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate

# %% [markdown]
# ## Configuration
#
# Auto-generated from \`${basename(configPath)}\` - DO NOT EDIT MANUALLY
# To modify, edit the TOML file and regenerate with: akk notebook build ${basename(configPath)}

# %%
# === AUTO-GENERATED CONFIG ===
# Source: ${basename(configPath)}
# Generated: ${timestamp}
# Version: ${config.meta.version}

CONFIG = {
    # Model
    "model_name": "${config.model.name}",

    # Language codes
    "src_lang": "${config.model.src_lang}",
    "tgt_lang": "${config.model.tgt_lang}",

    # Precision
    "training_precision": "${config.model.precision.training}",
    "checkpoint_precision": "${config.model.precision.checkpoint}",
    "fp16": ${config.model.precision.mixed_precision ? 'True' : 'False'},

    # Training
    "num_epochs": ${config.training.num_epochs},
    "batch_size": ${config.training.batch_size},
    "gradient_accumulation_steps": ${config.training.gradient_accumulation_steps},
    "learning_rate": ${config.training.learning_rate},
    "weight_decay": ${config.training.weight_decay},
    "warmup_ratio": ${config.training.warmup_ratio},

    # Sequence lengths
    "max_src_len": ${config.data.preprocessing.max_src_len},
    "max_tgt_len": ${config.data.preprocessing.max_tgt_len},

    # Evaluation
    "eval_steps": ${config.evaluation.eval_steps},
    "save_steps": ${config.evaluation.save_steps},
    "logging_steps": ${config.evaluation.logging_steps},
    "predict_with_generate": ${config.evaluation.predict_with_generate ? 'True' : 'False'},
    "metric_for_best_model": "${config.evaluation.metric}",
    "greater_is_better": ${config.evaluation.greater_is_better ? 'True' : 'False'},

    # Checkpoints
    "save_total_limit": ${config.checkpoints.save_total_limit},
    "save_only_model": ${config.checkpoints.save_optimizer === false ? 'True' : 'False'},  # True = skip optimizer.pt (saves ~4GB per checkpoint)
    "load_best_at_end": ${config.checkpoints.load_best_at_end ? 'True' : 'False'},

    # Early stopping
    "early_stopping_enabled": ${config.evaluation.early_stopping?.enabled ? 'True' : 'False'},
    "early_stopping_patience": ${config.evaluation.early_stopping?.patience ?? 3},

    # Validation split
    "val_split": ${config.data.val_split},
    "seed": ${config.data.seed},

    # Generation
    "num_beams": ${config.generation.num_beams},
    "max_new_tokens": ${config.generation.max_new_tokens},
    "repetition_penalty": ${config.generation.repetition_penalty},
    "no_repeat_ngram_size": ${config.generation.no_repeat_ngram_size},

    # Platform
    "clear_hf_cache": ${config.platform.clear_hf_cache ? 'True' : 'False'},

    # Output
    "output_dir": "${config.output.dir}",
}

print("Configuration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## Load Dataset

# %%
# Dataset sources (in priority order)
DATASET_SOURCES = [
${dataSources.map((s) => `    "${s.path}",`).join('\n')}
]

train_df = None
dataset_type = None

for source in DATASET_SOURCES:
    files = glob.glob(source, recursive=True)
    if files:
        print(f"Using dataset: {files[0]}")
        train_df = pd.read_csv(files[0])
        dataset_type = source.split("/")[-2] if "/" in source else "local"
        break

if train_df is None:
    raise RuntimeError(f"No training data found! Searched: {DATASET_SOURCES}")

print(f"\\nTraining samples: {len(train_df)}")
print(f"Columns: {train_df.columns.tolist()}")

# %%
# Data quality check
print(f"\\nData quality:")
print(f"  Null sources: {train_df['${config.data.source_column}'].isna().sum()}")
print(f"  Null targets: {train_df['${config.data.target_column}'].isna().sum()}")
print(f"  Avg source length: {train_df['${config.data.source_column}'].str.len().mean():.1f} chars")
print(f"  Avg target length: {train_df['${config.data.target_column}'].str.len().mean():.1f} chars")

train_df = train_df.dropna(subset=['${config.data.source_column}', '${config.data.target_column}'])
print(f"  After cleaning: {len(train_df)} samples")

# %% [markdown]
# ## Initialize Model

# %%
print(f"\\nLoading model: {CONFIG['model_name']}")
start = time.time()

tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"])

print(f"Model loaded in {time.time()-start:.1f}s")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# %%
# Clear HuggingFace cache to save disk space
if CONFIG["clear_hf_cache"]:
    hf_cache = os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(hf_cache):
        cache_size = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(hf_cache) for f in fn)
        print(f"Clearing HF cache: {cache_size / 1e9:.2f} GB")
        shutil.rmtree(hf_cache, ignore_errors=True)
        print("HF cache cleared to save disk space")

# %%
tokenizer.src_lang = CONFIG["src_lang"]
tokenizer.tgt_lang = CONFIG["tgt_lang"]

tgt_token_id = tokenizer.convert_tokens_to_ids(CONFIG["tgt_lang"])
print(f"Target language token ID: {tgt_token_id}")

# %% [markdown]
# ## Preprocessing

# %%
def preprocess_function(examples):
    inputs = examples["${config.data.source_column}"]
    targets = examples["${config.data.target_column}"]

    # Modern tokenization API (as_target_tokenizer is deprecated)
    model_inputs = tokenizer(
        inputs,
        text_target=targets,
        max_length=CONFIG["max_src_len"],
        truncation=True,
        padding=False,
    )

    return model_inputs

# %%
print("\\nPreparing datasets...")

dataset = Dataset.from_pandas(train_df[["${config.data.source_column}", "${config.data.target_column}"]])
split = dataset.train_test_split(test_size=CONFIG["val_split"], seed=CONFIG["seed"])
train_dataset = split["train"]
val_dataset = split["test"]

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# %%
train_tokenized = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
val_tokenized = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

print(f"Tokenized - Train: {len(train_tokenized)}, Val: {len(val_tokenized)}")

# %% [markdown]
# ## Training Setup

# %%
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=CONFIG["output_dir"],
    num_train_epochs=CONFIG["num_epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
    warmup_ratio=CONFIG["warmup_ratio"],
    fp16=CONFIG["fp16"],
    eval_strategy="steps",  # Modern API (evaluation_strategy is deprecated)
    eval_steps=CONFIG["eval_steps"],
    save_strategy="steps",
    save_steps=CONFIG["save_steps"],
    save_total_limit=CONFIG["save_total_limit"],
    save_only_model=CONFIG["save_only_model"],  # Skip optimizer.pt to save disk space
    logging_steps=CONFIG["logging_steps"],
    load_best_model_at_end=CONFIG["load_best_at_end"],
    metric_for_best_model=CONFIG["metric_for_best_model"],
    greater_is_better=CONFIG["greater_is_better"],
    predict_with_generate=CONFIG["predict_with_generate"],
    report_to="none",
    seed=CONFIG["seed"],
)

callbacks = []
if CONFIG["early_stopping_enabled"]:
    callbacks.append(EarlyStoppingCallback(early_stopping_patience=CONFIG["early_stopping_patience"]))

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=callbacks,
)

# %% [markdown]
# ## Train

# %%
print("\\n" + "="*60)
print("STARTING TRAINING")
print(f"Config: ${config.meta.name} v${config.meta.version}")
print(f"Dataset: {dataset_type} ({len(train_tokenized)} samples)")
print("="*60 + "\\n")

train_start = time.time()
train_result = trainer.train()
train_time = time.time() - train_start

print(f"\\nTraining completed in {train_time/60:.1f} minutes")
print(f"Final metrics: {train_result.metrics}")

# %% [markdown]
# ## Save Model

# %%
output_dir = CONFIG["output_dir"]
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# Save training config
config_path = os.path.join(output_dir, "training_config.json")
with open(config_path, "w") as f:
    json.dump({
        **{k: v for k, v in CONFIG.items() if not callable(v)},
        "dataset_type": dataset_type,
        "train_samples": len(train_tokenized),
        "val_samples": len(val_tokenized),
        "train_time_minutes": train_time / 60,
        "final_loss": train_result.metrics.get("train_loss"),
        "_meta": {
            "name": "${config.meta.name}",
            "version": "${config.meta.version}",
            "source_config": "${basename(configPath)}",
            "generated_at": "${timestamp}",
        },
    }, f, indent=2)

# Save length config for inference
length_config = {
    "length_ratio": 1.5,
    "length_slack": 20,
    "global_max": 512,
    "global_min": 32,
}
with open(os.path.join(output_dir, "length_config.json"), "w") as f:
    json.dump(length_config, f)

print(f"\\nModel saved to: {output_dir}")
print(f"Files: {os.listdir(output_dir)}")

# %% [markdown]
# ## Quick Evaluation

# %%
print("\\n" + "="*60)
print("SAMPLE TRANSLATIONS")
print("="*60)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

test_samples = val_dataset.select(range(min(5, len(val_dataset))))

for i, sample in enumerate(test_samples):
    src = sample["${config.data.source_column}"]
    ref = sample["${config.data.target_column}"]

    inputs = tokenizer(src, return_tensors="pt", max_length=CONFIG["max_src_len"], truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=CONFIG["max_new_tokens"],
            num_beams=CONFIG["num_beams"],
            forced_bos_token_id=tgt_token_id,
            repetition_penalty=CONFIG["repetition_penalty"],
            no_repeat_ngram_size=CONFIG["no_repeat_ngram_size"],
        )

    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\\n[{i}] SRC: {src[:80]}...")
    print(f"    REF: {ref[:80]}...")
    print(f"    OUT: {pred[:80]}...")

# %%
print("\\n" + "="*60)
print("TRAINING COMPLETE")
print(f"Config: ${config.meta.name} v${config.meta.version}")
print(f"Dataset: {dataset_type}")
print(f"Samples: {len(train_tokenized)} train, {len(val_tokenized)} val")
print(f"Time: {train_time/60:.1f} minutes")
print(f"Model saved to: {output_dir}")
print("="*60)
`
}

/**
 * Generate Kaggle kernel metadata
 */
function generateMetadata(config: TrainingConfig, outputPath: string): Record<string, unknown> {
  const dataSources = config.data.sources.map(parseDataSource)
  // Include version in slug for unique kernel per version (avoids Kaggle's versioning UX)
  const baseName = config.meta.name.toLowerCase().replace(/[^a-z0-9-]/g, '-')
  const version = config.meta.version.replace(/\./g, '-')
  const slug = `${baseName}-v${version}`

  return {
    id: `manwithacat/${slug}`,
    title: `${config.meta.name} v${config.meta.version}`,
    code_file: basename(outputPath),
    language: 'python',
    kernel_type: 'script',
    is_private: false,
    enable_gpu: true,
    enable_tpu: false,
    enable_internet: true,
    dataset_sources: dataSources.filter((s) => s.type === 'dataset').map((s) => s.name),
    competition_sources: dataSources.filter((s) => s.type === 'competition').map((s) => s.name),
    kernel_sources: [],
    model_sources: [],
  }
}

const BuildArgs = z.object({
  path: z.string().describe('Path to training.toml config file'),
  output: z.string().optional().describe('Output file path (default: derived from config)'),
  dryRun: z.boolean().default(false).describe('Show what would be generated without writing'),
  skipPreflight: z.boolean().default(false).describe('Skip preflight validation'),
})

export const build: CommandDefinition<typeof BuildArgs> = {
  name: 'notebook build',
  description: 'Generate a notebook from a TOML configuration file',
  help: `
Generate a training notebook deterministically from a TOML configuration file.

This ensures:
- All ML decisions are explicit and version-controlled
- Notebooks are reproducible from config changes
- Modern APIs are used (no deprecated patterns)
- Platform-specific optimizations are applied

The config file should contain all training parameters including:
- Model name and precision settings (fp16/fp32)
- Training hyperparameters (epochs, batch size, learning rate)
- Evaluation settings (step frequency, metrics)
- Checkpoint settings (save limits, optimizer states)
- Platform optimizations (HF cache clearing)

Example config structure: see notebooks/kaggle/training.toml
`,
  examples: [
    'akk notebook build training.toml',
    'akk notebook build training.toml -o train.py',
    'akk notebook build training.toml --dry-run',
  ],
  args: BuildArgs,

  async run(args, ctx) {
    // Validate config file exists
    if (!existsSync(args.path)) {
      return error(
        'CONFIG_NOT_FOUND',
        `Config file not found: ${args.path}`,
        'Provide a valid path to a training.toml file'
      )
    }

    // Parse config
    let config: TrainingConfig
    try {
      config = parseTrainingConfig(args.path)
    } catch (err) {
      return error(
        'CONFIG_PARSE_ERROR',
        `Failed to parse config: ${err instanceof Error ? err.message : 'Unknown error'}`,
        'Check the TOML syntax and required fields'
      )
    }

    // Determine output path (include version for unique kernels per version)
    const baseName = config.meta.name.toLowerCase().replace(/[^a-z0-9]/g, '_')
    const versionSuffix = `_v${config.meta.version.replace(/\./g, '_')}`
    const filename = `${baseName}${versionSuffix}.py`

    // Default to competition directory structure if no explicit output
    let outputPath = args.output
    if (!outputPath && ctx.config) {
      const competitionDir = getActiveCompetitionDir(ctx.cwd, ctx.config)
      // Detect if this is an inference or training notebook
      const isInference = config.meta.template === 'inference' || config.meta.name.toLowerCase().includes('inference')
      const notebookSubdir = isInference ? 'notebooks/inference' : 'notebooks/training'
      outputPath = join(competitionDir, notebookSubdir, filename)
    }
    // Fallback to input directory if no competition context
    if (!outputPath) {
      outputPath = join(dirname(args.path), filename)
    }

    // Generate notebook
    const notebook = generateNotebook(config, args.path)

    // Generate metadata
    const metadata = generateMetadata(config, outputPath)
    const metadataPath = outputPath.replace(/\.py$/, '-metadata.json')

    if (args.dryRun) {
      return success({
        message: 'Dry run - no files written',
        config: {
          name: config.meta.name,
          version: config.meta.version,
          model: config.model.name,
          precision: config.model.precision,
          training: {
            epochs: config.training.num_epochs,
            batch_size: config.training.batch_size,
            learning_rate: config.training.learning_rate,
          },
        },
        output: outputPath,
        metadata: metadataPath,
        notebook_preview: notebook.slice(0, 500) + '...',
      })
    }

    // Write files
    await Bun.write(outputPath, notebook)
    await Bun.write(metadataPath, JSON.stringify(metadata, null, 2))

    // Run preflight check
    let preflightResult = null
    if (!args.skipPreflight) {
      // Import preflight dynamically to avoid circular deps
      const { preflight } = await import('../preflight')
      preflightResult = await preflight.run(
        {
          path: outputPath,
          platform: config.platform.target,
          samples: 2000,
          verbose: false,
        },
        ctx
      )
    }

    return success({
      message: 'Notebook generated successfully',
      config: {
        name: config.meta.name,
        version: config.meta.version,
        source: args.path,
      },
      output: {
        notebook: outputPath,
        metadata: metadataPath,
      },
      model: {
        name: config.model.name,
        precision: config.model.precision,
      },
      training: {
        epochs: config.training.num_epochs,
        batch_size: config.training.batch_size,
        effective_batch: config.training.batch_size * config.training.gradient_accumulation_steps,
        learning_rate: config.training.learning_rate,
      },
      preflight: preflightResult?.success ? preflightResult.data : { skipped: true },
    })
  },
}

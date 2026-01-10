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
import { getActiveCompetitionDir } from '../../lib/config'
import { error, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

/**
 * Training/Inference config schema matching .toml structure
 */
interface TrainingConfig {
  meta: {
    name: string
    description: string
    version: string
    template: string
    /** Mode: "training" (default) or "inference" */
    mode?: 'training' | 'inference'
  }
  model: {
    name: string
    src_lang: string
    tgt_lang: string
    precision?: {
      training?: 'fp32' | 'fp16' | 'bf16'
      checkpoint?: 'fp32' | 'fp16'
      mixed_precision?: boolean
      inference?: 'fp32' | 'fp16' | 'bf16'
    }
    /** Model source for loading pre-trained models */
    source?: {
      /** Source type: huggingface (default), kaggle, or local */
      type: 'huggingface' | 'kaggle' | 'local'
      /** Kaggle model handle (e.g., "username/model/framework/variation") */
      handle?: string
      /** Local path to model directory */
      path?: string
    }
  }
  /** Data config - required for training, optional for inference */
  data?: {
    sources?: string[]
    source_column: string
    target_column?: string
    val_split?: number
    seed?: number
    preprocessing?: {
      max_src_len: number
      max_tgt_len: number
      dynamic_padding: boolean
    }
    /** Test data path for inference mode */
    test_source?: string
    /** ID column for submission */
    id_column?: string
  }
  /** Training config - required for training mode */
  training?: {
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
  /** Evaluation config - required for training mode */
  evaluation?: {
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
  /** Checkpoint config - required for training mode */
  checkpoints?: {
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
  kaggle_model?: {
    /** Kaggle model handle (e.g., "username/model-name") */
    handle: string
    /** Framework for the model (e.g., "transformers", "pytorch") */
    framework: string
    /** License name (e.g., "Apache 2.0") */
    license?: string
    // Note: variation is auto-derived from meta.version (e.g., 1.0.7 -> v1-0-7)

    // Model-level metadata (for Kaggle usability score)
    /** Brief subtitle (20-80 chars) */
    subtitle?: string
    /** Full model card description (markdown) */
    description?: string
    /** Data provenance/attribution */
    provenance?: string

    // Instance-level metadata
    /** Brief overview of the model instance */
    overview?: string
    /** Usage examples and documentation (markdown) */
    usage?: string
    /** Whether the model can be fine-tuned */
    fine_tunable?: boolean
    /** List of training data sources */
    training_data?: string[]
    /** URL to the base model (e.g., HuggingFace) */
    base_model_url?: string
  }
  /** Submission generation config (for competition notebooks) */
  submission?: {
    /** Enable submission.csv generation (default: false) */
    enabled: boolean
    /** Test data source path (default: competition test.csv) */
    test_source?: string
    /** Column containing source text (default: auto-detect 'transliteration' or 'source') */
    source_column?: string
    /** Column containing row IDs (default: 'id') */
    id_column?: string
    /** Output column name (default: 'translation') */
    output_column?: string
    /** Inference batch size (default: training batch_size) */
    batch_size?: number
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
function parseDataSource(source: string): {
  type: string
  name: string
  path: string
} {
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
 * Check if model is a T5/ByT5/mT5 variant (uses task prefix, not lang codes)
 */
function isT5Model(modelName: string): boolean {
  const t5Patterns = ['t5', 'byt5', 'mt5', 'flan-t5', 'long-t5']
  const lowerName = modelName.toLowerCase()
  return t5Patterns.some((p) => lowerName.includes(p))
}

/**
 * Generate model-specific tokenizer setup code
 */
function generateTokenizerSetup(config: TrainingConfig): string {
  if (isT5Model(config.model.name)) {
    // T5 models use task prefix, not lang codes
    return `# T5/ByT5 uses task prefix for translation
TASK_PREFIX = "translate Akkadian to English: "
print(f"Using T5 task prefix: {TASK_PREFIX}")`
  }

  // NLLB models use lang codes
  return `tokenizer.src_lang = CONFIG["src_lang"]
tokenizer.tgt_lang = CONFIG["tgt_lang"]

tgt_token_id = tokenizer.convert_tokens_to_ids(CONFIG["tgt_lang"])
print(f"Target language token ID: {tgt_token_id}")`
}

/**
 * Generate model-specific evaluation/inference code
 */
function generateEvalSampleCode(config: TrainingConfig): string {
  const srcCol = config.data.source_column

  if (isT5Model(config.model.name)) {
    // T5 models use task prefix
    return `for i, sample in enumerate(test_samples):
    src = sample["${srcCol}"]
    ref = sample["${config.data.target_column}"]

    # T5: prepend task prefix
    inputs = tokenizer(TASK_PREFIX + src, return_tensors="pt", max_length=CONFIG["max_src_len"], truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=CONFIG["max_new_tokens"],
            num_beams=CONFIG["num_beams"],
            repetition_penalty=CONFIG["repetition_penalty"],
            no_repeat_ngram_size=CONFIG["no_repeat_ngram_size"],
        )

    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\\n[{i}] SRC: {src[:80]}...")
    print(f"    REF: {ref[:80]}...")
    print(f"    OUT: {pred[:80]}...")`
  }

  // NLLB models use forced_bos_token_id
  return `for i, sample in enumerate(test_samples):
    src = sample["${srcCol}"]
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
    print(f"    OUT: {pred[:80]}...")`
}

/**
 * Generate submission code for competition (loads test.csv, translates, saves submission.csv)
 */
function generateSubmissionCode(config: TrainingConfig): string {
  if (!config.submission?.enabled) {
    return ''
  }

  const testSource = config.submission.test_source || '/kaggle/input/**/test.csv'
  const srcCol = config.submission.source_column || 'transliteration'
  const idCol = config.submission.id_column || 'id'
  const outCol = config.submission.output_column || 'translation'
  const batchSize = config.submission.batch_size || config.training.batch_size

  // Generate translate_batch function at module level (not nested in if block)
  const translateBatchFn = isT5Model(config.model.name)
    ? `def translate_batch(texts):
    """Translate a batch of texts."""
    # T5: prepend task prefix
    inputs = tokenizer(
        [TASK_PREFIX + t for t in texts],
        max_length=CONFIG["max_src_len"],
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=CONFIG["max_new_tokens"],
            num_beams=CONFIG["num_beams"],
            repetition_penalty=CONFIG["repetition_penalty"],
            no_repeat_ngram_size=CONFIG["no_repeat_ngram_size"],
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)`
    : `def translate_batch(texts):
    """Translate a batch of texts."""
    inputs = tokenizer(
        texts,
        max_length=CONFIG["max_src_len"],
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
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

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)`

  return `
# %% [markdown]
# ## Generate Competition Submission

# %%
import re

def clean_translation(text):
    """
    Clean ORACC academic conventions from translation output.
    These artifacts hurt scoring against clean English references.
    """
    # Remove indeterminate plural markers: "( s )" or "(s)"
    text = re.sub(r'\\s*\\(\\s*s\\s*\\)', 's', text)

    # Remove parenthesized alternatives: "( or another )" -> "or another"
    text = re.sub(r'\\s*\\(\\s*(or|and)\\s+([^)]+)\\s*\\)\\s*', r' \\1 \\2 ', text)

    # Remove parenthesized conjunctions alone: "( or )" "( and )" etc.
    text = re.sub(r'\\s*\\(\\s*(or|and|i\\.e\\.|etc\\.)\\s*\\)\\s*', r' \\1 ', text)

    # Remove Akkadian determinatives: (ki), (m), (f), (d), etc.
    text = re.sub(r'\\s*\\(\\s*(ki|m|f|d|pl|sg|lú|giš|kur|uru)\\s*\\)', '', text, flags=re.IGNORECASE)

    # Remove other short parentheticals that look like annotations
    text = re.sub(r'\\s*\\(\\s*[a-z]{1,3}\\s*\\)', '', text)

    # Normalize multiple spaces
    text = re.sub(r'\\s+', ' ', text)

    # Fix spacing around punctuation
    text = re.sub(r'\\s+([.,;:!?])', r'\\1', text)
    text = re.sub(r'([.,;:!?])(?=[A-Za-z])', r'\\1 ', text)

    return text.strip()

# %%
# Load test data
test_files = glob.glob("${testSource}", recursive=True)
print(f"Found test files: {test_files}")

if not test_files:
    print("WARNING: test.csv not found - skipping submission generation")
    test_df = None
else:
    test_df = pd.read_csv(test_files[0])
    print(f"Test samples: {len(test_df)}")
    print(test_df.head())

# %%
# Define batch translation function
${translateBatchFn}

# %%
if test_df is not None:
    # Detect source column
    source_col = "${srcCol}" if "${srcCol}" in test_df.columns else "source" if "source" in test_df.columns else test_df.columns[1]
    sources = test_df[source_col].tolist()

    print(f"\\nTranslating {len(sources)} test samples using column: {source_col}")

    translations = []
    batch_size = ${batchSize}
    for i in range(0, len(sources), batch_size):
        batch = sources[i:i+batch_size]
        batch_translations = translate_batch(batch)
        translations.extend(batch_translations)
        if (i + batch_size) % 100 == 0 or i + batch_size >= len(sources):
            print(f"  Translated {min(i + batch_size, len(sources))}/{len(sources)}")

    print(f"Generated {len(translations)} translations")

    # Clean translations (remove ORACC artifacts that hurt scoring)
    translations = [clean_translation(t) for t in translations]
    print("Cleaned ORACC artifacts from translations")

    # Create submission
    id_col = "${idCol}" if "${idCol}" in test_df.columns else "id"
    submission = pd.DataFrame({
        id_col: test_df[id_col] if id_col in test_df.columns else range(len(translations)),
        "${outCol}": translations
    })

    print("\\nSubmission preview:")
    print(submission.head(10))

    submission.to_csv("submission.csv", index=False)
    print("\\nSaved submission.csv")

    # Show sample translations
    print("\\n=== Sample Test Translations ===")
    for i in range(min(5, len(sources))):
        print(f"\\n[{i}]")
        print(f"SRC: {sources[i][:100]}...")
        print(f"OUT: {translations[i][:100]}...")
`
}

/**
 * Generate model-specific preprocessing function
 */
function generatePreprocessFunction(config: TrainingConfig): string {
  const srcCol = config.data.source_column
  const tgtCol = config.data.target_column

  if (isT5Model(config.model.name)) {
    // T5 models need task prefix prepended to inputs
    return `def preprocess_function(examples):
    # T5/ByT5: prepend task prefix to inputs
    inputs = [TASK_PREFIX + str(x) for x in examples["${srcCol}"]]
    targets = [str(x) for x in examples["${tgtCol}"]]

    model_inputs = tokenizer(
        inputs,
        max_length=CONFIG["max_src_len"],
        truncation=True,
        padding=False,
    )

    # Tokenize targets
    labels = tokenizer(
        targets,
        max_length=CONFIG["max_tgt_len"],
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs`
  }

  // NLLB models use text_target parameter
  return `def preprocess_function(examples):
    inputs = examples["${srcCol}"]
    targets = examples["${tgtCol}"]

    # Modern tokenization API (as_target_tokenizer is deprecated)
    model_inputs = tokenizer(
        inputs,
        text_target=targets,
        max_length=CONFIG["max_src_len"],
        truncation=True,
        padding=False,
    )

    return model_inputs`
}

/**
 * Generate model loading code based on source type
 */
function generateModelLoadingCode(config: TrainingConfig): string {
  const sourceType = config.model.source?.type || 'huggingface'

  if (sourceType === 'kaggle' && config.model.source?.handle) {
    // Load from Kaggle Model Registry
    return `# kagglehub is pre-installed on Kaggle
import kagglehub

# Download model from Kaggle registry
kaggle_handle = "${config.model.source.handle}"
print(f"\\nDownloading model from Kaggle: {kaggle_handle}")
start = time.time()

model_path = kagglehub.model_download(kaggle_handle)
print(f"Model downloaded to: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print(f"Model loaded in {time.time()-start:.1f}s")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")`
  }

  if (sourceType === 'local' && config.model.source?.path) {
    // Load from local path
    return `# Load model from local path
local_model_path = "${config.model.source.path}"
print(f"\\nLoading model from: {local_model_path}")
start = time.time()

tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)

print(f"Model loaded in {time.time()-start:.1f}s")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")`
  }

  // Default: HuggingFace
  return `print(f"\\nLoading model: {CONFIG['model_name']}")
start = time.time()

tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"])

print(f"Model loaded in {time.time()-start:.1f}s")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")`
}

/**
 * Generate inference-only notebook from config
 */
function generateInferenceNotebook(config: TrainingConfig, configPath: string): string {
  const timestamp = new Date().toISOString()
  const testSource =
    config.data?.test_source ||
    config.submission?.test_source ||
    '/kaggle/input/deep-past-initiative-machine-translation/test.csv'
  const sourceColumn = config.data?.source_column || config.submission?.source_column || 'transliteration'
  const idColumn = config.data?.id_column || config.submission?.id_column || 'id'
  const outputColumn = config.submission?.output_column || 'translation'
  const batchSize = config.submission?.batch_size || config.generation?.batch_size || 16
  const useFp16 = config.model.precision?.inference === 'fp16'

  // Determine model source
  const modelSource = config.model.source?.type || 'huggingface'
  const modelHandle = config.model.source?.handle || config.model.name

  return `# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   kaggle:
#     accelerator: gpu
#     dataSources:
#       - type: competition
#         name: deep-past-initiative-machine-translation
#     docker_image: gcr.io/kaggle-gpu-images/python
#     isGpuEnabled: true
#     isInternetEnabled: false
#     language: python
#     sourceType: script
# ---

# %% [markdown]
# # ${config.meta.name}
#
# ${config.meta.description}
#
# **Generated from**: \`${basename(configPath)}\`
# **Version**: ${config.meta.version}
# **Generated at**: ${timestamp}
# **Mode**: Inference (submission)
#
# Built with [Akkadian CLI](https://github.com/manwithacat/akkadian) - AI-first tooling for ML Competitions on Kaggle

# %%
import os
import sys
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import torch
import pandas as pd
from tqdm import tqdm

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()

# %% [markdown]
# ## Configuration

# %%
CONFIG = {
    "model_source": "${modelSource}",
    "model_handle": "${modelHandle}",
    "test_source": "${testSource}",
    "source_column": "${sourceColumn}",
    "id_column": "${idColumn}",
    "output_column": "${outputColumn}",
    "batch_size": ${batchSize},
    "num_beams": ${config.generation?.num_beams || 4},
    "max_new_tokens": ${config.generation?.max_new_tokens || 256},
    "use_fp16": ${useFp16 ? 'True' : 'False'},
}
print(f"Config: {CONFIG}")

# %% [markdown]
# ## Load Model from Registry

# %%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
${
  modelSource === 'kaggle'
    ? `
# kagglehub is pre-installed on Kaggle
import kagglehub

print(f"\\nDownloading model from Kaggle: {CONFIG['model_handle']}")
start = time.time()
model_path = kagglehub.model_download(CONFIG["model_handle"])
print(f"Downloaded in {time.time() - start:.1f}s to: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path${useFp16 ? ', torch_dtype=torch.float16' : ''})
`
    : `
print(f"\\nLoading model: {CONFIG['model_handle']}")
start = time.time()
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_handle"])
model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_handle"]${useFp16 ? ', torch_dtype=torch.float16' : ''})
`
}
print(f"Model loaded in {time.time() - start:.1f}s")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"Model on: {device}")

# %% [markdown]
# ## Load Test Data

# %%
test_df = pd.read_csv(CONFIG["test_source"])
print(f"Test samples: {len(test_df)}")
print(f"Columns: {list(test_df.columns)}")
print(f"\\nFirst few rows:")
print(test_df.head())

# %% [markdown]
# ## Generate Translations

# %%
def clean_translation(text):
    """Clean up translation output - remove ORACC academic conventions"""
    import re
    # Remove ( s ) plural markers
    text = re.sub(r'\\s*\\(\\s*s\\s*\\)\\s*', 's ', text)
    # Remove parenthesized conjunctions
    text = re.sub(r'\\s*\\(\\s*(or|and)\\s*\\)\\s*', r' \\1 ', text)
    # Remove determinatives like (ki), (d), (m), (f)
    text = re.sub(r'\\s*\\(\\s*(ki|d|m|f|lu|munus)\\s*\\)\\s*', ' ', text)
    # Clean up extra whitespace
    text = ' '.join(text.split())
    return text.strip()

# %%
print("\\n" + "="*60)
print("GENERATING TRANSLATIONS")
print("="*60)

texts = test_df[CONFIG["source_column"]].tolist()
ids = test_df[CONFIG["id_column"]].tolist()

# Add task prefix for ByT5
prefix = "translate Akkadian to English: "
prefixed_texts = [prefix + t for t in texts]

translations = []
batch_size = CONFIG["batch_size"]
start_time = time.time()

for i in tqdm(range(0, len(prefixed_texts), batch_size), desc="Translating"):
    batch = prefixed_texts[i:i+batch_size]

    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=CONFIG["num_beams"],
            max_new_tokens=CONFIG["max_new_tokens"],
            early_stopping=True
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    cleaned = [clean_translation(t) for t in decoded]
    translations.extend(cleaned)

    if (i // batch_size + 1) % 10 == 0:
        print(f"  Batch {i//batch_size + 1}/{(len(prefixed_texts) + batch_size - 1)//batch_size}")

elapsed = time.time() - start_time
print(f"\\nGenerated {len(translations)} translations in {elapsed:.1f}s")
print(f"Speed: {len(translations)/elapsed:.1f} samples/sec")

# %% [markdown]
# ## Sample Translations

# %%
print("\\n" + "="*60)
print("SAMPLE TRANSLATIONS")
print("="*60)
for i in range(min(5, len(texts))):
    print(f"\\n[{i+1}] Source: {texts[i][:100]}...")
    print(f"    Translation: {translations[i]}")

# %% [markdown]
# ## Create Submission

# %%
submission = pd.DataFrame({
    CONFIG["id_column"]: ids,
    CONFIG["output_column"]: translations
})
print(f"\\nSubmission shape: {submission.shape}")
print(submission.head())

submission.to_csv("submission.csv", index=False)
print(f"\\nSaved submission.csv")
print(f"Columns: {list(submission.columns)}")

# Verify file exists
import os
print(f"File size: {os.path.getsize('submission.csv'):,} bytes")

# %%
print("\\n" + "="*60)
print("INFERENCE COMPLETE")
print(f"Config: ${config.meta.name} v${config.meta.version}")
print(f"Model: {CONFIG['model_handle']}")
print(f"Samples: {len(translations)}")
print(f"Time: {elapsed:.1f}s")
print("="*60)
`
}

/**
 * Generate Python notebook from training config
 */
function generateNotebook(config: TrainingConfig, configPath: string): string {
  // Check for inference mode
  const mode = config.meta.mode || 'training'
  if (mode === 'inference') {
    return generateInferenceNotebook(config, configPath)
  }

  // Training mode - require data.sources
  if (!config.data?.sources) {
    throw new Error('Training mode requires data.sources. For inference, set meta.mode = "inference"')
  }

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
#
# Built with [Akkadian CLI](https://github.com/manwithacat/akkadian) - AI-first tooling for ML Competitions on Kaggle

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
os.environ["TQDM_DISABLE"] = "1"  # Disable progress bars for clean Kaggle logs
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # Disable HuggingFace download bars

import torch
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Suppress internal transformers deprecation warnings (past_key_values tuple format)
# These are library-internal and will be fixed in future transformers versions
warnings.filterwarnings("ignore", message=".*past_key_values.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

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
${
  config.submission?.enabled
    ? `# Competition submission mode: internet disabled, use pre-installed packages only
# Kaggle pre-installs: transformers, accelerate, sentencepiece, sacrebleu, datasets, etc.
print("Using Kaggle pre-installed packages (internet disabled for submission)")`
    : `# Install packages (internet enabled)
packages = ["sentencepiece", "sacrebleu"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + packages, check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.35.0", "accelerate"], check=True)
print("Packages installed")`
}

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
${
  config.submission?.enabled
    ? `# Note: sacrebleu not available offline - using eval_loss for model selection
# Competition scoring will compute chrF on their end`
    : `import sacrebleu  # Direct usage instead of evaluate.load() - works offline`
}

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
    "lr_scheduler_type": "${config.training.scheduler?.name || 'linear'}",

    # Sequence lengths
    "max_src_len": ${config.data.preprocessing.max_src_len},
    "max_tgt_len": ${config.data.preprocessing.max_tgt_len},

    # Evaluation
    "eval_steps": ${config.evaluation.eval_steps},
    "save_steps": ${config.evaluation.save_steps},
    "logging_steps": ${config.evaluation.logging_steps},
    "predict_with_generate": ${config.evaluation.predict_with_generate ? 'True' : 'False'},
    "metric_for_best_model": "${config.submission?.enabled ? 'eval_loss' : config.evaluation.metric}",
    "greater_is_better": ${config.submission?.enabled ? 'False' : config.evaluation.greater_is_better ? 'True' : 'False'},

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
${generateModelLoadingCode(config)}

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
${generateTokenizerSetup(config)}

# %% [markdown]
# ## Preprocessing

# %%
${generatePreprocessFunction(config)}

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

${
  config.submission?.enabled
    ? `# %%
# Submission mode: using eval_loss for model selection (sacrebleu not available offline)
# Competition will score using chrF on their end
compute_metrics = None`
    : `# %%
# Compute metrics using sacrebleu directly (no internet required)
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # Replace -100 (ignored tokens) with pad token for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Strip whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    # Compute chrF using sacrebleu directly
    chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels])
    return {"chrf": chrf.score}`
}

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
    lr_scheduler_type=CONFIG.get("lr_scheduler_type", "linear"),
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
    processing_class=tokenizer,  # Modern API (tokenizer= is deprecated in transformers>=4.46)
    data_collator=data_collator,
    compute_metrics=compute_metrics,
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
${
  config.kaggle_model
    ? `
# %% [markdown]
# ## Upload to Kaggle Model Registry

# %%
import subprocess
import json

# Model version is embedded in model name (not using Kaggle's variation system)
# e.g., akkadian-byt5-v1-0-10/transformers/default
model_version = "v${config.meta.version.replace(/\./g, '-')}"
owner_slug = "${config.kaggle_model.handle.split('/')[0]}"
model_slug = "${config.kaggle_model.handle.split('/')[1]}-" + model_version
instance_slug = "default"
framework = "${config.kaggle_model.framework}"

print(f"\\nPreparing model upload: {owner_slug}/{model_slug}/{framework}/{instance_slug}")

# Create model-metadata.json (model-level metadata)
model_metadata = {
    "ownerSlug": owner_slug,
    "title": f"${config.kaggle_model.handle.split('/')[1]}-{model_version}",
    "slug": model_slug,
    "subtitle": ${JSON.stringify(config.kaggle_model.subtitle || config.meta.description || '')},
    "isPrivate": False,
    "description": ${JSON.stringify(config.kaggle_model.description || `# ${config.meta.name}\n\n${config.meta.description || ''}`)},
    "publishTime": "",
    "provenanceSources": ${JSON.stringify(config.kaggle_model.provenance || '')}
}

model_metadata_path = os.path.join(output_dir, "model-metadata.json")
with open(model_metadata_path, "w") as f:
    json.dump(model_metadata, f, indent=2)
print(f"Created: {model_metadata_path}")

# Create model-instance-metadata.json (instance-level metadata)
instance_metadata = {
    "ownerSlug": owner_slug,
    "modelSlug": model_slug,
    "instanceSlug": instance_slug,
    "framework": framework,
    "overview": ${JSON.stringify(config.kaggle_model.overview || config.meta.description || '')},
    "usage": ${JSON.stringify(config.kaggle_model.usage || '')},
    "licenseName": "${config.kaggle_model.license || 'Apache 2.0'}",
    "fineTunable": ${config.kaggle_model.fine_tunable ?? true},
    "trainingData": ${JSON.stringify(config.kaggle_model.training_data || [])},
    "modelInstanceType": "Unspecified",
    "baseModelInstanceId": 0,
    "externalBaseModelUrl": ${JSON.stringify(config.kaggle_model.base_model_url || '')}
}

instance_metadata_path = os.path.join(output_dir, "model-instance-metadata.json")
with open(instance_metadata_path, "w") as f:
    json.dump(instance_metadata, f, indent=2)
print(f"Created: {instance_metadata_path}")

# Step 1: Create the model (if it doesn't exist)
print(f"\\nCreating model: {owner_slug}/{model_slug}")
result = subprocess.run(
    ["kaggle", "models", "create", "-p", output_dir],
    capture_output=True, text=True
)
if result.returncode == 0:
    print(f"Model created successfully")
elif "already exists" in result.stderr.lower() or "already exists" in result.stdout.lower():
    print(f"Model already exists, continuing with instance upload")
else:
    print(f"Model creation output: {result.stdout}")
    if result.stderr:
        print(f"Model creation stderr: {result.stderr}")

# Step 2: Create the model instance
print(f"\\nUploading instance: {owner_slug}/{model_slug}/{framework}/{instance_slug}")
result = subprocess.run(
    ["kaggle", "models", "instances", "create", "-p", output_dir],
    capture_output=True, text=True
)
if result.returncode == 0:
    print(f"Model uploaded successfully to: kaggle.com/models/{owner_slug}/{model_slug}")
else:
    print(f"Instance upload output: {result.stdout}")
    if result.stderr:
        print(f"Instance upload stderr: {result.stderr}")
    print("Model is saved locally and can be uploaded manually")
`
    : ''
}
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

${generateEvalSampleCode(config)}
${generateSubmissionCode(config)}
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
 *
 * IMPORTANT: enable_internet must be FALSE for competition submission kernels.
 * Kaggle runs scoring kernels with internet disabled, so any notebook that
 * generates submission.csv must work offline.
 */
function generateMetadata(config: TrainingConfig, outputPath: string): Record<string, unknown> {
  const mode = config.meta.mode || 'training'
  const isInferenceMode = mode === 'inference'

  // Include version in slug for unique kernel per version (avoids Kaggle's versioning UX)
  const baseName = config.meta.name.toLowerCase().replace(/[^a-z0-9-]/g, '-')
  const version = config.meta.version.replace(/\./g, '-')
  // Add suffix: T for training, I for inference
  const modeSuffix = isInferenceMode ? 'I' : 'T'
  const slug = `${baseName}-v${version}-${modeSuffix}`

  // Inference mode always has internet disabled (loads from Kaggle registry)
  // Training submission kernels also need internet disabled
  const isSubmissionKernel = config.submission?.enabled === true
  const enableInternet = !isInferenceMode && !isSubmissionKernel

  // Parse data sources for training mode, or use competition source for inference
  let datasetSources: string[] = []
  let competitionSources: string[] = []

  if (isInferenceMode) {
    // Inference mode: just needs competition test data
    competitionSources = ['deep-past-initiative-machine-translation']
  } else if (config.data?.sources) {
    const dataSources = config.data.sources.map(parseDataSource)
    datasetSources = dataSources.filter((s) => s.type === 'dataset').map((s) => s.name)
    competitionSources = dataSources.filter((s) => s.type === 'competition').map((s) => s.name)
  }

  // Add model source if loading from Kaggle registry
  const modelSources: string[] = []
  if (config.model.source?.type === 'kaggle' && config.model.source.handle) {
    // Extract model handle (owner/model-name from owner/model-name/framework/variation)
    const parts = config.model.source.handle.split('/')
    if (parts.length >= 2) {
      modelSources.push(`${parts[0]}/${parts[1]}`)
    }
  }

  return {
    id: `manwithacat/${slug}`,
    title: `${config.meta.name} v${config.meta.version}`,
    code_file: basename(outputPath),
    language: 'python',
    kernel_type: 'script',
    is_private: false,
    enable_gpu: true,
    enable_tpu: false,
    enable_internet: enableInternet,
    dataset_sources: datasetSources,
    competition_sources: competitionSources,
    kernel_sources: [],
    model_sources: modelSources,
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

    const mode = config.meta.mode || 'training'
    const isInferenceMode = mode === 'inference'

    // Run preflight check
    let preflightResult = null
    if (!args.skipPreflight && config.platform?.target) {
      // Import preflight dynamically to avoid circular deps
      const { preflight } = await import('../preflight')
      preflightResult = await preflight.run(
        {
          path: outputPath,
          platform: config.platform.target,
          samples: 2000,
          verbose: false,
          training: !isInferenceMode, // Training notebook - check for model save/upload
          competition: isInferenceMode, // Inference notebook - check for submission output
        },
        ctx
      )
    }

    // Build response based on mode
    const response: Record<string, unknown> = {
      message: `${isInferenceMode ? 'Inference' : 'Training'} notebook generated successfully`,
      mode,
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
        source: config.model.source?.type || 'huggingface',
        handle: config.model.source?.handle,
      },
      preflight: preflightResult?.success ? preflightResult.data : { skipped: true },
    }

    // Add training-specific info only for training mode
    if (!isInferenceMode && config.training) {
      response.training = {
        epochs: config.training.num_epochs,
        batch_size: config.training.batch_size,
        effective_batch: config.training.batch_size * config.training.gradient_accumulation_steps,
        learning_rate: config.training.learning_rate,
      }
    }

    // Add inference-specific info for inference mode
    if (isInferenceMode) {
      response.inference = {
        batch_size: config.submission?.batch_size || config.generation?.batch_size || 16,
        internet_enabled: false,
      }
    }

    return success(response)
  },
}

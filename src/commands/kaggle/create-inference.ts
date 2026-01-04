/**
 * Create Inference Kernel for Competition Submission
 *
 * Generates a complete inference notebook from a trained model in Kaggle registry.
 * Ensures all competition submission constraints are met:
 * - internet: OFF
 * - Uses kagglehub.model_download() for model loading
 * - Outputs submission.csv
 * - No pip installs or external downloads
 */

import { existsSync, mkdirSync } from 'fs'
import { basename, dirname, join } from 'path'
import { z } from 'zod'
import { findCompetitionConfig, loadCompetitionConfig } from '../../lib/config'
import { error, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

/**
 * Model type detection patterns
 */
const MODEL_PATTERNS = {
  byt5: {
    patterns: ['byt5', 'byte'],
    taskPrefix: 'translate Akkadian to English: ',
    preprocessor: null as string | null,
  },
  t5: {
    patterns: ['t5', 'mt5', 'flan'],
    taskPrefix: 'translate Akkadian to English: ',
    preprocessor: null,
  },
  nllb: {
    patterns: ['nllb'],
    taskPrefix: null,
    langCodes: { src: 'akk_Xsux', tgt: 'eng_Latn' },
  },
  qwen: {
    patterns: ['qwen'],
    taskPrefix: null,
    isInstruct: true,
  },
  gemma: {
    patterns: ['gemma'],
    taskPrefix: null,
    isInstruct: true,
  },
}

/**
 * Preprocessing type detection from model name
 */
const PREPROCESSING_PATTERNS = {
  consonantal: ['consonantal', 'skeleton', 'consonant'],
  annotated: ['annotated', 'enriched'],
  lexicon: ['lexicon', 'lex'],
}

/**
 * Detect model type from name
 */
function detectModelType(modelName: string): keyof typeof MODEL_PATTERNS | null {
  const lower = modelName.toLowerCase()
  for (const [type, config] of Object.entries(MODEL_PATTERNS)) {
    if (config.patterns.some((p) => lower.includes(p))) {
      return type as keyof typeof MODEL_PATTERNS
    }
  }
  return null
}

/**
 * Detect preprocessing type from model name
 */
function detectPreprocessing(modelName: string): string | null {
  const lower = modelName.toLowerCase()
  for (const [type, patterns] of Object.entries(PREPROCESSING_PATTERNS)) {
    if (patterns.some((p) => lower.includes(p))) {
      return type
    }
  }
  return null
}

/**
 * Parse Kaggle model handle
 * Format: owner/model-name/framework/variation or owner/model-name
 */
function parseModelHandle(handle: string): {
  owner: string
  model: string
  framework: string
  variation: string
  fullHandle: string
} {
  const parts = handle.split('/')
  if (parts.length >= 4) {
    return {
      owner: parts[0],
      model: parts[1],
      framework: parts[2],
      variation: parts[3],
      fullHandle: handle,
    }
  }
  if (parts.length >= 2) {
    return {
      owner: parts[0],
      model: parts[1],
      framework: 'transformers',
      variation: 'default',
      fullHandle: `${parts[0]}/${parts[1]}/transformers/default`,
    }
  }
  throw new Error(`Invalid model handle: ${handle}. Expected format: owner/model-name[/framework/variation]`)
}

/**
 * Extract version from model name (e.g., v1-0-10 -> 1.0.10)
 */
function extractVersion(modelName: string): string {
  const match = modelName.match(/v?(\d+)[.-](\d+)[.-](\d+)/)
  if (match) {
    return `${match[1]}.${match[2]}.${match[3]}`
  }
  return '1.0.0'
}

/**
 * Generate inference notebook content
 */
function generateInferenceNotebook(options: {
  modelHandle: string
  modelType: string
  preprocessing: string | null
  version: string
  name: string
  competition: string
}): string {
  const { modelHandle, modelType, preprocessing, version, name, competition } = options
  const timestamp = new Date().toISOString()

  // Generate preprocessing code based on type
  const preprocessingCode = ''
  let preprocessingImports = ''
  let applyPreprocessing = '    prefixed_texts = [prefix + t for t in texts]'

  if (preprocessing === 'consonantal') {
    preprocessingImports = `
# Consonantal preprocessing
VOWELS = set('aeiuāēīūâêîûàèìùáéíúAEIUÀÈÌÙÁÉÍÚÂÊÎÛ')
VOWEL_PATTERN = re.compile(r'[aeiuāēīūâêîûàèìùáéíúÀÈÌÙÁÉÍÚÂÊÎÛ]', re.IGNORECASE)
SUBSCRIPTS = '₀₁₂₃₄₅₆₇₈₉ₓ'

def is_logogram(token):
    clean = ''.join(c for c in token if c not in '.₀₁₂₃₄₅₆₇₈₉ₓ')
    return clean.isupper() and len(clean) > 0

def is_number(token):
    try:
        float(token.replace(',', '.'))
        return True
    except ValueError:
        return bool(re.match(r'^[\\d.,/]+$', token))

def strip_determinatives(token):
    return re.sub(r'^\\([a-zA-Z]{1,3}\\)', '', token)

def to_skeleton(text):
    if not text:
        return ""
    result = []
    for token in str(text).split():
        if is_logogram(token) or is_number(token):
            result.append(token)
            continue
        normalized = strip_determinatives(token)
        normalized = ''.join(c for c in normalized if c not in SUBSCRIPTS)
        skeleton = VOWEL_PATTERN.sub('', normalized)
        skeleton = re.sub(r'-+', '', skeleton)
        result.append(skeleton if skeleton else '_')
    return ' '.join(result)

def create_dual_input(text, mode='skeleton'):
    if not text:
        return ""
    normalized = to_skeleton(text)
    return f"[SKL:{normalized}] {text}"

print("Consonantal preprocessor loaded")
print(f"Example: 'šar-rum' -> '{create_dual_input('šar-rum')}'")
`
    applyPreprocessing = `    # Apply consonantal preprocessing
    prefixed_texts = [prefix + create_dual_input(t) for t in texts]
    print("Using consonantal skeleton preprocessing")`
  }

  // Generate model loading code based on type
  let modelLoadCode = ''
  let generateCode = ''

  if (modelType === 'nllb') {
    modelLoadCode = `from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print(f"\\nDownloading model from Kaggle: {CONFIG['model_handle']}")
start = time.time()
model_path = kagglehub.model_download(CONFIG["model_handle"])
print(f"Downloaded in {time.time() - start:.1f}s to: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16)

# Set language codes for NLLB
tokenizer.src_lang = "akk_Xsux"
tgt_lang_id = tokenizer.convert_tokens_to_ids("eng_Latn")

print(f"Model loaded, target lang ID: {tgt_lang_id}")`

    generateCode = `    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            num_beams=CONFIG["num_beams"],
            max_new_tokens=CONFIG["max_new_tokens"],
            early_stopping=True
        )`
  } else {
    // T5/ByT5 style
    modelLoadCode = `from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print(f"\\nDownloading model from Kaggle: {CONFIG['model_handle']}")
start = time.time()
model_path = kagglehub.model_download(CONFIG["model_handle"])
print(f"Downloaded in {time.time() - start:.1f}s to: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16)

print(f"Model loaded in {time.time() - start:.1f}s")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")`

    generateCode = `    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=CONFIG["num_beams"],
            max_new_tokens=CONFIG["max_new_tokens"],
            early_stopping=True
        )`
  }

  return `# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   kaggle:
#     accelerator: gpu
#     dataSources:
#       - type: model
#         name: ${modelHandle.split('/').slice(0, 2).join('/')}
#       - type: competition
#         name: ${competition}
#     docker_image: gcr.io/kaggle-gpu-images/python
#     isGpuEnabled: true
#     isInternetEnabled: false
#     language: python
#     sourceType: script
# ---

# %% [markdown]
# # ${name}
#
# Competition submission inference kernel.
#
# **Version**: ${version}
# **Model**: ${modelHandle}
# **Generated**: ${timestamp}
#
# ## Submission Requirements Met:
# - ✅ Internet: OFF (uses kagglehub for model loading)
# - ✅ Model: Loaded from Kaggle registry
# - ✅ Output: submission.csv

# %%
import os
import sys
import time
import re

# Suppress progress bars and warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*past_key_values.*")

# Disable tqdm globally
from functools import partialmethod
from tqdm import tqdm as tqdm_orig
tqdm_orig.__init__ = partialmethod(tqdm_orig.__init__, disable=True)

import torch
import pandas as pd
from tqdm import tqdm
import kagglehub

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
    "model_source": "kaggle",
    "model_handle": "${modelHandle}",
    "test_source": "/kaggle/input/${competition}/test.csv",
    "source_column": "transliteration",
    "id_column": "id",
    "output_column": "translation",
    "batch_size": 16,
    "num_beams": 4,
    "max_new_tokens": 256,
    "use_fp16": True,
}
print(f"Config: {CONFIG}")
${preprocessingImports}
# %% [markdown]
# ## Load Model

# %%
${modelLoadCode}

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
    """Clean up translation output - remove ORACC academic conventions."""
    text = re.sub(r'\\s*\\(\\s*s\\s*\\)\\s*', 's ', text)
    text = re.sub(r'\\s*\\(\\s*(or|and)\\s*\\)\\s*', r' \\1 ', text)
    text = re.sub(r'\\s*\\(\\s*(ki|d|m|f|lu|munus)\\s*\\)\\s*', ' ', text)
    text = ' '.join(text.split())
    return text.strip()

# %%
print("\\n" + "=" * 60)
print("GENERATING TRANSLATIONS")
print("=" * 60)

texts = test_df[CONFIG["source_column"]].tolist()
ids = test_df[CONFIG["id_column"]].tolist()

# Task prefix for T5/ByT5 models
prefix = "translate Akkadian to English: "

${applyPreprocessing}

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
        max_length=640
    ).to(device)

${generateCode}

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
print("\\n" + "=" * 60)
print("SAMPLE TRANSLATIONS")
print("=" * 60)
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
print(f"File size: {os.path.getsize('submission.csv'):,} bytes")

# %%
print("\\n" + "=" * 60)
print("INFERENCE COMPLETE")
print(f"Model: {CONFIG['model_handle']}")
print(f"Samples: {len(translations)}")
print(f"Time: {elapsed:.1f}s")
print("=" * 60)
`
}

/**
 * Generate kernel metadata
 */
function generateKernelMetadata(options: {
  owner: string
  name: string
  version: string
  codeFile: string
  modelHandle: string
  competition: string
}): Record<string, unknown> {
  const { owner, name, version, codeFile, modelHandle, competition } = options
  const slug = `${name.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-v${version.replace(/\./g, '-')}`

  // Extract model source (owner/model-name only)
  const modelParts = modelHandle.split('/')
  const modelSource = modelParts.length >= 2 ? `${modelParts[0]}/${modelParts[1]}` : modelHandle

  return {
    id: `${owner}/${slug}`,
    title: `${name} v${version}`,
    code_file: codeFile,
    language: 'python',
    kernel_type: 'script',
    is_private: false,
    enable_gpu: true,
    enable_tpu: false,
    enable_internet: false,
    dataset_sources: [],
    competition_sources: [competition],
    kernel_sources: [],
    model_sources: [modelSource],
  }
}

const CreateInferenceArgs = z.object({
  model: z.string().describe('Kaggle model handle (owner/model-name or full handle)'),
  name: z.string().optional().describe('Inference kernel name (default: derived from model)'),
  version: z.string().optional().describe('Version (default: extracted from model name)'),
  output: z.string().optional().describe('Output directory (default: notebooks/inference/)'),
  competition: z.string().optional().describe('Competition slug (default: from config)'),
  dryRun: z.boolean().default(false).describe('Preview without writing files'),
})

export const createInference: CommandDefinition<typeof CreateInferenceArgs> = {
  name: 'kaggle create-inference',
  description: 'Create inference kernel for competition submission',
  help: `
Generate a complete inference kernel from a trained model in Kaggle registry.

This command ensures all competition submission constraints are met:
- Internet: OFF (required for competition scoring)
- Model: Loaded via kagglehub.model_download()
- Output: submission.csv with correct format
- No pip installs or external downloads

The command auto-detects:
- Model type (ByT5, NLLB, T5, etc.)
- Preprocessing requirements (consonantal, annotated, etc.)
- Version from model name

Files generated:
- <name>_v<version>.py - The inference notebook
- kernel-metadata.json - Kaggle kernel configuration

After generation, push to Kaggle:
  kaggle kernels push -p <output-dir>
`,
  examples: [
    'akk kaggle create-inference --model manwithacat/akkadian-byt5-v1-0-10',
    'akk kaggle create-inference --model manwithacat/byt5-consonantal/transformers/default --name byt5-consonantal-inference',
    'akk kaggle create-inference --model manwithacat/akkadian-nllb-v1-2-0 --dry-run',
  ],
  args: CreateInferenceArgs,

  async run(args, ctx) {
    // Parse model handle
    let parsedModel
    try {
      parsedModel = parseModelHandle(args.model)
    } catch (err) {
      return error(
        'INVALID_MODEL',
        err instanceof Error ? err.message : 'Invalid model handle',
        'Format: owner/model-name[/framework/variation]',
        { model: args.model }
      )
    }

    // Detect model type and preprocessing
    const modelType = detectModelType(parsedModel.model) || 'byt5'
    const preprocessing = detectPreprocessing(parsedModel.model)

    // Extract version from model name if not provided
    const version = args.version || extractVersion(parsedModel.model)

    // Derive name if not provided
    const baseName = args.name || `${parsedModel.model}-inference`

    // Get competition from config or args
    let competition = args.competition
    if (!competition) {
      const compConfigPath = await findCompetitionConfig(ctx.cwd)
      if (compConfigPath) {
        const compConfig = await loadCompetitionConfig(compConfigPath)
        competition = compConfig?.competition?.slug
      }
    }
    if (!competition) {
      competition = 'deep-past-initiative-machine-translation'
    }

    // Determine output directory
    const outputDir = args.output || join(ctx.cwd, 'notebooks', 'inference', baseName.replace(/[^a-z0-9-]/gi, '_'))

    // Generate file names
    const pyFileName = `${baseName.toLowerCase().replace(/[^a-z0-9]+/g, '_')}_v${version.replace(/\./g, '_')}.py`
    const pyFilePath = join(outputDir, pyFileName)
    const metadataPath = join(outputDir, 'kernel-metadata.json')

    // Generate notebook content
    const notebook = generateInferenceNotebook({
      modelHandle: parsedModel.fullHandle,
      modelType,
      preprocessing,
      version,
      name: baseName,
      competition,
    })

    // Generate metadata
    const metadata = generateKernelMetadata({
      owner: parsedModel.owner,
      name: baseName,
      version,
      codeFile: pyFileName,
      modelHandle: parsedModel.fullHandle,
      competition,
    })

    if (args.dryRun) {
      return success({
        message: 'Dry run - files not written',
        model: {
          handle: parsedModel.fullHandle,
          type: modelType,
          preprocessing,
        },
        kernel: {
          name: baseName,
          version,
          slug: metadata.id,
        },
        output: {
          directory: outputDir,
          notebook: pyFileName,
          metadata: 'kernel-metadata.json',
        },
        constraints: {
          internet: 'OFF ✅',
          model_source: 'kagglehub ✅',
          submission_output: 'submission.csv ✅',
        },
        notebook_preview: notebook.slice(0, 800) + '\n...',
      })
    }

    // Create output directory
    if (!existsSync(outputDir)) {
      mkdirSync(outputDir, { recursive: true })
    }

    // Write files
    await Bun.write(pyFilePath, notebook)
    await Bun.write(metadataPath, JSON.stringify(metadata, null, 2))

    return success({
      message: 'Inference kernel created successfully',
      model: {
        handle: parsedModel.fullHandle,
        type: modelType,
        preprocessing: preprocessing || 'none',
      },
      kernel: {
        name: baseName,
        version,
        slug: metadata.id,
      },
      output: {
        directory: outputDir,
        notebook: pyFilePath,
        metadata: metadataPath,
      },
      constraints: {
        internet: 'OFF ✅',
        model_source: 'kagglehub ✅',
        submission_output: 'submission.csv ✅',
      },
      next_steps: [
        `Review the generated notebook: ${pyFilePath}`,
        `Push to Kaggle: kaggle kernels push -p ${outputDir}`,
        `Monitor: akk kaggle status ${metadata.id}`,
      ],
    })
  },
}

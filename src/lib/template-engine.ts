/**
 * Template Engine
 *
 * Parses and renders notebook templates with platform-specific injection.
 */

import { generateToolCell, generateToolDependenciesCell } from '../templates/tools'
import type { PlatformId } from '../types/platform'
import type {
  InjectionPoint,
  NotebookCell,
  NotebookContent,
  PlatformAdapter,
  PlatformPaths,
  TemplateContext,
  TemplateMetadata,
  TemplateRenderResult,
  TemplateVariable,
} from '../types/template'

// Embedded templates for bundled builds
const EMBEDDED_TEMPLATES: Record<string, string> = {
  training: `# @@TEMPLATE: training
# @@DESCRIPTION: Base training template for sequence-to-sequence translation models
# @@VERSION: 1.0.0
# @@PLATFORMS: kaggle-p100, kaggle-t4x2, colab-free, colab-pro, vertex-a100, runpod-a100

# --- INJECTION POINT: platform_setup ---
# Platform-specific setup will be injected here
# {{PLATFORM_SETUP}}
# --- END INJECTION POINT ---

# %% [markdown]
# # Training Configuration

# %%
# Training Configuration
CONFIG = {
    "model_name": "{{model_name}}",
    "src_lang": "{{src_lang}}",
    "tgt_lang": "{{tgt_lang}}",
    "num_epochs": {{num_epochs}},
    "batch_size": {{batch_size}},
    "gradient_accumulation_steps": {{gradient_accumulation_steps}},
    "learning_rate": {{learning_rate}},
    "max_src_len": {{max_src_len}},
    "max_tgt_len": {{max_tgt_len}},
    "warmup_ratio": {{warmup_ratio}},
    "weight_decay": {{weight_decay}},
    "fp16": {{fp16}},
    "save_steps": {{save_steps}},
    "eval_steps": {{eval_steps}},
    "logging_steps": {{logging_steps}},
    "save_total_limit": {{save_total_limit}},
}

print("Training Configuration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# %% [markdown]
# # Dependencies

# %%
# Install dependencies
import subprocess
import sys

def install_packages():
    packages = [
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "accelerate",
        "sentencepiece",
        "sacrebleu",
        "evaluate",
    ]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install_packages()

# --- INJECTION POINT: tools_setup ---
# ML tool integrations (sacrebleu, qlora, accelerate, onnx, streaming, hpo)
# {{TOOLS_SETUP}}
# --- END INJECTION POINT ---

# %%
# Imports
import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# --- INJECTION POINT: data_loading ---
# Platform-specific data loading will be injected here
# {{DATA_LOADING}}
# --- END INJECTION POINT ---

# %% [markdown]
# # Data Preprocessing

# %%
# Create train/val split if no validation set
if val_df is None:
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    print(f"Split: {len(train_df)} train, {len(val_df)} validation")

# Detect source and target columns
src_col = None
tgt_col = None
for col in train_df.columns:
    col_lower = col.lower()
    if 'source' in col_lower or 'src' in col_lower or 'input' in col_lower:
        src_col = col
    elif 'target' in col_lower or 'tgt' in col_lower or 'output' in col_lower or 'english' in col_lower:
        tgt_col = col

if src_col is None or tgt_col is None:
    # Fall back to first two columns
    src_col, tgt_col = train_df.columns[:2]

print(f"Source column: {src_col}")
print(f"Target column: {tgt_col}")

# Convert to HuggingFace datasets
train_dataset = Dataset.from_pandas(train_df[[src_col, tgt_col]].rename(
    columns={src_col: 'source', tgt_col: 'target'}
))
val_dataset = Dataset.from_pandas(val_df[[src_col, tgt_col]].rename(
    columns={src_col: 'source', tgt_col: 'target'}
))

print(f"Train dataset: {len(train_dataset)} samples")
print(f"Val dataset: {len(val_dataset)} samples")

# %% [markdown]
# # Model and Tokenizer

# %%
# Load model and tokenizer
print(f"Loading model: {CONFIG['model_name']}")

tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"])

# Set source/target languages for multilingual models
if hasattr(tokenizer, 'src_lang'):
    tokenizer.src_lang = CONFIG["src_lang"]
if hasattr(tokenizer, 'tgt_lang'):
    tokenizer.tgt_lang = CONFIG["tgt_lang"]

print(f"Model parameters: {model.num_parameters():,}")
print(f"Tokenizer vocab size: {tokenizer.vocab_size:,}")

# %%
# Preprocessing function
def preprocess_function(examples):
    inputs = examples["source"]
    targets = examples["target"]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=CONFIG["max_src_len"],
        truncation=True,
        padding=False,
    )

    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=CONFIG["max_tgt_len"],
            truncation=True,
            padding=False,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize datasets
print("Tokenizing datasets...")
train_tokenized = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing train",
)
val_tokenized = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names,
    desc="Tokenizing val",
)

# --- INJECTION POINT: checkpoint_save ---
# Platform-specific checkpoint saving will be injected here
# {{CHECKPOINT_SAVE}}
# --- END INJECTION POINT ---

# %% [markdown]
# # Training Setup

# %%
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8 if CONFIG["fp16"] else None,
)

# Metrics
bleu_metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Decode predictions
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute BLEU
    result = bleu_metric.compute(
        predictions=decoded_preds,
        references=[[ref] for ref in decoded_labels]
    )

    return {"bleu": result["score"]}

# %%
# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=CHECKPOINT_PATH,
    num_train_epochs=CONFIG["num_epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    warmup_ratio=CONFIG["warmup_ratio"],
    weight_decay=CONFIG["weight_decay"],
    fp16=CONFIG["fp16"],
    evaluation_strategy="steps",
    eval_steps=CONFIG["eval_steps"],
    save_strategy="steps",
    save_steps=CONFIG["save_steps"],
    save_total_limit=CONFIG["save_total_limit"],
    logging_steps=CONFIG["logging_steps"],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    predict_with_generate=False,  # Faster training without generation
    dataloader_num_workers=0,
    report_to="none",
)

# %%
# Create trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# %% [markdown]
# # Training

# %%
# Clear GPU cache before training
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Train
print("Starting training...")
train_result = trainer.train()

# Log results
print(f"\\nTraining completed!")
print(f"Training loss: {train_result.training_loss:.4f}")

# --- INJECTION POINT: output_save ---
# Platform-specific output saving will be injected here
# {{OUTPUT_SAVE}}
# --- END INJECTION POINT ---

# %% [markdown]
# # Evaluation

# %%
# Final evaluation with generation
print("\\nRunning final evaluation with BLEU...")

# Update args for generation
trainer.args.predict_with_generate = True
trainer.args.generation_max_length = CONFIG["max_tgt_len"]

eval_results = trainer.evaluate()
print(f"Final eval loss: {eval_results['eval_loss']:.4f}")
if 'eval_bleu' in eval_results:
    print(f"Final BLEU: {eval_results['eval_bleu']:.2f}")

# %%
# Save final model
print("\\nSaving final model...")
save_model_for_submission(model, tokenizer, output_name="final_model")

# Save training metrics
training_history = {
    "config": CONFIG,
    "train_loss": train_result.training_loss,
    "eval_results": eval_results,
    "train_samples": len(train_dataset),
    "val_samples": len(val_dataset),
}
save_training_log(training_history)

print("\\nTraining complete!")
`,
  inference: `# @@TEMPLATE: inference
# @@DESCRIPTION: Inference template for generating predictions
# @@VERSION: 1.0.0
# @@PLATFORMS: kaggle-p100, kaggle-t4x2, colab-free, colab-pro, vertex-a100, runpod-a100

# --- INJECTION POINT: platform_setup ---
# Platform-specific setup will be injected here
# --- END INJECTION POINT ---

# %% [markdown]
# # Inference Configuration

# %%
import os
import torch
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_PATH = "{{model_path}}"
SRC_LANG = "{{src_lang}}"
TGT_LANG = "{{tgt_lang}}"
MAX_LENGTH = {{max_length}}
BATCH_SIZE = {{batch_size}}

print(f"Loading model from: {MODEL_PATH}")

# %%
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

if torch.cuda.is_available():
    model = model.cuda()
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

model.eval()
print(f"Model loaded with {model.num_parameters():,} parameters")

# --- INJECTION POINT: data_loading ---
# Platform-specific data loading will be injected here
# --- END INJECTION POINT ---

# %%
# Generate predictions
def generate_translations(texts, batch_size=BATCH_SIZE):
    predictions = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_LENGTH,
                num_beams=4,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)

        if (i + batch_size) % 100 == 0:
            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)}")

    return predictions

# %%
# Run inference
print("Generating predictions...")
predictions = generate_translations(test_df["source"].tolist())

# --- INJECTION POINT: output_save ---
# Platform-specific output saving will be injected here
# --- END INJECTION POINT ---

# %%
# Save predictions
submission_df = pd.DataFrame({
    "id": test_df["id"] if "id" in test_df.columns else range(len(predictions)),
    "prediction": predictions
})

save_predictions(submission_df, "submission.csv")
print(f"Saved {len(predictions)} predictions")
`,
}

/**
 * Template marker patterns
 */
const MARKERS = {
  TEMPLATE: /^#\s*@@TEMPLATE:\s*(.+)$/m,
  DESCRIPTION: /^#\s*@@DESCRIPTION:\s*(.+)$/m,
  VERSION: /^#\s*@@VERSION:\s*(.+)$/m,
  PLATFORMS: /^#\s*@@PLATFORMS:\s*(.+)$/m,
  INJECTION_START: /^#\s*---\s*INJECTION POINT:\s*(\w+)\s*---$/,
  INJECTION_END: /^#\s*---\s*END INJECTION POINT\s*---$/,
  VARIABLE: /\{\{(\w+)\}\}/g,
  CONDITIONAL_START: /\{\{#if\s+(\w+)\}\}/,
  CONDITIONAL_END: /\{\{\/if\}\}/,
  CELL_MARKER: /^#\s*@@CELL:\s*(.+)$/m,
  CELL_PLATFORM: /^#\s*@@PLATFORM:\s*(.+)$/m,
}

/**
 * Parse template metadata from file header
 */
export function parseTemplateMetadata(content: string): TemplateMetadata {
  const templateMatch = content.match(MARKERS.TEMPLATE)
  const descMatch = content.match(MARKERS.DESCRIPTION)
  const versionMatch = content.match(MARKERS.VERSION)
  const platformsMatch = content.match(MARKERS.PLATFORMS)

  const name = templateMatch?.[1]?.trim() || 'unknown'
  const description = descMatch?.[1]?.trim() || ''
  const version = versionMatch?.[1]?.trim() || '1.0.0'
  const platforms = (platformsMatch?.[1]?.split(',').map((p) => p.trim()) as PlatformId[]) || []

  // Extract injection points
  const injectionPoints: InjectionPoint[] = []
  const lines = content.split('\n')
  for (let i = 0; i < lines.length; i++) {
    const match = lines[i].match(MARKERS.INJECTION_START)
    if (match) {
      injectionPoints.push({
        name: match[1],
        description: '',
        required: false,
      })
    }
  }

  // Extract variables from template
  const variables: TemplateVariable[] = []
  const variableMatches = content.matchAll(MARKERS.VARIABLE)
  const seenVars = new Set<string>()
  for (const match of variableMatches) {
    const varName = match[1]
    if (!seenVars.has(varName)) {
      seenVars.add(varName)
      variables.push({
        name: varName,
        type: 'string',
        required: true,
        description: '',
      })
    }
  }

  return {
    name,
    description,
    version,
    type: name.includes('train') ? 'training' : name.includes('infer') ? 'inference' : 'custom',
    platforms,
    variables,
    injection_points: injectionPoints,
  }
}

/**
 * Substitute variables in template content
 */
export function substituteVariables(content: string, variables: Record<string, unknown>): string {
  let result = content

  // Handle conditional blocks {{#if var}}...{{/if}}
  const conditionalPattern = /\{\{#if\s+(\w+)\}\}([\s\S]*?)\{\{\/if\}\}/g
  result = result.replace(conditionalPattern, (_, varName, block) => {
    const value = variables[varName]
    if (value) {
      return block
    }
    return ''
  })

  // Substitute simple variables {{var}}
  result = result.replace(MARKERS.VARIABLE, (match, varName) => {
    const value = variables[varName]
    if (value !== undefined) {
      return String(value)
    }
    return match // Keep placeholder if no value
  })

  return result
}

/**
 * Extract injection points and their content from template
 */
export function extractInjectionPoints(
  content: string
): Map<string, { start: number; end: number; placeholder: string }> {
  const points = new Map<string, { start: number; end: number; placeholder: string }>()
  const lines = content.split('\n')

  let currentPoint: string | null = null
  let startLine = 0
  let placeholderLines: string[] = []

  for (let i = 0; i < lines.length; i++) {
    const startMatch = lines[i].match(MARKERS.INJECTION_START)
    if (startMatch) {
      currentPoint = startMatch[1]
      startLine = i
      placeholderLines = []
      continue
    }

    if (currentPoint && MARKERS.INJECTION_END.test(lines[i])) {
      points.set(currentPoint, {
        start: startLine,
        end: i,
        placeholder: placeholderLines.join('\n'),
      })
      currentPoint = null
      continue
    }

    if (currentPoint) {
      placeholderLines.push(lines[i])
    }
  }

  return points
}

/**
 * Inject content into template at injection points
 */
export function injectContent(content: string, injections: Record<string, string>): string {
  const lines = content.split('\n')
  const result: string[] = []

  let currentPoint: string | null = null
  let skipUntilEnd = false

  for (let i = 0; i < lines.length; i++) {
    const startMatch = lines[i].match(MARKERS.INJECTION_START)
    if (startMatch) {
      currentPoint = startMatch[1]
      result.push(lines[i]) // Keep the marker

      // If we have injection content for this point, add it (empty string also counts)
      if (currentPoint in injections) {
        if (injections[currentPoint]) {
          result.push(injections[currentPoint])
        }
        skipUntilEnd = true
      }
      continue
    }

    if (MARKERS.INJECTION_END.test(lines[i])) {
      result.push(lines[i]) // Keep the end marker
      currentPoint = null
      skipUntilEnd = false
      continue
    }

    // Skip original placeholder content if we injected
    if (skipUntilEnd) {
      continue
    }

    result.push(lines[i])
  }

  return result.join('\n')
}

/**
 * Convert Python script to Jupyter notebook format
 */
export function scriptToNotebook(content: string): NotebookContent {
  const cells: NotebookCell[] = []

  // Split by cell markers or large code blocks
  const sections = content.split(/^# %%|^# In\[\d*\]:/m)

  for (const section of sections) {
    if (!section.trim()) continue

    // Check if it's a markdown section (starts with # followed by text, not code)
    const lines = section.trim().split('\n')
    const firstLine = lines[0]

    if (firstLine.startsWith('# ') && !firstLine.startsWith('# @') && lines.length === 1) {
      // Single line comment - could be markdown header
      cells.push({
        cell_type: 'markdown',
        source: [firstLine.replace(/^# /, '')],
        metadata: {},
      })
    } else {
      // Code cell
      cells.push({
        cell_type: 'code',
        source: lines.map((l, i) => (i < lines.length - 1 ? l + '\n' : l)),
        metadata: {},
        execution_count: null,
        outputs: [],
      })
    }
  }

  // If no cell markers, treat entire content as single code cell
  if (cells.length === 0) {
    cells.push({
      cell_type: 'code',
      source: content.split('\n').map((l, i, arr) => (i < arr.length - 1 ? l + '\n' : l)),
      metadata: {},
      execution_count: null,
      outputs: [],
    })
  }

  return {
    cells,
    metadata: {
      kernelspec: {
        display_name: 'Python 3',
        language: 'python',
        name: 'python3',
      },
      language_info: {
        name: 'python',
        version: '3.10.0',
      },
    },
    nbformat: 4,
    nbformat_minor: 5,
  }
}

/**
 * Template Engine class
 */
export class TemplateEngine {
  private adapters: Map<PlatformId, PlatformAdapter>

  constructor() {
    this.adapters = new Map()
  }

  /**
   * Register a platform adapter
   */
  registerAdapter(adapter: PlatformAdapter): void {
    this.adapters.set(adapter.id, adapter)
  }

  /**
   * Get registered adapter
   */
  getAdapter(platform: PlatformId): PlatformAdapter | undefined {
    return this.adapters.get(platform)
  }

  /**
   * List available templates (from embedded templates)
   */
  listTemplates(): TemplateMetadata[] {
    return Object.entries(EMBEDDED_TEMPLATES).map(([_name, content]) => {
      return parseTemplateMetadata(content)
    })
  }

  /**
   * Load a template by name (from embedded templates)
   */
  loadTemplate(name: string): { content: string; metadata: TemplateMetadata } | null {
    const content = EMBEDDED_TEMPLATES[name]
    if (!content) return null

    const metadata = parseTemplateMetadata(content)
    return { content, metadata }
  }

  /**
   * Render a template with context
   */
  render(templateName: string, ctx: TemplateContext): TemplateRenderResult {
    const template = this.loadTemplate(templateName)
    if (!template) {
      throw new Error(`Template not found: ${templateName}`)
    }

    const adapter = this.adapters.get(ctx.platform)
    const warnings: string[] = []
    const injections: Record<string, string> = {}

    // Generate platform-specific injections
    if (adapter) {
      injections.platform_setup = adapter.generateSetupCell(ctx)
      injections.data_loading = adapter.generateDataLoading(ctx)
      injections.checkpoint_save = adapter.generateCheckpointSave(ctx)
      injections.output_save = adapter.generateOutputSave(ctx)
    } else {
      warnings.push(`No adapter registered for platform: ${ctx.platform}`)
    }

    // Generate tool cells if enabled, or empty string to remove placeholder
    if (ctx.tools?.enabled && ctx.tools.enabled.length > 0) {
      const toolCells: string[] = []

      // Add dependencies cell first
      const depsCell = generateToolDependenciesCell(ctx.tools.enabled)
      if (depsCell) {
        toolCells.push(depsCell)
      }

      // Generate each tool cell
      for (const toolId of ctx.tools.enabled) {
        try {
          const cell = generateToolCell(toolId, ctx.tools.config)
          toolCells.push(cell)
        } catch (err) {
          warnings.push(
            `Failed to generate cell for tool ${toolId}: ${err instanceof Error ? err.message : 'Unknown error'}`
          )
        }
      }

      if (toolCells.length > 0) {
        injections.tools_setup = toolCells.join('\n\n')
      }
    } else {
      // No tools - inject empty string to remove the placeholder
      injections.tools_setup = ''
    }

    // Build variables from context
    const variables: Record<string, unknown> = {
      ...ctx.variables,
      // Model info
      model_name: ctx.model?.name || 'facebook/nllb-200-distilled-600M',
      model_path: ctx.model?.name || 'facebook/nllb-200-distilled-600M',
      // Languages
      src_lang: ctx.languages?.src || 'akk_Xsux',
      tgt_lang: ctx.languages?.tgt || 'eng_Latn',
      // Training params
      num_epochs: ctx.training?.epochs || 10,
      batch_size: ctx.training?.batch_size || 2,
      gradient_accumulation_steps: ctx.training?.gradient_accumulation_steps || 4,
      learning_rate: ctx.training?.learning_rate || 5e-5,
      max_src_len: ctx.training?.max_src_len || 128,
      max_tgt_len: ctx.training?.max_tgt_len || 128,
      max_length: ctx.training?.max_src_len || 128,
      warmup_ratio: ctx.training?.warmup_ratio || 0.1,
      weight_decay: ctx.training?.weight_decay || 0.01,
      fp16: ctx.training?.fp16 !== false ? 'True' : 'False',
      save_steps: ctx.training?.save_steps || 500,
      eval_steps: ctx.training?.eval_steps || 500,
      logging_steps: ctx.training?.logging_steps || 100,
      save_total_limit: ctx.training?.save_total_limit || 3,
    }

    // Substitute variables
    let content = substituteVariables(template.content, variables)

    // Inject platform-specific content
    content = injectContent(content, injections)

    // Convert to notebook
    const notebook = scriptToNotebook(content)

    // Get paths from adapter
    const paths: PlatformPaths = adapter?.getPaths(ctx) || {
      input: '/data',
      output: '/output',
      checkpoint: '/checkpoints',
      model_cache: '/cache',
    }

    // Get platform metadata (for Kaggle kernel-metadata.json)
    let platformMetadata: Record<string, unknown> | undefined
    if (adapter && 'generateMetadata' in adapter) {
      platformMetadata = (
        adapter as { generateMetadata(ctx: TemplateContext): Record<string, unknown> }
      ).generateMetadata(ctx)
    }

    return {
      script: content,
      notebook,
      paths,
      metadata: platformMetadata,
      warnings,
    }
  }

  /**
   * Render template directly to notebook JSON
   */
  renderToNotebook(templateName: string, ctx: TemplateContext): NotebookContent {
    const result = this.render(templateName, ctx)
    return result.notebook!
  }
}

/**
 * Create default template engine
 */
export function createTemplateEngine(): TemplateEngine {
  return new TemplateEngine()
}

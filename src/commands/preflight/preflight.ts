import { existsSync, readFileSync } from 'fs'
import { basename, dirname, extname, join } from 'path'
import { z } from 'zod'
import { error, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'
import { PLATFORMS, type PlatformProfile } from './platforms'

/**
 * Model size estimates (in GB for fp16)
 */
const MODEL_SIZES: Record<string, { params_b: number; size_gb: number; name: string }> = {
  // NLLB models
  'facebook/nllb-200-distilled-600M': {
    params_b: 0.6,
    size_gb: 1.2,
    name: 'NLLB-600M',
  },
  'facebook/nllb-200-1.3B': { params_b: 1.3, size_gb: 2.6, name: 'NLLB-1.3B' },
  'facebook/nllb-200-3.3B': { params_b: 3.3, size_gb: 6.6, name: 'NLLB-3.3B' },

  // T5 models
  'google-t5/t5-small': { params_b: 0.06, size_gb: 0.12, name: 'T5-Small' },
  'google-t5/t5-base': { params_b: 0.22, size_gb: 0.44, name: 'T5-Base' },
  'google-t5/t5-large': { params_b: 0.77, size_gb: 1.54, name: 'T5-Large' },

  // ByT5 models
  'google/byt5-small': { params_b: 0.3, size_gb: 0.6, name: 'ByT5-Small' },
  'google/byt5-base': { params_b: 0.58, size_gb: 1.16, name: 'ByT5-Base' },
  'google/byt5-large': { params_b: 1.2, size_gb: 2.4, name: 'ByT5-Large' },

  // mT5 models
  'google/mt5-small': { params_b: 0.3, size_gb: 0.6, name: 'mT5-Small' },
  'google/mt5-base': { params_b: 0.58, size_gb: 1.16, name: 'mT5-Base' },
}

/**
 * Extract configuration from notebook/script content
 */
interface ExtractedConfig {
  model_name?: string
  batch_size?: number
  gradient_accumulation_steps?: number
  max_src_len?: number
  max_tgt_len?: number
  num_epochs?: number
  fp16?: boolean
  save_total_limit?: number
  save_only_model?: boolean
  clear_hf_cache?: boolean
  dataloader_num_workers?: number
}

function extractConfig(content: string): ExtractedConfig {
  const config: ExtractedConfig = {}

  // Model name patterns
  const modelPatterns = [
    /["']model_name["']\s*:\s*["']([^"']+)["']/,
    /MODEL_NAME\s*=\s*["']([^"']+)["']/,
    /from_pretrained\s*\(\s*["']([^"']+)["']/,
  ]
  for (const pattern of modelPatterns) {
    const match = content.match(pattern)
    if (match) {
      config.model_name = match[1]
      break
    }
  }

  // Batch size
  const batchMatch = content.match(/["']?batch_size["']?\s*[=:]\s*(\d+)/)
  if (batchMatch) config.batch_size = parseInt(batchMatch[1], 10)

  // Gradient accumulation
  const gradAccumMatch = content.match(/["']?gradient_accumulation_steps["']?\s*[=:]\s*(\d+)/)
  if (gradAccumMatch) config.gradient_accumulation_steps = parseInt(gradAccumMatch[1], 10)

  // Max lengths
  const maxSrcMatch = content.match(/["']?max_src_len["']?\s*[=:]\s*(\d+)/)
  if (maxSrcMatch) config.max_src_len = parseInt(maxSrcMatch[1], 10)

  const maxTgtMatch = content.match(/["']?max_tgt_len["']?\s*[=:]\s*(\d+)/)
  if (maxTgtMatch) config.max_tgt_len = parseInt(maxTgtMatch[1], 10)

  // Epochs
  const epochsMatch = content.match(/["']?num_epochs["']?\s*[=:]\s*(\d+)/)
  if (epochsMatch) config.num_epochs = parseInt(epochsMatch[1], 10)

  // FP16 (handles both fp16=True and "fp16": True)
  config.fp16 = /["']?fp16["']?\s*[=:]\s*True/i.test(content)

  // Save total limit (handles both save_total_limit=1 and "save_total_limit": 1)
  const saveMatch = content.match(/["']?save_total_limit["']?\s*[=:]\s*(\d+)/)
  if (saveMatch) config.save_total_limit = parseInt(saveMatch[1], 10)

  // Save only model (skip optimizer states to save disk space)
  config.save_only_model = /["']?save_only_model["']?\s*[=:]\s*True/i.test(content)

  // Clear HF cache after model load (frees disk space)
  config.clear_hf_cache = /["']?clear_hf_cache["']?\s*[=:]\s*True/i.test(content)

  // Dataloader workers
  const workersMatch = content.match(/dataloader_num_workers\s*=\s*(\d+)/)
  if (workersMatch) config.dataloader_num_workers = parseInt(workersMatch[1], 10)

  return config
}

/**
 * Estimate GPU memory usage for training
 */
function estimateGpuMemory(config: ExtractedConfig): {
  peak_gb: number
  breakdown: Record<string, number>
} {
  const modelInfo = config.model_name ? MODEL_SIZES[config.model_name] : null
  const modelSize = modelInfo?.size_gb || 2.0 // Default 2GB

  const batchSize = config.batch_size || 8
  const seqLen = Math.max(config.max_src_len || 256, config.max_tgt_len || 256)
  const fp16 = config.fp16 !== false // Default to fp16

  // Memory breakdown (rough estimates)
  const breakdown: Record<string, number> = {}

  // Model weights
  breakdown.model_weights = fp16 ? modelSize : modelSize * 2

  // Optimizer states (Adam: 2x model size for momentum + variance)
  breakdown.optimizer_states = breakdown.model_weights * 2

  // Gradients
  breakdown.gradients = breakdown.model_weights

  // Activations (rough estimate based on batch size and sequence length)
  // This is highly variable but we use a heuristic
  const activationFactor = (batchSize * seqLen * seqLen) / (8 * 256 * 256)
  breakdown.activations = Math.min(modelSize * 2 * activationFactor, 16) // Cap at 16GB

  // KV cache and attention
  breakdown.attention_cache = batchSize * seqLen * 0.001 // ~1MB per token per batch

  // CUDA overhead
  breakdown.cuda_overhead = 0.5

  const peak_gb = Object.values(breakdown).reduce((a, b) => a + b, 0)

  return { peak_gb, breakdown }
}

/**
 * Estimate disk usage
 *
 * IMPORTANT: HuggingFace Trainer saves FULL checkpoints by default including:
 * - model weights (model.safetensors): fp32, ~2x fp16 model size
 * - optimizer states (optimizer.pt): ~2x model weights for Adam (momentum + variance)
 * - scheduler state, RNG states, trainer state: small (~1MB)
 *
 * Total checkpoint size ≈ 3x model weights in fp32 ≈ 6x fp16 model size
 *
 * Additionally, during checkpoint rotation (when save_total_limit > 1),
 * there can be up to (save_total_limit + 1) checkpoints on disk simultaneously
 * as the old checkpoint is deleted after the new one is saved.
 *
 * Empirical observation: NLLB-600M (1.2GB fp16):
 * - model.safetensors: 2.3 GB
 * - optimizer.pt: 4.6 GB (when disk space available)
 * - Total checkpoint: ~7 GB
 */
function estimateDiskUsage(config: ExtractedConfig): {
  total_gb: number
  breakdown: Record<string, number>
  peak_gb: number
  save_only_model: boolean
  clear_hf_cache: boolean
} {
  const modelInfo = config.model_name ? MODEL_SIZES[config.model_name] : null
  const modelSizeFp16 = modelInfo?.size_gb || 2.0
  const saveOnlyModel = config.save_only_model === true
  const clearHfCache = config.clear_hf_cache === true

  // Model weights in checkpoint (fp32 = 2x fp16)
  const modelWeightsFp32 = modelSizeFp16 * 2

  // Optimizer states in checkpoint: Adam saves momentum + variance (2x model weights)
  // If save_only_model=True, optimizer is NOT saved (significant disk savings!)
  const optimizerStateSize = saveOnlyModel ? 0 : modelWeightsFp32 * 2

  // Full checkpoint size: model + optimizer + small overhead
  const checkpointSize = modelWeightsFp32 + optimizerStateSize + 0.01

  const breakdown: Record<string, number> = {}

  // HuggingFace cache (model download - includes both pytorch and safetensors)
  // If clear_hf_cache=True, this is cleared after model load and doesn't count toward peak
  const hfCacheSize = modelSizeFp16 * 2.5
  breakdown.hf_cache = clearHfCache ? 0 : hfCacheSize
  breakdown.hf_cache_note = clearHfCache ? 0.01 : 0 // marker for cleared cache

  // Checkpoints at steady state (model + optimizer states)
  const saveLimit = config.save_total_limit || 3
  breakdown.checkpoints_model = modelWeightsFp32 * saveLimit
  breakdown.checkpoints_optimizer = optimizerStateSize * saveLimit

  // Peak checkpoint usage: during rotation, +1 checkpoint temporarily exists
  const peakCheckpoints = checkpointSize * (saveLimit + 1)

  // Final saved model (also fp32 by default from Trainer)
  breakdown.final_model = modelWeightsFp32

  // Tokenizer and configs
  breakdown.tokenizer_config = 0.1

  // Training logs, tensorboard, etc
  breakdown.logs = 0.05

  // Training outputs (submission.csv, etc)
  breakdown.outputs = 0.01

  const total_gb = Object.values(breakdown).reduce((a, b) => a + b, 0)

  // Peak includes the extra checkpoint during rotation
  const steadyCheckpoints = (modelWeightsFp32 + optimizerStateSize) * saveLimit
  const peak_gb = total_gb - steadyCheckpoints + peakCheckpoints

  return {
    total_gb,
    breakdown,
    peak_gb,
    save_only_model: saveOnlyModel,
    clear_hf_cache: clearHfCache,
  }
}

/**
 * Estimate training time
 *
 * Empirical calibration based on observed training times:
 * - NLLB-600M on P100: ~3 sec/step (batch=2, max_len=192)
 * - NLLB-1.3B on P100: ~6-8 sec/step (batch=2, max_len=192)
 * - NLLB-600M on T4: ~2 sec/step
 * - NLLB-600M on A100: ~0.5 sec/step
 */
function estimateTrainingTime(
  config: ExtractedConfig,
  platform: PlatformProfile,
  numSamples: number = 1500
): { hours: number; breakdown: Record<string, number> } {
  const batchSize = config.batch_size || 8
  const gradAccum = config.gradient_accumulation_steps || 1
  const epochs = config.num_epochs || 10

  const effectiveBatch = batchSize * gradAccum
  const stepsPerEpoch = Math.ceil(numSamples / effectiveBatch)
  const totalSteps = stepsPerEpoch * epochs

  // Get model size for calibration
  const modelInfo = config.model_name ? MODEL_SIZES[config.model_name] : null
  const modelParams = modelInfo?.params_b || 0.6 // Default to 600M

  // Empirically calibrated seconds per step
  // Base: P100 with 600M model at batch=2, max_len=192 ≈ 3 sec/step
  const baseSecondsPerStep = 3.0

  // Scale by model size (approximately linear with parameters)
  const modelScale = modelParams / 0.6

  // Scale by GPU performance relative to P100 (10.6 TFLOPS)
  const gpuScale = 10.6 / Math.max(platform.gpu.fp16_tflops, 1)

  // Scale by batch size (larger batch = proportionally more time, but less overhead)
  const batchScale = Math.sqrt(batchSize / 2)

  // Scale by sequence length (quadratic attention, but bounded)
  const seqLen = Math.max(config.max_src_len || 192, config.max_tgt_len || 192)
  const seqScale = (seqLen / 192) ** 1.5

  const secondsPerStep = baseSecondsPerStep * modelScale * gpuScale * batchScale * seqScale

  const breakdown: Record<string, number> = {}
  breakdown.training = (totalSteps * secondsPerStep) / 3600
  breakdown.evaluation = ((totalSteps / 100) * 30) / 3600 // ~30 sec per eval
  breakdown.model_loading = 0.1 // 6 minutes for model download/load
  breakdown.bleu_computation = 0.1 // 6 minutes for final BLEU

  const hours = Object.values(breakdown).reduce((a, b) => a + b, 0)

  return { hours, breakdown }
}

interface CheckResult {
  check: string
  status: 'pass' | 'warn' | 'fail'
  message: string
  details?: Record<string, unknown>
}

/**
 * Check for deprecated API usage that will cause runtime errors
 */
interface DeprecationIssue {
  pattern: RegExp
  message: string
  fix: string
  severity: 'error' | 'warning'
}

const DEPRECATION_CHECKS: DeprecationIssue[] = [
  {
    pattern: /evaluation_strategy\s*=/,
    message: 'evaluation_strategy is deprecated in transformers>=4.46',
    fix: 'Use eval_strategy instead',
    severity: 'error',
  },
  {
    pattern: /\.as_target_tokenizer\s*\(/,
    message: 'as_target_tokenizer() is deprecated in transformers>=4.40',
    fix: 'Use tokenizer(text, text_target=target) instead',
    severity: 'warning',
  },
  {
    pattern: /from_pretrained\([^)]*use_auth_token\s*=/,
    message: 'use_auth_token is deprecated in transformers>=4.35',
    fix: 'Use token= instead of use_auth_token=',
    severity: 'warning',
  },
]

function checkDeprecations(content: string): CheckResult[] {
  const results: CheckResult[] = []

  // Special check for Seq2SeqTrainer tokenizer deprecation
  // Look for trainer instantiation with tokenizer= but not processing_class=
  const hasTrainerInstantiation = /=\s*Seq2SeqTrainer\s*\(/.test(content)
  const hasOldTokenizerArg =
    hasTrainerInstantiation &&
    /Seq2SeqTrainer\s*\([\s\S]*?\btokenizer\s*=\s*tokenizer/.test(content) &&
    !/Seq2SeqTrainer\s*\([\s\S]*?processing_class\s*=/.test(content)

  if (hasOldTokenizerArg) {
    results.push({
      check: 'Deprecated API',
      status: 'warn',
      message: 'tokenizer= is deprecated in Seq2SeqTrainer (transformers>=4.46)',
      details: {
        fix: 'Use processing_class=tokenizer instead of tokenizer=tokenizer',
      },
    })
  }

  for (const check of DEPRECATION_CHECKS) {
    if (check.pattern.test(content)) {
      results.push({
        check: 'Deprecated API',
        status: check.severity === 'error' ? 'fail' : 'warn',
        message: check.message,
        details: { fix: check.fix },
      })
    }
  }

  return results
}

/**
 * Check if notebook produces a valid competition submission
 *
 * For Kaggle translation competitions, the submission must:
 * 1. Create a file named submission.csv
 * 2. Have columns: id, translation
 * 3. Write to the correct output path (/kaggle/working/)
 */
function checkSubmissionOutput(content: string): CheckResult[] {
  const results: CheckResult[] = []

  // Check for submission.csv creation
  const submissionPatterns = [
    /\.to_csv\s*\(\s*["'].*submission\.csv["']/i,
    /\.to_csv\s*\(\s*["']\/kaggle\/working\/submission\.csv["']/i,
    /open\s*\(\s*["'].*submission\.csv["']/i,
    /submission.*\.to_csv/i,
    /pd\.DataFrame.*\.to_csv.*submission/i,
  ]

  const hasSubmissionWrite = submissionPatterns.some((p) => p.test(content))

  if (!hasSubmissionWrite) {
    results.push({
      check: 'Submission Output',
      status: 'fail',
      message: 'No submission.csv output detected',
      details: {
        fix: "Add: submission_df.to_csv('/kaggle/working/submission.csv', index=False)",
        expected_columns: ['id', 'translation'],
        note: 'Competition notebooks must output submission.csv with id and translation columns',
      },
    })
  } else {
    // Check for correct column names
    const hasIdColumn = /["']id["']/.test(content) || /\bid\b\s*[=:]/.test(content)
    const hasTranslationColumn = /["']translation["']/.test(content) || /translation\s*[=:]/.test(content)

    if (!hasIdColumn || !hasTranslationColumn) {
      results.push({
        check: 'Submission Format',
        status: 'warn',
        message: 'Submission may be missing required columns (id, translation)',
        details: {
          detected_id_column: hasIdColumn,
          detected_translation_column: hasTranslationColumn,
          required: ['id', 'translation'],
        },
      })
    } else {
      results.push({
        check: 'Submission Output',
        status: 'pass',
        message: 'submission.csv output with correct columns detected',
      })
    }
  }

  // Check for index=False (common mistake)
  if (hasSubmissionWrite && !/index\s*=\s*False/i.test(content)) {
    results.push({
      check: 'CSV Index',
      status: 'warn',
      message: 'to_csv() may include unwanted index column',
      details: {
        fix: "Add index=False: df.to_csv('submission.csv', index=False)",
      },
    })
  }

  return results
}

/**
 * Check for internet enabled with submission generation
 *
 * CRITICAL: Competition submission kernels are run by Kaggle with internet DISABLED.
 * If enable_internet=true in kernel-metadata.json but the notebook generates submission.csv,
 * the kernel will fail during Kaggle's scoring run.
 *
 * This is a common mistake: training with internet enabled works fine locally,
 * but fails when Kaggle runs it for scoring.
 */
function checkSubmissionInternetMismatch(content: string, metadataPath?: string): CheckResult[] {
  const results: CheckResult[] = []

  // Check if notebook produces submission.csv
  const submissionPatterns = [/\.to_csv\s*\(\s*["'].*submission\.csv["']/i, /submission.*\.to_csv/i]
  const hasSubmissionOutput = submissionPatterns.some((p) => p.test(content))

  if (!hasSubmissionOutput) {
    return results // Not a submission kernel, no check needed
  }

  // Read kernel-metadata.json to check internet setting
  if (!metadataPath || !existsSync(metadataPath)) {
    results.push({
      check: 'Submission Internet',
      status: 'warn',
      message: 'No kernel-metadata.json found - cannot verify enable_internet setting',
      details: {
        note: 'Competition submission kernels MUST have enable_internet: false',
        fix: 'Generate metadata with: akk notebook build training.toml',
      },
    })
    return results
  }

  try {
    const metadata = JSON.parse(readFileSync(metadataPath, 'utf-8'))
    const internetEnabled = metadata.enable_internet === true

    if (internetEnabled) {
      results.push({
        check: 'Submission Internet',
        status: 'fail',
        message: 'enable_internet=true but notebook generates submission.csv - will FAIL on Kaggle scoring',
        details: {
          current: 'enable_internet: true',
          required: 'enable_internet: false',
          metadata_file: metadataPath,
          fix: 'Edit kernel-metadata.json and set "enable_internet": false',
          reason:
            'Kaggle runs scoring kernels with internet disabled. Any pip install, evaluate.load(), or HuggingFace model download will fail.',
          regenerate:
            'Or regenerate with: akk notebook build training.toml (submission.enabled=true auto-sets internet=false)',
        },
      })
    } else {
      results.push({
        check: 'Submission Internet',
        status: 'pass',
        message: 'enable_internet=false for submission kernel (correct)',
      })
    }
  } catch {
    results.push({
      check: 'Submission Internet',
      status: 'warn',
      message: 'Failed to parse kernel-metadata.json',
      details: {
        metadata_file: metadataPath,
      },
    })
  }

  return results
}

/**
 * Check if notebook uploads trained model to Kaggle Model Registry
 *
 * For training notebooks, models should be uploaded to Kaggle's registry using:
 * - kagglehub.model_upload() for direct upload
 * - Or saved as notebook output for manual publishing
 */
function checkModelRegistry(content: string): CheckResult[] {
  const results: CheckResult[] = []

  // Check for kagglehub model upload
  const modelUploadPatterns = [/kagglehub\.model_upload\s*\(/i, /model_upload\s*\(\s*handle\s*=/i]

  const hasModelUpload = modelUploadPatterns.some((p) => p.test(content))

  // Check for model save (trainer.save_model, model.save_pretrained, etc.)
  const modelSavePatterns = [
    /trainer\.save_model\s*\(/i,
    /model\.save_pretrained\s*\(/i,
    /\.save_model\s*\(/i,
    /torch\.save\s*\(/i,
  ]

  const hasModelSave = modelSavePatterns.some((p) => p.test(content))

  if (!hasModelSave) {
    results.push({
      check: 'Model Save',
      status: 'fail',
      message: 'No model save detected in training notebook',
      details: {
        fix: 'Add: trainer.save_model(output_dir) or model.save_pretrained(output_dir)',
      },
    })
  } else if (!hasModelUpload) {
    results.push({
      check: 'Kaggle Model Registry',
      status: 'warn',
      message: 'Model is saved but not uploaded to Kaggle Model Registry',
      details: {
        fix: 'Add kagglehub.model_upload() to upload trained model to Kaggle registry',
        example: `
import kagglehub
kagglehub.model_upload(
    handle="username/model-name/transformers/v1",
    local_model_dir=output_dir,
    version_notes="Training run description",
    license_name="Apache 2.0",
)`,
        note: 'Models in registry can be easily used in inference kernels',
        docs: 'https://github.com/Kaggle/kagglehub',
      },
    })
  } else {
    results.push({
      check: 'Kaggle Model Registry',
      status: 'pass',
      message: 'Model upload to Kaggle registry detected',
    })
  }

  return results
}

/**
 * Check if progress bars are disabled for clean Kaggle logs
 *
 * TQDM and HuggingFace progress bars create messy logs in Kaggle kernels.
 * Multiple approaches can disable progress bars:
 * 1. Environment variables (TQDM_DISABLE, HF_HUB_DISABLE_PROGRESS_BARS)
 * 2. partialmethod patching of tqdm.__init__
 * 3. disable=True on individual tqdm loops
 * 4. report_to="none" in HuggingFace TrainingArguments
 */
function checkProgressBarsDisabled(content: string): CheckResult[] {
  const results: CheckResult[] = []
  const missing: string[] = []
  const warnings: string[] = []

  // Check for TQDM_DISABLE env var
  const hasTqdmDisable = /os\.environ\s*\[\s*["']TQDM_DISABLE["']\s*\]\s*=\s*["']1["']/.test(content)

  // Check for HF_HUB_DISABLE_PROGRESS_BARS env var
  const hasHfDisable = /os\.environ\s*\[\s*["']HF_HUB_DISABLE_PROGRESS_BARS["']\s*\]\s*=\s*["']1["']/.test(content)

  // Check for partialmethod tqdm patching (more robust than env var)
  const hasPartialmethodPatch = /partialmethod\s*\(\s*tqdm.*__init__.*disable\s*=\s*True\s*\)/.test(content)

  // Check for Trainer's report_to="none" (disables HF progress logging)
  const hasReportToNone = /report_to\s*=\s*["']none["']/.test(content)

  // Check for disable_tqdm in Trainer
  const hasDisableTqdm = /disable_tqdm\s*=\s*True/.test(content)

  // Check for tqdm usage without disable=True
  const tqdmLoopMatches = content.match(/for\s+\w+\s+in\s+tqdm\s*\([^)]+\)/g) || []
  const tqdmLoopsWithoutDisable = tqdmLoopMatches.filter(
    (match) => !match.includes('disable=') && !match.includes('disable =')
  )

  // Determine if this is a training script (has Trainer/TrainingArguments)
  const isTrainingScript = /Seq2SeqTrainer|Trainer\s*\(|TrainingArguments/.test(content)

  // Build result based on what's missing
  if (!hasTqdmDisable && !hasPartialmethodPatch) {
    missing.push('TQDM_DISABLE')
  }

  if (!hasHfDisable) {
    missing.push('HF_HUB_DISABLE_PROGRESS_BARS')
  }

  // For training scripts, check Trainer progress bar settings
  if (isTrainingScript && !hasReportToNone && !hasDisableTqdm) {
    warnings.push('HuggingFace Trainer may show progress bars. Add report_to="none" to TrainingArguments')
  }

  // Warn about tqdm loops without explicit disable
  if (tqdmLoopsWithoutDisable.length > 0 && !hasTqdmDisable && !hasPartialmethodPatch) {
    warnings.push(
      `Found ${tqdmLoopsWithoutDisable.length} tqdm loop(s) without disable=True. Consider adding disable=True or using the partialmethod patch.`
    )
  }

  // Generate result
  if (missing.length === 0 && warnings.length === 0) {
    results.push({
      check: 'Progress Bars',
      status: 'pass',
      message: 'Progress bars disabled for clean Kaggle logs',
    })
  } else if (missing.length > 0) {
    const fixes = []
    if (missing.includes('TQDM_DISABLE')) {
      fixes.push('os.environ["TQDM_DISABLE"] = "1"')
    }
    if (missing.includes('HF_HUB_DISABLE_PROGRESS_BARS')) {
      fixes.push('os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"')
    }

    results.push({
      check: 'Progress Bars',
      status: 'warn',
      message: 'Progress bars not fully disabled - Kaggle logs will be messy',
      details: {
        fix: `Add at the start of your notebook:\n${fixes.join('\n')}`,
        missing,
        alternatives: [
          'Use partialmethod: tqdm_orig.__init__ = partialmethod(tqdm_orig.__init__, disable=True)',
          'Add disable=True to each tqdm() call',
        ],
        warnings: warnings.length > 0 ? warnings : undefined,
      },
    })
  } else if (warnings.length > 0) {
    results.push({
      check: 'Progress Bars',
      status: 'warn',
      message: 'Progress bars mostly disabled but some sources remain',
      details: {
        warnings,
      },
    })
  }

  return results
}

/**
 * Known dataset schemas for column validation
 * Maps Kaggle dataset slugs to their known column names
 */
const DATASET_SCHEMAS: Record<string, { columns: string[]; description: string }> = {
  'mtm24-akkadian-transliterated-20k': {
    columns: ['transliteration', 'target', 'original_cuneiform'],
    description: 'MTM24 Akkadian transliterated subset',
  },
  'mtm24-akkadian-transliterated': {
    columns: ['transliteration', 'target', 'original_cuneiform'],
    description: 'Full MTM24 Akkadian transliterated corpus',
  },
  'oracc-akkadian-english-parallel-corpus': {
    columns: ['akkadian', 'english', 'source_file'],
    description: 'ORACC Akkadian-English parallel corpus',
  },
  'deep-past-initiative-machine-translation': {
    columns: ['id', 'transliteration', 'translation'],
    description: 'Deep Past Initiative competition data',
  },
}

/**
 * Check for dataset column compatibility issues
 *
 * Training notebooks typically expect 'source' and 'target' columns,
 * but datasets may have different column names (akkadian/english, akk/eng, etc.)
 * This check ensures column mapping is present when needed.
 */
function checkDatasetColumns(content: string): CheckResult[] {
  const results: CheckResult[] = []

  // Extract column references from DataFrame access patterns
  // Be specific to avoid false positives from config dicts, env vars, etc.
  const dataframeNames = /(?:df|train_df|val_df|test_df|data|dataset|examples|sample|batch)\b/
  const columnAccessPatterns = [
    // DataFrame column access: df['col'], train_df["col"]
    new RegExp(`(?:${dataframeNames.source})\\s*\\[\\s*["'](\\w+)["']\\s*\\]`, 'g'),
    // DataFrame attribute column: df.col.str., df.col.isna()
    new RegExp(`(?:${dataframeNames.source})\\.(\\w+)\\.(?:str|isna|fillna)`, 'g'),
    // dropna/fillna subset: .dropna(subset=['col1', 'col2'])
    /\.(?:dropna|fillna)\s*\(\s*subset\s*=\s*\[([^\]]+)\]/g,
    // Multi-column selection: df[['col1', 'col2']]
    new RegExp(`(?:${dataframeNames.source})\\s*\\[\\s*\\[([^\\]]+)\\]\\s*\\]`, 'g'),
    // Dataset.from_pandas: Dataset.from_pandas(df[['col1', 'col2']])
    /Dataset\.from_pandas\s*\([^)]*\[\s*\[([^\]]+)\]/g,
  ]

  const referencedColumns = new Set<string>()
  for (const pattern of columnAccessPatterns) {
    let match = pattern.exec(content)
    while (match !== null) {
      const captured = match[1]
      // Handle multi-column matches (comma-separated in brackets)
      if (captured.includes(',') || captured.includes("'")) {
        // Extract individual column names from list like "'col1', 'col2'"
        const cols = captured.match(/["'](\w+)["']/g)
        if (cols) {
          for (const col of cols) {
            const name = col.replace(/["']/g, '')
            referencedColumns.add(name)
          }
        }
      } else {
        // Single column name
        const col = captured.trim()
        if (col && !['iloc', 'loc', 'values', 'index', 'columns', 'head', 'tail', 'shape'].includes(col)) {
          referencedColumns.add(col)
        }
      }
      match = pattern.exec(content)
    }
  }

  // Detect which datasets are being used
  const detectedDatasets: string[] = []
  for (const datasetSlug of Object.keys(DATASET_SCHEMAS)) {
    if (content.includes(datasetSlug)) {
      detectedDatasets.push(datasetSlug)
    }
  }

  // Validate column references against detected datasets
  for (const datasetSlug of detectedDatasets) {
    const schema = DATASET_SCHEMAS[datasetSlug]
    const invalidColumns: string[] = []

    for (const col of referencedColumns) {
      // Check if this column is used with this dataset's data
      // Look for patterns like: df['column'] where df is loaded from this dataset
      const isDatasetColumn =
        schema.columns.includes(col) ||
        // Allow common derived columns
        ['id', 'source', 'target', 'text', 'label', 'input', 'output'].includes(col)

      if (!isDatasetColumn && referencedColumns.has(col)) {
        // Check if this column is specifically accessed on data from this dataset
        // by looking for the column reference near the dataset reference
        const datasetIndex = content.indexOf(datasetSlug)
        const columnPattern = new RegExp(`\\[["']${col}["']\\]`)
        const columnMatch = content.match(columnPattern)

        if (columnMatch && datasetIndex !== -1) {
          // Simple heuristic: if column is referenced and dataset is used, validate
          if (!schema.columns.includes(col)) {
            invalidColumns.push(col)
          }
        }
      }
    }

    // Report invalid column references
    if (invalidColumns.length > 0) {
      results.push({
        check: 'Dataset Schema',
        status: 'fail',
        message: `Column(s) not found in ${datasetSlug}: ${invalidColumns.join(', ')}`,
        details: {
          dataset: datasetSlug,
          invalid_columns: invalidColumns,
          valid_columns: schema.columns,
          fix: `Check your TOML config - the dataset has columns: [${schema.columns.join(', ')}]`,
          common_issue: "MTM24 uses 'target' not 'translation', ORACC uses 'akkadian'/'english' not 'source'/'target'",
        },
      })
    }
  }

  // Check if code expects 'source' and 'target' columns
  const expectsSourceColumn = /\[["']source["']\]/.test(content)
  const expectsTargetColumn = /\[["']target["']\]/.test(content)

  // Check if code expects 'akkadian' and 'english' columns
  const expectsAkkadianColumn = /\[["']akkadian["']\]/.test(content)
  const expectsEnglishColumn = /\[["']english["']\]/.test(content)

  // Check if there's column renaming/mapping
  const hasColumnRename = /\.rename\s*\(\s*columns\s*=/.test(content)
  const hasColumnMapping = /column_mapping\s*=/.test(content)
  const hasColumnNormalization = hasColumnRename || hasColumnMapping

  // Check for dataset sources referencing our datasets
  const usesOraccDataset = /oracc-akkadian-english-parallel-corpus/.test(content)
  const _usesCompetitionData = /deep-past-initiative-machine-translation/.test(content)

  // Issue 1: Code expects source/target but uses our dataset without column mapping
  if (usesOraccDataset && expectsSourceColumn && !hasColumnNormalization && !expectsAkkadianColumn) {
    results.push({
      check: 'Dataset Columns',
      status: 'fail',
      message: 'Code expects source/target columns but ORACC dataset has akkadian/english',
      details: {
        fix: 'Add column mapping after loading CSV:\ncolumn_mapping = {"akkadian": "source", "english": "target"}\ndf = df.rename(columns=column_mapping)',
        dataset: 'oracc-akkadian-english-parallel-corpus',
        expected_columns: ['source', 'target'],
        actual_columns: ['akkadian', 'english'],
      },
    })
  }

  // Issue 2: Using competition data (which has sample_submission format) for training
  // Competition data has 'id' and 'translation' columns, not source/target
  const usesWildcardCompetition = /\/kaggle\/input\/deep-past-initiative-machine-translation\/\*\.csv/.test(content)
  if (usesWildcardCompetition && (expectsSourceColumn || expectsTargetColumn)) {
    results.push({
      check: 'Dataset Source',
      status: 'warn',
      message: 'Wildcard *.csv in competition dir may match sample_submission.csv (wrong format)',
      details: {
        fix: 'Use specific file path like train.csv instead of *.csv, or prioritize your curated dataset first',
        current: '/kaggle/input/deep-past-initiative-machine-translation/*.csv',
        suggestion: '/kaggle/input/deep-past-initiative-machine-translation/train.csv',
      },
    })
  }

  // Pass if using ORACC dataset with column normalization
  if (usesOraccDataset && hasColumnNormalization) {
    results.push({
      check: 'Dataset Columns',
      status: 'pass',
      message: 'Dataset column mapping detected',
    })
  }

  // Pass if directly using columns that match the dataset
  if (usesOraccDataset && expectsAkkadianColumn && expectsEnglishColumn) {
    results.push({
      check: 'Dataset Columns',
      status: 'pass',
      message: 'Code uses akkadian/english columns matching ORACC dataset',
    })
  }

  // Pass message if all detected datasets have valid column usage
  if (detectedDatasets.length > 0 && results.filter((r) => r.status === 'fail').length === 0) {
    results.push({
      check: 'Dataset Schema',
      status: 'pass',
      message: `Column references validated against ${detectedDatasets.length} dataset(s)`,
    })
  }

  return results
}

/**
 * Known packages pre-installed on Kaggle (as of Dec 2024)
 * These don't require pip install or internet access
 *
 * NOTE: sacrebleu and evaluate are NOT pre-installed!
 * They require pip install which fails with enable_internet=false
 */
const KAGGLE_PREINSTALLED = [
  'transformers',
  'torch',
  'tensorflow',
  'numpy',
  'pandas',
  'scikit-learn',
  'matplotlib',
  'seaborn',
  'nltk',
  // 'sacrebleu',   // NOT pre-installed - requires pip install
  // 'evaluate',    // NOT pre-installed - requires pip install
  'datasets',
  'accelerate',
  'sentencepiece',
  'tokenizers',
  'kagglehub',
  'huggingface_hub',
]

/**
 * Metrics available via evaluate that DON'T require internet download
 * because sacrebleu is pre-installed on Kaggle
 */
const _EVALUATE_METRICS_NEEDING_DOWNLOAD: Record<string, string[]> = {
  // These metrics download from HuggingFace hub
  chrf: ['sacrebleu'], // sacrebleu is pre-installed, but evaluate.load downloads
  bleu: ['sacrebleu'],
  sacrebleu: ['sacrebleu'],
  rouge: ['rouge_score'],
  meteor: ['nltk'],
  bertscore: ['bert_score'],
}

/**
 * Check for pip install with --no-deps flag
 *
 * Using --no-deps skips transitive dependencies which can cause
 * ModuleNotFoundError at runtime. For example, sacrebleu requires
 * portalocker, but --no-deps won't install it.
 *
 * Matches both shell syntax (!pip install) and subprocess syntax.
 */
function checkPipNoDeps(content: string): CheckResult[] {
  const results: CheckResult[] = []

  // Match various patterns of pip install with --no-deps:
  // 1. Shell syntax: !pip install --no-deps
  // 2. subprocess: subprocess.run([..."pip", "install"..."--no-deps"...])
  // 3. String form: "pip install --no-deps"
  const patterns = [
    /!pip\s+install[^\n]*--no-deps[^\n]*/g, // Shell syntax
    /pip[",\s]+install[^)\]]*--no-deps[^)\]]*/g, // subprocess or string
    /["']--no-deps["'][,\s]*\][^\n]*pip/g, // --no-deps in list before pip
  ]

  const matches: string[] = []
  for (const pattern of patterns) {
    const found = content.match(pattern) || []
    matches.push(...found)
  }

  // Also check for the specific subprocess pattern with --no-deps in array
  if (/subprocess\.run\s*\([^)]*["']--no-deps["']/.test(content)) {
    matches.push('subprocess.run with --no-deps')
  }

  // Check for pip install with --no-deps anywhere on the same logical line
  if (/pip.*install.*--no-deps|--no-deps.*pip.*install/.test(content)) {
    if (matches.length === 0) {
      matches.push('pip install --no-deps detected')
    }
  }

  if (matches.length > 0) {
    results.push({
      check: 'Pip No-Deps',
      status: 'fail',
      message: '--no-deps skips transitive dependencies, causing ModuleNotFoundError',
      details: {
        detected: matches.slice(0, 3), // Limit to first 3 matches
        fix: 'Remove --no-deps flag to install all dependencies, or explicitly add missing deps',
        example_error: "sacrebleu requires portalocker, but --no-deps won't install it",
        recommendation: 'Use: pip install -q <packages>  (without --no-deps)',
      },
    })
  }

  return results
}

/**
 * Check for operations that require internet access
 *
 * When enable_internet=false in kernel-metadata.json:
 * - pip install will fail
 * - evaluate.load() will fail (downloads from HuggingFace)
 * - HuggingFace model downloads will fail (use Kaggle Models instead)
 *
 * This check warns about these issues before pushing to Kaggle.
 */
function checkInternetDependencies(content: string, metadataPath?: string): CheckResult[] {
  const results: CheckResult[] = []

  // Try to read kernel-metadata.json to check internet setting
  let internetEnabled = true // Default to true (most permissive)
  let hasMetadata = false

  if (metadataPath && existsSync(metadataPath)) {
    try {
      const metadata = JSON.parse(readFileSync(metadataPath, 'utf-8'))
      internetEnabled = metadata.enable_internet !== false
      hasMetadata = true
    } catch {
      // Ignore parse errors
    }
  }

  // If internet is enabled, no concerns
  if (internetEnabled && hasMetadata) {
    return results
  }

  // Check for pip install commands (both shell-style and subprocess-style)
  const shellPipInstalls = content.match(/!pip\s+install\s+[^\n]+/g) || []
  const subprocessPipInstalls = content.match(/subprocess\.run\s*\([^)]*pip[^)]*install[^)]*\)/g) || []
  const allPipInstalls = [...shellPipInstalls, ...subprocessPipInstalls]

  // If internet is disabled and ANY pip install is detected, it will fail
  if (!internetEnabled && allPipInstalls.length > 0) {
    results.push({
      check: 'Pip Install Blocked',
      status: 'fail',
      message: 'pip install will fail with enable_internet=false',
      details: {
        detected: allPipInstalls.slice(0, 3).map((s) => s.slice(0, 80) + '...'),
        fix: 'Remove pip install commands - use Kaggle pre-installed packages instead',
        preinstalled: KAGGLE_PREINSTALLED.slice(0, 10),
        reason: 'Competition submission kernels run with internet disabled. pip install cannot reach PyPI.',
      },
    })
  } else if (allPipInstalls.length > 0) {
    // Internet enabled but warn about potential submission issues
    results.push({
      check: 'Pip Install',
      status: 'warn',
      message: 'pip install detected - will fail if used for competition submission',
      details: {
        note: 'Competition submissions require enable_internet=false',
      },
    })
  }

  // Check for evaluate.load() which downloads metrics from HuggingFace
  const evaluateLoadMatches = content.match(/evaluate\.load\s*\(\s*["']([^"']+)["']/g) || []
  const evaluateMetrics = evaluateLoadMatches
    .map((m) => {
      const match = m.match(/["']([^"']+)["']/)
      return match ? match[1] : ''
    })
    .filter(Boolean)

  if (evaluateMetrics.length > 0) {
    if (!internetEnabled) {
      results.push({
        check: 'Internet Required',
        status: 'fail',
        message: `evaluate.load() downloads metrics from HuggingFace but enable_internet=false`,
        details: {
          metrics: evaluateMetrics,
          fix: 'Use sacrebleu directly instead of evaluate.load(). Example:\nimport sacrebleu\nresult = sacrebleu.corpus_chrf(hypotheses, [references])',
          alternative: 'Or use loss-based early stopping: metric_for_best_model="eval_loss"',
        },
      })
    } else {
      results.push({
        check: 'Internet Required',
        status: 'warn',
        message: `evaluate.load() requires internet - will fail if enable_internet=false`,
        details: {
          metrics: evaluateMetrics,
          note: 'Competition submission kernels typically require enable_internet=false',
          fix: 'Use sacrebleu directly: sacrebleu.corpus_chrf(hypotheses, [references])',
        },
      })
    }
  }

  // Check for HuggingFace model downloads (not from Kaggle Models)
  const fromPretrainedMatches = content.match(/from_pretrained\s*\(\s*["']([^"']+)["']/g) || []
  const modelPaths = fromPretrainedMatches
    .map((m) => {
      const match = m.match(/["']([^"']+)["']/)
      return match ? match[1] : ''
    })
    .filter(Boolean)

  // Filter to only HuggingFace hub models (contain / but not local paths)
  const hubModels = modelPaths.filter((p) => {
    // Skip local paths
    if (p.startsWith('./') || p.startsWith('/') || p.startsWith('~')) return false
    // Skip kagglehub downloaded paths
    if (p.includes('/kaggle/')) return false
    // Skip variable references
    if (p.includes('model_path') || p.includes('MODEL_PATH')) return false
    // Hub models contain org/model format
    return p.includes('/')
  })

  if (hubModels.length > 0 && !internetEnabled) {
    results.push({
      check: 'HuggingFace Download',
      status: 'fail',
      message: `HuggingFace model download requires internet but enable_internet=false`,
      details: {
        models: hubModels,
        fix: 'Use Kaggle Models instead:\n1. Upload model to Kaggle Model Registry\n2. Add to kernel-metadata.json model_sources\n3. Use: model_path = kagglehub.model_download("user/model/framework/version")',
      },
    })
  }

  // Also check for from_pretrained(CONFIG[...]) pattern - indirect HuggingFace loading
  // This catches cases where model name is stored in config variable
  const hasConfigModelLoad = /from_pretrained\s*\(\s*CONFIG\s*\[/.test(content)
  const hasKaggleModelDownload = /kagglehub\.model_download/.test(content)
  const hasLocalModelPath = /model_path\s*=\s*["']\.?\//.test(content)

  // Extract model name from CONFIG if present
  const configModelMatch = content.match(/["']model_name["']\s*:\s*["']([^"']+)["']/)
  const configModelName = configModelMatch ? configModelMatch[1] : null
  const isHubModel = configModelName?.includes('/') && !configModelName.startsWith('/')

  if (!internetEnabled && hasConfigModelLoad && isHubModel && !hasKaggleModelDownload && !hasLocalModelPath) {
    results.push({
      check: 'HuggingFace Download',
      status: 'fail',
      message: `Model "${configModelName}" requires internet download but enable_internet=false`,
      details: {
        model: configModelName,
        pattern: 'from_pretrained(CONFIG["model_name"])',
        fix: 'For competition submission with internet disabled:\n1. First train with internet ON and upload model to Kaggle Model Registry\n2. Create inference notebook that loads from registry:\n   model_path = kagglehub.model_download("user/model/transformers/v1")\n   model = AutoModel.from_pretrained(model_path)',
        alternative: 'Or set enable_internet=true (but cannot submit for scoring)',
      },
    })
  }

  return results
}

/**
 * Check for compute_metrics callback when metric_for_best_model is set
 *
 * When using custom metrics like chrf, bleu for early stopping or best model selection,
 * the Seq2SeqTrainer requires a compute_metrics callback to calculate these metrics.
 */
function checkComputeMetrics(content: string): CheckResult[] {
  const results: CheckResult[] = []

  // Check if metric_for_best_model is set to a custom metric
  const metricMatch = content.match(/["']?metric_for_best_model["']?\s*[=:]\s*["']([^"']+)["']/)
  if (!metricMatch) {
    return results // No custom metric configured
  }

  const metric = metricMatch[1].toLowerCase()

  // Built-in metrics that don't require compute_metrics
  const builtInMetrics = ['loss', 'eval_loss']
  if (builtInMetrics.includes(metric)) {
    return results // Using built-in metric, no callback needed
  }

  // Custom metrics that require compute_metrics callback
  const customMetrics = ['chrf', 'bleu', 'sacrebleu', 'rouge', 'meteor', 'accuracy', 'f1']
  const isCustomMetric = customMetrics.some((m) => metric.includes(m))

  if (!isCustomMetric) {
    return results // Unknown metric, don't flag
  }

  // Check if compute_metrics is defined
  const hasComputeMetricsDef = /def\s+compute_metrics\s*\(/.test(content)

  // Check if compute_metrics is passed to trainer (handles multi-line calls)
  const hasComputeMetricsInTrainer = /Trainer\s*\([\s\S]*?compute_metrics\s*=/.test(content)

  if (!hasComputeMetricsDef) {
    results.push({
      check: 'Compute Metrics',
      status: 'fail',
      message: `metric_for_best_model="${metric}" requires compute_metrics callback`,
      details: {
        fix: `Define a compute_metrics function that returns {"${metric}": value} and pass it to the Trainer`,
        metric_configured: metric,
        compute_metrics_defined: false,
        example: `
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Compute metric
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"${metric}": result["score"]}`,
      },
    })
  } else if (!hasComputeMetricsInTrainer) {
    results.push({
      check: 'Compute Metrics',
      status: 'fail',
      message: `compute_metrics defined but not passed to Trainer`,
      details: {
        fix: 'Add compute_metrics=compute_metrics to Seq2SeqTrainer(...)',
        metric_configured: metric,
        compute_metrics_defined: true,
        passed_to_trainer: false,
      },
    })
  } else {
    results.push({
      check: 'Compute Metrics',
      status: 'pass',
      message: `compute_metrics callback configured for ${metric}`,
    })
  }

  return results
}

/**
 * Check if notebook downloads model from Kaggle Model Registry
 *
 * For fine-tuning or inference from a pre-trained model in Kaggle registry,
 * the notebook should use kagglehub.model_download()
 */
function checkKaggleModelSource(content: string): CheckResult[] {
  const results: CheckResult[] = []

  // Check for kagglehub model download
  const modelDownloadPatterns = [/kagglehub\.model_download\s*\(/i, /model_download\s*\(\s*["']/i]

  const hasModelDownload = modelDownloadPatterns.some((p) => p.test(content))

  if (hasModelDownload) {
    results.push({
      check: 'Kaggle Model Source',
      status: 'pass',
      message: 'Model download from Kaggle registry detected',
    })

    // Also check that the handle format looks correct
    const handleMatch = content.match(/model_download\s*\(\s*["']([^"']+)["']/i)
    if (handleMatch) {
      const handle = handleMatch[1]
      const parts = handle.split('/')
      if (parts.length < 4) {
        results.push({
          check: 'Kaggle Model Handle',
          status: 'warn',
          message: `Model handle "${handle}" may be incomplete`,
          details: {
            expected: 'username/model/framework/variation',
            got: handle,
          },
        })
      }
    }
  }

  return results
}

// CLI command
const PreflightArgs = z.object({
  path: z.string().describe('Path to notebook (.ipynb) or script (.py)'),
  platform: z.string().default('kaggle-p100').describe('Target platform profile'),
  samples: z.number().default(1500).describe('Estimated training samples'),
  verbose: z.boolean().default(false).describe('Show detailed breakdown'),
  competition: z.boolean().default(false).describe('Check for competition submission format (inference notebooks)'),
  training: z
    .boolean()
    .default(false)
    .describe('Check for training notebook requirements (model save, registry upload)'),
})

export const preflight: CommandDefinition<typeof PreflightArgs> = {
  name: 'preflight check',
  description: 'Check if a notebook will run within platform resource limits',
  help: `
Pre-flight check for ML training notebooks.

Analyzes a notebook or script to estimate:
- GPU memory usage
- Disk space requirements
- Training time

With --competition flag (for inference notebooks), also checks:
- submission.csv output is created
- Correct columns (id, translation) are present
- index=False is used in to_csv()

With --training flag (for training notebooks), also checks:
- Model is saved (trainer.save_model or model.save_pretrained)
- Model is uploaded to Kaggle Model Registry (kagglehub.model_upload)

Compares against platform limits (Kaggle P100, Colab, etc.) and reports
potential issues before deployment.

Use 'akk preflight platforms' to see available platforms.
`,
  examples: [
    'akk preflight check notebook.ipynb',
    'akk preflight check training.py --platform kaggle-p100 --training',
    'akk preflight check notebook.ipynb --platform colab-pro --samples 3000',
    'akk preflight check notebook.ipynb --verbose',
    'akk preflight check inference.py --competition',
  ],
  args: PreflightArgs,

  async run(args, _ctx) {
    // Validate file exists
    if (!existsSync(args.path)) {
      return error(
        'FILE_NOT_FOUND',
        `File not found: ${args.path}`,
        'Provide a valid path to a notebook (.ipynb) or script (.py)'
      )
    }

    // Validate platform
    const platform = PLATFORMS[args.platform]
    if (!platform) {
      return error(
        'INVALID_PLATFORM',
        `Unknown platform: ${args.platform}`,
        `Use 'akk preflight platforms' to see available options`,
        { available: Object.keys(PLATFORMS) }
      )
    }

    // Read and parse file
    const ext = extname(args.path).toLowerCase()
    let content: string

    if (ext === '.ipynb') {
      try {
        const notebook = JSON.parse(readFileSync(args.path, 'utf-8'))
        // Extract code from cells
        content =
          notebook.cells
            ?.filter((c: { cell_type: string }) => c.cell_type === 'code')
            ?.map((c: { source: string[] }) => (Array.isArray(c.source) ? c.source.join('') : c.source))
            ?.join('\n') || ''
      } catch (e) {
        return error(
          'PARSE_ERROR',
          `Failed to parse notebook: ${e instanceof Error ? e.message : 'Unknown error'}`,
          'Ensure the file is a valid Jupyter notebook'
        )
      }
    } else if (ext === '.py') {
      content = readFileSync(args.path, 'utf-8')
    } else {
      return error('INVALID_FORMAT', `Unsupported file format: ${ext}`, 'Provide a .ipynb or .py file')
    }

    // Extract configuration
    const config = extractConfig(content)

    // Run checks
    const checks: CheckResult[] = []

    // 0. Deprecation Checks (run first - these cause runtime failures)
    const deprecationResults = checkDeprecations(content)
    checks.push(...deprecationResults)

    // 0.5. Competition Submission Checks (if --competition flag is set)
    if (args.competition) {
      const submissionResults = checkSubmissionOutput(content)
      checks.push(...submissionResults)
    }

    // 0.6. Training Notebook Checks (if --training flag is set)
    if (args.training) {
      const modelRegistryResults = checkModelRegistry(content)
      checks.push(...modelRegistryResults)
    }

    // 0.7. Kaggle Model Source Checks (detect model_download usage)
    const kaggleSourceResults = checkKaggleModelSource(content)
    checks.push(...kaggleSourceResults)

    // 0.8. Progress Bar Checks (for clean Kaggle logs)
    const progressBarResults = checkProgressBarsDisabled(content)
    checks.push(...progressBarResults)

    // 0.9. Dataset Column Checks (for training notebooks)
    const datasetColumnResults = checkDatasetColumns(content)
    checks.push(...datasetColumnResults)

    // 0.10. Compute Metrics Check (for training notebooks with custom metrics)
    const computeMetricsResults = checkComputeMetrics(content)
    checks.push(...computeMetricsResults)

    // 0.11. Internet Dependencies Check (critical for competition kernels)
    // Look for kernel-metadata.json in same directory as the notebook
    const notebookDir = dirname(args.path)
    const metadataPath = join(notebookDir, 'kernel-metadata.json')
    const internetResults = checkInternetDependencies(content, metadataPath)
    checks.push(...internetResults)

    // 0.12. Submission Internet Mismatch Check (CRITICAL)
    // Catches: enable_internet=true with submission.csv output → will fail on Kaggle scoring
    const submissionInternetResults = checkSubmissionInternetMismatch(content, metadataPath)
    checks.push(...submissionInternetResults)

    // 0.13. Pip --no-deps Check (causes missing transitive dependencies)
    const pipNoDepsResults = checkPipNoDeps(content)
    checks.push(...pipNoDepsResults)

    // 1. GPU Memory Check
    const gpuEstimate = estimateGpuMemory(config)
    const gpuStatus =
      gpuEstimate.peak_gb <= platform.gpu.vram_gb * 0.9
        ? 'pass'
        : gpuEstimate.peak_gb <= platform.gpu.vram_gb
          ? 'warn'
          : 'fail'
    checks.push({
      check: 'GPU Memory',
      status: gpuStatus,
      message:
        gpuStatus === 'fail'
          ? `Estimated ${gpuEstimate.peak_gb.toFixed(1)}GB exceeds ${platform.gpu.vram_gb}GB VRAM`
          : gpuStatus === 'warn'
            ? `Estimated ${gpuEstimate.peak_gb.toFixed(1)}GB is close to ${platform.gpu.vram_gb}GB limit`
            : `Estimated ${gpuEstimate.peak_gb.toFixed(1)}GB fits in ${platform.gpu.vram_gb}GB VRAM`,
      details: args.verbose ? gpuEstimate.breakdown : undefined,
    })

    // 2. Disk Space Check (use peak_gb for actual limit, as checkpoint rotation creates temporary spikes)
    const diskEstimate = estimateDiskUsage(config)
    const diskStatus =
      diskEstimate.peak_gb <= platform.disk.working_gb * 0.8
        ? 'pass'
        : diskEstimate.peak_gb <= platform.disk.working_gb
          ? 'warn'
          : 'fail'
    checks.push({
      check: 'Disk Space',
      status: diskStatus,
      message:
        diskStatus === 'fail'
          ? `Peak ${diskEstimate.peak_gb.toFixed(1)}GB exceeds ${platform.disk.working_gb}GB working space`
          : diskStatus === 'warn'
            ? `Peak ${diskEstimate.peak_gb.toFixed(1)}GB is close to ${platform.disk.working_gb}GB limit`
            : `Peak ${diskEstimate.peak_gb.toFixed(1)}GB fits in ${platform.disk.working_gb}GB working space`,
      details: args.verbose ? { ...diskEstimate.breakdown, peak_gb: diskEstimate.peak_gb } : undefined,
    })

    // 3. Training Time Check
    const timeEstimate = estimateTrainingTime(config, platform, args.samples)
    const timeStatus =
      timeEstimate.hours <= platform.time.max_hours * 0.8
        ? 'pass'
        : timeEstimate.hours <= platform.time.max_hours
          ? 'warn'
          : 'fail'
    checks.push({
      check: 'Training Time',
      status: timeStatus,
      message:
        timeStatus === 'fail'
          ? `Estimated ${timeEstimate.hours.toFixed(1)}h exceeds ${platform.time.max_hours}h limit`
          : timeStatus === 'warn'
            ? `Estimated ${timeEstimate.hours.toFixed(1)}h is close to ${platform.time.max_hours}h limit`
            : `Estimated ${timeEstimate.hours.toFixed(1)}h fits in ${platform.time.max_hours}h limit`,
      details: args.verbose ? timeEstimate.breakdown : undefined,
    })

    // 4. Batch Size Recommendation
    if (gpuStatus === 'fail' && config.batch_size && config.batch_size > 2) {
      const recommendedBatch = Math.max(1, Math.floor((config.batch_size * platform.gpu.vram_gb) / gpuEstimate.peak_gb))
      checks.push({
        check: 'Recommendation',
        status: 'warn',
        message: `Consider reducing batch_size from ${config.batch_size} to ${recommendedBatch}`,
        details: {
          current_batch_size: config.batch_size,
          recommended_batch_size: recommendedBatch,
          gradient_accumulation: config.gradient_accumulation_steps || 1,
        },
      })
    }

    // 5. Checkpoint Space Recommendation
    if (diskStatus !== 'pass') {
      const currentLimit = config.save_total_limit || 3
      const modelSizeFp16 = MODEL_SIZES[config.model_name || '']?.size_gb || 2
      const checkpointSizeWithOptimizer = modelSizeFp16 * 6 // model + optimizer in fp32
      const checkpointSizeWithoutOptimizer = modelSizeFp16 * 2 // model only in fp32

      // Only recommend save_only_model if it's not already enabled
      if (!diskEstimate.save_only_model) {
        checks.push({
          check: 'Disk Recommendation',
          status: 'fail',
          message: `Checkpoint size ~${checkpointSizeWithOptimizer.toFixed(1)}GB includes optimizer states. Add: save_only_model=True`,
          details: {
            current_save_total_limit: currentLimit,
            checkpoint_with_optimizer_gb: checkpointSizeWithOptimizer,
            checkpoint_without_optimizer_gb: checkpointSizeWithoutOptimizer,
            savings_gb: checkpointSizeWithOptimizer - checkpointSizeWithoutOptimizer,
            note: 'HuggingFace Trainer saves optimizer states (2x model size) by default. Set save_only_model=True in TrainingArguments to save ~4GB per checkpoint for NLLB-600M.',
            training_args_fix: 'Seq2SeqTrainingArguments(..., save_only_model=True)',
          },
        })
      }

      if (currentLimit > 1) {
        checks.push({
          check: 'Checkpoint Limit',
          status: 'warn',
          message: `Reduce save_total_limit from ${currentLimit} to 1`,
          details: {
            current_save_total_limit: currentLimit,
            recommended: 1,
          },
        })
      }
    }

    // Summary
    const failed = checks.filter((c) => c.status === 'fail')
    const warned = checks.filter((c) => c.status === 'warn')
    const passed = checks.filter((c) => c.status === 'pass')

    const overallStatus = failed.length > 0 ? 'fail' : warned.length > 0 ? 'warn' : 'pass'

    return success({
      file: basename(args.path),
      platform: platform.name,
      status: overallStatus,
      summary: `${passed.length} passed, ${warned.length} warnings, ${failed.length} failed`,
      config_detected: {
        model: config.model_name || 'unknown',
        batch_size: config.batch_size,
        gradient_accumulation: config.gradient_accumulation_steps,
        max_length: Math.max(config.max_src_len || 0, config.max_tgt_len || 0) || undefined,
        epochs: config.num_epochs,
        fp16: config.fp16,
        save_total_limit: config.save_total_limit,
        save_only_model: config.save_only_model,
        clear_hf_cache: config.clear_hf_cache,
      },
      checks,
      recommendations:
        failed.length > 0
          ? [
              'Set save_only_model=True to skip optimizer states and save ~4GB per checkpoint',
              'Set save_total_limit=1 to minimize disk usage',
              'Reduce batch_size and increase gradient_accumulation_steps to maintain effective batch',
              'Reduce max_src_len/max_tgt_len if sequences are being heavily truncated anyway',
              'Set dataloader_num_workers=0 to reduce memory overhead',
            ]
          : undefined,
    })
  },
}

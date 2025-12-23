import { existsSync, readFileSync } from 'fs'
import { basename, extname } from 'path'
import { z } from 'zod'
import { error, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'
import { PLATFORMS, type PlatformProfile } from './platforms'

/**
 * Model size estimates (in GB for fp16)
 */
const MODEL_SIZES: Record<string, { params_b: number; size_gb: number; name: string }> = {
  // NLLB models
  'facebook/nllb-200-distilled-600M': { params_b: 0.6, size_gb: 1.2, name: 'NLLB-600M' },
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
  if (batchMatch) config.batch_size = parseInt(batchMatch[1])

  // Gradient accumulation
  const gradAccumMatch = content.match(/["']?gradient_accumulation_steps["']?\s*[=:]\s*(\d+)/)
  if (gradAccumMatch) config.gradient_accumulation_steps = parseInt(gradAccumMatch[1])

  // Max lengths
  const maxSrcMatch = content.match(/["']?max_src_len["']?\s*[=:]\s*(\d+)/)
  if (maxSrcMatch) config.max_src_len = parseInt(maxSrcMatch[1])

  const maxTgtMatch = content.match(/["']?max_tgt_len["']?\s*[=:]\s*(\d+)/)
  if (maxTgtMatch) config.max_tgt_len = parseInt(maxTgtMatch[1])

  // Epochs
  const epochsMatch = content.match(/["']?num_epochs["']?\s*[=:]\s*(\d+)/)
  if (epochsMatch) config.num_epochs = parseInt(epochsMatch[1])

  // FP16 (handles both fp16=True and "fp16": True)
  config.fp16 = /["']?fp16["']?\s*[=:]\s*True/i.test(content)

  // Save total limit (handles both save_total_limit=1 and "save_total_limit": 1)
  const saveMatch = content.match(/["']?save_total_limit["']?\s*[=:]\s*(\d+)/)
  if (saveMatch) config.save_total_limit = parseInt(saveMatch[1])

  // Save only model (skip optimizer states to save disk space)
  config.save_only_model = /["']?save_only_model["']?\s*[=:]\s*True/i.test(content)

  // Dataloader workers
  const workersMatch = content.match(/dataloader_num_workers\s*=\s*(\d+)/)
  if (workersMatch) config.dataloader_num_workers = parseInt(workersMatch[1])

  return config
}

/**
 * Estimate GPU memory usage for training
 */
function estimateGpuMemory(config: ExtractedConfig): { peak_gb: number; breakdown: Record<string, number> } {
  const modelInfo = config.model_name ? MODEL_SIZES[config.model_name] : null
  const modelSize = modelInfo?.size_gb || 2.0 // Default 2GB

  const batchSize = config.batch_size || 8
  const seqLen = Math.max(config.max_src_len || 256, config.max_tgt_len || 256)
  const fp16 = config.fp16 !== false // Default to fp16

  // Memory breakdown (rough estimates)
  const breakdown: Record<string, number> = {}

  // Model weights
  breakdown['model_weights'] = fp16 ? modelSize : modelSize * 2

  // Optimizer states (Adam: 2x model size for momentum + variance)
  breakdown['optimizer_states'] = breakdown['model_weights'] * 2

  // Gradients
  breakdown['gradients'] = breakdown['model_weights']

  // Activations (rough estimate based on batch size and sequence length)
  // This is highly variable but we use a heuristic
  const activationFactor = (batchSize * seqLen * seqLen) / (8 * 256 * 256)
  breakdown['activations'] = Math.min(modelSize * 2 * activationFactor, 16) // Cap at 16GB

  // KV cache and attention
  breakdown['attention_cache'] = batchSize * seqLen * 0.001 // ~1MB per token per batch

  // CUDA overhead
  breakdown['cuda_overhead'] = 0.5

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
} {
  const modelInfo = config.model_name ? MODEL_SIZES[config.model_name] : null
  const modelSizeFp16 = modelInfo?.size_gb || 2.0
  const saveOnlyModel = config.save_only_model === true

  // Model weights in checkpoint (fp32 = 2x fp16)
  const modelWeightsFp32 = modelSizeFp16 * 2

  // Optimizer states in checkpoint: Adam saves momentum + variance (2x model weights)
  // If save_only_model=True, optimizer is NOT saved (significant disk savings!)
  const optimizerStateSize = saveOnlyModel ? 0 : modelWeightsFp32 * 2

  // Full checkpoint size: model + optimizer + small overhead
  const checkpointSize = modelWeightsFp32 + optimizerStateSize + 0.01

  const breakdown: Record<string, number> = {}

  // HuggingFace cache (model download - includes both pytorch and safetensors)
  breakdown['hf_cache'] = modelSizeFp16 * 2.5 // Often downloads multiple formats

  // Checkpoints at steady state (model + optimizer states)
  const saveLimit = config.save_total_limit || 3
  breakdown['checkpoints_model'] = modelWeightsFp32 * saveLimit
  breakdown['checkpoints_optimizer'] = optimizerStateSize * saveLimit

  // Peak checkpoint usage: during rotation, +1 checkpoint temporarily exists
  const peakCheckpoints = checkpointSize * (saveLimit + 1)

  // Final saved model (also fp32 by default from Trainer)
  breakdown['final_model'] = modelWeightsFp32

  // Tokenizer and configs
  breakdown['tokenizer_config'] = 0.1

  // Training logs, tensorboard, etc
  breakdown['logs'] = 0.05

  // Training outputs (submission.csv, etc)
  breakdown['outputs'] = 0.01

  const total_gb = Object.values(breakdown).reduce((a, b) => a + b, 0)

  // Peak includes the extra checkpoint during rotation
  const steadyCheckpoints = (modelWeightsFp32 + optimizerStateSize) * saveLimit
  const peak_gb = total_gb - steadyCheckpoints + peakCheckpoints

  return { total_gb, breakdown, peak_gb, save_only_model: saveOnlyModel }
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
  breakdown['training'] = (totalSteps * secondsPerStep) / 3600
  breakdown['evaluation'] = ((totalSteps / 100) * 30) / 3600 // ~30 sec per eval
  breakdown['model_loading'] = 0.1 // 6 minutes for model download/load
  breakdown['bleu_computation'] = 0.1 // 6 minutes for final BLEU

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

  async run(args, ctx) {
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

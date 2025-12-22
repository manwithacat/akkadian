import { z } from 'zod'
import { existsSync, readFileSync } from 'fs'
import { basename, extname } from 'path'
import type { CommandDefinition } from '../../types/commands'
import { success, error } from '../../lib/output'
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

  // FP16
  config.fp16 = /fp16\s*[=:]\s*True/i.test(content)

  // Save total limit
  const saveMatch = content.match(/save_total_limit\s*=\s*(\d+)/)
  if (saveMatch) config.save_total_limit = parseInt(saveMatch[1])

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
  const modelSize = modelInfo?.size_gb || 2.0  // Default 2GB

  const batchSize = config.batch_size || 8
  const seqLen = Math.max(config.max_src_len || 256, config.max_tgt_len || 256)
  const fp16 = config.fp16 !== false  // Default to fp16

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
  breakdown['activations'] = Math.min(modelSize * 2 * activationFactor, 16)  // Cap at 16GB

  // KV cache and attention
  breakdown['attention_cache'] = batchSize * seqLen * 0.001  // ~1MB per token per batch

  // CUDA overhead
  breakdown['cuda_overhead'] = 0.5

  const peak_gb = Object.values(breakdown).reduce((a, b) => a + b, 0)

  return { peak_gb, breakdown }
}

/**
 * Estimate disk usage
 */
function estimateDiskUsage(config: ExtractedConfig): { total_gb: number; breakdown: Record<string, number> } {
  const modelInfo = config.model_name ? MODEL_SIZES[config.model_name] : null
  const modelSize = modelInfo?.size_gb || 2.0

  const breakdown: Record<string, number> = {}

  // HuggingFace cache (model download)
  breakdown['hf_cache'] = modelSize * 1.2  // Some overhead

  // Checkpoints (depends on save_total_limit)
  const saveLimit = config.save_total_limit || 3
  breakdown['checkpoints'] = modelSize * saveLimit

  // Final saved model
  breakdown['final_model'] = modelSize

  // Tokenizer and configs
  breakdown['tokenizer_config'] = 0.05

  // Logs and metrics
  breakdown['logs'] = 0.01

  // Training outputs (submission.csv, etc)
  breakdown['outputs'] = 0.01

  const total_gb = Object.values(breakdown).reduce((a, b) => a + b, 0)

  return { total_gb, breakdown }
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
  const modelParams = modelInfo?.params_b || 0.6  // Default to 600M

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
  const seqScale = Math.pow(seqLen / 192, 1.5)

  const secondsPerStep = baseSecondsPerStep * modelScale * gpuScale * batchScale * seqScale

  const breakdown: Record<string, number> = {}
  breakdown['training'] = (totalSteps * secondsPerStep) / 3600
  breakdown['evaluation'] = (totalSteps / 100) * 30 / 3600  // ~30 sec per eval
  breakdown['model_loading'] = 0.1  // 6 minutes for model download/load
  breakdown['bleu_computation'] = 0.1  // 6 minutes for final BLEU

  const hours = Object.values(breakdown).reduce((a, b) => a + b, 0)

  return { hours, breakdown }
}

interface CheckResult {
  check: string
  status: 'pass' | 'warn' | 'fail'
  message: string
  details?: Record<string, unknown>
}

// CLI command
const PreflightArgs = z.object({
  path: z.string().describe('Path to notebook (.ipynb) or script (.py)'),
  platform: z.string().default('kaggle-p100').describe('Target platform profile'),
  samples: z.number().default(1500).describe('Estimated training samples'),
  verbose: z.boolean().default(false).describe('Show detailed breakdown'),
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

Compares against platform limits (Kaggle P100, Colab, etc.) and reports
potential issues before deployment.

Use 'akk preflight platforms' to see available platforms.
`,
  examples: [
    'akk preflight check notebook.ipynb',
    'akk preflight check training.py --platform kaggle-p100',
    'akk preflight check notebook.ipynb --platform colab-pro --samples 3000',
    'akk preflight check notebook.ipynb --verbose',
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
        content = notebook.cells
          ?.filter((c: { cell_type: string }) => c.cell_type === 'code')
          ?.map((c: { source: string[] }) =>
            Array.isArray(c.source) ? c.source.join('') : c.source
          )
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
      return error(
        'INVALID_FORMAT',
        `Unsupported file format: ${ext}`,
        'Provide a .ipynb or .py file'
      )
    }

    // Extract configuration
    const config = extractConfig(content)

    // Run checks
    const checks: CheckResult[] = []

    // 1. GPU Memory Check
    const gpuEstimate = estimateGpuMemory(config)
    const gpuStatus = gpuEstimate.peak_gb <= platform.gpu.vram_gb * 0.9 ? 'pass' :
                      gpuEstimate.peak_gb <= platform.gpu.vram_gb ? 'warn' : 'fail'
    checks.push({
      check: 'GPU Memory',
      status: gpuStatus,
      message: gpuStatus === 'fail'
        ? `Estimated ${gpuEstimate.peak_gb.toFixed(1)}GB exceeds ${platform.gpu.vram_gb}GB VRAM`
        : gpuStatus === 'warn'
        ? `Estimated ${gpuEstimate.peak_gb.toFixed(1)}GB is close to ${platform.gpu.vram_gb}GB limit`
        : `Estimated ${gpuEstimate.peak_gb.toFixed(1)}GB fits in ${platform.gpu.vram_gb}GB VRAM`,
      details: args.verbose ? gpuEstimate.breakdown : undefined,
    })

    // 2. Disk Space Check
    const diskEstimate = estimateDiskUsage(config)
    const diskStatus = diskEstimate.total_gb <= platform.disk.working_gb * 0.8 ? 'pass' :
                       diskEstimate.total_gb <= platform.disk.working_gb ? 'warn' : 'fail'
    checks.push({
      check: 'Disk Space',
      status: diskStatus,
      message: diskStatus === 'fail'
        ? `Estimated ${diskEstimate.total_gb.toFixed(1)}GB exceeds ${platform.disk.working_gb}GB working space`
        : diskStatus === 'warn'
        ? `Estimated ${diskEstimate.total_gb.toFixed(1)}GB is close to ${platform.disk.working_gb}GB limit`
        : `Estimated ${diskEstimate.total_gb.toFixed(1)}GB fits in ${platform.disk.working_gb}GB working space`,
      details: args.verbose ? diskEstimate.breakdown : undefined,
    })

    // 3. Training Time Check
    const timeEstimate = estimateTrainingTime(config, platform, args.samples)
    const timeStatus = timeEstimate.hours <= platform.time.max_hours * 0.8 ? 'pass' :
                       timeEstimate.hours <= platform.time.max_hours ? 'warn' : 'fail'
    checks.push({
      check: 'Training Time',
      status: timeStatus,
      message: timeStatus === 'fail'
        ? `Estimated ${timeEstimate.hours.toFixed(1)}h exceeds ${platform.time.max_hours}h limit`
        : timeStatus === 'warn'
        ? `Estimated ${timeEstimate.hours.toFixed(1)}h is close to ${platform.time.max_hours}h limit`
        : `Estimated ${timeEstimate.hours.toFixed(1)}h fits in ${platform.time.max_hours}h limit`,
      details: args.verbose ? timeEstimate.breakdown : undefined,
    })

    // 4. Batch Size Recommendation
    if (gpuStatus === 'fail' && config.batch_size && config.batch_size > 2) {
      const recommendedBatch = Math.max(1, Math.floor(config.batch_size * platform.gpu.vram_gb / gpuEstimate.peak_gb))
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
    if (diskStatus === 'fail' && config.save_total_limit && config.save_total_limit > 1) {
      checks.push({
        check: 'Recommendation',
        status: 'warn',
        message: `Consider reducing save_total_limit from ${config.save_total_limit} to 1`,
      })
    }

    // Summary
    const failed = checks.filter(c => c.status === 'fail')
    const warned = checks.filter(c => c.status === 'warn')
    const passed = checks.filter(c => c.status === 'pass')

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
      },
      checks,
      recommendations: failed.length > 0 ? [
        'Reduce batch_size and increase gradient_accumulation_steps to maintain effective batch',
        'Reduce max_src_len/max_tgt_len if sequences are being heavily truncated anyway',
        'Set save_total_limit=1 to minimize disk usage',
        'Set dataloader_num_workers=0 to reduce memory overhead',
      ] : undefined,
    })
  },
}

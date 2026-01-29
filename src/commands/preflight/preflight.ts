import { existsSync, readFileSync } from 'fs'
import { basename, dirname, extname, join } from 'path'
import { z } from 'zod'
import { getModelInstanceFiles } from '../../lib/kaggle'
import { error, success } from '../../lib/output'
import { extractDatasetReferences } from '../../lib/utils'
import type { CommandDefinition } from '../../types/commands'
import { PLATFORMS, type PlatformProfile } from './platforms'

/**
 * Model size estimates (in GB for fp16)
 * size_gb = params_b * 2 (2 bytes per param in fp16)
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

  // T5 models (original)
  'google-t5/t5-small': { params_b: 0.06, size_gb: 0.12, name: 'T5-Small' },
  'google-t5/t5-base': { params_b: 0.22, size_gb: 0.44, name: 'T5-Base' },
  'google-t5/t5-large': { params_b: 0.77, size_gb: 1.54, name: 'T5-Large' },
  't5-small': { params_b: 0.06, size_gb: 0.12, name: 'T5-Small' },
  't5-base': { params_b: 0.22, size_gb: 0.44, name: 'T5-Base' },
  't5-large': { params_b: 0.77, size_gb: 1.54, name: 'T5-Large' },

  // Flan-T5 models
  'google/flan-t5-small': {
    params_b: 0.08,
    size_gb: 0.16,
    name: 'Flan-T5-Small',
  },
  'google/flan-t5-base': { params_b: 0.25, size_gb: 0.5, name: 'Flan-T5-Base' },
  'google/flan-t5-large': {
    params_b: 0.78,
    size_gb: 1.56,
    name: 'Flan-T5-Large',
  },
  'google/flan-t5-xl': { params_b: 3.0, size_gb: 6.0, name: 'Flan-T5-XL' },
  'google/flan-t5-xxl': { params_b: 11.0, size_gb: 22.0, name: 'Flan-T5-XXL' },

  // ByT5 models
  'google/byt5-small': { params_b: 0.3, size_gb: 0.6, name: 'ByT5-Small' },
  'google/byt5-base': { params_b: 0.58, size_gb: 1.16, name: 'ByT5-Base' },
  'google/byt5-large': { params_b: 1.2, size_gb: 2.4, name: 'ByT5-Large' },

  // mT5 models
  'google/mt5-small': { params_b: 0.3, size_gb: 0.6, name: 'mT5-Small' },
  'google/mt5-base': { params_b: 0.58, size_gb: 1.16, name: 'mT5-Base' },
  'google/mt5-large': { params_b: 1.2, size_gb: 2.4, name: 'mT5-Large' },

  // BART models
  'facebook/bart-base': { params_b: 0.14, size_gb: 0.28, name: 'BART-Base' },
  'facebook/bart-large': { params_b: 0.4, size_gb: 0.8, name: 'BART-Large' },

  // mBART models
  'facebook/mbart-large-50': {
    params_b: 0.61,
    size_gb: 1.22,
    name: 'mBART-50',
  },
  'facebook/mbart-large-50-many-to-many-mmt': {
    params_b: 0.61,
    size_gb: 1.22,
    name: 'mBART-50-M2M',
  },
}

/**
 * Fuzzy match model name to known models
 * Handles variations like local paths, different prefixes, etc.
 */
function findModelInfo(modelName: string | undefined): { params_b: number; size_gb: number; name: string } | null {
  if (!modelName) return null

  // Direct lookup first
  if (MODEL_SIZES[modelName]) {
    return MODEL_SIZES[modelName]
  }

  // Normalize the model name for fuzzy matching
  const normalized = modelName.toLowerCase()

  // Try to match by key patterns
  for (const [key, info] of Object.entries(MODEL_SIZES)) {
    const keyNorm = key.toLowerCase()
    // Check if the model name contains the key or vice versa
    if (normalized.includes(keyNorm) || keyNorm.includes(normalized)) {
      return info
    }
    // Check just the model part (after the last /)
    const keyModel = key.split('/').pop()?.toLowerCase() || ''
    const nameModel = modelName.split('/').pop()?.toLowerCase() || ''
    if (keyModel && nameModel && (keyModel.includes(nameModel) || nameModel.includes(keyModel))) {
      return info
    }
  }

  // Model family keyword matching for paths like "akkadian-byt5-v1-0-10"
  const modelFamilies: Array<{
    keywords: string[]
    defaultKey: string
  }> = [
    { keywords: ['byt5'], defaultKey: 'google/byt5-base' },
    {
      keywords: ['flan-t5', 'flan_t5', 'flant5'],
      defaultKey: 'google/flan-t5-base',
    },
    { keywords: ['mt5'], defaultKey: 'google/mt5-base' },
    { keywords: ['t5'], defaultKey: 'google-t5/t5-base' }, // After more specific T5 variants
    { keywords: ['nllb'], defaultKey: 'facebook/nllb-200-distilled-600M' },
    { keywords: ['bart'], defaultKey: 'facebook/bart-base' },
    { keywords: ['mbart'], defaultKey: 'facebook/mbart-large-50' },
  ]

  for (const { keywords, defaultKey } of modelFamilies) {
    if (keywords.some((kw) => normalized.includes(kw))) {
      // Found a family match, now try to determine size
      const sizeHints: Array<{ pattern: RegExp; sizeSuffix: string }> = [
        { pattern: /small/i, sizeSuffix: '-small' },
        { pattern: /base/i, sizeSuffix: '-base' },
        { pattern: /large/i, sizeSuffix: '-large' },
        { pattern: /xl(?!l)/i, sizeSuffix: '-xl' },
        { pattern: /xxl/i, sizeSuffix: '-xxl' },
      ]

      // Try to find size-specific variant
      for (const { pattern, sizeSuffix } of sizeHints) {
        if (pattern.test(normalized)) {
          const sizedKey = defaultKey.replace(/-base|-small|-large|-xl|-xxl/i, sizeSuffix)
          if (MODEL_SIZES[sizedKey]) {
            return MODEL_SIZES[sizedKey]
          }
        }
      }

      // Fall back to default (usually base) for this family
      if (MODEL_SIZES[defaultKey]) {
        return MODEL_SIZES[defaultKey]
      }
    }
  }

  // Pattern-based estimation for unknown models
  // Look for size indicators in the name
  const sizePatterns: Array<{ pattern: RegExp; params_b: number }> = [
    { pattern: /xxl|11b/i, params_b: 11.0 },
    { pattern: /xl|3b/i, params_b: 3.0 },
    { pattern: /large|1\.?[23]b|770m/i, params_b: 1.0 },
    { pattern: /base|[56]00m/i, params_b: 0.5 },
    { pattern: /small|[12]00m/i, params_b: 0.2 },
    { pattern: /tiny|60m/i, params_b: 0.06 },
  ]

  for (const { pattern, params_b } of sizePatterns) {
    if (pattern.test(normalized)) {
      return {
        params_b,
        size_gb: params_b * 2,
        name: `Unknown (${modelName.split('/').pop()})`,
      }
    }
  }

  // Default: assume small model (0.5B) rather than large to avoid false alarms
  return {
    params_b: 0.5,
    size_gb: 1.0,
    name: `Unknown (${modelName.split('/').pop() || 'model'})`,
  }
}

/**
 * Detect if this is an inference-only kernel (no training)
 */
function detectInferenceMode(content: string): boolean {
  // Signs of training
  const trainingIndicators = [
    /Trainer\s*\(/,
    /\.train\s*\(\)/,
    /training_args/i,
    /TrainingArguments/,
    /num_train_epochs/,
    /learning_rate\s*[=:]/,
    /optimizer\s*=/,
    /\.backward\s*\(\)/,
    /loss\.backward/,
  ]

  // Signs of inference-only
  const inferenceIndicators = [
    /model\.eval\s*\(\)/,
    /torch\.no_grad/,
    /model\.generate\s*\(/,
    /with\s+torch\.inference_mode/,
  ]

  const hasTraining = trainingIndicators.some((p) => p.test(content))
  const hasInference = inferenceIndicators.some((p) => p.test(content))

  // If has inference but no training indicators, it's inference-only
  if (hasInference && !hasTraining) {
    return true
  }

  // Filename heuristics
  const lowerContent = content.toLowerCase()
  if (lowerContent.includes('inference') || lowerContent.includes('submission') || lowerContent.includes('predict')) {
    if (!hasTraining) {
      return true
    }
  }

  return false
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

  // Model name patterns - direct string literals
  const modelPatterns = [
    /["']model_name["']\s*:\s*["']([^"']+)["']/,
    /MODEL_NAME\s*=\s*["']([^"']+)["']/,
    /from_pretrained\s*\(\s*["']([^"']+)["']/,
  ]

  // Also look for variable assignments like MODEL_PATH = "..."
  // Common patterns: BYT5_MODEL_PATH, PHILOLOGIST_MODEL_PATH, MODEL_DIR, etc.
  const modelPathPatterns = [/[A-Z_]*MODEL[A-Z_]*\s*=\s*["']([^"']+)["']/g, /[A-Z_]*_PATH\s*=\s*["']([^"']+)["']/g]

  // Collect all model paths found
  const modelPaths: string[] = []

  for (const pattern of modelPatterns) {
    const match = content.match(pattern)
    if (match?.[1]) {
      modelPaths.push(match[1])
    }
  }

  // Find variable assignments that look like model paths
  for (const pattern of modelPathPatterns) {
    for (const match of content.matchAll(pattern)) {
      const path = match[1]
      // Filter to paths that look like model locations
      if (
        path.includes('/kaggle/input/') ||
        path.includes('huggingface') ||
        path.includes('google/') ||
        path.includes('facebook/') ||
        path.includes('model') ||
        path.includes('transformer')
      ) {
        modelPaths.push(path)
      }
    }
  }

  // Use the first valid model path found
  if (modelPaths.length > 0) {
    config.model_name = modelPaths[0]
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
 * Estimate GPU memory usage
 *
 * @param config - Extracted configuration from the notebook
 * @param inferenceMode - If true, estimates for inference-only (much lower memory)
 * @param sequentialInfo - Optional sequential loading info for multi-stage pipelines
 * @param allModelPaths - Optional list of all model paths for sequential estimation
 */
function estimateGpuMemory(
  config: ExtractedConfig,
  inferenceMode: boolean = false,
  sequentialInfo?: SequentialLoadingInfo,
  allModelPaths?: string[]
): {
  peak_gb: number
  breakdown: Record<string, number>
  mode: 'training' | 'inference'
  model_detected: string
  sequential_loading?: {
    detected: boolean
    stages: number
    peak_stage_models: number
    note: string
  }
} {
  // Use fuzzy model matching
  const modelInfo = findModelInfo(config.model_name)
  const modelSize = modelInfo?.size_gb || 1.0 // Default 1GB (smaller default)
  const modelName = modelInfo?.name || 'Unknown'

  const batchSize = config.batch_size || 8
  const seqLen = Math.max(config.max_src_len || 256, config.max_tgt_len || 256)
  const fp16 = config.fp16 !== false // Default to fp16

  const breakdown: Record<string, number> = {}

  // Check for sequential loading - if detected, calculate peak memory differently
  if (sequentialInfo?.isSequential && allModelPaths && allModelPaths.length > 1) {
    // Sequential loading detected - calculate peak memory for largest stage
    // instead of summing all models

    // Calculate memory for each stage and find the peak
    let peakStageMemory = 0
    let peakStageModels: string[] = []

    for (const stage of sequentialInfo.stages) {
      // Estimate memory for this stage's models
      let stageMemory = 0
      for (const modelVar of stage.models) {
        // Try to find matching model path
        const matchingPath = allModelPaths.find((p) =>
          p.toLowerCase().includes(modelVar.toLowerCase().replace('_model', '').replace('model', ''))
        )
        const stageModelInfo = matchingPath ? findModelInfo(matchingPath) : modelInfo
        const stageModelSize = stageModelInfo?.size_gb || modelSize

        if (inferenceMode) {
          // Inference memory per model
          const weights = fp16 ? stageModelSize : stageModelSize * 2
          const activations = stageModelSize * 0.2 * (batchSize / 4)
          stageMemory += weights + activations
        } else {
          // Training memory per model
          const weights = fp16 ? stageModelSize : stageModelSize * 2
          stageMemory += weights * 4 // weights + optimizer + gradients
        }
      }

      // Add KV cache and overhead once per stage
      if (inferenceMode) {
        const numLayers = Math.ceil((modelInfo?.params_b || 0.5) * 24)
        stageMemory += (batchSize * seqLen * numLayers * 2) / (1024 * 1024) // KV cache
        stageMemory += 0.3 // CUDA overhead
      } else {
        stageMemory += 0.5 // CUDA overhead
      }

      if (stageMemory > peakStageMemory) {
        peakStageMemory = stageMemory
        peakStageModels = stage.models
      }
    }

    // Use peak stage memory instead of sum
    breakdown.peak_stage_models = peakStageMemory
    breakdown.sequential_note = 0.01 // Marker

    return {
      peak_gb: peakStageMemory,
      breakdown,
      mode: inferenceMode ? 'inference' : 'training',
      model_detected: `${modelName} (sequential: ${sequentialInfo.stages.length} stages)`,
      sequential_loading: {
        detected: true,
        stages: sequentialInfo.stages.length,
        peak_stage_models: peakStageModels.length,
        note: `Memory calculated for peak stage (${peakStageModels.join(', ')}) - models freed between stages`,
      },
    }
  }

  // Standard (non-sequential) memory estimation
  if (inferenceMode) {
    // INFERENCE MODE: Much simpler memory requirements
    // Only need model weights + small activation buffer
    breakdown.model_weights = fp16 ? modelSize : modelSize * 2

    // KV cache for generation (smaller than training activations)
    // ~2KB per token per layer for typical models
    const numLayers = Math.ceil((modelInfo?.params_b || 0.5) * 24) // Estimate layers
    breakdown.kv_cache = (batchSize * seqLen * numLayers * 2) / (1024 * 1024) // KB to GB

    // Small activation buffer for current forward pass
    breakdown.activations = modelSize * 0.2 * (batchSize / 4)

    // CUDA overhead
    breakdown.cuda_overhead = 0.3
  } else {
    // TRAINING MODE: Full memory requirements
    // Model weights
    breakdown.model_weights = fp16 ? modelSize : modelSize * 2

    // Optimizer states (Adam: 2x model size for momentum + variance)
    breakdown.optimizer_states = breakdown.model_weights * 2

    // Gradients
    breakdown.gradients = breakdown.model_weights

    // Activations (rough estimate based on batch size and sequence length)
    // More refined: scale with model size and sequence length
    const activationFactor = (batchSize * seqLen) / (8 * 256)
    breakdown.activations = Math.min(modelSize * 1.5 * activationFactor, 12) // Cap at 12GB

    // KV cache and attention
    breakdown.attention_cache = batchSize * seqLen * 0.0005 // ~0.5MB per token per batch

    // CUDA overhead
    breakdown.cuda_overhead = 0.5
  }

  const peak_gb = Object.values(breakdown).reduce((a, b) => a + b, 0)

  return {
    peak_gb,
    breakdown,
    mode: inferenceMode ? 'inference' : 'training',
    model_detected: modelName,
  }
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
function estimateDiskUsage(
  config: ExtractedConfig,
  inferenceMode: boolean = false
): {
  total_gb: number
  breakdown: Record<string, number>
  peak_gb: number
  save_only_model: boolean
  clear_hf_cache: boolean
  mode: 'training' | 'inference'
} {
  // Use fuzzy model matching
  const modelInfo = findModelInfo(config.model_name)
  const modelSizeFp16 = modelInfo?.size_gb || 1.0 // Default 1GB (smaller)
  const saveOnlyModel = config.save_only_model === true
  const clearHfCache = config.clear_hf_cache === true

  const breakdown: Record<string, number> = {}

  if (inferenceMode) {
    // INFERENCE MODE: Much simpler disk requirements
    // No checkpoints, no optimizer states, just model + outputs

    // Model loaded from Kaggle input (already on disk, doesn't count toward working space)
    // Or HF cache if downloading
    const hfCacheSize = modelSizeFp16 * 2.5
    breakdown.hf_cache = clearHfCache ? 0 : hfCacheSize

    // Output files (submission.csv, etc)
    breakdown.outputs = 0.1

    // Temp files
    breakdown.temp = 0.1

    const total_gb = Object.values(breakdown).reduce((a, b) => a + b, 0)

    return {
      total_gb,
      breakdown,
      peak_gb: total_gb,
      save_only_model: saveOnlyModel,
      clear_hf_cache: clearHfCache,
      mode: 'inference',
    }
  }

  // TRAINING MODE: Full disk requirements
  // Model weights in checkpoint (fp32 = 2x fp16)
  const modelWeightsFp32 = modelSizeFp16 * 2

  // Optimizer states in checkpoint: Adam saves momentum + variance (2x model weights)
  // If save_only_model=True, optimizer is NOT saved (significant disk savings!)
  const optimizerStateSize = saveOnlyModel ? 0 : modelWeightsFp32 * 2

  // Full checkpoint size: model + optimizer + small overhead
  const checkpointSize = modelWeightsFp32 + optimizerStateSize + 0.01

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
    mode: 'training',
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
 * Check that all dataset paths referenced in code are attached to the kernel
 *
 * CRITICAL: Kaggle kernels can only access datasets explicitly attached
 * in kernel-metadata.json dataset_sources. If a notebook references
 * /kaggle/input/my-dataset/ but it's not attached, the kernel will fail
 * with FileNotFoundError.
 */
function checkDatasetSources(content: string, metadataPath?: string): CheckResult[] {
  const results: CheckResult[] = []

  // Extract all dataset references from code
  const referencedDatasets = extractDatasetReferences(content)

  if (referencedDatasets.length === 0) {
    return results // No dataset references found
  }

  // Check for competition source references (these are attached separately)
  const competitionSlugs = ['deep-past-initiative-machine-translation']

  // Get attached datasets from metadata
  let attachedDatasets: string[] = []
  let attachedCompetitions: string[] = []
  let attachedModels: string[] = []
  let attachedKernels: string[] = []
  let hasMetadata = false

  if (metadataPath && existsSync(metadataPath)) {
    try {
      const metadata = JSON.parse(readFileSync(metadataPath, 'utf-8'))
      attachedDatasets = (metadata.dataset_sources || []).map((ds: string) => {
        // Extract slug from full path like "manwithacat/dataset-name"
        const parts = ds.split('/')
        return parts.length > 1 ? parts[1] : parts[0]
      })
      attachedCompetitions = metadata.competition_sources || []
      // Extract model slugs from model_sources (format: owner/model/Framework/variation/version)
      attachedModels = (metadata.model_sources || []).map((ms: string) => {
        const parts = ms.split('/')
        return parts.length > 1 ? parts[1] : parts[0]
      })
      // Extract kernel slugs from kernel_sources (format: owner/kernel-name)
      // Kernel outputs are available at /kaggle/input/{kernel-slug}/
      attachedKernels = (metadata.kernel_sources || []).map((ks: string) => {
        const parts = ks.split('/')
        return parts.length > 1 ? parts[1] : parts[0]
      })
      hasMetadata = true
    } catch {
      // Ignore parse errors
    }
  }

  // Find missing datasets
  const missingDatasets = referencedDatasets.filter((ds) => {
    // Check if it's a competition source
    if (competitionSlugs.includes(ds)) {
      return !attachedCompetitions.includes(ds)
    }
    // Check if it's attached as a model (models appear at /kaggle/input/<model-slug>/...)
    if (attachedModels.includes(ds)) {
      return false
    }
    // Check if it's attached as a kernel output (kernel outputs at /kaggle/input/<kernel-slug>/...)
    if (attachedKernels.includes(ds)) {
      return false
    }
    // Check if it's attached as a dataset
    return !attachedDatasets.includes(ds)
  })

  if (missingDatasets.length > 0) {
    results.push({
      check: 'Dataset Sources',
      status: 'fail',
      message: `Dataset(s) referenced in code but not attached: ${missingDatasets.join(', ')}`,
      details: {
        referenced: referencedDatasets,
        attached_datasets: attachedDatasets,
        attached_competitions: attachedCompetitions,
        missing: missingDatasets,
        fix: hasMetadata
          ? `Add to kernel-metadata.json dataset_sources: ${missingDatasets.map((ds) => `"manwithacat/${ds}"`).join(', ')}`
          : `Use --datasets flag: akk kaggle upload-notebook ... --datasets ${missingDatasets.map((ds) => `manwithacat/${ds}`).join(',')}`,
        metadata_file: metadataPath || 'not found',
        note: 'Kaggle kernels can only access datasets explicitly attached in metadata',
      },
    })
  } else if (referencedDatasets.length > 0) {
    results.push({
      check: 'Dataset Sources',
      status: 'pass',
      message: `All ${referencedDatasets.length} referenced dataset(s) are attached`,
      details: {
        datasets: referencedDatasets,
      },
    })
  }

  return results
}

/**
 * Check that model_sources in kernel-metadata.json uses the full path format.
 *
 * Kaggle Models require the full path format to be attached properly:
 *   CORRECT: owner/model-name/Framework/variation/version
 *   WRONG:   owner/model-name (will NOT be attached to the kernel!)
 *
 * This is a critical check because the Kaggle API silently accepts the short format
 * but the model won't actually be mounted in the kernel environment.
 */
function checkModelSourcesFormat(content: string, metadataPath?: string): CheckResult[] {
  const results: CheckResult[] = []

  // Extract model path references from code
  // Pattern: /kaggle/input/{model-slug}/{framework}/{variation}/{version}
  const modelPathPattern = /\/kaggle\/input\/([a-z0-9-]+)\/([a-z]+)\/([a-z0-9-]+)\/(\d+)/gi
  const modelRefs: string[] = []

  for (const match of content.matchAll(modelPathPattern)) {
    const [, modelSlug] = match
    if (modelSlug && !modelRefs.includes(modelSlug)) {
      modelRefs.push(modelSlug)
    }
  }

  // If no model paths referenced, no check needed
  if (modelRefs.length === 0) {
    return results
  }

  // Get model_sources from metadata
  if (!metadataPath || !existsSync(metadataPath)) {
    results.push({
      check: 'Model Sources',
      status: 'warn',
      message: `Model path(s) referenced but no kernel-metadata.json found`,
      details: {
        referenced_models: modelRefs,
        fix: 'Create kernel-metadata.json with model_sources array',
      },
    })
    return results
  }

  try {
    const metadata = JSON.parse(readFileSync(metadataPath, 'utf-8'))
    const modelSources: string[] = metadata.model_sources || []

    if (modelSources.length === 0) {
      results.push({
        check: 'Model Sources',
        status: 'fail',
        message: `Model path(s) referenced in code but model_sources is empty`,
        details: {
          referenced_models: modelRefs,
          fix: 'Add model_sources to kernel-metadata.json with full path format: owner/model-name/Framework/variation/version',
          example: 'manwithacat/byt5-skeleton-akkadian/Transformers/transformers/1',
        },
      })
      return results
    }

    // Check each model_source for correct format
    // Full format: owner/model-name/Framework/variation/version (5 parts)
    // Short format: owner/model-name (2 parts) - WILL NOT WORK
    const fullPathPattern = /^[a-z0-9_-]+\/[a-z0-9_-]+\/[A-Za-z]+\/[a-z0-9_-]+\/\d+$/
    const shortPathPattern = /^[a-z0-9_-]+\/[a-z0-9_-]+$/

    const invalidSources: { source: string; issue: string }[] = []
    const validSources: string[] = []

    for (const source of modelSources) {
      if (fullPathPattern.test(source)) {
        validSources.push(source)
      } else if (shortPathPattern.test(source)) {
        invalidSources.push({
          source,
          issue: 'Short format (owner/model) - model will NOT be attached!',
        })
      } else {
        invalidSources.push({
          source,
          issue: 'Invalid format',
        })
      }
    }

    if (invalidSources.length > 0) {
      results.push({
        check: 'Model Sources Format',
        status: 'fail',
        message: `model_sources uses incorrect format - models will NOT be attached`,
        details: {
          invalid_sources: invalidSources,
          valid_sources: validSources,
          required_format: 'owner/model-name/Framework/variation/version',
          example: 'manwithacat/byt5-skeleton-akkadian/Transformers/transformers/1',
          note: 'Kaggle silently accepts short format but model is NOT mounted. Use full path with Framework/variation/version.',
          fix: `Update model_sources in ${metadataPath}`,
        },
      })
    } else if (validSources.length > 0) {
      // Check if referenced models are actually attached
      const attachedModelSlugs = validSources.map((s) => s.split('/')[1])
      const missingModels = modelRefs.filter((ref) => !attachedModelSlugs.includes(ref))

      if (missingModels.length > 0) {
        results.push({
          check: 'Model Sources',
          status: 'fail',
          message: `Model(s) referenced in code but not in model_sources: ${missingModels.join(', ')}`,
          details: {
            referenced_models: modelRefs,
            attached_models: attachedModelSlugs,
            missing: missingModels,
            fix: `Add missing models to model_sources in ${metadataPath}`,
          },
        })
      } else {
        results.push({
          check: 'Model Sources',
          status: 'pass',
          message: `All ${modelRefs.length} model(s) attached with correct format`,
          details: {
            models: validSources,
          },
        })
      }
    }
  } catch {
    // Ignore parse errors
  }

  return results
}

/**
 * Verify that model_sources actually exist on Kaggle.
 *
 * This makes API calls to verify each model exists, preventing the common issue
 * where a kernel is uploaded with a model reference that doesn't exist or
 * uses an incorrect path format.
 */
async function verifyModelSourcesExist(metadataPath?: string): Promise<CheckResult[]> {
  const results: CheckResult[] = []

  if (!metadataPath || !existsSync(metadataPath)) {
    return results
  }

  try {
    const metadata = JSON.parse(readFileSync(metadataPath, 'utf-8'))
    const modelSources: string[] = metadata.model_sources || []

    if (modelSources.length === 0) {
      return results
    }

    // Verify each model source exists
    const verificationResults: {
      source: string
      exists: boolean
      error?: string
    }[] = []

    for (const source of modelSources) {
      // Only verify full-format sources (owner/model/framework/instance/version)
      const parts = source.split('/')
      if (parts.length < 5) {
        continue // Skip short format, already flagged by format check
      }

      // Use full version path (owner/model/framework/instance/version)
      // Note: We use the full path because `kaggle models instances files`
      // has a bug, but `kaggle models instances versions files` works correctly
      const versionPath = source

      try {
        const result = await getModelInstanceFiles(versionPath)
        verificationResults.push({
          source,
          exists: result.success,
          error: result.success ? undefined : result.message,
        })
      } catch (err) {
        verificationResults.push({
          source,
          exists: false,
          error: err instanceof Error ? err.message : 'Unknown error',
        })
      }
    }

    const missingModels = verificationResults.filter((r) => !r.exists)
    const existingModels = verificationResults.filter((r) => r.exists)

    if (missingModels.length > 0) {
      results.push({
        check: 'Model Existence',
        status: 'fail',
        message: `${missingModels.length} model(s) not found on Kaggle`,
        details: {
          missing: missingModels.map((m) => ({
            source: m.source,
            error: m.error,
          })),
          existing: existingModels.map((m) => m.source),
          fix: "Verify model paths are correct and models are published. Use 'kaggle models instances files <path>' to check.",
        },
      })
    } else if (existingModels.length > 0) {
      results.push({
        check: 'Model Existence',
        status: 'pass',
        message: `All ${existingModels.length} model(s) verified on Kaggle`,
        details: {
          models: existingModels.map((m) => m.source),
        },
      })
    }
  } catch {
    // Ignore parse errors
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
        if (
          col &&
          ![
            // DataFrame methods/attributes
            'iloc',
            'loc',
            'values',
            'index',
            'columns',
            'head',
            'tail',
            'shape',
            // PyTorch tensor keys (from tokenization, not DataFrame columns)
            'input_ids',
            'attention_mask',
            'labels',
            'decoder_input_ids',
            'decoder_attention_mask',
            'token_type_ids',
            'pixel_values',
            'logits',
            'loss',
          ].includes(col)
        ) {
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

  // Extract column rename targets - columns created by .rename(columns={"old": "new"})
  // These are valid even if they don't exist in the original dataset
  const renamedColumns = new Set<string>()
  const renamePattern = /\.rename\s*\(\s*columns\s*=\s*\{([^}]+)\}/g
  let renameMatch = renamePattern.exec(content)
  while (renameMatch !== null) {
    // Extract the target column names (values in the dict)
    // Pattern: "old": "new" or 'old': 'new'
    const dictContent = renameMatch[1]
    const targetPattern = /["'][^"']+["']\s*:\s*["']([^"']+)["']/g
    let targetMatch = targetPattern.exec(dictContent)
    while (targetMatch !== null) {
      renamedColumns.add(targetMatch[1])
      targetMatch = targetPattern.exec(dictContent)
    }
    renameMatch = renamePattern.exec(content)
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
        ['id', 'source', 'target', 'text', 'label', 'input', 'output'].includes(col) ||
        // Allow columns created via .rename(columns={"old": "new"})
        renamedColumns.has(col)

      if (!isDatasetColumn && referencedColumns.has(col)) {
        // Check if this column is specifically accessed on data from this dataset
        // by looking for the column reference near the dataset reference
        const datasetIndex = content.indexOf(datasetSlug)
        const columnPattern = new RegExp(`\\[["']${col}["']\\]`)
        const columnMatch = content.match(columnPattern)

        if (columnMatch && datasetIndex !== -1) {
          // Simple heuristic: if column is referenced and dataset is used, validate
          if (!schema.columns.includes(col) && !renamedColumns.has(col)) {
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

  // Check for pip install commands (both shell-style and subprocess-style)
  const shellPipInstalls = content.match(/!pip\s+install\s+[^\n]+/g) || []
  const subprocessPipInstalls = content.match(/subprocess\.\w+\s*\([^)]*pip[^)]*install[^)]*\)/g) || []
  const subprocessCheckCall = content.match(/subprocess\.check_call\s*\([^)]*pip[^)]*install[^)]*\)/g) || []
  const allPipInstalls = [...shellPipInstalls, ...subprocessPipInstalls, ...subprocessCheckCall]

  // If internet is disabled and ANY pip install is detected, it will fail
  if (!internetEnabled && hasMetadata && allPipInstalls.length > 0) {
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
    // Internet enabled (or no metadata) - warn about potential submission issues
    results.push({
      check: 'Pip Install Warning',
      status: 'warn',
      message: 'pip install detected - will fail if converted to competition submission (internet disabled)',
      details: {
        detected: allPipInstalls.slice(0, 3).map((s) => s.slice(0, 60) + '...'),
        note: 'Training kernels can use pip install, but submission kernels require enable_internet=false',
        preinstalled: 'Run `akk help kaggle-packages` to see pre-installed packages',
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
 * Check for AutoTokenizer/AutoModel with local Kaggle Model paths
 *
 * ISSUE: Newer versions of HuggingFace transformers validate the path argument
 * before checking if it's a local directory. Paths like:
 *   /kaggle/input/model-name/pytorch/transformers/1
 *
 * Are rejected with:
 *   HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'
 *
 * FIX: Use model-specific classes (T5Tokenizer, T5ForConditionalGeneration) instead
 * of Auto classes, or use the model config to load explicitly.
 */
function checkAutoClassWithLocalPath(content: string): CheckResult[] {
  const results: CheckResult[] = []

  // Find Auto class usage with Kaggle Model paths
  // Pattern: AutoTokenizer.from_pretrained(PATH) where PATH is a variable containing /kaggle/input/
  const autoClassPatterns = [
    /AutoTokenizer\.from_pretrained\s*\(\s*([A-Z_]+)\s*(?:,|\))/g,
    /AutoModelForSeq2SeqLM\.from_pretrained\s*\(\s*([A-Z_]+)\s*(?:,|\))/g,
    /AutoModel\.from_pretrained\s*\(\s*([A-Z_]+)\s*(?:,|\))/g,
    /AutoModelForCausalLM\.from_pretrained\s*\(\s*([A-Z_]+)\s*(?:,|\))/g,
  ]

  // Collect variable names used in Auto class calls
  const autoClassVars = new Set<string>()
  for (const pattern of autoClassPatterns) {
    let match
    while ((match = pattern.exec(content)) !== null) {
      autoClassVars.add(match[1])
    }
  }

  // Check if any of these variables are Kaggle Model paths (deep paths with multiple segments)
  // Kaggle Model paths look like: /kaggle/input/model-slug/pytorch/transformers/1
  const kaggleModelPathPattern = /([A-Z_]+)\s*=\s*["']?(\/kaggle\/input\/[^"'\s]+\/[^"'\s]+\/[^"'\s]+\/\d+)["']?/g
  const kaggleModelPaths: Array<{ varName: string; path: string }> = []

  let pathMatch
  while ((pathMatch = kaggleModelPathPattern.exec(content)) !== null) {
    kaggleModelPaths.push({ varName: pathMatch[1], path: pathMatch[2] })
  }

  // Find Auto class calls using Kaggle Model path variables
  const problematicCalls: Array<{ varName: string; path: string }> = []
  for (const { varName, path } of kaggleModelPaths) {
    if (autoClassVars.has(varName)) {
      problematicCalls.push({ varName, path })
    }
  }

  // Also check for direct string literals
  const directAutoClassPatterns = [
    /AutoTokenizer\.from_pretrained\s*\(\s*["'](\/kaggle\/input\/[^"']+\/[^"']+\/[^"']+\/\d+)["']/g,
    /AutoModelForSeq2SeqLM\.from_pretrained\s*\(\s*["'](\/kaggle\/input\/[^"']+\/[^"']+\/[^"']+\/\d+)["']/g,
  ]

  for (const pattern of directAutoClassPatterns) {
    let match
    while ((match = pattern.exec(content)) !== null) {
      problematicCalls.push({ varName: 'direct', path: match[1] })
    }
  }

  if (problematicCalls.length > 0) {
    // Detect model type from path to suggest correct class
    const modelTypeHints: Record<string, { tokenizer: string; model: string }> = {
      t5: { tokenizer: 'T5Tokenizer', model: 'T5ForConditionalGeneration' },
      'flan-t5': {
        tokenizer: 'T5Tokenizer',
        model: 'T5ForConditionalGeneration',
      },
      byt5: {
        tokenizer: 'ByT5Tokenizer',
        model: 'T5ForConditionalGeneration',
      },
      bart: {
        tokenizer: 'BartTokenizer',
        model: 'BartForConditionalGeneration',
      },
      mbart: {
        tokenizer: 'MBartTokenizer',
        model: 'MBartForConditionalGeneration',
      },
      nllb: { tokenizer: 'NllbTokenizer', model: 'AutoModelForSeq2SeqLM' },
    }

    // Try to detect model type from path
    const path = problematicCalls[0].path.toLowerCase()
    let suggestion = {
      tokenizer: 'T5Tokenizer',
      model: 'T5ForConditionalGeneration',
    } // default
    for (const [hint, classes] of Object.entries(modelTypeHints)) {
      if (path.includes(hint)) {
        suggestion = classes
        break
      }
    }

    results.push({
      check: 'Auto Class with Local Path',
      status: 'fail',
      message: 'AutoTokenizer/AutoModel with Kaggle Model paths will fail due to HF repo ID validation',
      details: {
        problematic_paths: problematicCalls.map((p) => p.path),
        error: "HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'",
        reason: 'Newer transformers validates paths as HuggingFace repo IDs before checking if local',
        fix: `Use model-specific classes instead of Auto classes:
  from transformers import ${suggestion.tokenizer}, ${suggestion.model}
  tokenizer = ${suggestion.tokenizer}.from_pretrained(MODEL_PATH)
  model = ${suggestion.model}.from_pretrained(MODEL_PATH)`,
        note: 'This issue affects Kaggle Models with deep paths like /kaggle/input/model/pytorch/transformers/1',
      },
    })
  }

  return results
}

/**
 * Check for model-specific classes used with Kaggle Model paths without local_files_only=True
 *
 * Even model-specific classes like T5Tokenizer, T5ForConditionalGeneration, etc.
 * will fail with HF repo ID validation when loading from Kaggle Model paths
 * unless local_files_only=True is specified.
 *
 * Kaggle Model paths look like: /kaggle/input/model-slug/pytorch/transformers/1
 *
 * FIX: Add local_files_only=True to the from_pretrained() call
 */
function checkLocalFilesOnlyMissing(content: string): CheckResult[] {
  const results: CheckResult[] = []

  // Model-specific classes that need local_files_only=True for Kaggle Model paths
  const modelClasses = [
    'T5Tokenizer',
    'T5ForConditionalGeneration',
    'ByT5Tokenizer',
    'BartTokenizer',
    'BartForConditionalGeneration',
    'MBartTokenizer',
    'MBartForConditionalGeneration',
    'NllbTokenizer',
    'GPT2Tokenizer',
    'GPT2LMHeadModel',
    'BertTokenizer',
    'BertModel',
    'RobertaTokenizer',
    'RobertaModel',
  ]

  // Find Kaggle Model path variables (deep paths with /pytorch/transformers/1 or similar)
  // Handle various assignment patterns:
  // 1. VAR = "/kaggle/input/.../1"
  // 2. VAR = (\n    "/kaggle/input/.../1"\n)
  // 3. var_name = "/kaggle/input/.../1"
  const kaggleModelPaths = new Map<string, string>()

  // Pattern 1: Single-line assignment
  const singleLinePattern = /([A-Z_][A-Z0-9_]*)\s*=\s*["'](\/kaggle\/input\/[^"']+\/[^"']+\/[^"']+\/\d+)["']/gi
  let pathMatch
  while ((pathMatch = singleLinePattern.exec(content)) !== null) {
    kaggleModelPaths.set(pathMatch[1], pathMatch[2])
  }

  // Pattern 2: Multi-line with parentheses (common Python formatting)
  const multiLinePattern = /([A-Z_][A-Z0-9_]*)\s*=\s*\(\s*["'](\/kaggle\/input\/[^"']+\/[^"']+\/[^"']+\/\d+)["']\s*\)/gi
  while ((pathMatch = multiLinePattern.exec(content)) !== null) {
    kaggleModelPaths.set(pathMatch[1], pathMatch[2])
  }

  // Pattern 3: Lowercase variable names (single-line)
  const lowerSinglePattern = /([a-z_][a-z0-9_]*)\s*=\s*["'](\/kaggle\/input\/[^"']+\/[^"']+\/[^"']+\/\d+)["']/gi
  while ((pathMatch = lowerSinglePattern.exec(content)) !== null) {
    kaggleModelPaths.set(pathMatch[1], pathMatch[2])
  }

  // Pattern 4: Lowercase variable names (multi-line)
  const lowerMultiPattern =
    /([a-z_][a-z0-9_]*)\s*=\s*\(\s*["'](\/kaggle\/input\/[^"']+\/[^"']+\/[^"']+\/\d+)["']\s*\)/gi
  while ((pathMatch = lowerMultiPattern.exec(content)) !== null) {
    kaggleModelPaths.set(pathMatch[1], pathMatch[2])
  }

  if (kaggleModelPaths.size === 0) {
    return results // No Kaggle Model paths detected
  }

  // Check each model class for from_pretrained calls without local_files_only
  const problematicCalls: Array<{
    className: string
    path: string
    line: string
  }> = []

  for (const className of modelClasses) {
    // Pattern: ClassName.from_pretrained(VAR_NAME or "path", ...) without local_files_only
    const classPattern = new RegExp(`(${className}\\.from_pretrained\\s*\\([^)]+\\))`, 'g')

    let classMatch
    while ((classMatch = classPattern.exec(content)) !== null) {
      const fullCall = classMatch[1]

      // Check if this call uses a Kaggle Model path variable
      let usesKagglePath = false
      let pathUsed = ''

      for (const [varName, path] of kaggleModelPaths) {
        // Check if variable name is used in the call (case insensitive for matching)
        if (fullCall.includes(varName) || fullCall.toLowerCase().includes(varName.toLowerCase())) {
          usesKagglePath = true
          pathUsed = path
          break
        }
      }

      // Also check for direct string literals
      const directPathMatch = fullCall.match(/["'](\/kaggle\/input\/[^"']+\/[^"']+\/[^"']+\/\d+)["']/)
      if (directPathMatch) {
        usesKagglePath = true
        pathUsed = directPathMatch[1]
      }

      if (usesKagglePath) {
        // Check if local_files_only=True is present
        if (!fullCall.includes('local_files_only')) {
          problematicCalls.push({
            className,
            path: pathUsed,
            line: fullCall.slice(0, 100),
          })
        }
      }
    }
  }

  if (problematicCalls.length > 0) {
    const uniqueClasses = [...new Set(problematicCalls.map((c) => c.className))]
    const uniquePaths = [...new Set(problematicCalls.map((c) => c.path))]

    results.push({
      check: 'Local Files Only Missing',
      status: 'fail',
      message: 'Model classes with Kaggle Model paths will fail without local_files_only=True',
      details: {
        classes: uniqueClasses,
        paths: uniquePaths,
        error: "HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'",
        reason:
          'Newer transformers validates paths as HuggingFace repo IDs before checking if local, even for model-specific classes',
        fix: `Add local_files_only=True to from_pretrained() calls:
  tokenizer = ${uniqueClasses[0]}.from_pretrained(MODEL_PATH, local_files_only=True)
  model = ${uniqueClasses.length > 1 ? uniqueClasses[1] : uniqueClasses[0]}.from_pretrained(MODEL_PATH, local_files_only=True)`,
        note: 'This bypasses HuggingFace Hub validation and loads directly from local path',
      },
    })
  }

  return results
}

/**
 * Detect sequential model loading pattern for multi-stage pipelines
 *
 * Sequential loading pattern:
 * 1. Load model A
 * 2. Run inference
 * 3. Delete model A (del model + gc.collect() + torch.cuda.empty_cache())
 * 4. Load model B
 *
 * This pattern allows larger models to fit in limited VRAM by not loading
 * all models simultaneously.
 */
interface SequentialLoadingInfo {
  isSequential: boolean
  stages: Array<{
    models: string[]
    hasCleanup: boolean
  }>
  peakModels: string[]
  estimatedPeakCount: number
}

function detectSequentialLoading(content: string): SequentialLoadingInfo {
  // Detect memory cleanup patterns
  const cleanupPatterns = [
    /del\s+\w+.*\n.*gc\.collect\s*\(\)/s,
    /gc\.collect\s*\(\).*\n.*torch\.cuda\.empty_cache\s*\(\)/s,
    /del\s+\w+_model.*\n/g,
    /torch\.cuda\.empty_cache\s*\(\)/g,
  ]

  const hasCleanupPattern = cleanupPatterns.some((p) => p.test(content))

  // Find all model loading calls
  const modelLoadPattern =
    /(\w+)\s*=\s*(?:AutoModelForSeq2SeqLM|T5ForConditionalGeneration|AutoModel\w*)\.from_pretrained/g
  const modelLoads: Array<{ varName: string; position: number }> = []
  let match
  while ((match = modelLoadPattern.exec(content)) !== null) {
    modelLoads.push({ varName: match[1], position: match.index })
  }

  // Find all model deletions
  const modelDelPattern = /del\s+(\w+)(?:\s*,\s*(\w+))*/g
  const modelDeletions: Array<{ varNames: string[]; position: number }> = []
  while ((match = modelDelPattern.exec(content)) !== null) {
    const fullMatch = match[0]
    const vars = fullMatch
      .replace('del ', '')
      .split(',')
      .map((v) => v.trim())
    modelDeletions.push({ varNames: vars, position: match.index })
  }

  // Find gc.collect() + empty_cache() calls
  const gcPattern = /gc\.collect\s*\(\)/g
  const gcCalls: number[] = []
  while ((match = gcPattern.exec(content)) !== null) {
    gcCalls.push(match.index)
  }

  const emptyCachePattern = /torch\.cuda\.empty_cache\s*\(\)/g
  const emptyCacheCalls: number[] = []
  while ((match = emptyCachePattern.exec(content)) !== null) {
    emptyCacheCalls.push(match.index)
  }

  // Determine if this is sequential loading
  // Heuristics:
  // 1. Multiple model loads exist
  // 2. Deletions exist between loads
  // 3. gc.collect() and/or empty_cache() are called

  if (modelLoads.length <= 1) {
    return {
      isSequential: false,
      stages: [{ models: modelLoads.map((m) => m.varName), hasCleanup: false }],
      peakModels: modelLoads.map((m) => m.varName),
      estimatedPeakCount: modelLoads.length,
    }
  }

  // Check for cleanup between model loads
  const stages: Array<{ models: string[]; hasCleanup: boolean }> = []
  let currentStageModels: string[] = []
  let lastLoadPosition = 0

  for (let i = 0; i < modelLoads.length; i++) {
    const load = modelLoads[i]

    // Check if there's a deletion + gc.collect between previous load and this one
    const hasCleanupBetween = modelDeletions.some(
      (d) =>
        d.position > lastLoadPosition &&
        d.position < load.position &&
        gcCalls.some((gc) => gc > d.position && gc < load.position)
    )

    if (hasCleanupBetween && currentStageModels.length > 0) {
      // End current stage, start new one
      stages.push({ models: currentStageModels, hasCleanup: true })
      currentStageModels = [load.varName]
    } else {
      currentStageModels.push(load.varName)
    }

    lastLoadPosition = load.position
  }

  // Add final stage
  if (currentStageModels.length > 0) {
    stages.push({ models: currentStageModels, hasCleanup: false })
  }

  // Determine peak models (max models loaded simultaneously in any stage)
  const peakStage = stages.reduce((max, stage) => (stage.models.length > max.models.length ? stage : max), stages[0])

  const isSequential = stages.length > 1 && hasCleanupPattern

  return {
    isSequential,
    stages,
    peakModels: peakStage.models,
    estimatedPeakCount: peakStage.models.length,
  }
}

/**
 * Extract all model paths from content for sequential memory estimation
 */
function extractAllModelPaths(content: string): string[] {
  const modelPaths: string[] = []

  // Model path variable patterns
  const pathPatterns = [
    /([A-Z_]*MODEL[A-Z_]*_PATH|[A-Z_]*_PATH|MODEL\d*_PATH)\s*=\s*["']([^"']+)["']/g,
    /([A-Z_]*PHILOLOGIST[A-Z_]*)\s*=\s*["']([^"']+)["']/g,
  ]

  for (const pattern of pathPatterns) {
    let match
    while ((match = pattern.exec(content)) !== null) {
      const path = match[2]
      if (
        path.includes('/kaggle/input/') ||
        path.includes('google/') ||
        path.includes('facebook/') ||
        path.includes('model')
      ) {
        modelPaths.push(path)
      }
    }
  }

  // Also extract from direct from_pretrained calls
  const fromPretrainedPattern = /from_pretrained\s*\(\s*["']([^"']+)["']/g
  let match
  while ((match = fromPretrainedPattern.exec(content)) !== null) {
    const path = match[1]
    if (!modelPaths.includes(path)) {
      modelPaths.push(path)
    }
  }

  return modelPaths
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

    // 0.14. Dataset Sources Check (CRITICAL - datasets referenced must be attached)
    const datasetSourcesResults = checkDatasetSources(content, metadataPath)
    checks.push(...datasetSourcesResults)

    // 0.15. Model Sources Format Check (CRITICAL - must use full path format)
    const modelSourcesResults = checkModelSourcesFormat(content, metadataPath)
    checks.push(...modelSourcesResults)

    // 0.16. Model Existence Verification (verify models exist on Kaggle)
    // This makes API calls so only run if format check passed
    const formatPassed = !modelSourcesResults.some((r) => r.status === 'fail')
    if (formatPassed && metadataPath) {
      const modelExistenceResults = await verifyModelSourcesExist(metadataPath)
      checks.push(...modelExistenceResults)
    }

    // 0.17. Auto Class with Local Path Check (HF repo ID validation issue)
    const autoClassResults = checkAutoClassWithLocalPath(content)
    checks.push(...autoClassResults)

    // 0.17. Local Files Only Missing Check (model-specific classes need local_files_only=True)
    const localFilesOnlyResults = checkLocalFilesOnlyMissing(content)
    checks.push(...localFilesOnlyResults)

    // Detect if this is inference-only (no training)
    const isInferenceMode = detectInferenceMode(content)

    // Detect sequential loading pattern for multi-stage pipelines
    const sequentialInfo = detectSequentialLoading(content)
    const allModelPaths = extractAllModelPaths(content)

    // 1. GPU Memory Check (with sequential loading awareness)
    const gpuEstimate = estimateGpuMemory(config, isInferenceMode, sequentialInfo, allModelPaths)
    const gpuStatus =
      gpuEstimate.peak_gb <= platform.gpu.vram_gb * 0.9
        ? 'pass'
        : gpuEstimate.peak_gb <= platform.gpu.vram_gb
          ? 'warn'
          : 'fail'
    const modeLabel = gpuEstimate.mode === 'inference' ? ' (inference)' : ' (training)'
    checks.push({
      check: 'GPU Memory',
      status: gpuStatus,
      message:
        gpuStatus === 'fail'
          ? `Estimated ${gpuEstimate.peak_gb.toFixed(1)}GB exceeds ${platform.gpu.vram_gb}GB VRAM${modeLabel}`
          : gpuStatus === 'warn'
            ? `Estimated ${gpuEstimate.peak_gb.toFixed(1)}GB is close to ${platform.gpu.vram_gb}GB limit${modeLabel}`
            : `Estimated ${gpuEstimate.peak_gb.toFixed(1)}GB fits in ${platform.gpu.vram_gb}GB VRAM${modeLabel}`,
      details: args.verbose
        ? {
            ...gpuEstimate.breakdown,
            mode: gpuEstimate.mode,
            model: gpuEstimate.model_detected,
            ...(gpuEstimate.sequential_loading && {
              sequential_loading: gpuEstimate.sequential_loading,
            }),
          }
        : undefined,
    })

    // 2. Disk Space Check (use peak_gb for actual limit, as checkpoint rotation creates temporary spikes)
    const diskEstimate = estimateDiskUsage(config, isInferenceMode)
    const diskStatus =
      diskEstimate.peak_gb <= platform.disk.working_gb * 0.8
        ? 'pass'
        : diskEstimate.peak_gb <= platform.disk.working_gb
          ? 'warn'
          : 'fail'
    const diskModeLabel = diskEstimate.mode === 'inference' ? ' (inference)' : ' (training)'
    checks.push({
      check: 'Disk Space',
      status: diskStatus,
      message:
        diskStatus === 'fail'
          ? `Peak ${diskEstimate.peak_gb.toFixed(1)}GB exceeds ${platform.disk.working_gb}GB working space${diskModeLabel}`
          : diskStatus === 'warn'
            ? `Peak ${diskEstimate.peak_gb.toFixed(1)}GB is close to ${platform.disk.working_gb}GB limit${diskModeLabel}`
            : `Peak ${diskEstimate.peak_gb.toFixed(1)}GB fits in ${platform.disk.working_gb}GB working space${diskModeLabel}`,
      details: args.verbose
        ? {
            ...diskEstimate.breakdown,
            peak_gb: diskEstimate.peak_gb,
            mode: diskEstimate.mode,
          }
        : undefined,
    })

    // 3. Training Time Check (skip for inference-only kernels)
    if (!isInferenceMode) {
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
    } else {
      checks.push({
        check: 'Training Time',
        status: 'pass',
        message: 'N/A (inference-only kernel)',
      })
    }

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

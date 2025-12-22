/**
 * ML Tools Integration Types
 *
 * Defines types for optional ML tool integrations that can be
 * injected into training/inference notebooks.
 */

import { z } from 'zod'

/**
 * Available tool integrations
 */
export type ToolId =
  | 'sacrebleu'      // Reproducible MT evaluation
  | 'qlora'          // QLoRA (4-bit quantization + LoRA)
  | 'accelerate'     // Distributed training wrapper
  | 'onnx'           // ONNX export for inference
  | 'streaming'      // Streaming dataloader for large datasets
  | 'hpo'            // Hyperparameter optimization (Optuna + MLflow)

/**
 * Tool configuration options
 */
export interface ToolConfig {
  sacrebleu?: {
    tokenizer?: '13a' | 'intl' | 'flores200' | 'zh' | 'ja-mecab' | 'ko-mecab'
    metrics?: ('bleu' | 'chrf' | 'ter')[]
    lowercase?: boolean
  }
  qlora?: {
    bits?: 4 | 8
    quant_type?: 'nf4' | 'fp4'
    double_quant?: boolean
    lora_r?: number
    lora_alpha?: number
    lora_dropout?: number
    target_modules?: string[] | 'all-linear'
  }
  accelerate?: {
    mixed_precision?: 'no' | 'fp16' | 'bf16'
    gradient_checkpointing?: boolean
    gradient_accumulation_steps?: number
  }
  onnx?: {
    optimization_level?: 'O1' | 'O2' | 'O3' | 'O4'
    quantize?: boolean
  }
  streaming?: {
    buffer_size?: number
    num_workers?: number
  }
  hpo?: {
    n_trials?: number
    metric?: string
    direction?: 'minimize' | 'maximize'
    search_space?: {
      learning_rate?: { min: number; max: number; log?: boolean }
      batch_size?: number[]
      weight_decay?: { min: number; max: number }
      warmup_ratio?: { min: number; max: number }
      lora_r?: number[]
      lora_alpha?: number[]
    }
  }
}

/**
 * Tool metadata
 */
export interface ToolMetadata {
  id: ToolId
  name: string
  description: string
  dependencies: string[]
  incompatible_with?: ToolId[]
  requires?: ToolId[]
}

/**
 * Tool registry with metadata
 */
export const TOOL_REGISTRY: Record<ToolId, ToolMetadata> = {
  sacrebleu: {
    id: 'sacrebleu',
    name: 'SacreBLEU Evaluation',
    description: 'Reproducible MT evaluation with BLEU, chrF, and TER metrics',
    dependencies: ['sacrebleu>=2.3.0'],
  },
  qlora: {
    id: 'qlora',
    name: 'QLoRA Training',
    description: '4-bit quantized training with LoRA adapters for memory efficiency',
    dependencies: ['bitsandbytes>=0.41.0', 'peft>=0.6.0'],
  },
  accelerate: {
    id: 'accelerate',
    name: 'Accelerate Distributed',
    description: 'Distributed training wrapper for multi-GPU and mixed precision',
    dependencies: ['accelerate>=0.24.0'],
  },
  onnx: {
    id: 'onnx',
    name: 'ONNX Export',
    description: 'Export and optimize models for production inference',
    dependencies: ['optimum[onnxruntime]>=1.14.0'],
  },
  streaming: {
    id: 'streaming',
    name: 'Streaming DataLoader',
    description: 'Memory-efficient data loading for large datasets',
    dependencies: ['datasets>=2.14.0'],
  },
  hpo: {
    id: 'hpo',
    name: 'Hyperparameter Optimization',
    description: 'Automated hyperparameter search with Optuna and MLflow tracking',
    dependencies: ['optuna>=3.4.0', 'mlflow>=2.8.0'],
  },
}

/**
 * Default tool configurations
 */
export const DEFAULT_TOOL_CONFIG: ToolConfig = {
  sacrebleu: {
    tokenizer: '13a',
    metrics: ['bleu', 'chrf'],
    lowercase: false,
  },
  qlora: {
    bits: 4,
    quant_type: 'nf4',
    double_quant: true,
    lora_r: 16,
    lora_alpha: 32,
    lora_dropout: 0.05,
    target_modules: ['q', 'v'],
  },
  accelerate: {
    mixed_precision: 'fp16',
    gradient_checkpointing: true,
    gradient_accumulation_steps: 4,
  },
  onnx: {
    optimization_level: 'O2',
    quantize: false,
  },
  streaming: {
    buffer_size: 10000,
    num_workers: 4,
  },
  hpo: {
    n_trials: 20,
    metric: 'eval_bleu',
    direction: 'maximize',
    search_space: {
      learning_rate: { min: 1e-5, max: 5e-4, log: true },
      batch_size: [2, 4, 8],
      weight_decay: { min: 0.0, max: 0.1 },
      warmup_ratio: { min: 0.0, max: 0.2 },
    },
  },
}

/**
 * Get dependencies for selected tools
 */
export function getToolDependencies(tools: ToolId[]): string[] {
  const deps = new Set<string>()
  for (const tool of tools) {
    const meta = TOOL_REGISTRY[tool]
    if (meta) {
      meta.dependencies.forEach(d => deps.add(d))
    }
  }
  return Array.from(deps)
}

/**
 * Check tool compatibility
 */
export function checkToolCompatibility(tools: ToolId[]): { valid: boolean; errors: string[] } {
  const errors: string[] = []

  for (const tool of tools) {
    const meta = TOOL_REGISTRY[tool]
    if (!meta) {
      errors.push(`Unknown tool: ${tool}`)
      continue
    }

    // Check incompatibilities
    if (meta.incompatible_with) {
      for (const incompatible of meta.incompatible_with) {
        if (tools.includes(incompatible)) {
          errors.push(`${tool} is incompatible with ${incompatible}`)
        }
      }
    }

    // Check requirements
    if (meta.requires) {
      for (const required of meta.requires) {
        if (!tools.includes(required)) {
          errors.push(`${tool} requires ${required}`)
        }
      }
    }
  }

  return { valid: errors.length === 0, errors }
}

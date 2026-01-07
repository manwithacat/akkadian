/**
 * Template System Types
 *
 * Defines types for the hybrid template system that combines
 * base template files with programmatic cell injection.
 */

import type { PlatformId, PlatformPaths } from './platform'

export type { PlatformPaths }

import type { CompetitionConfig } from './competition'
import type { ToolConfig, ToolId } from './tools'

/**
 * Template variable types
 */
export type TemplateVariableType = 'string' | 'number' | 'boolean' | 'path' | 'array'

/**
 * Template variable definition
 */
export interface TemplateVariable {
  name: string
  type: TemplateVariableType
  required: boolean
  default?: string | number | boolean
  description: string
  platform_specific?: boolean
  validation?: {
    min?: number
    max?: number
    pattern?: string
    enum?: string[]
  }
}

/**
 * Injection point in a template
 */
export interface InjectionPoint {
  name: string
  description: string
  required: boolean
  platforms?: PlatformId[]
}

/**
 * Template metadata extracted from template file headers
 */
export interface TemplateMetadata {
  name: string
  description: string
  version: string
  type: 'training' | 'inference' | 'evaluation' | 'custom'
  platforms: PlatformId[]
  variables: TemplateVariable[]
  injection_points: InjectionPoint[]
  dependencies?: string[]
  author?: string
}

/**
 * Template context for rendering
 */
export interface TemplateContext {
  platform: PlatformId
  competition?: {
    competition: CompetitionConfig['competition']
  }
  model?: {
    name: string
    base?: string
  }
  training?: {
    epochs?: number
    batch_size?: number
    learning_rate?: number
    gradient_accumulation_steps?: number
    max_src_len?: number
    max_tgt_len?: number
    warmup_ratio?: number
    weight_decay?: number
    fp16?: boolean
    save_steps?: number
    eval_steps?: number
    logging_steps?: number
    save_total_limit?: number
  }
  languages?: {
    src?: string
    tgt?: string
  }
  gcs?: {
    bucket?: string
    project?: string
    cross_account?: boolean
  }
  tools?: {
    enabled: ToolId[]
    config?: ToolConfig
  }
  variables?: Record<string, unknown>
}

/**
 * Rendered cell in a notebook
 */
export interface NotebookCell {
  cell_type: 'code' | 'markdown'
  source: string[]
  metadata?: Record<string, unknown>
  execution_count?: number | null
  outputs?: unknown[]
}

/**
 * Notebook content structure
 */
export interface NotebookContent {
  cells: NotebookCell[]
  metadata: {
    kernelspec?: {
      display_name: string
      language: string
      name: string
    }
    language_info?: {
      name: string
      version?: string
    }
    [key: string]: unknown
  }
  nbformat: number
  nbformat_minor: number
}

/**
 * Template rendering result
 */
export interface TemplateRenderResult {
  script: string
  notebook: NotebookContent
  paths: PlatformPaths
  metadata?: Record<string, unknown>
  warnings: string[]
}

/**
 * Platform adapter interface
 */
export interface PlatformAdapter {
  id: PlatformId
  name: string

  /**
   * Generate platform-specific setup code
   */
  generateSetupCell(ctx: TemplateContext): string

  /**
   * Generate data loading code for platform paths
   */
  generateDataLoading(ctx: TemplateContext): string

  /**
   * Generate checkpoint saving code
   */
  generateCheckpointSave(ctx: TemplateContext): string

  /**
   * Generate output/artifact saving code
   */
  generateOutputSave(ctx: TemplateContext): string

  /**
   * Get platform-specific paths
   */
  getPaths(ctx: TemplateContext): PlatformPaths

  /**
   * Get recommended batch size for a model on this platform
   */
  getRecommendedBatchSize(modelName: string): number

  /**
   * Get recommended gradient accumulation steps
   */
  getRecommendedGradientAccumulation(modelName: string, targetBatchSize: number): number

  /**
   * Generate kernel/job metadata (e.g., kernel-metadata.json for Kaggle)
   */
  generateMetadata?(ctx: TemplateContext): Record<string, unknown>

  /**
   * Validate that a notebook is compatible with this platform
   */
  validateNotebook?(content: NotebookContent): ValidationResult
}

/**
 * Validation result
 */
export interface ValidationResult {
  valid: boolean
  errors: ValidationError[]
  warnings: ValidationWarning[]
}

/**
 * Validation error
 */
export interface ValidationError {
  type: 'error'
  code: string
  message: string
  line?: number
  suggestion?: string
}

/**
 * Validation warning
 */
export interface ValidationWarning {
  type: 'warning'
  code: string
  message: string
  line?: number
  suggestion?: string
}

/**
 * Template file header markers
 */
export const TEMPLATE_MARKERS = {
  TEMPLATE: '@@TEMPLATE:',
  DESCRIPTION: '@@DESCRIPTION:',
  VERSION: '@@VERSION:',
  PLATFORMS: '@@PLATFORMS:',
  VARIABLES: '@@VARIABLES:',
  INJECTION_START: '--- INJECTION POINT:',
  INJECTION_END: '--- END INJECTION POINT ---',
  CELL_START: '@@CELL:',
  CELL_PLATFORM: '@@PLATFORM:',
} as const

/**
 * Common model batch size recommendations by platform
 */
export const BATCH_SIZE_RECOMMENDATIONS: Record<string, Record<string, number>> = {
  'kaggle-p100': {
    'facebook/nllb-200-distilled-600M': 4,
    'facebook/nllb-200-1.3B': 2,
    'facebook/nllb-200-3.3B': 1,
    'google/byt5-small': 16,
    'google/byt5-base': 8,
    'google/byt5-large': 4,
    'google/mt5-small': 16,
    'google/mt5-base': 8,
    default: 4,
  },
  'kaggle-t4x2': {
    'facebook/nllb-200-distilled-600M': 8,
    'facebook/nllb-200-1.3B': 4,
    'facebook/nllb-200-3.3B': 2,
    'google/byt5-small': 32,
    'google/byt5-base': 16,
    'google/byt5-large': 8,
    default: 8,
  },
  'colab-free': {
    'facebook/nllb-200-distilled-600M': 4,
    'facebook/nllb-200-1.3B': 2,
    'facebook/nllb-200-3.3B': 1,
    'google/byt5-small': 16,
    'google/byt5-base': 8,
    'google/byt5-large': 4,
    default: 4,
  },
  'colab-pro': {
    'facebook/nllb-200-distilled-600M': 16,
    'facebook/nllb-200-1.3B': 8,
    'facebook/nllb-200-3.3B': 4,
    'google/byt5-small': 64,
    'google/byt5-base': 32,
    'google/byt5-large': 16,
    default: 16,
  },
  'vertex-a100': {
    'facebook/nllb-200-distilled-600M': 16,
    'facebook/nllb-200-1.3B': 8,
    'facebook/nllb-200-3.3B': 4,
    default: 16,
  },
  'runpod-a100': {
    'facebook/nllb-200-distilled-600M': 32,
    'facebook/nllb-200-1.3B': 16,
    'facebook/nllb-200-3.3B': 8,
    default: 32,
  },
  'runpod-3090': {
    'facebook/nllb-200-distilled-600M': 8,
    'facebook/nllb-200-1.3B': 4,
    'facebook/nllb-200-3.3B': 2,
    default: 8,
  },
  local: {
    default: 4,
  },
}

/**
 * Get recommended batch size for a model on a platform
 */
export function getRecommendedBatchSize(platform: PlatformId, modelName: string): number {
  const platformRecs = BATCH_SIZE_RECOMMENDATIONS[platform] || BATCH_SIZE_RECOMMENDATIONS.local
  return platformRecs[modelName] || platformRecs.default || 4
}

/**
 * Calculate gradient accumulation steps to achieve target effective batch size
 */
export function calculateGradientAccumulation(actualBatchSize: number, targetEffectiveBatch: number): number {
  return Math.max(1, Math.ceil(targetEffectiveBatch / actualBatchSize))
}

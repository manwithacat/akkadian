/**
 * Model Registry Types
 *
 * Tracks models uploaded to Kaggle Model Registry for reuse in training/inference.
 */

/**
 * A single model entry in the registry
 */
export interface ModelEntry {
  /** Full Kaggle model handle (e.g., "username/model/framework/variation") */
  handle: string

  /** ISO timestamp when model was uploaded */
  uploaded_at: string

  /** Source training config file */
  training_config: string

  /** Version from training config */
  training_version: string

  /** Base model used for fine-tuning (e.g., "facebook/nllb-200-distilled-600M") */
  base_model: string

  /** Training metrics at time of upload */
  metrics?: {
    eval_loss?: number
    train_loss?: number
    bleu?: number
    [key: string]: number | undefined
  }

  /** Kaggle notebook that trained this model */
  kaggle_notebook?: string

  /** Optional notes about the model */
  notes?: string
}

/**
 * Model registry stored in models.toml
 */
export interface ModelRegistry {
  models: ModelEntry[]
}

/**
 * Parse a Kaggle model handle into components
 */
export function parseModelHandle(handle: string): {
  username: string
  model: string
  framework: string
  variation: string
  version?: string
} | null {
  const parts = handle.split('/')
  if (parts.length < 4 || parts.length > 5) {
    return null
  }
  return {
    username: parts[0],
    model: parts[1],
    framework: parts[2],
    variation: parts[3],
    version: parts[4],
  }
}

/**
 * Validate a Kaggle model handle format
 */
export function isValidModelHandle(handle: string): boolean {
  return parseModelHandle(handle) !== null
}

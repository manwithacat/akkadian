/**
 * Register a model in the model registry
 */

import TOML from '@iarna/toml'
import { existsSync, readFileSync, writeFileSync } from 'fs'
import { join } from 'path'
import { z } from 'zod'
import { getActiveCompetitionDir } from '../../lib/config'
import { error, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'
import type { ModelEntry, ModelRegistry } from '../../types/models'
import { isValidModelHandle } from '../../types/models'

const RegisterArgs = z.object({
  // The CLI parser stores first positional arg as 'path', so we accept either
  path: z.string().optional().describe('Kaggle model handle (positional)'),
  handle: z.string().optional().describe('Kaggle model handle (--handle)'),
  config: z.string().optional().describe('Training config file used'),
  version: z.string().optional().describe('Training config version'),
  baseModel: z.string().optional().describe('Base model name (e.g., facebook/nllb-200-distilled-600M)'),
  metrics: z.string().optional().describe('JSON object of metrics (e.g., {"eval_loss": 0.82})'),
  notebook: z.string().optional().describe('Kaggle notebook that trained this model'),
  notes: z.string().optional().describe('Notes about the model'),
  competition: z.string().optional().describe('Competition directory to use'),
})

/**
 * Load model registry from models.toml
 */
function loadModelRegistry(competitionDir: string): ModelRegistry {
  const registryPath = join(competitionDir, 'models.toml')

  if (!existsSync(registryPath)) {
    return { models: [] }
  }

  const content = readFileSync(registryPath, 'utf-8')
  const parsed = TOML.parse(content) as unknown as ModelRegistry

  return {
    models: parsed.models || [],
  }
}

/**
 * Save model registry to models.toml
 */
function saveModelRegistry(competitionDir: string, registry: ModelRegistry): void {
  const registryPath = join(competitionDir, 'models.toml')

  // Create TOML content with header
  const header = `# Model Registry
# Auto-generated - tracks models uploaded to Kaggle Model Registry
# Do not edit manually - use 'akk model register' to add models

`
  const tomlContent = TOML.stringify(registry as unknown as TOML.JsonMap)

  writeFileSync(registryPath, header + tomlContent)
}

export const register: CommandDefinition<typeof RegisterArgs> = {
  name: 'model register',
  description: 'Register a model in the model registry',
  help: `
Registers a model that has been uploaded to Kaggle Model Registry,
allowing it to be used as a source for training/inference notebooks.

The handle should be the full Kaggle model path:
  username/model-name/framework/variation

Example:
  akk model register manwithacat/akkadian-nllb/transformers/v1 \\
    --config training.toml --version 1.2.0 \\
    --base-model facebook/nllb-200-distilled-600M \\
    --metrics '{"eval_loss": 0.82, "train_loss": 0.71}'
`,
  examples: [
    'akk model register manwithacat/akkadian-nllb/transformers/v1',
    'akk model register manwithacat/akkadian-nllb/transformers/v1 --version 1.2.0 --metrics \'{"eval_loss": 0.82}\'',
  ],
  args: RegisterArgs,

  async run(args, ctx) {
    // Get handle from positional arg (path) or --handle flag
    const handle = args.handle || args.path

    if (!handle) {
      return error(
        'MISSING_HANDLE',
        'Model handle is required',
        'Usage: akk model register <handle> or --handle <handle>'
      )
    }

    // Validate handle format
    if (!isValidModelHandle(handle)) {
      return error(
        'INVALID_HANDLE',
        `Invalid model handle format: ${handle}`,
        'Handle should be: username/model/framework/variation'
      )
    }

    // Get competition directory
    let competitionDir: string

    if (args.competition) {
      competitionDir = args.competition
    } else if (ctx.config) {
      competitionDir = getActiveCompetitionDir(ctx.cwd, ctx.config)
    } else {
      return error(
        'NO_COMPETITION',
        'No competition directory found',
        'Specify with --competition or configure in akk.toml'
      )
    }

    if (!existsSync(competitionDir)) {
      return error(
        'COMPETITION_NOT_FOUND',
        `Competition directory not found: ${competitionDir}`,
        'Check the path or run akk competition init'
      )
    }

    // Parse metrics if provided
    let metrics: Record<string, number> | undefined
    if (args.metrics) {
      try {
        metrics = JSON.parse(args.metrics)
      } catch {
        return error('INVALID_METRICS', 'Invalid metrics JSON', 'Provide valid JSON like: {"eval_loss": 0.82}')
      }
    }

    // Load existing registry
    const registry = loadModelRegistry(competitionDir)

    // Check if model already exists
    const existingIndex = registry.models.findIndex((m) => m.handle === handle)
    if (existingIndex >= 0) {
      // Update existing entry
      registry.models[existingIndex] = {
        ...registry.models[existingIndex],
        uploaded_at: new Date().toISOString(),
        training_config: args.config || registry.models[existingIndex].training_config,
        training_version: args.version || registry.models[existingIndex].training_version,
        base_model: args.baseModel || registry.models[existingIndex].base_model,
        metrics: metrics || registry.models[existingIndex].metrics,
        kaggle_notebook: args.notebook || registry.models[existingIndex].kaggle_notebook,
        notes: args.notes || registry.models[existingIndex].notes,
      }
    } else {
      // Create new entry
      const newEntry: ModelEntry = {
        handle,
        uploaded_at: new Date().toISOString(),
        training_config: args.config || 'training.toml',
        training_version: args.version || '1.0.0',
        base_model: args.baseModel || 'unknown',
        metrics,
        kaggle_notebook: args.notebook,
        notes: args.notes,
      }
      registry.models.push(newEntry)
    }

    // Save registry
    saveModelRegistry(competitionDir, registry)

    return success({
      message: existingIndex >= 0 ? 'Model updated in registry' : 'Model registered successfully',
      handle,
      registry_path: join(competitionDir, 'models.toml'),
      entry: registry.models.find((m) => m.handle === handle),
    })
  },
}

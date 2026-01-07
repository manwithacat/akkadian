/**
 * List registered models from the model registry
 */

import TOML from '@iarna/toml'
import { existsSync, readFileSync } from 'fs'
import { join } from 'path'
import { z } from 'zod'
import { getActiveCompetitionDir } from '../../lib/config'
import { error, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'
import type { ModelEntry, ModelRegistry } from '../../types/models'

const ListArgs = z.object({
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

export const list: CommandDefinition<typeof ListArgs> = {
  name: 'model list',
  description: 'List registered models from the model registry',
  help: `
Lists all models registered in the competition's models.toml file.

These are models that have been uploaded to Kaggle Model Registry
and can be used as sources for training/inference notebooks.

The registry tracks:
- Kaggle model handle
- Training config and version
- Base model used
- Training metrics
- Upload timestamp
`,
  examples: ['akk model list', 'akk model list --competition competitions/my-competition'],
  args: ListArgs,

  async run(args, ctx) {
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

    // Load registry
    const registry = loadModelRegistry(competitionDir)

    if (registry.models.length === 0) {
      return success({
        models: [],
        registry_path: join(competitionDir, 'models.toml'),
        message: 'No models registered. Train and upload a model, then register with: akk model register',
      })
    }

    // Format for display
    const models = registry.models.map((m: ModelEntry) => ({
      handle: m.handle,
      version: m.training_version,
      base_model: m.base_model,
      uploaded: m.uploaded_at,
      metrics: m.metrics
        ? Object.entries(m.metrics)
            .filter(([_, v]) => v !== undefined)
            .map(([k, v]) => `${k}=${(v as number).toFixed(4)}`)
            .join(', ')
        : undefined,
      notebook: m.kaggle_notebook,
      notes: m.notes,
    }))

    return success({
      models,
      count: models.length,
      registry_path: join(competitionDir, 'models.toml'),
    })
  },
}

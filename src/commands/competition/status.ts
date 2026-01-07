/**
 * Competition Status Command
 *
 * Display current competition status including models, submissions, and scores.
 */

import { z } from 'zod'
import { loadCompetitionDirectory } from '../../lib/config'
import { error, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const StatusArgs = z.object({
  path: z.string().optional().describe('Competition directory (default: current directory)'),
  verbose: z.boolean().default(false).describe('Show detailed information'),
})

export const status: CommandDefinition<typeof StatusArgs> = {
  name: 'competition status',
  description: 'Show competition status and configuration',
  help: `
Display the current competition status including:
- Competition metadata (name, platform, metric)
- Active model and its score
- Model registry with scores and status
- Submission history and statistics
- Training configuration defaults

Use --verbose for full configuration details.
`,
  examples: [
    'akk competition status',
    'akk competition status --verbose',
    'akk competition status --path ./my-competition',
  ],
  args: StatusArgs,

  async run(args, _ctx) {
    const competitionDir = await loadCompetitionDirectory(args.path || process.cwd())

    if (!competitionDir) {
      return error(
        'NOT_FOUND',
        'No competition.toml found in current directory or parents',
        'Run "akk competition init" to initialize a competition directory'
      )
    }

    const { config } = competitionDir

    // Build status summary
    const modelCount = Object.keys(config.models).length
    const activeModels = Object.entries(config.models)
      .filter(([_, m]) => m.status === 'active')
      .map(([name, m]) => ({ name, ...m }))

    const recentSubmissions = config.submissions.history.slice(-5).reverse()

    // Basic status
    const statusData: Record<string, unknown> = {
      competition: {
        name: config.competition.name,
        slug: config.competition.slug,
        platform: config.competition.platform,
        metric: config.competition.metric,
        metric_direction: config.competition.metric_direction,
        deadline: config.competition.deadline,
      },
      active_model: config.active_model
        ? {
            name: config.active_model.name,
            base: config.active_model.base,
            current_score: config.active_model.current_score,
            best_score: config.active_model.best_score,
          }
        : null,
      models: {
        total: modelCount,
        active: activeModels.length,
        best_score: config.submissions.best_score,
      },
      submissions: {
        total: config.submissions.total,
        remaining_today: config.submissions.remaining_today,
        best_score: config.submissions.best_score,
        recent: recentSubmissions.map((s) => ({
          id: s.id,
          model: s.model,
          score: s.public_score,
          timestamp: s.timestamp,
        })),
      },
      paths: {
        root: competitionDir.root,
        notebooks: competitionDir.notebooks,
        models: competitionDir.models,
      },
    }

    // Add verbose details
    if (args.verbose) {
      statusData.all_models = config.models
      statusData.kernels = config.kernels
      statusData.training_defaults = config.training
      statusData.gcs = config.gcs
      statusData.mlflow = config.mlflow
      statusData.full_submission_history = config.submissions.history
    }

    return success(statusData)
  },
}

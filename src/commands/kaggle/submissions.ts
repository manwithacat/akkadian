/**
 * List competition submissions and their scoring status
 */

import { z } from 'zod'
import { getCompetitionSubmissions } from '../../lib/kaggle'
import { error, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const SubmissionsArgs = z.object({
  competition: z.string().optional().describe('Competition slug (default: from config)'),
  limit: z.number().default(10).describe('Number of submissions to show'),
  pending: z.boolean().default(false).describe('Show only pending submissions'),
})

export const submissions: CommandDefinition<typeof SubmissionsArgs> = {
  name: 'kaggle submissions',
  description: 'List competition submissions and scoring status',
  help: `
Lists your submissions to a Kaggle competition with their scoring status.

This is different from kernel status:
  - Kernel status: Whether the notebook finished running
  - Submission status: Whether Kaggle has scored your submission

Submission statuses:
  - pending: Kaggle is scoring the submission
  - complete: Scoring finished, score available
  - error: Scoring failed

Use --pending to filter for submissions still being scored.
`,
  examples: [
    'akk kaggle submissions',
    'akk kaggle submissions --competition deep-past-initiative-machine-translation',
    'akk kaggle submissions --limit 5',
    'akk kaggle submissions --pending',
  ],
  args: SubmissionsArgs,

  async run(args, ctx) {
    const { competition: competitionArg, limit, pending } = args

    // Get competition from args or config
    const competition = competitionArg || ctx.config?.kaggle?.competition
    if (!competition) {
      return error(
        'NO_COMPETITION',
        'Competition not specified',
        'Use --competition or set kaggle.competition in akk.toml',
        {}
      )
    }

    try {
      let subs = await getCompetitionSubmissions(competition)

      // Filter pending if requested
      if (pending) {
        subs = subs.filter((s) => s.status === 'pending')
      }

      // Limit results
      subs = subs.slice(0, limit)

      // Format for display
      const formatted = subs.map((s) => ({
        date: s.date,
        status: s.status,
        score: s.publicScore ?? null,
        description: s.description || null,
      }))

      return success({
        competition,
        count: formatted.length,
        total: subs.length,
        submissions: formatted,
      })
    } catch (err) {
      return error(
        'SUBMISSIONS_ERROR',
        err instanceof Error ? err.message : String(err),
        'Check competition slug is correct',
        { competition }
      )
    }
  },
}

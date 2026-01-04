/**
 * List running Kaggle kernels
 */

import { z } from 'zod'
import { listRunningKernels } from '../../lib/kaggle'
import { error, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const ListRunningArgs = z.object({
  user: z.string().optional().describe('Kaggle username (defaults to config)'),
})

export const listRunning: CommandDefinition<typeof ListRunningArgs> = {
  name: 'kaggle list-running',
  description: 'List currently running or queued kernels',
  help: `
Checks your recent kernels and shows any that are currently running or queued.

This is useful for monitoring active jobs without needing to know the exact kernel slug.

The command checks the 10 most recent kernels by default.
`,
  examples: ['akk kaggle list-running', 'akk kaggle list-running --user manwithacat'],
  args: ListRunningArgs,

  async run(args, ctx) {
    const user = args.user || ctx.config?.kaggle?.username

    if (!user) {
      return error('NO_USER', 'Kaggle username not specified', 'Provide --user or set kaggle.username in akk.toml', {})
    }

    try {
      const running = await listRunningKernels(user)

      if (running.length === 0) {
        return success({
          running: [],
          message: 'No kernels currently running or queued',
        })
      }

      return success({
        running: running.map((k) => ({
          slug: k.slug,
          status: k.status,
          url: `https://www.kaggle.com/code/${k.slug}`,
        })),
        count: running.length,
      })
    } catch (err) {
      return error('LIST_ERROR', err instanceof Error ? err.message : String(err), 'Check your Kaggle credentials', {
        user,
      })
    }
  },
}

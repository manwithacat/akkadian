/**
 * Check Kaggle kernel status
 */

import { z } from 'zod'
import type { CommandDefinition } from '../../types/commands'
import { success, error } from '../../lib/output'
import { getKernelStatus } from '../../lib/kaggle'

const StatusArgs = z.object({
  path: z.string().describe('Kernel slug (user/kernel-name)'),
})

export const status: CommandDefinition<typeof StatusArgs> = {
  name: 'kaggle status',
  description: 'Check kernel execution status',
  help: `
Checks the current status of a Kaggle kernel.

Possible statuses:
  - queued: Waiting to run
  - running: Currently executing
  - complete: Finished successfully
  - error: Failed with error
  - cancelled: Manually cancelled
`,
  examples: [
    'akk kaggle status manwithacat/nllb-train',
  ],
  args: StatusArgs,

  async run(args, ctx) {
    const { path: slug } = args

    if (!slug.includes('/')) {
      return error(
        'INVALID_SLUG',
        'Kernel slug must be in format "user/kernel-name"',
        'Example: manwithacat/nllb-train',
        { slug }
      )
    }

    try {
      const status = await getKernelStatus(slug)

      return success({
        slug,
        status: status.status,
        failureMessage: status.failureMessage || null,
      })
    } catch (err) {
      return error(
        'STATUS_ERROR',
        err instanceof Error ? err.message : String(err),
        'Check kernel slug is correct',
        { slug }
      )
    }
  },
}
